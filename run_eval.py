#!/usr/bin/env python
"""End-to-end evaluation driver.

Examples
--------
# retrieval only, no LLM cost (sanity check + retrieval ceiling)
python run_eval.py --city PHO --n 30 --mode retrieval_only

# full method
python run_eval.py --city PHO --n 30 --mode ours

# ablations / baselines
python run_eval.py --city PHO --n 30 --mode no_agent
python run_eval.py --city PHO --n 30 --mode free

# several methods in one shot
python run_eval.py --city PHO --n 30 --mode ours,no_agent,free
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from grpoi import config  # noqa: E402
from grpoi.agent import AdaptiveAgent, Prediction  # noqa: E402
from grpoi.baselines import FreeFormAgent  # noqa: E402
from grpoi.data import load_city  # noqa: E402
from grpoi.llm import LLMClient  # noqa: E402
from grpoi.metrics import evaluate, format_table  # noqa: E402

MODES = {
    "ours": "multi-route retrieval + tools + adaptive verification",
    "no_agent": "multi-route retrieval + tools, single LLM pass",
    "no_geo": "ablation: geo route removed from fusion",
    "free": "baseline: free-form LLM, no retrieval (RALLM-POI w/o RAG)",
    "rag_free": "baseline: RALLM-POI style TF-IDF RAG, free-form output",
    "retrieval_only": "diagnostic: fusion ranking without any LLM call",
}


def retrieval_only(city, engine, sample: "Sample", target: int) -> Prediction:
    cands = engine.candidates(sample)
    pred = Prediction(ranking=[c.poi for c in cands][:10])
    pred.candidate_recall = target in [c.poi for c in cands]
    return pred


def build_runner(mode: str, city, client: LLMClient | None, alpha: float | None = None):
    if mode == "ours":
        return AdaptiveAgent(city, client=client, alpha=alpha)
    if mode == "no_agent":
        return AdaptiveAgent(city, client=client, enable_agent=False, alpha=alpha)
    if mode == "no_geo":
        return AdaptiveAgent(city, client=client, use_geo=False, alpha=alpha)
    if mode == "free":
        return FreeFormAgent(city, client=client, use_rag=False)
    if mode == "rag_free":
        return FreeFormAgent(city, client=client, use_rag=True)
    raise ValueError(f"unknown mode {mode}")


def out_jsonl(args, mode: str) -> Path:
    stem = args.out and Path(args.out).stem or f"{args.city}_{args.split}_{args.n}"
    d = ROOT / "results" / "dump"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stem}_{mode}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="PHO", choices=list(config.CITIES))
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--n", type=int, default=20, help="number of samples")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--mode", default="ours", help="comma separated: " + ", ".join(MODES))
    ap.add_argument("--model", default=config.AGENT.model)
    ap.add_argument("--alpha", type=float, default=None,
                    help="weight of the retrieval prior in the blend (default from config)")
    ap.add_argument("--workers", type=int, default=1, help="concurrent samples (each may issue 1-2 LLM calls)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"unknown mode {m}; choose from {list(MODES)}")

    config.AGENT.model = args.model
    print(f"loading {args.city} ...", flush=True)
    city = load_city(args.city)
    samples = getattr(city, args.split)[args.offset : args.offset + args.n]
    print(
        f"{args.city}/{args.split}: using {len(samples)} samples "
        f"(offset={args.offset}); total split={len(getattr(city, args.split))}"
    )

    engine = None
    if "retrieval_only" in modes:
        from grpoi.retrieval import RetrievalEngine

        engine = RetrievalEngine(city)

    client = None
    results: dict[str, dict] = {}

    for mode in modes:
        t0 = time.time()
        if mode == "retrieval_only":
            preds = [retrieval_only(city, engine, s, s.target) for s in samples]
        else:
            if client is None:
                client = LLMClient(config.AGENT)
            base_calls = client.usage.calls
            runner = build_runner(mode, city, client, args.alpha)
            preds = [None] * len(samples)
            errors = 0
            t_start = time.time()
            if args.workers > 1:
                done = 0
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    fut = {ex.submit(runner.run, s, s.target): i for i, s in enumerate(samples)}
                    for f in as_completed(fut):
                        i = fut[f]
                        try:
                            preds[i] = f.result()
                        except Exception as exc:  # noqa: BLE001
                            errors += 1
                            preds[i] = Prediction(ranking=[])
                            print(f"  [{mode}] sample {i} failed: {type(exc).__name__}: {exc}", flush=True)
                        done += 1
                        if done % 100 == 0:
                            rate = done / max(time.time() - t_start, 1e-9)
                            print(f"  [{mode}] {done}/{len(samples)}  ({rate:.1f}/s)", flush=True)
            else:
                for i, s in enumerate(samples):
                    try:
                        preds[i] = runner.run(s, s.target)
                    except Exception as exc:  # noqa: BLE001
                        errors += 1
                        preds[i] = Prediction(ranking=[])
                        print(f"  [{mode}] sample {i} failed: {exc}", flush=True)
                    if (i + 1) % 10 == 0:
                        print(f"  [{mode}] {i + 1}/{len(samples)}", flush=True)
            if errors:
                print(f"  [{mode}] {errors}/{len(samples)} samples failed", flush=True)
            used = client.usage.calls - base_calls
            print(f"  [{mode}] LLM calls: {used}", flush=True)

        # keep the raw orderings so alpha / thresholds can be re-tuned offline
        dump = out_jsonl(args, mode)
        with open(dump, "w") as fh:
            for s, p in zip(samples, preds):
                fh.write(json.dumps({
                    "target": int(s.target),
                    "user": int(s.user),
                    "fusion_order": list(getattr(p, "fusion_order", [])),
                    "llm_order": list(getattr(p, "llm_order", [])),
                    "final": list(p.ranking),
                    "confidence": getattr(p, "confidence", None),
                    "verified": bool(getattr(p, "verified", False)),
                }) + "\n")

        metrics = evaluate(samples, preds, city)
        metrics["wall_s"] = round(time.time() - t0, 2)
        metrics["mode"] = mode
        metrics["model"] = args.model if mode != "retrieval_only" else "-"
        results[mode] = metrics

        if client is not None and mode != "retrieval_only":
            # report the marginal usage of this run
            pass

    print()
    print(format_table(results))
    print()

    out = Path(args.out) if args.out else (
        ROOT / "results" / f"{args.city}_{args.split}_{args.n}_{args.model.replace('/', '_')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"city": args.city, "split": args.split, "n": len(samples), "results": results}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"saved -> {out}")

    md = out.with_suffix(".md")
    md.write_text(format_table(results) + "\n")
    print(f"saved -> {md}")


if __name__ == "__main__":
    main()
