"""Full-split check of the soft-hint design (P1s) with a paired significance test.

For each sample: one intent call (rich context) + one generation call conditioned on
the top-3 intents.  Compares against the direct-generation dump (G0) on the same
sample indices, with a paired bootstrap on HR@10 / NDCG@10.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grpoi.data import load_city  # noqa: E402
from grpoi.generative import SYSTEM  # noqa: E402
from grpoi.llm import LLMClient  # noqa: E402
from grpoi.retrieval import RetrievalEngine  # noqa: E402
from run_hierarchical_pilot2 import (  # noqa: E402
    INTENT_SYSTEM, build_cat_stats, gen_prompt, intent_prompt, parse_ids,
)


def hit_vec(samples, rankings, k=10):
    return np.array([1.0 if s.target in (r or [])[:k] else 0.0 for s, r in zip(samples, rankings)])


def ndcg_vec(samples, rankings, k=10):
    out = []
    for s, r in zip(samples, rankings):
        r = (r or [])[:k]
        out.append(1.0 / math.log2(r.index(s.target) + 2) if s.target in r else 0.0)
    return np.array(out)


def boot_ci(diff, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="NYC")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    city = load_city(args.city)
    samples = city.test[: args.n]
    engine = RetrievalEngine(city)
    client = LLMClient()
    stats = build_cat_stats(city)
    cache = ROOT / "results" / "dump" / f"soft_hint_{args.city}_{args.n}.jsonl"
    print(f"{args.city}: {len(samples)} samples", flush=True)

    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["i"]] = r
        print(f"resume: {len(done)} cached", flush=True)

    todo = [i for i in range(len(samples)) if i not in done]
    t0 = time.time()
    lock = None

    def one(i):
        s = samples[i]
        d, _ = client.chat_json(INTENT_SYSTEM, intent_prompt(city, s, engine, stats))
        ints = []
        for it in (d or {}).get("intents", []):
            c = str(it.get("category", "")).strip()
            if c in city.cat_names and c not in ints:
                ints.append(c)
        ints = ints[:3]
        if ints:
            d, _ = client.chat_json(SYSTEM, gen_prompt(city, s, engine, stats, ints))
            ids = parse_ids(d)
        else:
            ids = []
        return i, ints, ids

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, i): i for i in todo}
        finished = 0
        for f in as_completed(futs):
            try:
                i, ints, ids = f.result()
                done[i] = {"i": i, "intents": ints, "ids": ids}
            except Exception as exc:  # noqa: BLE001
                i = futs[f]
                done[i] = {"i": i, "intents": [], "ids": []}
                print(f"  sample {i} failed: {exc}", flush=True)
            finished += 1
            if finished % 100 == 0:
                print(f"  {finished}/{len(todo)} ({finished / (time.time() - t0):.1f}/s)", flush=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as fh:
        for i in range(len(samples)):
            fh.write(json.dumps(done.get(i, {"i": i, "intents": [], "ids": []})) + "\n")

    p1s = [done[i]["ids"] for i in range(len(samples))]
    g0 = None
    if args.dump and Path(args.dump).exists():
        rows = [json.loads(l) for l in Path(args.dump).read_text().splitlines() if l.strip()][: args.n]
        g0 = [r.get("final") or [] for r in rows]

    def report(name, r):
        print(f"{name:18s} HR@1={r['HR@1']:.3f} HR@5={r['HR@5']:.3f} HR@10={r['HR@10']:.3f} NDCG@10={r['NDCG@10']:.3f}")

    def metrics(rankings):
        h = {k: 0 for k in (1, 5, 10)}
        nd = 0.0
        n = len(samples)
        for s, r in zip(samples, rankings):
            r = (r or [])[:10]
            if s.target in r:
                p = r.index(s.target) + 1
                for k in h:
                    if p <= k:
                        h[k] += 1
                nd += 1.0 / math.log2(p + 1)
        return {f"HR@{k}": h[k] / n for k in h} | {"NDCG@10": nd / n}

    print()
    report("P1s soft (full)", metrics(p1s))
    if g0:
        report("G0 direct (full)", metrics(g0))

        d_hit = hit_vec(samples, p1s) - hit_vec(samples, g0)
        d_nd = ndcg_vec(samples, p1s) - ndcg_vec(samples, g0)
        lo, hi = boot_ci(d_hit)
        lo2, hi2 = boot_ci(d_nd)
        print(f"\n配对差值 HR@10: {d_hit.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"  {'显著' if lo > 0 or hi < 0 else '不显著'}")
        print(f"配对差值 NDCG@10: {d_nd.mean():+.4f}  95% CI [{lo2:+.4f}, {hi2:+.4f}]"
              f"  {'显著' if lo2 > 0 or hi2 < 0 else '不显著'}")

    acc = sum(1 for s, i in zip(samples, [done[k]["intents"] for k in range(len(samples))])
              if i and i[0] == city.cat_name(city.poi_cat(s.target))) / len(samples)
    print(f"\n意图 top-1 准确率: {acc:.3f} | 调用 {client.usage.calls} | 成本 ${client.usage.cost_usd:.4f}")

    out = ROOT / "results" / f"soft_hint_{args.city}_{args.n}.json"
    out.write_text(json.dumps({
        "n": len(samples),
        "soft": metrics(p1s),
        "direct": metrics(g0) if g0 else None,
        "intent_top1": acc,
        "calls": client.usage.calls,
        "cost_usd": client.usage.cost_usd,
    }, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
