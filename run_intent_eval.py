"""Evaluate LLM intent expansion: does it add candidate reachability?

The LLM only produces intents (category + radius); POIs are grounded by the
retrieval layer.  Intents are cached so that per-intent budget and slot
allocation can be swept without extra API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from grpoi import fusion  # noqa: E402
from grpoi.data import load_city  # noqa: E402
from grpoi.intent import Intent, IntentExpander  # noqa: E402
from grpoi.llm import LLMClient  # noqa: E402
from grpoi.retrieval import RetrievalEngine  # noqa: E402


def load_or_fetch(expander, samples, cache: Path, workers: int):
    if cache.exists():
        rows = [json.loads(l) for l in cache.read_text().splitlines() if l.strip()]
        if len(rows) == len(samples):
            print(f"loaded cached intents: {cache}", flush=True)
            return [[Intent(**i) for i in r] for r in rows], True

    out = [None] * len(samples)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(expander.expand, s): i for i, s in enumerate(samples)}
        for f in as_completed(futs):
            i = futs[f]
            try:
                ic = f.result()
                out[i] = ic.intents
            except Exception as exc:  # noqa: BLE001
                print(f"  sample {i} failed: {exc}", flush=True)
                out[i] = []
            done += 1
            if done % 50 == 0:
                print(f"  intents {done}/{len(samples)} ({done / (time.time() - t0):.1f}/s)", flush=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as fh:
        for ints in out:
            fh.write(json.dumps([{"category": i.category, "radius_km": i.radius_km, "reason": i.reason} for i in ints]) + "\n")
    return out, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="NYC")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    city = load_city(args.city)
    samples = city.test[: args.n]
    engine = RetrievalEngine(city)
    client = LLMClient()
    expander = IntentExpander(city, client=client)

    print(f"{args.city}: {len(samples)} samples", flush=True)
    routes_all = [engine.route_lists(s) for s in samples]
    base10 = [[p for p, _ in fusion.two_stage(r, engine.cfg.weights)] for r in routes_all]

    cache = ROOT / "results" / "dump" / f"intents_{args.city}_{args.n}.jsonl"
    intents, cached = load_or_fetch(expander, samples, cache, args.workers)
    print(f"avg intents/sample: {np.mean([len(x) for x in intents]):.2f}", flush=True)

    n = len(samples)
    base_hits = sum(1 for s, b in zip(samples, base10) if s.target in b)

    print("\n| per_intent | intent POIs | intent hit | union reach | recoverable |")
    print("|---|---|---|---|---|")
    best = None
    for per in (5, 10, 20, 40):
        pois = [expander._ground(s, ints, per) for s, ints in zip(samples, intents)]
        ih = sum(1 for s, p in zip(samples, pois) if s.target in [x for x, _ in p])
        uh = sum(1 for s, b, p in zip(samples, base10, pois) if s.target in set(b) | {x for x, _ in p})
        rec = sum(1 for s, b, p in zip(samples, base10, pois) if s.target not in b and s.target in {x for x, _ in p})
        print(f"| {per} | {np.mean([len(p) for p in pois]):.1f} | {ih / n:.3f} | {uh / n:.3f} | {rec / n:.3f} |")
        if best is None or uh > best[0]:
            best = (uh, per, pois)

    print(f"\nbaseline top-10 hit: {base_hits / n:.3f}")

    print("\n| head slots (graph) | + intent slots | HR@10 |")
    print("|---|---|---|")
    pois = best[2]
    for k_graph in (8, 7, 6, 5, 4):
        hits = 0
        for s, r, b, p in zip(samples, routes_all, base10, pois):
            g = fusion._minmax(r.get("graph", []))
            out = sorted(g, key=lambda x: -g[x])[:k_graph]
            for poi, _ in p:
                if poi not in out:
                    out.append(poi)
                if len(out) >= 10:
                    break
            for poi in b:
                if poi not in out:
                    out.append(poi)
                if len(out) >= 10:
                    break
            if s.target in out[:10]:
                hits += 1
        print(f"| {k_graph} | {10 - k_graph} | {hits / n:.3f} |")

    out = Path(args.out) if args.out else ROOT / "results" / f"intent_{args.city}.json"
    out.write_text(json.dumps({
        "n": n,
        "baseline_hit": base_hits / n,
        "best_per_intent": best[1],
        "best_union": best[0],
        "llm_calls": client.usage.calls,
        "prompt_tokens": client.usage.prompt_tokens,
        "cost_usd": client.usage.cost_usd,
        "cached": cached,
    }, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
