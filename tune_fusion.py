"""Grid-search fusion weights on the *validation* split, no LLM calls.

Usage: python tune_fusion.py --city PHO --split valid --n 0
`--n 0` means the whole split.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from grpoi import config  # noqa: E402
from grpoi.data import load_city  # noqa: E402
from grpoi.retrieval import RetrievalEngine  # noqa: E402


def score(engine, samples, ks=(1, 5, 10)):
    rec = {k: 0 for k in ks}
    nd = {k: 0.0 for k in ks}
    cand = 0
    for s in samples:
        cands = engine.candidates(s)
        ranking = [c.poi for c in cands]
        if s.target in ranking[: engine.cfg.n_candidates]:
            cand += 1
        if s.target in ranking:
            pos = ranking.index(s.target) + 1
            for k in ks:
                if pos <= k:
                    rec[k] += 1
                    nd[k] += 1.0 / np.log2(pos + 1)
    n = len(samples)
    return {
        "cand_recall": cand / n,
        **{f"R@{k}": rec[k] / n for k in ks},
        **{f"N@{k}": nd[k] / n for k in ks},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="PHO")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    city = load_city(args.city)
    samples = getattr(city, args.split)
    if args.n:
        samples = samples[: args.n]
    engine = RetrievalEngine(city)

    grid = {
        "semantic": [0.0, 0.5, 1.0, 2.0],
        "graph": [0.5, 1.0, 2.0],
        "geo": [1.0, 2.0, 3.0],
        "memory": [1.0, 2.5, 4.0],
    }
    keys = list(grid)
    results = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        w = dict(zip(keys, combo))
        engine.cfg.weights = w
        m = score(engine, samples)
        results.append({"weights": w, **m})
        print(f"{w} -> R@5={m['R@5']:.3f} R@10={m['R@10']:.3f} cand={m['cand_recall']:.3f}", flush=True)

    results.sort(key=lambda r: (r["R@10"], r["R@5"]), reverse=True)
    print("\n=== top configs by R@10 ===")
    for r in results[: args.top]:
        print(json.dumps(r, ensure_ascii=False))

    out = ROOT / "results" / f"tune_{args.city}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
