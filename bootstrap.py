"""Bootstrap confidence intervals for a dumped run.

Usage:
  python bootstrap.py --file results/dump/PHO_test_39_ours.jsonl --compare results/dump/PHO_test_39_retrieval_only.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def hit_vector(records: list[dict], k: int) -> np.ndarray:
    out = []
    for r in records:
        final = r.get("final") or []
        out.append(1.0 if r["target"] in final[:k] else 0.0)
    return np.asarray(out)


def ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05):
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--compare", default="")
    args = ap.parse_args()

    a = load(Path(args.file))
    print(f"{args.file}: n={len(a)}")
    for k in (1, 5, 10):
        m, lo, hi = ci(hit_vector(a, k))
        print(f"  Recall@{k}: {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    if args.compare:
        b = load(Path(args.compare))
        print(f"\n{args.compare}: n={len(b)}")
        for k in (1, 5, 10):
            m, lo, hi = ci(hit_vector(b, k))
            print(f"  Recall@{k}: {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        # paired bootstrap on the difference (assumes aligned samples)
        if len(a) == len(b):
            for k in (1, 5, 10):
                va, vb = hit_vector(a, k), hit_vector(b, k)
                diff = va - vb
                _, lo, hi = ci(diff)
                print(f"  Δ Recall@{k}: {diff.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
                      f"{'significant' if lo > 0 or hi < 0 else 'not significant'}")


if __name__ == "__main__":
    main()
