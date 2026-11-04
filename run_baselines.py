"""Evaluate the classical (training-free) baselines on a split.

Usage: python run_baselines.py --city NYC --split test --n 600
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from grpoi import config  # noqa: E402
from grpoi.agent import Prediction  # noqa: E402
from grpoi.classic_baselines import ALL_BASELINES  # noqa: E402
from grpoi.data import load_city  # noqa: E402
from grpoi.metrics import evaluate, format_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="NYC", choices=list(config.CITIES))
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    city = load_city(args.city)
    samples = getattr(city, args.split)
    if args.n:
        samples = samples[: args.n]
    print(f"{args.city}/{args.split}: {len(samples)} samples")

    results = {}
    for cls in ALL_BASELINES:
        model = cls(city)
        preds = []
        for s in samples:
            ranking = model.predict(s, 10)
            p = Prediction(ranking=ranking)
            p.candidate_recall = s.target in ranking
            preds.append(p)
        m = evaluate(samples, preds, city)
        m["calls_per_sample"] = 0.0
        m["cost_usd_per_sample"] = 0.0
        results[model.name] = m
        print(f"  {model.name:9s} HR@5={m['HR@5']:.3f} HR@10={m['HR@10']:.3f} "
              f"N@10={m['NDCG@10']:.3f} M@10={m['MAP@10']:.3f}")

    print()
    print(format_table(results))

    out = Path(args.out) if args.out else (
        ROOT / "results" / f"{args.city}_{args.split}_classic.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "city": args.city,
        "split": args.split,
        "n": len(samples),
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
