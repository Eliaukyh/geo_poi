"""Offline tuning of the prior-anchor weight alpha from a dumped run.

No LLM calls: re-reads fusion_order / llm_order saved by run_eval.py --dump.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from grpoi.agent import blend_orders  # noqa: E402
from grpoi.metrics import map_at, ndcg_at  # noqa: E402


def scores(records, alpha: float, ks=(1, 5, 10)) -> dict:
    rec = {k: 0.0 for k in ks}
    nd = {k: 0.0 for k in ks}
    mp = {k: 0.0 for k in ks}
    for r in records:
        final = blend_orders(r["fusion_order"], r["llm_order"], alpha)[:10]
        t = r["target"]
        pos = final.index(t) + 1 if t in final else None
        if pos is not None:
            for k in ks:
                if pos <= k:
                    rec[k] += 1
                    nd[k] += ndcg_at(pos, k)
                    mp[k] += map_at(pos, k)
    n = len(records)
    return {f"R@{k}": rec[k] / n for k in ks} | {f"N@{k}": nd[k] / n for k in ks} | {f"M@{k}": mp[k] / n for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="jsonl dump produced by run_eval.py")
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.file).read_text().splitlines() if l.strip()]
    print(f"{len(records)} records; verified={sum(r['verified'] for r in records)}")

    print(f"{'alpha':>6} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'N@5':>6} {'N@10':>6} {'M@10':>6}")
    best = None
    for alpha in np.arange(0.0, 1.0001, 0.05):
        s = scores(records, float(alpha))
        print(
            f"{alpha:6.2f} {s['R@1']:6.3f} {s['R@5']:6.3f} {s['R@10']:6.3f} "
            f"{s['N@5']:6.3f} {s['N@10']:6.3f} {s['M@10']:6.3f}"
        )
        key = (s["R@10"], s["N@10"])
        if best is None or key > best[0]:
            best = (key, float(alpha), s)
    print(f"\nbest by (R@10, N@10): alpha={best[1]:.2f} -> {json.dumps(best[2], indent=None)}")


if __name__ == "__main__":
    main()
