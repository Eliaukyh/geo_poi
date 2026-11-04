"""Compare fusion strategies and separate reachability from ordering.

Reports, per strategy:
  * candidate recall @25 (does the union reach the target at all?)
  * HR@1/5/10 and NDCG@10 of the fused top-10 (does ordering exploit it?)

Also reports how much the four routes overlap, which is the condition under which
RRF double-counts evidence.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from grpoi import fusion  # noqa: E402
from grpoi import config  # noqa: E402
from grpoi.data import load_city  # noqa: E402
from grpoi.retrieval import RetrievalEngine  # noqa: E402

WEIGHT_GRID = [(1, 1), (5, 0.5), (20, 0.3), (50, 0.1)]  # (graph weight, others)


def prep(engine, samples):
    return [engine.route_lists(s) for s in samples]


def score_fusion(strategy, routes_cache, samples, weights, top_n=25, ks=(1, 5, 10)):
    rec = {k: 0 for k in ks}
    ndcg = 0.0
    cand = 0
    for routes, s in zip(routes_cache, samples):
        ranked = fusion.fuse(strategy, routes, weights)
        pois = [p for p, _ in ranked]
        if s.target in pois[:top_n]:
            cand += 1
        if s.target in pois[:10]:
            pos = pois.index(s.target) + 1
            for k in ks:
                if pos <= k:
                    rec[k] += 1
            ndcg += 1.0 / math.log2(pos + 1)
    n = len(samples)
    return {
        "cand_recall": cand / n,
        **{f"HR@{k}": rec[k] / n for k in ks},
        "NDCG@10": ndcg / n,
    }


def overlap_stats(routes_cache, k=20):
    from collections import Counter

    jac = []
    for routes in routes_cache:
        sets = [set(p for p, _ in v[:k]) for v in routes.values() if v]
        if len(sets) < 2:
            continue
        s = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                u = sets[i] | sets[j]
                s.append(len(sets[i] & sets[j]) / len(u) if u else 0.0)
        jac.append(np.mean(s))
    return float(np.mean(jac)) if jac else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="NYC")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    city = load_city(args.city)
    engine = RetrievalEngine(city)
    val, test = city.valid, city.test
    print(f"{args.city}: val {len(val)} | test {len(test)}", flush=True)

    vcache = prep(engine, val)
    tcache = prep(engine, test)
    print(f"route top-20 mean Jaccard overlap: {overlap_stats(tcache):.3f}", flush=True)

    rows = {}
    # single best route (no fusion)
    for route in ["graph", "geo", "memory", "semantic"]:
        rec = {1: 0, 5: 0, 10: 0}
        nd = 0.0
        cand = 0
        for routes, s in zip(tcache, test):
            pois = [p for p, _ in routes.get(route, [])]
            if s.target in pois[:25]:
                cand += 1
            if s.target in pois[:10]:
                pos = pois.index(s.target) + 1
                for k in rec:
                    if pos <= k:
                        rec[k] += 1
                nd += 1.0 / math.log2(pos + 1)
        n = len(test)
        rows[f"single:{route}"] = {
            "cand_recall": cand / n, "HR@1": rec[1] / n, "HR@5": rec[5] / n,
            "HR@10": rec[10] / n, "NDCG@10": nd / n,
        }

    for strategy in ["rrf", "minmax", "softmax", "union"]:
        best = None
        for g, o in WEIGHT_GRID:
            w = {"graph": g, "geo": o, "memory": o, "semantic": o}
            r = score_fusion(strategy, vcache, val, w)
            if best is None or r["HR@10"] > best[0]:
                best = (r["HR@10"], w)
        w = best[1]
        rows[f"{strategy}"] = score_fusion(strategy, tcache, test, w)
        rows[f"{strategy}"]["weights"] = w
        print(f"  {strategy}: {rows[strategy]}", flush=True)

    print("\n| method | cand@25 | HR@1 | HR@5 | HR@10 | NDCG@10 |")
    print("|---|---|---|---|---|---|")
    for k, v in rows.items():
        print(f"| {k} | {v['cand_recall']:.3f} | {v['HR@1']:.3f} | {v['HR@5']:.3f} | {v['HR@10']:.3f} | {v['NDCG@10']:.3f} |")

    out = Path(args.out) if args.out else ROOT / "results" / f"fusion_{args.city}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
