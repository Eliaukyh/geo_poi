"""Two-stage / prior-anchored fusion under a strong-channel prior.

Motivation (see results/FUSION_ANALYSIS.md):
  * equal-weight RRF destroys the strong temporal channel (NYC HR@1 0.538 -> 0.302)
  * weighted RRF recovers, but conversion (HR@10 / cand@25) drops 0.983 -> 0.948
  * the remaining headroom is reachability: cand@25 0.84 vs HR@10 0.80

Strategies compared:
  graph_only    -- strong channel alone (reference)
  graph_eps     -- s = g + eps * a          (aux only breaks near-ties / fills)
  blend_fixed   -- s = alpha*g + (1-alpha)*a
  blend_adapt   -- alpha varies per instance with the confidence of g
  two_stage     -- top-k from g, then fill remaining slots from a
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grpoi.data import load_city  # noqa: E402
from grpoi.retrieval import RetrievalEngine  # noqa: E402

AUX = ("memory", "geo", "semantic")


def _minmax(lst):
    if not lst:
        return {}
    vals = np.array([s for _, s in lst], dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return {p: 1.0 for p, _ in lst}
    return {p: float((s - lo) / (hi - lo)) for p, s in lst}


def scores_for(routes, aux_w=(1.0, 0.5, 0.5)):
    """Strong-channel score g and auxiliary score a, both in [0, 1]."""
    g = _minmax(routes.get("graph", []))
    a = defaultdict(float)
    for name, w in zip(AUX, aux_w):
        for p, v in _minmax(routes.get(name, [])).items():
            a[p] += w * v
    if a:
        m = max(a.values()) or 1.0
        a = {p: v / m for p, v in a.items()}
    return g, dict(a)


def graph_confidence(g, top=10):
    """1 - normalised entropy of the strong channel's top-k; high = confident."""
    vals = sorted(g.values(), reverse=True)[:top]
    if len(vals) < 2 or sum(vals) <= 0:
        return 0.0
    p = np.array(vals, dtype=float)
    p = p / p.sum()
    ent = -(p * np.log(p + 1e-12)).sum() / math.log(len(p))
    return 1.0 - float(ent)


def rank_two_stage(g, a, k_graph=8, k=10):
    head = sorted(g, key=lambda p: -g[p])[:k_graph]
    rest = sorted((p for p in a if p not in head), key=lambda p: -a[p])
    tail = sorted((p for p in g if p not in head), key=lambda p: -g[p])
    out = head[:]
    for p in rest:
        if len(out) >= k:
            break
        out.append(p)
    for p in tail:
        if len(out) >= k:
            break
        out.append(p)
    return out[:k]


def rank_eps(g, a, eps):
    s = defaultdict(float)
    for p, v in g.items():
        s[p] += v
    if eps > 0:
        for p, v in a.items():
            s[p] += eps * v
    return [p for p, _ in sorted(s.items(), key=lambda x: -x[1])]


def rank_blend(g, a, alpha):
    s = defaultdict(float)
    if alpha > 0:
        for p, v in g.items():
            s[p] += alpha * v
    if alpha < 1:
        for p, v in a.items():
            s[p] += (1 - alpha) * v
    return [p for p, _ in sorted(s.items(), key=lambda x: -x[1])]


def rank_adapt(g, a, lo=0.25, hi=0.75, alpha_min=0.6, alpha_max=0.95):
    c = graph_confidence(g)
    t = min(max((c - lo) / (hi - lo + 1e-9), 0.0), 1.0)
    alpha = alpha_max - (alpha_max - alpha_min) * t  # confident -> more graph
    return rank_blend(g, a, alpha)


def evaluate(cache, samples, ranker, k=10):
    hr = {1: 0, 5: 0, 10: 0}
    nd = 0.0
    cand = 0
    for routes, s in zip(cache, samples):
        g, a = scores_for(routes)
        pois = ranker(g, a)
        if s.target in pois[:25]:
            cand += 1
        if s.target in pois[:k]:
            pos = pois.index(s.target) + 1
            for kk in hr:
                if pos <= kk:
                    hr[kk] += 1
            nd += 1.0 / math.log2(pos + 1)
    n = len(samples)
    return {**{f"HR@{k}": hr[k] / n for k in hr}, "NDCG@10": nd / n, "cand": cand / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="NYC")
    args = ap.parse_args()

    city = load_city(args.city)
    eng = RetrievalEngine(city)
    print(f"{args.city}: val {len(city.valid)} test {len(city.test)}", flush=True)
    vcache = [eng.route_lists(s) for s in city.valid]
    tcache = [eng.route_lists(s) for s in city.test]

    cands = {
        "graph_only": (lambda g, a: rank_blend(g, a, 1.0)),
        "two_stage": (lambda g, a: rank_two_stage(g, a)),
        "adapt": (lambda g, a: rank_adapt(g, a)),
    }
    for eps in (0.01, 0.05, 0.1, 0.3):
        cands[f"eps={eps}"] = (lambda g, a, e=eps: rank_eps(g, a, e))
    for al in (0.7, 0.8, 0.9):
        cands[f"blend={al}"] = (lambda g, a, x=al: rank_blend(g, a, x))

    print("\n| strategy | HR@1 | HR@5 | HR@10 | NDCG@10 | conv |")
    print("|---|---|---|---|---|---|")
    out = {}
    for name, fn in cands.items():
        r_test = evaluate(tcache, city.test, fn)
        r_val = evaluate(vcache, city.valid, fn)
        out[name] = {"test": r_test, "val": r_val}
        conv = r_test["HR@10"] / r_test["cand"] if r_test["cand"] else 0
        print(f"| {name} | {r_test['HR@1']:.3f} | {r_test['HR@5']:.3f} | {r_test['HR@10']:.3f} "
              f"| {r_test['NDCG@10']:.3f} | {conv:.3f} |")

    p = ROOT / "results" / f"two_stage_{args.city}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
