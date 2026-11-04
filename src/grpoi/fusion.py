"""Fusion strategies for multi-route retrieval.

RRF only uses ranks and assumes the routes are of comparable quality.  In POI
retrieval that assumption fails: the temporal-transition route is far stronger
than the others, and the geographic route carries meaningful magnitude
(100 m vs 5 km both collapse to adjacent ranks under RRF).

Strategies implemented:
  rrf        -- reciprocal rank fusion (baseline)
  minmax     -- per-route min-max normalisation, then weighted sum (keeps magnitude)
  softmax    -- per-route softmax to a probability scale, then weighted sum
  union      -- round-robin union, ordered by the strongest route (reachability only)
"""

from __future__ import annotations

import numpy as np


def rrf(route_lists, weights, k=60):
    score = {}
    for route, ranked in route_lists.items():
        w = weights.get(route, 1.0)
        if w <= 0:
            continue
        for rank, (poi, _) in enumerate(ranked, start=1):
            score[poi] = score.get(poi, 0.0) + w / (k + rank)
    return sorted(score.items(), key=lambda x: -x[1])


def _minmax(ranked):
    vals = np.array([s for _, s in ranked], dtype=float)
    if len(vals) == 0:
        return {}
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return {p: 1.0 for p, _ in ranked}
    return {p: float((s - lo) / (hi - lo)) for p, s in ranked}


def minmax(route_lists, weights):
    score = {}
    for route, ranked in route_lists.items():
        w = weights.get(route, 1.0)
        if w <= 0:
            continue
        for poi, v in _minmax(ranked).items():
            score[poi] = score.get(poi, 0.0) + w * v
    return sorted(score.items(), key=lambda x: -x[1])


def softmax(route_lists, weights, temp=0.5):
    score = {}
    for route, ranked in route_lists.items():
        w = weights.get(route, 1.0)
        if w <= 0 or not ranked:
            continue
        vals = np.array([s for _, s in ranked], dtype=float)
        z = (vals - vals.max()) / max(temp * (vals.std() + 1e-9), 1e-9)
        p = np.exp(z)
        p = p / p.sum()
        for (poi, _), pv in zip(ranked, p):
            score[poi] = score.get(poi, 0.0) + w * float(pv)
    return sorted(score.items(), key=lambda x: -x[1])


def union(route_lists, weights, k=60):
    """Round-robin over routes, but the strongest route sets the head order."""
    order = sorted(route_lists, key=lambda r: -weights.get(r, 1.0))
    head = [p for p, _ in route_lists[order[0]]] if order else []
    seen, out = set(), []
    for p in head:
        if p not in seen:
            seen.add(p)
            out.append(p)
    queues = [list(route_lists[r]) for r in order[1:] if route_lists[r]]
    i = 0
    while queues and len(out) < 200:
        any_left = False
        for qi, q in enumerate(queues):
            if i < len(q):
                any_left = True
                p = q[i][0]
                if p not in seen:
                    seen.add(p)
                    out.append(p)
        if not any_left:
            break
        i += 1
    return [(p, 0.0) for p in out]


STRATEGIES = {"rrf": rrf, "minmax": minmax, "softmax": softmax, "union": union}


def fuse(strategy: str, route_lists, weights, **kw):
    fn = STRATEGIES[strategy]
    if strategy == "rrf":
        return fn(route_lists, weights, k=kw.get("rrf_k", 60))
    if strategy == "softmax":
        return fn(route_lists, weights, temp=kw.get("temp", 0.5))
    return fn(route_lists, weights)
