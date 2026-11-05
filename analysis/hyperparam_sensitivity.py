"""Hyper-parameter sensitivity (free part: no LLM calls).

Measures how the evidence / candidate coverage responds to

  1. the spatial decay scale lambda and the retrieval radius
  2. the transition mixing weights (they affect the candidate pool, not the
     evidence text -- the evidence line uses raw first-order counts)
  3. the number of items rendered per evidence view

Coverage = the fraction of samples whose target appears in the corresponding set.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grpoi.data import load_city  # noqa: E402
from grpoi.retrieval import GraphRoute, RetrievalEngine  # noqa: E402


def evidence_union(city, sample, engine, k):
    last = int(sample.last_poi)
    succ = [p for p, _ in Counter(city.transitions.get(last, {})).most_common(k)]
    ucnt = Counter(p for p, _, _ in city.user_hist.get(sample.user, []))
    usr = [p for p, _ in ucnt.most_common(k)]
    r = engine.route_lists(sample)
    geo = [p for p, _ in r.get("geo", [])[:k]]
    sem = [p for p, _ in r.get("semantic", [])[:k]]
    return set(succ) | set(usr) | set(geo) | set(sem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="NYC")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    city = load_city(args.city)
    samples = getattr(city, args.split)[: args.n]
    engine = RetrievalEngine(city)
    n = len(samples)

    def frac(hits):
        return hits / n

    # ---------------- 1) lambda x radius ----------------
    print(f"\n[{args.city}/{args.split} n={n}] 1) 空间尺度 λ 与半径")
    print(f"{'λ':>6s} {'半径':>6s} {'geo@8':>7s} {'证据并集@8':>11s} {'候选池':>7s}")
    base = (engine.cfg.geo_scale_km, engine.cfg.geo_radius_km)
    for lam, rad in product([0.5, 1.0, 2.5, 5.0, 10.0], [2.0, 5.0, 10.0, 20.0]):
        engine.cfg.geo_scale_km = lam
        engine.cfg.geo_radius_km = rad
        g = e = p = 0
        for s in samples:
            r = engine.route_lists(s)
            geo = [x for x, _ in r.get("geo", [])[:8]]
            g += s.target in geo
            e += s.target in evidence_union(city, s, engine, 8)
            p += s.target in [c.poi for c in engine.candidates(s)]
        print(f"{lam:6.1f} {rad:6.1f} {frac(g):7.3f} {frac(e):11.3f} {frac(p):7.3f}")
    engine.cfg.geo_scale_km, engine.cfg.geo_radius_km = base

    # ---------------- 2) transition weights ----------------
    print("\n2) 转移混合权重 (w1=1.0 固定)")
    print(f"{'w2':>5s} {'w3':>5s} {'转移@8':>7s} {'候选池':>7s}")
    orig = GraphRoute.WEIGHTS
    for w2, w3 in product([0.0, 0.3, 0.75, 1.5], [0.0, 0.75, 1.5, 3.0]):
        GraphRoute.WEIGHTS = (1.0, w2, w3)
        tr = p = 0
        for s in samples:
            r = engine.route_lists(s)
            tr += s.target in [x for x, _ in r.get("graph", [])[:8]]
            p += s.target in [c.poi for c in engine.candidates(s)]
        print(f"{w2:5.2f} {w3:5.2f} {frac(tr):7.3f} {frac(p):7.3f}")
    GraphRoute.WEIGHTS = orig

    # ---------------- 3) items per view ----------------
    print("\n3) 每视图证据条数")
    print(f"{'k':>4s} {'证据并集':>9s} {'转移':>6s} {'个体':>6s} {'空间':>6s} {'语义':>6s}")
    for k in (3, 5, 8, 12, 16, 24):
        u = tr = us = ge = se = 0
        for s in samples:
            last = int(s.last_poi)
            succ = [x for x, _ in Counter(city.transitions.get(last, {})).most_common(k)]
            ucnt = Counter(x for x, _, _ in city.user_hist.get(s.user, []))
            usr = [x for x, _ in ucnt.most_common(k)]
            r = engine.route_lists(s)
            geo = [x for x, _ in r.get("geo", [])[:k]]
            sem = [x for x, _ in r.get("semantic", [])[:k]]
            tr += s.target in succ
            us += s.target in usr
            ge += s.target in geo
            se += s.target in sem
            u += s.target in (set(succ) | set(usr) | set(geo) | set(sem))
        print(f"{k:4d} {frac(u):9.3f} {frac(tr):6.3f} {frac(us):6.3f} {frac(ge):6.3f} {frac(se):6.3f}")


if __name__ == "__main__":
    main()
