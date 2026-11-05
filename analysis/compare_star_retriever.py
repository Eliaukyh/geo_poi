"""Compare our semantic retrieval with RALLM-STAR's HybridTemporalRAG retriever.

Both operate on the same prebuilt pairs:
  experiments_rag/{city}_train_short_term_history      (trajectory ids)
  experiments_rag/{city}_train_short_term_recommendation (its next POI)

Variants:
  single      our current view: one TF-IDF over full history, top-50, vote next stops
  dual_rrf    STAR without GMM: TF-IDF over full + last-40, RRF fused
  dual_rrf_gmm  STAR with GMM spatial rerank (slower; --gmm to enable)

Metric: does the target appear among the top-k next-stops voted from the retrieved
trajectories?  No LLM calls involved.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grpoi import config  # noqa: E402
from grpoi.data import load_city  # noqa: E402


def load_pairs(city: str):
    hist = pickle.load(open(config.RAG_DIR / f"{city}_train_short_term_history", "rb"))
    rec = pickle.load(open(config.RAG_DIR / f"{city}_train_short_term_recommendation", "rb"))
    return hist, [int(r) for r in rec]


def rrf_fuse(short_rank, long_rank, alpha=0.6, top_k=10, k=60):
    score = defaultdict(float)
    for r, idx in enumerate(short_rank[:200]):
        score[idx] += alpha / (k + r)
    for r, idx in enumerate(long_rank[:200]):
        score[idx] += (1 - alpha) / (k + r)
    return [i for i, _ in sorted(score.items(), key=lambda x: -x[1])[:top_k]]


def votes_to_candidates(indices, rec, sims, top_n=8):
    score = defaultdict(float)
    for i in indices:
        score[rec[i]] += float(sims[i]) if sims is not None else 1.0
    return [p for p, _ in sorted(score.items(), key=lambda x: -x[1])[:top_n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="NYC")
    ap.add_argument("--split", default="test")
    ap.add_argument("--top_n", type=int, default=8)
    ap.add_argument("--gmm", action="store_true", help="also test STAR's GMM spatial rerank")
    ap.add_argument("--gmm_pool", type=int, default=64)
    ap.add_argument("--gmm_weight", type=float, default=0.2)
    args = ap.parse_args()

    city = load_city(args.city)
    samples = getattr(city, args.split)
    hist, rec = load_pairs(args.city)
    print(f"{args.city}/{args.split}: {len(samples)} samples | {len(hist)} training pairs")

    texts = [" ".join(map(str, h)) for h in hist]
    long_texts = [" ".join(map(str, h[-40:])) for h in hist]
    vec_s = TfidfVectorizer().fit(texts)
    vec_l = TfidfVectorizer().fit(long_texts)
    M_s, M_l = vec_s.transform(texts), vec_l.transform(long_texts)

    cov = {name: 0 for name in ("single", "dual_rrf")}
    for s in samples:
        q = " ".join(map(str, s.recent_pois))
        ql = " ".join(map(str, s.recent_pois[-40:]))

        sim_s = cosine_similarity(vec_s.transform([q]), M_s)[0]
        order_s = np.argsort(-sim_s)[:50]
        cand_single = votes_to_candidates(order_s, rec, sim_s, args.top_n)
        if s.target in cand_single:
            cov["single"] += 1

        sim_l = cosine_similarity(vec_l.transform([ql]), M_l)[0]
        fused = rrf_fuse(np.argsort(-sim_s), np.argsort(-sim_l), top_k=50)
        cand_dual = votes_to_candidates(fused, rec, None, args.top_n)
        if s.target in cand_dual:
            cov["dual_rrf"] += 1

    if args.gmm:
        print("  fitting GMMs for training trajectories ...", flush=True)
        coords_all = city.coords
        models = []
        for h in hist:
            idx = [p for p in h if 0 <= p < len(coords_all)]
            if len(idx) < 2:
                models.append(None)
                continue
            try:
                g = GaussianMixture(n_components=min(3, len(idx)), covariance_type="full", random_state=0)
                g.fit(coords_all[idx])
                models.append(g)
            except Exception:
                models.append(None)
        fitted = sum(m is not None for m in models)
        print(f"  fitted {fitted}/{len(models)} models", flush=True)
        cov["dual_gmm"] = 0
        for s in samples:
            q = " ".join(map(str, s.recent_pois))
            ql = " ".join(map(str, s.recent_pois[-40:]))
            sim_s = cosine_similarity(vec_s.transform([q]), M_s)[0]
            sim_l = cosine_similarity(vec_l.transform([ql]), M_l)[0]

            # base RRF score over both streams (STAR: alpha for short stream)
            base = defaultdict(float)
            for r, i in enumerate(np.argsort(-sim_s)[:200]):
                base[i] += 0.6 / (60 + r)
            for r, i in enumerate(np.argsort(-sim_l)[:200]):
                base[i] += 0.4 / (60 + r)
            pool = [i for i, _ in sorted(base.items(), key=lambda x: -x[1])[: args.gmm_pool]]

            q_idx = [p for p in s.recent_pois if 0 <= p < len(coords_all)]
            if len(q_idx) >= 2 and pool:
                qc = coords_all[q_idx]
                ll = np.array([
                    float(np.mean(models[i].score_samples(qc))) if models[i] is not None else np.nan
                    for i in pool
                ])
                finite = np.isfinite(ll)
                if finite.any():
                    lo, hi = ll[finite].min(), ll[finite].max()
                    rng = hi - lo if hi - lo > 1e-9 else 1.0
                    norm = np.where(finite, (ll - lo) / rng, 0.0)
                    combined = {i: base[i] + args.gmm_weight * float(norm[k]) for k, i in enumerate(pool)}
                    pool = [i for i, _ in sorted(combined.items(), key=lambda x: -x[1])]

            cand = votes_to_candidates(pool[:50], rec, None, args.top_n)
            if s.target in cand:
                cov["dual_gmm"] += 1

    n = len(samples)
    print(f"\n候选覆盖（top-{args.top_n}，无 LLM）:")
    for k, v in cov.items():
        print(f"  {k:10s} {v/n:.3f}")


if __name__ == "__main__":
    main()
