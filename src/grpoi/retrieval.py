"""Multi-route candidate retrieval and reciprocal-rank fusion.

Four complementary evidence sources replace RALLM-POI's single TF-IDF channel:

* ``semantic``  -- trajectory-level TF-IDF neighbour retrieval (the RALLM-POI channel)
* ``graph``     -- global POI transition graph (where do people go next after these POIs?)
* ``geo``       -- spatial neighbourhood with distance decay and heading consistency
* ``memory``    -- the user's own revisit history conditioned on category-hour preference
"""

from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .data import CityData, Sample, haversine_vec
from .tools import category_hour_fit


@dataclass
class Candidate:
    poi: int
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)

    def add_evidence(self, text: str) -> None:
        if text and text not in self.evidence:
            self.evidence.append(text)


def _rrf_fuse(route_lists: dict[str, list[tuple[int, float]]], weights: dict[str, float], k: int) -> list[Candidate]:
    """Reciprocal rank fusion over per-route ranked lists."""
    pool: dict[int, Candidate] = {}
    for route, ranked in route_lists.items():
        w = weights.get(route, 1.0)
        for rank, (poi, raw) in enumerate(ranked, start=1):
            cand = pool.setdefault(poi, Candidate(poi=poi))
            cand.score += w / (k + rank)
            cand.ranks[route] = rank
    return sorted(pool.values(), key=lambda c: c.score, reverse=True)


class SemanticRoute:
    """TF-IDF retrieval over training trajectories (RALLM-POI style)."""

    def __init__(self, city: CityData):
        self.city = city
        hist_path = config.RAG_DIR / f"{city.city}_train_short_term_history"
        rec_path = config.RAG_DIR / f"{city.city}_train_short_term_recommendation"
        with open(hist_path, "rb") as f:
            self.train_hist = pickle.load(f)
        with open(rec_path, "rb") as f:
            self.train_rec = pickle.load(f)
        self.train_texts = [" ".join(map(str, h)) for h in self.train_hist]
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.train_texts)

    def retrieve(self, sample: Sample, top_k: int) -> list[tuple[int, float]]:
        query = " ".join(map(str, sample.recent_pois))
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        order = np.argsort(-sims)[:top_k]
        votes: dict[int, float] = {}
        for i in order:
            if sims[i] <= 0:
                continue
            poi = int(self.train_rec[i])
            votes[poi] = votes.get(poi, 0.0) + float(sims[i])
        return sorted(votes.items(), key=lambda x: x[1], reverse=True)


def hour_bucket(hour: int) -> int:
    """Coarse time-of-day bucket (3h) used by the transition model."""
    return int(hour) // 3


class GraphRoute:
    """Temporal POI transition model.

    Three counting statistics are combined:
      * first-order, conditioned on the hour of day:  P(v | u, bucket(t))
      * first-order, unconditioned:                   P(v | u)
      * second-order:                                 P(v | u_{t-1}, u_t)

    Each is normalised to a probability before mixing, so the weights are
    interpretable.  This is a counting model -- no training, no LLM.
    """

    #: mixing weights for (hour-conditioned, first-order, second-order)
    WEIGHTS = (1.0, 0.3, 1.5)

    def __init__(self, city: CityData):
        self.city = city
        self.t1 = {int(k): v for k, v in city.transitions.items()}
        self.t1h: dict[tuple[int, int], Counter] = {}
        self.t2: dict[tuple[int, int], Counter] = {}
        for s in city.train:
            pois, hours = s.pois, s.hours
            for i in range(1, len(pois)):
                a, b, h = int(pois[i - 1]), int(pois[i]), int(hours[i])
                key = (a, hour_bucket(h))
                self.t1h.setdefault(key, Counter())[b] += 1
                if i >= 2:
                    k2 = (int(pois[i - 2]), a)
                    self.t2.setdefault(k2, Counter())[b] += 1

    @staticmethod
    def _prob(counter: Counter | None) -> dict[int, float]:
        if not counter:
            return {}
        total = sum(counter.values())
        return {p: c / total for p, c in counter.items()}

    def retrieve(self, sample: Sample, top_k: int) -> list[tuple[int, float]]:
        last = int(sample.last_poi)
        hb = hour_bucket(sample.last_hour)
        w1h, w1, w2 = self.WEIGHTS
        scores: dict[int, float] = {}

        def add(probs: dict[int, float], weight: float) -> None:
            if weight <= 0:
                return
            for p, v in probs.items():
                scores[p] = scores.get(p, 0.0) + weight * v

        add(self._prob(self.t1h.get((last, hb))), w1h)
        add(self._prob(self.t1.get(last)), w1)
        if len(sample.recent_pois) >= 2:
            prev = int(sample.recent_pois[-2])
            add(self._prob(self.t2.get((prev, last))), w2)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked


class GeoRoute:
    """Spatial neighbourhood around the last POI, with distance decay and heading."""

    def __init__(self, city: CityData):
        self.city = city

    def retrieve(self, sample: Sample, top_k: int, radius_km: float, scale_km: float) -> list[tuple[int, float]]:
        city = self.city
        last = sample.last_poi
        if not (0 <= last < city.n_pois()):
            return []
        anchor = city.coords[last]
        dist = haversine_vec(anchor, city.coords)
        valid = np.isfinite(dist) & (dist <= radius_km)
        valid[last] = False

        # heading consistency with the most recent movement
        heading_bonus = 1.0
        if len(sample.recent_pois) >= 2:
            prev = sample.recent_pois[-2]
            if 0 <= prev < city.n_pois():
                v = anchor - city.coords[prev]
                norm = np.linalg.norm(v)
                if norm > 0:
                    d = city.coords - anchor
                    dn = np.linalg.norm(d, axis=1) + 1e-9
                    cos = (d @ v) / (dn * norm)
                    heading_bonus = 1.0 + 0.3 * np.clip(cos, -1, 1)

        recent_set = set(sample.recent_pois)
        scores = np.exp(-dist / scale_km) * heading_bonus
        scores = np.where(np.isin(np.arange(len(dist)), list(recent_set)), scores * 0.5, scores)

        idx = np.where(valid)[0]
        idx = idx[np.argsort(-scores[idx])][:top_k]
        return [(int(i), float(scores[i])) for i in idx]


class MemoryRoute:
    """The user's own visit history, conditioned on category-hour preference."""

    def __init__(self, city: CityData):
        self.city = city

    def retrieve(self, sample: Sample, top_k: int) -> list[tuple[int, float]]:
        hist = self.city.user_hist.get(sample.user, [])
        if not hist:
            return []
        hour = sample.last_hour
        counts: Counter = Counter(p for p, _, _ in hist)
        recent_set = set(sample.recent_pois)
        scores = {}
        for poi, cnt in counts.items():
            fit = category_hour_fit(self.city, self.city.poi_cat(poi), hour)
            s = np.log1p(cnt) * (0.25 + fit) * (0.6 if poi in recent_set else 1.0)
            scores[poi] = float(s)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


class RetrievalEngine:
    def __init__(self, city: CityData, cfg: config.RetrievalConfig | None = None):
        self.city = city
        self.cfg = cfg or config.RETRIEVAL
        self.semantic = SemanticRoute(city)
        self.graph = GraphRoute(city)
        self.geo = GeoRoute(city)
        self.memory = MemoryRoute(city)

    def route_lists(self, sample: Sample) -> dict[str, list[tuple[int, float]]]:
        cfg = self.cfg
        return {
            "semantic": self.semantic.retrieve(sample, cfg.rag_top_k),
            "graph": self.graph.retrieve(sample, cfg.graph_top_k),
            "geo": self.geo.retrieve(sample, cfg.geo_top_k, cfg.geo_radius_km, cfg.geo_scale_km),
            "memory": self.memory.retrieve(sample, cfg.memory_top_k),
        }

    def candidates(self, sample: Sample) -> list[Candidate]:
        routes = self.route_lists(sample)
        fused = _rrf_fuse(routes, self.cfg.weights, self.cfg.rrf_k)
        # never propose the POI the user is standing at right now
        fused = [c for c in fused if c.poi != sample.last_poi][: self.cfg.n_candidates]
        self._annotate(sample, fused)
        return fused

    def _annotate(self, sample: Sample, cands: list[Candidate]) -> None:
        """Attach human-readable evidence for the prompt."""
        city = self.city
        last = sample.last_poi
        succ = city.transitions.get(int(last), {})
        for c in cands:
            if "graph" in c.ranks:
                cnt = succ.get(c.poi, 0)
                if cnt:
                    c.add_evidence(f"follows POI {last} in {cnt} training transitions")
            if "memory" in c.ranks:
                visits = sum(1 for p, _, _ in city.user_hist.get(sample.user, []) if p == c.poi)
                if visits:
                    c.add_evidence(f"visited by this user {visits}x before")
            if "geo" in c.ranks:
                c.add_evidence("locally close to the current location")

    def route_agreement(self, routes: dict[str, list[tuple[int, float]]], k: int = 5) -> float:
        """1 - mean pairwise Jaccard overlap of the routes' top-k sets."""
        sets = [set(p for p, _ in r[:k]) for r in routes.values() if r]
        if len(sets) < 2:
            return 1.0
        sims = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                union = sets[i] | sets[j]
                sims.append(len(sets[i] & sets[j]) / len(union) if union else 0.0)
        return float(1.0 - np.mean(sims))
