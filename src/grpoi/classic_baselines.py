"""Classical, training-free baselines for next-POI recommendation.

Every baseline returns a ranked list of 10 POIs over the full POI space, so it is
scored by exactly the same code path as our method.  None of them calls an LLM and
none of them is trained, which makes them the honest floor for a zero-shot system.

* POP        -- global most popular POIs in the training set
* UPop       -- the user's own most frequent POIs (falls back to POP)
* Markov1    -- first-order transition from the last POI (falls back to UPop)
* ItemKNN    -- co-visitation neighbours of the last POI (cosine on counts)
* GeoN       -- the 10 geographically nearest POIs
* CatPop     -- popular POIs restricted to the user's recent categories
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .data import CityData, Sample, haversine_vec


def _topk(counter: Counter, k: int) -> list[int]:
    return [p for p, _ in counter.most_common(k)]


class PopBaseline:
    name = "POP"

    def __init__(self, city: CityData):
        self.city = city
        c: Counter = Counter()
        for s in city.train:
            for p in s.pois:
                c[int(p)] += 1
        self.order = _topk(c, 50)

    def predict(self, s: Sample, k: int = 10) -> list[int]:
        return self.order[:k]


class UserPopBaseline:
    name = "UPop"

    def __init__(self, city: CityData):
        self.city = city
        self.per_user = {}
        for s in city.train:
            c = self.per_user.setdefault(s.user, Counter())
            for p in s.pois:
                c[int(p)] += 1
        self.backoff = PopBaseline(city)

    def predict(self, s: Sample, k: int = 10) -> list[int]:
        c = self.per_user.get(s.user)
        if not c:
            return self.backoff.predict(s, k)
        return _topk(c, k)


class MarkovBaseline:
    name = "Markov1"

    def __init__(self, city: CityData):
        self.city = city
        self.backoff = UserPopBaseline(city)

    def predict(self, s: Sample, k: int = 10) -> list[int]:
        score: Counter = Counter()
        for pos, src in enumerate(reversed(s.recent_pois[-3:])):
            w = 0.5 ** pos
            for dst, cnt in self.city.transitions.get(int(src), {}).items():
                score[int(dst)] += w * cnt
        out = _topk(score, k)
        if len(out) < k:
            for p in self.backoff.predict(s, k):
                if p not in out:
                    out.append(p)
                if len(out) == k:
                    break
        return out[:k]


class ItemKNNBaseline:
    """Co-visitation similarity: sim(a,b) = c(a,b) / sqrt(c(a)*c(b))."""

    name = "ItemKNN"

    def __init__(self, city: CityData):
        self.city = city
        self.sim = defaultdict(dict)
        pair: Counter = Counter()
        deg: Counter = Counter()
        for s in city.train:
            for a, b in zip(s.pois[:-1], s.pois[1:]):
                a, b = int(a), int(b)
                if a == b:
                    continue
                pair[(a, b)] += 1
                deg[a] += 1
                deg[b] += 1
        for (a, b), c in pair.items():
            w = c / np.sqrt(deg[a] * deg[b])
            # symmetric
            self.sim[a][b] = w
            if w > self.sim[b].get(a, 0.0):
                self.sim[b][a] = w
        self.backoff = PopBaseline(city)

    def predict(self, s: Sample, k: int = 10) -> list[int]:
        score: Counter = Counter()
        for pos, src in enumerate(reversed(s.recent_pois[-3:])):
            w = 0.5 ** pos
            for dst, sc in self.sim.get(int(src), {}).items():
                score[dst] += w * sc
        out = _topk(score, k)
        if len(out) < k:
            for p in self.backoff.order:
                if p not in out:
                    out.append(p)
                if len(out) == k:
                    break
        return out[:k]


class GeoNearestBaseline:
    name = "GeoN"

    def __init__(self, city: CityData):
        self.city = city

    def predict(self, s: Sample, k: int = 10) -> list[int]:
        city = self.city
        last = s.last_poi
        if not (0 <= last < city.n_pois()):
            return []
        d = haversine_vec(city.coords[last], city.coords)
        d[last] = np.inf
        idx = np.argsort(d)[:k]
        return [int(i) for i in idx if np.isfinite(d[i])]


class CatPopBaseline:
    """Popular POIs among the categories the user visited recently."""

    name = "CatPop"

    def __init__(self, city: CityData):
        self.city = city
        self.by_cat: dict[int, list[int]] = defaultdict(list)
        c: Counter = Counter()
        for s in city.train:
            for p in s.pois:
                c[int(p)] += 1
        for cat in range(len(city.cat_names)):
            self.by_cat[cat] = sorted(
                (p for p in c if city.poi_cat(p) == cat), key=lambda p: c[p], reverse=True
            )
        self.backoff = PopBaseline(city)

    def predict(self, s: Sample, k: int = 10) -> list[int]:
        cats = list(dict.fromkeys(s.recent_cats[::-1]))
        out: list[int] = []
        for cat in cats:
            for p in self.by_cat.get(cat, [])[:k]:
                if p not in out:
                    out.append(p)
            if len(out) >= k:
                break
        if len(out) < k:
            for p in self.backoff.order:
                if p not in out:
                    out.append(p)
                if len(out) == k:
                    break
        return out[:k]


class TemporalMarkovBaseline:
    """Hour-conditioned 2nd-order transition model (the strongest classical baseline)."""

    name = "TemporalMarkov"

    def __init__(self, city: CityData):
        from .retrieval import GraphRoute

        self.route = GraphRoute(city)

    def predict(self, s: Sample, k: int = 10) -> list[int]:
        return [p for p, _ in self.route.retrieve(s, k)]


ALL_BASELINES = [
    PopBaseline,
    UserPopBaseline,
    MarkovBaseline,
    ItemKNNBaseline,
    GeoNearestBaseline,
    CatPopBaseline,
    TemporalMarkovBaseline,
]
