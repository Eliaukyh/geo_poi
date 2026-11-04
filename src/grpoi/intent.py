"""LLM as an intent generator, not a re-ranker.

Rationale: a re-ranker sees exactly the same facts as the retriever and can only
add noise (measured: HR@1 0.542 -> 0.500 at alpha=0.75).  Here the LLM instead
produces information the retriever does not have -- *what the user is about to
do* -- and the retrieval layer grounds each intent into concrete POIs by
category, radius and popularity.  The LLM never emits a POI id itself.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import config
from .data import CityData, Sample, haversine_vec
from .llm import LLMClient
from .tools import summarize_trajectory

SYSTEM = (
    "You are an urban mobility analyst. Given a user's recent check-ins you infer "
    "what the user is most likely to do next, in terms of activity category and a "
    "plausible travel radius. You never invent POI ids. Answer with JSON only."
)


def intent_prompt(city: CityData, sample: Sample, n_intents: int = 4) -> str:
    cats = ", ".join(city.cat_names)
    recent = summarize_trajectory(city, sample.recent_pois, max_len=6)
    hist = city.user_hist.get(sample.user, [])
    if hist:
        c = Counter(city.poi_cat(p) for p, _, _ in hist)
        top = ", ".join(f"{city.cat_name(k)} ({v / len(hist):.0%})" for k, v in c.most_common(2))
        profile = f"The user has {len(hist)} historical check-ins; dominant categories: {top}."
    else:
        profile = "This user has no history (cold-start user)."

    return f"""A user in {city.city} is moving through the city.

Recent check-ins (oldest to newest): {recent}
Current hour of day: {sample.last_hour}:00
{profile}

Task: infer the {n_intents} most plausible intentions for this user's NEXT visit.
For each intention give:
- "category": exactly one of [{cats}]
- "radius_km": a plausible search radius around the current location (0.2 to 10)
- "reason": at most 8 words

Rules:
- The category must be copied exactly from the list above.
- Vary the intentions; do not repeat the same category.
- Base them on the time of day, the recent trajectory and the user's habits.

Return JSON only: {{"intents": [{{"category": "...", "radius_km": 2.0, "reason": "..."}}]}}"""


@dataclass
class Intent:
    category: str
    radius_km: float
    reason: str = ""


@dataclass
class IntentCandidates:
    intents: list[Intent] = field(default_factory=list)
    pois: list[tuple[int, float]] = field(default_factory=list)  # (poi, score) ranked
    usage: object = None


class IntentExpander:
    """Turn LLM intents into concrete POI candidates (LLM never outputs ids)."""

    def __init__(self, city: CityData, client: LLMClient | None = None, per_intent: int = 5):
        self.city = city
        self.client = client or LLMClient()
        self.per_intent = per_intent
        self.pop: Counter = Counter()
        for s in city.train:
            for p in s.pois:
                self.pop[int(p)] += 1
        self.cat_to_id = {name: i for i, name in enumerate(city.cat_names)}

    def _parse(self, data: dict | None) -> list[Intent]:
        out = []
        for it in (data or {}).get("intents", [])[:6]:
            cat = str(it.get("category", "")).strip()
            if cat not in self.cat_to_id:
                continue
            try:
                r = float(it.get("radius_km", 2.0))
            except (TypeError, ValueError):
                r = 2.0
            out.append(Intent(category=cat, radius_km=min(max(r, 0.2), 10.0), reason=str(it.get("reason", ""))[:80]))
        return out

    def _ground(self, sample: Sample, intents: list[Intent], per_intent: int | None) -> list[tuple[int, float]]:
        k = per_intent or self.per_intent
        city = self.city
        last = sample.last_poi
        if not (0 <= last < city.n_pois()):
            return []
        anchor = city.coords[last]
        dist = haversine_vec(anchor, city.coords)
        recent = set(sample.recent_pois)
        score: dict[int, float] = {}
        for it in intents:
            cid = self.cat_to_id[it.category]
            mask = (city.cat_ids == cid) & np.isfinite(dist) & (dist <= it.radius_km)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            # rank by popularity, penalise POIs already in the recent window
            vals = np.array(
                [self.pop.get(int(i), 0) * (0.5 if int(i) in recent else 1.0) for i in idx],
                dtype=float,
            )
            if vals.max() <= 0:
                continue
            order = np.argsort(-vals)[:k]
            for rank, j in enumerate(order, start=1):
                poi = int(idx[j])
                s = 1.0 / rank
                score[poi] = max(score.get(poi, 0.0), s)
        return sorted(score.items(), key=lambda x: -x[1])

    def expand(self, sample: Sample, per_intent: int | None = None) -> IntentCandidates:
        prompt = intent_prompt(self.city, sample)
        data, usage = self.client.chat_json(SYSTEM, prompt)
        intents = self._parse(data)
        pois = self._ground(sample, intents, per_intent)
        return IntentCandidates(intents=intents, pois=pois, usage=usage)
