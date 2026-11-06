"""Generative mode: the LLM produces POI ids from its own knowledge.

Difference from `agent.py` (constrained selection):
  * retrieval does NOT hand the model a candidate list to choose from;
    it hands an *evidence pack* (transitions, user history, nearby POIs,
    similar trajectories) written as reference facts;
  * the model may output any POI id in the city;
  * invalid ids (unknown / duplicate / too few) are repaired afterwards
    (policy B): unknown ids are dropped and replaced from the evidence pool,
    but the model's own choices are never overwritten.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from . import config
from .data import CityData, Sample
from .llm import LLMClient
from .retrieval import RetrievalEngine
from .tools import build_facts, summarize_trajectory

SYSTEM = (
    "You are an expert urban mobility analyst. You predict the next point of interest "
    "of a user from their check-in trajectory and reference evidence. Answer with JSON only."
)


def _fmt_pois(city: CityData, pois, scores=None, limit=8, with_dist=None) -> str:
    out = []
    for i, p in enumerate(list(pois)[:limit]):
        p = int(p)
        cat = city.cat_name(city.poi_cat(p))
        extra = ""
        if with_dist is not None and p < city.n_pois():
            extra = f", {with_dist[p]:.2f}km"
        out.append(f"{p} ({cat}{extra})")
    return "; ".join(out) if out else "none"


def evidence_block(city: CityData, sample: Sample, engine: RetrievalEngine, per_view: int = 8,
                   drop: tuple[str, ...] = (), transition_evidence: str = "counts") -> str:
    """Render retrieval output as facts, not as a menu to pick from."""
    routes = engine.route_lists(sample)
    last = sample.last_poi
    hour = sample.last_hour

    succ = city.transitions.get(int(last), {})
    if transition_evidence == "score":
        # hour-conditioned first-order + second-order mix (GraphRoute.WEIGHTS)
        succ_top = [p for p, _ in routes.get("graph", [])[:per_view]]
    else:
        succ_top = [p for p, _ in Counter(succ).most_common(per_view)]

    hist = city.user_hist.get(sample.user, [])
    ucount = Counter(p for p, _, _ in hist)
    u_top = [p for p, _ in ucount.most_common(per_view)]

    geo = routes.get("geo", [])
    from .data import haversine_vec
    import numpy as np
    dist = haversine_vec(city.coords[last], city.coords) if 0 <= last < city.n_pois() else None

    sem = routes.get("semantic", [])[:per_view]

    parts = []
    if "transition" not in drop:
        parts.append(f"- After POI {last}, people most often went to: "
                     f"{_fmt_pois(city, succ_top, with_dist=dist, limit=per_view)}")
    if "memory" not in drop:
        if u_top:
            parts.append(f"- This user has visited most often: {_fmt_pois(city, u_top, limit=per_view)}")
        else:
            parts.append("- This user has no recorded history (cold start).")
    if "geo" not in drop:
        parts.append(f"- Near the current location: {_fmt_pois(city, [p for p, _ in geo], with_dist=dist, limit=per_view)}")
    if "semantic" not in drop:
        parts.append(f"- Users with similar recent trajectories next visited: "
                     f"{_fmt_pois(city, [p for p, _ in sem], with_dist=dist, limit=per_view)}")
    return "\n".join(parts) if parts else "- (no evidence available)"


def generation_prompt(city: CityData, sample: Sample, engine: RetrievalEngine,
                      profile_text: str | None = None, per_view: int = 8,
                      drop: tuple[str, ...] = (), transition_evidence: str = "counts") -> str:
    hist = city.user_hist.get(sample.user, [])
    if profile_text:
        profile = profile_text
    elif hist:
        c = Counter(city.poi_cat(p) for p, _, _ in hist)
        top = ", ".join(f"{city.cat_name(k)} ({v / len(hist):.0%})" for k, v in c.most_common(2))
        profile = f"This user has {len(hist)} historical check-ins; dominant categories: {top}."
    else:
        profile = "This user has no history in the training period (cold-start user)."

    return f"""A user in {city.city} is moving through the city. Predict their next point of interest.

Recent check-ins (oldest to newest): {summarize_trajectory(city, sample.recent_pois)}
Current hour of day: {sample.last_hour}:00
{profile}

Reference evidence retrieved from historical check-ins (use it as a hint; you may also rely on your own knowledge of this city):
{evidence_block(city, sample, engine, per_view, drop, transition_evidence)}

Instructions:
1. Recommend exactly 10 distinct POI ids, most likely first.
2. Use POI ids that exist in this city. You are not limited to the ids listed above.
3. Prefer places that are plausible given the hour, the recent trajectory and the user's habits.
4. Do not include line breaks inside the JSON.

Return JSON only: {{"recommendation": [10 integer ids, most likely first], "reason": "one short sentence"}}"""


class GenerativeAgent:
    """Evidence-grounded free generation with post-hoc validity repair (policy B)."""

    def __init__(self, city: CityData, engine: RetrievalEngine | None = None,
                 client: LLMClient | None = None, cfg: config.AgentConfig | None = None):
        self.city = city
        self.cfg = cfg or config.AGENT
        self.engine = engine or RetrievalEngine(city)
        self.client = client or LLMClient(self.cfg)

    # ---------------------------------------------------------------- repair
    def repair(self, ids: list[int], fill_pool: list[int], k: int = 10) -> tuple[list[int], int, int]:
        """Drop unknown/duplicate ids, then fill the tail from the evidence pool."""
        n = self.city.n_pois()
        out, invalid = [], 0
        for x in ids:
            if not (0 <= x < n) or x in out:
                invalid += 1
                continue
            out.append(x)
        padded = 0
        for p in fill_pool:
            if len(out) >= k:
                break
            if p not in out:
                out.append(int(p))
                padded += 1
        return out[:k], invalid, padded

    # -------------------------------------------------------------- main entry
    def run(self, sample: Sample, target: int | None = None, profile_text: str | None = None):
        from .agent import Prediction

        pred = Prediction()
        cands = self.engine.candidates(sample)
        pred.fusion_order = [c.poi for c in cands]
        pred.candidate_recall = bool(target is not None and target in pred.fusion_order)

        data, usage = self.client.chat_json(
            SYSTEM, generation_prompt(self.city, sample, self.engine, profile_text)
        )
        pred.usage.add(usage)
        data = data or {}
        pred.reason = str(data.get("reason", ""))

        raw = data.get("recommendation") or data.get("ranking") or []
        ids: list[int] = []
        for x in raw:
            try:
                ids.append(int(str(x).strip()))
            except (TypeError, ValueError):
                continue

        repaired, invalid, padded = self.repair(ids, pred.fusion_order)
        pred.ranking = repaired
        pred.invalid_ids = invalid
        pred.llm_order = repaired
        return pred
