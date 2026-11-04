"""Adaptive agentic re-ranking with a retrieval prior anchor.

Design decisions (each is an ablation in the paper):

1. The four retrieval routes are fused first; the fused order is a *prior*.
2. The LLM sees the candidate list together with its fusion rank and evidence,
   and returns a re-ordering plus a self-reported confidence.
3. The final order is a convex blend
       score = alpha * fusion_prior + (1 - alpha) * llm_order
   so the LLM can nudge but cannot destroy a good retrieval ranking.  With
   alpha = 1 this degenerates to pure retrieval; alpha = 0 is pure LLM reranking.
4. The critique pass (tool-grounded verification) fires only when the LLM
   reports low confidence or its output fails validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config
from .data import CityData, Sample
from .llm import LLMClient, Usage
from .retrieval import Candidate, RetrievalEngine
from .tools import build_facts, summarize_trajectory, violations

SYSTEM = (
    "You are an expert urban mobility analyst. You predict where a person goes next "
    "using their check-in trajectory and retrieved evidence. You always answer with "
    "a JSON object and you never invent POI ids that are not in the candidate list."
)


@dataclass
class Prediction:
    ranking: list[int] = field(default_factory=list)
    reason: str = ""
    verified: bool = False
    confidence: float = 0.0
    uncertainty: float = 0.0
    invalid_ids: int = 0
    usage: Usage = field(default_factory=Usage)
    candidate_recall: bool = False
    fusion_order: list[int] = field(default_factory=list)
    llm_order: list[int] = field(default_factory=list)


def _candidate_block(city: CityData, sample: Sample, cands: list[Candidate]) -> str:
    hour = sample.last_hour
    lines = []
    for i, c in enumerate(cands):
        f = build_facts(city, c.poi, sample.last_poi, hour)
        ev = "; ".join(c.evidence[:3]) if c.evidence else "no strong evidence"
        lines.append(
            f"[C{i + 1}] id={c.poi} | {f['cat']} | {f['dist_km']}km | ~{int(f['travel_min'])}min | "
            f"hour-fit={f['cat_hour_p']:.2f} | {ev}"
        )
    return "\n".join(lines)


def _user_profile(city: CityData, sample: Sample) -> str:
    hist = city.user_hist.get(sample.user, [])
    if not hist:
        return "This user has no history in the training period (cold-start user)."
    cats = [city.poi_cat(p) for p, _, _ in hist]
    top: dict[int, int] = {}
    for c in cats:
        top[c] = top.get(c, 0) + 1
    ranked = sorted(top.items(), key=lambda x: x[1], reverse=True)[:2]
    desc = ", ".join(f"{city.cat_name(c)} ({n / len(hist):.0%})" for c, n in ranked)
    return f"This user has {len(hist)} historical check-ins; dominant categories: {desc}."


def ranking_prompt(city: CityData, sample: Sample, cands: list[Candidate]) -> str:
    return f"""A user in {city.city} is moving through the city. Predict their next point of interest.

Recent check-ins (oldest to newest): {summarize_trajectory(city, sample.recent_pois)}
Current hour of day: {sample.last_hour}:00
{_user_profile(city, sample)}

Candidate POIs are listed in the order produced by a statistical retriever that combines
trajectory similarity, POI transition counts, spatial proximity and the user's own history.
Treat that order as a strong prior: [C1] is the retriever's top guess.
Each line: id | category | km from current location | estimated travel minutes | plausibility of this category at this hour | evidence.

{_candidate_block(city, sample, cands)}

Instructions:
1. Output the 10 most likely next POIs, most likely first.
2. Use ONLY ids from the candidate list. Never invent an id.
3. The 10 ids must be distinct.
4. Keep the retriever's order unless you have a concrete reason to change it; a concrete reason is, for example,
   a candidate that is unreachable in a plausible time, a category that makes no sense at this hour, or evidence
   from the user's own history that is clearly stronger than locality.
5. Do not promote a candidate merely because it is nearby if it has no supporting evidence.

Return JSON only:
{{"ranking": [10 integer ids, most likely first], "confidence": 0.0-1.0, "reason": "one short sentence"}}
`confidence` is how sure you are that the first item is the correct next POI."""


def critique_prompt(city: CityData, sample: Sample, cands: list[Candidate], first: list[int]) -> str:
    allowed = {c.poi for c in cands}
    rows = []
    for rank, poi in enumerate(first[:10], start=1):
        if poi not in allowed:
            rows.append(f"{rank}. id={poi} -- NOT IN CANDIDATE LIST (must be removed)")
            continue
        f = build_facts(city, poi, sample.last_poi, sample.last_hour)
        flags = violations(city, poi, sample.last_poi, sample.last_hour, config.EVAL)
        rows.append(
            f"{rank}. id={poi} | {f['cat']} | {f['dist_km']}km | ~{int(f['travel_min'])}min | "
            f"hour-fit={f['cat_hour_p']:.2f} | check={'; '.join(flags) if flags else 'ok'}"
        )
    return f"""You are auditing a POI recommendation before it is returned to the user.

User context:
- Recent check-ins: {summarize_trajectory(city, sample.recent_pois)}
- Current hour: {sample.last_hour}:00
- {_user_profile(city, sample)}

The first-pass answer was:
{chr(10).join(rows)}

Candidate POIs available (id | category | km | minutes | hour-fit):
{_candidate_block(city, sample, cands)}

Audit checklist:
- every id must exist in the candidate list (drop any that do not)
- no duplicates, exactly 10 ids
- replace a POI flagged implausible_jump or cat_hour_mismatch unless the evidence is strong
- the ordering should reflect probability

Return JSON only: {{"ranking": [10 corrected integer ids], "confidence": 0.0-1.0, "reason": "one short sentence"}}"""


def blend_orders(fusion: list[int], llm: list[int], alpha: float) -> list[int]:
    """Convex blend of a fusion prior and an LLM ordering (both rank based)."""
    n = max(len(fusion), len(llm), 1)

    def rank_scores(order: list[int]) -> dict[int, float]:
        return {poi: (n - i) / n for i, poi in enumerate(order)}

    fs = rank_scores(fusion)
    ls = rank_scores(llm)
    pool = list(dict.fromkeys(list(fusion) + list(llm)))
    scored = {poi: alpha * fs.get(poi, 0.0) + (1 - alpha) * ls.get(poi, 0.0) for poi in pool}
    return sorted(scored, key=lambda x: scored[x], reverse=True)


class AdaptiveAgent:
    def __init__(
        self,
        city: CityData,
        engine: RetrievalEngine | None = None,
        client: LLMClient | None = None,
        cfg: config.AgentConfig | None = None,
        enable_agent: bool = True,
        use_geo: bool = True,
        alpha: float | None = None,
    ):
        self.city = city
        self.cfg = cfg or config.AGENT
        self.engine = engine or RetrievalEngine(city)
        self.client = client or LLMClient(self.cfg)
        self.enable_agent = enable_agent
        self.use_geo = use_geo
        self.alpha = self.cfg.llm_blend_alpha if alpha is None else alpha
        if not use_geo:
            w = dict(self.engine.cfg.weights)
            w["geo"] = 0.0
            self.engine.cfg.weights = w

    # ------------------------------------------------------------------ utils
    def valid_ids(self, ranking, allowed: set[int]) -> tuple[list[int], int]:
        out, invalid = [], 0
        for x in ranking or []:
            try:
                poi = int(str(x).strip())
            except (TypeError, ValueError):
                invalid += 1
                continue
            if poi in allowed:
                if poi not in out:
                    out.append(poi)
            else:
                invalid += 1
        return out, invalid

    def uncertainty(self, cands: list[Candidate]) -> float:
        scores = np.asarray([c.score for c in cands], dtype=float)
        if len(scores) < 2 or scores.sum() <= 0:
            return 1.0
        p = scores / scores.sum()
        top = p[:10]
        return float(-(top * np.log(top + 1e-12)).sum() / np.log(len(top)))

    # -------------------------------------------------------------- main entry
    def run(self, sample: Sample, target: int | None = None) -> Prediction:
        cands = self.engine.candidates(sample)
        allowed = {c.poi for c in cands}
        pred = Prediction()
        pred.fusion_order = [c.poi for c in cands]
        pred.candidate_recall = bool(target is not None and target in allowed)
        if not cands:
            return pred
        pred.uncertainty = self.uncertainty(cands)

        data, u = self.client.chat_json(SYSTEM, ranking_prompt(self.city, sample, cands))
        pred.usage.add(u)
        data = data or {}
        pred.reason = data.get("reason", "")
        pred.confidence = _as_float(data.get("confidence"), 0.5)
        ids, invalid = self.valid_ids(data.get("ranking", []), allowed)
        pred.invalid_ids = invalid
        llm_order = list(ids)

        triggered = self.enable_agent and (
            pred.confidence < self.cfg.confidence_threshold
            or (self.cfg.always_verify_on_invalid and (invalid > 0 or len(ids) < 10))
        )
        if triggered:
            data2, u2 = self.client.chat_json(
                SYSTEM, critique_prompt(self.city, sample, cands, ids or pred.fusion_order[:10])
            )
            pred.usage.add(u2)
            data2 = data2 or {}
            ids2, invalid2 = self.valid_ids(data2.get("ranking", []), allowed)
            if ids2:
                llm_order = ids2
                pred.reason = data2.get("reason", pred.reason)
                pred.confidence = _as_float(data2.get("confidence"), pred.confidence)
            pred.verified = True

        pred.llm_order = llm_order
        final = blend_orders(pred.fusion_order, llm_order, self.alpha)
        if len(final) < 10:  # pad with fusion order, never with invented ids
            for c in cands:
                if c.poi not in final:
                    final.append(c.poi)
                if len(final) == 10:
                    break
        pred.ranking = final[:10]
        return pred


def _as_float(v, default: float) -> float:
    try:
        f = float(v)
        return f if 0.0 <= f <= 1.0 else default
    except (TypeError, ValueError):
        return default
