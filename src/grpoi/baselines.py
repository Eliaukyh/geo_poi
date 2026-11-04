"""Baselines that isolate the contribution of each component.

* ``FreeFormAgent``      -- RALLM-POI style prompting: the model must emit POI ids
                            from memory, with no candidate constraint (optional RAG context)
* ``CandidateOnlyAgent`` -- our retrieval, but the LLM merely re-orders; no tools, no agent
"""

from __future__ import annotations

from . import config
from .agent import SYSTEM, Prediction, _user_profile
from .data import CityData, Sample
from .llm import LLMClient
from .retrieval import RetrievalEngine
from .tools import summarize_trajectory


class FreeFormAgent:
    def __init__(self, city: CityData, client: LLMClient | None = None, use_rag: bool = False):
        self.city = city
        self.client = client or LLMClient()
        self.use_rag = use_rag
        self.engine = RetrievalEngine(city)

    def run(self, sample: Sample, target: int | None = None) -> Prediction:
        ctx = ""
        if self.use_rag:
            hits = self.engine.semantic.retrieve(sample, 10)
            ctx = (
                "\nContext: other users with similar recent check-ins next visited these POIs: "
                f"{[p for p, _ in hits]}. Use them only as a reference.\n"
            )
        prompt = f"""A user in {self.city.city} is moving through the city. Predict their next point of interest.

Recent check-ins (oldest to newest): {summarize_trajectory(self.city, sample.recent_pois)}
Current hour of day: {sample.last_hour}:00
{_user_profile(self.city, sample)}
{ctx}
Instructions:
1. Recommend exactly 10 distinct POI ids, most likely first.
2. Use real POI ids that exist in this city.
3. Do not include line breaks inside the JSON.

Return JSON only: {{"ranking": [10 integer ids, most likely first], "reason": "one short sentence"}}"""
        data, u = self.client.chat_json(SYSTEM, prompt)
        pred = Prediction()
        pred.usage.add(u)
        raw = (data or {}).get("ranking", [])
        ids, invalid = [], 0
        for x in raw:
            try:
                poi = int(str(x).strip())
            except (TypeError, ValueError):
                invalid += 1
                continue
            if 0 <= poi < self.city.n_pois():
                if poi not in ids:
                    ids.append(poi)
            else:
                invalid += 1
        pred.invalid_ids = invalid
        pred.ranking = ids[:10]
        pred.reason = (data or {}).get("reason", "")
        if target is not None:
            pred.candidate_recall = target in ids[:10]
        return pred


class CandidateOnlyAgent:
    """Our multi-route retrieval + LLM ranking, with tools and agent disabled."""

    def __init__(self, city: CityData, client: LLMClient | None = None):
        from .agent import AdaptiveAgent

        self.inner = AdaptiveAgent(city, client=client, enable_agent=False)

    def run(self, sample: Sample, target: int | None = None) -> Prediction:
        return self.inner.run(sample, target)
