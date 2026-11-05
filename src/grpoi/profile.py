"""LLM-generated user profiles (borrowed from AgentMove's personal-memory idea).

A profile is generated once per user (not per sample) from their training history
statistics, then injected into the generation prompt.  Cached on disk.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .data import CityData

SYSTEM = (
    "You are a user modeling analyst for a location-based service. "
    "You summarise a user's check-in statistics into a short mobility profile. "
    "Answer with JSON only."
)


def _summarise(city: CityData, user: int) -> str | None:
    hist = city.user_hist.get(user, [])
    if not hist:
        return None
    n = len(hist)
    poi_cnt = Counter(p for p, _, _ in hist)
    cat_cnt = Counter(city.poi_cat(p) for p, _, _ in hist)
    hour_cnt = Counter()
    days = set()
    for p, c, h in hist:
        hour_cnt[h // 6] += 1  # 4 buckets: 0-5, 6-11, 12-17, 18-23

    top_pois = "; ".join(
        f"{p} ({city.cat_name(city.poi_cat(p))}, {c}x)" for p, c in poi_cnt.most_common(10)
    )
    cats = ", ".join(f"{city.cat_name(k)} {v / n:.0%}" for k, v in cat_cnt.most_common(4))
    buckets = ["00-05", "06-11", "12-17", "18-23"]
    hours = ", ".join(f"{buckets[k]} {v / n:.0%}" for k, v in sorted(hour_cnt.items()))

    return f"""Total check-ins: {n}
Most frequent places: {top_pois}
Category mix: {cats}
Time-of-day mix: {hours}"""


def profile_prompt(city: CityData, user: int) -> str | None:
    stats = _summarise(city, user)
    if stats is None:
        return None
    return f"""A user in {city.city} has the following check-in statistics from a historical period.

{stats}

Write:
1. "profile": a concise mobility profile in at most 60 words (habits, preferred categories, typical areas,
   and whether the user tends to revisit the same places or explore new ones).
2. "routine": one short sentence about the user's typical rhythm during the day.

Base your answer only on the statistics above. Do not invent place names.

Return JSON only: {{"profile": "...", "routine": "..."}}"""


class ProfileStore:
    """user_id -> {"profile": str, "routine": str}, cached as jsonl."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[int, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.data[int(r["user"])] = {"profile": r.get("profile", ""), "routine": r.get("routine", "")}

    def get(self, user: int) -> dict | None:
        return self.data.get(user)

    def missing(self, users) -> list[int]:
        return [u for u in users if u not in self.data]

    def put(self, user: int, profile: dict) -> None:
        self.data[user] = profile

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as fh:
            for u, v in sorted(self.data.items()):
                fh.write(json.dumps({"user": u, "profile": v.get("profile", ""),
                                     "routine": v.get("routine", "")}) + "\n")


def render_profile(store: ProfileStore, user: int) -> str:
    p = store.get(user)
    if not p:
        return "This user has no history in the training period (cold-start user)."
    txt = p.get("profile", "").strip()
    routine = p.get("routine", "").strip()
    out = f"User profile: {txt}"
    if routine:
        out += f" Routine: {routine}"
    return out
