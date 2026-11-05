"""Agent loop for output verification: Judge -> Fixer.

Design constraints (learned from earlier ablations):
  * the loop may only touch items that violate OBJECTIVE rules (format, distance,
    travel time, category-hour); it must not re-rank items that are fine, because
    an extra LLM pass was measured to hurt ranking quality;
  * a fix is accepted only if it strictly reduces the number of violations,
    otherwise the previous ranking is restored (so the loop cannot make things
    worse than the deterministic baseline);
  * the Fixer may only choose from the evidence pool, never invent ids.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .data import CityData, Sample
from .llm import LLMClient
from .tools import build_facts, summarize_trajectory, travel_time_min, violations

JUDGE_SYSTEM = (
    "You are a strict verifier of POI recommendations. You check a recommendation "
    "list against a fixed set of objective rules and report violations. "
    "You never propose replacements. Answer with JSON only."
)

FIXER_SYSTEM = (
    "You repair a POI recommendation list. You only replace the items that were "
    "flagged as violations, keep every other item in its original position, and "
    "choose replacements only from the given pool. Answer with JSON only."
)

MAX_DIST_KM = 50.0
MAX_TRAVEL_MIN = 120.0


@dataclass
class Issue:
    kind: str          # duplicate | out_of_range | short_list | jump | travel | cat_hour
    poi: int | None
    detail: str = ""


@dataclass
class LoopResult:
    ranking: list[int] = field(default_factory=list)
    calls: int = 0
    iterations: int = 0
    accepted_fixes: int = 0
    issues_before: int = 0
    issues_after: int = 0


def objective_issues(city: CityData, sample: Sample, ranking: list[int]) -> list[Issue]:
    """Deterministic violation checks (used as ground truth for the Judge)."""
    out: list[Issue] = []
    n = city.n_pois()
    seen = set()
    for p in ranking:
        if not (0 <= p < n):
            out.append(Issue("out_of_range", p, "id not in POI table"))
        elif p in seen:
            out.append(Issue("duplicate", p, "repeated id"))
        seen.add(p)
    if len(ranking) < 10:
        out.append(Issue("short_list", None, f"only {len(ranking)} items"))

    for p in ranking:
        if not (0 <= p < n):
            continue
        f = build_facts(city, p, sample.last_poi, sample.last_hour)
        if f["dist_km"] > MAX_DIST_KM:
            out.append(Issue("jump", p, f"{f['dist_km']}km from current location"))
        if f["travel_min"] > MAX_TRAVEL_MIN:
            out.append(Issue("travel", p, f"~{int(f['travel_min'])}min"))
        if f["cat_hour_p"] < 0.02:
            out.append(Issue("cat_hour", p, f"{f['cat']} at {sample.last_hour}h (p={f['cat_hour_p']})"))
    return out


def _facts_block(city: CityData, sample: Sample, ranking: list[int]) -> str:
    rows = []
    for i, p in enumerate(ranking, start=1):
        if not (0 <= p < city.n_pois()):
            rows.append(f"{i}. id={p} | NOT IN POI TABLE")
            continue
        f = build_facts(city, p, sample.last_poi, sample.last_hour)
        rows.append(
            f"{i}. id={p} | {f['cat']} | {f['dist_km']}km | ~{int(f['travel_min'])}min | hour-fit={f['cat_hour_p']:.2f}"
        )
    return "\n".join(rows)


def judge_prompt(city: CityData, sample: Sample, ranking: list[int], task_issues: list[Issue]) -> str:
    flags = "\n".join(f"- {i.kind}: {i.detail}" for i in task_issues) or "- (none detected by the rule engine)"
    return f"""A user in {city.city} is at POI {sample.last_poi} at {sample.last_hour}:00.
Recent check-ins: {summarize_trajectory(city, sample.recent_pois)}

A 10-item recommendation list was produced:
{_facts_block(city, sample, ranking)}

The rule engine reports:
{flags}

Check the list against these objective rules ONLY:
R1. exactly 10 items, all distinct, all ids exist
R2. straight-line distance from the current location <= {MAX_DIST_KM:.0f} km
R3. estimated travel time <= {MAX_TRAVEL_MIN:.0f} min
R4. the category must be plausible at this hour (hour-fit >= 0.02)

Do NOT judge whether a POI is "better" than another; only report rule violations.

Return JSON only:
{{"verdict": "pass" | "fix",
  "issues": [{{"kind": "duplicate|out_of_range|short_list|jump|travel|cat_hour",
              "poi": <id or null>, "detail": "<short>"}}]}}"""


def fixer_prompt(city: CityData, sample: Sample, ranking: list[int],
                 issues: list[Issue], pool: list[int]) -> str:
    issue_txt = "\n".join(f"- {i.kind}: poi={i.poi} ({i.detail})" for i in issues)
    pool_txt = _facts_block(city, sample, pool)
    return f"""A recommendation list has rule violations and must be repaired.

Current list (position: id):
{_facts_block(city, sample, ranking)}

Violations:
{issue_txt}

Replacement pool (choose only from here):
{pool_txt}

Repair rules:
0. NEVER change the item at position 1; it is protected. If position 1 is flagged, leave it as is.
1. Replace ONLY the items flagged above; keep all other items in their original positions.
2. A replacement should satisfy: distance <= {MAX_DIST_KM:.0f} km, travel time <= {MAX_TRAVEL_MIN:.0f} min,
   hour-fit >= 0.02, and must not duplicate an item already in the list.
3. Return exactly 10 distinct ids, ordered as before (replacements take the flagged slot).

Return JSON only: {{"ranking": [10 integer ids]}}"""


class AgentLoop:
    """Judge -> Fixer loop with a monotone acceptance rule."""

    def __init__(self, city: CityData, client: LLMClient, max_iters: int = 2, pool_size: int = 15):
        self.city = city
        self.client = client
        self.max_iters = max_iters
        self.pool_size = pool_size

    def run(self, sample: Sample, ranking: list[int], fill_pool: list[int]) -> LoopResult:
        state = list(dict.fromkeys(ranking))[:10]
        anchor = state[0] if state else None
        res = LoopResult(ranking=state)
        res.issues_before = len(objective_issues(self.city, sample, state))

        for _ in range(self.max_iters):
            if not state:
                break
            before = objective_issues(self.city, sample, state)
            if not before:
                break
            res.iterations += 1

            data, usage = self.client.chat_json(
                JUDGE_SYSTEM, judge_prompt(self.city, sample, state, before)
            )
            res.calls += 1
            data = data or {}
            llm_issues = data.get("issues", []) or []
            # merge: rule engine is ground truth, LLM may add detail but not remove
            issues = before + [
                Issue(kind=str(it.get("kind", "unknown")), poi=it.get("poi"), detail=str(it.get("detail", ""))[:80])
                for it in llm_issues
                if str(it.get("kind", "")) in {"duplicate", "out_of_range", "short_list", "jump", "travel", "cat_hour"}
            ]
            if not issues:
                break

            pool = [p for p in fill_pool if p not in state][: self.pool_size]
            if not pool:
                break
            data2, usage2 = self.client.chat_json(
                FIXER_SYSTEM, fixer_prompt(self.city, sample, state, issues, pool)
            )
            res.calls += 1
            new = (data2 or {}).get("ranking") or (data2 or {}).get("recommendation") or []
            new_ids = []
            for x in new:
                try:
                    v = int(str(x).strip())
                except (TypeError, ValueError):
                    continue
                if 0 <= v < self.city.n_pois() and v not in new_ids:
                    new_ids.append(v)
            if not new_ids:
                break

            # hard protection: position 1 is never replaced
            if anchor is not None:
                new_ids = [anchor] + [x for x in new_ids if x != anchor]
                new_ids = new_ids[:10]
            after = len(objective_issues(self.city, sample, new_ids))
            if after < len(before):
                state = new_ids
                res.accepted_fixes += 1
            else:
                break  # reject the fix, keep the previous ranking

        res.ranking = state
        res.issues_after = len(objective_issues(self.city, sample, state))
        return res
