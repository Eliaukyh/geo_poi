"""Evaluation metrics.

Headline metrics follow RALLM-POI's `test_model` exactly so numbers are directly
comparable:

    HR@k     = #{label in preds[:k]} / N
    NDCG@k   = sum(1 / log2(pos + 1)) / N      (pos is 1-based)
    MAP@k    = sum(1 / pos) / N

Naming note: with exactly one ground-truth POI per sample, HitRate@k, Recall@k and
Accuracy@k are the same quantity -- each user contributes a binary indicator
1[rank <= k].  RALLM-POI's *paper* reports "HR@k" while its *code* calls the same
function `calc_recall`; we emit both key names with identical values and display HR.

Supplementary metrics (not in RALLM-POI) measure spatial quality, constraint
violations and inference cost.
"""

from __future__ import annotations

import numpy as np

from . import config
from .data import CityData, haversine_scalar
from .tools import violations

KS = (1, 5, 10)


def ndcg_at(hit_pos: int | None, k: int) -> float:
    if hit_pos is None or hit_pos > k:
        return 0.0
    return 1.0 / np.log2(hit_pos + 1)


def map_at(hit_pos: int | None, k: int) -> float:
    if hit_pos is None or hit_pos > k:
        return 0.0
    return 1.0 / hit_pos


def evaluate(
    samples: list,
    preds: list,
    city: CityData,
    cfg: config.EvalConfig | None = None,
) -> dict:
    cfg = cfg or config.EVAL
    n = len(samples)
    if n == 0:
        return {}

    recall = {k: 0.0 for k in KS}
    ndcg = {k: 0.0 for k in KS}
    mapk = {k: 0.0 for k in KS}

    geo_err: list[float] = []
    jump = cat_hour_bad = invalid = dup = cand_recall = 0
    calls = tokens_in = tokens_out = 0
    latency = cost = verified = 0.0

    for s, p in zip(samples, preds):
        target = s.target
        ranking = [int(x) for x in p.ranking]
        hit_pos = ranking.index(target) + 1 if target in ranking else None

        for k in KS:
            if hit_pos is not None and hit_pos <= k:
                recall[k] += 1
                ndcg[k] += ndcg_at(hit_pos, k)
                mapk[k] += map_at(hit_pos, k)

        if ranking:
            top1 = ranking[0]
            if 0 <= top1 < city.n_pois() and 0 <= target < city.n_pois():
                geo_err.append(haversine_scalar(city.poi_coord(top1), city.poi_coord(target)))
            flags = violations(city, top1, s.last_poi, s.last_hour, cfg)
            jump += any(f.startswith("implausible_jump") for f in flags)
            cat_hour_bad += any(f.startswith("cat_hour_mismatch") for f in flags)

        invalid += getattr(p, "invalid_ids", 0)
        dup += len(ranking) - len(set(ranking))
        cand_recall += int(getattr(p, "candidate_recall", False))
        verified += int(getattr(p, "verified", False))

        u = p.usage
        calls += u.calls
        tokens_in += u.prompt_tokens
        tokens_out += u.completion_tokens
        latency += u.latency_s
        cost += u.cost_usd

    out: dict = {"n": n}
    for k in KS:
        out[f"HR@{k}"] = recall[k] / n
        out[f"Recall@{k}"] = recall[k] / n  # identical under a single ground truth
        out[f"Accuracy@{k}"] = recall[k] / n
    for k in KS:
        out[f"NDCG@{k}"] = ndcg[k] / n
    for k in KS:
        out[f"MAP@{k}"] = mapk[k] / n

    out.update(
        {
            "geo_err_km": float(np.mean(geo_err)) if geo_err else None,
            "jump_rate": jump / n,
            "cat_hour_violation_rate": cat_hour_bad / n,
            "invalid_ids_per_sample": invalid / n,
            "dup_per_sample": dup / n,
            "candidate_recall": cand_recall / n,
            "calls_per_sample": calls / n,
            "prompt_tokens": tokens_in,
            "completion_tokens": tokens_out,
            "latency_s_per_sample": latency / n,
            "cost_usd_total": cost,
            "cost_usd_per_sample": cost / n,
            "verified_rate": verified / n,
        }
    )
    return out


COLUMNS = [
    ("n", "n"),
    ("HR@1", "HR@1"),
    ("HR@5", "HR@5"),
    ("HR@10", "HR@10"),
    ("NDCG@5", "N@5"),
    ("NDCG@10", "N@10"),
    ("MAP@5", "M@5"),
    ("MAP@10", "M@10"),
    ("geo_err_km", "GeoErr(km)"),
    ("jump_rate", "Jump"),
    ("candidate_recall", "CandRec"),
    ("calls_per_sample", "Calls"),
    ("latency_s_per_sample", "Sec/s"),
    ("cost_usd_per_sample", "USD/s"),
]


def format_table(rows: dict[str, dict]) -> str:
    head = "| Method | " + " | ".join(label for _, label in COLUMNS) + " |"
    sep = "|" + "---|" * (len(COLUMNS) + 1)
    lines = [head, sep]
    for name, m in rows.items():
        cells = []
        for key, _ in COLUMNS:
            v = m.get(key)
            if v is None:
                cells.append("-")
            elif isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
