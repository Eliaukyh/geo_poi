"""Prompt-diversity ensemble vs sampling diversity at equal compute.

Same budget (3 generations per sample), different source of diversity:
  S1  same prompt, temperature 0.7, 3 samples
  S2  3 different prompts, temperature 0, 1 sample each
Aggregation: Borda over the three lists.

If S2 > S1, the gain comes from prompt diversity rather than from spending more calls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from grpoi.data import load_city  # noqa: E402
from grpoi.generative import SYSTEM, evidence_block  # noqa: E402
from grpoi.llm import LLMClient  # noqa: E402
from grpoi.retrieval import RetrievalEngine  # noqa: E402
from grpoi.tools import summarize_trajectory  # noqa: E402

PERSONA = (
    "You are a long-time resident of {city} who knows its neighbourhoods, "
    "opening patterns and daily rhythms well. "
    "Think about the time of day and the user's routine before deciding."
)


def context_lines(city, sample):
    hist = city.user_hist.get(sample.user, [])
    if hist:
        c = Counter(city.poi_cat(p) for p, _, _ in hist)
        top = ", ".join(f"{city.cat_name(k)} ({v / len(hist):.0%})" for k, v in c.most_common(2))
        profile = f"This user has {len(hist)} check-ins; dominant categories: {top}."
    else:
        profile = "This user has no history (cold-start)."
    return (f"Recent check-ins (oldest to newest): {summarize_trajectory(city, sample.recent_pois)}\n"
            f"Current hour of day: {sample.last_hour}:00\n{profile}")


def build_prompt(variant: str, city, sample, engine) -> str:
    lines = evidence_block(city, sample, engine).split("\n")
    header = ""
    if variant == "B" and len(lines) >= 4:          # evidence order reversed
        lines = [lines[1], lines[2], lines[3], lines[0]]
        header = "Note: the evidence below is ordered by personal relevance first.\n"
    elif variant == "C":                             # persona + explicit reasoning
        header = PERSONA.format(city=city.city) + "\n"

    ev = "\n".join(lines)
    return f"""A user in {city.city} is moving through the city. Predict their next point of interest.

{header}{context_lines(city, sample)}

Reference evidence retrieved from historical check-ins:
{ev}

Instructions:
1. Recommend exactly 10 distinct POI ids, most likely first.
2. Use POI ids that exist in this city. You are not limited to the ids listed above.
3. Do not include line breaks inside the JSON.

Return JSON only: {{"recommendation": [10 integer ids, most likely first], "reason": "one short sentence"}}"""


def parse_ids(data):
    raw = (data or {}).get("recommendation") or (data or {}).get("ranking") or []
    out = []
    for x in raw:
        try:
            v = int(str(x).strip())
        except (TypeError, ValueError):
            continue
        if v not in out:
            out.append(v)
    return out


def borda(lists, k=10):
    score = Counter()
    for lst in lists:
        for rank, p in enumerate(lst[:k]):
            score[p] += (k - rank)
    return [p for p, _ in score.most_common(k)]


def jaccard(a, b, k=10):
    sa, sb = set(a[:k]), set(b[:k])
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def hit_vec(samples, rankings, k=10):
    return np.array([1.0 if s.target in (r or [])[:k] else 0.0 for s, r in zip(samples, rankings)])


def ndcg_vec(samples, rankings, k=10):
    out = []
    for s, r in zip(samples, rankings):
        r = (r or [])[:k]
        out.append(1.0 / math.log2(r.index(s.target) + 2) if s.target in r else 0.0)
    return np.array(out)


def paired(a, b, label):
    d = a - b
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(d), size=(2000, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    sig = "显著" if (lo > 0 or hi < 0) else "不显著"
    print(f"  {label}: {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {sig}")


def metrics(samples, rankings):
    n = len(samples)
    h = {k: 0 for k in (1, 5, 10)}
    nd = 0.0
    for s, r in zip(samples, rankings):
        r = (r or [])[:10]
        if s.target in r:
            p = r.index(s.target) + 1
            for k in h:
                if p <= k:
                    h[k] += 1
            nd += 1.0 / math.log2(p + 1)
    return {f"HR@{k}": h[k] / n for k in h} | {"NDCG@10": nd / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="NYC")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    city = load_city(args.city)
    samples = city.test[: args.n]
    engine = RetrievalEngine(city)
    client = LLMClient()
    print(f"{args.city}: {len(samples)} samples | 6 calls/sample", flush=True)

    def one(i):
        s = samples[i]
        # S1: same prompt, temperature 0.7, 3 samples
        s1 = []
        for _ in range(3):
            d, _ = client.chat_json(SYSTEM, build_prompt("A", city, s, engine), temperature=0.7)
            s1.append(parse_ids(d))
        # S2: three prompts, temperature 0
        s2 = []
        for v in ("A", "B", "C"):
            d, _ = client.chat_json(SYSTEM, build_prompt(v, city, s, engine), temperature=0.0)
            s2.append(parse_ids(d))
        return i, s1, s2

    res = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, i): i for i, s in enumerate(samples)}
        done = 0
        for f in as_completed(futs):
            i = futs[f]
            try:
                _, s1, s2 = f.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  sample {i} failed: {exc}", flush=True)
                s1, s2 = [], []
            res[i] = {"s1": s1, "s2": s2}
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(samples)} ({done / (time.time() - t0):.1f}/s)", flush=True)

    single = [parse_ids({"recommendation": (res[i]["s1"][0] if res[i]["s1"] else [])}) for i in range(len(samples))]
    s1_agg = [borda(res[i]["s1"]) for i in range(len(samples))]
    s2_agg = [borda(res[i]["s2"]) for i in range(len(samples))]

    g0 = None
    if args.dump and Path(args.dump).exists():
        rows = [json.loads(l) for l in Path(args.dump).read_text().splitlines() if l.strip()][: args.n]
        g0 = [r.get("final") or [] for r in rows]

    print("\n| 配置 | HR@1 | HR@5 | HR@10 | NDCG@10 |")
    print("|---|---|---|---|---|")
    for name, rk in [("G0 direct (1 call)", g0), ("S1 same-prompt x3", s1_agg), ("S2 prompt-ensemble x3", s2_agg)]:
        if rk is None:
            continue
        m = metrics(samples, rk)
        print(f"| {name} | {m['HR@1']:.3f} | {m['HR@5']:.3f} | {m['HR@10']:.3f} | {m['NDCG@10']:.3f} |")

    div1 = np.mean([jaccard(a, b) for i in range(len(samples)) for a, b in
                    [(res[i]["s1"][0], res[i]["s1"][1]), (res[i]["s1"][1], res[i]["s1"][2]), (res[i]["s1"][0], res[i]["s1"][2])]
                    if a and b])
    div2 = np.mean([jaccard(a, b) for i in range(len(samples)) for a, b in
                    [(res[i]["s2"][0], res[i]["s2"][1]), (res[i]["s2"][1], res[i]["s2"][2]), (res[i]["s2"][0], res[i]["s2"][2])]
                    if a and b])
    print(f"\n列表两两 Jaccard：S1 同提示词 = {div1:.3f} | S2 异提示词 = {div2:.3f}")
    print(f"（越低越多样）调用 {client.usage.calls} | 成本 ${client.usage.cost_usd:.4f}")

    # paired significance
    print("\n配对差值（bootstrap 95% CI）:")
    if g0:
        paired(hit_vec(samples, s2_agg), hit_vec(samples, g0), "S2 - G0  HR@10")
        paired(hit_vec(samples, s1_agg), hit_vec(samples, g0), "S1 - G0  HR@10")
    paired(hit_vec(samples, s2_agg), hit_vec(samples, s1_agg), "S2 - S1  HR@10")
    paired(ndcg_vec(samples, s2_agg), ndcg_vec(samples, s1_agg), "S2 - S1  NDCG@10")
    if g0:
        paired(ndcg_vec(samples, s2_agg), ndcg_vec(samples, g0), "S2 - G0  NDCG@10")

    # per-sample dump for later analysis
    dump = ROOT / "results" / "dump" / f"prompt_ensemble_{args.city}_{args.n}.jsonl"
    dump.parent.mkdir(parents=True, exist_ok=True)
    with open(dump, "w") as fh:
        for i in range(len(samples)):
            fh.write(json.dumps({"i": i, "target": int(samples[i].target),
                                 "s1": res[i]["s1"], "s2": res[i]["s2"]}) + "\n")

    out = ROOT / "results" / f"prompt_ensemble_{args.city}_{args.n}.json"
    out.write_text(json.dumps({
        "n": len(samples),
        "single": metrics(samples, single),
        "s1": metrics(samples, s1_agg),
        "s2": metrics(samples, s2_agg),
        "direct": metrics(samples, g0) if g0 else None,
        "jaccard_s1": float(div1),
        "jaccard_s2": float(div2),
        "calls": client.usage.calls,
        "cost_usd": client.usage.cost_usd,
    }, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
