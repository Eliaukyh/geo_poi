"""Pilot 2: richer intent context + soft (multi-category) hint.

Compared on the same samples:
  G0  direct generation (from dump)
  G1  oracle category hint (upper bound)
  P1h plan-then-generate, hard hint (top-1 intent), rich context
  P1s plan-then-generate, soft hint (top-3 intents), rich context
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from grpoi.data import load_city  # noqa: E402
from grpoi.generative import SYSTEM, evidence_block  # noqa: E402
from grpoi.llm import LLMClient  # noqa: E402
from grpoi.retrieval import RetrievalEngine  # noqa: E402
from grpoi.tools import summarize_trajectory  # noqa: E402

INTENT_SYSTEM = (
    "You are an urban mobility analyst. You infer what a user is about to do next. "
    "Answer with JSON only."
)


def build_cat_stats(city):
    trans = defaultdict(Counter)
    for s in city.train:
        cats = [city.poi_cat(p) for p in s.pois]
        for i in range(1, len(cats)):
            trans[cats[i - 1]][cats[i]] += 1
    return trans


def rich_context(city, sample, engine, cat_stats):
    hist = city.user_hist.get(sample.user, [])
    if hist:
        c = Counter(city.poi_cat(p) for p, _, _ in hist)
        top = ", ".join(f"{city.cat_name(k)} ({v / len(hist):.0%})" for k, v in c.most_common(3))
        profile = f"This user has {len(hist)} check-ins; category mix: {top}."
    else:
        profile = "This user has no history (cold-start)."

    last_cat = city.poi_cat(sample.last_poi)
    tr = cat_stats.get(last_cat, Counter())
    tot = sum(tr.values()) or 1
    trans_txt = ", ".join(f"{city.cat_name(k)} {v / tot:.0%}" for k, v in tr.most_common(3))

    return f"""Recent check-ins (oldest to newest): {summarize_trajectory(city, sample.recent_pois, max_len=20)}
Current hour of day: {sample.last_hour}:00
{profile}
Historical category transitions after "{city.cat_name(last_cat)}": {trans_txt}

Reference evidence (use as a hint; you may also rely on your own knowledge of this city):
{evidence_block(city, sample, engine)}"""


def intent_prompt(city, sample, engine, cat_stats):
    cats = ", ".join(city.cat_names)
    return f"""A user in {city.city} is moving through the city.

{rich_context(city, sample, engine, cat_stats)}

Task: which CATEGORY of place will the user visit next?
Choose from exactly: [{cats}]

Return JSON only: {{"intents": [{{"category": "<one of the above>", "p": 0.0-1.0}}, ...]}}
Give the 3 most likely categories, ordered by probability."""


def gen_prompt(city, sample, engine, cat_stats, hints):
    if len(hints) == 1:
        hint_line = (f"\nThe user's most likely next activity is: {hints[0]}. "
                     f"Recommend places consistent with it.\n")
    else:
        joined = ", ".join(hints)
        hint_line = (f"\nThe user's next activity is likely among: {joined}. "
                     f"Recommend places consistent with these possibilities, favouring the most likely one.\n")
    return f"""A user in {city.city} is moving through the city. Predict their next point of interest.

{rich_context(city, sample, engine, cat_stats)}
{hint_line}
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


def evaluate(samples, rankings, ks=(1, 5, 10)):
    n = len(samples)
    hr = {k: 0 for k in ks}
    nd = 0.0
    for s, r in zip(samples, rankings):
        r = r[:10]
        if s.target in r:
            pos = r.index(s.target) + 1
            for k in ks:
                if pos <= k:
                    hr[k] += 1
            nd += 1.0 / math.log2(pos + 1)
    return {**{f"HR@{k}": hr[k] / n for k in ks}, "NDCG@10": nd / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="NYC")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    city = load_city(args.city)
    samples = city.test[: args.n]
    engine = RetrievalEngine(city)
    client = LLMClient()
    cat_stats = build_cat_stats(city)
    print(f"{args.city}: {len(samples)} samples", flush=True)

    t0 = time.time()

    def run_one(i, s):
        out = {}
        d, _ = client.chat_json(INTENT_SYSTEM, intent_prompt(city, s, engine, cat_stats))
        ints = []
        for it in (d or {}).get("intents", []):
            cat = str(it.get("category", "")).strip()
            if cat in city.cat_names and cat not in ints:
                ints.append(cat)
        ints = ints[:3]
        out["intents"] = ints

        true_cat = city.cat_name(city.poi_cat(s.target))
        d, _ = client.chat_json(SYSTEM, gen_prompt(city, s, engine, cat_stats, [true_cat]))
        out["g1"] = parse_ids(d)

        if ints:
            d, _ = client.chat_json(SYSTEM, gen_prompt(city, s, engine, cat_stats, [ints[0]]))
            out["p1h"] = parse_ids(d)
            d, _ = client.chat_json(SYSTEM, gen_prompt(city, s, engine, cat_stats, ints))
            out["p1s"] = parse_ids(d)
        else:
            out["p1h"] = []
            out["p1s"] = []
        return out

    results = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, i, s): i for i, s in enumerate(samples)}
        done = 0
        for f in as_completed(futs):
            i = futs[f]
            try:
                results[i] = f.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  sample {i} failed: {exc}", flush=True)
                results[i] = {"g1": [], "p1h": [], "p1s": [], "intents": []}
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(samples)} ({done / (time.time() - t0):.1f}/s)", flush=True)

    g0 = None
    if args.dump:
        rows = [json.loads(l) for l in Path(args.dump).read_text().splitlines() if l.strip()][: args.n]
        g0 = [r.get("final") or [] for r in rows]

    table = []
    if g0:
        table.append(("G0 direct", evaluate(samples, g0)))
    table.append(("G1 oracle-cat", evaluate(samples, [r["g1"] for r in results])))
    table.append(("P1h hard hint", evaluate(samples, [r["p1h"] for r in results])))
    table.append(("P1s soft hint", evaluate(samples, [r["p1s"] for r in results])))

    acc = sum(1 for s, r in zip(samples, results)
              if r["intents"] and r["intents"][0] == city.cat_name(city.poi_cat(s.target))) / len(samples)
    acc3 = sum(1 for s, r in zip(samples, results)
               if city.cat_name(city.poi_cat(s.target)) in r["intents"]) / len(samples)

    print(f"\n[rich context] LLM 意图 top-1 = {acc:.3f} | top-3 = {acc3:.3f}")
    print("\n| 配置 | HR@1 | HR@5 | HR@10 | NDCG@10 |")
    print("|---|---|---|---|---|")
    for name, m in table:
        print(f"| {name} | {m['HR@1']:.3f} | {m['HR@5']:.3f} | {m['HR@10']:.3f} | {m['NDCG@10']:.3f} |")
    print(f"\n调用 {client.usage.calls} | 成本 ${client.usage.cost_usd:.4f}")

    out = ROOT / "results" / f"pilot_hier2_{args.city}_{args.n}.json"
    out.write_text(json.dumps({
        "n": len(samples), "intent_top1": acc, "intent_top3": acc3,
        "results": {name: m for name, m in table},
        "calls": client.usage.calls, "cost": client.usage.cost_usd,
    }, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
