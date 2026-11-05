"""Does the LLM's output depend on the ORDER in which evidence is presented?

Content is held fixed (same four evidence lines, same facts); only the line order
changes.  Three measurements:

  1. output instability: mean pairwise Jaccard of the 10-item lists across orders
  2. accuracy per order, and accuracy of Borda-ensembling the orders
  3. line-position effect: average output rank of POIs that appear in the first
     evidence line vs the last one

If (1) is low and (2) shows ensemble > single, order sensitivity is real and SICA
exploits it.  If (1) is ~1.0, the model is order-invariant and the motivation fails.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grpoi.data import load_city  # noqa: E402
from grpoi.generative import SYSTEM, evidence_block  # noqa: E402
from grpoi.llm import LLMClient  # noqa: E402
from grpoi.profile import ProfileStore, render_profile  # noqa: E402
from grpoi.retrieval import RetrievalEngine  # noqa: E402
from grpoi.tools import summarize_trajectory  # noqa: E402


def build_prompt(city, sample, engine, profile_text, lines):
    ev = "\n".join(lines)
    return f"""A user in {city.city} is moving through the city. Predict their next point of interest.

Recent check-ins (oldest to newest): {summarize_trajectory(city, sample.recent_pois)}
Current hour of day: {sample.last_hour}:00
{profile_text}

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


def hr10(rankings, samples):
    return sum(1 for s, r in zip(samples, rankings) if s.target in (r or [])[:10]) / len(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="NYC")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=5, help="number of evidence orderings")
    ap.add_argument("--workers", type=int, default=50)
    args = ap.parse_args()

    city = load_city(args.city)
    samples = city.test[: args.n]
    engine = RetrievalEngine(city)
    client = LLMClient()
    store = ProfileStore(ROOT / "results" / "dump" / f"profiles_{args.city}.jsonl")
    print(f"{args.city}: {len(samples)} samples x {args.k} orderings", flush=True)

    # fixed set of orderings: identity, reverse, then seeded shuffles
    base_order = [0, 1, 2, 3]
    orders = [base_order, list(reversed(base_order))]
    rng = random.Random(0)
    while len(orders) < args.k:
        p = base_order[:]
        rng.shuffle(p)
        if p not in orders:
            orders.append(p)

    def one(i):
        s = samples[i]
        prof = render_profile(store, s.user)
        lines = evidence_block(city, s, engine).split("\n")
        lists = []
        for perm in orders:
            prompt = build_prompt(city, s, engine, prof, [lines[j] for j in perm])
            d, _ = client.chat_json(SYSTEM, prompt, temperature=0.0)
            lists.append(parse_ids(d))
        return i, lists

    res = [None] * len(samples)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, i): i for i, s in enumerate(samples)}
        done = 0
        for f in as_completed(futs):
            i, lists = f.result()
            res[i] = lists
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(samples)} ({done / (time.time() - t0):.1f}/s)", flush=True)

    # 1) instability across orders
    jac = []
    for lists in res:
        pairs = [jaccard(lists[a], lists[b]) for a in range(len(lists)) for b in range(a + 1, len(lists))]
        jac.append(np.mean(pairs))
    print(f"\n[1] 不同顺序输出的平均两两 Jaccard: {np.mean(jac):.3f}  (1.0 = 完全不敏感)")

    # 2) accuracy per ordering + ensemble
    per_order = []
    for k in range(len(orders)):
        rankings = [res[i][k] for i in range(len(samples))]
        per_order.append(hr10(rankings, samples))
    ens = [borda(res[i]) for i in range(len(samples))]
    print("\n[2] 各顺序的 HR@10:")
    for k, (perm, hr) in enumerate(zip(orders, per_order)):
        print(f"    order {k} {perm}: {hr:.3f}")
    print(f"    Borda 集成: {hr10(ens, samples):.3f}  (最好单序 {max(per_order):.3f}, 最差 {min(per_order):.3f})")

    # 3) line-position effect: where do POIs from line 0 vs line 3 end up?
    def avg_rank_of_line(line_idx):
        ranks = []
        for i, lists in enumerate(res):
            import re
            line = evidence_block(city, samples[i], engine).split("\n")[line_idx]
            ids = [int(x) for x in re.findall(r"(\d+) \(", line)]
            if not ids:
                continue
            for lst in lists:
                for pid in ids:
                    if pid in lst:
                        ranks.append(lst.index(pid))
        return float(np.mean(ranks)) if ranks else float("nan")

    r0, r3 = avg_rank_of_line(0), avg_rank_of_line(3)
    print(f"\n[3] 输出中的平均名次：第 1 行证据的 POI = {r0:.2f}，第 4 行证据的 POI = {r3:.2f}")
    print("    （若前者明显更小，说明模型偏好排在前面的证据——即位置偏置）")

    dump = ROOT / "results" / "dump" / f"order_sensitivity_{args.city}_{args.n}.jsonl"
    dump.parent.mkdir(parents=True, exist_ok=True)
    with open(dump, "w") as fh:
        for i, lists in enumerate(res):
            fh.write(json.dumps({"i": i, "target": int(samples[i].target), "lists": lists}) + "\n")

    out = ROOT / "results" / f"order_sensitivity_{args.city}_{args.n}.json"
    out.write_text(json.dumps({
        "n": len(samples), "k": args.k,
        "mean_jaccard": float(np.mean(jac)),
        "per_order_hr10": [float(x) for x in per_order],
        "ensemble_hr10": float(hr10(ens, samples)),
        "rank_line0": r0, "rank_line3": r3,
        "calls": client.usage.calls,
        "cost": client.usage.cost_usd,
    }, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
