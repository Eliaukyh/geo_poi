"""Final configuration: multi-view evidence + LLM profile + prompt-level TTA.

Per sample:
  context = recent check-ins + hour + LLM user profile + four-view evidence
  for variant in {A, B, C}: LLM generates 10 POIs (temperature 0)
  aggregate by Borda -> repair (drop invalid, pad from evidence) -> top-10

Cost: 3 calls/sample (profiles are cached per user).
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
from grpoi.profile import ProfileStore, SYSTEM as PROFILE_SYSTEM, profile_prompt, render_profile  # noqa: E402
from grpoi.retrieval import RetrievalEngine  # noqa: E402
from grpoi.tools import summarize_trajectory  # noqa: E402

PERSONA = (
    "You are a long-time resident of {city} who knows its neighbourhoods, "
    "opening patterns and daily rhythms well. "
    "Think about the time of day and the user's routine before deciding."
)


def build_prompt(variant: str, city, sample, engine, profile_text: str, per_view: int = 8,
                 drop: tuple[str, ...] = ()) -> str:
    lines = evidence_block(city, sample, engine, per_view, drop).split("\n")
    header = ""
    if variant == "B" and len(lines) >= 4:
        lines = [lines[1], lines[2], lines[3], lines[0]]
        header = "Note: the evidence below is ordered by personal relevance first.\n"
    elif variant == "C":
        header = PERSONA.format(city=city.city) + "\n"
    ev = "\n".join(lines)

    return f"""A user in {city.city} is moving through the city. Predict their next point of interest.

{header}Recent check-ins (oldest to newest): {summarize_trajectory(city, sample.recent_pois)}
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


def repair(ids, fill_pool, n_pois, k=10):
    out, invalid = [], 0
    for p in ids:
        if not (0 <= p < n_pois) or p in out:
            invalid += 1
            continue
        out.append(p)
    for p in fill_pool:
        if len(out) >= k:
            break
        if p not in out:
            out.append(p)
    return out[:k], invalid


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
    print(f"  {label}: {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"{'显著' if lo > 0 or hi < 0 else '不显著'}")


def metrics(samples, rankings):
    n = len(samples)
    h = {k: 0 for k in (1, 5, 10)}
    nd = {k: 0.0 for k in (5, 10)}
    for s, r in zip(samples, rankings):
        r = (r or [])[:10]
        if s.target in r:
            p = r.index(s.target) + 1
            for k in h:
                if p <= k:
                    h[k] += 1
            for k in nd:
                if p <= k:
                    nd[k] += 1.0 / math.log2(p + 1)
    out = {f"HR@{k}": h[k] / n for k in h}
    out |= {f"NDCG@{k}": nd[k] / n for k in nd}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="NYC")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dump", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-profile", action="store_true", help="disable the LLM user profile")
    ap.add_argument("--no-tta", action="store_true", help="single view only (equivalent to plain generation)")
    ap.add_argument("--drop", default="geo", help="comma list: transition,memory,geo,semantic (default: geo)")
    ap.add_argument("--weights", default="", help="w1,w2,w3 for the transition mix")
    ap.add_argument("--with-geo", action="store_true", help="re-enable the spatial evidence line")
    ap.add_argument("--variants", default="ABC")
    ap.add_argument("--per-view", type=int, default=8, help="evidence items per view")
    ap.add_argument("--agent-loop", action="store_true", help="Judge->Fixer loop on rule violations")
    args = ap.parse_args()

    city = load_city(args.city)
    samples = getattr(city, args.split)[: args.n]
    engine = RetrievalEngine(city)
    client = LLMClient()
    tag = "single" if args.no_tta else ("3 views" if not args.no_profile else "3 views, no profile")
    print(f"{args.city}/{args.split}: {len(samples)} samples | {tag}", flush=True)

    # ---- profiles (cached per user) ----
    store = ProfileStore(ROOT / "results" / "dump" / f"profiles_{args.city}.jsonl")
    users = sorted({s.user for s in samples})
    todo = [] if args.no_profile else store.missing(users)
    if todo:
        print(f"building {len(todo)} profiles ...", flush=True)

        def mk(u):
            p = profile_prompt(city, u)
            if p is None:
                return u, {"profile": "", "routine": ""}
            d, _ = client.chat_json(PROFILE_SYSTEM, p)
            d = d or {}
            return u, {"profile": str(d.get("profile", ""))[:600], "routine": str(d.get("routine", ""))[:200]}

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for u, p in ex.map(mk, todo):
                store.put(u, p)
        store.save()

    # ---- 3-variant generation ----
    variants = tuple(args.variants) if args.variants else ("A",)
    if args.no_tta:
        variants = ("A",)
    if args.weights:
        from grpoi.retrieval import GraphRoute

        GraphRoute.WEIGHTS = tuple(float(x) for x in args.weights.split(","))
    drop = tuple(x.strip() for x in args.drop.split(",") if x.strip())
    if args.with_geo:
        drop = tuple(d for d in drop if d != "geo")

    def one(i):
        s = samples[i]
        prof = render_profile(store, s.user) if not args.no_profile else None
        lists = []
        for v in variants:
            d, _ = client.chat_json(SYSTEM, build_prompt(v, city, s, engine, prof, args.per_view, drop), temperature=0.0)
            lists.append(parse_ids(d))
        agg = borda(lists) if len(lists) > 1 else (lists[0] if lists else [])
        fill = [c.poi for c in engine.candidates(s)]
        final, invalid = repair(agg, fill, city.n_pois())
        n_calls = len(lists)
        if args.agent_loop:
            from grpoi.agent_loop import AgentLoop

            loop = AgentLoop(city, client)
            out = loop.run(s, final, fill)
            final = repair(out.ranking, fill, city.n_pois())[0]
            n_calls += out.calls
        return i, final, invalid, n_calls

    preds = [None] * len(samples)
    inv = 0
    extra_calls = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, i): i for i, s in enumerate(samples)}
        done = 0
        for f in as_completed(futs):
            i, r, bad, nc = f.result()
            preds[i] = r
            inv += bad
            extra_calls += nc
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(samples)} ({done / (time.time() - t0):.1f}/s)", flush=True)

    g0 = None
    if args.dump and Path(args.dump).exists():
        rows = [json.loads(l) for l in Path(args.dump).read_text().splitlines() if l.strip()][: args.n]
        g0 = [r.get("final") or [] for r in rows]

    print("\n| 配置 | HR@1 | HR@5 | HR@10 | NDCG@10 |")
    print("|---|---|---|---|---|")
    if g0:
        m = metrics(samples, g0)
        print(f"| 单次生成 | {m['HR@1']:.3f} | {m['HR@5']:.3f} | {m['HR@10']:.3f} | {m['NDCG@10']:.3f} |")
    m = metrics(samples, preds)
    print(f"| **{tag}** | {m['HR@1']:.3f} | {m['HR@5']:.3f} | {m['HR@10']:.3f} | {m['NDCG@10']:.3f} |")

    if g0:
        print("\n配对差值（vs 单次生成）:")
        paired(hit_vec(samples, preds), hit_vec(samples, g0), "HR@10")
        paired(ndcg_vec(samples, preds), ndcg_vec(samples, g0), "NDCG@10")

    jump = cath = 0
    for s, r in zip(samples, preds):
        if not r:
            continue
        from grpoi.agent_loop import objective_issues

        kinds = {i.kind for i in objective_issues(city, s, r)}
        jump += "jump" in kinds
        cath += "cat_hour" in kinds
    print(f"\n非法 ID {inv} | 违规样本: 距离跳变 {jump} / 类别-时段 {cath} | "
          f"调用 {client.usage.calls} | 成本 ${client.usage.cost_usd:.4f}")
    # per-sample dump for paired analysis
    dump_path = (Path(args.out).with_suffix(".jsonl") if args.out
                 else ROOT / "results" / f"final_{args.city}_{args.split}_{args.n}.jsonl")
    with open(dump_path, "w") as fh:
        for s, r in zip(samples, preds):
            fh.write(json.dumps({"target": int(s.target), "final": list(r or [])}) + "\n")

    out = Path(args.out) if args.out else ROOT / "results" / f"final_{args.city}_{args.split}_{args.n}.json"
    out.write_text(json.dumps({
        "n": len(samples), "final": m, "single": metrics(samples, g0) if g0 else None,
        "invalid_ids": inv, "calls": client.usage.calls, "cost_usd": client.usage.cost_usd,
    }, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
