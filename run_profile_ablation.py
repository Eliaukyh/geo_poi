"""Ablation: does an LLM-generated user profile improve generation?

Profiles are generated once per user (cached), then injected into the generation
prompt.  Compared against the direct-generation dump on the same samples, with a
paired bootstrap.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from grpoi.data import load_city  # noqa: E402
from grpoi.generative import GenerativeAgent  # noqa: E402
from grpoi.llm import LLMClient  # noqa: E402
from grpoi.profile import ProfileStore, profile_prompt, render_profile  # noqa: E402
from grpoi.profile import SYSTEM as PROFILE_SYSTEM  # noqa: E402
from grpoi.retrieval import RetrievalEngine  # noqa: E402


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
    agent = GenerativeAgent(city, engine=engine, client=client)

    store = ProfileStore(ROOT / "results" / "dump" / f"profiles_{args.city}.jsonl")
    users = sorted({s.user for s in samples})
    todo = store.missing(users)
    print(f"{args.city}: {len(samples)} samples | {len(users)} users | profiles to build: {len(todo)}", flush=True)

    # ---- build profiles (once per user) ----
    t0 = time.time()
    if todo:
        def make_profile(u):
            prompt = profile_prompt(city, u)
            if prompt is None:
                return u, {"profile": "", "routine": ""}
            d, _ = client.chat_json(PROFILE_SYSTEM, prompt)
            d = d or {}
            return u, {"profile": str(d.get("profile", ""))[:600], "routine": str(d.get("routine", ""))[:200]}

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(make_profile, u): u for u in todo}
            for f in as_completed(futs):
                u, p = f.result()
                store.put(u, p)
                done += 1
                if done % 50 == 0:
                    print(f"  profiles {done}/{len(todo)} ({done / (time.time() - t0):.1f}/s)", flush=True)
        store.save()

    # ---- generate with profile ----
    def run_one(i):
        s = samples[i]
        txt = render_profile(store, s.user)
        pred = agent.run(s, s.target, profile_text=txt)
        return i, pred.ranking

    preds = [None] * len(samples)
    t1 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, i): i for i, s in enumerate(samples)}
        for f in as_completed(futs):
            i, r = f.result()
            preds[i] = r
            done += 1
            if done % 100 == 0:
                print(f"  gen {done}/{len(samples)} ({done / (time.time() - t1):.1f}/s)", flush=True)

    g0 = None
    if args.dump and Path(args.dump).exists():
        rows = [json.loads(l) for l in Path(args.dump).read_text().splitlines() if l.strip()][: args.n]
        g0 = [r.get("final") or [] for r in rows]

    print("\n| 配置 | HR@1 | HR@5 | HR@10 | NDCG@10 |")
    print("|---|---|---|---|---|")
    if g0:
        m = metrics(samples, g0)
        print(f"| G0 无画像 | {m['HR@1']:.3f} | {m['HR@5']:.3f} | {m['HR@10']:.3f} | {m['NDCG@10']:.3f} |")
    m = metrics(samples, preds)
    print(f"| +LLM 画像 | {m['HR@1']:.3f} | {m['HR@5']:.3f} | {m['HR@10']:.3f} | {m['NDCG@10']:.3f} |")

    if g0:
        print("\n配对差值:")
        paired(hit_vec(samples, preds), hit_vec(samples, g0), "画像 - 无画像  HR@10")
        paired(ndcg_vec(samples, preds), ndcg_vec(samples, g0), "画像 - 无画像  NDCG@10")

    print(f"\n调用 {client.usage.calls} | 成本 ${client.usage.cost_usd:.4f}")
    out = ROOT / "results" / f"profile_{args.city}_{args.n}.json"
    out.write_text(json.dumps({
        "n": len(samples), "users": len(users),
        "with_profile": metrics(samples, preds),
        "without_profile": metrics(samples, g0) if g0 else None,
        "calls": client.usage.calls, "cost_usd": client.usage.cost_usd,
    }, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
