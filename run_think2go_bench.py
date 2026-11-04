"""Evaluate our training-free method on the Think2Go / GNPR-SID benchmark.

Data: GNPR-SID `V1/datasets/nyc/llm_{train,val,test}.json`
      each item = {instruction, input: "<history> ... When <ts> user_u is likely to visit:", output: SID}

The public artifact ships no coordinates or categories, so the geo and
category-hour routes cannot be used; the temporal transition model (which is the
dominant channel in our framework) runs unchanged.

Metrics follow Think2Go: Acc@1/5/10 and MRR.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

TRIPLE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) visited ((?:<[a-z]_\d+>)+)")
USER = re.compile(r"[Uu]ser_(\d+)")
TARGET_TS = re.compile(r"When (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def parse(path: Path):
    items = json.loads(path.read_text())
    out = []
    for it in items:
        inp = it["input"]
        hist = [(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"), sid) for ts, sid in TRIPLE.findall(inp)]
        m = TARGET_TS.search(inp)
        target_ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") if m else None
        user = int(USER.search(inp).group(1))
        out.append({"user": user, "hist": hist, "target_ts": target_ts, "answer": it["output"].strip()})
    return out


def bucket(h: int) -> int:
    return h // 3


class TemporalModel:
    def __init__(self, train):
        self.t1 = defaultdict(Counter)
        self.t1h = defaultdict(Counter)
        self.t2 = defaultdict(Counter)
        self.user_hist = defaultdict(Counter)
        for it in train:
            seq = it["hist"] + [(it["target_ts"], it["answer"])]
            for ts, sid in seq:
                self.user_hist[it["user"]][sid] += 1
            for i in range(1, len(seq)):
                prev_ts, prev = seq[i - 1]
                cur_ts, cur = seq[i]
                self.t1[prev][cur] += 1
                self.t1h[(prev, bucket(cur_ts.hour))][cur] += 1
                if i >= 2:
                    self.t2[(seq[i - 2][1], prev)][cur] += 1

    @staticmethod
    def _p(c):
        if not c:
            return {}
        tot = sum(c.values())
        return {k: v / tot for k, v in c.items()}

    def rank(self, it, k=10, weights=(1.0, 0.3, 1.5)):
        hist = it["hist"]
        if not hist:
            return []
        w1h, w1, w2 = weights
        last = hist[-1][1]
        hb = bucket(it["target_ts"].hour)
        score = defaultdict(float)
        for d, v in self._p(self.t1h.get((last, hb))).items():
            score[d] += w1h * v
        for d, v in self._p(self.t1.get(last)).items():
            score[d] += w1 * v
        if len(hist) >= 2:
            for d, v in self._p(self.t2.get((hist[-2][1], last))).items():
                score[d] += w2 * v
        ranked = sorted(score.items(), key=lambda x: -x[1])[:k]
        return [p for p, _ in ranked]


def evaluate(model, items, k_list=(1, 5, 10)):
    scores = {k: 0 for k in k_list}
    mrr = 0.0
    for it in items:
        ranked = model.rank(it, max(k_list))
        if it["answer"] in ranked:
            pos = ranked.index(it["answer"]) + 1
            for k in k_list:
                if pos <= k:
                    scores[k] += 1
            mrr += 1.0 / pos
    n = len(items)
    return {f"Acc@{k}": scores[k] / n for k in k_list} | {"MRR": mrr / n, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/GNPR-SID/V1/datasets/nyc")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    d = Path(args.data)
    train = parse(d / "llm_train.json")
    test = parse(d / f"llm_{args.split}.json")
    print(f"train items {len(train)} | {args.split} items {len(test)}")

    # sanity: statistics
    users = {it["user"] for it in test}
    ans_in_hist = sum(1 for it in test if it["answer"] in [s for _, s in it["hist"]])
    print(f"test users {len(users)} | answer already in history {ans_in_hist / len(test):.3f}")

    model = TemporalModel(train)

    # baselines for context
    pop = Counter()
    for it in train:
        for _, s in it["hist"]:
            pop[s] += 1
    pop_top = [p for p, _ in pop.most_common(10)]

    ufreq = TemporalModel.__new__(TemporalModel)  # reuse counters only
    ufreq.user_hist = model.user_hist

    class Pop:
        def rank(self, it, k=10):
            return pop_top[:k]

    class UserFreq:
        def rank(self, it, k=10):
            c = model.user_hist.get(it["user"])
            return [p for p, _ in c.most_common(k)] if c else pop_top[:k]

    print("\n| model | Acc@1 | Acc@5 | Acc@10 | MRR |")
    print("|---|---|---|---|---|")
    for name, m in [("POP", Pop()), ("UserFreq", UserFreq()), ("TemporalMarkov", model)]:
        r = evaluate(m, test)
        print(f"| {name} | {r['Acc@1']:.4f} | {r['Acc@5']:.4f} | {r['Acc@10']:.4f} | {r['MRR']:.4f} |")


if __name__ == "__main__":
    main()
