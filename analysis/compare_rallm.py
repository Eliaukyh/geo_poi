"""Re-score a RALLM-POI predictions file with this project's metric definitions.

RALLM-POI writes lines of the form

    Sample 3: tensor([ 35, 243, ...]), 39

which we parse into a ranking and score with exactly the same Recall/NDCG/MAP
implementation used for our own method, so the comparison is apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grpoi import config  # noqa: E402
from grpoi.data import load_city  # noqa: E402
from grpoi.metrics import KS, map_at, ndcg_at  # noqa: E402

LINE = re.compile(r"Sample\s+(\d+):\s*tensor\(\[(.*?)\]\),\s*(\d+)")


def parse(path: Path) -> dict[int, tuple[list[int], int]]:
    out = {}
    for line in path.read_text().splitlines():
        m = LINE.search(line)
        if not m:
            continue
        idx = int(m.group(1))
        nums = [int(x) for x in re.findall(r"-?\d+", m.group(2))]
        label = int(m.group(3))
        out[idx] = (nums, label)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="PHO")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--file", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    city = load_city(args.city)
    samples = getattr(city, args.split)
    preds = parse(Path(args.file))
    print(f"parsed {len(preds)} predictions; split has {len(samples)} samples")

    n = args.n or len(samples)
    rec = {k: 0.0 for k in KS}
    nd = {k: 0.0 for k in KS}
    mp = {k: 0.0 for k in KS}
    missing = 0
    dup = 0
    invalid = 0
    for i, s in enumerate(samples[:n]):
        if i not in preds:
            missing += 1
            continue
        ranking, label = preds[i]
        target = int(s.target)  # cross-check against the pickle
        if label != target:
            print(f"  ! label mismatch at {i}: file={label} pickle={target}")
        dup += len(ranking) - len(set(ranking))
        invalid += sum(1 for x in ranking if not (0 <= x < city.n_pois()))
        pos = ranking.index(target) + 1 if target in ranking else None
        if pos is not None:
            for k in KS:
                if pos <= k:
                    rec[k] += 1
                    nd[k] += ndcg_at(pos, k)
                    mp[k] += map_at(pos, k)

    n_ok = n - missing
    out = {"n": n_ok, "missing": missing, "dup_per_sample": dup / max(n_ok, 1), "invalid_per_sample": invalid / max(n_ok, 1)}
    for k in KS:
        out[f"Recall@{k}"] = rec[k] / max(n_ok, 1)
    for k in KS:
        out[f"NDCG@{k}"] = nd[k] / max(n_ok, 1)
    for k in KS:
        out[f"MAP@{k}"] = mp[k] / max(n_ok, 1)

    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
