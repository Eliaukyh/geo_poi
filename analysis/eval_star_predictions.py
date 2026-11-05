"""Score RALLM-STAR predictions (possibly sharded) with our metric implementation.

RALLM-STAR writes lines of the form
    Sample 12: tensor([ ... ]), 39
and, when run in shards, one file per shard (`predictions_*_shard{i}.txt`).
This script takes the newest file per shard, merges them, and scores them against
the same split, so the numbers are directly comparable with ours.

Usage:
  python analysis/eval_star_predictions.py --city NYC --dir /tmp/RALLM-STAR/experiments_reproduce
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grpoi.data import load_city  # noqa: E402

LINE = re.compile(r"Sample\s+(\d+):\s*tensor\(\[(.*?)\]\),\s*(\d+)")


def newest_per_shard(folder: Path) -> list[Path]:
    files = sorted(folder.glob("predictions_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    by_shard = {}
    for f in files:
        m = re.search(r"_shard(\d+)\.txt$", f.name)
        key = m.group(1) if m else "all"
        if key not in by_shard:
            by_shard[key] = f
    return list(by_shard.values())


def parse(path: Path) -> dict[int, list[int]]:
    out = {}
    for line in path.read_text().splitlines():
        m = LINE.search(line)
        if not m:
            continue
        out[int(m.group(1))] = [int(x) for x in re.findall(r"-?\d+", m.group(2))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="NYC")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--dir", default="/tmp/RALLM-STAR/experiments_reproduce")
    args = ap.parse_args()

    city = load_city(args.city)
    samples = getattr(city, args.split)
    folder = Path(args.dir) / next(
        (d.name for d in Path(args.dir).iterdir() if d.is_dir() and args.city in d.name), ""
    )
    files = newest_per_shard(folder)
    print(f"{args.city}/{args.split}: {len(samples)} samples | {len(files)} prediction file(s)")

    preds: dict[int, list[int]] = {}
    for f in files:
        got = parse(f)
        preds.update(got)
        print(f"  {f.name}: {len(got)} samples")

    missing = [i for i in range(len(samples)) if i not in preds]
    if missing:
        print(f"  ! missing {len(missing)} samples (e.g. {missing[:5]})")

    ks = (1, 5, 10)
    rec = {k: 0.0 for k in ks}
    nd = {k: 0.0 for k in ks}
    mp = {k: 0.0 for k in ks}
    n = 0
    for i, s in enumerate(samples):
        if i not in preds:
            continue
        n += 1
        ranking = preds[i][:10]
        if s.target in ranking:
            pos = ranking.index(s.target) + 1
            for k in ks:
                if pos <= k:
                    rec[k] += 1
                    nd[k] += 1.0 / math.log2(pos + 1)
                    mp[k] += 1.0 / pos

    print(f"\nscored n={n}")
    for k in ks:
        print(f"  Recall@{k} {rec[k]/n:.4f} | NDCG@{k} {nd[k]/n:.4f} | MAP@{k} {mp[k]/n:.4f}")


if __name__ == "__main__":
    main()
