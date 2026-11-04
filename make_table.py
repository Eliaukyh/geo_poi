"""Merge our runs with a RALLM-POI re-scored file into one comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLS = [
    ("Recall@1", "R@1"),
    ("Recall@5", "R@5"),
    ("Recall@10", "R@10"),
    ("NDCG@5", "N@5"),
    ("NDCG@10", "N@10"),
    ("MAP@5", "M@5"),
    ("MAP@10", "M@10"),
    ("geo_err_km", "GeoErr"),
    ("calls_per_sample", "Calls"),
    ("cost_usd_per_sample", "USD/s"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="results json from run_eval.py")
    ap.add_argument("--rallm", default="", help="metrics json from compare_rallm.py --out")
    ap.add_argument("--out", default="")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    ours = json.loads(Path(args.ours).read_text())["results"]
    rows: dict[str, dict] = {}
    if args.rallm:
        r = json.loads(Path(args.rallm).read_text())
        r.setdefault("calls_per_sample", 2.0)
        rows["RALLM-POI (repro)"] = r
    rows.update(ours)

    head = "| Method | " + " | ".join(label for _, label in COLS) + " |"
    lines = [head, "|" + "---|" * (len(COLS) + 1)]
    for name, m in rows.items():
        cells = []
        for key, _ in COLS:
            v = m.get(key)
            if v is None:
                cells.append("-")
            elif isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    if args.title:
        table = f"**{args.title}**\n\n" + table
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n")
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
