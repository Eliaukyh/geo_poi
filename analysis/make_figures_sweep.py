"""Sensitivity figures for the two-term transition model.

Two sweeps (NYC/test, n=200, full pipeline):
  fig_ratio.png : w1:w3 ratio (w2 fixed to 0)  over {1:0, 1:1, 1:2, 1:3}
  fig_count.png : evidence items per view       over {0, 2, 4, 8, 10, 12}

Each figure puts the four metrics in one panel.

Usage:
    cd geo-rag-poi && source ../.tools/uv-env.sh
    UV_CACHE_DIR=../.tools/uv-cache MPLCONFIGDIR=../.tools/mplcache \
        uv run --with matplotlib python analysis/make_figures_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figs" / "final"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 220,
                     "axes.spines.top": False, "axes.spines.right": False})

METRICS = ["HR@5", "HR@10", "NDCG@5", "NDCG@10"]
COLORS = ["#2c7fb8", "#e6550d", "#31a354", "#756bb1"]


def read(name):
    d = json.load(open(ROOT / "results" / f"{name}.json"))["final"]
    return [d[m] * 100 for m in METRICS]


def draw(xs, runs, xlabel, title, fname):
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    xs_pos = list(range(len(xs)))
    for i, m in enumerate(METRICS):
        ys = [read(r)[i] for r in runs]
        ax.plot(xs_pos, ys, marker="o", lw=1.7, color=COLORS[i], label=m)
        for x, y in zip(xs_pos, ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=6.5, color=COLORS[i])
    ax.set_xticks(xs_pos); ax.set_xticklabels(xs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("score (%)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw(["1:0", "1:1", "1:2", "1:3"],
         ["r_1_0", "r_1_1", "r_1_2", "r_1_3"],
         "$w_1 : w_3$   ($w_2 = 0$, middle term removed)",
         "Transition ratio (NYC, n=200)", "fig_ratio.png")

    draw(["0", "2", "4", "8", "10", "12"],
         ["k_0", "k_2", "k_4", "r_1_2", "k_10", "k_12"],
         "evidence items per view (k)   ($w_1:w_3=1:2$)",
         "Evidence budget (NYC, n=200)", "fig_count.png")

    print("figures ->", OUT)
