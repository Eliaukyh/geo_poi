"""Final figures: module/source ablation + per-view & weight sensitivity.

All numbers are read from results/*.json written by run.py.
Every figure puts the four metrics (HR@5, HR@10, NDCG@5, NDCG@10) together.

Usage:
    cd geo-rag-poi
    source ../.tools/uv-env.sh
    UV_CACHE_DIR=../.tools/uv-cache MPLCONFIGDIR=../.tools/mplcache \
        uv run --with matplotlib python analysis/make_figures_final.py

Outputs (results/figs/final/):
    fig_ablation.png          module-level: Ours / -MMEG / -SICA / -CARE
    fig_ablation_source.png   source-level: Full / -collective / -user / +spatial
    fig_perview.png           four metric curves vs items per view
    fig_weights.png           four metric curves vs transition weights
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figs" / "final"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "figure.dpi": 220,
                     "axes.spines.top": False, "axes.spines.right": False})

METRICS = ["HR@5", "HR@10", "NDCG@5", "NDCG@10"]
COLORS = ["#2c7fb8", "#e6550d", "#31a354", "#756bb1"]


def load(name: str) -> list[float]:
    d = json.load(open(ROOT / "results" / f"{name}.json"))["final"]
    return [d[m] for m in METRICS]


def bar_panels(entries, title, fname, note=""):
    """2x2 panels, one per metric; entries = [(label, results_name), ...]."""
    labels = [e[0] for e in entries]
    vals = np.array([load(e[1]) for e in entries])      # (config, 4)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.8), sharex=True)
    x = np.arange(len(labels))
    for ax, j, m in zip(axes.ravel(), range(4), METRICS):
        base = vals[0, j]
        colors = ["#34495e"] + ["#c0392b" if v < base else "#27ae60" for v in vals[1:, j]]
        ax.bar(x, vals[:, j], color=colors, alpha=0.9, width=0.65)
        ax.axhline(base, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(m)
        lo = min(vals[:, j]); hi = max(vals[:, j])
        ax.set_ylim(lo - (hi - lo) * 0.25, hi + (hi - lo) * 0.15)
        for xi, v in zip(x, vals[:, j]):
            ax.text(xi, v, f"{v*100:.1f}", ha="center", va="bottom", fontsize=6.5)
    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    fig.suptitle(title + ("  " + note if note else ""), y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)


def line_chart(xs, entries, xlabel, title, fname):
    """One panel, four metric curves."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for i, m in enumerate(METRICS):
        ys = [load(e[1])[i] * 100 for e in entries]
        ax.plot(xs, ys, marker="o", label=m, color=COLORS[i], lw=1.6)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=6.5, color=COLORS[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("score (%)")
    ax.set_title(title)
    ax.set_xticks(xs)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # 1. module-level ablation (final configuration, CARE on)
    bar_panels(
        [("Ours", "f_ours"), ("− MMEG", "f_no_mmeg_pure"),
         ("− SICA", "f_sica_A"), ("− CARE", "f_no_care")],
        "Module ablation (NYC/test, n=200)", "fig_ablation.png",
        note="dashed = Ours",
    )

    # 2. source-level ablation (calibration off, otherwise it masks the effect)
    bar_panels(
        [("Full", "s2_full"), ("− collective", "s2_no_collective"),
         ("− user model", "s2_no_user"), ("+ spatial", "s2_with_geo")],
        "Evidence-source ablation (NYC/test, n=200, CARE off)", "fig_ablation_source.png",
        note="dashed = Full",
    )

    # 3. per-view sensitivity
    line_chart(
        [5, 8, 12, 16],
        [("5", "f_pv_5"), ("8", "f_ours"), ("12", "f_pv_12"), ("16", "f_pv_16")],
        "evidence items per view (k)",
        "Evidence budget (NYC, n=200, full pipeline)",
        "fig_perview.png",
    )

    # 4. transition-weight sensitivity
    line_chart(
        ["0/0", "0.3/0", "0.3/1.5", "0/1.5", "1.5/1.5", "0.3/3.0"],
        [("a", "f_w_0_0"), ("b", "f_w_3_0"), ("c", "f_ours"),
         ("d", "f_w_0_15"), ("e", "f_w_15_15"), ("f", "f_w_3_30")],
        "$w_2$ / $w_3$   ($w_1$=1.0)",
        "Transition weights (NYC, n=200, full pipeline)",
        "fig_weights.png",
    )

    print("figures ->", OUT)
