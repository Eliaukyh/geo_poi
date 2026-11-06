"""Ablation and sensitivity figures (4 metrics each), NYC, n=200."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figs"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})

METRICS = ["HR@5", "HR@10", "NDCG@5", "NDCG@10"]


def load(name):
    d = json.load(open(ROOT / "results" / f"{name}.json"))["final"]
    return [d[m] for m in METRICS]


# ------------------------------------------------------------------ 1. ablation
ABL = [
    ("Full", "abl3_full"),
    ("− collective\n(transition+analogy)", "abl3_no_collective"),
    ("− user model\n(view+profile)", "abl3_no_user"),
    ("+ spatial view", "abl3_with_geo"),
    ("SICA: single", "abl3_sica_A"),
    ("SICA: A+B", "abl3_sica_AB"),
    ("+ CARE", "abl3_care"),
]


def fig_ablation():
    labels = [a for a, _ in ABL]
    vals = np.array([load(f) for _, f in ABL])          # (configs, 4)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.6), sharex=True)
    x = np.arange(len(labels))
    for ax, j, title in zip(axes.ravel(), range(4), METRICS):
        colors = ["#34495e"] + ["#c0392b" if v < vals[0, j] else "#27ae60" for v in vals[1:, j]]
        ax.bar(x, vals[:, j], color=colors, alpha=0.9, width=0.65)
        ax.axhline(vals[0, j], color="black", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(title)
        ax.set_ylim(min(vals[:, j]) * 0.85, max(vals[:, j]) * 1.06)
        for xi, v in zip(x, vals[:, j]):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=7)
    fig.suptitle("Ablation on NYC/test (n=200); dashed line = full model", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ablation.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 2. per-view
PERVIEW = [(5, "hp3_perview_5"), (8, "abl3_full"), (12, "hp3_perview_12"), (16, "hp3_perview_16")]


def fig_perview():
    ks = [k for k, _ in PERVIEW]
    vals = np.array([load(f) for _, f in PERVIEW])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    markers = ["o", "s", "^", "D"]
    for j, m in enumerate(METRICS):
        ax.plot(ks, vals[:, j], marker=markers[j], label=m)
    ax.set_xticks(ks)
    ax.set_xlabel("evidence items per view (k)")
    ax.set_ylabel("score")
    ax.set_title("Evidence budget sensitivity (NYC, n=200)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_perview.png", bbox_inches="tight")
    plt.close(fig)
    return list(zip(ks, vals.tolist()))


# ------------------------------------------------------------------ 3. weights
WEIGHTS = [
    ("0.0 / 0.0", "hp3_w_0_0"),
    ("0.3 / 0.0", "hp3_w_3_0"),
    ("0.3 / 1.5", "abl3_full"),
    ("0.0 / 1.5", "hp3_w_0_15"),
    ("1.5 / 1.5", "hp3_w_15_15"),
    ("0.3 / 3.0", "hp3_w_3_30"),
]


def fig_weights():
    labels = [a for a, _ in WEIGHTS]
    vals = np.array([load(f) for _, f in WEIGHTS])
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.6), sharex=True)
    x = np.arange(len(labels))
    for ax, j, title in zip(axes.ravel(), range(4), METRICS):
        ax.bar(x, vals[:, j], color=["#c0392b" if v < vals[2, j] else "#27ae60" for v in vals[:, j]],
               alpha=0.9, width=0.65)
        ax.axhline(vals[2, j], color="black", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(title)
        ax.set_ylim(min(vals[:, j]) * 0.85, max(vals[:, j]) * 1.06)
        for xi, v in zip(x, vals[:, j]):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=7)
        ax.set_xlabel("$w_2$ / $w_3$  ($w_1$=1.0)")
    fig.suptitle("Transition weight sensitivity (NYC, n=200); dashed = default 0.3/1.5", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig_weights.png", bbox_inches="tight")
    plt.close(fig)
    return list(zip(labels, vals.tolist()))


if __name__ == "__main__":
    fig_ablation()
    print("perview:", fig_perview())
    print("weights:", fig_weights())
    print("figures ->", FIG)
