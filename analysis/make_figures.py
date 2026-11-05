"""Generate the ablation and sensitivity figures (no LLM calls)."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grpoi.data import load_city  # noqa: E402
from grpoi.retrieval import RetrievalEngine  # noqa: E402

FIG = ROOT / "results" / "figs"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})


def load_rows(p):
    return [json.loads(l) for l in open(p)]

def hit(rows, k=10):
    return np.array([1.0 if r["target"] in (r["final"] or [])[:k] else 0.0 for r in rows])

def ndcg(rows, k=10):
    out = []
    for r in rows:
        f = (r["final"] or [])[:k]
        out.append(1.0 / math.log2(f.index(r["target"]) + 2) if r["target"] in f else 0.0)
    return np.array(out)

def paired(a, b, n=2000, seed=0):
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boot = d[idx].mean(axis=1)
    return d.mean(), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ---------------------------------------------------------------- figure 1
def figure_ablation():
    base = load_rows(ROOT / "results" / "main_test_NYC.jsonl")
    configs = [
        ("- transition (local)", "abl2_no_transition"),
        ("- trajectory analogy (global)", "abl2_no_semantic"),
        ("- individual view", "abl2_no_memory"),
        ("- user profile", "abl2_no_profile"),
        ("+ spatial view (removed)", "abl2_with_geo"),
        ("SICA: single variant", "abl2_sica_A"),
        ("SICA: A+B", "abl2_sica_AB"),
        ("+ CARE", "abl2_care"),
    ]
    labels, dh, dn, lo, hi = [], [], [], [], []
    for name, f in configs:
        rows = load_rows(ROOT / "results" / f"{f}.jsonl")
        m, l, h = paired(hit(rows), hit(base))
        mn, _, _ = paired(ndcg(rows), ndcg(base))
        labels.append(name); dh.append(m * 100); dn.append(mn * 100); lo.append(l * 100); hi.append(h * 100)

    order = np.argsort(dh)
    labels = [labels[i] for i in order]
    dh = np.array(dh)[order]; dn = np.array(dn)[order]
    lo = np.array(lo)[order]; hi = np.array(hi)[order]

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    y = np.arange(len(labels))
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in dh]
    ax.barh(y, dh, color=colors, alpha=0.85, height=0.62)
    ax.errorbar(dh, y, xerr=[dh - lo, hi - dh], fmt="none", ecolor="black", elinewidth=1, capsize=2)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Δ HR@10 vs. full model (percentage points)")
    ax.set_title("Component ablation (NYC/test, n=600)")
    for yi, v in zip(y, dh):
        ax.text(v + (0.3 if v >= 0 else -0.3), yi, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ablation.png", bbox_inches="tight")
    plt.close(fig)
    return list(zip(labels, dh, dn))


# ---------------------------------------------------------------- figure 2
def evidence_cov(city, samples, engine, k):
    u = 0
    for s in samples:
        last = int(s.last_poi)
        succ = [p for p, _ in Counter(city.transitions.get(last, {})).most_common(k)]
        usr = [p for p, _ in Counter(p for p, _, _ in city.user_hist.get(s.user, [])).most_common(k)]
        sem = [p for p, _ in engine.route_lists(s).get("semantic", [])[:k]]
        u += s.target in (set(succ) | set(usr) | set(sem))
    return u / len(samples)


def figure_perview():
    city = load_city("NYC")
    samples = city.test[:100]
    engine = RetrievalEngine(city)
    ks = [5, 8, 12, 16]
    cov = [evidence_cov(city, samples, engine, k) for k in ks]
    hr, n100 = [], []
    for k in ks:
        p = ROOT / "results" / f"hp2_perview_{k}.json"
        if p.exists():
            d = json.load(open(p))["final"]
            hr.append(d["HR@10"]); n100.append(d["NDCG@10"])
        else:
            hr.append(np.nan); n100.append(np.nan)

    fig, ax1 = plt.subplots(figsize=(4.6, 3.0))
    ax1.plot(ks, cov, "o-", color="#2980b9", label="evidence coverage")
    ax1.set_xlabel("items per evidence view (k)")
    ax1.set_ylabel("evidence coverage", color="#2980b9")
    ax1.tick_params(axis="y", labelcolor="#2980b9")
    ax2 = ax1.twinx()
    ax2.plot(ks, hr, "s--", color="#c0392b", label="HR@10")
    ax2.set_ylabel("HR@10 (n=100)", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax2.set_ylim(0.45, 0.72)          # common-sense range: makes the flatness visible
    ax1.set_ylim(0.55, 0.85)
    ax1.set_xticks(ks)
    ax1.set_title("Evidence budget vs. accuracy (NYC, n=100)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_perview.png", bbox_inches="tight")
    plt.close(fig)
    return list(zip(ks, cov, hr))


# ---------------------------------------------------------------- figure 3
def figure_weights():
    city = load_city("NYC")
    samples = city.test[:100]
    engine = RetrievalEngine(city)
    from grpoi.retrieval import GraphRoute
    orig = GraphRoute.WEIGHTS
    w2s = [0.0, 0.3, 0.75, 1.5]
    w3s = [0.0, 0.75, 1.5, 3.0]
    M = np.zeros((len(w2s), len(w3s)))
    for i, w2 in enumerate(w2s):
        for j, w3 in enumerate(w3s):
            GraphRoute.WEIGHTS = (1.0, w2, w3)
            c = 0
            for s in samples:
                tr = [p for p, _ in engine.route_lists(s).get("graph", [])[:8]]
                c += s.target in tr
            M[i, j] = c / len(samples)
    GraphRoute.WEIGHTS = orig

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    im = ax.imshow(M, cmap="YlGnBu", aspect="auto", vmin=M.min(), vmax=M.max())
    ax.set_xticks(range(len(w3s))); ax.set_xticklabels(w3s)
    ax.set_yticks(range(len(w2s))); ax.set_yticklabels(w2s)
    ax.set_xlabel("$w_3$ (second-order weight)")
    ax.set_ylabel("$w_2$ (unconditional weight)")
    for i in range(len(w2s)):
        for j in range(len(w3s)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] > M.max() * 0.75 else "black", fontsize=8)
    ax.set_title("Transition weight grid: coverage@8\n($w_1$=1.0, NYC, n=100)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "fig_weights.png", bbox_inches="tight")
    plt.close(fig)
    return M.tolist()


if __name__ == "__main__":
    print("ablation:", figure_ablation())
    print("perview:", figure_perview())
    print("weights:", figure_weights())
    print("figures ->", FIG)
