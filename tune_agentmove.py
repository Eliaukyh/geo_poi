"""Tune and evaluate our method on AgentMove's Tokyo protocol (fast: components cached)."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_agentmove_bench import (  # noqa: E402
    Model, build_city_pois, build_trajectories, evaluate, load_checkins, sample_tests, split_and_build,
)

DATA = Path("/tmp/agentmove_data/dataset_tist2015")


def main(city: str = "Tokyo"):
    pois_all = build_city_pois(DATA, Path("/tmp/agentmove_data/pois_by_city.pkl"))
    df = load_checkins(city, pois_all, DATA)
    df = build_trajectories(df)
    train, val, items = split_and_build(df)
    tests = sample_tests(items)

    val_items = [(u[0], g) for u, g in val.groupby(["user", "traj"]) if len(g) >= 4]
    vtests = sample_tests(val_items, n=200)
    pois = df[["venue", "lat", "lng"]].drop_duplicates("venue").reset_index(drop=True)
    model = Model(train, pois)
    print(f"{city}: train rows {len(train):,} | test {len(tests)} | val {len(vtests)}")

    print("precomputing components ...", flush=True)
    vcomp = [model.components(u, g.iloc[:-1]) for u, g in vtests]
    tcomp = [model.components(u, g.iloc[:-1]) for u, g in tests]

    best = None
    grid = list(itertools.product([1, 2, 3], [0.5, 1, 2], [0, 0.5, 1, 2], [0.2, 0.5, 1]))
    print(f"grid size {len(grid)}", flush=True)
    for w in grid:
        model.w = w
        r = evaluate(model, vtests, k=5, comps=vcomp)
        if best is None or r["Acc@5"] > best[0]:
            best = (r["Acc@5"], w, r)

    print("best on VAL:", best[1], {k: round(v, 4) if isinstance(v, float) else v for k, v in best[2].items()})
    model.w = best[1]
    print("TEST:", {k: round(v, 4) if isinstance(v, float) else v for k, v in evaluate(model, tests, comps=tcomp).items()})

    # single-route ablations on test
    for name, w in [("trans only", (1, 0, 0, 0)), ("user only", (0, 1, 0, 0)),
                    ("geo only", (0, 0, 1, 0)), ("pop only", (0, 0, 0, 1))]:
        model.w = w
        r = evaluate(model, tests, comps=tcomp)
        print(f"  {name:11s} Acc@1={r['Acc@1']:.3f} Acc@5={r['Acc@5']:.3f} NDCG@5={r['NDCG@5']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Tokyo")
