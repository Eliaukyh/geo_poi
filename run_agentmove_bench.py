"""Run our training-free method under AgentMove's evaluation protocol.

Protocol reproduced from the AgentMove repo (tsinghua-fib-lab/AgentMove):
  * TIST2015 global Foursquare data, POIs assigned to the nearest city centroid
  * per user, check-ins grouped into 72-hour trajectories
  * users with < 5 trajectories dropped; test trajectories with < 4 stays dropped
  * per user: first 70% trajectories = train, next 10% = val, last 20% = test
  * a test instance = (user, test trajectory); history comes from that user's train
    trajectories (last 40 stays), context = the test trajectory, answer = its last stay
  * 200 test instances per city, one trajectory per user, seed 1234
Metrics: Acc@1, Acc@5, NDCG@5 (as in AgentMove Table 1).

Usage:
  python run_agentmove_bench.py --city Tokyo --data /tmp/agentmove_data/dataset_tist2015
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HOUR_BINS = 72
TRAJ_MIN_LEN = 5
MIN_TEST_STAYS = 4
HISTORY_STAYS = 40
TEST_SAMPLE = 200
SEED = 1234


def haversine_matrix(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


TARGET_CITIES = ("Tokyo", "Moscow", "SaoPaulo")


def build_city_pois(data_dir: Path, cache: Path) -> pd.DataFrame:
    """Assign every POI to its nearest city centroid, chunked; cache the result."""
    if cache.exists():
        return pd.read_pickle(cache)
    cities = pd.read_csv(
        data_dir / "dataset_TIST2015_Cities.txt", sep="\t", header=None,
        names=["name", "lat", "lng", "a", "b", "c"], dtype={0: str}, quoting=3,
    )
    keep = []
    reader = pd.read_csv(
        data_dir / "dataset_TIST2015_POIs.txt", sep="\t", header=None,
        names=["venue", "lat", "lng", "cat", "country"],
        dtype={0: str}, quoting=3, chunksize=300_000,
    )
    for chunk in reader:
        d = haversine_matrix(
            chunk["lat"].values[:, None], chunk["lng"].values[:, None],
            cities["lat"].values[None, :], cities["lng"].values[None, :],
        )
        chunk = chunk.assign(city=cities["name"].values[np.argmin(d, axis=1)])
        keep.append(chunk[chunk["city"].isin(TARGET_CITIES)])
    pois = pd.concat(keep, ignore_index=True)
    pois.to_pickle(cache)
    return pois


def load_checkins(city: str, pois: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    cache = Path(f"/tmp/agentmove_data/checkins_{city}.pkl")
    if cache.exists():
        return pd.read_pickle(cache)
    venues = set(pois.loc[pois["city"] == city, "venue"])
    coord = pois[pois["city"] == city][["venue", "lat", "lng", "cat"]]
    parts = []
    reader = pd.read_csv(
        data_dir / "dataset_TIST2015_Checkins.txt", sep="\t", header=None,
        names=["user", "venue", "utc_time", "offset"],
        dtype={0: str, 1: str}, quoting=3, chunksize=2_000_000,
    )
    for chunk in reader:
        chunk = chunk[chunk["venue"].isin(venues)]
        if len(chunk):
            parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    df = df.merge(coord, on="venue", how="left")
    df["dt"] = pd.to_datetime(df["utc_time"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["dt"]).sort_values(["user", "dt"]).reset_index(drop=True)
    df.to_pickle(cache)
    return df


def build_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """72-hour anchored windows per user (mirrors _group_by_time_window)."""
    traj_ids = np.zeros(len(df), dtype=np.int64)
    counter = 0
    for _, idx in df.groupby("user").groups.items():  # indices, already time-sorted
        idx = np.asarray(idx)
        start = df.at[idx[0], "dt"]
        for i in idx:
            if df.at[i, "dt"] > start + pd.Timedelta(hours=HOUR_BINS):
                counter += 1
                start = df.at[i, "dt"]
            traj_ids[i] = counter
        counter += 1
    df = df.copy()
    df["traj"] = traj_ids
    return df


def split_and_build(df: pd.DataFrame):
    train_rows, val_rows, test_items = [], [], []
    for user, g in df.groupby("user"):
        trajs = list(dict.fromkeys(g["traj"].tolist()))  # chronological order of appearance
        if len(trajs) < TRAJ_MIN_LEN:
            continue
        n = len(trajs)
        train_ids = trajs[: int(0.7 * n)]
        val_ids = trajs[int(0.7 * n) : int(0.8 * n)]
        test_ids = trajs[int(0.8 * n) :]

        hist = g[g["traj"].isin(train_ids)]
        train_rows.append(hist)
        val_rows.append(g[g["traj"].isin(val_ids)])
        for t in test_ids:
            tg = g[g["traj"] == t]
            if len(tg) < MIN_TEST_STAYS:
                continue
            test_items.append((user, tg))
    return pd.concat(train_rows) if train_rows else df.iloc[:0], \
           pd.concat(val_rows) if val_rows else df.iloc[:0], test_items


def sample_tests(items, n=TEST_SAMPLE, seed=SEED):
    rng = random.Random(seed)
    by_user = defaultdict(list)
    for u, g in items:
        by_user[u].append(g)
    users = list(by_user)
    rng.shuffle(users)
    out = []
    for u in users:
        out.append((u, rng.choice(by_user[u])))
        if len(out) == n:
            break
    return out


class Model:
    """Temporal transition + user profile + geography + popularity."""

    def __init__(self, train: pd.DataFrame, pois: pd.DataFrame, w=(1.0, 1.0, 0.5, 0.5)):
        self.w = w
        self.pois = pois
        self.coords = pois[["lat", "lng"]].values.astype(float)
        self.index = {v: i for i, v in enumerate(pois["venue"].values)}
        self.t1, self.t1h, self.t2 = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        self.upois = defaultdict(Counter)
        self.pop = Counter()
        for _, g in train.groupby(["user", "traj"]):
            seq = list(zip(g["venue"], g["dt"]))
            for v, _ in seq:
                self.upois[g["user"].iloc[0]][v] += 1
                self.pop[v] += 1
            for i in range(1, len(seq)):
                a, b = seq[i - 1][0], seq[i][0]
                hb = seq[i][1].hour // 3
                self.t1[a][b] += 1
                self.t1h[(a, hb)][b] += 1
                if i >= 2:
                    self.t2[(seq[i - 2][0], a)][b] += 1

    @staticmethod
    def _p(c):
        if not c:
            return {}
        tot = sum(c.values()) or 1
        return {k: v / tot for k, v in c.items()}

    def components(self, user, ctx: pd.DataFrame, top_geo=300):
        """Per-route score dicts, computed once per instance (weight-free)."""
        last = ctx["venue"].iloc[-1]
        hb = ctx["dt"].iloc[-1].hour // 3
        comp = {}
        t = defaultdict(float)
        for d, v in self._p(self.t1h.get((last, hb))).items():
            t[d] += v
        for d, v in self._p(self.t1.get(last)).items():
            t[d] += v
        if len(ctx) >= 2:
            prev = ctx["venue"].iloc[-2]
            for d, v in self._p(self.t2.get((prev, last))).items():
                t[d] += v
        comp["trans"] = dict(t)
        comp["user"] = self._p(self.upois.get(user))
        comp["pop"] = self._p(self.pop)
        g = {}
        if last in self.index:
            anchor = self.coords[self.index[last]]
            dist = haversine_matrix(anchor[0], anchor[1], self.coords[:, 0], self.coords[:, 1])
            idx = np.argsort(dist)[:top_geo]
            for i in idx:
                v = float(np.exp(-dist[i] / 3.0))
                if v > 1e-3:
                    g[self.pois["venue"].values[i]] = v
        comp["geo"] = g
        return comp

    def rank_from_components(self, comp, w, k=5):
        wt, wu, wg, wp = w
        score = defaultdict(float)
        for name, weight in (("trans", wt), ("user", wu), ("geo", wg), ("pop", wp)):
            if weight <= 0:
                continue
            for d, v in comp[name].items():
                score[d] += weight * v
        return [p for p, _ in sorted(score.items(), key=lambda x: -x[1])[:k]]

def evaluate(model, tests, k=5, comps=None):
    acc1 = acc5 = 0
    ndcg = 0.0
    n_ok = 0
    for i, (user, ctx) in enumerate(tests):
        target = ctx["venue"].iloc[-1]
        context = ctx.iloc[:-1]          # never let the model see the answer
        if len(context) == 0:
            continue
        n_ok += 1
        comp = comps[i] if comps is not None else model.components(user, context)
        ranked = model.rank_from_components(comp, model.w, k)
        if target in ranked:
            pos = ranked.index(target) + 1
            if pos == 1:
                acc1 += 1
            acc5 += 1
            ndcg += 1.0 / math.log2(pos + 1)
    n = n_ok or len(tests)
    return {"Acc@1": acc1 / n, "Acc@5": acc5 / n, "NDCG@5": ndcg / n, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Tokyo")
    ap.add_argument("--data", default="/tmp/agentmove_data/dataset_tist2015")
    args = ap.parse_args()

    data_dir = Path(args.data)
    print("mapping POIs to cities (cached) ...")
    pois_all = build_city_pois(data_dir, Path("/tmp/agentmove_data/pois_by_city.pkl"))
    print(f"target-city POIs: {len(pois_all):,}")
    print(f"loading {args.city} check-ins ...")
    df = load_checkins(args.city, pois_all, data_dir)
    print(f"check-ins: {len(df):,} | users: {df['user'].nunique():,} | POIs: {df['venue'].nunique():,}")
    df = build_trajectories(df)
    train, val, items = split_and_build(df)
    tests = sample_tests(items)
    print(f"train rows {len(train):,} | val rows {len(val):,} | test instances {len(tests)}")

    pois = df[["venue", "lat", "lng"]].drop_duplicates("venue").reset_index(drop=True)
    model = Model(train, pois)
    print(evaluate(model, tests))


if __name__ == "__main__":
    main()
