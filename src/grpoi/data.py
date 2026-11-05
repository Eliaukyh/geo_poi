"""Data adapter: turn RALLM-POI pickles into a clean, modelling-friendly form.

RALLM-POI stores each sample as a list of days; every day is six parallel arrays
    [POI_id, category_id, user_id, hour, flag, day_number]
so a sample is a ragged (days x check-ins) trajectory.  We flatten it into
    Sample(history_pois, recent_pois, target)
where `recent_pois` are the check-ins of the final day (all but the last) and
`target` is the POI to predict -- exactly the convention used by RALLM-POI.
"""

from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


@dataclass
class Sample:
    city: str
    user: int
    pois: list[int]
    cats: list[int]
    hours: list[int]
    target: int
    recent_len: int
    days: list[int] = field(default_factory=list)   # absolute day index per check-in

    @property
    def history_pois(self) -> list[int]:
        return self.pois[:-1]

    @property
    def recent_pois(self) -> list[int]:
        return self.pois[-self.recent_len - 1 : -1]

    @property
    def recent_cats(self) -> list[int]:
        return self.cats[-self.recent_len - 1 : -1]

    @property
    def last_poi(self) -> int:
        return self.pois[-2]

    @property
    def last_hour(self) -> int:
        return self.hours[-2]

    @property
    def target_cat(self) -> int:
        return self.cats[-1]


def _parse(sample, city: str) -> Sample | None:
    poi_seqs, cat_seqs, hour_seqs, day_seqs = [], [], [], []
    for day in sample:
        if len(day) < 6:
            return None
        poi_seqs.append(np.asarray(day[0], dtype=np.int64))
        cat_seqs.append(np.asarray(day[1], dtype=np.int64))
        hour_seqs.append(np.asarray(day[3], dtype=np.int64))
        day_seqs.append(np.asarray(day[5], dtype=np.int64))
    pois = np.concatenate(poi_seqs)
    cats = np.concatenate(cat_seqs)
    hours = np.concatenate(hour_seqs)
    if len(pois) < 2:
        return None
    user = int(np.asarray(sample[-1][2]).ravel()[0])
    return Sample(
        city=city,
        user=user,
        pois=pois.tolist(),
        cats=cats.tolist(),
        hours=hours.tolist(),
        target=int(pois[-1]),
        recent_len=len(poi_seqs[-1]) - 1,
        days=np.concatenate(day_seqs).tolist(),
    )


@dataclass
class CityData:
    city: str
    coords: np.ndarray                 # (N, 2) float32  [lat, lon]
    cat_ids: np.ndarray                # (N,) int32
    cat_names: list[str]
    regions: np.ndarray                # (N,) int32, -1 when missing
    train: list[Sample]
    valid: list[Sample]
    test: list[Sample]
    user_hist: dict[int, list[tuple[int, int, int]]]
    transitions: dict[int, Counter]
    step_km: np.ndarray                # consecutive-check-in distances in train
    cat_hour_prob: np.ndarray          # (n_cats, 24) P(category | hour)

    # ---------------------------------------------------------------- helpers
    def poi_cat(self, poi: int) -> int:
        return int(self.cat_ids[poi]) if 0 <= poi < len(self.cat_ids) else -1

    def cat_name(self, cat: int) -> str:
        return self.cat_names[cat] if 0 <= cat < len(self.cat_names) else "unknown"

    def poi_coord(self, poi: int) -> tuple[float, float]:
        lat, lon = self.coords[poi]
        return float(lat), float(lon)

    def n_pois(self) -> int:
        return len(self.coords)


def _build_poi_table(city: str):
    raw = config.RAW_DIR / f"{city}_poi_full_info.csv"
    mapping = pd.read_csv(config.RAW_DIR / f"{city}_poi_mapping.csv")
    df = pd.read_csv(raw)
    df = df.drop_duplicates("index").set_index("index")
    mapping = mapping.set_index(mapping["Unnamed: 0"])

    n = int(max(df.index.max(), mapping.index.max())) + 1
    coords = np.full((n, 2), np.nan, dtype=np.float64)
    coords[mapping.index.values, 1] = mapping["longitude"].values
    coords[mapping.index.values, 0] = mapping["latitude"].values

    meta = pickle.load(open(config.PROCESSED_DIR / f"{city}_meta", "rb"))
    cat_names = [str(c) for c in meta["cat"]]
    name_to_id = {name: i for i, name in enumerate(cat_names)}

    cat_ids = np.full(n, -1, dtype=np.int32)
    for idx, name in df["L1_Category"].items():
        cat_ids[int(idx)] = name_to_id.get(str(name), -1)

    regions = np.full(n, -1, dtype=np.int32)
    if "RegionId" in df.columns:
        regions[df.index.values] = (
            pd.to_numeric(df["RegionId"], errors="coerce").fillna(-1).astype(int).values
        )
    return coords, cat_ids, cat_names, regions


def load_city(city: str) -> CityData:
    coords, cat_ids, cat_names, regions = _build_poi_table(city)

    splits = {}
    for name in ("train", "valid", "test"):
        raw = pickle.load(open(config.PROCESSED_DIR / f"{city}_{name}", "rb"))
        parsed = [s for s in (_parse(x, city) for x in raw) if s is not None]
        splits[name] = parsed

    train = splits["train"]
    user_hist: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    transitions: dict[int, Counter] = defaultdict(Counter)
    steps: list[float] = []

    for s in train:
        for p, c, h in zip(s.pois, s.cats, s.hours):
            user_hist[s.user].append((int(p), int(c), int(h)))
        # consecutive transitions inside each day
        for day_pois in _day_slices(s):
            for a, b in zip(day_pois[:-1], day_pois[1:]):
                transitions[int(a)][int(b)] += 1
                steps.append(_dist(coords, int(a), int(b)))

    steps_arr = np.asarray([d for d in steps if np.isfinite(d)], dtype=np.float64)

    cat_hour = np.zeros((len(cat_names), 24), dtype=np.float64)
    for s in train:
        for c, h in zip(s.cats, s.hours):
            cat_hour[int(c) % len(cat_names), int(h) % 24] += 1
    # Laplace smoothing then normalise per hour
    cat_hour = cat_hour + 0.5
    cat_hour /= cat_hour.sum(axis=0, keepdims=True)

    return CityData(
        city=city,
        coords=coords,
        cat_ids=cat_ids,
        cat_names=cat_names,
        regions=regions,
        train=train,
        valid=splits["valid"],
        test=splits["test"],
        user_hist=dict(user_hist),
        transitions=dict(transitions),
        step_km=steps_arr,
        cat_hour_prob=cat_hour,
    )


def _day_slices(sample: Sample):
    """Recover per-day POI sequences from the flattened sample."""
    # recent block length is known; everything before it is history days.
    total = len(sample.pois)
    yield sample.pois[: total - sample.recent_len - 1]
    yield sample.pois[total - sample.recent_len - 1 :]


def _dist(coords: np.ndarray, a: int, b: int) -> float:
    if not (0 <= a < len(coords) and 0 <= b < len(coords)):
        return float("nan")
    return float(haversine_scalar(coords[a], coords[b]))


def haversine_scalar(p1, p2) -> float:
    lat1, lon1 = np.radians(p1[0]), np.radians(p1[1])
    lat2, lon2 = np.radians(p2[0]), np.radians(p2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def haversine_vec(coord, coords: np.ndarray) -> np.ndarray:
    """Vectorised distance (km) from one [lat, lon] to an (N, 2) array."""
    lat1, lon1 = np.radians(coord[0]), np.radians(coord[1])
    lat2 = np.radians(coords[:, 0])
    lon2 = np.radians(coords[:, 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
