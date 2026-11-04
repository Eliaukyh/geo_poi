"""Geographic / temporal tools.

The tool layer exists so the LLM never has to do spatial arithmetic itself:
distance, travel time and category-hour plausibility are computed here and
handed to the model as facts.  (The Foursquare splits carry no opening-hours
field, so opening status is replaced by an empirical category-hour prior.)
"""

from __future__ import annotations

from .data import CityData, haversine_scalar

#: detour factor applied to great-circle distance when estimating travel time
DETOUR = 1.3
WALK_SPEED_KMH = 4.5
TRANSIT_SPEED_KMH = 22.0
WALK_MAX_KM = 1.5


def travel_time_min(km: float) -> float:
    """Rough door-to-door travel time in minutes."""
    if km <= 0:
        return 0.0
    if km <= WALK_MAX_KM:
        return DETOUR * km / WALK_SPEED_KMH * 60.0
    return DETOUR * km / TRANSIT_SPEED_KMH * 60.0


def category_hour_fit(city: CityData, cat: int, hour: int) -> float:
    """P(category | hour) estimated from the training check-ins."""
    if cat < 0:
        return 0.0
    return float(city.cat_hour_prob[cat % city.cat_hour_prob.shape[0], hour % 24])


def build_facts(city: CityData, poi: int, last_poi: int, hour: int) -> dict:
    """Facts about one candidate POI, used in prompts and for hard checks."""
    lat1, lon1 = city.poi_coord(last_poi)
    lat2, lon2 = city.poi_coord(poi)
    km = haversine_scalar((lat1, lon1), (lat2, lon2))
    cat = city.poi_cat(poi)
    return {
        "poi": poi,
        "cat": city.cat_name(cat),
        "dist_km": round(float(km), 2),
        "travel_min": round(travel_time_min(km), 0),
        "cat_hour_p": round(category_hour_fit(city, cat, hour), 3),
    }


def violations(city: CityData, poi: int, last_poi: int, hour: int, cfg) -> list[str]:
    """Hard-constraint violations for a single predicted POI."""
    out = []
    if not (0 <= poi < city.n_pois()):
        return ["unknown_poi"]
    facts = build_facts(city, poi, last_poi, hour)
    if facts["dist_km"] > cfg.jump_km:
        out.append(f"implausible_jump:{facts['dist_km']}km")
    if facts["cat_hour_p"] < cfg.cat_hour_min_prob:
        out.append(f"cat_hour_mismatch:{facts['cat']}@{hour}h(p={facts['cat_hour_p']})")
    return out


def summarize_trajectory(city: CityData, pois: list[int], max_len: int = 8) -> str:
    """Compact, human-readable rendering of recent check-ins for prompts."""
    parts = []
    for p in pois[-max_len:]:
        if 0 <= p < city.n_pois():
            parts.append(f"{p}({city.cat_name(city.poi_cat(p))})")
        else:
            parts.append(f"{p}(?)")
    return " -> ".join(parts)
