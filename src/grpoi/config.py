"""Global configuration for the geo-rag-poi project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# RALLM-POI stays pristine as the external baseline; we only read its data.
# config.py lives at <workspace>/geo-rag-poi/src/grpoi/config.py
WORKSPACE = Path(
    os.environ.get("GRPOI_WORKSPACE", Path(__file__).resolve().parents[3])
)
RALLM_DIR = WORKSPACE / "RALLM-POI"
PROCESSED_DIR = RALLM_DIR / "processed_data" / "original"
RAW_DIR = RALLM_DIR / "raw_data"
RAG_DIR = RALLM_DIR / "experiments_rag"

CITIES = ("PHO", "NYC", "SIN")


def _load_env() -> None:
    """Load DMX_API_KEY / DMX_BASE_URL from the private env file if present."""
    env_file = WORKSPACE / ".tools" / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()


@dataclass
class RetrievalConfig:
    """Multi-route retrieval hyper-parameters."""

    rag_top_k: int = 50          # semantic route: top-k similar training trajectories
    graph_top_k: int = 20        # transition-graph route: top-k successors
    geo_top_k: int = 20          # geo route: top-k nearby POIs
    memory_top_k: int = 15       # individual-memory route: top-k revisited POIs
    geo_radius_km: float = 5.0   # neighbourhood radius around the last POI
    geo_scale_km: float = 2.5    # distance-decay scale
    rrf_k: int = 60              # reciprocal-rank-fusion constant
    #: The temporal transition model is by far the strongest single channel on
    #: these splits; the other routes act as tie-breakers and for cold start.
    weights: dict = field(
        default_factory=lambda: {
            "graph": 20.0,
            "geo": 0.3,
            "memory": 0.3,
            "semantic": 0.3,
        }
    )
    n_candidates: int = 25       # candidates handed to the LLM
    fusion: str = "rrf"          # rrf | minmax | softmax | union


@dataclass
class AgentConfig:
    """Adaptive-verification hyper-parameters."""

    model: str = os.getenv("GRPOI_MODEL", "gpt-4o-mini")
    temperature: float = 0.0
    max_tokens: int = 900
    timeout: float = 90.0
    max_retries: int = 3
    #: weight of the retrieval prior in the final blend (1.0 = pure retrieval)
    llm_blend_alpha: float = 0.75
    #: trigger the critique pass when the LLM self-reports confidence below this
    confidence_threshold: float = 0.55
    #: trigger when the LLM output fails validation (dupes/invalid ids)
    always_verify_on_invalid: bool = True


@dataclass
class EvalConfig:
    cities: tuple = CITIES
    ks: tuple = (5, 10)
    #: a predicted jump larger than this is counted as a spatial violation
    jump_km: float = 50.0
    #: P(category | hour) below this is treated as an implausible category-hour pair
    cat_hour_min_prob: float = 0.02


RETRIEVAL = RetrievalConfig()
AGENT = AgentConfig()
EVAL = EvalConfig()

# Approximate USD per 1M tokens (input, output); unknown models are reported as 0.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
}


def client_kwargs() -> dict:
    return {
        "api_key": os.environ.get("DMX_API_KEY", ""),
        "base_url": os.environ.get("DMX_BASE_URL", "https://www.dmxapi.com/v1"),
        "timeout": AGENT.timeout,
        "max_retries": 0,  # we retry ourselves to keep accounting exact
    }
