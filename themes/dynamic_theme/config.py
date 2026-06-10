"""Central configuration for the dynamic_theme module."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Source data ──────────────────────────────────────────────────────────────
# Raw daily bars per ticker (shared Alpaca cache)
DAILY_BARS_DIR = REPO_ROOT / "Data" / "shared" / "bars" / "1d"

# News records from the news module
NEWS_RECORDS_PATH = REPO_ROOT / "signals" / "news" / "data" / "processed" / "news_records.parquet"

# Yahoo Finance ticker profiles (from news module)
TICKER_PROFILES_PATH = REPO_ROOT / "signals" / "news" / "data" / "processed" / "ticker_profiles.parquet"

# Per-ticker JSON profile cache (from legacy theme_expansion)
TICKER_PROFILES_CACHE_DIR = REPO_ROOT / "themes" / "theme_expansion_legacy" / "data" / "ticker_profiles_cache"

# Forward guidance / events
EVENTS_FG_PATH = REPO_ROOT / "signals" / "events" / "forward_guidance" / "data" / "processed"

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUTS_DIR = REPO_ROOT / "themes" / "dynamic_theme" / "outputs"

TICKER_DOCUMENTS_PATH   = OUTPUTS_DIR / "ticker_documents.parquet"
TICKER_EMBEDDINGS_PATH  = OUTPUTS_DIR / "ticker_embeddings.parquet"
TICKER_CLUSTERS_PATH    = OUTPUTS_DIR / "ticker_clusters.parquet"
THEME_REGISTRY_PATH     = OUTPUTS_DIR / "theme_registry.parquet"
THEME_RELATIONSHIPS_PATH = OUTPUTS_DIR / "theme_relationships.parquet"
TICKER_MEMBERSHIP_PATH  = OUTPUTS_DIR / "ticker_theme_membership.parquet"
TICKER_THEME_FEATURES_PATH = OUTPUTS_DIR / "ticker_theme_features.parquet"

# ── Embedding model ───────────────────────────────────────────────────────────
BGE_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # bge-small-en-v1.5

# ── Clustering ───────────────────────────────────────────────────────────────
HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_MIN_SAMPLES = 2
HDBSCAN_METRIC = "euclidean"  # cosine-normalized embeddings → euclidean OK

# ── Discovery engine ─────────────────────────────────────────────────────────
# Centroid cosine similarity below this → new theme detected
NEW_THEME_SIMILARITY_THRESHOLD = 0.75

# ── News window ───────────────────────────────────────────────────────────────
NEWS_LOOKBACK_DAYS = 30
MAX_HEADLINES_PER_TICKER = 10

# ── Theme heat features ───────────────────────────────────────────────────────
HEAT_WINDOW_DAYS = 5          # short window for heat / breadth
HEAT_PRIOR_WINDOW_DAYS = 10   # prior window for acceleration base
BREADTH_THRESHOLD = 0.0       # return > 0 = constituent is "up"
TOP_Q_STRENGTH = 0.75         # top-quartile cutoff for theme_strength

# ── Claude ────────────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-6"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Max tokens returned per Claude call (labeling / relationship graph)
CLAUDE_MAX_TOKENS = 1024


def ensure_outputs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
