"""Configuration for the Reddit social-attention pipeline."""

from __future__ import annotations

import os
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODULE_ROOT.parent

DATA_DIR = MODULE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
MODELS_DIR = MODULE_ROOT / "models"

REDDIT_SUBMISSIONS_PATH = RAW_DIR / "reddit_submissions.parquet"
REDDIT_COMMENTS_PATH = RAW_DIR / "reddit_comments.parquet"
REDDIT_POSTS_PATH = PROCESSED_DIR / "reddit_posts.parquet"
REDDIT_MENTIONS_PATH = PROCESSED_DIR / "reddit_mentions.parquet"
ATTENTION_FEATURES_PATH = PROCESSED_DIR / "attention_features.parquet"
SOCIAL_EMBEDDINGS_PATH = PROCESSED_DIR / "social_embeddings.parquet"
NARRATIVE_CLUSTERS_PATH = PROCESSED_DIR / "narrative_clusters.parquet"
NARRATIVE_FEATURES_PATH = PROCESSED_DIR / "narrative_features.parquet"
SOCIAL_LABELS_PATH = PROCESSED_DIR / "social_labels.parquet"
TRAINING_MATRIX_PATH = PROCESSED_DIR / "training_matrix.parquet"

MODEL_PATH = MODELS_DIR / "social_attention_lgbm.txt"
MODEL_MANIFEST_PATH = MODELS_DIR / "social_attention_manifest.json"
METRICS_PATH = MODELS_DIR / "social_attention_metrics.json"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "social_attention_feature_importance.csv"

DEFAULT_PULLPUSH_BASE_URL = os.getenv("PULLPUSH_BASE_URL", "https://api.pullpush.io")
DEFAULT_SUBREDDITS = (
    "stocks",
    "wallstreetbets",
    "smallstreetbets",
    "options",
    "investing",
    "pennystocks",
    "SPACs",
)
DEFAULT_START = "2023-01-01"

REDDIT_CLIENT_ID_ENV = "REDDIT_CLIENT_ID"
REDDIT_CLIENT_SECRET_ENV = "REDDIT_CLIENT_SECRET"
REDDIT_USER_AGENT_ENV = "REDDIT_USER_AGENT"

AMBIGUOUS_BARE_TICKERS = {
    "A",
    "AA",
    "AI",
    "ALL",
    "AM",
    "ARE",
    "AT",
    "BE",
    "BIG",
    "BY",
    "CAN",
    "DD",
    "DO",
    "FOR",
    "GO",
    "HAS",
    "HE",
    "I",
    "IN",
    "IT",
    "KEY",
    "LOW",
    "ME",
    "NEW",
    "NOW",
    "ON",
    "OPEN",
    "OR",
    "OUT",
    "PLAY",
    "REAL",
    "SO",
    "STAY",
    "T",
    "TO",
    "U",
    "UP",
    "USA",
    "VERY",
    "WE",
    "WELL",
    "YOU",
}

LIGHTGBM_CONFIG = {
    "n_estimators": 600,
    "learning_rate": 0.035,
    "num_leaves": 31,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "objective": "binary",
    "random_state": 42,
    "n_jobs": -1,
}


def ensure_dirs() -> None:
    for path in (DATA_DIR, RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)

