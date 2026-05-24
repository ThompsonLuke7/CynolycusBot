"""Configuration for context meta-model artifacts."""

from __future__ import annotations

from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parent
DATA_DIR = MODULE_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = MODULE_ROOT / "models"

META_TRAINING_MATRIX_PATH = PROCESSED_DIR / "meta_training_matrix.parquet"
META_MODEL_PATH = MODELS_DIR / "meta_xgb_model.json"
CONTEXT_BACKTEST_UNIVERSE_PATH = PROCESSED_DIR / "context_backtest_universe.csv"
CONTEXT_BACKTEST_TIMESTAMPS_PATH = PROCESSED_DIR / "context_backtest_timestamps.parquet"
CONTEXT_FORWARD_LABELS_PATH = PROCESSED_DIR / "context_forward_labels.parquet"

META_FEATURE_COLUMNS = [
    "ta_expansion_prob",
    "news_catalyst_score",
    "social_velocity_score",
    "event_risk_score",
    "theme_rotation_score",
    "leader_follower_score",
]


def ensure_dirs() -> None:
    for path in (DATA_DIR, PROCESSED_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
