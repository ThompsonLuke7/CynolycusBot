"""Configuration for multi-ticker swing higher-time-frame research."""
from __future__ import annotations

from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODULE_ROOT.parents[0]

DATA_DIR = MODULE_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = MODULE_ROOT / "models"

FEATURES_COMBINED = PROCESSED_DIR / "features_4h.parquet"
LABELS_COMBINED = PROCESSED_DIR / "pivot_swing_labels_4h.parquet"
TRAINING_MATRIX = PROCESSED_DIR / "training_matrix_4h.parquet"

CORRELATED_FEATURE_REPORT = PROCESSED_DIR / "correlated_features_dropped.csv"

TRAIN_START = "2020-01-01"
# End-exclusive date used by the shared fetchers; includes 2026-06-01.
TRAIN_END = "2026-06-02"

PIVOT_LABEL_CONFIG: dict = {
    "pivot_left_bars": 3,
    "pivot_right_bars": 3,
    "label_shift_bars": 1,
    "positive_window_bars": 1,
    "ambiguous_window_bars": 1,
    "forward_min_bars": 13,   # about 5 trading days at 2 4H bars/day
    "forward_max_bars": 38,   # about 15 trading days
    "alpha_benchmark": "SPY",
    "top_quantile": 0.15,
    "composite_weights": {
        "alpha": 0.35,
        "atr_adjusted": 0.25,
        "drawdown": 0.25,
        "persistence": 0.15,
    },
}
