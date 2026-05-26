"""Configuration for unified catalyst artifacts."""

from __future__ import annotations

from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parent
DATA_DIR = MODULE_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

CATALYST_RECORDS_PATH = PROCESSED_DIR / "catalyst_records.parquet"
CATALYST_SCORES_PATH = PROCESSED_DIR / "catalyst_scores.parquet"
CATALYST_FEATURE_MATRIX_PATH = PROCESSED_DIR / "catalyst_feature_matrix.parquet"

CATALYST_FEATURE_COLUMNS = [
    "catalyst_score",
    "news_catalyst_score",
    "scheduled_event_score",
    "event_risk_score",
    "catalyst_count_24h",
    "direct_catalyst_count_24h",
    "hours_since_catalyst",
    "latest_catalyst_relation_confidence",
    "latest_catalyst_is_direct",
]


def ensure_dirs() -> None:
    for path in (DATA_DIR, PROCESSED_DIR):
        path.mkdir(parents=True, exist_ok=True)
