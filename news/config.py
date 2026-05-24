"""Configuration for unscheduled catalyst news features."""

from __future__ import annotations

from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODULE_ROOT.parent

DATA_DIR = MODULE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

NEWS_RECORDS_PATH = PROCESSED_DIR / "news_records.parquet"
NEWS_EMBEDDINGS_PATH = PROCESSED_DIR / "news_embeddings.parquet"
NEWS_LABELS_PATH = PROCESSED_DIR / "news_labels.parquet"
NEWS_FEATURE_MATRIX_PATH = PROCESSED_DIR / "news_feature_matrix.parquet"
WINNER_LIBRARY_PATH = PROCESSED_DIR / "winner_news_library.parquet"
LOSER_LIBRARY_PATH = PROCESSED_DIR / "loser_news_library.parquet"

DEFAULT_BGE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_FINBERT_MODEL = "ProsusAI/finbert"

NEWS_FEATURE_COLUMNS = [
    "news_count_24h",
    "hours_since_news",
    "finbert_positive_score",
    "finbert_negative_score",
    "finbert_neutral_score",
    "news_cluster_id",
    "winner_similarity_max",
    "loser_similarity_max",
    "news_edge_score",
]


def ensure_data_dirs() -> None:
    for path in (DATA_DIR, RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR):
        path.mkdir(parents=True, exist_ok=True)

