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
NEWS_SCORES_PATH = PROCESSED_DIR / "news_scores.parquet"
NEWS_FEATURE_MATRIX_PATH = PROCESSED_DIR / "news_feature_matrix.parquet"
WINNER_LIBRARY_PATH = PROCESSED_DIR / "winner_news_library.parquet"
LOSER_LIBRARY_PATH = PROCESSED_DIR / "loser_news_library.parquet"

# Per-ticker reference + per-day market data parquets
TICKER_PROFILE_PATH = PROCESSED_DIR / "ticker_profiles.parquet"
FINRA_SHORT_VOLUME_PATH = PROCESSED_DIR / "finra_short_volume.parquet"
CBOE_OPTIONS_SUMMARY_PATH = PROCESSED_DIR / "cboe_options_summary.parquet"
EARNINGS_CALENDAR_PATH = PROCESSED_DIR / "ticker_earnings_calendar.parquet"
ECONOMIC_CALENDAR_PATH = PROCESSED_DIR / "economic_calendar.parquet"
NASDAQ_SHORT_INTEREST_PATH = PROCESSED_DIR / "nasdaq_short_interest.parquet"
USASPENDING_CONTRACTS_PATH = PROCESSED_DIR / "usaspending_contracts.parquet"

DEFAULT_BGE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_FINBERT_MODEL = "ProsusAI/finbert"

LABEL_HORIZON_DAYS = 10
LABEL_HORIZON_BARS_PER_DAY = 13

SEC_ALPHA_FORMS = [
    "8-K",
    "10-Q",
    "10-K",
    "6-K",
    "S-1",
    "F-1",
    "S-3",
    "S-4",
    "424B3",
    "424B4",
    "424B5",
    "SC 13D",
    "SC 13D/A",
    "SC TO-I",
    "SC TO-T",
    "DEFM14A",
    "PREM14A",
    "4",       # insider transactions — open market purchases are strong signal
    "4/A",     # amended Form 4
]

SEC_LOW_SIGNAL_FORMS = [
    "SC 13G",
    "SC 13G/A",
]

SEC_ADMIN_DESCRIPTION_PATTERNS = [
    "total voting rights",
    "holding(s) in company",
    "block listing",
    "director/pdmr shareholding",
    "notification of major holdings",
    "monthly return",
]

NEWS_FEATURE_COLUMNS = [
    "news_count_24h",
    "direct_news_count_24h",
    "hours_since_news",
    "finbert_positive_score",
    "finbert_negative_score",
    "finbert_neutral_score",
    "news_cluster_id",
    "news_relation_confidence",
    "news_is_direct_catalyst",
    "news_similarity_score",
    "news_similarity_neighbor_count",
    "news_similarity_max",
    "winner_similarity_max",
    "loser_similarity_max",
    "news_edge_score",
    "earnings_catalyst_count_24h",
    "earnings_language_score",
    "earnings_guidance_score",
    "earnings_beat_miss_score",
    "earnings_relevance_score",
]


def ensure_data_dirs() -> None:
    for path in (DATA_DIR, RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
