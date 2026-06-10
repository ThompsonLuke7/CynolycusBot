"""Configuration for scheduled event context features."""

from __future__ import annotations

from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODULE_ROOT.parents[1]

DATA_DIR = MODULE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MACRO_EVENTS_PATH = PROCESSED_DIR / "macro_events.parquet"
EARNINGS_EVENTS_PATH = PROCESSED_DIR / "earnings_dates.parquet"
EVENT_FEATURES_PATH = PROCESSED_DIR / "event_features.parquet"

DEFAULT_TIMEZONE = "America/New_York"

ALLOWED_MACRO_EVENT_TYPES = {
    "cpi",
    "fomc_decision",
    "fomc_minutes",
    "fed_speech",
    "nfp",
    "jobs",
    "ppi",
    "gdp",
    "opex",
}

DISALLOWED_EVENT_TYPES = {"treasury_auction", "treasury_auctions"}


def ensure_data_dirs() -> None:
    for path in (DATA_DIR, RAW_DIR, PROCESSED_DIR):
        path.mkdir(parents=True, exist_ok=True)

