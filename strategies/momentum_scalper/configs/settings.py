"""Central configuration for the momentum scalper MVP."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]

DATA_DIR = PACKAGE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MINUTE_BARS_DIR = DATA_DIR / "minute_bars"
NEWS_DIR = DATA_DIR / "news"
SCANNER_SNAPSHOTS_DIR = DATA_DIR / "scanner_snapshots"
HALTS_DIR = DATA_DIR / "halts"
METADATA_DIR = DATA_DIR / "metadata"
MODELS_DIR = PACKAGE_ROOT / "models"
MODEL_ARTIFACTS_DIR = MODELS_DIR / "artifacts"
PLOTS_OUTPUT_DIR = PACKAGE_ROOT / "plots" / "output"

ALL_EQUITIES_PATH = RAW_DIR / "all_equities.parquet"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
LABELS_PATH = PROCESSED_DIR / "labels.parquet"
TRAINING_MATRIX_PATH = PROCESSED_DIR / "training_matrix.parquet"

POLYGON_API_KEY_ENV = "POLYGON_API_KEY"


@dataclass(frozen=True)
class ScannerConfig:
    gap_pct_min: float = 10.0
    premarket_volume_min: int = 250_000
    relative_volume_min: float = 5.0
    min_price: float = 1.0
    max_price: float = 20.0
    max_float: float = 100_000_000.0
    require_news: bool = False


@dataclass(frozen=True)
class EntryConfig:
    max_chase_pct_above_trigger: float = 2.0
    max_spread_pct: float = 1.5
    min_liquidity_score: float = 0.15
    halt_cooldown_minutes: int = 30
    opening_range_minutes: int = 5


@dataclass(frozen=True)
class ExitConfig:
    reward_risk: float = 2.0
    stop_risk: float = 1.0
    atr_trail_mult: float = 1.5
    max_hold_minutes: int = 15
    volume_fade_ratio: float = 0.35


def polygon_api_key() -> str:
    return os.getenv(POLYGON_API_KEY_ENV, "").strip()


def ensure_data_dirs() -> None:
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        MINUTE_BARS_DIR,
        NEWS_DIR,
        SCANNER_SNAPSHOTS_DIR,
        HALTS_DIR,
        METADATA_DIR,
        MODEL_ARTIFACTS_DIR,
        PLOTS_OUTPUT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
