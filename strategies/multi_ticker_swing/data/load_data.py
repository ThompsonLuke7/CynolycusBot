"""
Loaders for raw and processed parquet files.
All functions return DataFrames with a tz-aware UTC DatetimeIndex.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from strategies.multi_ticker_swing.config.pipeline_config import (
    RAW_30M_DIR,
    RAW_5M_DIR,
    RAW_10M_DIR,
    FEATURES_COMBINED,
    LABELS_COMBINED,
    TRAINING_MATRIX,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw bar loaders
# ---------------------------------------------------------------------------

def load_raw_30m(ticker: str, base_dir: Path = RAW_30M_DIR) -> pd.DataFrame:
    path = base_dir / f"{ticker}.parquet"
    return _load(path, ticker)


def load_raw_5m(ticker: str, base_dir: Path = RAW_5M_DIR) -> pd.DataFrame:
    path = base_dir / f"{ticker}.parquet"
    return _load(path, ticker)


def load_raw_10m(ticker: str, base_dir: Path = RAW_10M_DIR) -> pd.DataFrame:
    path = base_dir / f"{ticker}.parquet"
    return _load(path, ticker)


def _load(path: Path, ticker: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"[{ticker}] raw data not found at {path}. Run --fetch first."
        )
    df = pd.read_parquet(path)
    df = _ensure_utc_index(df)
    return df


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee DatetimeIndex is UTC and sorted."""
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to find a timestamp-like column
        for col in ("timestamp", "t", "time"):
            if col in df.columns:
                df = df.set_index(col)
                break
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()


# ---------------------------------------------------------------------------
# Processed loaders
# ---------------------------------------------------------------------------

def load_features(path: Path = FEATURES_COMBINED) -> pd.DataFrame:
    """Load combined features parquet with (ticker, timestamp) MultiIndex."""
    if not path.exists():
        raise FileNotFoundError(
            f"Features not found at {path}. Run --build-features first."
        )
    return pd.read_parquet(path)


def load_labels(path: Path = LABELS_COMBINED) -> pd.DataFrame:
    """Load combined labels parquet with (ticker, timestamp) MultiIndex."""
    if not path.exists():
        raise FileNotFoundError(
            f"Labels not found at {path}. Run --build-labels first."
        )
    return pd.read_parquet(path)


def load_training_matrix(path: Path = TRAINING_MATRIX) -> pd.DataFrame:
    """Load final training matrix (features + labels, NaN-dropped)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Training matrix not found at {path}. Run --build-matrix first."
        )
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Per-ticker processed loaders (intermediate files)
# ---------------------------------------------------------------------------

def load_ticker_features(ticker: str, processed_30m_dir: Path) -> pd.DataFrame | None:
    path = processed_30m_dir / f"{ticker}_features.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_ticker_labels(ticker: str, processed_lbl_dir: Path) -> pd.DataFrame | None:
    path = processed_lbl_dir / f"{ticker}_labels.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)
