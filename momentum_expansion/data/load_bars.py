"""Readers for cached momentum_expansion bar files."""
from __future__ import annotations

import pandas as pd

from momentum_expansion.config.momentum_config import (
    RAW_1D_DIR,
    RAW_1H_DIR,
    RAW_4H_DIR,
    RAW_CONTEXT_DIR,
)


def _read(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


def load_1h(ticker: str) -> pd.DataFrame:
    return _read(RAW_1H_DIR / f"{ticker}.parquet")


def load_4h(ticker: str) -> pd.DataFrame:
    p4 = RAW_4H_DIR / f"{ticker}.parquet"
    if p4.exists():
        return _read(p4)
    # Context tickers store 4H in raw/4h too once context fetcher runs.
    raise FileNotFoundError(f"No 4h cache for {ticker}")


def load_1d(ticker: str) -> pd.DataFrame:
    return _read(RAW_1D_DIR / f"{ticker}.parquet")


def load_context_1h(ticker: str) -> pd.DataFrame:
    return _read(RAW_CONTEXT_DIR / f"{ticker}.parquet")
