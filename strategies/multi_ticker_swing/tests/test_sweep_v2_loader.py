"""Regression test for the sweep_v2 raw-bar loader schema fix.

sweep_v2.load_raw_30m/load_raw_5m assumed "timestamp" was always a DataFrame
column. Most raw 30m/5m caches now store it as a DatetimeIndex instead (an
on-disk format change since sweep_v2.py was last touched), which silently
dropped most tickers from every sweep_v2 run via the broad except-Exception
in run_sweep's ticker-build loop (see research/capstone/leakage_audit.md §0.6).
"""
from __future__ import annotations

import pandas as pd
import pytest

from strategies.multi_ticker_swing.backtest.sweep_v2 import _ensure_timestamp_column

pytestmark = pytest.mark.safe


def _ohlcv(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2026-01-02 14:30", periods=n, freq="30min", tz="UTC")
    return pd.DataFrame(
        {"open": range(n), "high": range(n), "low": range(n), "close": range(n), "volume": range(n)},
        index=idx.rename("timestamp"),
    )


def test_index_based_timestamp_normalized_to_column():
    df = _ensure_timestamp_column(_ohlcv())
    assert "timestamp" in df.columns
    assert df["timestamp"].is_monotonic_increasing
    assert len(df) == 5


def test_column_based_timestamp_passthrough():
    raw = _ohlcv().reset_index()
    df = _ensure_timestamp_column(raw)
    assert "timestamp" in df.columns
    assert df["timestamp"].is_monotonic_increasing
    assert len(df) == 5


def test_unnamed_datetime_index_normalized():
    df_raw = _ohlcv()
    df_raw.index.name = None
    df = _ensure_timestamp_column(df_raw)
    assert "timestamp" in df.columns
    assert len(df) == 5
