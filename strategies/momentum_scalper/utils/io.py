"""Parquet and market-time helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


NY_TZ = "America/New_York"


def clean_ticker(value: object) -> str:
    return str(value).strip().upper().replace("$", "")


def read_parquet_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def normalize_timestamp_column(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    out = df.copy()
    out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")
    out = out.dropna(subset=[column])
    return out.sort_values(column).reset_index(drop=True)


def trading_day(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize(None).normalize()


def month_starts(start: str, end: str) -> Iterable[pd.Timestamp]:
    first = pd.Timestamp(start).to_period("M").to_timestamp()
    last = pd.Timestamp(end).to_period("M").to_timestamp()
    for month in pd.date_range(first, last, freq="MS"):
        yield month


def add_session_columns(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    out = normalize_timestamp_column(df, ts_col)
    if out.empty:
        return out
    local = out[ts_col].dt.tz_convert(NY_TZ)
    out["date"] = local.dt.date.astype(str)
    out["minute_of_day"] = local.dt.hour * 60 + local.dt.minute
    out["is_premarket"] = out["minute_of_day"].between(4 * 60, 9 * 60 + 29)
    out["is_rth"] = out["minute_of_day"].between(9 * 60 + 30, 16 * 60)
    return out
