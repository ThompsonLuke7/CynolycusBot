"""Point-in-time scheduled event feature builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from events.config import EVENT_FEATURES_PATH, ensure_data_dirs


FEATURE_COLUMNS = [
    "hours_to_cpi",
    "hours_to_fomc",
    "macro_event_today",
    "macro_event_next_24h",
    "opex_week",
    "earnings_next_7d",
]


def _utc_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _hours_to(timestamp: pd.Timestamp, events: pd.DataFrame, event_types: set[str]) -> float:
    if events.empty:
        return float("nan")
    future = events.loc[(events["event_type"].isin(event_types)) & (events["timestamp"] >= timestamp)]
    if future.empty:
        return float("nan")
    delta = future["timestamp"].min() - timestamp
    return float(delta.total_seconds() / 3600.0)


def _event_today(timestamp: pd.Timestamp, events: pd.DataFrame) -> float:
    if events.empty:
        return 0.0
    local_day = timestamp.tz_convert("America/New_York").date()
    days = events["timestamp"].dt.tz_convert("America/New_York").dt.date
    return float(bool((days == local_day).any()))


def _event_next_hours(timestamp: pd.Timestamp, events: pd.DataFrame, hours: int) -> float:
    if events.empty:
        return 0.0
    end = timestamp + pd.Timedelta(hours=hours)
    return float(bool(events["timestamp"].between(timestamp, end).any()))


def _opex_week(timestamp: pd.Timestamp, events: pd.DataFrame) -> float:
    if events.empty:
        return 0.0
    local = timestamp.tz_convert("America/New_York")
    week_end = local.normalize() + pd.Timedelta(days=7 - local.weekday())
    opex = events.loc[events["event_type"].eq("opex")]
    if opex.empty:
        return 0.0
    opex_local = opex["timestamp"].dt.tz_convert("America/New_York")
    return float(bool(opex_local.between(local.normalize(), week_end).any()))


def _earnings_next_7d(timestamp: pd.Timestamp, ticker: str, earnings: pd.DataFrame) -> float:
    if earnings.empty or not ticker:
        return 0.0
    clean = str(ticker).upper().replace("$", "")
    future = earnings.loc[earnings["ticker"].astype(str).str.upper().eq(clean)]
    if future.empty:
        return 0.0
    return float(bool(future["timestamp"].between(timestamp, timestamp + pd.Timedelta(days=7)).any()))


def build_event_features(
    timestamps: pd.DataFrame,
    macro_events: pd.DataFrame,
    earnings_events: pd.DataFrame | None = None,
    *,
    output_path: Path | str = EVENT_FEATURES_PATH,
) -> pd.DataFrame:
    """Build scheduled event features for rows containing `timestamp` and optional `ticker`."""
    ensure_data_dirs()
    if timestamps.empty:
        out = pd.DataFrame(columns=["timestamp", "ticker", *FEATURE_COLUMNS])
        out.to_parquet(output_path, index=False)
        return out

    base = timestamps.copy()
    base["timestamp"] = _utc_series(base["timestamp"])
    if "ticker" not in base.columns:
        base["ticker"] = ""

    macro = macro_events.copy()
    if not macro.empty:
        macro["timestamp"] = _utc_series(macro["timestamp"])
        macro = macro.dropna(subset=["timestamp"])
    earnings = earnings_events.copy() if earnings_events is not None else pd.DataFrame()
    if not earnings.empty:
        earnings["timestamp"] = _utc_series(earnings["timestamp"])
        earnings = earnings.dropna(subset=["timestamp"])

    rows = []
    for row in base.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        ticker = str(row.ticker).upper().replace("$", "")
        rows.append(
            {
                "timestamp": ts,
                "ticker": ticker,
                "hours_to_cpi": _hours_to(ts, macro, {"cpi"}),
                "hours_to_fomc": _hours_to(ts, macro, {"fomc_decision", "fomc_minutes"}),
                "macro_event_today": _event_today(ts, macro.loc[~macro["event_type"].eq("opex")] if not macro.empty else macro),
                "macro_event_next_24h": _event_next_hours(ts, macro.loc[~macro["event_type"].eq("opex")] if not macro.empty else macro, 24),
                "opex_week": _opex_week(ts, macro),
                "earnings_next_7d": _earnings_next_7d(ts, ticker, earnings),
            }
        )
    out = pd.DataFrame(rows)
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    out.to_parquet(output_path, index=False)
    return out

