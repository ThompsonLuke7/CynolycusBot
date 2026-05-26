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

    def _clean_times(frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.asarray([], dtype="datetime64[ns]")
        return (
            frame["timestamp"]
            .sort_values()
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .to_numpy(dtype="datetime64[ns]")
        )

    def _clean_ts(value: pd.Timestamp) -> np.datetime64:
        return pd.Timestamp(value).tz_convert("UTC").tz_localize(None).to_datetime64()

    def _next_hours(timestamp: pd.Timestamp, times: np.ndarray) -> float:
        if len(times) == 0:
            return float("nan")
        ts64 = _clean_ts(timestamp)
        idx = int(np.searchsorted(times, ts64, side="left"))
        if idx >= len(times):
            return float("nan")
        delta_hours = (times[idx] - ts64) / np.timedelta64(1, "h")
        return float(delta_hours)

    cpi_times = _clean_times(macro.loc[macro["event_type"].eq("cpi")]) if not macro.empty else np.asarray([], dtype="datetime64[ns]")
    fomc_times = _clean_times(macro.loc[macro["event_type"].isin({"fomc_decision", "fomc_minutes"})]) if not macro.empty else np.asarray([], dtype="datetime64[ns]")
    non_opex = macro.loc[~macro["event_type"].eq("opex")].copy() if not macro.empty else pd.DataFrame()
    non_opex_times = _clean_times(non_opex)
    non_opex_days = set(non_opex["timestamp"].dt.tz_convert("America/New_York").dt.date) if not non_opex.empty else set()
    opex = macro.loc[macro["event_type"].eq("opex")].copy() if not macro.empty else pd.DataFrame()
    opex_days = set(opex["timestamp"].dt.tz_convert("America/New_York").dt.date) if not opex.empty else set()
    earnings_by_ticker: dict[str, np.ndarray] = {}
    if not earnings.empty:
        earnings["ticker"] = earnings["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
        for ticker, ticker_earnings in earnings.groupby("ticker", sort=False):
            earnings_by_ticker[str(ticker)] = _clean_times(ticker_earnings)

    rows = []
    for row in base.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        ticker = str(row.ticker).upper().replace("$", "")
        ts64 = _clean_ts(ts)
        local = ts.tz_convert("America/New_York")
        local_day = local.date()
        next_24h = ts64 + np.timedelta64(24, "h")
        earnings_times = earnings_by_ticker.get(ticker, np.asarray([], dtype="datetime64[ns]"))
        earnings_idx = int(np.searchsorted(earnings_times, ts64, side="left")) if len(earnings_times) else 0
        has_earnings_next_7d = bool(
            len(earnings_times)
            and earnings_idx < len(earnings_times)
            and earnings_times[earnings_idx] <= ts64 + np.timedelta64(7, "D")
        )
        week_end = (local.normalize() + pd.Timedelta(days=7 - local.weekday())).date()
        rows.append(
            {
                "timestamp": ts,
                "ticker": ticker,
                "hours_to_cpi": _next_hours(ts, cpi_times),
                "hours_to_fomc": _next_hours(ts, fomc_times),
                "macro_event_today": float(local_day in non_opex_days),
                "macro_event_next_24h": float(bool(len(non_opex_times) and int(np.searchsorted(non_opex_times, ts64, side="left")) < len(non_opex_times) and non_opex_times[int(np.searchsorted(non_opex_times, ts64, side="left"))] <= next_24h)),
                "opex_week": float(any(local_day <= day <= week_end for day in opex_days)),
                "earnings_next_7d": float(has_earnings_next_7d),
            }
        )
    out = pd.DataFrame(rows)
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    out.to_parquet(output_path, index=False)
    return out
