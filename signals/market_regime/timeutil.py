"""Shared time-correctness helpers for signals.market_regime.

Every helper here is causal: rolling windows end at ``t`` (never centered),
and z-scores require an explicit ``min_periods`` with NaN — never a
zero-fill — before that point. See the time-correctness contract in
docs/superpowers/plans/2026-07-26-market-regime-and-sector-context.md §3.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MARKET_CLOSE_ET


def session_date_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the tz-naive ET calendar (session) date for each row of a
    UTC-tz-aware-indexed daily bar frame (``load_1d`` output).

    Daily bars are stamped at ET midnight (04:00 or 05:00 UTC depending on
    DST), so converting to America/New_York and normalizing recovers the
    exact session date with no off-by-one risk across DST transitions.
    """
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert("America/New_York").normalize().tz_localize(None)


def available_at_for_dates(
    dates: pd.DatetimeIndex,
    *,
    settle_margin_minutes: int,
    market_close_et: str = MARKET_CLOSE_ET,
) -> pd.DatetimeIndex:
    """UTC timestamp at which each session's daily bar becomes knowable:
    that session's ET close plus a settle margin, converted to UTC.

    ``dates`` must be tz-naive calendar dates (as returned by
    ``session_date_index``). Uses wall-clock localization to America/New_York
    so the UTC offset is correct across DST transitions automatically.
    """
    naive_dates = pd.DatetimeIndex(dates)
    if naive_dates.tz is not None:
        naive_dates = naive_dates.tz_convert("UTC").tz_localize(None)
    close_naive = naive_dates + pd.Timedelta(market_close_et)
    close_et = close_naive.tz_localize("America/New_York")
    return (close_et + pd.Timedelta(minutes=settle_margin_minutes)).tz_convert("UTC")


def trailing_zscore(series: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    """Trailing (never centered) z-score over a rolling window.

    Rows before ``min_periods`` valid trailing observations are NaN, never
    zero-filled, per the time-correctness contract.
    """
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def align_with_staleness(
    raw: pd.Series, calendar: pd.DatetimeIndex
) -> tuple[pd.Series, pd.Series]:
    """Reindex ``raw`` (indexed by session date) onto ``calendar``.

    Forward-fills genuine mid-series gaps (a missing session after the
    series has already started trading) and returns
    ``(filled_values, stale_days)``: ``stale_days`` counts consecutive
    sessions since the last real (non-forward-filled) observation, 0 on a
    real observation. Dates before the series' first valid observation stay
    NaN in both outputs — that is short history, not staleness (there is
    nothing yet to be stale from), per plan §3's "short-history rows are NaN,
    not zero" requirement.
    """
    aligned = raw.reindex(calendar)
    first_valid = aligned.first_valid_index()
    if first_valid is None:
        nan_series = pd.Series(np.nan, index=calendar)
        return nan_series, nan_series.copy()

    is_real = aligned.notna().to_numpy()
    filled = aligned.ffill()
    filled = filled.where(calendar >= first_valid, np.nan)

    stale = np.full(len(calendar), np.nan)
    counter = 0
    started = False
    for i, ts in enumerate(calendar):
        if ts < first_valid:
            continue
        started = True
        if is_real[i]:
            counter = 0
        else:
            counter += 1
        stale[i] = float(counter)
    assert started or len(calendar) == 0
    return filled, pd.Series(stale, index=calendar)


def atomic_write_parquet(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Write via temp file + atomic rename, matching the pattern used by
    strategies/intraday_structure/state_store.py and
    strategies/momentum_expansion/data/bars.py, so a concurrent reader never
    sees a torn/partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    df.to_parquet(tmp, **kwargs)
    tmp.replace(path)
