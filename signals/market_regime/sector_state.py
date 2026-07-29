"""Per-sector-ETF-per-date relative-strength state.

Builds, per docs/superpowers/plans/2026-07-26-market-regime-and-sector-context.md §4 (WS-A):
``excess_21d``, ``excess_63d``, ``rank_21d``, ``rank_63d``,
``rs_accel = excess_21d - excess_63d/3``, ``above_20d``, ``above_50d`` for
each of the 11 sector ETFs on each SPY trading session. All statistics are
trailing/causal (window ends at ``t``); ranks are cross-sectional (same date,
across sector ETFs) so they carry no lookahead either.
"""
from __future__ import annotations

import pandas as pd

from strategies.momentum_expansion.data.load_bars import load_1d

from .config import (
    COMPONENT_WINDOW_LONG,
    COMPONENT_WINDOW_SHORT,
    EXCESS_RETURN_WINDOW,
    EXCESS_RETURN_WINDOW_LONG,
    SECTOR_ETFS_LIST,
    SETTLE_MARGIN_MINUTES,
)
from .timeutil import align_with_staleness, available_at_for_dates, session_date_index


def _load_close(ticker: str, *, loader=load_1d) -> pd.Series:
    df = loader(ticker)
    dates = session_date_index(df)
    close = pd.Series(df["close"].to_numpy(), index=dates, name=ticker)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close


def build_sector_state(
    *,
    loader=load_1d,
    sector_etfs: list[str] = SECTOR_ETFS_LIST,
    excess_return_window: int = EXCESS_RETURN_WINDOW,
    excess_return_window_long: int = EXCESS_RETURN_WINDOW_LONG,
    component_window_short: int = COMPONENT_WINDOW_SHORT,
    component_window_long: int = COMPONENT_WINDOW_LONG,
    settle_margin_minutes: int = SETTLE_MARGIN_MINUTES,
) -> pd.DataFrame:
    """Build the long-format ``(date, sector_etf)`` sector-state table.

    Columns: ``date, sector_etf, available_at, excess_21d, excess_63d,
    rank_21d, rank_63d, rs_accel, above_20d, above_50d, stale_days``.
    ``loader`` defaults to ``load_1d`` and is injectable for tests.
    """
    spy_raw = _load_close("SPY", loader=loader)
    calendar = spy_raw.index
    spy_close, _spy_stale = align_with_staleness(spy_raw, calendar)
    spy_ret_short = spy_close.pct_change(excess_return_window)
    spy_ret_long = spy_close.pct_change(excess_return_window_long)

    available_at = available_at_for_dates(calendar, settle_margin_minutes=settle_margin_minutes)

    excess_short_cols: dict[str, pd.Series] = {}
    excess_long_cols: dict[str, pd.Series] = {}
    per_etf: dict[str, pd.DataFrame] = {}

    for etf in sector_etfs:
        raw = _load_close(etf, loader=loader)
        close, stale = align_with_staleness(raw, calendar)
        excess_21d = close.pct_change(excess_return_window) - spy_ret_short
        excess_63d = close.pct_change(excess_return_window_long) - spy_ret_long
        sma20 = close.rolling(component_window_short, min_periods=component_window_short).mean()
        sma50 = close.rolling(component_window_long, min_periods=component_window_long).mean()
        above_20d = (close > sma20).astype(float).where(close.notna() & sma20.notna())
        above_50d = (close > sma50).astype(float).where(close.notna() & sma50.notna())

        excess_short_cols[etf] = excess_21d
        excess_long_cols[etf] = excess_63d
        per_etf[etf] = pd.DataFrame({
            "excess_21d": excess_21d,
            "excess_63d": excess_63d,
            "above_20d": above_20d,
            "above_50d": above_50d,
            "stale_days": stale,
        })

    excess_short_df = pd.DataFrame(excess_short_cols)
    excess_long_df = pd.DataFrame(excess_long_cols)
    rank_short_df = excess_short_df.rank(axis=1, pct=True)
    rank_long_df = excess_long_df.rank(axis=1, pct=True)

    frames: list[pd.DataFrame] = []
    for etf in sector_etfs:
        block = per_etf[etf].copy()
        block["rank_21d"] = rank_short_df[etf]
        block["rank_63d"] = rank_long_df[etf]
        block["rs_accel"] = block["excess_21d"] - block["excess_63d"] / 3.0
        block["date"] = calendar
        block["available_at"] = available_at
        block["sector_etf"] = etf
        frames.append(block.reset_index(drop=True))

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out[[
        "date", "sector_etf", "available_at",
        "excess_21d", "excess_63d", "rank_21d", "rank_63d", "rs_accel",
        "above_20d", "above_50d", "stale_days",
    ]]
    out = out.sort_values(["date", "sector_etf"]).reset_index(drop=True)
    return out
