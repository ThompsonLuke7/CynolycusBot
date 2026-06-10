"""
Bar download + cache + 1H -> 4H resampler for momentum_expansion.

Wraps the existing Alpaca fetcher (`API.Alpaca_API.market_data.fetch_intraday`)
so credentials, pagination, and adjustment policy match the SPY live system.

Storage layout:
  Data/shared/bars/1h/{TICKER}.parquet   — native 1H pull
  Data/shared/bars/4h/{TICKER}.parquet   — derived from 1H
  Data/shared/bars/1d/{TICKER}.parquet   — native daily pull
  Data/shared/bars/context/{TICKER}.parquet — SPY/QQQ/IWM/VIXY/sector ETFs at 1H

All parquets store UTC tz-aware timestamps in column `timestamp` (or index
fall-through). The 4H resample anchors to NY local 09:30 / 13:30 so each
RTH session yields exactly two 4H bars (the second may be partial).
"""
from __future__ import annotations

import logging
import time as time_mod
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from core.API.Alpaca_API.market_data.fetch_intraday import fetch_intraday

from strategies.momentum_expansion.config.momentum_config import (
    BAR_CONFIG,
    CONTEXT_TICKERS,
    RAW_1D_DIR,
    RAW_1H_DIR,
    RAW_4H_DIR,
    RAW_CONTEXT_DIR,
    SECTOR_ETFS,
    TRAIN_END,
    TRAIN_START,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _path_for(ticker: str, kind: str) -> Path:
    base = {
        "1h":      RAW_1H_DIR,
        "4h":      RAW_4H_DIR,
        "1d":      RAW_1D_DIR,
        "context": RAW_CONTEXT_DIR,
    }[kind]
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{ticker}.parquet"


def _load_cached(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Single-ticker fetcher (incremental, resumable)
# ---------------------------------------------------------------------------

def fetch_one(
    *,
    ticker: str,
    kind: str,
    start: str = TRAIN_START,
    end: str = TRAIN_END,
    force: bool = False,
) -> pd.DataFrame | None:
    """
    Fetch a single ticker at the given timeframe. If a cache exists, only
    new bars are appended (incremental).

    kind: "1h", "1d", or "context" (1H bars for context tickers).
    Returns the on-disk DataFrame after merge, or None on failure.
    """
    timeframe = {
        "1h":      BAR_CONFIG["primary_timeframe"],
        "1d":      BAR_CONFIG["daily_timeframe"],
        "context": BAR_CONFIG["context_timeframe"],
    }[kind]

    out_path = _path_for(ticker, kind)
    cached = None if force else _load_cached(out_path)

    fetch_start = start
    if cached is not None and "timestamp" in cached.columns and len(cached):
        last_ts = cached["timestamp"].max()
        if pd.notna(last_ts):
            fetch_start = (last_ts + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            if pd.Timestamp(fetch_start) >= pd.Timestamp(end, tz="UTC"):
                logger.info("[%s/%s] cache fresh (last=%s) — skip", ticker, kind, last_ts)
                return cached

    try:
        df_new = fetch_intraday(
            ticker=ticker,
            start=fetch_start,
            end=end,
            timeframe=timeframe,
            limit=10_000,
            adjustment=BAR_CONFIG["adjustment"],
            feed=BAR_CONFIG.get("feed", "sip"),
            save_path="",
        )
    except Exception as exc:
        logger.warning("[%s/%s] fetch failed: %s", ticker, kind, exc)
        return cached

    if df_new is None or df_new.empty:
        if cached is not None:
            return cached
        logger.info("[%s/%s] no bars returned and no cache", ticker, kind)
        return None

    if cached is not None and not cached.empty:
        merged = pd.concat([cached, df_new], axis=0, ignore_index=True)
    else:
        merged = df_new
    merged = (
        merged.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
              .sort_values("timestamp")
              .reset_index(drop=True)
    )
    merged.to_parquet(out_path, index=False)
    logger.info("[%s/%s] cached %d bars -> %s", ticker, kind, len(merged), out_path)
    return merged


# ---------------------------------------------------------------------------
# 1H -> 4H RTH-aligned resample
# ---------------------------------------------------------------------------

def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def resample_1h_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1H RTH bars to 4H bars anchored at NY 09:30 and 13:30.

    Rules:
      - Keep RTH only (already filtered upstream by fetch_intraday, but enforced).
      - Each RTH session contributes up to two 4H bars:
          bar A: 09:30 - 13:30 ET  (4 hourly bars)
          bar B: 13:30 - 17:30 ET  (the remaining hourly bars; may be 2-3)
      - Bar timestamp = bar start, in UTC.
    """
    if df_1h is None or df_1h.empty:
        return pd.DataFrame()
    df = _ensure_utc_index(df_1h.copy())
    df.columns = [c.lower() for c in df.columns]

    idx_ny = df.index.tz_convert("America/New_York")
    minutes = idx_ny.hour * 60 + idx_ny.minute

    # Slot index per row: 0 for [09:30, 13:30), 1 for [13:30, 17:30)
    slot = np.where(minutes < 13 * 60 + 30, 0, 1)
    session_date = pd.Series(idx_ny.date, index=df.index).astype("string")

    bucket = session_date + "_" + pd.Series(slot, index=df.index).astype(str)
    df = df.assign(_bucket=bucket.values)

    grouped = df.groupby("_bucket", sort=True)
    agg = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    # First timestamp per bucket = bar start
    first_ts = grouped.apply(lambda g: g.index[0], include_groups=False)
    agg.index = pd.DatetimeIndex(first_ts.values, tz="UTC")
    agg = agg.sort_index()
    agg.index.name = "timestamp"
    return agg


def build_4h_for(ticker: str, *, force: bool = False) -> pd.DataFrame | None:
    """Resample cached 1H bars to 4H and persist."""
    out_path = _path_for(ticker, "4h")
    src = _path_for(ticker, "1h")
    if out_path.exists() and not force:
        try:
            cached_4h = pd.read_parquet(out_path)
            if not src.exists() or cached_4h.empty:
                return cached_4h
            df_1h_head = pd.read_parquet(src, columns=["timestamp"])
            last_1h = pd.to_datetime(df_1h_head["timestamp"], utc=True, errors="coerce").max()
            last_4h = pd.to_datetime(cached_4h["timestamp"], utc=True, errors="coerce").max()
            if pd.notna(last_1h) and pd.notna(last_4h) and last_4h >= last_1h - pd.Timedelta(hours=4):
                return cached_4h
            logger.info("[%s] rebuilding stale 4h cache (last_4h=%s, last_1h=%s)", ticker, last_4h, last_1h)
        except Exception as exc:
            logger.warning("[%s] existing 4h cache unreadable, rebuilding: %s", ticker, exc)

    if not src.exists():
        logger.warning("[%s] no 1h cache to resample to 4h", ticker)
        return None
    df_1h = pd.read_parquet(src)
    df_4h = resample_1h_to_4h(df_1h)
    if df_4h.empty:
        logger.warning("[%s] 1h->4h yielded 0 bars", ticker)
        return None
    df_4h.reset_index().to_parquet(out_path, index=False)
    return df_4h


# ---------------------------------------------------------------------------
# Multi-ticker orchestrators
# ---------------------------------------------------------------------------

def fetch_universe_bars(
    *,
    tickers: Iterable[str],
    fetch_1h: bool = True,
    fetch_1d: bool = True,
    build_4h: bool = True,
    sleep_between: float = 0.05,
    force: bool = False,
) -> dict:
    """
    Pull 1H, 1D bars for a list of tickers, then resample to 4H.

    Returns a status dict {ticker: {"1h": n_bars, "1d": n_bars, "4h": n_bars}}.
    """
    out: dict[str, dict] = {}
    tickers = list(tickers)
    n = len(tickers)
    for i, t in enumerate(tickers, 1):
        st: dict = {"1h": 0, "1d": 0, "4h": 0}
        if fetch_1h:
            df = fetch_one(ticker=t, kind="1h", force=force)
            st["1h"] = 0 if df is None else len(df)
        if fetch_1d:
            df = fetch_one(ticker=t, kind="1d", force=force)
            st["1d"] = 0 if df is None else len(df)
        if build_4h:
            df = build_4h_for(t, force=force)
            st["4h"] = 0 if df is None else len(df)
        out[t] = st
        if i % 20 == 0 or i == n:
            logger.info("(%d/%d) bars cached for %s -> %s", i, n, t, st)
        time_mod.sleep(sleep_between)
    return out


def fetch_context_bars(
    *,
    tickers: Iterable[str] = tuple(CONTEXT_TICKERS) + tuple(SECTOR_ETFS),
    force: bool = False,
) -> dict:
    """Pull 1H bars for context tickers (regime + sector ETFs)."""
    out: dict[str, int] = {}
    for t in tickers:
        df = fetch_one(ticker=t, kind="context", force=force)
        out[t] = 0 if df is None else len(df)
        # Also build 4H for context tickers so feature builder can use them at the
        # same cadence as the names being scored.
        try:
            df_1h = pd.read_parquet(_path_for(t, "context"))
            df_4h = resample_1h_to_4h(df_1h)
            if not df_4h.empty:
                df_4h.reset_index().to_parquet(_path_for(t, "4h"), index=False)
        except Exception as exc:
            logger.warning("[%s] context 4h build failed: %s", t, exc)
    logger.info("context bars cached: %s", out)
    return out


def fetch_daily_for_universe_scoring(
    *,
    tickers: Iterable[str],
    force: bool = False,
) -> dict:
    """Pull 1D only — used by the universe selector before any intraday cost."""
    out: dict[str, int] = {}
    for t in tickers:
        df = fetch_one(ticker=t, kind="1d", force=force)
        out[t] = 0 if df is None else len(df)
    logger.info("daily bars cached for %d tickers", len(out))
    return out
