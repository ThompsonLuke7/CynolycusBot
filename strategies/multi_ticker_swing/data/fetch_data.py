"""
Fetch and cache 30-minute and 10-minute OHLCV bars for all universe tickers.

Wraps the existing fetch_intraday utility so credentials + pagination
are handled identically to the SPY live system.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from core.API.Alpaca_API.market_data.fetch_intraday import fetch_intraday
from strategies.multi_ticker_swing.config.pipeline_config import (
    RAW_30M_DIR,
    RAW_5M_DIR,
    RAW_10M_DIR,
    TRAIN_START,
    TRAIN_END,
    UNIVERSE_CSV,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Universe helpers
# ---------------------------------------------------------------------------

def load_universe(csv_path: Path | str = UNIVERSE_CSV) -> pd.DataFrame:
    """Return universe DataFrame with columns: ticker, sector, cap_bucket, asset_type."""
    df = pd.read_csv(csv_path)
    if "ticker" in df.columns:
        df = df.copy()
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper().str.replace("$", "", regex=False)
        df = df[df["ticker"] != ""]

    # The shared universe is broader than the swing model universe.  When the
    # caller points this loader at Data/shared/universe/shared_universe.csv,
    # keep only names that passed the shared liquidity/eligibility filter.
    if "is_eligible" in df.columns:
        df = df[df["is_eligible"].fillna(False).astype(bool)].copy()

    for col in ("sector", "market_cap_bucket", "type", "asset_type"):
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    return df.drop_duplicates("ticker").reset_index(drop=True)


def universe_tickers(csv_path: Path | str = UNIVERSE_CSV) -> list[str]:
    return load_universe(csv_path)["ticker"].tolist()


# ---------------------------------------------------------------------------
# Single-ticker fetch
# ---------------------------------------------------------------------------

def _raw_path(ticker: str, base_dir: Path) -> Path:
    return base_dir / f"{ticker}.parquet"


def fetch_ticker(
    ticker: str,
    *,
    start: str = TRAIN_START,
    end: str = TRAIN_END,
    alpaca_timeframe: str,        # e.g. "30Min" or "10Min"
    base_dir: Path,
    force: bool = False,
) -> pd.DataFrame | None:
    """
    Fetch a single ticker and save to base_dir/{ticker}.parquet.
    Returns None (and skips) if the file already exists and force=False.
    """
    out_path = _raw_path(ticker, base_dir)
    if out_path.exists() and not force:
        logger.info("[%s] cache hit at %s — skipping", ticker, out_path)
        return None

    base_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[%s] fetching %s  %s → %s", ticker, alpaca_timeframe, start, end)
    try:
        df = fetch_intraday(
            ticker=ticker,
            start=start,
            end=end,
            timeframe=alpaca_timeframe,
            limit=500_000,
            adjustment="split",
            save_path=str(out_path),
        )
        logger.info("[%s] saved %d bars → %s", ticker, len(df), out_path)
        return df
    except Exception as exc:
        logger.error("[%s] fetch failed: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Multi-ticker orchestrator
# ---------------------------------------------------------------------------

def fetch_all(
    *,
    start: str = TRAIN_START,
    end: str = TRAIN_END,
    universe_csv: Path | str = UNIVERSE_CSV,
    fetch_30m: bool = True,
    fetch_5m: bool = False,
    fetch_10m: bool = False,
    force: bool = False,
) -> None:
    """
    Fetch 30m, 5m, and/or 10m data for every ticker in the universe CSV.
    Already-cached files are skipped unless force=True.
    """
    tickers = universe_tickers(universe_csv)
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        logger.info("(%d/%d) %s", i, total, ticker)

        if fetch_30m:
            fetch_ticker(
                ticker,
                start=start,
                end=end,
                alpaca_timeframe="30Min",
                base_dir=RAW_30M_DIR,
                force=force,
            )

        if fetch_5m:
            fetch_ticker(
                ticker,
                start=start,
                end=end,
                alpaca_timeframe="5Min",
                base_dir=RAW_5M_DIR,
                force=force,
            )

        if fetch_10m:
            fetch_ticker(
                ticker,
                start=start,
                end=end,
                alpaca_timeframe="10Min",
                base_dir=RAW_10M_DIR,
                force=force,
            )

    logger.info("fetch_all complete (%d tickers)", total)
