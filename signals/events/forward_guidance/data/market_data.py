"""Alpaca-backed market-window fetch and cache helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from core.API.Alpaca_API.market_data.fetch_intraday import fetch_intraday
from signals.events.forward_guidance.config import (
    CONTEXT_TICKERS,
    DEFAULT_ALPACA_FEED,
    DEFAULT_MARKET_TIMEFRAME,
    MARKET_FORWARD_DAYS,
    MARKET_LOOKBACK_DAYS,
    MARKET_WINDOWS_DIR,
    ensure_data_dirs,
)
from signals.events.forward_guidance.data.schema import EarningsEvent
from signals.events.forward_guidance.utils.universe import ticker_sector_etf

logger = logging.getLogger(__name__)


def event_market_dir(event: EarningsEvent) -> Path:
    return MARKET_WINDOWS_DIR / event.event_id


def market_symbols_for_event(event: EarningsEvent) -> list[str]:
    symbols = {event.clean_ticker, *CONTEXT_TICKERS}
    sector = event.sector_etf or ticker_sector_etf(event.clean_ticker)
    if sector:
        symbols.add(str(sector).upper())
    return sorted(symbols)


def _window_bounds(event: EarningsEvent) -> tuple[str, str]:
    reaction = pd.Timestamp(event.reaction_date)
    start = reaction - pd.Timedelta(days=MARKET_LOOKBACK_DAYS)
    end = reaction + pd.Timedelta(days=MARKET_FORWARD_DAYS)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def market_window_path(event: EarningsEvent, symbol: str, timeframe: str = DEFAULT_MARKET_TIMEFRAME) -> Path:
    return event_market_dir(event) / f"{symbol.upper()}_{timeframe.lower()}.parquet"


def fetch_symbol_window(
    event: EarningsEvent,
    symbol: str,
    *,
    timeframe: str = DEFAULT_MARKET_TIMEFRAME,
    force: bool = False,
    feed: str = DEFAULT_ALPACA_FEED,
) -> pd.DataFrame | None:
    ensure_data_dirs()
    out_path = market_window_path(event, symbol, timeframe)
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)
    start, end = _window_bounds(event)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[%s] fetching %s %s bars %s -> %s", event.event_id, symbol, timeframe, start, end)
    try:
        df = fetch_intraday(
            ticker=symbol,
            start=start,
            end=end,
            timeframe=timeframe,
            limit=500_000,
            adjustment="split",
            feed=feed,
            save_path=str(out_path),
        )
    except Exception as exc:
        logger.warning("[%s] %s market fetch failed: %s", event.event_id, symbol, exc)
        return None
    return df


def fetch_event_market_window(
    event: EarningsEvent,
    *,
    timeframe: str = DEFAULT_MARKET_TIMEFRAME,
    force: bool = False,
    feed: str = DEFAULT_ALPACA_FEED,
) -> dict[str, pd.DataFrame]:
    bars: dict[str, pd.DataFrame] = {}
    for symbol in market_symbols_for_event(event):
        df = fetch_symbol_window(event, symbol, timeframe=timeframe, force=force, feed=feed)
        if df is not None and not df.empty:
            bars[symbol.upper()] = df
    return bars


def load_event_market_window(
    event: EarningsEvent,
    *,
    timeframe: str = DEFAULT_MARKET_TIMEFRAME,
) -> dict[str, pd.DataFrame]:
    bars: dict[str, pd.DataFrame] = {}
    for path in event_market_dir(event).glob(f"*_{timeframe.lower()}.parquet"):
        symbol = path.name[: -len(f"_{timeframe.lower()}.parquet")]
        bars[symbol.upper()] = pd.read_parquet(path)
    return bars
