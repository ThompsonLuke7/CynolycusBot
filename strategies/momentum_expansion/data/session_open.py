"""Today's official session open, for the live edge of the daily feature block.

``daily_gap_pct`` / ``daily_gap_z_60`` are built from the daily bar's open (the
09:30 auction print). Intraday the daily cache has no row for today, and the 1H
cache starts at 10:00 with a first print that can sit far from the auction
(AAPL 2026-11-18: 355.79 vs 351.94), so it is not a usable stand-in. This module
fetches today's (partial) daily bar from Alpaca, keeps only its open, and adds it
to the daily frame as a partial row (open set, every other field NaN), which
``feature_matrix_4h`` then treats exactly like the completed row history has.
Nothing here is written to the shared bar cache.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
# The SIP subscription refuses the most recent ~15 minutes (same buffer as the
# shared-bar catch-up).
SIP_DELAY = dt.timedelta(minutes=20)


def session_day(now: dt.datetime | None = None) -> pd.Timestamp:
    """Today's ET calendar date (tz-naive midnight)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    return pd.Timestamp(now.astimezone(ET).date())


def fetch_session_opens(tickers: Iterable[str], *, now: dt.datetime | None = None,
                        batch_size: int = 200, client=None) -> dict[str, float]:
    """``{ticker: official open}`` for today's session, from Alpaca SIP daily bars.

    Returns what it could get: a failed batch is logged and skipped, and before
    the open (or on a holiday) the result is empty. Callers fall back to the
    previous session's gap for any ticker missing here.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    day = session_day(now)
    start = dt.datetime.combine(day.date(), dt.time(0, 0), tzinfo=ET)
    end = now - SIP_DELAY
    if end <= dt.datetime.combine(day.date(), dt.time(9, 31), tzinfo=ET):
        return {}
    symbols = sorted({str(t).upper() for t in tickers if str(t).strip()})
    if not symbols:
        return {}
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    if client is None:
        from alpaca.data.historical import StockHistoricalDataClient
        from core.API.Alpaca_API.core.config import AlpacaConfig

        cfg = AlpacaConfig.from_env()
        client = StockHistoricalDataClient(api_key=cfg.key_id, secret_key=cfg.secret_key)
    out: dict[str, float] = {}
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        try:
            frame = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=batch, timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=start, end=end, adjustment=Adjustment.RAW, feed=DataFeed.SIP,
            )).df
        except Exception as exc:  # noqa: BLE001 - a failed batch degrades, never aborts
            logger.warning("session opens: batch %d/%d failed: %s", i // batch_size + 1,
                           -(-len(symbols) // batch_size), exc)
            continue
        if frame is None or frame.empty:
            continue
        frame = frame.reset_index()
        ts = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(ET).dt.normalize().dt.tz_localize(None)
        today = frame.loc[ts == day]
        for sym, px in zip(today["symbol"], today["open"]):
            if px is not None and np.isfinite(px) and px > 0:
                out[str(sym).upper()] = float(px)
    return out


def append_session_open(daily: pd.DataFrame | None, open_px: float | None,
                        day: pd.Timestamp) -> pd.DataFrame | None:
    """``daily`` plus a partial row for ``day`` (open only), if it has no row for
    ``day`` yet and the open is known. Otherwise ``daily`` unchanged."""
    if daily is None or daily.empty or open_px is None or not np.isfinite(open_px):
        return daily
    idx = pd.to_datetime(daily.index)
    tz = idx.tz
    days = (idx.tz_convert(ET).tz_localize(None) if tz is not None else idx).normalize()
    if (days >= day).any():
        return daily
    stamp = pd.Timestamp(day).tz_localize(ET)
    stamp = stamp.tz_convert(tz) if tz is not None else stamp.tz_localize(None)
    row = pd.DataFrame(np.nan, index=pd.DatetimeIndex([stamp], name=daily.index.name), columns=daily.columns)
    for col in daily.columns:
        if not pd.api.types.is_numeric_dtype(daily[col]):
            row[col] = row[col].astype(object)
    open_col = next((c for c in daily.columns if str(c).lower() == "open"), None)
    if open_col is None:
        return daily
    row[open_col] = float(open_px)
    out = pd.concat([daily, row])
    out.index = pd.DatetimeIndex(out.index, name=daily.index.name)
    return out
