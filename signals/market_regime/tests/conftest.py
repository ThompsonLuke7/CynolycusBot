"""Synthetic-fixture helpers for signals.market_regime tests.

No network, no on-disk dependency: every test builds an in-memory loader
(matching the return shape of ``strategies.momentum_expansion.data.load_bars.load_1d``
— a UTC-tz-aware-DatetimeIndex-ed OHLCV frame) and injects it via the
``loader=`` parameter every builder function exposes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.momentum_expansion.config.momentum_config import SECTOR_ETFS

from signals.market_regime.config import CREDIT_DENOMINATOR

REQUIRED_TICKERS = sorted({"SPY", "IWM", "HYG", "LQD", "RSP", CREDIT_DENOMINATOR} | set(SECTOR_ETFS))


def utc_midnight_et(date) -> pd.Timestamp:
    """The load_1d daily-bar timestamp convention: ET midnight for `date`, in UTC
    (04:00 UTC in EDT, 05:00 UTC in EST — matches the real cached bars)."""
    local = pd.Timestamp(date).tz_localize("America/New_York")
    return local.tz_convert("UTC")


def session_calendar(start: str, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def make_bars(*, dates: pd.DatetimeIndex, close, volume=None) -> pd.DataFrame:
    """Build one ticker's OHLCV frame in load_1d's return shape."""
    idx = pd.DatetimeIndex([utc_midnight_et(d) for d in dates])
    close = np.asarray(close, dtype=float)
    assert len(close) == len(dates), "close series must match the date index length"
    if volume is None:
        volume = np.full(len(dates), 1_000_000.0)
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.asarray(volume, dtype=float),
        },
        index=idx,
    ).sort_index()


def random_walk_close(n: int, *, seed: int, start: float = 100.0, sigma: float = 0.01, drift: float = 0.0002):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, sigma, size=n)
    return start * np.cumprod(1.0 + rets)


def make_loader(bars: dict[str, pd.DataFrame]):
    """Build a fake ``load_1d``-shaped loader over an in-memory ticker->frame map.
    Missing tickers raise FileNotFoundError, matching real load_1d on a cache miss."""

    def _loader(ticker: str) -> pd.DataFrame:
        if ticker not in bars:
            raise FileNotFoundError(ticker)
        return bars[ticker]

    return _loader


def build_full_universe_bars(
    *, start: str, n: int, seed_base: int = 0, extra_tickers: dict[str, object] | None = None
) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    """Random-walk OHLCV fixtures for every ticker daily_regime/sector_state need,
    each with a distinct seed so cross-ticker correlations aren't degenerate."""
    dates = session_calendar(start, n)
    bars: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(REQUIRED_TICKERS):
        close = random_walk_close(n, seed=seed_base + i, start=100.0 + 5 * i)
        bars[ticker] = make_bars(dates=dates, close=close)
    if extra_tickers:
        for ticker, close in extra_tickers.items():
            bars[ticker] = make_bars(dates=dates, close=close)
    return dates, bars
