"""Causal daily-bar context for the sizing research engine.

Backed by ``strategies.momentum_expansion.data.load_bars.load_1d`` (per the
WS-D task spec: "Use load_1d, do not read the parquet naively" -- that
function already normalizes the flat ``Data/shared/bars/1d/*.parquet`` frame
into a sorted, tz-aware ``DatetimeIndex``). Each ticker's full history is
loaded once and cached in-process; every lookup takes an explicit ``asof``
and only ever returns sessions strictly before it -- this module has no
notion of "now" and cannot look ahead on its own.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from strategies.momentum_expansion.data.load_bars import load_1d

from research.portfolio_lab import covariance as cov_mod

logger = logging.getLogger(__name__)


def _to_utc_session(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.normalize()


@dataclass
class DailyContext:
    """Lazy, cached wide-frame view of daily closes/dollar-volume for a fixed
    ticker universe. Not thread-safe (matches the rest of this research code
    -- single-process backtest loops only)."""

    tickers: list[str]
    _close: dict = field(default_factory=dict, repr=False)
    _dollar_vol: dict = field(default_factory=dict, repr=False)
    _loaded: set = field(default_factory=set, repr=False)
    _wide_close: pd.DataFrame | None = field(default=None, repr=False)

    def _load(self, ticker: str) -> None:
        if ticker in self._loaded:
            return
        self._loaded.add(ticker)
        try:
            df = load_1d(ticker)
        except FileNotFoundError:
            logger.debug("DailyContext: no 1d bars cached for %s", ticker)
            return
        if df.empty or "close" not in df.columns:
            return
        idx = df.index.tz_convert("UTC").normalize()
        close = pd.Series(df["close"].to_numpy(dtype=float), index=idx)
        close = close[~close.index.duplicated(keep="last")].sort_index()
        close = close[close > 0]
        if close.empty:
            return
        self._close[ticker] = close
        if "volume" in df.columns:
            vol = pd.Series(df["volume"].to_numpy(dtype=float), index=idx)
            vol = vol[~vol.index.duplicated(keep="last")].sort_index()
            self._dollar_vol[ticker] = (close * vol.reindex(close.index)).dropna()

    def wide_close(self, tickers: list[str] | None = None) -> pd.DataFrame:
        """Full wide close frame (all cached tickers) if no subset requested."""
        want = tickers if tickers is not None else self.tickers
        for t in want:
            self._load(t)
        if tickers is None:
            if self._wide_close is None:
                self._wide_close = pd.DataFrame(self._close).sort_index()
            return self._wide_close
        cols = {t: self._close[t] for t in want if t in self._close}
        return pd.DataFrame(cols).sort_index()

    def trailing_returns_subset(
        self, tickers: list[str], *, asof, window: int, min_periods: int,
    ) -> pd.DataFrame | None:
        """Causal log-return frame restricted to ``tickers`` (kept small on
        purpose -- see portfolio_backtest.py: covariance is only ever fit on
        the currently-relevant book/candidate subset, not the whole universe,
        so a single illiquid name's data gap can't wipe out the whole-window
        row alignment for everyone else)."""
        if not tickers:
            return None
        wide = self.wide_close(tickers)
        if wide.empty:
            return None
        return cov_mod.trailing_log_returns(wide, asof=asof, window=window, min_periods=min_periods)

    def trailing_adv(
        self, ticker: str, *, asof, window: int = 20, min_periods: int = 5,
    ) -> float | None:
        """Trailing mean daily dollar volume over up to ``window`` sessions
        strictly before ``asof``. ``None`` when unknown/insufficient -- never
        zero-filled or inferred."""
        self._load(ticker)
        dv = self._dollar_vol.get(ticker)
        if dv is None or dv.empty:
            return None
        asof = _to_utc_session(asof)
        hist = dv.loc[dv.index < asof].iloc[-window:].dropna()
        if len(hist) < min_periods:
            return None
        return float(hist.mean())

    def last_close_before(self, ticker: str, *, asof) -> float | None:
        self._load(ticker)
        close = self._close.get(ticker)
        if close is None or close.empty:
            return None
        asof = _to_utc_session(asof)
        hist = close.loc[close.index < asof]
        if hist.empty:
            return None
        return float(hist.iloc[-1])
