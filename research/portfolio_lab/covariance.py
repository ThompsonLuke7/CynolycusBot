"""Ledoit-Wolf shrunk covariance from trailing daily log returns.

``scikit-learn`` is already a pinned dependency (``requirements.txt``:
``scikit-learn>=1.5,<1.6``), so this wraps ``sklearn.covariance.LedoitWolf``
rather than reimplementing the shrinkage constant.

Causality contract (see AGENTS.md "Data integrity and time correctness" and
the WS-D task spec): every function here takes an explicit ``asof`` and uses
only sessions strictly BEFORE it. A decision made intraday (a 4H bar) or at
the open of session ``asof`` cannot see that session's own close, so the
session dated ``asof`` is always excluded from the trailing window -- never
just "the last N rows including today". ``trailing_log_returns`` and
``ledoit_wolf_asof`` both enforce this with a strict ``<`` filter; there is
no parameter to relax it. ``tests/test_covariance.py::test_asof_is_causal``
proves a row's output is byte-identical whether or not future sessions have
already been appended to the input frame.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class ShrunkCovResult:
    """Annualized shrunk covariance/correlation as of one decision timestamp."""

    asof: pd.Timestamp
    tickers: list[str]
    cov: pd.DataFrame          # annualized covariance (daily cov * 252), index/cols = tickers
    corr: pd.DataFrame         # correlation implied by ``cov``, diag forced to 1.0
    shrinkage: float           # sklearn's fitted shrinkage constant, in [0, 1]
    n_obs: int                 # trailing sessions actually used


def trailing_log_returns(
    prices: pd.DataFrame, *, asof: pd.Timestamp, window: int, min_periods: int
) -> pd.DataFrame | None:
    """Log returns for up to ``window`` sessions strictly before ``asof``.

    ``prices``: wide frame, sorted ``DatetimeIndex`` (daily), one column per
    ticker, values = close (or any single causal price field). Columns with
    no data anywhere in the window are dropped; rows with any remaining NaN
    (a ticker missing a session within an otherwise-covered window) are
    dropped so every returned column is fully dense over the same dates --
    ``LedoitWolf`` requires a rectangular, NaN-free matrix.

    Returns ``None`` (never a zero-filled or short-window frame) when fewer
    than ``min_periods`` dense rows are available, per AGENTS.md: "rows before
    min_periods are NaN, never zero-filled."
    """
    asof = pd.Timestamp(asof)
    hist = prices.loc[prices.index < asof]
    if hist.empty:
        return None
    # need window+1 raw price rows to form `window` diffs
    hist = hist.iloc[-(window + 1):]
    rets = np.log(hist).diff().iloc[1:]
    rets = rets.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if rets.shape[0] < min_periods or rets.shape[1] == 0:
        return None
    return rets


def ledoit_wolf_shrunk_cov(
    returns: pd.DataFrame, *, asof, annualize: bool = True,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> ShrunkCovResult | None:
    """Fit Ledoit-Wolf shrinkage on an already-causal returns frame.

    ``returns`` must already be restricted to information available at
    ``asof`` (see ``trailing_log_returns``) -- this function does no date
    filtering itself, it only fits and annualizes. Returns ``None`` when
    there are fewer than 2 observations or 0 columns (degenerate input).
    """
    r = returns.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if r.shape[0] < 2 or r.shape[1] < 1:
        return None
    lw = LedoitWolf().fit(r.values)
    cov = lw.covariance_ * (trading_days if annualize else 1.0)
    cov_df = pd.DataFrame(cov, index=r.columns, columns=r.columns)
    std = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(std, std)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    corr_df = pd.DataFrame(corr, index=r.columns, columns=r.columns)
    return ShrunkCovResult(
        asof=pd.Timestamp(asof), tickers=list(r.columns), cov=cov_df, corr=corr_df,
        shrinkage=float(lw.shrinkage_), n_obs=int(r.shape[0]),
    )


def ledoit_wolf_asof(
    prices: pd.DataFrame, *, asof, window: int = 120, min_periods: int = 60,
    annualize: bool = True, trading_days: int = TRADING_DAYS_PER_YEAR,
) -> ShrunkCovResult | None:
    """Convenience: ``trailing_log_returns`` + ``ledoit_wolf_shrunk_cov`` in one call."""
    rets = trailing_log_returns(prices, asof=asof, window=window, min_periods=min_periods)
    if rets is None:
        return None
    return ledoit_wolf_shrunk_cov(rets, asof=asof, annualize=annualize, trading_days=trading_days)
