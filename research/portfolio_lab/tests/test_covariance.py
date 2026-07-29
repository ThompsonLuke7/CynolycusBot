"""Tests for research/portfolio_lab/covariance.py.

Covers: shrinkage correctness against sklearn's own LedoitWolf fit directly
on the same causal window, min_periods short-history behavior (None, never
zero-filled), and the causality/leakage guarantee required by AGENTS.md and
the WS-D task spec -- appending future sessions to the input frame must not
change a past asof's output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from research.portfolio_lab import covariance as cov_mod


def _synthetic_prices(n_days=300, n_tickers=6, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, tz="UTC")
    # correlated returns: shared factor + idiosyncratic noise
    factor = rng.normal(0, 0.01, n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    rets = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for i, t in enumerate(tickers):
        beta = 0.3 + 0.1 * i
        idio = rng.normal(0, 0.008, n_days)
        rets[t] = beta * factor + idio
    prices = 100 * (1 + rets).cumprod()
    return prices


# --------------------------------------------------------------------------
# trailing_log_returns
# --------------------------------------------------------------------------

def test_trailing_log_returns_excludes_asof_session():
    prices = _synthetic_prices(n_days=50, n_tickers=3)
    asof = prices.index[30]
    rets = cov_mod.trailing_log_returns(prices, asof=asof, window=10, min_periods=5)
    assert rets is not None
    assert rets.index.max() < asof
    # window=10 -> at most 10 diffs
    assert len(rets) <= 10


def test_trailing_log_returns_none_below_min_periods():
    prices = _synthetic_prices(n_days=50, n_tickers=3)
    asof = prices.index[3]  # only ~3 sessions of history before this
    rets = cov_mod.trailing_log_returns(prices, asof=asof, window=120, min_periods=60)
    assert rets is None


def test_trailing_log_returns_drops_incomplete_columns():
    prices = _synthetic_prices(n_days=40, n_tickers=3)
    prices = prices.copy()
    prices.iloc[:20, prices.columns.get_loc("T1")] = np.nan  # T1 has a long gap
    asof = prices.index[35]
    rets = cov_mod.trailing_log_returns(prices, asof=asof, window=30, min_periods=5)
    assert rets is not None
    assert not rets.isna().any().any()


# --------------------------------------------------------------------------
# shrinkage correctness
# --------------------------------------------------------------------------

def test_shrinkage_matches_sklearn_direct_fit():
    prices = _synthetic_prices(n_days=200, n_tickers=5)
    asof = prices.index[150]
    result = cov_mod.ledoit_wolf_asof(prices, asof=asof, window=100, min_periods=30, annualize=False)
    assert result is not None

    rets = cov_mod.trailing_log_returns(prices, asof=asof, window=100, min_periods=30)
    ref = LedoitWolf().fit(rets.values)
    np.testing.assert_allclose(result.cov.values, ref.covariance_, rtol=1e-10)
    assert result.shrinkage == pytest.approx(ref.shrinkage_)


def test_shrinkage_shrinks_toward_diagonal_vs_sample_cov():
    """In a small-sample, correlated-factor setting, Ledoit-Wolf shrinkage
    should be strictly between 0 and 1, and the resulting covariance's
    off-diagonal structure should be damped relative to the raw sample
    covariance (the entire point of shrinkage estimation)."""
    prices = _synthetic_prices(n_days=120, n_tickers=8, seed=11)
    asof = prices.index[100]
    rets = cov_mod.trailing_log_returns(prices, asof=asof, window=40, min_periods=20)
    assert rets is not None
    result = cov_mod.ledoit_wolf_shrunk_cov(rets, asof=asof, annualize=False)
    assert result is not None
    assert 0.0 < result.shrinkage < 1.0

    sample_cov = np.cov(rets.values, rowvar=False, ddof=1)
    off_diag_mask = ~np.eye(sample_cov.shape[0], dtype=bool)
    sample_off_mag = np.abs(sample_cov[off_diag_mask]).mean()
    shrunk_off_mag = np.abs(result.cov.values[off_diag_mask]).mean()
    assert shrunk_off_mag <= sample_off_mag


def test_annualization_scales_by_trading_days():
    prices = _synthetic_prices(n_days=150, n_tickers=4)
    asof = prices.index[120]
    daily = cov_mod.ledoit_wolf_asof(prices, asof=asof, window=60, min_periods=20, annualize=False)
    annual = cov_mod.ledoit_wolf_asof(prices, asof=asof, window=60, min_periods=20, annualize=True)
    np.testing.assert_allclose(annual.cov.values, daily.cov.values * 252, rtol=1e-10)
    # correlation is scale-invariant
    np.testing.assert_allclose(annual.corr.values, daily.corr.values, atol=1e-8)


def test_correlation_diagonal_is_one():
    prices = _synthetic_prices(n_days=150, n_tickers=4)
    asof = prices.index[120]
    result = cov_mod.ledoit_wolf_asof(prices, asof=asof, window=60, min_periods=20)
    np.testing.assert_allclose(np.diag(result.corr.values), 1.0)


# --------------------------------------------------------------------------
# Causality / leakage guarantee
# --------------------------------------------------------------------------

def test_asof_is_causal_future_rows_do_not_change_past_result():
    """The core leakage guarantee: fitting on a frame truncated at `asof` vs.
    a frame with many additional FUTURE sessions appended must produce a
    byte-identical result for that same `asof`. If this test fails, the
    estimator is (or could become) look-ahead-biased."""
    full_prices = _synthetic_prices(n_days=400, n_tickers=6, seed=42)
    asof = full_prices.index[200]

    truncated_prices = full_prices.loc[full_prices.index <= asof]

    result_full = cov_mod.ledoit_wolf_asof(full_prices, asof=asof, window=120, min_periods=60)
    result_truncated = cov_mod.ledoit_wolf_asof(truncated_prices, asof=asof, window=120, min_periods=60)

    assert result_full is not None and result_truncated is not None
    assert result_full.n_obs == result_truncated.n_obs
    assert result_full.shrinkage == pytest.approx(result_truncated.shrinkage)
    pd.testing.assert_frame_equal(result_full.cov, result_truncated.cov)
    pd.testing.assert_frame_equal(result_full.corr, result_truncated.corr)


def test_asof_excludes_same_day_close():
    """A price move ON the asof session itself must not be visible: bump
    just that day's close after the asof date's row is included and confirm
    the causal result is unchanged (only sessions strictly before asof may
    move the estimate)."""
    prices = _synthetic_prices(n_days=200, n_tickers=5, seed=3)
    asof = prices.index[150]
    result_before = cov_mod.ledoit_wolf_asof(prices, asof=asof, window=80, min_periods=30)

    bumped = prices.copy()
    bumped.loc[asof, :] = bumped.loc[asof, :] * 5.0  # huge same-day move
    # also bump every future session so a same-day OR later leak would show up
    bumped.loc[bumped.index > asof, :] = bumped.loc[bumped.index > asof, :] * 5.0
    result_after = cov_mod.ledoit_wolf_asof(bumped, asof=asof, window=80, min_periods=30)

    assert result_before is not None and result_after is not None
    pd.testing.assert_frame_equal(result_before.cov, result_after.cov)
