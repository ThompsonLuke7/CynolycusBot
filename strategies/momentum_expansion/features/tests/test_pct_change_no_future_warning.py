"""Regression: every `.pct_change()` call in feature_matrix_4h.py must pin
`fill_method=None`.

2026-07-20 live audit found ~5,700 FutureWarning lines in one session's server
log from pandas' `Series.pct_change` default `fill_method='pad'` deprecation
(it only fires when the input has a genuine internal/non-leading NaN -- a
trivially-clean synthetic series never triggers it, which is why this test
deliberately punches a one-day gap into the daily close series, mimicking a
real missing/halted trading day). `weekly_ret_1`/`weekly_ret_4`
(feature_matrix_4h.py:421-422) were the dominant contributors and already had
`fill_method=None` pinned in an uncommitted fix at audit time, but several
other `.pct_change()` calls in the same file (ema_slope_*, ret_*, rs_*,
regime_spy_ret_20, daily d_ret_*) were still on the deprecated default and
would have started firing the same warning once the dominant two stopped.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from strategies.momentum_expansion.features.feature_matrix_4h import build_ticker_features_4h
from strategies.momentum_expansion.features.tests.test_live_feature_panel_4h import _synthetic_4h_bars

N_DAYS = 260


def _daily_bars_with_internal_gap(seed: int) -> pd.DataFrame:
    """Daily OHLCV with one non-leading NaN close -- e.g. a halted session --
    so pct_change over this series has real fill_method='pad' vs None to
    disagree on and pandas' deprecation warning actually has something to
    fire about."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-02", periods=N_DAYS, freq="B", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 1, N_DAYS))
    close = np.clip(close, 5, None)
    frame = pd.DataFrame({
        "open": close + rng.normal(0, 0.3, N_DAYS),
        "high": close + rng.uniform(0, 1, N_DAYS),
        "low": close - rng.uniform(0, 1, N_DAYS),
        "close": close,
        "volume": rng.uniform(1e5, 1e6, N_DAYS),
    }, index=idx)
    frame.loc[frame.index[150], "close"] = np.nan  # the internal gap
    return frame


def _future_warnings(caught) -> list:
    return [w for w in caught if issubclass(w.category, FutureWarning) and "pct_change" in str(w.message)]


def test_build_ticker_features_4h_raises_no_pct_change_future_warning():
    df_4h = _synthetic_4h_bars(seed=1)
    df_1d = _daily_bars_with_internal_gap(seed=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = build_ticker_features_4h(ticker="AAA", df_4h=df_4h, df_1d=df_1d, ctx_4h={})
    assert out is not None
    assert _future_warnings(caught) == []


def test_fixture_actually_exercises_the_deprecated_path(monkeypatch):
    """Efficacy check for the test above: reverting fill_method=None on the
    daily d_ret_1/d_ret_5 calls must make the warning reappear against the
    SAME gapped fixture, proving it isn't a false-negative from clean data.
    """
    df_1d = _daily_bars_with_internal_gap(seed=2)
    dc = df_1d["close"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dc.pct_change(1)  # deprecated default, deliberately not fixed here
    assert _future_warnings(caught) != []
