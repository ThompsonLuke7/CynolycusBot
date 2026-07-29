"""Correctness tests for signals.market_regime.daily_regime.

Every composite's arithmetic is checked against an independent numpy/pandas
reference implementation written directly from the plan's formulas (not by
importing daily_regime's internals), over a synthetic random-walk fixture
with no network or on-disk dependency. Small window overrides (window=10,
component windows 5/8, excess window 3) keep the fixture short (60-80
sessions) while still exercising post-warmup rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.momentum_expansion.config.momentum_config import SECTOR_ETFS

from signals.market_regime.config import CREDIT_DENOMINATOR
from signals.market_regime.daily_regime import build_daily_regime
from signals.market_regime.tests.conftest import (
    build_full_universe_bars,
    make_loader,
    utc_midnight_et,
)

WINDOW = 10
MIN_PERIODS = 10
COMP_SHORT = 5
COMP_LONG = 8
EXCESS_WINDOW = 3
N_SESSIONS = 60

BUILD_KWARGS = dict(
    zscore_window=WINDOW,
    zscore_min_periods=MIN_PERIODS,
    component_window_short=COMP_SHORT,
    component_window_long=COMP_LONG,
    excess_return_window=EXCESS_WINDOW,
)


def _ref_zscore(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    s = pd.Series(arr)
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()
    return ((s - mean) / std.replace(0, np.nan)).to_numpy()


def _closes(bars, ticker):
    return bars[ticker]["close"].to_numpy()


@pytest.fixture(scope="module")
def universe():
    return build_full_universe_bars(start="2021-01-04", n=N_SESSIONS, seed_base=100)


@pytest.fixture(scope="module")
def regime(universe):
    dates, bars = universe
    return build_daily_regime(loader=make_loader(bars), **BUILD_KWARGS)


def test_risk_appetite_arithmetic(universe, regime):
    dates, bars = universe
    xly, xlp = _closes(bars, "XLY"), _closes(bars, "XLP")
    iwm, spy = _closes(bars, "IWM"), _closes(bars, "SPY")
    hyg, iei = _closes(bars, "HYG"), _closes(bars, CREDIT_DENOMINATOR)
    rsp = _closes(bars, "RSP")

    z1 = _ref_zscore(np.log(xly / xlp), WINDOW, MIN_PERIODS)
    z2 = _ref_zscore(np.log(iwm / spy), WINDOW, MIN_PERIODS)
    z3 = _ref_zscore(np.log(hyg / iei), WINDOW, MIN_PERIODS)  # duration-controlled credit leg
    z4 = _ref_zscore(np.log(rsp / spy), WINDOW, MIN_PERIODS)

    stacked = np.vstack([z1, z2, z3, z4])
    expected = np.nanmean(stacked, axis=0)
    expected[np.all(np.isnan(stacked), axis=0)] = np.nan
    expected_n = np.sum(~np.isnan(stacked), axis=0)

    np.testing.assert_allclose(
        regime["risk_appetite_z"].to_numpy(), expected, equal_nan=True, rtol=1e-8, atol=1e-10
    )
    np.testing.assert_array_equal(regime["risk_appetite_z_n_components"].to_numpy(), expected_n)
    # Sanity: the composite is meaningfully populated past warmup, not degenerate all-NaN.
    assert regime["risk_appetite_z"].notna().sum() > N_SESSIONS - WINDOW - 2


def test_liquidity_stress_and_related_arithmetic(universe, regime):
    dates, bars = universe
    spy_close = _closes(bars, "SPY")
    spy_vol = bars["SPY"]["volume"].to_numpy()
    hyg, iei = _closes(bars, "HYG"), _closes(bars, CREDIT_DENOMINATOR)

    spy_ret1 = np.concatenate([[np.nan], np.diff(np.log(spy_close))])
    dollar_vol = spy_close * spy_vol
    amihud_raw = np.abs(spy_ret1) / dollar_vol
    amihud_20 = pd.Series(amihud_raw).rolling(COMP_SHORT, min_periods=COMP_SHORT).mean().to_numpy()
    dollar_vol_20 = pd.Series(dollar_vol).rolling(COMP_SHORT, min_periods=COMP_SHORT).mean().to_numpy()
    rv20 = pd.Series(spy_ret1).rolling(COMP_SHORT, min_periods=COMP_SHORT).std().to_numpy()
    credit_ratio = np.log(hyg / iei)  # duration-controlled: HYG/IEI, NOT HYG/LQD

    z_amihud = _ref_zscore(amihud_20, WINDOW, MIN_PERIODS)
    z_dollar_vol = _ref_zscore(dollar_vol_20, WINDOW, MIN_PERIODS)
    z_rv20 = _ref_zscore(rv20, WINDOW, MIN_PERIODS)
    z_credit = _ref_zscore(credit_ratio, WINDOW, MIN_PERIODS)

    stacked = np.vstack([z_amihud, -z_dollar_vol, -z_credit, z_rv20])
    expected_liquidity = np.nanmean(stacked, axis=0)
    expected_liquidity[np.all(np.isnan(stacked), axis=0)] = np.nan

    np.testing.assert_allclose(
        regime["liquidity_stress_z"].to_numpy(), expected_liquidity, equal_nan=True, rtol=1e-8, atol=1e-10
    )
    # credit_risk_z is the LITERAL (non-negated) z(log(HYG/IEI)), duration-controlled.
    np.testing.assert_allclose(
        regime["credit_risk_z"].to_numpy(), z_credit, equal_nan=True, rtol=1e-8, atol=1e-10
    )
    # spy_rv20_z is the same RV20 z reused standalone.
    np.testing.assert_allclose(
        regime["spy_rv20_z"].to_numpy(), z_rv20, equal_nan=True, rtol=1e-8, atol=1e-10
    )
    # Sign convention: liquidity_stress_z's dollar-volume and credit legs are negated.
    np.testing.assert_allclose(
        regime["liquidity_stress_credit_z"].to_numpy(), -z_credit, equal_nan=True, rtol=1e-8, atol=1e-10
    )


def test_credit_risk_hyg_lqd_diagnostic_column_does_not_feed_any_composite(universe, regime):
    """credit_risk_hyg_lqd_z must be the plan's literal (duration-contaminated)
    HYG/LQD ratio, present for comparison, but numerically distinct from — and
    not equal to — credit_risk_z (which is now HYG/IEI-based)."""
    dates, bars = universe
    hyg, lqd = _closes(bars, "HYG"), _closes(bars, "LQD")
    expected_diag = _ref_zscore(np.log(hyg / lqd), WINDOW, MIN_PERIODS)

    assert "credit_risk_hyg_lqd_z" in regime.columns
    np.testing.assert_allclose(
        regime["credit_risk_hyg_lqd_z"].to_numpy(), expected_diag, equal_nan=True, rtol=1e-8, atol=1e-10
    )
    # The two series must not be identical post-warmup (proves the composite
    # leg and the diagnostic leg are genuinely wired to different tickers).
    post_warmup = regime["credit_risk_z"].notna() & regime["credit_risk_hyg_lqd_z"].notna()
    assert post_warmup.sum() > 5
    assert not np.allclose(
        regime.loc[post_warmup, "credit_risk_z"].to_numpy(),
        regime.loc[post_warmup, "credit_risk_hyg_lqd_z"].to_numpy(),
    )


def test_breadth_and_dispersion_arithmetic(universe, regime):
    dates, bars = universe
    spy_close = _closes(bars, "SPY")

    above20_rows, above50_rows, excess_rows = [], [], []
    for etf in SECTOR_ETFS:
        c = _closes(bars, etf)
        s = pd.Series(c)
        sma20 = s.rolling(COMP_SHORT, min_periods=COMP_SHORT).mean().to_numpy()
        sma50 = s.rolling(COMP_LONG, min_periods=COMP_LONG).mean().to_numpy()
        above20_rows.append(np.where(np.isnan(sma20), np.nan, (c > sma20).astype(float)))
        above50_rows.append(np.where(np.isnan(sma50), np.nan, (c > sma50).astype(float)))
        ret_sector = s.pct_change(EXCESS_WINDOW).to_numpy()
        ret_spy = pd.Series(spy_close).pct_change(EXCESS_WINDOW).to_numpy()
        excess_rows.append(ret_sector - ret_spy)

    above20_arr = np.vstack(above20_rows)
    above50_arr = np.vstack(above50_rows)
    breadth_raw = np.nanmean(np.vstack([above20_arr, above50_arr]), axis=0)
    n_valid = np.sum(~np.isnan(above20_arr) & ~np.isnan(above50_arr), axis=0)
    breadth_raw = np.where(n_valid > 0, breadth_raw, np.nan)
    expected_breadth_z = _ref_zscore(breadth_raw, WINDOW, MIN_PERIODS)

    np.testing.assert_allclose(
        regime["breadth_z"].to_numpy(), expected_breadth_z, equal_nan=True, rtol=1e-8, atol=1e-10
    )
    np.testing.assert_array_equal(regime["breadth_z_n_components"].to_numpy(), n_valid)

    excess_arr = np.vstack(excess_rows)
    n_excess = np.sum(~np.isnan(excess_arr), axis=0)
    dispersion_raw = np.nanstd(excess_arr, axis=0, ddof=1)
    dispersion_raw = np.where(n_excess >= 2, dispersion_raw, np.nan)
    expected_dispersion_z = _ref_zscore(dispersion_raw, WINDOW, MIN_PERIODS)

    np.testing.assert_allclose(
        regime["sector_dispersion_z"].to_numpy(), expected_dispersion_z, equal_nan=True, rtol=1e-6, atol=1e-8
    )
    np.testing.assert_array_equal(regime["sector_dispersion_z_n_components"].to_numpy(), n_excess)


def test_spy_trend_state_arithmetic(universe, regime):
    dates, bars = universe
    spy_close = pd.Series(_closes(bars, "SPY"))
    ema20 = spy_close.ewm(span=20, adjust=False).mean()
    ema50 = spy_close.ewm(span=50, adjust=False).mean()
    warm = spy_close.rolling(COMP_LONG, min_periods=COMP_LONG).count() >= COMP_LONG
    expected = np.sign(ema20 - ema50).where(warm).to_numpy()
    np.testing.assert_allclose(regime["spy_trend_state"].to_numpy(), expected, equal_nan=True)


def test_available_at_monotonic_and_dst_correct():
    # Winter fixture (EST, UTC-5): 16:00 ET close + 30min settle = 21:30 UTC.
    dates_w, bars_w = build_full_universe_bars(start="2021-01-04", n=20, seed_base=5)
    regime_w = build_daily_regime(loader=make_loader(bars_w), zscore_window=5, zscore_min_periods=5,
                                   component_window_short=3, component_window_long=4, excess_return_window=2)
    assert regime_w["available_at"].is_monotonic_increasing
    row = regime_w[regime_w["date"] == pd.Timestamp("2021-01-05")]
    assert not row.empty
    assert row["available_at"].iloc[0] == pd.Timestamp("2021-01-05 21:30:00", tz="UTC")

    # Summer fixture (EDT, UTC-4): 16:00 ET close + 30min settle = 20:30 UTC.
    dates_s, bars_s = build_full_universe_bars(start="2021-07-05", n=20, seed_base=6)
    regime_s = build_daily_regime(loader=make_loader(bars_s), zscore_window=5, zscore_min_periods=5,
                                   component_window_short=3, component_window_long=4, excess_return_window=2)
    row_s = regime_s[regime_s["date"] == pd.Timestamp("2021-07-06")]
    assert not row_s.empty
    assert row_s["available_at"].iloc[0] == pd.Timestamp("2021-07-06 20:30:00", tz="UTC")


def test_short_history_ticker_is_nan_not_zero():
    dates, bars = build_full_universe_bars(start="2021-01-04", n=N_SESSIONS, seed_base=42)
    cutoff = 30
    bars["RSP"] = bars["RSP"].iloc[cutoff:]
    regime = build_daily_regime(loader=make_loader(bars), **BUILD_KWARGS)

    before = regime.iloc[:cutoff]
    assert before["risk_appetite_rsp_spy_z"].isna().all()
    assert not (before["risk_appetite_rsp_spy_z"] == 0).any()
    assert (before["risk_appetite_z_n_components"] <= 3).all()
    assert not (before["risk_appetite_z"] == 0).any()

    # Once RSP has accrued enough real history past the cutoff, its leg activates.
    assert regime["risk_appetite_rsp_spy_z"].iloc[cutoff + WINDOW:].notna().any()
    assert (regime["risk_appetite_z_n_components"].iloc[cutoff + WINDOW + 1:] == 4).any()


def test_staleness_flag_on_data_gap():
    dates, bars = build_full_universe_bars(start="2021-01-04", n=N_SESSIONS, seed_base=7)
    gap_date = dates[40]
    bars["HYG"] = bars["HYG"].drop(index=utc_midnight_et(gap_date))
    regime = build_daily_regime(loader=make_loader(bars), **BUILD_KWARGS)

    gap_row = regime[regime["date"] == pd.Timestamp(gap_date)]
    assert gap_row["credit_risk_z_stale_days"].iloc[0] == 1.0
    assert pd.notna(gap_row["credit_risk_z"].iloc[0])  # forward-filled, not dropped

    next_row = regime[regime["date"] == pd.Timestamp(dates[41])]
    assert next_row["credit_risk_z_stale_days"].iloc[0] == 0.0

    prewarm_row = regime[regime["date"] == pd.Timestamp(dates[35])]
    assert prewarm_row["credit_risk_z_stale_days"].iloc[0] == 0.0


def test_leakage_row_identical_when_future_sessions_appended():
    dates_full, bars_full = build_full_universe_bars(start="2021-01-04", n=80, seed_base=11)
    truncated_bars = {t: df.iloc[:60] for t, df in bars_full.items()}

    regime_trunc = build_daily_regime(loader=make_loader(truncated_bars), **BUILD_KWARGS)
    regime_full = build_daily_regime(loader=make_loader(bars_full), **BUILD_KWARGS)

    expected = regime_full.iloc[:60].reset_index(drop=True)
    pd.testing.assert_frame_equal(regime_trunc.reset_index(drop=True), expected, check_exact=False, rtol=1e-10)
