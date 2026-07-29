"""Correctness tests for signals.market_regime.sector_state."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.momentum_expansion.config.momentum_config import SECTOR_ETFS

from signals.market_regime.sector_state import build_sector_state
from signals.market_regime.tests.conftest import build_full_universe_bars, make_loader

COMP_SHORT = 5
COMP_LONG = 8
EXCESS_SHORT = 3
EXCESS_LONG = 6
N_SESSIONS = 40

BUILD_KWARGS = dict(
    component_window_short=COMP_SHORT,
    component_window_long=COMP_LONG,
    excess_return_window=EXCESS_SHORT,
    excess_return_window_long=EXCESS_LONG,
)


@pytest.fixture(scope="module")
def universe():
    return build_full_universe_bars(start="2021-01-04", n=N_SESSIONS, seed_base=200)


@pytest.fixture(scope="module")
def state(universe):
    dates, bars = universe
    return build_sector_state(loader=make_loader(bars), **BUILD_KWARGS)


def test_columns_and_shape(universe, state):
    dates, bars = universe
    assert set(state.columns) == {
        "date", "sector_etf", "available_at", "excess_21d", "excess_63d",
        "rank_21d", "rank_63d", "rs_accel", "above_20d", "above_50d", "stale_days",
    }
    assert len(state) == N_SESSIONS * len(SECTOR_ETFS)
    assert set(state["sector_etf"].unique()) == set(SECTOR_ETFS)


def test_excess_return_and_rs_accel_arithmetic(universe, state):
    dates, bars = universe
    spy_close = bars["SPY"]["close"].to_numpy()
    spy_ret_s = pd.Series(spy_close).pct_change(EXCESS_SHORT).to_numpy()
    spy_ret_l = pd.Series(spy_close).pct_change(EXCESS_LONG).to_numpy()

    for etf in SECTOR_ETFS:
        c = bars[etf]["close"].to_numpy()
        ret_s = pd.Series(c).pct_change(EXCESS_SHORT).to_numpy()
        ret_l = pd.Series(c).pct_change(EXCESS_LONG).to_numpy()
        expected_excess_21d = ret_s - spy_ret_s
        expected_excess_63d = ret_l - spy_ret_l
        expected_rs_accel = expected_excess_21d - expected_excess_63d / 3.0

        row = state[state["sector_etf"] == etf].sort_values("date")
        np.testing.assert_allclose(row["excess_21d"].to_numpy(), expected_excess_21d, equal_nan=True, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(row["excess_63d"].to_numpy(), expected_excess_63d, equal_nan=True, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(row["rs_accel"].to_numpy(), expected_rs_accel, equal_nan=True, rtol=1e-8, atol=1e-10)

        sma20 = pd.Series(c).rolling(COMP_SHORT, min_periods=COMP_SHORT).mean().to_numpy()
        sma50 = pd.Series(c).rolling(COMP_LONG, min_periods=COMP_LONG).mean().to_numpy()
        expected_above20 = np.where(np.isnan(sma20), np.nan, (c > sma20).astype(float))
        expected_above50 = np.where(np.isnan(sma50), np.nan, (c > sma50).astype(float))
        np.testing.assert_allclose(row["above_20d"].to_numpy(), expected_above20, equal_nan=True)
        np.testing.assert_allclose(row["above_50d"].to_numpy(), expected_above50, equal_nan=True)


def test_ranks_are_cross_sectional_same_date(universe, state):
    """rank_21d/rank_63d must be a percentile rank across sector ETFs at each
    date, independent of other dates."""
    dates, bars = universe
    for d in state["date"].unique()[EXCESS_SHORT + 1:]:
        day_rows = state[state["date"] == d]
        expected = day_rows["excess_21d"].rank(pct=True)
        pd.testing.assert_series_equal(
            day_rows["rank_21d"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


def test_leakage_row_identical_when_future_sessions_appended():
    dates_full, bars_full = build_full_universe_bars(start="2021-01-04", n=60, seed_base=21)
    truncated_bars = {t: df.iloc[:40] for t, df in bars_full.items()}

    state_trunc = build_sector_state(loader=make_loader(truncated_bars), **BUILD_KWARGS)
    state_full = build_sector_state(loader=make_loader(bars_full), **BUILD_KWARGS)

    expected = state_full[state_full["date"] < pd.Timestamp(dates_full[40])].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        state_trunc.reset_index(drop=True), expected, check_exact=False, rtol=1e-10
    )


def test_short_history_sector_etf_is_nan_not_zero():
    dates, bars = build_full_universe_bars(start="2021-01-04", n=N_SESSIONS, seed_base=33)
    cutoff = 15
    bars["XLRE"] = bars["XLRE"].iloc[cutoff:]
    state = build_sector_state(loader=make_loader(bars), **BUILD_KWARGS)

    xlre = state[state["sector_etf"] == "XLRE"].sort_values("date").reset_index(drop=True)
    before = xlre.iloc[:cutoff]
    assert before["excess_21d"].isna().all()
    assert before["above_20d"].isna().all()
    assert not (before["excess_21d"] == 0).any()
