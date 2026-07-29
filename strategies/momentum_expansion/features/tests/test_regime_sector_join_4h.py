"""
Tests for the WS-B market-regime / sector-context join in feature_matrix_4h.py
(docs/superpowers/plans/2026-07-26-market-regime-and-sector-context.md §4).

Covers:
  - the as-of join never uses a regime row whose `available_at` is after the
    bar timestamp (the core leakage hazard: a naive date-merge would leak)
  - `rs_sector_5`/`rs_sector_20` are byte-identical whether or not the regime
    tables are present (the old XLK-fallback sector logic must not change)
  - a missing regime table yields NaN + regime_available == 0, never a
    zero-fill
  - the new columns never enter FEATURE_COLUMNS_4H (plan D3)
  - the three interaction columns are arithmetically correct
  - `market_breadth_z` (the genuine cross-sector fraction, daily_regime's
    `breadth_z`) is ticker-invariant on a given bar, while `sector_above_20d`/
    `sector_above_50d` (renamed from the misleading sector_breadth_20/50 --
    they are one sector's own above/below-SMA state, not breadth) vary by
    the ticker's resolved sector
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.momentum_expansion.config.momentum_config import MIN_4H_BARS
from strategies.momentum_expansion.features import feature_matrix_4h as fm4h
from strategies.momentum_expansion.features.feature_matrix_4h import (
    FEATURE_COLUMNS_4H,
    REGIME_FEATURE_COLUMNS_4H,
    _asof_join_regime,
    build_ticker_features_4h,
)

N_BARS = MIN_4H_BARS + 30

REGIME_COLS = [
    "risk_appetite_z", "liquidity_stress_z", "credit_risk_z", "sector_dispersion_21d",
    "market_breadth_z",
]
SECTOR_COLS = [
    "sector_excess_21d", "sector_excess_63d", "sector_rank_21d", "sector_rank_63d",
    "sector_rs_accel", "sector_above_20d", "sector_above_50d",
]
INTERACTION_COLS = [
    "int_rs_spy_20_x_sector_rank_63d",
    "int_ret_20_x_risk_appetite_z",
    "int_breakout_20_x_liquidity_stress_z",
]


def _synthetic_4h_bars(seed: int, start: str = "2024-01-01", periods: int = N_BARS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="4h", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 1, periods))
    close = np.clip(close, 5, None)
    high = close + rng.uniform(0, 1, periods)
    low = close - rng.uniform(0, 1, periods)
    open_ = close + rng.normal(0, 0.3, periods)
    volume = rng.uniform(1e5, 1e6, periods)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def _calendar_and_available_at(bar_index: pd.DatetimeIndex):
    from signals.market_regime.timeutil import available_at_for_dates

    ny_dates = bar_index.tz_convert("America/New_York").normalize().tz_localize(None)
    calendar = pd.DatetimeIndex(sorted(pd.unique(ny_dates)))
    return calendar, available_at_for_dates(calendar, settle_margin_minutes=30)


def _synthetic_daily_regime_table(calendar: pd.DatetimeIndex, available_at: pd.DatetimeIndex, *, seed: int = 7):
    rng = np.random.default_rng(seed)
    n = len(calendar)
    return pd.DataFrame({
        "date": calendar,
        "available_at": available_at,
        "risk_appetite_z": rng.normal(0, 1, n),
        "liquidity_stress_z": rng.normal(0, 1, n),
        "credit_risk_z": rng.normal(0, 1, n),
        "sector_dispersion_raw": rng.uniform(0, 0.05, n),
        "breadth_z": rng.normal(0, 1, n),
    })


def _synthetic_sector_state_rows(
    calendar: pd.DatetimeIndex, available_at: pd.DatetimeIndex, *, sector_etf: str, seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(calendar)
    return pd.DataFrame({
        "date": calendar,
        "sector_etf": sector_etf,
        "available_at": available_at,
        "excess_21d": rng.normal(0, 0.05, n),
        "excess_63d": rng.normal(0, 0.05, n),
        "rank_21d": rng.uniform(0, 1, n),
        "rank_63d": rng.uniform(0, 1, n),
        "rs_accel": rng.normal(0, 0.02, n),
        "above_20d": rng.integers(0, 2, n).astype(float),
        "above_50d": rng.integers(0, 2, n).astype(float),
    })


def _synthetic_regime_and_sector_tables(bar_index: pd.DatetimeIndex, sector_etf: str = "XLK"):
    """Build daily_regime-shaped / sector_state-shaped tables covering bar_index's
    ET calendar-date range, with a real ET-close + 30min-settle `available_at`
    (reuses signals.market_regime.timeutil so the fixture matches production
    time semantics exactly, not a hand-rolled approximation)."""
    calendar, available_at = _calendar_and_available_at(bar_index)
    daily_regime = _synthetic_daily_regime_table(calendar, available_at, seed=7)
    sector_state = _synthetic_sector_state_rows(calendar, available_at, sector_etf=sector_etf, seed=7)
    return daily_regime, sector_state


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def test_asof_join_regime_never_uses_a_row_available_after_the_bar():
    """Core leakage hazard: a naive merge on calendar `date` would let a 09:30
    ET bar on session D see D's own (same-day) regime row, which isn't
    knowable until D's close + settle margin. The as-of join on `available_at`
    must instead fall back to D-1's row for that bar, and only pick up D's row
    once a later bar's timestamp has passed D's `available_at`."""
    from signals.market_regime.timeutil import available_at_for_dates

    d_minus_1 = pd.Timestamp("2026-07-08")  # Wed, firmly EDT
    d = pd.Timestamp("2026-07-09")          # Thu
    calendar = pd.DatetimeIndex([d_minus_1, d])
    available_at = available_at_for_dates(calendar, settle_margin_minutes=30)

    table = pd.DataFrame({"date": calendar, "available_at": available_at, "value": [1.0, 2.0]})

    morning_bar_d = pd.DatetimeIndex(
        [pd.Timestamp("2026-07-09 09:30", tz="America/New_York").tz_convert("UTC")]
    )
    afternoon_bar_d = pd.DatetimeIndex(
        [pd.Timestamp("2026-07-09 17:00", tz="America/New_York").tz_convert("UTC")]
    )

    morning_join = _asof_join_regime(morning_bar_d, table, ["value"])
    afternoon_join = _asof_join_regime(afternoon_bar_d, table, ["value"])

    # The same-day (D) row exists in the table but is not yet available at
    # 09:30 ET -- the morning bar must fall back to D-1's value, never D's.
    assert morning_join["value"].iloc[0] == 1.0
    assert pd.Timestamp(morning_join["_regime_date"].iloc[0]) == d_minus_1

    # By 17:00 ET, D's settle margin (16:00 close + 30min) has passed.
    assert afternoon_join["value"].iloc[0] == 2.0
    assert pd.Timestamp(afternoon_join["_regime_date"].iloc[0]) == d

    # Direct assertion of the actual invariant: the matched row's available_at
    # is never after the bar's own timestamp.
    for bar_ts, join in [(morning_bar_d[0], morning_join), (afternoon_bar_d[0], afternoon_join)]:
        matched_date = join["_regime_date"].iloc[0]
        matched_avail = table.loc[table["date"] == matched_date, "available_at"].iloc[0]
        assert matched_avail <= bar_ts


# ---------------------------------------------------------------------------
# Missing data -> NaN, not zero
# ---------------------------------------------------------------------------

def test_missing_regime_table_yields_nan_not_zero(monkeypatch):
    monkeypatch.setattr(fm4h, "_load_daily_regime_table", lambda: None)
    monkeypatch.setattr(fm4h, "_load_sector_state_table", lambda: None)

    bars = _synthetic_4h_bars(seed=1)
    feats = build_ticker_features_4h(ticker="AAPL", df_4h=bars, df_1d=None, ctx_4h={})
    assert feats is not None

    for col in REGIME_COLS + SECTOR_COLS + INTERACTION_COLS:
        assert col in feats.columns
        assert feats[col].isna().all(), f"{col} should be all-NaN when the regime table is missing"

    assert (feats["regime_available"] == 0.0).all()
    assert feats["regime_stale_days"].isna().all()


def test_unresolvable_sector_yields_nan_sector_columns_even_with_regime_present(monkeypatch):
    """A ticker absent from the curated SECTOR_MAP (and the empirical resolver
    is OFF by default) must get NaN sector_* features, not a silent XLK guess,
    even when the daily_regime table itself is present."""
    bars = _synthetic_4h_bars(seed=8)
    daily_regime, sector_state = _synthetic_regime_and_sector_tables(bars.index, sector_etf="XLK")
    monkeypatch.setattr(fm4h, "_load_daily_regime_table", lambda: daily_regime)
    monkeypatch.setattr(fm4h, "_load_sector_state_table", lambda: sector_state)

    feats = build_ticker_features_4h(ticker="ZZZZ_NOT_CURATED", df_4h=bars, df_1d=None, ctx_4h={})
    assert feats is not None
    for col in SECTOR_COLS:
        assert feats[col].isna().all()
    # But market-wide regime composites are unaffected by sector resolution.
    assert feats["regime_available"].sum() > 0


# ---------------------------------------------------------------------------
# rs_sector_5 / rs_sector_20 parity
# ---------------------------------------------------------------------------

def test_rs_sector_unchanged_by_regime_join(monkeypatch):
    bars = _synthetic_4h_bars(seed=4)
    ctx_4h = {"SPY": _synthetic_4h_bars(seed=5), "XLK": _synthetic_4h_bars(seed=6)}

    monkeypatch.setattr(fm4h, "_load_daily_regime_table", lambda: None)
    monkeypatch.setattr(fm4h, "_load_sector_state_table", lambda: None)
    feats_no_regime = build_ticker_features_4h(
        ticker="AAPL", df_4h=bars.copy(), df_1d=None, ctx_4h=ctx_4h,
    )

    daily_regime, sector_state = _synthetic_regime_and_sector_tables(bars.index, sector_etf="XLK")
    monkeypatch.setattr(fm4h, "_load_daily_regime_table", lambda: daily_regime)
    monkeypatch.setattr(fm4h, "_load_sector_state_table", lambda: sector_state)
    feats_with_regime = build_ticker_features_4h(
        ticker="AAPL", df_4h=bars.copy(), df_1d=None, ctx_4h=ctx_4h,
    )

    assert feats_no_regime["rs_sector_5"].notna().any()  # sanity: not a vacuous NaN==NaN check
    pd.testing.assert_series_equal(feats_no_regime["rs_sector_5"], feats_with_regime["rs_sector_5"])
    pd.testing.assert_series_equal(feats_no_regime["rs_sector_20"], feats_with_regime["rs_sector_20"])


# ---------------------------------------------------------------------------
# Column-list hygiene (plan D3)
# ---------------------------------------------------------------------------

def test_regime_columns_not_in_training_feature_list():
    assert set(REGIME_FEATURE_COLUMNS_4H).isdisjoint(set(FEATURE_COLUMNS_4H))
    assert len(REGIME_FEATURE_COLUMNS_4H) == len(set(REGIME_FEATURE_COLUMNS_4H))


# ---------------------------------------------------------------------------
# Interaction arithmetic
# ---------------------------------------------------------------------------

def test_regime_columns_populate_and_interactions_are_arithmetically_correct(monkeypatch):
    bars = _synthetic_4h_bars(seed=2)
    ctx_4h = {"SPY": _synthetic_4h_bars(seed=3)}
    daily_regime, sector_state = _synthetic_regime_and_sector_tables(bars.index, sector_etf="XLK")
    monkeypatch.setattr(fm4h, "_load_daily_regime_table", lambda: daily_regime)
    monkeypatch.setattr(fm4h, "_load_sector_state_table", lambda: sector_state)

    feats = build_ticker_features_4h(ticker="AAPL", df_4h=bars, df_1d=None, ctx_4h=ctx_4h)
    assert feats is not None

    # Most bars should resolve a regime row (calendar padding covers ~all of
    # bar_index's date range; only the very first session's early-morning
    # bars, before any prior row exists, are legitimately unresolved).
    assert feats["regime_available"].isin([0.0, 1.0]).all()
    assert (feats["regime_available"] == 1.0).mean() > 0.9
    assert feats["rs_spy_20"].notna().any()

    expected_1 = feats["rs_spy_20"] * feats["sector_rank_63d"]
    expected_2 = feats["ret_20"] * feats["risk_appetite_z"]
    expected_3 = feats["breakout_20"] * feats["liquidity_stress_z"]
    pd.testing.assert_series_equal(
        feats["int_rs_spy_20_x_sector_rank_63d"], expected_1, check_names=False)
    pd.testing.assert_series_equal(
        feats["int_ret_20_x_risk_appetite_z"], expected_2, check_names=False)
    pd.testing.assert_series_equal(
        feats["int_breakout_20_x_liquidity_stress_z"], expected_3, check_names=False)

    assert feats["sector_above_20d"].dropna().isin([0.0, 1.0]).all()
    assert feats["sector_above_50d"].dropna().isin([0.0, 1.0]).all()


# ---------------------------------------------------------------------------
# market_breadth_z (genuine cross-sector fraction) vs sector_above_20d/50d
# (per-ticker, sector-varying) taxonomy
# ---------------------------------------------------------------------------

def test_market_breadth_z_is_ticker_invariant_while_sector_above_20d_varies(monkeypatch):
    """The exact distinction the rename exists to make explicit: on the same
    bar, two tickers in different sectors must see the SAME market_breadth_z
    (it's daily_regime's market-wide breadth_z, joined once per bar,
    independent of ticker/sector) but CAN see different sector_above_20d/50d
    (each ticker's own resolved sector ETF's above/below-SMA state)."""
    bars = _synthetic_4h_bars(seed=9)
    calendar, available_at = _calendar_and_available_at(bars.index)
    daily_regime = _synthetic_daily_regime_table(calendar, available_at, seed=11)
    # Two sectors seeded differently so above_20d/above_50d actually diverge
    # (a shared seed would make them coincidentally identical, defeating the
    # point of the assertion).
    sector_state = pd.concat([
        _synthetic_sector_state_rows(calendar, available_at, sector_etf="XLK", seed=101),
        _synthetic_sector_state_rows(calendar, available_at, sector_etf="XLF", seed=202),
    ], ignore_index=True)

    monkeypatch.setattr(fm4h, "_load_daily_regime_table", lambda: daily_regime)
    monkeypatch.setattr(fm4h, "_load_sector_state_table", lambda: sector_state)

    feats_xlk = build_ticker_features_4h(ticker="AAPL", df_4h=bars.copy(), df_1d=None, ctx_4h={})  # -> XLK
    feats_xlf = build_ticker_features_4h(ticker="JPM", df_4h=bars.copy(), df_1d=None, ctx_4h={})   # -> XLF
    assert fm4h.SECTOR_MAP["AAPL"] == "XLK"
    assert fm4h.SECTOR_MAP["JPM"] == "XLF"

    # Market-wide: byte-identical across tickers on every bar.
    pd.testing.assert_series_equal(
        feats_xlk["market_breadth_z"], feats_xlf["market_breadth_z"], check_names=False)

    # Sector-level: genuinely varies -- not a vacuous check.
    assert not feats_xlk["sector_above_20d"].equals(feats_xlf["sector_above_20d"])
    assert not feats_xlk["sector_above_50d"].equals(feats_xlf["sector_above_50d"])
