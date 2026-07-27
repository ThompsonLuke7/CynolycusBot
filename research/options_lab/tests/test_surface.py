"""Tests for research/options_lab/surface.py.

Covers: building an IV surface from synthetic (but internally consistent,
priced-from-known-vol) contract bars/meta, ATM IV / term structure / skew
summaries, IV rank/percentile, the Yang-Zhang realized-vol estimator
against a synthetic series with known vol, and the asof lookahead guard
actually raising.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from research.options_lab import pricing, surface


UTC = "UTC"


def _mk_contract_bars_and_meta(asof: pd.Timestamp, spot: float, r: float, q: float,
                                iv_by_key: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a synthetic (osi_symbol -> price) contract bars/meta pair by
    PRICING each contract from a known IV via bsm_price, so build_iv_surface
    round-trips through implied_vol to (approximately) recover iv_by_key.

    iv_by_key: {(expiry_days, strike, right): sigma_true}
    """
    bars_rows = []
    meta_rows = []
    for (expiry_days, strike, right), sigma_true in iv_by_key.items():
        expiry_date = (asof + pd.Timedelta(days=expiry_days)).date()
        T = expiry_days / 365.25
        price = pricing.bsm_price(spot, strike, T, r, q, sigma_true, right)
        osi = f"TEST{expiry_date.strftime('%y%m%d')}{right}{int(strike * 1000):08d}"
        bars_rows.append({
            "osi_symbol": osi,
            "ts": asof,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 100.0,
            "trade_count": 10.0,
            "vwap": price,
        })
        meta_rows.append({
            "osi_symbol": osi,
            "ticker": "TEST",
            "expiry": expiry_date,
            "strike": strike,
            "right": right,
        })
    return pd.DataFrame(bars_rows), pd.DataFrame(meta_rows)


# --------------------------------------------------------------------------
# build_iv_surface
# --------------------------------------------------------------------------


def test_build_iv_surface_round_trips_known_vols():
    asof = pd.Timestamp("2026-06-01", tz=UTC)
    spot = 100.0
    r, q = 0.04, 0.0
    iv_by_key = {
        (30, 90.0, "P"): 0.35,
        (30, 100.0, "C"): 0.30,
        (30, 110.0, "C"): 0.32,
        (60, 100.0, "C"): 0.31,
    }
    bars, meta = _mk_contract_bars_and_meta(asof, spot, r, q, iv_by_key)
    result = surface.build_iv_surface(bars, meta, asof=asof, spot=spot, r=r, q=q)

    assert len(result) == len(iv_by_key)
    assert set(result.columns) == {
        "osi_symbol", "ticker", "expiry", "strike", "right", "T",
        "log_moneyness", "price", "iv",
    }
    for (expiry_days, strike, right), sigma_true in iv_by_key.items():
        expiry_date = (asof + pd.Timedelta(days=expiry_days)).date()
        row = result[
            (result["strike"] == strike)
            & (result["right"] == right)
            & (result["expiry"] == expiry_date)
        ].iloc[0]
        assert row["iv"] is not None
        assert row["iv"] == pytest.approx(sigma_true, rel=0.02, abs=1e-3)


def test_build_iv_surface_expired_contract_gets_none_iv():
    asof = pd.Timestamp("2026-06-01", tz=UTC)
    spot = 100.0
    bars = pd.DataFrame([{
        "osi_symbol": "EXPIRED1", "ts": asof, "open": 5.0, "high": 5.0, "low": 5.0,
        "close": 5.0, "volume": 1.0, "trade_count": 1.0, "vwap": 5.0,
    }])
    meta = pd.DataFrame([{
        "osi_symbol": "EXPIRED1", "ticker": "TEST", "expiry": (asof - pd.Timedelta(days=5)).date(),
        "strike": 100.0, "right": "C",
    }])
    result = surface.build_iv_surface(bars, meta, asof=asof, spot=spot, r=0.04, q=0.0)
    assert len(result) == 1
    assert result.iloc[0]["iv"] is None
    assert result.iloc[0]["T"] < 0


def test_build_iv_surface_missing_columns_raise():
    asof = pd.Timestamp("2026-06-01", tz=UTC)
    # contract_bars missing the required 'ts' column.
    bars_missing_ts = pd.DataFrame([{"osi_symbol": "X", "close": 1.0}])
    meta = pd.DataFrame([{"osi_symbol": "X", "ticker": "TEST", "expiry": asof.date(),
                           "strike": 100.0, "right": "C"}])
    with pytest.raises(ValueError):
        surface.build_iv_surface(bars_missing_ts, meta, asof=asof, spot=100.0, r=0.04)

    # contract_meta missing the required 'strike' column.
    bars = pd.DataFrame([{"osi_symbol": "X", "ts": asof, "close": 1.0}])
    meta_missing_strike = pd.DataFrame([{"osi_symbol": "X", "ticker": "TEST", "expiry": asof.date(),
                                          "right": "C"}])
    with pytest.raises(ValueError):
        surface.build_iv_surface(bars, meta_missing_strike, asof=asof, spot=100.0, r=0.04)


def test_build_iv_surface_lookahead_guard_raises():
    asof = pd.Timestamp("2026-06-01", tz=UTC)
    future_ts = asof + pd.Timedelta(days=1)
    bars = pd.DataFrame([{
        "osi_symbol": "X", "ts": future_ts, "open": 1.0, "high": 1.0, "low": 1.0,
        "close": 1.0, "volume": 1.0, "trade_count": 1.0, "vwap": 1.0,
    }])
    meta = pd.DataFrame([{"osi_symbol": "X", "ticker": "TEST", "expiry": (asof + pd.Timedelta(days=30)).date(),
                           "strike": 100.0, "right": "C"}])
    with pytest.raises(ValueError, match="lookahead"):
        surface.build_iv_surface(bars, meta, asof=asof, spot=100.0, r=0.04)


# --------------------------------------------------------------------------
# atm_iv_by_expiry / term_structure_slope / fit_skew
# --------------------------------------------------------------------------


def test_atm_iv_by_expiry_and_term_structure_slope():
    asof = pd.Timestamp("2026-06-01", tz=UTC)
    spot = 100.0
    r, q = 0.04, 0.0
    iv_by_key = {
        (30, 100.0, "C"): 0.30,
        (30, 100.0, "P"): 0.30,
        (90, 100.0, "C"): 0.40,
        (90, 100.0, "P"): 0.40,
    }
    bars, meta = _mk_contract_bars_and_meta(asof, spot, r, q, iv_by_key)
    surf = surface.build_iv_surface(bars, meta, asof=asof, spot=spot, r=r, q=q)
    atm = surface.atm_iv_by_expiry(surf)
    assert len(atm) == 2
    near = atm[atm["T"] < 0.2].iloc[0]
    far = atm[atm["T"] > 0.2].iloc[0]
    assert near["atm_iv"] == pytest.approx(0.30, abs=0.01)
    assert far["atm_iv"] == pytest.approx(0.40, abs=0.01)

    slope = surface.term_structure_slope(atm)
    assert slope > 0  # upward sloping: far expiry has higher IV


def test_term_structure_slope_requires_two_expiries():
    with pytest.raises(ValueError):
        surface.term_structure_slope(pd.DataFrame({"T": [0.1], "atm_iv": [0.3]}))


def test_fit_skew_recovers_put_skew_shape():
    asof = pd.Timestamp("2026-06-01", tz=UTC)
    spot = 100.0
    r, q = 0.04, 0.0
    # Put skew: lower strikes (k<0) have HIGHER iv than higher strikes (k>0).
    iv_by_key = {
        (30, 80.0, "P"): 0.50,
        (30, 90.0, "P"): 0.40,
        (30, 100.0, "C"): 0.30,
        (30, 110.0, "C"): 0.28,
        (30, 120.0, "C"): 0.27,
    }
    bars, meta = _mk_contract_bars_and_meta(asof, spot, r, q, iv_by_key)
    surf = surface.build_iv_surface(bars, meta, asof=asof, spot=spot, r=r, q=q)
    expiry = surf["expiry"].iloc[0]
    fit = surface.fit_skew(surf, expiry)
    assert fit["n_points"] == 5
    assert fit["b"] < 0  # negative slope -> put skew (higher iv at low k)


def test_fit_skew_requires_three_points():
    asof = pd.Timestamp("2026-06-01", tz=UTC)
    spot = 100.0
    r, q = 0.04, 0.0
    iv_by_key = {(30, 100.0, "C"): 0.30, (30, 110.0, "C"): 0.28}
    bars, meta = _mk_contract_bars_and_meta(asof, spot, r, q, iv_by_key)
    surf = surface.build_iv_surface(bars, meta, asof=asof, spot=spot, r=r, q=q)
    with pytest.raises(ValueError):
        surface.fit_skew(surf, surf["expiry"].iloc[0])


# --------------------------------------------------------------------------
# IV rank / percentile
# --------------------------------------------------------------------------


def _mk_history(dates: list[str], ivs: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates), "atm_iv": ivs})


def test_iv_rank_and_percentile_basic():
    dates = pd.date_range("2026-05-01", periods=20, freq="D")
    ivs = np.linspace(0.20, 0.50, 20)  # monotonically increasing, last = max
    history = pd.DataFrame({"date": dates, "atm_iv": ivs})
    asof = dates[-1]

    rank = surface.iv_rank(history, asof, lookback_days=30)
    pct = surface.iv_percentile(history, asof, lookback_days=30)
    assert rank == pytest.approx(100.0)
    assert pct == pytest.approx(100.0)

    # A U-shaped window (mid-window IV is the LOW, not the extreme) so the
    # rank/percentile at the midpoint is neither 0 nor 100 -- a real,
    # non-degenerate case. iv_rank's lookahead guard rejects any future
    # rows in the input frame, not just the trailing window, so slice to
    # <= asof first (matching how a live caller would only have data
    # through "today" available in the first place).
    u_shape = np.concatenate([np.linspace(0.50, 0.20, 10), np.linspace(0.20, 0.50, 10)])
    history_u = pd.DataFrame({"date": dates, "atm_iv": u_shape})
    asof_mid = dates[9]
    history_u_to_mid = history_u[history_u["date"] <= asof_mid]
    rank_mid = surface.iv_rank(history_u_to_mid, asof_mid, lookback_days=30)
    pct_mid = surface.iv_percentile(history_u_to_mid, asof_mid, lookback_days=30)
    assert rank_mid == pytest.approx(0.0, abs=1e-6)  # lowest value in the window so far
    assert pct_mid == pytest.approx(10.0, abs=1e-6)  # only itself (1 of 10 days) is <= current


def test_iv_rank_degenerate_flat_history_returns_50():
    dates = pd.date_range("2026-05-01", periods=10, freq="D")
    history = pd.DataFrame({"date": dates, "atm_iv": [0.30] * 10})
    rank = surface.iv_rank(history, dates[-1], lookback_days=30)
    assert rank == pytest.approx(50.0)


def test_iv_rank_requires_row_at_asof():
    history = _mk_history(["2026-05-01", "2026-05-02"], [0.3, 0.32])
    with pytest.raises(ValueError):
        surface.iv_rank(history, "2026-05-05", lookback_days=30)


def test_iv_rank_lookahead_guard_raises():
    history = _mk_history(["2026-05-01", "2026-05-02", "2026-05-03"], [0.3, 0.32, 0.31])
    with pytest.raises(ValueError, match="lookahead"):
        surface.iv_rank(history, "2026-05-02", lookback_days=30)


# --------------------------------------------------------------------------
# Realized vol estimators
# --------------------------------------------------------------------------


def _synthetic_daily_ohlc(n_days: int, sigma_true: float, seed: int = 7,
                           bars_per_day: int = 78) -> pd.DataFrame:
    """Simulate a continuous GBM path at intraday resolution (zero drift)
    and aggregate into daily OHLC bars with a genuine overnight gap (the
    path is continuous across the day boundary, so overnight variance is
    consistent with the same sigma_true) -- a fair test for Yang-Zhang.
    """
    rng = np.random.default_rng(seed)
    total_steps = n_days * bars_per_day
    dt = 1.0 / 252.0 / bars_per_day
    log_returns = rng.normal(loc=0.0, scale=sigma_true * math.sqrt(dt), size=total_steps)
    log_price = np.concatenate([[math.log(100.0)], math.log(100.0) + np.cumsum(log_returns)])
    price_path = np.exp(log_price)

    rows = []
    start = pd.Timestamp("2024-01-02", tz=UTC)
    for day in range(n_days):
        day_slice = price_path[day * bars_per_day: (day + 1) * bars_per_day + 1]
        o = float(day_slice[0])
        c = float(day_slice[-1])
        h = float(day_slice.max())
        lo = float(day_slice.min())
        rows.append({
            "symbol": "SYN",
            "timestamp": start + pd.Timedelta(days=day),
            "open": o, "high": h, "low": lo, "close": c,
            "volume": 1000.0, "trade_count": 100.0, "vwap": (o + c) / 2.0,
        })
    return pd.DataFrame(rows)


def test_yang_zhang_recovers_known_vol():
    sigma_true = 0.40
    bars = _synthetic_daily_ohlc(n_days=250, sigma_true=sigma_true, seed=11)
    asof = bars["timestamp"].iloc[-1]
    estimated = surface.yang_zhang_vol(bars, asof, window=249)
    assert estimated == pytest.approx(sigma_true, rel=0.20)


def test_close_to_close_recovers_known_vol():
    sigma_true = 0.25
    bars = _synthetic_daily_ohlc(n_days=250, sigma_true=sigma_true, seed=13)
    asof = bars["timestamp"].iloc[-1]
    estimated = surface.close_to_close_vol(bars, asof, window=249)
    assert estimated == pytest.approx(sigma_true, rel=0.20)


def test_yang_zhang_requires_enough_bars():
    bars = _synthetic_daily_ohlc(n_days=5, sigma_true=0.3)
    asof = bars["timestamp"].iloc[-1]
    with pytest.raises(ValueError):
        surface.yang_zhang_vol(bars, asof, window=20)


def test_realized_vol_lookahead_guard_raises():
    bars = _synthetic_daily_ohlc(n_days=30, sigma_true=0.3)
    asof = bars["timestamp"].iloc[10]  # earlier than the frame's max date
    with pytest.raises(ValueError, match="lookahead"):
        surface.yang_zhang_vol(bars, asof, window=5)
    with pytest.raises(ValueError, match="lookahead"):
        surface.close_to_close_vol(bars, asof, window=5)


# --------------------------------------------------------------------------
# iv_rv_premium
# --------------------------------------------------------------------------


def test_iv_rv_premium_basic():
    assert surface.iv_rv_premium(0.40, 0.20) == pytest.approx(2.0)


def test_iv_rv_premium_rejects_nonpositive_rv():
    with pytest.raises(ValueError):
        surface.iv_rv_premium(0.40, 0.0)
    with pytest.raises(ValueError):
        surface.iv_rv_premium(0.40, -0.1)
