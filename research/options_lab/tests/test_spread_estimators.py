"""Tests for research/options_lab/spread_estimators.py.

No network: every estimator here is a pure function over synthetic price/bar
series with a KNOWN injected spread, or `combine_spread_estimates`'s pure
fallback-ladder logic, or `estimate_contract_spread` with `chain_cache`
monkeypatched to synthetic frames (no real HTTP call). Covers:
  - Roll recovers an injected bid/ask-bounce spread within tolerance.
  - Roll returns None (never 0, never abs()) on non-negative serial
    covariance.
  - Corwin-Schultz recovers an injected spread from synthetic high/low bars
    within tolerance, and its negative-window-floored-at-zero convention.
  - Price clustering recovers an exact two-level injected spread.
  - The fallback ladder picks the first available method in the documented
    order and reports it via `.method`.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from research.options_lab import fills, spread_estimators as se


# --------------------------------------------------------------------------
# Roll (1984)
# --------------------------------------------------------------------------

def test_roll_recovers_known_bounce_spread():
    # Classic Roll setup: price = fixed mid +/- half-spread * iid +-1 trade
    # direction. No drift, so the ONLY source of serial covariance is the
    # bid/ask bounce -> should recover close to the true round-trip spread.
    rng = np.random.RandomState(1)
    mid = 100.0
    half_spread = 0.10
    true_spread_pct = (2 * half_spread) / mid
    q = rng.choice([-1, 1], size=5000)
    prices = mid + half_spread * q

    result = se.roll_effective_spread_pct(prices.tolist())
    assert result is not None
    assert result == pytest.approx(true_spread_pct, rel=0.15)


def test_roll_recovers_known_spread_at_different_price_level():
    rng = np.random.RandomState(7)
    mid = 12.50
    half_spread = 0.05
    true_spread_pct = (2 * half_spread) / mid
    q = rng.choice([-1, 1], size=4000)
    prices = mid + half_spread * q

    result = se.roll_effective_spread_pct(prices.tolist())
    assert result is not None
    assert result == pytest.approx(true_spread_pct, rel=0.2)


def test_roll_returns_none_on_positive_serial_covariance():
    # Momentum-block price changes ([5,5,1,1] repeating): consecutive diffs
    # cluster together (big-follows-big, small-follows-small) -> positive
    # lag-1 covariance, the opposite signature of bid/ask bounce.
    diffs = np.array([5, 5, 1, 1] * 6, dtype=float)
    prices = 100.0 + np.concatenate([[0.0], np.cumsum(diffs)])
    cov = float(np.cov(np.diff(prices)[1:], np.diff(prices)[:-1], ddof=1)[0, 1])
    assert cov > 0  # sanity: this construction really is positive-covariance

    assert se.roll_effective_spread_pct(prices.tolist()) is None


def test_roll_returns_none_never_abs_or_clamped():
    # Regression guard: a positive-covariance series must come back None,
    # not abs(2*sqrt(cov)) or 0.0 -- both would fabricate a number.
    diffs = np.array([5, 5, 1, 1] * 6, dtype=float)
    prices = (100.0 + np.concatenate([[0.0], np.cumsum(diffs)])).tolist()
    result = se.roll_effective_spread_pct(prices)
    assert result is None
    assert result != 0.0


def test_roll_returns_none_with_too_few_prints():
    assert se.roll_effective_spread_pct([100.0, 100.1, 99.9]) is None


def test_roll_returns_none_on_empty_or_degenerate_input():
    assert se.roll_effective_spread_pct([]) is None
    assert se.roll_effective_spread_pct([100.0] * 20) is None  # zero variance -> cov==0 -> not negative


# --------------------------------------------------------------------------
# Corwin & Schultz (2012)
# --------------------------------------------------------------------------

def _simulate_cs_bars(true_spread_pct: float, *, n_bars: int, seed: int, sigma_tick: float = 0.0005):
    """Synthetic intraday bars: mid follows a small random walk, each print
    within a bar bounces +/- true_spread_pct/2 around the current mid; the
    bar's high/low is the max/min of its prints. This is the standard way
    the Corwin-Schultz literature validates the estimator against a KNOWN
    injected spread."""
    rng = np.random.RandomState(seed)
    mid = 100.0
    highs, lows = [], []
    for _ in range(n_bars):
        ticks = []
        for _ in range(100):
            mid *= 1 + rng.normal(0, sigma_tick)
            q = rng.choice([-1, 1])
            ticks.append(mid * (1 + (true_spread_pct / 2) * q))
        highs.append(max(ticks))
        lows.append(min(ticks))
    return highs, lows


def test_corwin_schultz_recovers_known_spread():
    true_spread_pct = 0.02
    highs, lows = _simulate_cs_bars(true_spread_pct, n_bars=40, seed=42)

    result = se.corwin_schultz_spread_pct(highs, lows)
    assert result is not None
    assert result == pytest.approx(true_spread_pct, abs=0.008)


def test_corwin_schultz_recovers_known_spread_wider_market():
    true_spread_pct = 0.08
    highs, lows = _simulate_cs_bars(true_spread_pct, n_bars=40, seed=123)

    result = se.corwin_schultz_spread_pct(highs, lows)
    assert result is not None
    assert result == pytest.approx(true_spread_pct, rel=0.35)


def test_corwin_schultz_negative_windows_floored_at_zero_not_discarded():
    # Two bars with essentially identical, tiny high-low range (near-zero
    # diffusion, near-zero implied spread) should push alpha negative in at
    # least one window. Confirm the estimator does not raise / does not go
    # negative, and stays finite (the floor-at-zero convention keeps the
    # mean well-defined instead of blowing up or being undefined).
    highs = [100.01, 100.01, 100.02, 100.01, 100.015]
    lows = [100.00, 100.00, 100.00, 100.00, 100.005]
    result = se.corwin_schultz_spread_pct(highs, lows, min_windows=1)
    assert result is not None
    assert result >= 0.0


def test_corwin_schultz_returns_none_with_too_few_windows():
    assert se.corwin_schultz_spread_pct([100.0], [99.0]) is None
    assert se.corwin_schultz_spread_pct([100.0, 100.5], [99.5, 99.8]) is None  # only 1 window, min is 3


def test_corwin_schultz_returns_none_on_mismatched_lengths():
    assert se.corwin_schultz_spread_pct([100.0, 101.0, 102.0], [99.0, 99.5]) is None


def test_corwin_schultz_skips_invalid_bars_without_raising():
    # A bad bar (high < low) inside an otherwise-valid series should be
    # skipped, not raise or poison the whole estimate.
    highs = [100.0, 99.0, 101.0, 102.0, 103.0]  # index 1: high < low below
    lows = [99.5, 99.5, 99.8, 100.0, 100.5]
    result = se.corwin_schultz_spread_pct(highs, lows, min_windows=1)
    assert result is not None and math.isfinite(result)


# --------------------------------------------------------------------------
# Price clustering
# --------------------------------------------------------------------------

def test_clustering_recovers_exact_two_level_spread():
    rng = np.random.RandomState(3)
    bid, ask = 99.95, 100.05
    true_spread_pct = (ask - bid) / 100.0
    prices = [bid if x else ask for x in rng.choice([True, False], size=50)]

    result = se.price_clustering_spread_pct(prices)
    assert result is not None
    assert result == pytest.approx(true_spread_pct, rel=0.02)


def test_clustering_returns_none_with_single_price_level():
    assert se.price_clustering_spread_pct([100.0] * 20) is None


def test_clustering_returns_none_with_too_few_trades():
    assert se.price_clustering_spread_pct([100.0, 100.1, 99.9]) is None


# --------------------------------------------------------------------------
# Fallback ladder
# --------------------------------------------------------------------------

def test_ladder_prefers_roll_when_available():
    est = se.combine_spread_estimates(
        roll_pct=0.05, corwin_schultz_pct=0.08, clustering_pct=0.10, regression_pct=0.20
    )
    assert est.method == "roll"
    assert est.spread_pct == pytest.approx(0.05)


def test_ladder_falls_back_to_corwin_schultz_when_roll_none():
    est = se.combine_spread_estimates(
        roll_pct=None, corwin_schultz_pct=0.08, clustering_pct=0.10, regression_pct=0.20
    )
    assert est.method == "corwin_schultz"
    assert est.spread_pct == pytest.approx(0.08)


def test_ladder_falls_back_to_clustering_when_roll_and_cs_none():
    est = se.combine_spread_estimates(
        roll_pct=None, corwin_schultz_pct=None, clustering_pct=0.10, regression_pct=0.20
    )
    assert est.method == "clustering"
    assert est.spread_pct == pytest.approx(0.10)


def test_ladder_falls_back_to_regression_as_last_resort():
    est = se.combine_spread_estimates(
        roll_pct=None, corwin_schultz_pct=None, clustering_pct=None, regression_pct=0.20
    )
    assert est.method == "regression"
    assert est.spread_pct == pytest.approx(0.20)


def test_ladder_returns_none_method_when_nothing_available():
    est = se.combine_spread_estimates(
        roll_pct=None, corwin_schultz_pct=None, clustering_pct=None, regression_pct=None
    )
    assert est.method == "none"
    assert est.spread_pct is None


def test_ladder_skips_non_positive_or_non_finite_candidates():
    # A zero, negative, or NaN candidate must be treated the same as None --
    # never selected as "the" estimate.
    est = se.combine_spread_estimates(
        roll_pct=0.0, corwin_schultz_pct=-0.02, clustering_pct=float("nan"), regression_pct=0.20
    )
    assert est.method == "regression"
    assert est.spread_pct == pytest.approx(0.20)


# --------------------------------------------------------------------------
# estimate_contract_spread orchestration (chain_cache monkeypatched -- no
# network call actually happens)
# --------------------------------------------------------------------------

def test_estimate_contract_spread_wires_trades_and_bars_through_ladder(monkeypatch):
    sym = "SPY260501C00724000"

    rng = np.random.RandomState(9)
    mid, half_spread = 1.00, 0.03
    q = rng.choice([-1, 1], size=200)
    trade_prices = mid + half_spread * q
    trades_df = pd.DataFrame({
        "osi_symbol": [sym] * len(trade_prices),
        "t": pd.date_range("2026-05-01T14:00:00Z", periods=len(trade_prices), freq="s"),
        "p": trade_prices,
        "s": [1] * len(trade_prices),
    })

    def fake_fetch_trades(symbols, start, end, *, env_file=".env"):
        assert symbols == [sym]
        return trades_df

    def fake_fetch_bars(symbols, timeframe, start, end, *, env_file=".env"):
        assert timeframe == "30Min"
        return pd.DataFrame(columns=["osi_symbol", "t", "o", "h", "l", "c", "v", "n", "vw"])

    monkeypatch.setattr(se.chain_cache, "fetch_trades", fake_fetch_trades)
    monkeypatch.setattr(se.chain_cache, "fetch_bars", fake_fetch_bars)

    est = se.estimate_contract_spread(
        sym, "2026-05-01T14:00:00Z", "2026-05-01T14:05:00Z",
        moneyness=0.01, dte=0, oi=1000, volume=500, underlying_adv=1_000_000,
    )
    # Bars are empty (no CS candidate) but trades carry a clean bounce
    # signature -> Roll should win over the regression fallback.
    assert est.method == "roll"
    assert est.spread_pct == pytest.approx((2 * half_spread) / mid, rel=0.2)


def test_estimate_contract_spread_falls_back_to_regression_with_no_data(monkeypatch):
    sym = "AAPL260501C00200000"

    def empty_trades(symbols, start, end, *, env_file=".env"):
        return pd.DataFrame(columns=["osi_symbol", "t", "x", "p", "s", "c"])

    def empty_bars(symbols, timeframe, start, end, *, env_file=".env"):
        return pd.DataFrame(columns=["osi_symbol", "t", "o", "h", "l", "c", "v", "n", "vw"])

    monkeypatch.setattr(se.chain_cache, "fetch_trades", empty_trades)
    monkeypatch.setattr(se.chain_cache, "fetch_bars", empty_bars)

    est = se.estimate_contract_spread(
        sym, "2026-05-01T14:00:00Z", "2026-05-01T14:05:00Z",
        moneyness=0.02, dte=5, oi=100, volume=50, underlying_adv=1_000_000,
    )
    assert est.method == "regression"
    expected = fills.estimate_spread(0.02, 5, 100, 50, 1_000_000)
    assert est.spread_pct == pytest.approx(expected)


def test_estimate_contract_spread_none_when_everything_unmeasurable(monkeypatch):
    sym = "AAPL260501C00200000"

    def empty_trades(symbols, start, end, *, env_file=".env"):
        return pd.DataFrame(columns=["osi_symbol", "t", "x", "p", "s", "c"])

    def empty_bars(symbols, timeframe, start, end, *, env_file=".env"):
        return pd.DataFrame(columns=["osi_symbol", "t", "o", "h", "l", "c", "v", "n", "vw"])

    monkeypatch.setattr(se.chain_cache, "fetch_trades", empty_trades)
    monkeypatch.setattr(se.chain_cache, "fetch_bars", empty_bars)

    # No liquidity features given -> regression also returns None.
    est = se.estimate_contract_spread(sym, "2026-05-01T14:00:00Z", "2026-05-01T14:05:00Z")
    assert est.method == "none"
    assert est.spread_pct is None


# --------------------------------------------------------------------------
# fills.py hook
# --------------------------------------------------------------------------

def test_fills_hook_delegates_to_spread_estimators(monkeypatch):
    sym = "AAPL260501C00200000"

    def empty_trades(symbols, start, end, *, env_file=".env"):
        return pd.DataFrame(columns=["osi_symbol", "t", "x", "p", "s", "c"])

    def empty_bars(symbols, timeframe, start, end, *, env_file=".env"):
        return pd.DataFrame(columns=["osi_symbol", "t", "o", "h", "l", "c", "v", "n", "vw"])

    monkeypatch.setattr(se.chain_cache, "fetch_trades", empty_trades)
    monkeypatch.setattr(se.chain_cache, "fetch_bars", empty_bars)

    result = fills.estimate_spread_empirical(
        sym, "2026-05-01T14:00:00Z", "2026-05-01T14:05:00Z",
        moneyness=0.02, dte=5, oi=100, volume=50, underlying_adv=1_000_000,
    )
    assert result.method == "regression"
    assert result.spread_pct == pytest.approx(fills.estimate_spread(0.02, 5, 100, 50, 1_000_000))
