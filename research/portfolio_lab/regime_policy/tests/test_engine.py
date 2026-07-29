"""Tests for the regime-conditional policy engine.

Synthetic fixtures only (no disk/network dependency), matching the pattern
in ``research/portfolio_lab/tests/test_portfolio_backtest.py``. The primary
requirement under test (per the task's verification section): a decision at
signal_ts=t must never change when the regime table is extended with rows
whose ``available_at > t``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.momentum_expansion.features.feature_matrix_4h import _asof_join_regime

from research.portfolio_lab.regime_policy import engine as E
from research.portfolio_lab.regime_policy import rules as R


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

class _FakeBarCache:
    def __init__(self, bars_by_ticker: dict):
        self._bars = bars_by_ticker

    def get(self, ticker):
        return self._bars.get(ticker)


def _make_bars(start="2025-06-02", periods=60, price0=100.0, vol=0.01, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=periods, freq="4h", tz="UTC")
    rets = rng.normal(0.0, vol, periods)
    close = price0 * np.cumprod(1 + rets)
    open_ = np.roll(close, 1)
    open_[0] = price0
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    atr = np.full(periods, price0 * vol * 2)
    return {"ts": ts.asi8, "ts_dt": ts.to_numpy(), "open": open_.astype(float),
            "high": high.astype(float), "low": low.astype(float), "close": close.astype(float), "atr": atr}


def _regime_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def _signal_frame(rows):
    df = pd.DataFrame(rows, columns=["timestamp", "ticker", "score"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["direction"] = 1
    return df


# ---------------------------------------------------------------------------
# Causality: a candidate's admission at signal_ts=t must not depend on any
# regime row whose available_at > t.
# ---------------------------------------------------------------------------

def test_asof_join_plus_rule_admission_is_causal_to_future_regime_rows():
    tickers = ["AAA", "BBB"]
    bars = {t: _make_bars(seed=i) for i, t in enumerate(tickers)}
    bar_cache = _FakeBarCache(bars)
    ts0 = pd.Timestamp("2025-06-02 14:00", tz="UTC")
    sig = _signal_frame([(ts0, "AAA", 0.9), (ts0, "BBB", 0.8)])
    candidates = E.resolve_fixed_stop_candidates(_pack(sig), bar_cache, top_k=E.WIDE_TOP_K)
    assert len(candidates) == 2

    # table known "as of" ts0: liquidity_stress_z = 0.2 (below the H1-liq-1.0
    # threshold) on the settlement date at/just before ts0.
    table_known = _regime_table([
        {"date": "2025-05-30", "available_at": "2025-05-30T20:15:00Z",
         "risk_appetite_z": 0.1, "liquidity_stress_z": 0.2, "sector_dispersion_z": 0.0,
         "spy_rv20_z": 0.0, "spy_trend_state": 1.0},
    ])
    joined_known = _asof_join_regime(pd.DatetimeIndex(candidates["signal_ts"]), table_known, E.REGIME_COLS)
    cand_known = candidates.copy()
    for c in E.REGIME_COLS:
        cand_known[c] = joined_known[c].to_numpy()

    rule = R.ADMISSION_SIZING_RULES[0]  # H1-liq-1.0
    assert rule.rule_id == "H1-liq-1.0"
    trades_known, _ = E.apply_rule(cand_known, rule)
    assert len(trades_known) == 2  # liquidity_stress_z=0.2 < 1.0 -> both admitted

    # NOW append a FUTURE row (available_at strictly after ts0) that, if it
    # leaked backward, would flip the gate (liquidity_stress_z=5.0 >> 1.0).
    table_with_future = pd.concat([table_known, _regime_table([
        {"date": "2025-06-05", "available_at": "2025-06-05T20:15:00Z",
         "risk_appetite_z": -3.0, "liquidity_stress_z": 5.0, "sector_dispersion_z": 3.0,
         "spy_rv20_z": 3.0, "spy_trend_state": -1.0},
    ])], ignore_index=True)
    joined_future = _asof_join_regime(pd.DatetimeIndex(candidates["signal_ts"]), table_with_future, E.REGIME_COLS)
    cand_future = candidates.copy()
    for c in E.REGIME_COLS:
        cand_future[c] = joined_future[c].to_numpy()

    trades_future, _ = E.apply_rule(cand_future, rule)
    assert len(trades_future) == 2, "future regime row leaked backward into an entry-day gate decision"
    assert [t.dollar_size for t in trades_known] == [t.dollar_size for t in trades_future]


def _pack(sig_df: pd.DataFrame) -> pd.DataFrame:
    return sig_df[["timestamp", "ticker", "score"]]


# ---------------------------------------------------------------------------
# apply_rule: baseline == rank<=10 subset of the same resolved stream
# ---------------------------------------------------------------------------

def test_baseline_rule_is_rank_le_10_subset():
    tickers = [f"T{i}" for i in range(15)]
    bars = {t: _make_bars(seed=i) for i, t in enumerate(tickers)}
    bar_cache = _FakeBarCache(bars)
    ts0 = pd.Timestamp("2025-06-02 14:00", tz="UTC")
    sig = _signal_frame([(ts0, t, 1.0 - i * 0.01) for i, t in enumerate(tickers)])
    candidates = E.resolve_fixed_stop_candidates(_pack(sig), bar_cache, top_k=E.WIDE_TOP_K)
    assert len(candidates) == 15
    for c in E.REGIME_COLS:
        candidates[c] = 0.0  # neutral regime, irrelevant to this test

    trades, _ = E.apply_rule(candidates, E.baseline_rule())
    assert len(trades) == 10
    assert set(t.ticker for t in trades) == set(tickers[:10])


def test_h1_gate_suspends_entries_above_threshold():
    tickers = ["AAA"]
    bars = {t: _make_bars(seed=0) for t in tickers}
    bar_cache = _FakeBarCache(bars)
    ts0 = pd.Timestamp("2025-06-02 14:00", tz="UTC")
    sig = _signal_frame([(ts0, "AAA", 1.0)])
    candidates = E.resolve_fixed_stop_candidates(_pack(sig), bar_cache, top_k=E.WIDE_TOP_K)
    candidates["liquidity_stress_z"] = 2.0  # above every H1-liq threshold
    for c in ["risk_appetite_z", "sector_dispersion_z", "spy_rv20_z", "spy_trend_state"]:
        candidates[c] = 0.0

    rule = next(r for r in R.ADMISSION_SIZING_RULES if r.rule_id == "H1-liq-1.0")
    trades, _ = E.apply_rule(candidates, rule)
    assert trades == []


# ---------------------------------------------------------------------------
# already-held dedupe preserved in the regime-conditional overlay
# ---------------------------------------------------------------------------

def test_already_held_dedupe_preserved():
    ts0 = pd.Timestamp("2025-06-02 14:00", tz="UTC")
    ts1 = pd.Timestamp("2025-06-02 18:00", tz="UTC")
    bars = {"AAA": _make_bars(seed=0)}
    bar_cache = _FakeBarCache(bars)
    sig = _signal_frame([(ts0, "AAA", 0.9), (ts1, "AAA", 0.8)])
    candidates = E.resolve_fixed_stop_candidates(_pack(sig), bar_cache, top_k=E.WIDE_TOP_K)
    assert len(candidates) == 2
    for c in E.REGIME_COLS:
        candidates[c] = 0.0
    trades, _ = E.apply_rule(candidates, E.baseline_rule())
    assert len(trades) == 1


# ---------------------------------------------------------------------------
# weekly_diff_bootstrap sanity
# ---------------------------------------------------------------------------

def test_weekly_diff_bootstrap_zero_when_identical():
    ts0 = pd.Timestamp("2025-06-02 14:00", tz="UTC")
    bars = {"AAA": _make_bars(seed=0)}
    bar_cache = _FakeBarCache(bars)
    sig = _signal_frame([(ts0, "AAA", 1.0)])
    candidates = E.resolve_fixed_stop_candidates(_pack(sig), bar_cache, top_k=E.WIDE_TOP_K)
    for c in E.REGIME_COLS:
        candidates[c] = 0.0
    trades, _ = E.apply_rule(candidates, E.baseline_rule())
    result = E.weekly_diff_bootstrap(trades, trades)
    assert result["point"] == pytest.approx(0.0)


def test_weekly_diff_bootstrap_empty_when_no_trades():
    result = E.weekly_diff_bootstrap([], [])
    assert result["n_weeks"] == 0
    assert np.isnan(result["point"])
