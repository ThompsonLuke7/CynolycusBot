"""Integration tests for research/portfolio_lab/portfolio_backtest.py.

Uses fully synthetic 4H bars (a lightweight stand-in for
family_backtest.BarCache, duck-typed to the same `.get(ticker)` interface)
and a synthetic DailyContext (pre-populated in-memory, bypassing load_1d) so
these tests need no real market data on disk. Covers: the "already held"
dedupe, the target_concurrent slot cap (present for equal-weight/cov-vol-
target, absent for fixed-notional -- matching the live baseline), per-sector
cap propagation through the full admission loop, and the causality guarantee
at the engine level (sizing at signal_ts must not change when the daily
context is later extended with FUTURE sessions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.portfolio_lab import portfolio_backtest as pb
from research.portfolio_lab import sizing as sz
from research.portfolio_lab.daily_context import DailyContext


# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------

class _FakeBarCache:
    """Duck-types family_backtest.BarCache without touching disk."""

    def __init__(self, bars_by_ticker: dict):
        self._bars = bars_by_ticker

    def get(self, ticker):
        return self._bars.get(ticker)


def _make_bars(start="2025-06-02", periods=60, price0=100.0, drift=0.0, vol=0.01, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=periods, freq="4h", tz="UTC")
    rets = rng.normal(drift, vol, periods)
    close = price0 * np.cumprod(1 + rets)
    open_ = np.roll(close, 1)
    open_[0] = price0
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    atr = np.full(periods, price0 * vol * 2)  # constant ATR, big enough to avoid same-bar TP/SL
    return {
        "ts": ts.asi8, "ts_dt": ts.to_numpy(),
        "open": open_.astype(float), "high": high.astype(float),
        "low": low.astype(float), "close": close.astype(float), "atr": atr,
    }


def _make_daily_context(tickers, start="2025-01-02", periods=250, price0=100.0, vol=0.015, seed=1):
    """Populate a DailyContext in-memory (no load_1d / no disk I/O)."""
    ctx = DailyContext(tickers=list(tickers))
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=periods, tz="UTC").normalize()
    for i, t in enumerate(tickers):
        rets = rng.normal(0, vol, periods)
        close = price0 * np.cumprod(1 + rets)
        vol_shares = rng.integers(1_000_000, 5_000_000, periods).astype(float)
        ctx._close[t] = pd.Series(close, index=dates)
        ctx._dollar_vol[t] = pd.Series(close * vol_shares, index=dates)
        ctx._loaded.add(t)
    return ctx


def _signal_frame(rows):
    """rows: list of (timestamp, ticker, score). direction always +1 (long-only, momentum)."""
    df = pd.DataFrame(rows, columns=["timestamp", "ticker", "score"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["direction"] = 1
    return df


# --------------------------------------------------------------------------
# Dedupe ("already held")
# --------------------------------------------------------------------------

def test_already_held_dedupe_across_all_policies():
    tickers = ["AAA", "BBB"]
    bars = {t: _make_bars(seed=i) for i, t in enumerate(tickers)}
    bar_cache = _FakeBarCache(bars)

    ts0 = pd.Timestamp("2025-06-02 14:00", tz="UTC")
    ts1 = pd.Timestamp("2025-06-02 18:00", tz="UTC")  # re-signaled one bar later, same ticker
    sig = _signal_frame([(ts0, "AAA", 0.9), (ts1, "AAA", 0.8)])
    candidates = pb._resolve_candidates(sig, bar_cache, tp_mult=5.0, sl_mult=5.0, max_hold=20)
    assert len(candidates) == 2  # both resolve to trades

    ctx = _make_daily_context(tickers)
    for policy in [
        pb.PolicyConfig(name="fixed", fixed_notional=5000.0),
        pb.PolicyConfig(name="ew", equal_weight=True, target_concurrent=10),
        pb.PolicyConfig(name="cov", sizing_cfg=sz.SizingConfig(target_concurrent=10), target_concurrent=10),
    ]:
        trades, _ = pb.run_policy(candidates, policy=policy, daily_ctx=ctx, cost_bps_round_trip=0.0)
        aaa_trades = [t for t in trades if t.ticker == "AAA"]
        assert len(aaa_trades) == 1, f"{policy.name}: expected exactly one AAA entry (dedupe), got {len(aaa_trades)}"


# --------------------------------------------------------------------------
# target_concurrent slot cap
# --------------------------------------------------------------------------

def test_slot_cap_enforced_for_constrained_policies_not_fixed_notional():
    tickers = [f"T{i}" for i in range(6)]
    bars = {t: _make_bars(seed=i) for i, t in enumerate(tickers)}
    bar_cache = _FakeBarCache(bars)
    ts0 = pd.Timestamp("2025-06-02 14:00", tz="UTC")
    sig = _signal_frame([(ts0, t, 1.0 - i * 0.01) for i, t in enumerate(tickers)])
    candidates = pb._resolve_candidates(sig, bar_cache, tp_mult=5.0, sl_mult=5.0, max_hold=20)
    assert len(candidates) == 6

    ctx = _make_daily_context(tickers)

    fixed = pb.PolicyConfig(name="fixed", fixed_notional=5000.0)  # no slot cap
    trades_fixed, _ = pb.run_policy(candidates, policy=fixed, daily_ctx=ctx, cost_bps_round_trip=0.0)
    assert len(trades_fixed) == 6  # unconstrained -- all six admitted

    ew = pb.PolicyConfig(name="ew", equal_weight=True, target_concurrent=3)
    trades_ew, snaps_ew = pb.run_policy(candidates, policy=ew, daily_ctx=ctx, cost_bps_round_trip=0.0)
    assert len(trades_ew) == 3  # slot-capped
    assert max(s.n_open for s in snaps_ew) <= 3

    cov = pb.PolicyConfig(name="cov", sizing_cfg=sz.SizingConfig(target_concurrent=3, capital=100_000),
                          target_concurrent=3)
    trades_cov, snaps_cov = pb.run_policy(candidates, policy=cov, daily_ctx=ctx, cost_bps_round_trip=0.0)
    assert len(trades_cov) == 3
    assert max(s.n_open for s in snaps_cov) <= 3


# --------------------------------------------------------------------------
# Sector cap propagation through the full admission loop
# --------------------------------------------------------------------------

def test_sector_cap_blocks_over_concentrated_admission():
    tickers = ["S1", "S2", "S3", "S4"]  # all same sector
    bars = {t: _make_bars(seed=i) for i, t in enumerate(tickers)}
    bar_cache = _FakeBarCache(bars)
    ts0 = pd.Timestamp("2025-06-02 14:00", tz="UTC")
    sig = _signal_frame([(ts0, t, 1.0 - i * 0.01) for i, t in enumerate(tickers)])
    candidates = pb._resolve_candidates(sig, bar_cache, tp_mult=5.0, sl_mult=5.0, max_hold=20)

    ctx = _make_daily_context(tickers)
    cfg = sz.SizingConfig(capital=100_000, target_concurrent=10, per_sector_cap_pct=0.10,
                          per_position_cap_pct=1.0, corr_penalty_strength=0.0,
                          adv_participation_cap_pct=None)
    policy = pb.PolicyConfig(name="cov", sizing_cfg=cfg, target_concurrent=10)
    sector_map = {t: "tech" for t in tickers}

    trades, _ = pb.run_policy(candidates, policy=policy, daily_ctx=ctx, sector_map=sector_map,
                              cost_bps_round_trip=0.0)
    total_sector_dollar = sum(t.dollar_size for t in trades if t.sector == "tech")
    # 10% of $100k capital = $10k sector ceiling, regardless of how many names wanted in
    assert total_sector_dollar <= 10_000.0 + 500.0  # small slack for whole-share rounding


# --------------------------------------------------------------------------
# Causality at the engine level
# --------------------------------------------------------------------------

def test_admission_sizing_is_causal_future_daily_data_does_not_change_past_sizing():
    tickers = ["AAA", "BBB", "CCC"]
    bars = {t: _make_bars(seed=i) for i, t in enumerate(tickers)}
    bar_cache = _FakeBarCache(bars)
    ts0 = pd.Timestamp("2025-06-02 14:00", tz="UTC")
    sig = _signal_frame([(ts0, "AAA", 1.0), (ts0, "BBB", 0.9), (ts0, "CCC", 0.8)])
    candidates = pb._resolve_candidates(sig, bar_cache, tp_mult=5.0, sl_mult=5.0, max_hold=20)

    ctx_present = _make_daily_context(tickers, periods=150)
    cfg = sz.SizingConfig(capital=100_000, target_concurrent=10)
    policy = pb.PolicyConfig(name="cov", sizing_cfg=cfg, target_concurrent=10)
    trades_present, _ = pb.run_policy(candidates, policy=policy, daily_ctx=ctx_present, cost_bps_round_trip=0.0)
    sizes_present = {t.ticker: t.dollar_size for t in trades_present}

    # Extend the SAME context with 100 additional future sessions (well past ts0) --
    # a leak would show up here if trailing_returns_subset/trailing_adv used them.
    ctx_future = _make_daily_context(tickers, periods=150, seed=1)
    rng = np.random.default_rng(999)
    for t in tickers:
        future_dates = pd.bdate_range(ctx_future._close[t].index.max() + pd.Timedelta(days=1),
                                      periods=100, tz="UTC").normalize()
        future_close = ctx_future._close[t].iloc[-1] * np.cumprod(1 + rng.normal(0, 0.05, 100))
        ctx_future._close[t] = pd.concat([ctx_future._close[t], pd.Series(future_close, index=future_dates)])
        future_dv = future_close * rng.integers(1_000_000, 5_000_000, 100)
        ctx_future._dollar_vol[t] = pd.concat([ctx_future._dollar_vol[t], pd.Series(future_dv, index=future_dates)])

    trades_future, _ = pb.run_policy(candidates, policy=policy, daily_ctx=ctx_future, cost_bps_round_trip=0.0)
    sizes_future = {t.ticker: t.dollar_size for t in trades_future}

    assert sizes_present == sizes_future


# --------------------------------------------------------------------------
# Whole-share integration + fixed-notional baseline fidelity
# --------------------------------------------------------------------------

def test_admitted_trade_dollar_size_matches_whole_shares_times_price():
    tickers = ["AAA"]
    bars = {t: _make_bars(seed=0) for t in tickers}
    bar_cache = _FakeBarCache(bars)
    ts0 = pd.Timestamp("2025-06-02 14:00", tz="UTC")
    sig = _signal_frame([(ts0, "AAA", 1.0)])
    candidates = pb._resolve_candidates(sig, bar_cache, tp_mult=5.0, sl_mult=5.0, max_hold=20)
    ctx = _make_daily_context(tickers)

    for policy in [
        pb.PolicyConfig(name="fixed", fixed_notional=5000.0),
        pb.PolicyConfig(name="ew", equal_weight=True, target_concurrent=10),
    ]:
        trades, _ = pb.run_policy(candidates, policy=policy, daily_ctx=ctx, cost_bps_round_trip=0.0)
        assert len(trades) == 1
        tr = trades[0]
        assert tr.dollar_size == pytest.approx(tr.shares * tr.entry_price)
        assert tr.shares >= 1  # both fixed-notional and equal-weight floor >=1 share here


def test_fixed_notional_floors_at_one_share_like_live_baseline():
    assert pb.shares_for_notional_min1(50_000.0, 5_000.0) == 1  # would round to 0 without the floor
    assert pb.shares_for_notional_min1(None, 5_000.0) == 1
    assert pb.shares_for_notional_min1(0.0, 5_000.0) == 1
