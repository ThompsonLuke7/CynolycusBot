import math

import numpy as np
import pandas as pd
import pytest

from core.personal_portfolio import trade_analysis as ta
from core.personal_portfolio import trade_timing as tt
from core.personal_portfolio.tests.test_trade_analysis import _daily, _ep


def test_black_scholes_parity_and_implied_vol_round_trip():
    s, k, t, r, v = 100.0, 105.0, 0.25, 0.04, 0.45
    c, p = tt.bs_price(s, k, t, r, v, True), tt.bs_price(s, k, t, r, v, False)
    assert c - p == pytest.approx(s - k * math.exp(-r * t), abs=1e-9)
    assert tt.implied_vol(c, s, k, t, r, True) == pytest.approx(v, abs=1e-6)
    assert tt.implied_vol(p, s, k, t, r, False) == pytest.approx(v, abs=1e-6)
    assert tt.implied_vol(14.0, 120.0, 105.0, t, r, True) is None        # below intrinsic: no vol fits
    assert tt.bs_price(110.0, 100.0, 0.0, r, v, True) == 10.0


def _option_trade(sell_price=3.0, exit_time="2026-02-10T15:00:00+00:00"):
    return _ep("2026-01-15T15:00:00+00:00", exit_time, strike=100.0, expiry="2026-03-20", status="closed",
               fills=[{"time": "2026-01-15T15:00:00+00:00", "role": "entry", "qty": 1, "price": 2.0, "cash": -200.0,
                       "fees": -0.66, "source": "trade"},
                      {"time": exit_time, "role": "exit", "qty": -1, "price": sell_price, "cash": sell_price * 100,
                       "fees": -0.66, "source": "trade"}],
               pnl=sell_price * 100 - 200 - 1.32, cost=200.0, fees=-1.32, multiplier=100.0,
               exit_reason="closed")


def test_hold_to_expiry_is_the_exact_intrinsic_payoff():
    split, raw = _daily(n=320)
    ep = _option_trade()
    out = tt.option_counterfactuals(ep, raw, None, None)
    s_t = float(raw.at[pd.Timestamp("2026-03-20"), "close"])
    assert out["expiry_close"] == s_t
    assert out["hold_to_expiry_pnl"] == pytest.approx(ep["pnl"] - 300 + 100 * max(0.0, s_t - 100.0))


def test_entry_style_and_exit_trigger_ignore_what_came_after_them():
    split, raw = _daily(n=320)
    ep = _option_trade()
    ctx = ta.episode_features(ep, split, raw, None, None, None)
    base = tt.timing_metrics(ep, ctx, split, raw)
    # change everything after the entry: the entry style must not move
    s2, r2 = split.copy(), raw.copy()
    later = s2.index > pd.Timestamp("2026-01-14")
    for df in (s2, r2):
        df.loc[later, ["open", "high", "low", "close"]] *= 1.7
    moved = tt.timing_metrics(ep, ctx, s2, r2)
    for k in ("entry_style", "ret_10d_dir", "range_pos_20"):
        assert base[k] == moved[k], k
    # change the exit day's own high and everything after the exit: the exit trigger must not move
    s3 = split.copy()
    s3.loc[s3.index >= pd.Timestamp("2026-02-10"), "high"] *= 2.0
    s3.loc[s3.index > pd.Timestamp("2026-02-10"), ["open", "low", "close"]] *= 2.0
    assert tt.timing_metrics(ep, ctx, s3, raw)["exit_trigger"] == base["exit_trigger"]
    assert base["entry_style"] in ("buying weakness", "buying strength", "in between")


def test_bars_problem_catches_a_misaligned_reverse_split_and_foreign_bars():
    split, raw = _daily(n=320, drift=0.0)
    i = split.index.get_loc(pd.Timestamp("2026-01-20"))
    raw.iloc[i + 1:, :4] *= 14                       # a 1:14 reverse split on 01-21 in the raw series...
    split.iloc[:i, :4] *= 14                         # ...applied one session early in the split series
    ep = _ep("2026-01-15T15:00:00+00:00", "2026-02-10T15:00:00+00:00")
    assert "misaligned" in ta.bars_problem(ep, split, raw)
    s2, r2 = _daily(n=320)
    stock = _ep("2026-01-15T15:00:00+00:00", "2026-02-10T15:00:00+00:00", kind="equity",
                fills=[{"time": "2026-01-15T15:00:00+00:00", "role": "entry", "qty": 10, "price": 0.15, "source": "trade"}])
    assert "not this security" in ta.bars_problem(stock, s2, r2)
    assert ta.bars_problem(_ep("2026-01-15T15:00:00+00:00", "2026-02-10T15:00:00+00:00"), s2, r2) is None


def test_roll_pnl_buys_whole_contracts_and_pays_intrinsic():
    split, raw = _daily(n=320, drift=0.003)
    ep = _option_trade(sell_price=6.0)
    ctx = {"under_exit": float(raw.at[pd.Timestamp("2026-02-10"), "close"]), "iv_exit": 0.4}
    s = ctx["under_exit"]
    px = tt.bs_price(s, s, 30 / 365, tt._rate(pd.Timestamp("2026-02-10")), 0.4, True)
    n = math.floor(600 / (px * 100))
    s_t = float(raw.loc[:pd.Timestamp("2026-03-12"), "close"].iloc[-1])
    expect = ep["pnl"] + n * (max(0.0, s_t - s) - px) * 100 - 2 * n * 0.65
    assert tt.roll_pnl(ep, ctx, raw, budget="all") == pytest.approx(expect)
    assert tt.roll_pnl({**ep, "exit_reason": "expired"}, ctx, raw) is None


def test_repo_policy_constants_match_the_live_exec_policy():
    from core.live_4h_exec import ExecPolicy

    live = ExecPolicy()
    for k in ("underlying_stop_atr", "take_profit", "scale_frac", "horizon_bars", "stop_loss"):
        assert tt.REPO_POLICY[k] == getattr(live, k), k


def test_premium_stop_entry_variants_and_drawdown_outcome():
    split, raw = _daily(n=380, drift=0.0, seed=5)
    after = split.index > pd.Timestamp("2026-01-15")
    decay = np.cumprod(np.where(after, 0.99, 1.0))            # flat into the entry, then -1%/day: the call bleeds out
    for df in (split, raw):
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].mul(decay, axis=0)
        df[["open", "high", "low", "close"]] *= 100.0 / float(df.at[pd.Timestamp("2026-01-14"), "close"])
    ep = _option_trade(sell_price=0.05, exit_time="2026-03-19T19:00:00+00:00")
    ctx = ta.episode_features(ep, split, raw, None, None, None)
    ctx["iv_entry"] = tt.implied_vol(2.0, ctx["under_entry"], 100.0,
                                     tt._years_to_expiry(pd.Timestamp("2026-01-15T15:00Z").to_pydatetime(), "2026-03-20"),
                                     0.038, True) or 0.6
    stop = tt.premium_stop_pnl(ep, ctx, raw, 0.5, 60)
    assert stop is not None and -150 < stop < -90 and stop > ep["pnl"]   # sold near half value, not at 0.05
    assert tt.premium_stop_pnl(ep, ctx, raw, 0.5, 14) <= stop     # the late version fires later, for less
    ev = tt.entry_variant_returns(ep, ctx, raw, split)
    assert set(ev) == {f"ev_{v}" for v in tt.ENTRY_VARIANTS} and ev["ev_confirm4"] is None
    dd = tt.drawdown_outcome(ep, ctx, raw, 0.5)
    assert dd["dd50_expiry_value"] == pytest.approx(max(0.0, float(raw.at[pd.Timestamp("2026-03-20"), "close"]) - 100) / 2.0)
    assert tt.comeback_stats([{**ep, "ctx": {**ctx, **dd, "stop50_late_pnl": stop}}]) is None   # < 10 cases
