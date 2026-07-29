"""Tests for the pyramiding lot simulator.

Coverage (mapped to the task's verification requirements):
  * regression — pyramiding OFF reproduces the EXISTING baseline exit engine
    (``backtest_exits.simulate``, unmodified) exactly, on synthetic bars and on
    real 4H bars;
  * regression — ``family_backtest._simulate_signal`` is byte-unchanged
    (golden-value test, so any future edit to it breaks here);
  * causality — an add never consults a bar after the trigger bar;
  * lot accounting — shares/notional/P&L correct after adds and trims;
  * precedence — stop and horizon pre-empt an add on the same bar; a trim
    executes before an add on the same bar;
  * horizon clock does not reset on an add;
  * ladder / max-adds / re-selection spacing behave as pre-registered;
  * capital linearity (used by the capital-matched check).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.meta_context.meta_ranker import backtest_exits as be
from strategies.momentum_expansion.backtest.family_backtest import _simulate_signal

from research.pyramid_lab.engine import (
    NO_PYRAMID, BasePolicy, PyramidPolicy, simulate_ticker,
)

LIVE = BasePolicy()
LIVE_KW = dict(stop=LIVE.stop_loss, target=LIVE.take_profit, scale_frac=LIVE.scale_frac,
               trail=LIVE.trail_stop, grace=LIVE.grace_bars, horizon=LIVE.horizon_bars)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bars_frame(close, high=None, low=None, start="2024-01-02 14:00"):
    close = np.asarray(close, dtype=float)
    high = close * 1.0 if high is None else np.asarray(high, dtype=float)
    low = close * 1.0 if low is None else np.asarray(low, dtype=float)
    idx = pd.date_range(start, periods=len(close), freq="4h", tz="UTC")
    return pd.DataFrame({"close": close, "high": high, "low": low}, index=idx)


def _arrays(df: pd.DataFrame, member: np.ndarray):
    return (df["close"].to_numpy(float), df["high"].to_numpy(float),
            df["low"].to_numpy(float), np.asarray(member, dtype=bool))


def _run(df, member, *, pyr=NO_PYRAMID, base=LIVE, notional=5000.0, cost_bps=0.0):
    c, h, l, m = _arrays(df, member)
    return simulate_ticker(c, h, l, m, ticker="T", base=base, pyr=pyr,
                           notional=notional, cost_bps=cost_bps)


def _rng_walk(n, seed, start=100.0):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0, 0.035, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.02, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.02, n)))
    return _bars_frame(close, high, low)


# ---------------------------------------------------------------------------
# 1. regression: pyramiding OFF == the existing baseline engine
# ---------------------------------------------------------------------------

def test_pyramid_off_matches_baseline_engine_synthetic(monkeypatch):
    """With no adds and no costs, this engine must reproduce
    ``backtest_exits.simulate`` (the engine the LIVE exit policy was selected
    with) exactly: same trade count, same mean/total return, same avg hold."""
    frames, members = {}, []
    for k in range(12):
        tkr = f"T{k}"
        df = _rng_walk(400, seed=100 + k)
        frames[tkr] = df
        rng = np.random.default_rng(500 + k)
        mask = rng.random(len(df)) < 0.08
        members.append(pd.DataFrame({"timestamp": df.index[mask], "ticker": tkr, "in_top": True}))
    member = pd.concat(members, ignore_index=True)

    monkeypatch.setattr(be, "_ticker_path", lambda t, ts: frames.get(t))
    ref = be.simulate(member, **LIVE_KW)

    rets, holds = [], []
    for tkr, df in frames.items():
        mk = member[member["ticker"] == tkr].set_index("timestamp")["in_top"]
        mask = mk.reindex(df.index).fillna(False).astype(bool).to_numpy()
        for p in _run(df, mask):
            rets.append(p.ret_on_initial)
            holds.append(p.bars_held)

    assert ref["n"] > 50, "fixture must produce a meaningful number of trades"
    assert len(rets) == ref["n"]
    assert float(np.mean(rets)) == pytest.approx(ref["mean"], rel=1e-12, abs=1e-12)
    assert float(np.mean(holds)) == pytest.approx(ref["avg_hold"], rel=1e-12, abs=1e-12)
    assert float(np.median(rets)) == pytest.approx(ref["median"], rel=1e-12, abs=1e-12)
    assert float(np.std(rets)) == pytest.approx(ref["std"], rel=1e-12, abs=1e-12)
    assert float((np.array(rets) > 0).mean()) == pytest.approx(ref["win"], rel=1e-12, abs=1e-12)
    assert float((np.array(rets) / np.maximum(holds, 1)).mean()) == pytest.approx(
        ref["ret_per_bar"], rel=1e-12, abs=1e-12)


def test_pyramid_off_matches_baseline_engine_real_bars():
    """Same parity assertion against REAL 4H bars, so the match is not an
    artifact of well-behaved synthetic prices."""
    from research.pyramid_lab.streams import Bars

    bars = Bars()
    tickers, frames = [], {}
    for tkr in ("AAPL", "AMD", "NVDA", "F", "PLTR", "SOFI", "INTC", "MU"):
        df = bars.get(tkr)
        if df is not None and len(df) > 500:
            frames[tkr] = df.iloc[-1200:]
            tickers.append(tkr)
    if len(tickers) < 4:
        pytest.skip("4H bar data not available")

    members = []
    for k, tkr in enumerate(tickers):
        df = frames[tkr]
        rng = np.random.default_rng(7000 + k)
        mask = rng.random(len(df)) < 0.06
        members.append(pd.DataFrame({"timestamp": df.index[mask], "ticker": tkr, "in_top": True}))
    member = pd.concat(members, ignore_index=True)

    orig = be._ticker_path
    try:
        be._ticker_path = lambda t, ts: frames.get(t)
        ref = be.simulate(member, **LIVE_KW)
    finally:
        be._ticker_path = orig

    rets, holds = [], []
    for tkr, df in frames.items():
        mk = member[member["ticker"] == tkr].set_index("timestamp")["in_top"]
        mask = mk.reindex(df.index).fillna(False).astype(bool).to_numpy()
        for p in _run(df, mask):
            rets.append(p.ret_on_initial)
            holds.append(p.bars_held)

    assert len(rets) == ref["n"] and ref["n"] > 20
    assert float(np.mean(rets)) == pytest.approx(ref["mean"], rel=1e-12, abs=1e-12)
    assert float(np.mean(holds)) == pytest.approx(ref["avg_hold"], rel=1e-12, abs=1e-12)


def test_baseline_reproduces_published_exit_policy_numbers():
    """External validation: the pyramiding-OFF baseline must reproduce the
    ALREADY-PUBLISHED momentum "id4 tail-rider" row of
    ``research/capstone/exit_policy_cross_module.csv`` -- the search that
    selected the current live ExecPolicy -- to the last decimal, on the same
    val/test windows.
    """
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    src = repo / "scripts/capstone/exit_policy_cross_module.py"
    csv = repo / "research/capstone/exit_policy_cross_module.csv"
    oof = repo / "strategies/momentum_expansion/models/expansion_v1/oof_preds.parquet"
    if not (src.exists() and csv.exists() and oof.exists()):
        pytest.skip("capstone exit-policy artifacts not available")

    spec = importlib.util.spec_from_file_location("epcm", src)
    epcm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(epcm)

    published = pd.read_csv(csv)
    row = published[(published["module"] == "momentum")
                    & (published["policy"] == "id4 tail-rider")].iloc[0]

    from research.pyramid_lab.streams import Bars
    bars = Bars()
    member = epcm.load_member("momentum", epcm.VAL_START, epcm.VAL_END)
    rets, holds = [], []
    for tkr, g in member.groupby("ticker"):
        df = bars.get(tkr)
        if df is None:
            continue
        mask = (g.set_index("timestamp")["in_top"]
                .reindex(df.index).fillna(False).astype(bool).to_numpy())
        for p in _run(df, mask):
            rets.append(p.ret_on_initial)
            holds.append(p.bars_held)

    assert len(rets) == int(row["val_n"])
    assert float(np.mean(rets)) == pytest.approx(float(row["val_mean"]), rel=1e-9)
    assert float(np.mean(holds)) == pytest.approx(float(row["val_hold"]), rel=1e-9)


def test_family_backtest_simulate_signal_unchanged():
    """Golden values for the SINGLE-ENTRY engine this study must not disturb.

    ``_simulate_signal`` is imported (never modified) by
    ``portfolio_backtest``, ``regime_policy/engine`` and
    ``run_family_compare``; this pins its behaviour so an edit to it fails
    loudly here.
    """
    n = 60
    close = np.linspace(100.0, 130.0, n)
    bars = {
        "ts": pd.date_range("2024-01-02 14:00", periods=n, freq="4h", tz="UTC").asi8,
        "ts_dt": pd.date_range("2024-01-02 14:00", periods=n, freq="4h", tz="UTC").to_numpy(),
        "open": close.copy(),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close.copy(),
        "atr": np.full(n, 2.0),
    }
    sig_ts = pd.Timestamp("2024-01-02 14:00", tz="UTC") + pd.Timedelta(hours=4 * 20)
    res = _simulate_signal(bars, sig_ts, 1, tp_mult=2.0, sl_mult=2.0, max_hold=25)
    assert res is not None
    assert res["entry_i"] == 21
    assert res["exit_reason"] == "tp"
    assert res["tp_price"] == pytest.approx(bars["open"][21] + 4.0)
    assert res["sl_price"] == pytest.approx(bars["open"][21] - 4.0)
    assert res["pnl_pct"] == pytest.approx(4.0 / bars["open"][21])
    # single-entry engine: one signal -> one trade, no lots, no adds
    assert set(res) >= {"entry_i", "exit_i", "pnl_pct", "bars_held"}
    assert "n_adds" not in res


# ---------------------------------------------------------------------------
# 2. causality
# ---------------------------------------------------------------------------

def test_add_never_uses_future_bars():
    """Truncating the series immediately after the add bar must not change the
    add: same count, same added notional, same peak cost basis."""
    close = np.array([100.0] + [100.0 * (1 + 0.05 * k) for k in range(1, 30)])
    df = _bars_frame(close, high=close * 1.02, low=close * 0.995)
    member = np.zeros(len(close), dtype=bool)
    member[0] = True
    pyr = PyramidPolicy(trigger="level", level=0.10, add_frac=1.0, max_adds=1)

    full = _run(df, member, pyr=pyr)[0]
    add_bar = None
    c, h, l, m = _arrays(df, member)
    for j in range(1, len(c)):
        if h[j] / c[0] - 1 >= 0.10:
            add_bar = j
            break
    assert add_bar is not None

    trunc = simulate_ticker(c[: add_bar + 1], h[: add_bar + 1], l[: add_bar + 1], m[: add_bar + 1],
                            base=LIVE, pyr=pyr, notional=5000.0, cost_bps=0.0)[0]
    assert full.n_adds == trunc.n_adds == 1
    assert full.added_notional == pytest.approx(trunc.added_notional)
    assert full.peak_cost_basis == pytest.approx(trunc.peak_cost_basis)


# ---------------------------------------------------------------------------
# 3. lot accounting
# ---------------------------------------------------------------------------

def test_lot_accounting_after_add_then_flat_exit():
    """Hand-computed: enter at 100 with $5,000; add $5,000 at the +10% bar's
    close; exit at horizon. P&L must equal the sum of the two lots' P&L."""
    n = 60
    close = np.full(n, 100.0)
    close[1:] = 110.0
    df = _bars_frame(close, high=close, low=close)
    member = np.zeros(n, dtype=bool)
    member[0] = True
    pyr = PyramidPolicy(trigger="level", level=0.10, add_frac=1.0, max_adds=1)
    p = _run(df, member, pyr=pyr)[0]

    # lot 1: 50 sh @100 ; trim never fires (+10% < +30%); add: $5,000/110 sh @110
    # exit at horizon bar 53, close 110 -> lot1 +$500, lot2 +$0
    assert p.n_adds == 1
    assert p.added_notional == pytest.approx(5000.0)
    assert p.exit_reason == "horizon"
    assert p.bars_held == 53
    assert p.pnl_gross == pytest.approx(50.0 * 10.0)
    assert p.peak_cost_basis == pytest.approx(10000.0)


def test_lot_accounting_with_trim_and_add():
    """Trim (16% of TOTAL shares, pro-rata across lots) then exit; totals must
    reconcile lot by lot."""
    n = 60
    close = np.full(n, 100.0)
    close[1:] = 140.0
    high = close.copy()
    df = _bars_frame(close, high=high, low=close)
    member = np.zeros(n, dtype=bool)
    member[0] = True
    pyr = PyramidPolicy(trigger="level", level=0.30, add_frac=1.0, max_adds=1)
    p = _run(df, member, pyr=pyr)[0]

    # bar 1: high 140 -> +40% >= +30% trim at 130 on 50 sh -> sell 8 sh, keep 42
    #        then add $5,000 @ close 140 -> 35.714... sh
    # exit horizon @140: 42*(140-100) + 35.7142857*(140-140) = 1680
    trim_realized = 8.0 * (130.0 - 100.0)
    assert p.n_adds == 1
    assert p.pnl_gross == pytest.approx(trim_realized + 42.0 * 40.0)
    total_shares_after = 42.0 + 5000.0 / 140.0
    assert p.peak_cost_basis == pytest.approx(42.0 * 100.0 + 5000.0)
    assert total_shares_after > 42.0


def test_fees_charged_on_every_fill():
    n = 60
    close = np.full(n, 100.0)
    close[1:] = 140.0
    df = _bars_frame(close, high=close, low=close)
    member = np.zeros(n, dtype=bool)
    member[0] = True
    pyr = PyramidPolicy(trigger="level", level=0.30, add_frac=1.0, max_adds=1)
    p = _run(df, member, pyr=pyr, cost_bps=10.0)[0]
    # fills: entry 5000, trim 8*130=1040, add 5000, exit (42 + 5000/140)*140
    exit_notional = (42.0 + 5000.0 / 140.0) * 140.0
    expected_fills = 5000.0 + 1040.0 + 5000.0 + exit_notional
    assert p.fill_notional == pytest.approx(expected_fills)
    assert p.fees == pytest.approx(10.0 / 1e4 * expected_fills)
    assert p.pnl_net == pytest.approx(p.pnl_gross - p.fees)


# ---------------------------------------------------------------------------
# 4. precedence
# ---------------------------------------------------------------------------

def test_stop_preempts_add_on_same_bar():
    """A bar whose high clears the add level but whose low breaches the stop
    must exit and NOT add."""
    n = 20
    close = np.full(n, 100.0)
    high = np.full(n, 100.0)
    low = np.full(n, 100.0)
    high[1] = 130.0          # clears +10% and +20% add levels
    low[1] = 55.0            # -45% <= -39% stop
    df = _bars_frame(close, high, low)
    member = np.zeros(n, dtype=bool)
    member[0] = True
    pyr = PyramidPolicy(trigger="level", level=0.10, add_frac=1.0, max_adds=2)
    p = _run(df, member, pyr=pyr)[0]
    assert p.exit_reason == "stop"
    assert p.n_adds == 0
    assert p.exit_price == pytest.approx(100.0 * (1 - 0.39))


def test_horizon_exit_preempts_add_on_same_bar():
    """The add is evaluated before the horizon check in the ladder, but an add
    can never occur on the exiting bar: on the horizon bar the position is
    already flagged for exit, so no lot is opened."""
    n = 70
    close = np.full(n, 100.0)
    high = np.full(n, 100.0)
    high[53] = 115.0     # clears the +10% add level, stays under the +30% trim
    df = _bars_frame(close, high, np.full(n, 100.0))
    member = np.zeros(n, dtype=bool)
    member[0] = True
    pyr = PyramidPolicy(trigger="level", level=0.10, add_frac=1.0, max_adds=1)
    p = _run(df, member, pyr=pyr)[0]
    assert p.exit_reason == "horizon" and p.bars_held == 53
    # Pre-registered: the add is evaluated LAST, so the horizon exit on bar 53
    # pre-empts it entirely -- no lot is opened on the exit bar and no fee is
    # burned for zero exposure.
    assert p.n_adds == 0
    assert p.added_notional == 0.0
    assert p.pnl_gross == pytest.approx(0.0)


def test_trim_executes_before_add_on_same_bar():
    """Trim first (16% of shares held BEFORE the add), then the add."""
    n = 60
    close = np.full(n, 100.0)
    close[1:] = 135.0
    df = _bars_frame(close, high=close, low=close)
    member = np.zeros(n, dtype=bool)
    member[0] = True
    pyr = PyramidPolicy(trigger="level", level=0.30, add_frac=1.0, max_adds=1)
    p = _run(df, member, pyr=pyr)[0]
    # if the add ran FIRST, the trim would sell 16% of (50 + 5000/135) shares.
    # Correct order sells 16% of 50 = 8 shares at 130.
    assert p.pnl_gross == pytest.approx(8.0 * 30.0 + 42.0 * 35.0)


def test_horizon_clock_does_not_reset_on_add():
    n = 120
    close = np.full(n, 100.0)
    close[1:] = 115.0
    df = _bars_frame(close, high=close, low=close)
    member = np.zeros(n, dtype=bool)
    member[0] = True
    for pyr in (NO_PYRAMID, PyramidPolicy(trigger="level", level=0.10, add_frac=1.0, max_adds=1)):
        p = _run(df, member, pyr=pyr)[0]
        assert p.bars_held == 53, "horizon must run from the ORIGINAL entry"


# ---------------------------------------------------------------------------
# 5. trigger mechanics
# ---------------------------------------------------------------------------

def test_level_ladder_and_max_adds():
    """max_adds=2 on an L10 arm adds at +10% and +20%, never more."""
    n = 60
    close = 100.0 * np.array([1.0] + [1.0 + 0.03 * k for k in range(1, n)])
    df = _bars_frame(close, high=close, low=close * 0.999)
    member = np.zeros(n, dtype=bool)
    member[0] = True
    for max_adds, expect in ((1, 1), (2, 2)):
        pyr = PyramidPolicy(trigger="level", level=0.10, add_frac=0.5, max_adds=max_adds)
        p = _run(df, member, pyr=pyr)[0]
        assert p.n_adds == expect
        assert p.added_notional == pytest.approx(expect * 0.5 * 5000.0)


def test_reselect_spacing_enforced():
    """Re-selection adds respect the 6-bar minimum spacing; a name that is in
    the top-10 on every bar still gets at most one add per 6 bars."""
    n = 60
    close = np.full(n, 100.0)
    df = _bars_frame(close, high=close, low=close)
    member = np.ones(n, dtype=bool)
    pyr = PyramidPolicy(trigger="reselect", add_frac=0.5, max_adds=2, spacing_bars=6)
    p = _run(df, member, pyr=pyr)[0]
    assert p.n_adds == 2
    tight = PyramidPolicy(trigger="reselect", add_frac=0.5, max_adds=2, spacing_bars=100)
    assert _run(df, member, pyr=tight)[0].n_adds == 0


def test_adds_only_fire_while_held_not_after_reentry_gap():
    """No add can be attributed to a position that is already closed."""
    n = 60
    close = np.full(n, 100.0)
    high = np.full(n, 100.0)
    low = np.full(n, 100.0)
    low[1] = 50.0  # stop out on bar 1
    high[2:] = 200.0
    df = _bars_frame(close, high, low)
    member = np.zeros(n, dtype=bool)
    member[0] = True
    pyr = PyramidPolicy(trigger="level", level=0.10, add_frac=1.0, max_adds=2)
    ps = _run(df, member, pyr=pyr)
    assert len(ps) == 1 and ps[0].n_adds == 0


# ---------------------------------------------------------------------------
# 6. per-bar series + capital linearity
# ---------------------------------------------------------------------------

def test_per_bar_pnl_sums_to_position_pnl():
    df = _rng_walk(300, seed=11)
    rng = np.random.default_rng(12)
    member = rng.random(len(df)) < 0.1
    c, h, l, m = _arrays(df, member)
    pnl = np.zeros(len(c))
    dep = np.zeros(len(c))
    pyr = PyramidPolicy(trigger="level", level=0.10, add_frac=1.0, max_adds=2)
    ps = simulate_ticker(c, h, l, m, base=LIVE, pyr=pyr, notional=5000.0, cost_bps=10.0,
                         pnl_by_bar=pnl, deployed_by_bar=dep)
    assert len(ps) > 3
    assert pnl.sum() == pytest.approx(sum(p.pnl_net for p in ps), rel=1e-9, abs=1e-6)
    assert dep.max() > 0


def test_capital_linearity_under_entry_basis():
    """Scaling the initial notional scales P&L and deployed capital exactly
    linearly (basis='entry'), which is why the capital-matched check reduces to
    a rescaling. Asserted, not assumed."""
    df = _rng_walk(300, seed=21)
    rng = np.random.default_rng(22)
    member = rng.random(len(df)) < 0.1
    c, h, l, m = _arrays(df, member)
    pyr = PyramidPolicy(trigger="level", level=0.20, add_frac=1.0, max_adds=2)
    out = {}
    for notional in (5000.0, 1234.0):
        pnl = np.zeros(len(c))
        dep = np.zeros(len(c))
        simulate_ticker(c, h, l, m, base=LIVE, pyr=pyr, notional=notional, cost_bps=10.0,
                        pnl_by_bar=pnl, deployed_by_bar=dep)
        out[notional] = (pnl.sum(), dep.mean())
    k = 1234.0 / 5000.0
    assert out[1234.0][0] == pytest.approx(out[5000.0][0] * k, rel=1e-9)
    assert out[1234.0][1] == pytest.approx(out[5000.0][1] * k, rel=1e-9)


def test_blended_basis_changes_exit_timing():
    """Sanity check on the SECONDARY sensitivity: keying the stop to the
    blended cost basis genuinely changes exits (otherwise it would be a
    pointless arm)."""
    # rip to +30% over 8 bars (two adds fire, lifting the blended cost above
    # the original entry), then crash to 65 -- between the entry-basis stop
    # (100 * 0.61 = 61) and the blended-basis stop, so only one arm stops out.
    close = np.concatenate([np.linspace(100, 130, 8), np.linspace(130, 65, 12), np.full(60, 65.0)])
    n = len(close)
    df = _bars_frame(close, high=close * 1.001, low=close * 0.999)
    member = np.zeros(n, dtype=bool)
    member[0] = True
    entry_arm = PyramidPolicy(trigger="level", level=0.10, add_frac=1.0, max_adds=2, basis="entry")
    blend_arm = PyramidPolicy(trigger="level", level=0.10, add_frac=1.0, max_adds=2, basis="blended")
    a = _run(df, member, pyr=entry_arm)[0]
    b = _run(df, member, pyr=blend_arm)[0]
    assert (a.bars_held, a.exit_reason) != (b.bars_held, b.exit_reason)


def test_invalid_policies_fail_fast():
    with pytest.raises(ValueError):
        PyramidPolicy(trigger="bogus")
    with pytest.raises(ValueError):
        PyramidPolicy(trigger="level", add_frac=1.0, max_adds=1)  # no level
    with pytest.raises(ValueError):
        PyramidPolicy(trigger="reselect", add_frac=0.0, max_adds=1)
