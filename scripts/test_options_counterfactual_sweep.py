"""Regression tests for scripts/run_options_counterfactual_sweep.py.

Focused on the shares (B0) max_loss fix: a coordinator review of the smoke test found that
`long_shares.max_loss` was being read straight from `strategies.Structure.max_loss`, which for a
naked share leg is defined as "stock goes to zero" -- the entire notional. That badly overstates
the real risk every one of the three powered modules actually takes (they all exit at a stop), and
under `matched_max_loss` sizing (the pre-registration's PRIMARY metric) it makes shares look
artificially risk-hungry next to defined-risk option structures, biasing the whole experiment
toward "use options." These tests pin the fix: shares' reported/used max_loss must track the
module's real stop distance, never the to-zero notional figure.

No network calls: chain_cache/Alpaca are never touched by anything exercised here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_options_counterfactual_sweep as sweep  # noqa: E402


# --------------------------------------------------------------------------
# stop_distance_for_trade
# --------------------------------------------------------------------------


def test_stop_distance_uses_real_sl_price_when_present():
    dist, source = sweep.stop_distance_for_trade(
        "multi_ticker_swing_htf", "AAPL", pd.Timestamp("2026-01-05", tz="UTC"),
        entry_px=100.0, sl_price=97.0,
    )
    assert dist == pytest.approx(3.0)
    assert source == "sl_price"


def test_stop_distance_real_sl_price_is_far_from_to_zero_notional():
    """The bug this fix addresses: a real stop is a small fraction of entry price, not the
    whole position. Assert the fixed distance is nowhere near "entry price" (to-zero)."""
    entry_px = 250.0
    dist, _source = sweep.stop_distance_for_trade(
        "momentum_expansion", "NVDA", pd.Timestamp("2026-01-05", tz="UTC"),
        entry_px=entry_px, sl_price=entry_px * (1 - 0.198),  # median momentum_expansion stop ~19.8%
    )
    assert dist < entry_px * 0.5  # real stop, not "stock to zero"
    assert dist == pytest.approx(entry_px * 0.198, rel=1e-6)


def test_stop_distance_missing_sl_on_powered_4h_module_is_flagged_not_silent():
    """momentum_expansion / multi_ticker_swing_htf carry sl_price on 100% of real spine rows --
    a miss here must be a labeled fallback, never silently treated as notional risk."""
    dist, source = sweep.stop_distance_for_trade(
        "momentum_expansion", "AAPL", pd.Timestamp("2026-01-05", tz="UTC"), entry_px=100.0, sl_price=np.nan,
    )
    assert source == "atr_proxy_fallback_missing_sl"
    assert dist == pytest.approx(100.0 * sweep.MTS_STOP_PCT_FALLBACK)


def test_stop_distance_multi_ticker_swing_uses_atr_proxy_with_ticker_multiple(monkeypatch):
    fake_bars = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC"),
        "atr_14": [np.nan] * 13 + [2.0] * 7,  # first 14-1 rows have no ATR yet
    })
    monkeypatch.setattr(sweep, "_load_hourly_bars", lambda ticker: fake_bars)
    monkeypatch.setattr(sweep, "_MTS_UNIVERSE", {"XYZ": 4.0})

    dist, source = sweep.stop_distance_for_trade(
        "multi_ticker_swing", "XYZ", pd.Timestamp("2026-01-01T18:00:00", tz="UTC"),
        entry_px=100.0, sl_price=np.nan,
    )
    assert source == "atr_proxy_ticker_mult"
    assert dist == pytest.approx(4.0 * 2.0)  # sl_atr(4.0) x atr_14(2.0)


def test_stop_distance_multi_ticker_swing_defaults_when_ticker_not_in_universe(monkeypatch):
    fake_bars = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC"),
        "atr_14": [np.nan] * 13 + [1.5] * 7,
    })
    monkeypatch.setattr(sweep, "_load_hourly_bars", lambda ticker: fake_bars)
    monkeypatch.setattr(sweep, "_MTS_UNIVERSE", {})  # ticker not curated in trading_universe.json

    dist, source = sweep.stop_distance_for_trade(
        "multi_ticker_swing", "NOTLISTED", pd.Timestamp("2026-01-01T18:00:00", tz="UTC"),
        entry_px=100.0, sl_price=np.nan,
    )
    assert source == "atr_proxy_default_4x"
    assert dist == pytest.approx(sweep.DEFAULT_SL_ATR_MULT * 1.5)


def test_stop_distance_multi_ticker_swing_falls_back_when_no_1h_bars(monkeypatch):
    monkeypatch.setattr(sweep, "_load_hourly_bars", lambda ticker: None)
    dist, source = sweep.stop_distance_for_trade(
        "multi_ticker_swing", "THIN", pd.Timestamp("2026-01-01T18:00:00", tz="UTC"),
        entry_px=100.0, sl_price=np.nan,
    )
    assert source == "atr_proxy_fallback_no_1h_bars"
    assert dist == pytest.approx(100.0 * sweep.MTS_STOP_PCT_FALLBACK)


# --------------------------------------------------------------------------
# emit_shares_row: the actual max_loss used downstream must track stop
# distance, never entry notional.
# --------------------------------------------------------------------------


def test_shares_max_loss_tracks_stop_distance_not_notional():
    base = dict(
        trade_id="t1", module="multi_ticker_swing_htf", ticker="AAPL", direction=1,
        entry_ts=pd.Timestamp("2026-01-05", tz="UTC"), exit_ts=pd.Timestamp("2026-01-06", tz="UTC"),
        holding_days=1.0, week_key="2026-W02", entry_px_underlying=100.0, exit_px_underlying=102.0,
        atr_at_entry=np.nan, score=np.nan, exit_reason="tp", bars_held=8.0, provenance="backtest",
        realized_move_atr=np.nan,
    )
    stop_distance_dollars = 3.0  # a real 3% stop on a $100 stock
    # This is exactly how process_trade derives the matched-max-loss target_notional: scale up
    # by 1/stop_pct so that a stop-out loses TARGET_MAX_LOSS, using the REAL stop distance.
    stop_pct = stop_distance_dollars / 100.0
    matched_max_loss_notional = sweep.TARGET_MAX_LOSS / stop_pct

    row = sweep.emit_shares_row(
        base, "AAPL", base["entry_ts"], 102.0, 100.0, "long",
        "matched_max_loss", matched_max_loss_notional, stop_distance_dollars, "sl_price",
    )
    # Sizing: shares sized so a stop-out loses exactly TARGET_MAX_LOSS -- with a 3% stop that
    # means a MUCH larger notional than $5,000, the entire point of the fix.
    shares_qty = matched_max_loss_notional / 100.0
    assert row["max_loss"] == pytest.approx(sweep.TARGET_MAX_LOSS, rel=1e-6)
    assert row["entry_cost"] == pytest.approx(shares_qty * 100.0, rel=1e-6)
    # The bug this guards against: max_loss must NOT equal the to-zero notional (entry_cost),
    # since stop_distance_dollars (3.0) is nowhere near entry price (100.0).
    assert row["max_loss"] != pytest.approx(row["entry_cost"])
    assert row["shares_stop_source"] == "sl_price"
    assert row["shares_stop_distance_dollars"] == pytest.approx(3.0)


def test_shares_matched_notional_mode_still_sizes_to_5000_dollars():
    base = dict(
        trade_id="t2", module="momentum_expansion", ticker="TSLA", direction=-1,
        entry_ts=pd.Timestamp("2026-01-05", tz="UTC"), exit_ts=pd.Timestamp("2026-01-06", tz="UTC"),
        holding_days=1.0, week_key="2026-W02", entry_px_underlying=200.0, exit_px_underlying=195.0,
        atr_at_entry=np.nan, score=np.nan, exit_reason="tp", bars_held=4.0, provenance="backtest",
        realized_move_atr=np.nan,
    )
    row = sweep.emit_shares_row(
        base, "TSLA", base["entry_ts"], 195.0, 200.0, "short",
        "matched_notional", sweep.TARGET_NOTIONAL, 40.0, "sl_price",
    )
    assert abs(row["entry_cost"]) == pytest.approx(sweep.TARGET_NOTIONAL, rel=1e-6)
    # max_loss here reflects the real stop (40.0/200.0 = 20% of a $5,000 position = $1,000),
    # not the $5,000 to-zero figure.
    assert row["max_loss"] == pytest.approx(sweep.TARGET_NOTIONAL * (40.0 / 200.0), rel=1e-6)
    assert row["max_loss"] < sweep.TARGET_NOTIONAL


def test_shares_gross_and_net_pnl_agree_across_all_three_cost_assumptions():
    """Documented assumption: no equity bid/ask spread and no commission are modeled for shares,
    so gross == net under optimistic/calibrated/pessimistic alike."""
    base = dict(
        trade_id="t3", module="multi_ticker_swing", ticker="MSFT", direction=1,
        entry_ts=pd.Timestamp("2026-01-05", tz="UTC"), exit_ts=pd.Timestamp("2026-01-06", tz="UTC"),
        holding_days=1.0, week_key="2026-W02", entry_px_underlying=300.0, exit_px_underlying=306.0,
        atr_at_entry=np.nan, score=np.nan, exit_reason="tp", bars_held=6.0, provenance="backtest",
        realized_move_atr=np.nan,
    )
    row = sweep.emit_shares_row(
        base, "MSFT", base["entry_ts"], 306.0, 300.0, "long",
        "matched_notional", sweep.TARGET_NOTIONAL, 12.0, "atr_proxy_default_4x",
    )
    assert row["gross_pnl"] == pytest.approx(row["net_pnl_optimistic"])
    assert row["net_pnl_optimistic"] == pytest.approx(row["net_pnl_calibrated"])
    assert row["net_pnl_calibrated"] == pytest.approx(row["net_pnl_pessimistic"])
    assert row["commission_dollars"] == 0.0
    assert row["gross_pnl"] > 0  # long, entry 300 -> exit 306


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
