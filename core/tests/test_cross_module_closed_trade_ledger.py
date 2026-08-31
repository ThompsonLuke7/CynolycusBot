"""Every module that closes a position writes the SAME ledger row.

The 30m swing and the SPY daytrader kept no `closed_trades.jsonl` at all, so the
2026-08 execution study had to rebuild their 227 position lifecycles from raw
Alpaca fills just to see them — and Alpaca's order retention only reaches back to
2026-07-13, so four sessions of live history were unrecoverable.

These pin the contract that makes a cross-module study a join instead of a
reconstruction: one schema, written from one constructor.
"""
from __future__ import annotations

import json

from core.live_4h_exec import (
    CLOSED_TRADE_SCHEMA_VERSION,
    append_closed_trade,
    closed_trade_record,
)

_ENTRY = {
    "entry_bar": "2026-08-18T18:20:00+00:00",
    "runs_held": 3,
    "entry_order_id": "oid-1",
    "entry_submitted_at": "2026-08-18T18:20:00Z",
    "entry_filled_at": "2026-08-18T18:20:01Z",
    "entry_fill_price": 2.00,
    "entry_filled_qty": 5.0,
    "u_entry": 11.40,
    "u_atr": 0.62,
}


def _record(**over):
    base = dict(module="m", bar="b", ticker="ABC", order_symbol="ABC260918C00010000",
                route="option", qty=5, exit_reason="take_profit_full_+30%",
                entry_avg_price=2.00, exit_fill_price=2.60, realized_pnl=300.0,
                entry_state=_ENTRY, order_id="oid-2")
    base.update(over)
    return closed_trade_record(**base)


def test_the_row_carries_both_sides_of_the_clock():
    r = _record()
    assert r["schema_version"] == CLOSED_TRADE_SCHEMA_VERSION
    for k in ("entry_submitted_at", "entry_filled_at", "entry_fill_price",
              "entry_order_id", "entry_filled_qty"):
        assert r[k] == _ENTRY[k], k
    assert r["order_id"] == "oid-2"


def test_the_row_carries_the_underlying_anchor_next_to_the_option_price():
    """entry_avg_price is PREMIUM; u_entry is the underlying. Both, or an option
    row cannot be compared against the move it was a bet on."""
    r = _record()
    assert r["entry_avg_price"] == 2.00
    assert r["u_entry"] == 11.40 and r["u_atr"] == 0.62


def test_fill_gain_and_stop_overshoot_are_derived_consistently():
    r = _record(exit_reason="stop_-39%", exit_fill_price=1.16)
    assert r["fill_gain"] == -0.42
    assert r["stop_overshoot"] == -0.03      # filled 3pts worse than the -39% policy
    assert _record()["stop_overshoot"] is None   # not a stop -> no threshold


def test_a_missing_entry_state_nulls_the_entry_side_rather_than_raising():
    r = _record(entry_state=None)
    assert r["entry_filled_at"] is None and r["u_entry"] is None
    assert r["exit_fill_price"] == 2.60       # the exit side still stands


def test_append_writes_one_json_line_per_close(tmp_path):
    append_closed_trade("multi_ticker_swing", _record(module="multi_ticker_swing"),
                        ledger_root=str(tmp_path))
    append_closed_trade("multi_ticker_swing", _record(module="multi_ticker_swing"),
                        ledger_root=str(tmp_path))
    path = tmp_path / "multi_ticker_swing" / "closed_trades.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(rows) == 2
    assert rows[0]["module"] == "multi_ticker_swing"


def test_a_broken_row_never_propagates(tmp_path):
    """A ledger write must not be able to stop trading."""
    class _Unserializable:
        def __repr__(self):
            raise RuntimeError("boom")

    append_closed_trade("m", {"bad": _Unserializable()}, ledger_root=str(tmp_path))


def test_every_module_uses_the_same_key_set():
    """The join contract. If one module drifts, a cross-module study silently
    loses whichever column it dropped."""
    keys = set(_record(module="momentum_expansion"))
    for module in ("multi_ticker_swing", "spy_daytrader", "meta_ranker", "dealer_ranker"):
        assert set(_record(module=module)) == keys
