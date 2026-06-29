"""Regression: swing assignment-flatten must stay inside its own option book.

The Alpaca account is shared with the Meta Ranker / Momentum modules, whose
deliberate 100-share equity lots also live in the swing universe. The flattener
must only ever touch lots the swing book actually held options on — otherwise it
liquidates another module's positions (the 2026-06-22 friendly-fire incident).
"""
from __future__ import annotations

from strategies.multi_ticker_swing.live.position_manager import SwingPositionManager


class _FakeClient:
    def __init__(self, positions):
        self._positions = positions

    def get_positions(self):
        return self._positions


def _universe(*tickers: str) -> dict:
    # The detector only does a membership check on the universe mapping.
    return {t: None for t in tickers}


def test_meta_equity_lots_are_not_claimed_as_assignment():
    # Account holds Meta's deliberate equity lot (WDC) and swing's own assigned
    # equity (ASTS, which swing held an option on before it expired).
    client = _FakeClient([
        {"symbol": "WDC", "qty": 100, "side": "long", "avg_entry_price": 771.0},
        {"symbol": "ASTS", "qty": 100, "side": "long", "avg_entry_price": 80.0},
    ])
    pm = SwingPositionManager(client, dry_run=False, auto_flatten_assigned_equities=True)

    # Swing's ownership scope: it holds/held an ASTS option, never a WDC one.
    owned = {"ASTS"}
    detected = pm._broker_assigned_equity_positions(_universe("WDC", "ASTS"), owned)

    symbols = {d["symbol"] for d in detected}
    assert symbols == {"ASTS"}, f"flattener escaped its scope: {symbols}"


def test_unowned_lot_is_ignored_even_if_in_universe():
    client = _FakeClient([{"symbol": "KLAC", "qty": 100, "side": "long", "avg_entry_price": 265.0}])
    pm = SwingPositionManager(client, dry_run=False, auto_flatten_assigned_equities=True)
    detected = pm._broker_assigned_equity_positions(_universe("KLAC"), owned_tickers=set())
    assert detected == []


def test_flatten_cooldown_blocks_immediate_resubmit():
    submitted = {"n": 0}

    class _SubClient(_FakeClient):
        def submit_order(self, **kwargs):
            submitted["n"] += 1
            return {"id": "x", "status": "pending_new"}

    pm = SwingPositionManager(_SubClient([]), dry_run=False, auto_flatten_assigned_equities=True)
    lot = [{"symbol": "ASTS", "qty": 100, "suggested_close_side": "sell"}]
    pm.flatten_assigned_equity_positions(lot)
    pm.flatten_assigned_equity_positions(lot)  # immediate retry must be suppressed
    assert submitted["n"] == 1
