"""Regression: Swing's broker reconciliation must never adopt or flatten a
position a SIBLING module's own managed state currently claims.

2026-07-21 incident: HTF Swing bought 100 FIG shares as a real, intentional
entry. Swing's own "assigned equity" detector found those same shares in the
shared broker account, misdiagnosed them as an accidental option-exercise
assignment (Swing's universe happens to also include FIG), and auto-sold
them -- closing HTF's legitimate position without HTF's own exit logic ever
being consulted. Separately, Swing's broker-position-adoption path (no
ownership check at all) could just as easily have adopted a sibling's open
OPTION position as its own. Both paths now consult every sibling module's own
persisted `managed` state (that's what it exists for) before touching
anything.
"""
from __future__ import annotations

import json

import strategies.multi_ticker_swing.live.position_manager as pm_module
from strategies.multi_ticker_swing.live.position_manager import SwingPositionManager


class _FakeClient:
    def __init__(self, positions):
        self._positions = positions

    def get_positions(self):
        return self._positions


def _universe(*tickers: str) -> dict:
    return {t: None for t in tickers}


def _write_sibling_state(path, managed: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"managed": managed}))


def test_assigned_equity_never_flags_a_symbol_a_sibling_currently_manages(tmp_path, monkeypatch):
    htf_state = tmp_path / "htf_live_state.json"
    _write_sibling_state(htf_state, {"FIG": {"route": "equity", "symbol": "FIG", "shares": 100}})
    monkeypatch.setattr(pm_module, "_SIBLING_MODULE_STATE_PATHS", (htf_state,))

    client = _FakeClient([{"symbol": "FIG", "qty": 100, "side": "long", "avg_entry_price": 21.92}])
    swing = SwingPositionManager(client, dry_run=False, auto_flatten_assigned_equities=True)

    # Even if Swing's OWN history looks like a match (e.g. a stale cache
    # entry), the sibling veto must still win.
    detected = swing._broker_assigned_equity_positions(
        _universe("FIG"), owned_tickers={"FIG"}, sibling_owned=pm_module._sibling_module_owned_symbols(),
    )
    assert detected == []


def test_reconcile_does_not_adopt_an_option_position_a_sibling_owns(tmp_path, monkeypatch):
    dealer_state = tmp_path / "dealer_ranker_live_state.json"
    _write_sibling_state(dealer_state, {"FIG": {"route": "option", "occ": "FIG260724C00022000", "contracts": 1}})
    monkeypatch.setattr(pm_module, "_SIBLING_MODULE_STATE_PATHS", (dealer_state,))

    client = _FakeClient([
        {"symbol": "FIG260724C00022000", "qty": 1, "side": "long", "avg_entry_price": 0.97},
    ])
    swing = SwingPositionManager(client, dry_run=False, auto_flatten_assigned_equities=True)

    result = swing.reconcile_with_broker(
        universe=_universe("FIG"),
        price_lookup=lambda t: 22.0,
        atr_lookup=lambda t: 1.0,
    )

    assert result["restored"] == 0
    assert "FIG" not in swing._positions
    reasons = {row["reason"] for row in result["ignored_positions"]}
    assert "owned_by_other_module" in reasons


def test_sync_from_broker_does_not_adopt_an_option_position_a_sibling_owns(tmp_path, monkeypatch):
    meta_state = tmp_path / "meta_live_state.json"
    _write_sibling_state(meta_state, {"FIG": {"route": "option", "occ": "FIG260724C00022000", "contracts": 1}})
    monkeypatch.setattr(pm_module, "_SIBLING_MODULE_STATE_PATHS", (meta_state,))

    client = _FakeClient([
        {"symbol": "FIG260724C00022000", "qty": 1, "side": "long", "avg_entry_price": 0.97},
    ])
    swing = SwingPositionManager(client, dry_run=False, auto_flatten_assigned_equities=True)

    result = swing.sync_from_broker(
        universe=_universe("FIG"),
        price_lookup=lambda t: 22.0,
        atr_lookup=lambda t: 1.0,
    )

    assert result["restored"] == 0
    assert "FIG" not in swing._positions
    reasons = {row["reason"] for row in result["ignored_positions"]}
    assert "owned_by_other_module" in reasons


def test_sibling_owned_symbols_is_missing_file_tolerant(tmp_path, monkeypatch):
    monkeypatch.setattr(pm_module, "_SIBLING_MODULE_STATE_PATHS", (tmp_path / "does_not_exist.json",))
    assert pm_module._sibling_module_owned_symbols() == set()


def test_sibling_owned_symbols_collects_both_equity_and_option_keys(tmp_path, monkeypatch):
    state_a = tmp_path / "a.json"
    state_b = tmp_path / "b.json"
    _write_sibling_state(state_a, {"FIG": {"route": "equity", "symbol": "FIG"}})
    _write_sibling_state(state_b, {"SNDK": {"route": "option", "occ": "SNDK260814C01700000"}})
    monkeypatch.setattr(pm_module, "_SIBLING_MODULE_STATE_PATHS", (state_a, state_b))

    owned = pm_module._sibling_module_owned_symbols()
    assert owned == {"FIG", "SNDK260814C01700000"}
