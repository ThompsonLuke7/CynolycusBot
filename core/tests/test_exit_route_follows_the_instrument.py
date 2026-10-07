"""An exit is booked by the instrument that was sold, not by a missing label.

2026-09-24..10-02: five Meta share sales (EQPT, ACHR, CAVA, LITE, WDAY) reached
the exit ledger labelled "option" and were booked at the x100 contract
multiplier: -$173,019 and +$121,435 recorded against a true -$1,730 and +$1,214.
The broker-fill certificate did not catch it, because the order registry was
written with the same wrong route and so both sides of its comparison were x100.

The label came from a plan-level flag (see the governed-path tests). These tests
cover the shared bookkeeping underneath it: when no route travels with an exit,
the fallback is the symbol itself, never a bare "option".
"""
from __future__ import annotations

import json

import pytest

from core.live_4h_exec import plan_row_routes, route_for_symbol, track_exit_submission

_OCC = "NTSK260918C00015000"
_BAR = "2026-09-29 18:00:00+00:00"


class _Client:
    """Paper client whose sell order is already filled when it is read back."""

    _trading_base = "https://paper-api.alpaca.markets"

    def __init__(self, *, filled_qty, fill_price):
        self._order = {"status": "filled", "filled_qty": str(filled_qty),
                       "filled_avg_price": str(fill_price)}

    def get_order(self, order_id):
        return {"id": order_id, **self._order}


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("symbol, expected", [
    (_OCC, "option"),
    ("ntsk260918c00015000", "option"),
    ("CAVA", "equity"),
    ("BRK.B", "equity"),
    ("", "equity"),
    (None, "equity"),
])
def test_route_for_symbol(symbol, expected):
    assert route_for_symbol(symbol) == expected


def test_plan_row_routes_reads_each_rows_own_route():
    plan = [(_OCC, "sell", 5, "take_profit_+30%", "option"),
            ("CAVA", "sell", 80, "horizon", "equity"),
            ("OLD", "sell", 10, "horizon")]  # 4-field row: no route of its own

    assert plan_row_routes(plan) == {_OCC: "option", "CAVA": "equity"}


def test_a_share_exit_with_no_route_label_is_booked_per_share(tmp_path):
    # CAVA 2026-09-29: 80 shares, $62.33 -> $52.51. Recorded -$78,560.
    state = {"route": "equity", "symbol": "CAVA", "shares": 80, "entry_avg_price": 62.33,
             "entry_bar": "2026-08-07 18:00:00+00:00", "runs_held": 53}
    new_managed: dict = {}

    result, row = track_exit_submission(
        _Client(filled_qty=80, fill_price=52.51), module="meta_ranker",
        item=("CAVA", "sell", 80, "horizon"), resp={"id": "order-cava"},
        new_managed=new_managed, exit_context={"CAVA": ("CAVA", state)},
        pos_lookup={"CAVA": {"avg_entry": 62.33}}, bar=_BAR, ledger_root=tmp_path)

    assert result == "closed" and "CAVA" not in new_managed
    assert row["route"] == "equity"
    assert row["realized_pnl"] == pytest.approx(-785.60)

    (fill,) = _rows(tmp_path / "meta_ranker" / "exit_fills.jsonl")
    assert fill["proceeds"] == pytest.approx(80 * 52.51)
    assert fill["realized_pnl"] == pytest.approx(-785.60)

    # The registry feeds the broker-fill certificate's multiplier; a wrong route
    # here is what let the x100 rows pass that check.
    (owner,) = _rows(tmp_path / "broker_reconciliation" / "paper" / "order_registry.jsonl")
    assert owner["route"] == "equity"


def test_an_option_exit_with_no_route_label_keeps_the_contract_multiplier(tmp_path):
    state = {"route": "option", "occ": _OCC, "contracts": 33, "entry_avg_price": 1.10,
             "entry_bar": "2026-09-01 18:00:00+00:00", "runs_held": 11}
    new_managed: dict = {}

    result, row = track_exit_submission(
        _Client(filled_qty=33, fill_price=2.30), module="meta_ranker",
        item=(_OCC, "sell", 33, "expiring_before_closure"), resp={"id": "order-ntsk"},
        new_managed=new_managed, exit_context={_OCC: ("NTSK", state)},
        pos_lookup={_OCC: {"avg_entry": 1.10}}, bar=_BAR, ledger_root=tmp_path)

    assert result == "closed" and "NTSK" not in new_managed
    assert row["route"] == "option"
    assert row["realized_pnl"] == pytest.approx(3960.00)
