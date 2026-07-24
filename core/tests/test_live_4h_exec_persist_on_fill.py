"""Regression coverage for the 2026-07-23 IOT incident.

Dealer Ranker bought 23 contracts of IOT260724C00031500 at 15:52:43 ET; the
fill succeeded, but the module's `live_state.json` was only rewritten once,
after its *entire* multi-symbol plan finished executing. Swing's independent
broker reconciliation polled Alpaca directly ~19ms after the fill, saw a real
option position for IOT with no owner in any sibling module's on-disk managed
state (because Dealer Ranker hadn't persisted yet), adopted it as an "unknown
restored" position, and defensively liquidated it two minutes later for
-$4,945 -- money that belonged to Dealer Ranker's brand-new, healthy position.

`execute_plan` now accepts a `persist_managed` callback invoked after every
order attempt (success or failure) so callers can flush managed state to disk
immediately, shrinking the race window from "whole plan duration" to a single
order round trip.
"""
from __future__ import annotations

from core.live_4h_exec import execute_plan


class _FillClient:
    def __init__(self, *, reject_symbols: set[str] | None = None):
        self.reject_symbols = reject_symbols or set()
        self.submitted: list[tuple[str, str]] = []

    def _submit(self, side, sym):
        self.submitted.append((side, sym))
        if sym in self.reject_symbols:
            raise RuntimeError("HTTP Error 403: Forbidden")
        return {"id": f"{sym}-{side}", "status": "accepted"}

    def submit_option_order(self, *, symbol, qty, side, **k):
        return self._submit(side, symbol)

    def submit_order(self, *, symbol, qty, side, **k):
        return self._submit(side, symbol)


def _bypass_gates(monkeypatch):
    monkeypatch.setattr("core.live_4h_exec.filter_entry_orders_for_readiness",
                        lambda plan, new_managed=None: (plan, [], ""))
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: True)


def test_persist_managed_called_after_each_successful_order(monkeypatch):
    _bypass_gates(monkeypatch)
    c = _FillClient()
    new_managed = {
        "IOT": {"route": "option", "occ": "IOT260724C00031500", "contracts": 23},
        "STM": {"route": "option", "occ": "STM260724C00050000", "contracts": 5},
    }
    plan = [
        ("IOT260724C00031500", "buy", 23, "entry", "option"),
        ("STM260724C00050000", "buy", 5, "entry", "option"),
    ]
    snapshots: list[dict] = []
    execute_plan(
        c, plan=plan, limits={}, submit=True, equity_tif_fn=lambda: "day",
        new_managed=new_managed, exit_context={}, module="dealer_ranker",
        pos_lookup={}, bar="2026-07-23 15:51",
        persist_managed=lambda: snapshots.append(dict(new_managed)),
    )
    # One persist call per order, and by the first callback IOT is already
    # visible -- a sibling reconciling right then would see it as owned.
    assert len(snapshots) == 2
    assert "IOT" in snapshots[0]


def test_persist_managed_called_even_when_order_fails(monkeypatch):
    _bypass_gates(monkeypatch)
    c = _FillClient(reject_symbols={"BAD260724C00010000"})
    new_managed = {"BAD": {"route": "option", "occ": "BAD260724C00010000", "contracts": 1}}
    plan = [("BAD260724C00010000", "buy", 1, "entry", "option")]
    calls = []
    execute_plan(
        c, plan=plan, limits={}, submit=True, equity_tif_fn=lambda: "day",
        new_managed=new_managed, exit_context={}, module="dealer_ranker",
        pos_lookup={}, bar="2026-07-23 15:51",
        persist_managed=lambda: calls.append(dict(new_managed)),
    )
    # drop_failed_entry already removed BAD by the time we persist, so the
    # on-disk state never claims a phantom position for a rejected buy either.
    assert len(calls) == 1
    assert "BAD" not in calls[0]


def test_persist_managed_callback_error_does_not_block_remaining_orders(monkeypatch):
    _bypass_gates(monkeypatch)
    c = _FillClient()
    new_managed = {
        "A": {"route": "option", "occ": "A260724C00010000", "contracts": 1},
        "B": {"route": "option", "occ": "B260724C00010000", "contracts": 1},
    }
    plan = [
        ("A260724C00010000", "buy", 1, "entry", "option"),
        ("B260724C00010000", "buy", 1, "entry", "option"),
    ]

    def _boom():
        raise RuntimeError("disk full")

    execute_plan(
        c, plan=plan, limits={}, submit=True, equity_tif_fn=lambda: "day",
        new_managed=new_managed, exit_context={}, module="dealer_ranker",
        pos_lookup={}, bar="2026-07-23 15:51", persist_managed=_boom,
    )
    assert c.submitted == [("buy", "A260724C00010000"), ("buy", "B260724C00010000")]


def test_no_persist_managed_is_a_no_op(monkeypatch):
    _bypass_gates(monkeypatch)
    c = _FillClient()
    new_managed = {"A": {"route": "option", "occ": "A260724C00010000", "contracts": 1}}
    plan = [("A260724C00010000", "buy", 1, "entry", "option")]
    failed = execute_plan(
        c, plan=plan, limits={}, submit=True, equity_tif_fn=lambda: "day",
        new_managed=new_managed, exit_context={}, module="dealer_ranker",
        pos_lookup={}, bar="2026-07-23 15:51",
    )
    assert failed == set()
    assert c.submitted == [("buy", "A260724C00010000")]
