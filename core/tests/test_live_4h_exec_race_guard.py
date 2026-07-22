"""Cross-module option-position race guard in the shared 4H engine.

Regression coverage for the 2026-07-17 incident: Dealer Ranker and the Swing
runner independently ranked AUR and both bought the identical nearest-ATM
contract (AUR260724C00006000) within ~20 seconds of each other, because each
module's own `pos_info` snapshot was taken once near the top of its run and
never refreshed before submit. `execute_plan` now re-checks live positions
immediately before submitting BUY orders and drops any that another process
has since opened.
"""
from __future__ import annotations

from core.live_4h_exec import drop_failed_entry, execute_plan


class _RaceClient:
    """Reports `held_symbols` as already-open broker positions."""

    def __init__(self, *, held_symbols: set[str] | None = None, positions_raise: bool = False):
        self.held_symbols = held_symbols or set()
        self.positions_raise = positions_raise
        self.submitted: list[tuple[str, str]] = []

    def get_positions(self):
        if self.positions_raise:
            raise RuntimeError("positions endpoint down")
        return [{"symbol": s, "qty": "1"} for s in self.held_symbols]

    def _submit(self, side, sym):
        self.submitted.append((side, sym))
        return {"id": f"{sym}-{side}", "status": "accepted"}

    def submit_option_order(self, *, symbol, qty, side, **k):
        return self._submit(side, symbol)

    def submit_order(self, *, symbol, qty, side, **k):
        return self._submit(side, symbol)


def _bypass_entry_gates(monkeypatch):
    monkeypatch.setattr("core.live_4h_exec.filter_entry_orders_for_readiness",
                        lambda plan, new_managed=None: (plan, [], ""))
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: True)


def test_race_guard_drops_buy_already_held_by_another_module(monkeypatch):
    _bypass_entry_gates(monkeypatch)
    c = _RaceClient(held_symbols={"AUR260724C00006000"})
    new_managed = {"AUR": {"route": "option", "occ": "AUR260724C00006000", "contracts": 1}}
    plan = [("AUR260724C00006000", "buy", 1, "entry", "option")]
    failed = execute_plan(
        c, plan=plan, limits={}, submit=True, equity_tif_fn=lambda: "day",
        new_managed=new_managed, exit_context={},
        module="dealer_ranker", pos_lookup={}, bar="2026-07-17 19:52",
    )
    assert c.submitted == []          # never even attempted
    assert failed == set()            # not a submit failure, just skipped
    assert "AUR" not in new_managed   # phantom entry dropped, matches drop_failed_entry


def test_race_guard_leaves_unheld_buys_alone(monkeypatch):
    _bypass_entry_gates(monkeypatch)
    c = _RaceClient(held_symbols=set())
    new_managed = {"GOOD": {"route": "option", "occ": "GOOD260724C00010000", "contracts": 1}}
    plan = [("GOOD260724C00010000", "buy", 1, "entry", "option")]
    execute_plan(
        c, plan=plan, limits={}, submit=True, equity_tif_fn=lambda: "day",
        new_managed=new_managed, exit_context={},
        module="dealer_ranker", pos_lookup={}, bar="2026-07-17 19:52",
    )
    assert c.submitted == [("buy", "GOOD260724C00010000")]
    assert "GOOD" in new_managed


def test_race_guard_ignores_sell_orders():
    # Held-position lookups should never block exits -- only new entries.
    c = _RaceClient(held_symbols={"AAA260724C00010000"})
    plan = [("AAA260724C00010000", "sell", 1, "horizon", "option")]
    execute_plan(
        c, plan=plan, limits={}, submit=True, equity_tif_fn=lambda: "day",
        new_managed={}, exit_context={},
        module="dealer_ranker", pos_lookup={}, bar="2026-07-17 19:52",
    )
    assert c.submitted == [("sell", "AAA260724C00010000")]


def test_race_guard_skips_positions_lookup_when_plan_has_no_buys():
    class _NoPositionsMethodClient(_RaceClient):
        def get_positions(self):  # pragma: no cover - must never be called
            raise AssertionError("get_positions should not be called for a sell-only plan")

    c = _NoPositionsMethodClient()
    plan = [("AAA260724C00010000", "sell", 1, "horizon", "option")]
    execute_plan(
        c, plan=plan, limits={}, submit=True, equity_tif_fn=lambda: "day",
        new_managed={}, exit_context={},
        module="dealer_ranker", pos_lookup={}, bar="2026-07-17 19:52",
    )
    assert c.submitted == [("sell", "AAA260724C00010000")]


def test_race_guard_falls_through_on_positions_endpoint_error(monkeypatch):
    _bypass_entry_gates(monkeypatch)
    # A transient get_positions() failure must not block real entries -- fail open.
    c = _RaceClient(held_symbols={"AUR260724C00006000"}, positions_raise=True)
    new_managed = {"GOOD": {"route": "option", "occ": "GOOD260724C00010000", "contracts": 1}}
    plan = [("GOOD260724C00010000", "buy", 1, "entry", "option")]
    execute_plan(
        c, plan=plan, limits={}, submit=True, equity_tif_fn=lambda: "day",
        new_managed=new_managed, exit_context={},
        module="dealer_ranker", pos_lookup={}, bar="2026-07-17 19:52",
    )
    assert c.submitted == [("buy", "GOOD260724C00010000")]


def test_drop_failed_entry_still_works_standalone():
    nm = {"AUR": {"occ": "AUR260724C00006000", "contracts": 1}}
    drop_failed_entry(nm, "AUR260724C00006000")
    assert nm == {}
