"""A deferred exit must stay OWNED and must still reach the realized-PnL ledger.

Two defects from 2026-08-10/11, both rooted in the same gap: build_mixed_plan
removes a position from managed the moment it plans a full exit, and execute_plan
only restores it when the *submission* raises. A deferred exit is pulled from the
plan before submission, so neither path runs.

  1. Ownership. The position ends up held at the broker and claimed by nobody.
     Swing's reconcile adopts unclaimed positions (it reads siblings' `managed`
     via position_manager._sibling_module_owned_symbols), so on 2026-08-11 it
     restored HTF's VSH (19x VSH260821C00035000) as its own at 09:06:28 ET and
     HTF's own deferred exit sold those contracts at 09:37:04. Self-healed 31s
     later, but two modules owned one position for half an hour.

  2. Ledger. submit_pending_exit_orders never called record_exit_realized_pnl,
     so the AMAT260821C00550000 and VSH260821C00035000 stops that flushed on
     08-11 produced no closed_trades.jsonl row at all. Realized P&L was
     understated and the fills were recorded nowhere — and deferred exits are
     precisely the population the stop-overshoot fields exist to measure, since
     they sit unpriced from the 16:20 ET decision to the 09:35 ET flush.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from core.live_4h_exec import (
    defer_exits_if_opg_unavailable,
    execute_plan,
    submit_pending_exit_orders,
)

_ET = ZoneInfo("America/New_York")


def _et(hour, minute, day=10):
    return datetime(2026, 8, day, hour, minute, tzinfo=_ET).astimezone(timezone.utc)


class _Client:
    def __init__(self, fill=1.20):
        self.fill = fill
        self.option_orders = []

    def submit_order(self, *, symbol, qty, side, **k):
        return {"id": f"{symbol}-{side}"}

    def get_option_quotes(self, symbols, **k):
        return {"quotes": {symbols: {"bp": 1.15, "ap": 1.25}}}

    def submit_option_order(self, *, symbol, qty, side, **k):
        self.option_orders.append((side, symbol, qty))
        return {"id": f"{symbol}-{side}"}

    def get_order(self, _oid):
        return {"status": "filled", "filled_avg_price": self.fill, "filled_qty": "19"}


def _queue(tmp_path, module, entries):
    d = tmp_path / module
    d.mkdir(parents=True, exist_ok=True)
    (d / "pending_exit_orders.json").write_text(json.dumps({"updated": "x", "entries": entries}))


# --- 1. ownership ---------------------------------------------------------------

_VSH = "VSH260821C00035000"
# The real contract from the 2026-08-11 incident, kept verbatim so the fixture
# still names what it documents. It has since expired, and the flush now refuses
# to send an order for an expired contract, so every flush test below pins the
# clock to the incident's session rather than swapping in a synthetic symbol.
_VSH_SESSION = datetime(2026, 8, 11, 9, 35, tzinfo=_ET)


def test_deferred_exit_restores_the_position_to_managed(monkeypatch, tmp_path):
    """The VSH case. build_mixed_plan already dropped it; the deferral puts it back."""
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: False)
    new_managed = {}  # build_mixed_plan removed VSH when it planned the exit
    exit_context = {_VSH: ("VSH", {"route": "option", "occ": _VSH, "contracts": 19,
                                   "runs_held": 4, "entry_bar": "2026-08-04 14:00:00+00:00"})}
    plan = [(_VSH, "sell", 19, "stop_-39%", "option")]
    out = defer_exits_if_opg_unavailable(
        "multi_ticker_swing_htf", "2026-08-10 18:00", plan, {}, now=_et(16, 25),
        ledger_root=str(tmp_path), new_managed=new_managed, exit_context=exit_context)
    assert out == []                                  # queued, not submitted
    assert "VSH" in new_managed                       # and still ours
    assert new_managed["VSH"]["occ"] == _VSH


def test_restored_position_is_visible_to_a_sibling_reconcile(monkeypatch, tmp_path):
    """The ownership contract in one assertion: the OCC a sibling would scan for
    must appear in the module's managed state while the exit is pending."""
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: False)
    new_managed, exit_context = {}, {_VSH: ("VSH", {"route": "option", "occ": _VSH})}
    defer_exits_if_opg_unavailable("htf", "bar", [(_VSH, "sell", 19, "stop_-39%", "option")], {},
                                   now=_et(16, 25), ledger_root=str(tmp_path),
                                   new_managed=new_managed, exit_context=exit_context)
    claimed = {e.get("occ") or e.get("symbol") for e in new_managed.values()}
    assert _VSH in claimed


def test_a_kept_exit_is_not_restored(monkeypatch, tmp_path):
    """Inside the OPG window an equity sell goes out now, so managed stays pruned."""
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: False)
    new_managed = {}
    exit_context = {"CRWV": ("CRWV", {"route": "equity", "symbol": "CRWV"})}
    out = defer_exits_if_opg_unavailable("meta_ranker", "bar",
                                         [("CRWV", "sell", 12, "take_profit_+30%", "equity")], {},
                                         now=_et(22, 0), ledger_root=str(tmp_path),
                                         new_managed=new_managed, exit_context=exit_context)
    assert out == [("CRWV", "sell", 12, "take_profit_+30%", "equity")]
    assert new_managed == {}


def test_execute_plan_wires_the_restore_through(monkeypatch, tmp_path):
    """End to end: the runner path, not just the helper in isolation."""
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: False)
    new_managed = {}
    exit_context = {_VSH: ("VSH", {"route": "option", "occ": _VSH, "contracts": 19})}
    failed = execute_plan(
        _Client(), plan=[(_VSH, "sell", 19, "stop_-39%", "option")], limits={}, submit=True,
        equity_tif_fn=lambda: "day", new_managed=new_managed, exit_context=exit_context,
        module="multi_ticker_swing_htf", pos_lookup={}, bar="2026-08-10 18:00",
        ledger_root=str(tmp_path))
    assert failed == set()
    assert "VSH" in new_managed
    q = json.loads((tmp_path / "multi_ticker_swing_htf" / "pending_exit_orders.json").read_text())
    assert [e["order_symbol"] for e in q["entries"]] == [_VSH]


# --- 2. ledger ------------------------------------------------------------------

def _rows(tmp_path, module):
    p = tmp_path / module / "closed_trades.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_flush_writes_a_realized_pnl_row(tmp_path):
    """The 08-11 gap: VSH flushed with an order id and no ledger row."""
    _queue(tmp_path, "htf", [{"order_symbol": _VSH, "side": "sell", "qty": 19,
                              "route": "option", "reason": "stop_-39%",
                              "bar": "2026-08-10 18:00:00+00:00"}])
    res = submit_pending_exit_orders(
        _Client(fill=1.20), "htf", equity_tif_fn=lambda: "day",
        pos_lookup={_VSH: {"qty": 19, "avg_entry": 2.00}},
        managed={"VSH": {"route": "option", "occ": _VSH, "runs_held": 4,
                         "entry_bar": "2026-08-04 14:00:00+00:00", "unrealized_gain": -0.42}},
        ledger_root=str(tmp_path), now=_VSH_SESSION)
    assert res["count"] == 1
    rows = _rows(tmp_path, "htf")
    assert len(rows) == 1
    r = rows[0]
    assert r["order_symbol"] == _VSH
    assert r["ticker"] == "VSH"
    assert r["exit_reason"] == "stop_-39%"
    assert r["entry_avg_price"] == 2.00
    assert r["exit_fill_price"] == 1.20
    assert r["realized_pnl"] == (1.20 - 2.00) * 100 * 19       # -1520.0
    assert r["bar"] == "2026-08-10 18:00:00+00:00"             # the DECISION bar


def test_flush_row_carries_the_overshoot_decomposition(tmp_path):
    """Deferred stops are the population these fields exist for: they sit
    unpriced from the 16:20 ET decision to the 09:35 ET flush."""
    _queue(tmp_path, "htf", [{"order_symbol": _VSH, "side": "sell", "qty": 19,
                              "route": "option", "reason": "stop_-39%", "bar": "b"}])
    submit_pending_exit_orders(
        _Client(fill=1.20), "htf", equity_tif_fn=lambda: "day",
        pos_lookup={_VSH: {"qty": 19, "avg_entry": 2.00}},
        managed={"VSH": {"route": "option", "occ": _VSH, "unrealized_gain": -0.42}},
        ledger_root=str(tmp_path), now=_VSH_SESSION)
    r = _rows(tmp_path, "htf")[0]
    assert r["decision_gain"] == -0.42          # where it was when we looked
    assert r["fill_gain"] == -0.40              # where it actually filled
    assert r["stop_overshoot"] == -0.01         # fill_gain + 0.39


def test_flush_provenance_is_null_without_managed_state(tmp_path):
    """Row is still written; only entry_bar / runs_held / decision_gain go null."""
    _queue(tmp_path, "htf", [{"order_symbol": _VSH, "side": "sell", "qty": 19,
                              "route": "option", "reason": "stop_-39%", "bar": "b"}])
    submit_pending_exit_orders(_Client(fill=1.20), "htf", equity_tif_fn=lambda: "day",
                               pos_lookup={_VSH: {"qty": 19, "avg_entry": 2.00}},
                               ledger_root=str(tmp_path), now=_VSH_SESSION)
    r = _rows(tmp_path, "htf")[0]
    assert r["realized_pnl"] == (1.20 - 2.00) * 100 * 19
    assert r["entry_bar"] is None and r["runs_held"] is None and r["decision_gain"] is None


def test_a_skipped_exit_writes_no_row(tmp_path):
    """Nothing was sold, so nothing is realized."""
    _queue(tmp_path, "htf", [{"order_symbol": _VSH, "side": "sell", "qty": 19, "route": "option"}])
    res = submit_pending_exit_orders(_Client(), "htf", equity_tif_fn=lambda: "day",
                                     pos_lookup={}, ledger_root=str(tmp_path))
    assert res["count"] == 0
    assert _rows(tmp_path, "htf") == []


def test_an_equity_flush_row_uses_the_share_multiplier(tmp_path):
    _queue(tmp_path, "meta_ranker", [{"order_symbol": "CRWV", "side": "sell", "qty": 12,
                                      "route": "equity", "reason": "take_profit_+30%", "bar": "b"}])
    submit_pending_exit_orders(_Client(fill=90.0), "meta_ranker", equity_tif_fn=lambda: "day",
                               pos_lookup={"CRWV": {"qty": 75, "avg_entry": 66.30}},
                               managed={"CRWV": {"route": "equity", "symbol": "CRWV"}},
                               ledger_root=str(tmp_path))
    r = _rows(tmp_path, "meta_ranker")[0]
    assert r["route"] == "equity"
    assert r["realized_pnl"] == round((90.0 - 66.30) * 12, 2)
    assert r["stop_overshoot"] is None      # not a stop
