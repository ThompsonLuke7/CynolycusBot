"""The entry side of the clock, and what became of every planned order.

Two gaps the 2026-08 execution study ran into, both fixed here:

  1. NOTHING persisted an entry's submit time, fill time, fill price or order id.
     Exits have carried all of that since the ledger existed. Reconstructing the
     entry side meant querying Alpaca order history, which works but only back to
     the broker's retention floor (2026-07-13) — four sessions of live history
     are gone for good.

  2. The audit recorded the plan and the ledger recorded the fills, and nothing
     recorded the middle. The study could establish that only 54% of planned
     entries became positions but not why.

Plus the trim-lineage defect: `exit_context` holds FULL exits only, so a partial
take-profit passed entry_state=None and wrote entry_bar=null. All 45 such rows in
the study were trims, none corrupt.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from core.live_4h_exec import execute_plan, order_plan_audit_record

_ET = ZoneInfo("America/New_York")
_OPEN = datetime(2026, 8, 20, 11, 0, tzinfo=_ET).astimezone(timezone.utc)


class _Client:
    """Paper broker. `fill` None means accepted-but-never-filled."""

    def __init__(self, fill=2.50, submitted_at="2026-08-20T15:00:00Z"):
        self.fill = fill
        self.submitted_at = submitted_at
        self.sent = []

    def _resp(self, symbol, side):
        return {"id": f"oid-{symbol}-{side}", "submitted_at": self.submitted_at}

    def submit_order(self, *, symbol, qty, side, **k):
        self.sent.append((side, symbol, qty))
        return self._resp(symbol, side)

    def submit_option_order(self, *, symbol, qty, side, **k):
        self.sent.append((side, symbol, qty))
        return self._resp(symbol, side)

    def get_option_quotes(self, symbols, **k):
        return {"quotes": {symbols: {"bp": 2.45, "ap": 2.55}}}

    def get_order(self, _oid):
        if self.fill is None:
            return {"status": "new", "filled_avg_price": None, "filled_qty": "0"}
        return {"status": "filled", "filled_avg_price": self.fill,
                "filled_at": "2026-08-20T15:00:02Z", "filled_qty": "5"}


def _open_market(monkeypatch):
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: True)
    monkeypatch.setattr("core.live_readiness.filter_entry_orders_for_readiness",
                        lambda plan, **k: (plan, [], "ready"))


# --- 1. the entry clock ---------------------------------------------------------

def test_entry_records_submit_time_fill_time_price_and_order_id(monkeypatch, tmp_path):
    _open_market(monkeypatch)
    client = _Client(fill=2.50)
    managed = {"ABC": {"route": "option", "occ": "ABC260918C00010000", "contracts": 5}}
    execute_plan(client, plan=[("ABC260918C00010000", "buy", 5, "entry", "option")],
                 limits={}, submit=True, equity_tif_fn=lambda: "day",
                 new_managed=managed, module="m", bar="2026-08-20 14:00:00+00:00",
                 ledger_root=str(tmp_path))
    st = managed["ABC"]
    assert st["entry_order_id"] == "oid-ABC260918C00010000-buy"
    assert st["entry_submitted_at"] == "2026-08-20T15:00:00Z"
    assert st["entry_filled_at"] == "2026-08-20T15:00:02Z"
    assert st["entry_fill_price"] == 2.50
    assert st["entry_filled_qty"] == 5.0
    assert st["pending_fill"] is False


def test_an_unfilled_entry_stays_pending_and_records_no_fill(monkeypatch, tmp_path):
    """The ladder case. Accepted is not filled, and the flag must survive."""
    _open_market(monkeypatch)
    client = _Client(fill=None)
    managed = {"ABC": {"route": "option", "occ": "ABC260918C00010000", "contracts": 5}}
    execute_plan(client, plan=[("ABC260918C00010000", "buy", 5, "entry", "option")],
                 limits={}, submit=True, equity_tif_fn=lambda: "day",
                 new_managed=managed, module="m", bar="b", ledger_root=str(tmp_path))
    st = managed["ABC"]
    assert st["pending_fill"] is True
    assert st["entry_fill_price"] is None
    assert st["entry_order_id"]          # the id is still recorded


def test_entry_clock_reaches_the_closed_trade_ledger(monkeypatch, tmp_path):
    """A closed trade must say when it was OPENED, not only when it was closed."""
    _open_market(monkeypatch)
    client = _Client(fill=3.10)
    entry_state = {"route": "option", "occ": "ABC260918C00010000", "contracts": 5,
                   "entry_bar": "2026-08-18 14:00:00+00:00", "runs_held": 3,
                   "entry_order_id": "oid-entry-1",
                   "entry_submitted_at": "2026-08-18T18:20:00Z",
                   "entry_filled_at": "2026-08-18T18:20:01Z",
                   "entry_fill_price": 2.00, "entry_filled_qty": 5.0,
                   "u_entry": 11.4, "u_atr": 0.62}
    execute_plan(client, plan=[("ABC260918C00010000", "sell", 5, "take_profit_full_+30%", "option")],
                 limits={"ABC260918C00010000": 3.10}, submit=True, equity_tif_fn=lambda: "day",
                 new_managed={}, exit_context={"ABC260918C00010000": ("ABC", entry_state)},
                 module="m", pos_lookup={"ABC260918C00010000": {"avg_entry": 2.00}},
                 bar="2026-08-20 14:00:00+00:00", ledger_root=str(tmp_path))
    row = json.loads((tmp_path / "m" / "closed_trades.jsonl").read_text().strip())
    assert row["entry_order_id"] == "oid-entry-1"
    assert row["entry_submitted_at"] == "2026-08-18T18:20:00Z"
    assert row["entry_filled_at"] == "2026-08-18T18:20:01Z"
    assert row["entry_fill_price"] == 2.00
    assert row["u_entry"] == 11.4 and row["u_atr"] == 0.62


# --- 2. trim lineage ------------------------------------------------------------

def test_a_trim_keeps_its_entry_lineage(monkeypatch, tmp_path):
    """A partial take-profit is not in exit_context; managed state supplies it."""
    _open_market(monkeypatch)
    client = _Client(fill=3.10)
    managed = {"ABC": {"route": "option", "occ": "ABC260918C00010000", "contracts": 10,
                       "entry_bar": "2026-08-18 14:00:00+00:00", "runs_held": 3,
                       "entry_order_id": "oid-entry-9", "entry_fill_price": 2.00}}
    execute_plan(client, plan=[("ABC260918C00010000", "sell", 5, "take_profit_+30%", "option")],
                 limits={"ABC260918C00010000": 3.10}, submit=True, equity_tif_fn=lambda: "day",
                 new_managed=managed, exit_context={},   # <- the trim is absent, by design
                 module="m", pos_lookup={"ABC260918C00010000": {"avg_entry": 2.00}},
                 bar="2026-08-20 14:00:00+00:00", ledger_root=str(tmp_path))
    row = json.loads((tmp_path / "m" / "closed_trades.jsonl").read_text().strip())
    assert row["exit_reason"] == "take_profit_+30%"
    assert row["entry_bar"] == "2026-08-18 14:00:00+00:00"   # was null before
    assert row["runs_held"] == 3
    assert row["entry_order_id"] == "oid-entry-9"


# --- 3. dispositions ------------------------------------------------------------

def test_a_filled_entry_is_labelled_filled(monkeypatch, tmp_path):
    _open_market(monkeypatch)
    disp: dict[str, str] = {}
    execute_plan(_Client(fill=2.50), plan=[("ABC260918C00010000", "buy", 5, "entry", "option")],
                 limits={}, submit=True, equity_tif_fn=lambda: "day",
                 new_managed={"ABC": {"route": "option", "occ": "ABC260918C00010000"}},
                 module="m", bar="b", ledger_root=str(tmp_path), dispositions=disp)
    assert disp["ABC260918C00010000"] == "filled"


def test_an_accepted_but_unfilled_entry_is_distinguishable(monkeypatch, tmp_path):
    """The 54%-conversion question: 'submitted' and 'filled' are not the same."""
    _open_market(monkeypatch)
    disp: dict[str, str] = {}
    execute_plan(_Client(fill=None), plan=[("ABC260918C00010000", "buy", 5, "entry", "option")],
                 limits={}, submit=True, equity_tif_fn=lambda: "day",
                 new_managed={"ABC": {"route": "option", "occ": "ABC260918C00010000"}},
                 module="m", bar="b", ledger_root=str(tmp_path), dispositions=disp)
    assert disp["ABC260918C00010000"] == "accepted_unfilled"


def test_a_deferred_entry_is_labelled_not_lost(monkeypatch, tmp_path):
    """After the close an entry is queued. It must not read as 'submitted'."""
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: False)
    disp: dict[str, str] = {}
    execute_plan(_Client(), plan=[("ABC", "buy", 100, "entry", "equity")],
                 limits={}, submit=True, equity_tif_fn=lambda: "day",
                 new_managed={"ABC": {"route": "equity", "symbol": "ABC"}},
                 module="m", bar="2026-08-20 18:00:00+00:00",
                 ledger_root=str(tmp_path), dispositions=disp)
    assert disp["ABC"] == "deferred_entry_market_closed"


def test_a_failed_submission_names_the_failure(monkeypatch, tmp_path):
    _open_market(monkeypatch)

    class _Boom(_Client):
        def submit_order(self, **k):
            raise RuntimeError("403 uncovered")

    disp: dict[str, str] = {}
    execute_plan(_Boom(), plan=[("ABC", "buy", 100, "entry", "equity")],
                 limits={}, submit=True, equity_tif_fn=lambda: "day",
                 new_managed={"ABC": {"route": "equity", "symbol": "ABC"}},
                 module="m", bar="b", ledger_root=str(tmp_path), dispositions=disp)
    assert disp["ABC"] == "submit_failed:RuntimeError"


def test_the_audit_record_carries_the_disposition_per_symbol():
    rec = order_plan_audit_record(
        module="m", bar="b", mode="options", submit=True, targets=["ABC"],
        plan=[("ABC", "buy", 100, "entry", "equity")],
        signal_audits={}, order_audits={}, contract_selection={},
        dispositions={"ABC": "accepted_unfilled"})
    assert rec["planned"][0]["disposition"] == "accepted_unfilled"
    assert rec["dispositions"] == {"ABC": "accepted_unfilled"}
