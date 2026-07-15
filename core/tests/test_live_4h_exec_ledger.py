"""Realized-PnL ledger + phantom-position reconciliation in the shared 4H engine.

Covers the two behaviours added so the 4H modules (Momentum, HTF Swing, Meta)
stop leaving phantom positions from rejected orders and finally record a true
realized PnL (broker cost basis vs exit fill) on every exit.
"""
from __future__ import annotations

import json

from core.live_4h_exec import (
    ExecPolicy,
    build_mixed_plan,
    drop_failed_entry,
    execute_plan,
    record_exit_realized_pnl,
)


class _FillClient:
    """Submits succeed; sells report a fill price; buys reject (after-hours 403)."""

    def __init__(self, *, fill_price="0.20", reject_buys=False):
        self.fill_price = fill_price
        self.reject_buys = reject_buys
        self.submitted = []

    def _submit(self, side, sym):
        self.submitted.append((side, sym))
        if side == "buy" and self.reject_buys:
            raise RuntimeError("HTTP Error 403: Forbidden")
        return {"id": f"{sym}-{side}", "status": "accepted"}

    def submit_option_order(self, *, symbol, qty, side, **k):
        return self._submit(side, symbol)

    def submit_order(self, *, symbol, qty, side, **k):
        return self._submit(side, symbol)

    def get_order(self, oid):
        return {"id": oid, "status": "filled", "filled_avg_price": self.fill_price}


def test_record_exit_realized_pnl_option_and_equity(tmp_path):
    c = _FillClient(fill_price="0.20")
    record_exit_realized_pnl(
        c, module="meta_ranker",
        item=("WEN260717C00009000", "sell", 10, "horizon", "option"),
        resp={"id": "o1"}, entry_state={"symbol": "WEN", "entry_bar": "2026-07-01"},
        pos_lookup={"WEN260717C00009000": {"avg_entry": 0.60}},
        bar="2026-07-08 18:00", ledger_root=str(tmp_path),
    )
    rows = [json.loads(l) for l in (tmp_path / "meta_ranker" / "closed_trades.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    # (0.20 - 0.60) * 100 * 10 = -400
    assert rows[0]["realized_pnl"] == -400.0
    assert rows[0]["ticker"] == "WEN" and rows[0]["exit_reason"] == "horizon"


def test_record_exit_ignores_buys(tmp_path):
    c = _FillClient()
    record_exit_realized_pnl(
        c, module="momentum_expansion",
        item=("RXT", "buy", 100, "entry", "equity"),
        resp={"id": "o2"}, entry_state=None, pos_lookup={}, bar="x", ledger_root=str(tmp_path),
    )
    assert not (tmp_path / "momentum_expansion" / "closed_trades.jsonl").exists()


def test_drop_failed_entry_equity_and_option():
    nm = {"RXT": {"qty": 100}, "WEN": {"occ": "WEN260717C00009000", "contracts": 10}}
    drop_failed_entry(nm, "RXT")                       # equity keyed by ticker
    drop_failed_entry(nm, "WEN260717C00009000")        # option keyed by ticker, matched via occ
    assert nm == {}


def test_execute_plan_rejected_entry_not_tracked(monkeypatch):
    # An after-hours 403 buy must NOT stay in managed (no phantom position),
    # while a successful entry stays tracked.
    monkeypatch.setattr("core.live_4h_exec.filter_entry_orders_for_readiness",
                        lambda plan, new_managed=None: (plan, [], ""))
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: True)  # test the submit path, not the after-close defer
    c = _FillClient(reject_buys=True)
    new_managed = {"AAA": {"qty": 100, "route": "equity"},   # this buy 403s
                   "GOOD": {"qty": 100, "route": "equity"}}  # this buy fills
    c_good = _FillClient(reject_buys=False)

    def _submit(side, sym):
        c.submitted.append((side, sym))
        if sym == "AAA":
            raise RuntimeError("HTTP Error 403: Forbidden")
        return {"id": f"{sym}-{side}", "status": "accepted"}

    c.submit_order = lambda **k: _submit(k["side"], k["symbol"])
    plan = [("AAA", "buy", 100, "entry", "equity"),
            ("GOOD", "buy", 100, "entry", "equity")]
    failed = execute_plan(
        c, plan=plan, limits={}, submit=True, equity_tif_fn=lambda: "day",
        new_managed=new_managed, exit_context={},
        module="momentum_expansion", pos_lookup={}, bar="2026-07-08 18:00",
    )
    assert failed == {"AAA"}
    assert "AAA" not in new_managed        # phantom entry dropped
    assert "GOOD" in new_managed           # filled entry retained


def test_build_mixed_plan_skip_route_does_not_buy_equity():
    def _skip_route(client, ticker, px, **kwargs):
        return "skip", {"option_type": "call"}, "no_non_0dte_call_contracts"

    res = build_mixed_plan(
        _FillClient(),
        targets=["AAA"],
        managed={},
        pos_info={},
        bar="2026-07-09T19:45:00Z",
        signal_audits={},
        policy=ExecPolicy(contracts=1, shares=100),
        route_fn=_skip_route,
        ref_price_fn=lambda ticker: 10.0,
        verbose=False,
    )
    assert res.plan == []
    assert res.new_managed == {}
    assert res.contract_selection["AAA"]["action"] == "skip"
    assert res.contract_selection["AAA"]["reason"] == "no_non_0dte_call_contracts"


def test_build_mixed_plan_persists_authoritative_open_mark():
    managed = {
        "AAA": {
            "route": "equity",
            "symbol": "AAA",
            "shares": 100,
            "runs_held": 0,
            "bars_out": 0,
            "trimmed": False,
            "entry_bar": "2026-07-13 14:00:00+00:00",
        }
    }
    res = build_mixed_plan(
        _FillClient(),
        targets=["AAA"],
        managed=managed,
        pos_info={"AAA": {"qty": 100, "avg_entry": 10.0, "current": 10.5}},
        bar="2026-07-13 18:00:00+00:00",
        signal_audits={},
        policy=ExecPolicy(contracts=1, shares=100),
        route_fn=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no entry route")),
        ref_price_fn=lambda ticker: 10.0,
        verbose=False,
    )

    state = res.new_managed["AAA"]
    assert state["entry_avg_price"] == 10.0
    assert state["last_mark_price"] == 10.5
    assert state["last_mark_bar"] == "2026-07-13 18:00:00+00:00"
    assert round(state["unrealized_gain"], 8) == 0.05
