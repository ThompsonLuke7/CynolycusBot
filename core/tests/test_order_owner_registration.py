"""Every accepted order of an owned client is registered to its module (2026-11-24).

The certificate read 7 of 127 fills as unattributed: SPY day trader (both sides) and
intraday-structure entries never registered, and the 30m swing had no registration at all.
"""
from __future__ import annotations

import json

import pytest

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
from core.broker_fill_reconciliation import reconcile_account

pytestmark = pytest.mark.safe


def _client(monkeypatch, owner, response=None):
    obj = object.__new__(AlpacaOptionsClient)
    obj._trading_base = "https://paper-api.alpaca.markets"
    obj.owner = owner
    sent = {}

    def _request(method, url, *, json_body=None, **_kw):
        sent["payload"] = json_body
        return response if response is not None else {"id": "ord-1", "status": "accepted"}

    monkeypatch.setattr(obj, "_request", _request, raising=False)
    obj.sent = sent
    return obj


def _registry(root):
    path = root / "Data/inference/broker_reconciliation/paper/order_registry.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_owned_client_registers_and_names_the_owner_at_the_broker(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    c = _client(monkeypatch, "spy_day_trader")
    c.submit_option_order(symbol="SPY261125C00719000", qty=1, side="buy", order_type="limit", limit_price=3.4)
    (row,) = _registry(tmp_path)
    assert row["order_id"] == "ord-1" and row["module"] == "spy_day_trader"
    assert row["route"] == "option" and row["side"] == "buy" and row["ledger"] == "none"
    assert c.sent["payload"]["client_order_id"].startswith("spy_day_trader-")
    assert row["client_order_id"] == c.sent["payload"]["client_order_id"]


def test_caller_client_order_id_is_kept(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    c = _client(monkeypatch, "multi_ticker_swing_30m")
    c.submit_order(symbol="AAPL", qty=5, side="sell", client_order_id="mine-1")
    assert c.sent["payload"]["client_order_id"] == "mine-1"
    assert _registry(tmp_path)[0]["route"] == "equity"


def test_unowned_client_is_unchanged(monkeypatch, tmp_path):
    """The shared 4H engine registers its own orders: its client must not register again."""
    monkeypatch.chdir(tmp_path)
    c = _client(monkeypatch, None)
    c.submit_order(symbol="AAPL", qty=5, side="buy")
    assert "client_order_id" not in c.sent["payload"]
    assert _registry(tmp_path) == []


def test_registration_failure_never_fails_the_order(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    c = _client(monkeypatch, "spy_day_trader", response={"status": "accepted"})  # no id
    assert c.submit_order(symbol="SPY", qty=1, side="buy") == {"status": "accepted"}
    assert _registry(tmp_path) == []


def test_owned_sell_is_attributed_without_a_ledger_comparison(monkeypatch, tmp_path):
    """SPY/30m keep their own books (no exit_fills.jsonl): attributed, not a ledger mismatch."""
    monkeypatch.chdir(tmp_path)
    c = _client(monkeypatch, "spy_day_trader", response={"id": "ord-9"})
    c.submit_option_order(symbol="SPY261125C00719000", qty=1, side="sell")

    class Broker:
        def get_account_activities(self, **_kw):
            return [{"id": "a1", "activity_type": "FILL", "order_id": "ord-9",
                     "symbol": "SPY261125C00719000", "side": "sell", "qty": "1", "price": "3.07"}]

    cert = reconcile_account(Broker(), root=tmp_path / "Data/inference")
    assert cert["status"] == "PASS", cert


def test_intraday_structure_entry_is_registered(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import strategies.intraday_structure.execution as execution
    from strategies.intraday_structure.config import ExecutionPolicy

    monkeypatch.setattr(execution, "is_market_open_now", lambda *_a, **_k: True)

    class Client:
        _trading_base = "https://paper-api.alpaca.markets"   # the account label comes from it

        def submit_option_order(self, *, symbol, qty, side, **_k):
            return {"id": f"oid-{symbol}", "filled_avg_price": "1.0", "filled_qty": str(qty)}

    def contract(_client, ticker, price, *, option_type, **_k):
        return ({"occ": f"{ticker}260904C00010000", "limit": 1.0, "mid": 0.98,
                 "strike": 10.0, "expiry": "2026-09-04"}, "ok")

    policy = ExecutionPolicy(enabled=True, target_notional=1000.0, state_path=str(tmp_path / "open.json"))
    ex = execution.IntradayOptionExecutor(Client(), policy, select_option_fn=contract, ledger_root=str(tmp_path))
    setup = SimpleNamespace(setup_id="ABC:long:x", ticker="ABC", direction="long", setup_type="x",
                            entry_price=10.0, spot=10.2, invalidation=9.8, metadata={})
    ex.on_entry(setup, spot=10.0)
    rows = [json.loads(l) for l in
            (tmp_path / "broker_reconciliation/paper/order_registry.jsonl").read_text().splitlines()]
    assert [(r["module"], r["side"], r["reason"], r["route"], r["order_id"]) for r in rows] == [
        ("intraday_structure", "buy", "entry", "option", "oid-ABC260904C00010000")]


def test_order_fills_joins_registry_and_broker_fills(tmp_path):
    from core.broker_fill_reconciliation import order_fills, register_order_ownership
    from core.live_state import append_once

    for oid, side, qty in [("o1", "buy", 10), ("o2", "sell", 10), ("o3", "buy", 5)]:
        register_order_ownership(order_id=oid, module="m", symbol="AAPL", side=side, qty=qty,
                                 route="equity", root=tmp_path)
    journal = tmp_path / "broker_reconciliation/paper/fills.jsonl"
    for aid, oid, q, px, t in [("a1", "o1", 4, 10.0, "2026-11-24T15:00:00Z"),
                               ("a2", "o1", 6, 11.0, "2026-11-24T15:00:05Z"),
                               ("a3", "o2", 3, 12.0, "2026-11-24T16:00:00Z")]:
        append_once(journal, {"activity": {"id": aid, "order_id": oid, "qty": str(q), "price": str(px),
                                           "transaction_time": t}}, identity=f"fill:{aid}")
    rows = {r["order_id"]: r for r in order_fills(root=tmp_path)}
    assert rows["o1"]["status"] == "filled" and rows["o1"]["avg_fill_price"] == pytest.approx(10.6)
    assert rows["o1"]["first_fill_at"] == "2026-11-24T15:00:00Z" and rows["o1"]["n_fills"] == 2
    assert rows["o2"]["status"] == "partial" and rows["o2"]["filled_qty"] == 3
    assert rows["o3"]["status"] == "unfilled" and rows["o3"]["avg_fill_price"] is None
