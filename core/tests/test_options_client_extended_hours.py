"""The governed path's equity submit must not die on `extended_hours`.

2026-09-11 pre-open flush: all 14 of Meta Ranker's queued exits failed with
`AlpacaOptionsClient.submit_order() got an unexpected keyword argument
'extended_hours'`. `alpaca_adapter._submit_equity` has always passed that flag;
this client never accepted it, so no governed equity order could reach the
broker. The exits stayed queued and the positions stayed unmanaged.
"""
from __future__ import annotations

import pytest

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient


@pytest.fixture
def client(monkeypatch):
    """A client with the network boundary replaced, so only the payload is under test."""
    obj = object.__new__(AlpacaOptionsClient)
    obj._trading_base = "https://paper-api.example"
    sent = {}

    def _request(method, url, *, json_body=None, **_kw):
        sent.update({"method": method, "url": url, "payload": json_body})
        return {"id": "order-1", "status": "accepted"}

    monkeypatch.setattr(obj, "_request", _request, raising=False)
    obj.sent = sent
    return obj


def test_an_equity_sell_accepts_extended_hours(client):
    client.submit_order(symbol="METC", qty=100, side="sell", extended_hours=True)

    assert client.sent["payload"]["extended_hours"] is True
    assert client.sent["payload"]["symbol"] == "METC"


def test_the_flag_is_omitted_when_not_requested(client):
    """Existing call sites must produce the same payload they always did."""
    client.submit_order(symbol="METC", qty=100, side="sell")

    assert "extended_hours" not in client.sent["payload"]


def test_the_adapter_call_shape_is_accepted(client):
    """Exactly what `alpaca_adapter._submit_equity` passes."""
    client.submit_order(
        symbol="BE", qty=16, side="sell", order_type="market", time_in_force="day",
        extended_hours=False, limit_price=None, client_order_id="cid-1",
    )

    payload = client.sent["payload"]
    assert payload["client_order_id"] == "cid-1"
    assert "extended_hours" not in payload
    assert "limit_price" not in payload


def test_the_client_accepts_every_keyword_the_governed_adapter_passes():
    """The contract test whose absence let this ship.

    `alpaca_adapter._submit_equity` and `AlpacaOptionsClient.submit_order` are
    written in different packages and were never checked against each other, so
    a keyword the adapter always sent was simply not a parameter of the method
    it called. Read the adapter's actual call site rather than restating it, so
    this fails the next time the two drift.
    """
    import ast
    import inspect
    from pathlib import Path

    import core.nervous_system.execution.alpaca_adapter as adapter_mod

    source = Path(inspect.getfile(adapter_mod)).read_text()
    tree = ast.parse(source)

    submit_equity = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_submit_equity"
    )
    calls = [
        node for node in ast.walk(submit_equity)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "submit_order"
    ]
    assert calls, "expected _submit_equity to call submit_order"

    passed = {kw.arg for call in calls for kw in call.keywords if kw.arg is not None}
    accepted = set(inspect.signature(AlpacaOptionsClient.submit_order).parameters)

    assert passed, "expected keyword arguments at the call site"
    assert passed <= accepted, f"client cannot accept: {sorted(passed - accepted)}"
