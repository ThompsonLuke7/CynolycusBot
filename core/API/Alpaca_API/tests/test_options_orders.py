"""HTTP payload tests for the Alpaca options client (Task 19).

Every test mocks the ``urllib`` boundary.  Nothing here reads process
credentials or contacts Alpaca.
"""

from __future__ import annotations

import json
import urllib.error
from decimal import Decimal
from typing import Any

import pytest

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient


KEY = "PKTESTKEYID000000000"
SECRET = "TESTSECRETVALUE0000000000000000000000000"


class _Response:
    def __init__(self, payload: Any, *, raw: str | None = None) -> None:
        self._body = raw if raw is not None else json.dumps(payload)

    def read(self) -> bytes:
        return self._body.encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _Transport:
    """Records every request instead of performing it."""

    def __init__(self, *responses: Any) -> None:
        self.requests: list[Any] = []
        self._responses = list(responses)
        self.calls = 0

    def __call__(self, request, timeout=None):  # noqa: ANN001 - urlopen shape
        self.calls += 1
        self.requests.append(request)
        if not self._responses:
            return _Response({})
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, _Response):
            return result
        return _Response(result)

    @property
    def last(self):
        return self.requests[-1]

    def body(self, index: int = -1) -> dict[str, Any]:
        data = self.requests[index].data
        return json.loads(data.decode("utf-8")) if data else {}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> AlpacaOptionsClient:
    """A client with injected credentials; never reads the environment."""

    instance = AlpacaOptionsClient.__new__(AlpacaOptionsClient)
    instance._key = KEY
    instance._secret = SECRET
    instance._trading_base = "https://paper-api.alpaca.markets"
    instance._data_base = "https://data.alpaca.markets"
    instance._timeout = 5
    return instance


def _patch(monkeypatch: pytest.MonkeyPatch, transport: _Transport) -> None:
    monkeypatch.setattr(
        "core.API.Alpaca_API.options.options_api.urllib.request.urlopen", transport
    )


# --------------------------------------------------------------------------
# client_order_id on existing submit paths
# --------------------------------------------------------------------------


def test_equity_submit_sends_client_order_id(client, monkeypatch) -> None:
    transport = _Transport({"id": "abc"})
    _patch(monkeypatch, transport)

    client.submit_order(
        symbol="AMD", qty=10, side="buy", client_order_id="cyn-0001"
    )

    body = transport.body()
    assert body["client_order_id"] == "cyn-0001"
    assert body["symbol"] == "AMD"
    assert body["qty"] == 10
    assert transport.last.get_method() == "POST"


def test_option_submit_sends_client_order_id_and_intent(client, monkeypatch) -> None:
    transport = _Transport({"id": "abc"})
    _patch(monkeypatch, transport)

    client.submit_option_order(
        symbol="AMD260918C00200000",
        qty=2,
        side="buy",
        order_type="limit",
        limit_price=10.05,
        position_intent="buy_to_open",
        client_order_id="cyn-0002",
    )

    body = transport.body()
    assert body["client_order_id"] == "cyn-0002"
    assert body["position_intent"] == "buy_to_open"
    assert body["limit_price"] == 10.05


def test_existing_callers_are_unaffected(client, monkeypatch) -> None:
    """Omitting client_order_id must not add the key to the payload."""

    transport = _Transport({"id": "abc"})
    _patch(monkeypatch, transport)

    client.submit_order(symbol="AMD", qty=1, side="buy")

    assert "client_order_id" not in transport.body()


# --------------------------------------------------------------------------
# Multi-leg
# --------------------------------------------------------------------------


def _legs(count: int = 2) -> list[dict[str, Any]]:
    template = [
        {
            "symbol": "AMD260918C00200000",
            "ratio_qty": 1,
            "side": "buy",
            "position_intent": "buy_to_open",
        },
        {
            "symbol": "AMD260918C00210000",
            "ratio_qty": 1,
            "side": "sell",
            "position_intent": "sell_to_open",
        },
        {
            "symbol": "AMD260918P00190000",
            "ratio_qty": 1,
            "side": "buy",
            "position_intent": "buy_to_open",
        },
        {
            "symbol": "AMD260918P00180000",
            "ratio_qty": 1,
            "side": "sell",
            "position_intent": "sell_to_open",
        },
    ]
    return template[:count]


@pytest.mark.parametrize("leg_count", [1, 2, 3, 4])
def test_multileg_sends_mleg_order_class(client, monkeypatch, leg_count) -> None:
    transport = _Transport({"id": "abc"})
    _patch(monkeypatch, transport)

    client.submit_multileg_order(
        legs=_legs(leg_count),
        qty=3,
        order_type="limit",
        time_in_force="day",
        limit_price=Decimal("4.70"),
        client_order_id="cyn-mleg",
    )

    body = transport.body()
    assert body["order_class"] == "mleg"
    assert body["qty"] == 3
    assert body["type"] == "limit"
    assert body["time_in_force"] == "day"
    assert body["limit_price"] == "4.70"
    assert len(body["legs"]) == leg_count
    # A multi-leg parent carries no symbol or side of its own.
    assert "symbol" not in body
    assert "side" not in body
    for leg in body["legs"]:
        assert set(leg) == {"symbol", "ratio_qty", "side", "position_intent"}


def test_multileg_credit_uses_a_negative_limit_price(client, monkeypatch) -> None:
    transport = _Transport({"id": "abc"})
    _patch(monkeypatch, transport)

    client.submit_multileg_order(
        legs=_legs(2),
        qty=1,
        order_type="limit",
        time_in_force="day",
        limit_price=Decimal("-1.25"),
        client_order_id="cyn-credit",
    )

    assert transport.body()["limit_price"] == "-1.25"


def test_multileg_ratios_are_reduced_to_gcd_one(client, monkeypatch) -> None:
    transport = _Transport({"id": "abc"})
    _patch(monkeypatch, transport)

    legs = _legs(2)
    legs[0]["ratio_qty"] = 4
    legs[1]["ratio_qty"] = 2
    client.submit_multileg_order(
        legs=legs,
        qty=1,
        order_type="market",
        time_in_force="day",
        limit_price=None,
        client_order_id="cyn-ratio",
    )

    assert [leg["ratio_qty"] for leg in transport.body()["legs"]] == [2, 1]


def test_multileg_quantity_must_be_whole_contracts(client, monkeypatch) -> None:
    transport = _Transport({"id": "abc"})
    _patch(monkeypatch, transport)

    with pytest.raises(ValueError, match="positive whole number"):
        client.submit_multileg_order(
            legs=_legs(2),
            qty=0,
            order_type="market",
            time_in_force="day",
            limit_price=None,
            client_order_id="cyn",
        )
    assert transport.calls == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"legs": [], "match": None}, "at least one leg"),
        ({"legs": _legs(4) + _legs(1)}, "at most four legs"),
        ({"order_type": "stop"}, "unsupported multi-leg order type"),
        ({"order_type": "limit", "limit_price": None}, "requires limit_price"),
        ({"order_type": "market", "limit_price": Decimal("1")}, "cannot carry limit_price"),
        ({"client_order_id": "  "}, "client_order_id is required"),
    ],
)
def test_unsupported_multileg_combinations_are_rejected_before_http(
    client, monkeypatch, kwargs, match
) -> None:
    transport = _Transport({"id": "abc"})
    _patch(monkeypatch, transport)

    payload = {
        "legs": _legs(2),
        "qty": 1,
        "order_type": "limit",
        "time_in_force": "day",
        "limit_price": Decimal("1.00"),
        "client_order_id": "cyn",
    }
    payload.update({k: v for k, v in kwargs.items() if k != "match"})
    with pytest.raises(ValueError, match=match):
        client.submit_multileg_order(**payload)
    assert transport.calls == 0, "an invalid request must never reach the network"


def test_multileg_rejects_non_positive_ratio(client, monkeypatch) -> None:
    transport = _Transport({"id": "abc"})
    _patch(monkeypatch, transport)

    legs = _legs(2)
    legs[0]["ratio_qty"] = 0
    with pytest.raises(ValueError, match="ratio_qty must be positive"):
        client.submit_multileg_order(
            legs=legs,
            qty=1,
            order_type="market",
            time_in_force="day",
            limit_price=None,
            client_order_id="cyn",
        )
    assert transport.calls == 0


# --------------------------------------------------------------------------
# Lookup, cancel, replace
# --------------------------------------------------------------------------


def test_lookup_uses_the_documented_client_order_id_endpoint(client, monkeypatch) -> None:
    transport = _Transport({"id": "abc", "client_order_id": "cyn-1"})
    _patch(monkeypatch, transport)

    result = client.get_order_by_client_order_id("cyn-1")

    assert result["client_order_id"] == "cyn-1"
    url = transport.last.full_url
    assert "/v2/orders:by_client_order_id" in url
    assert "client_order_id=cyn-1" in url
    assert transport.last.get_method() == "GET"


def test_unknown_client_order_id_returns_none(client, monkeypatch) -> None:
    error = urllib.error.HTTPError(
        "https://paper-api.alpaca.markets", 404, "Not Found", None, None
    )
    transport = _Transport(error)
    _patch(monkeypatch, transport)

    assert client.get_order_by_client_order_id("missing") is None


def test_replace_uses_patch(client, monkeypatch) -> None:
    transport = _Transport({"id": "abc"})
    _patch(monkeypatch, transport)

    client.replace_order("order-1", {"qty": 2, "limit_price": "4.50"})

    assert transport.last.get_method() == "PATCH"
    assert transport.last.full_url.endswith("/v2/orders/order-1")
    assert transport.body() == {"qty": 2, "limit_price": "4.50"}


def test_replace_refuses_structural_fields(client, monkeypatch) -> None:
    transport = _Transport({"id": "abc"})
    _patch(monkeypatch, transport)

    with pytest.raises(ValueError, match="does not support fields"):
        client.replace_order("order-1", {"legs": []})
    assert transport.calls == 0


def test_cancel_sends_delete_and_returns_no_body(client, monkeypatch) -> None:
    """DELETE returns 204 with no body; the caller must refetch, not assume."""

    transport = _Transport(_Response(None, raw=""))
    _patch(monkeypatch, transport)

    assert client.cancel_order("order-1") is None
    assert transport.last.get_method() == "DELETE"


# --------------------------------------------------------------------------
# Retry and credential safety
# --------------------------------------------------------------------------


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://paper-api.alpaca.markets/v2/orders", code, "Server Error", None, None
    )


def test_post_5xx_is_surfaced_and_never_retried(client, monkeypatch) -> None:
    """A 5xx POST may already have been accepted; retrying could double-submit."""

    transport = _Transport(_http_error(503))
    _patch(monkeypatch, transport)

    with pytest.raises(urllib.error.HTTPError):
        client.submit_order(symbol="AMD", qty=1, side="buy")

    assert transport.calls == 1


def test_patch_5xx_is_never_retried(client, monkeypatch) -> None:
    transport = _Transport(_http_error(502))
    _patch(monkeypatch, transport)

    with pytest.raises(urllib.error.HTTPError):
        client.replace_order("order-1", {"qty": 1})

    assert transport.calls == 1


def test_multileg_post_5xx_is_never_retried(client, monkeypatch) -> None:
    transport = _Transport(_http_error(500))
    _patch(monkeypatch, transport)

    with pytest.raises(urllib.error.HTTPError):
        client.submit_multileg_order(
            legs=_legs(2),
            qty=1,
            order_type="market",
            time_in_force="day",
            limit_price=None,
            client_order_id="cyn",
        )

    assert transport.calls == 1


def test_get_5xx_is_retried(client, monkeypatch) -> None:
    transport = _Transport(_http_error(503), {"ok": True})
    _patch(monkeypatch, transport)
    monkeypatch.setattr(
        "core.API.Alpaca_API.options.options_api.time.sleep", lambda _seconds: None
    )

    assert client.get_account() == {"ok": True}
    assert transport.calls == 2


def test_error_text_redacts_credentials(client, monkeypatch) -> None:
    class _Leaky(urllib.error.HTTPError):
        def read(self) -> bytes:
            return f"denied for key {KEY} secret {SECRET}".encode("utf-8")

    error = _Leaky(
        f"https://paper-api.alpaca.markets/v2/orders?key={KEY}",
        403,
        "Forbidden",
        None,
        None,
    )
    transport = _Transport(error)
    _patch(monkeypatch, transport)

    with pytest.raises(urllib.error.HTTPError) as caught:
        client.submit_order(symbol="AMD", qty=1, side="buy")

    text = f"{caught.value} {caught.value.url} {caught.value.reason}"
    assert KEY not in text
    assert SECRET not in text
    assert "***redacted***" in text
