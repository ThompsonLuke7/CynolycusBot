"""The account-wide entry gate: what it refuses, what it never touches."""
from __future__ import annotations

import json
import urllib.error

import pytest

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
from core.trading_controls import (
    TradingControls,
    load_controls,
    order_refusal,
    write_controls,
)

FROZEN = TradingControls(entries_frozen=True, reason="test freeze", source="test")
OPEN = TradingControls(source="test")


def _no_position(symbol):
    return None


def _account(**fields):
    base = {"equity": "100000", "long_market_value": "100000", "maintenance_margin": "30000"}
    base.update(fields)
    return lambda: base


def _refuse(payload, controls, *, position=_no_position, account=None):
    return order_refusal(
        payload, get_position=position, get_account=account or _account(), controls=controls
    )


# --------------------------------------------------------------------------
# control file
# --------------------------------------------------------------------------


def test_no_file_means_entries_open_and_no_limits(tmp_path) -> None:
    ctl = load_controls(tmp_path / "missing.json")
    assert not ctl.entries_frozen
    assert ctl.max_gross_leverage is None and ctl.min_margin_cushion_pct is None


def test_written_freeze_round_trips(tmp_path) -> None:
    path = write_controls(
        entries="frozen", reason="margin", set_at_utc="2026-11-03T07:00:00Z",
        max_gross_leverage=1.5, path=tmp_path / "c.json",
    )
    ctl = load_controls(path)
    assert ctl.entries_frozen and ctl.reason == "margin"
    assert ctl.max_gross_leverage == 1.5 and ctl.min_margin_cushion_pct is None


@pytest.mark.parametrize(
    "content",
    ["{not json", json.dumps({"entries": "maybe"}), json.dumps({"limits": {"max_gross_leverage": -1}})],
)
def test_an_unreadable_file_freezes_entries(tmp_path, content) -> None:
    path = tmp_path / "c.json"
    path.write_text(content, encoding="utf-8")
    ctl = load_controls(path)
    assert ctl.entries_frozen
    assert ctl.reason.startswith("unreadable trading controls file")


# --------------------------------------------------------------------------
# what counts as opening exposure
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"symbol": "AMD", "qty": 10, "side": "sell"},
        {"symbol": "AMD261120C00200000", "qty": 2, "side": "sell", "position_intent": "sell_to_close"},
        {"symbol": "AMD261120P00200000", "qty": 1, "side": "buy", "position_intent": "buy_to_close"},
    ],
)
def test_a_freeze_never_blocks_an_exit(payload) -> None:
    assert _refuse(payload, FROZEN) is None


@pytest.mark.parametrize(
    "payload",
    [
        {"symbol": "VSTM", "qty": 506, "side": "buy"},
        {"symbol": "TWST261120C00175000", "qty": 2, "side": "buy"},
        {"symbol": "AMD261120C00200000", "qty": 2, "side": "buy", "position_intent": "buy_to_open"},
        {"symbol": "AMD261120P00180000", "qty": 1, "side": "sell", "position_intent": "sell_to_open"},
        {
            "order_class": "mleg", "qty": 1,
            "legs": [
                {"symbol": "A261120C00010000", "side": "buy", "position_intent": "buy_to_open", "ratio_qty": 1},
                {"symbol": "A261120C00012000", "side": "sell", "position_intent": "sell_to_open", "ratio_qty": 1},
            ],
        },
    ],
)
def test_a_freeze_blocks_every_opening_order(payload) -> None:
    refusal = _refuse(payload, FROZEN)
    assert refusal is not None and refusal["code"] == "entries_frozen"
    assert "test freeze" in refusal["message"]


def test_a_buy_covering_an_existing_short_passes_a_freeze() -> None:
    """The SPY daytrader flattens assigned shares with a plain buy."""

    short = lambda symbol: {"symbol": symbol, "qty": "-100"}  # noqa: E731
    assert _refuse({"symbol": "SPY", "qty": 100, "side": "buy"}, FROZEN, position=short) is None
    # Buying more than the short would open a long.
    refusal = _refuse({"symbol": "SPY", "qty": 150, "side": "buy"}, FROZEN, position=short)
    assert refusal is not None


def test_a_failed_position_lookup_is_treated_as_opening() -> None:
    def broken(symbol):
        raise urllib.error.URLError("dns")

    refusal = _refuse({"symbol": "SPY", "qty": 100, "side": "buy"}, FROZEN, position=broken)
    assert refusal is not None and "position lookup failed" in refusal["message"]


def test_open_entries_and_no_limits_never_reads_the_broker() -> None:
    def explode(*_args):
        raise AssertionError("no broker read is needed")

    assert order_refusal(
        {"symbol": "AMD", "qty": 1, "side": "buy"},
        get_position=explode, get_account=explode, controls=OPEN,
    ) is None


# --------------------------------------------------------------------------
# account limits (entries open)
# --------------------------------------------------------------------------


def test_gross_leverage_limit_refuses_an_entry_at_the_cap() -> None:
    ctl = TradingControls(max_gross_leverage=1.5, source="test")
    at_cap = _account(equity="100000", long_market_value="150000")
    refusal = _refuse({"symbol": "AMD", "qty": 1, "side": "buy"}, ctl, account=at_cap)
    assert refusal["code"] == "gross_leverage_limit"
    under = _account(equity="100000", long_market_value="149000")
    assert _refuse({"symbol": "AMD", "qty": 1, "side": "buy"}, ctl, account=under) is None


def test_margin_cushion_limit_uses_maintenance_margin() -> None:
    ctl = TradingControls(min_margin_cushion_pct=0.15, source="test")
    # The 2026-11-03 01:31 account: $371.8k equity, $357.2k maintenance.
    thin = _account(equity="371825.63", long_market_value="886767.59", maintenance_margin="357224.81")
    refusal = _refuse({"symbol": "AMD", "qty": 1, "side": "buy"}, ctl, account=thin)
    assert refusal["code"] == "margin_cushion_limit"


def test_limits_never_block_an_exit() -> None:
    ctl = TradingControls(max_gross_leverage=1.0, min_margin_cushion_pct=0.5, source="test")
    assert _refuse({"symbol": "AMD", "qty": 1, "side": "sell"}, ctl) is None


def test_an_unreadable_account_refuses_an_entry_under_limits() -> None:
    def down():
        raise urllib.error.URLError("timed out")

    ctl = TradingControls(max_gross_leverage=2.0, source="test")
    refusal = _refuse({"symbol": "AMD", "qty": 1, "side": "buy"}, ctl, account=down)
    assert refusal["code"] == "risk_limit_unverified"


# --------------------------------------------------------------------------
# the client hook
# --------------------------------------------------------------------------


class _Response:
    def __init__(self, payload):
        self._body = json.dumps(payload)

    def read(self):
        return self._body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


@pytest.fixture
def client() -> AlpacaOptionsClient:
    instance = AlpacaOptionsClient.__new__(AlpacaOptionsClient)
    instance._key = "PKTEST"
    instance._secret = "SECRET"
    instance._trading_base = "https://paper-api.alpaca.markets"
    instance._data_base = "https://data.alpaca.markets"
    instance._timeout = 5
    return instance


def _freeze(tmp_path, monkeypatch) -> None:
    path = write_controls(
        entries="frozen", reason="salvage mode", set_at_utc="2026-11-03T07:00:00Z",
        path=tmp_path / "controls.json",
    )
    monkeypatch.setenv("CYNOLYCUS_TRADING_CONTROLS", str(path))


def test_client_refuses_a_buy_without_posting_it(client, tmp_path, monkeypatch) -> None:
    _freeze(tmp_path, monkeypatch)
    sent = []

    def transport(request, timeout=None):
        sent.append((request.get_method(), request.full_url))
        if request.get_method() == "GET" and "/v2/positions/" in request.full_url:
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)
        return _Response({"id": "should-not-exist"})

    monkeypatch.setattr("core.API.Alpaca_API.options.options_api.urllib.request.urlopen", transport)
    with pytest.raises(urllib.error.HTTPError) as caught:
        client.submit_order(symbol="VSTM", qty=506, side="buy")
    assert caught.value.code == 403
    assert "entries_frozen" in str(caught.value.reason)
    assert [method for method, _ in sent] == ["GET"]


def test_client_sends_a_sell_while_frozen(client, tmp_path, monkeypatch) -> None:
    _freeze(tmp_path, monkeypatch)
    sent = []

    def transport(request, timeout=None):
        sent.append(request.get_method())
        return _Response({"id": "exit-1"})

    monkeypatch.setattr("core.API.Alpaca_API.options.options_api.urllib.request.urlopen", transport)
    assert client.submit_order(symbol="JDZG", qty=1399, side="sell") == {"id": "exit-1"}
    assert sent == ["POST"]


def test_client_without_a_controls_file_is_unchanged(client, monkeypatch) -> None:
    sent = []

    def transport(request, timeout=None):
        sent.append(request.get_method())
        return _Response({"id": "entry-1"})

    monkeypatch.setattr("core.API.Alpaca_API.options.options_api.urllib.request.urlopen", transport)
    assert client.submit_order(symbol="AMD", qty=1, side="buy") == {"id": "entry-1"}
    assert sent == ["POST"]
