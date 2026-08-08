"""Option orders through the governed path (Task 23, increment 5).

Entries are hard: they always carry the two-sided market select_option
observed. Exits are soft: a failed quote fetch must not trap a position, so a
close degrades to a market order carrying the reason rather than being blocked
or priced from an invented number.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest

from core.nervous_system.contracts.enums import (
    DebitCredit,
    DecisionKind,
    InstrumentFamily,
    OptionType,
    OrderSide,
    PositionIntent,
    QuoteAssurance,
)
from core.nervous_system.execution.options.quotes import OptionQuote
from signals.meta_context.meta_ranker.gateway_execution import option_order_request


NOW = datetime(2026, 8, 3, 20, 0, tzinfo=timezone.utc)
OCC = "AMD260821C00200000"
D = Decimal


def _quote(**updates: Any) -> OptionQuote:
    payload: dict[str, Any] = {
        "symbol": OCC,
        "underlying": "AMD",
        "option_type": OptionType.CALL,
        "strike": D("200"),
        "expiration": date(2026, 8, 21),
        "quote_at": NOW - timedelta(minutes=1),
        "bid": D("4.00"),
        "ask": D("4.40"),
    }
    payload.update(updates)
    return OptionQuote(**payload)


def _request(**updates: Any):
    payload: dict[str, Any] = {
        "decision_id": uuid5(NAMESPACE_URL, "opt/decision"),
        "policy_decision_id": uuid5(NAMESPACE_URL, "opt/policy"),
        "environment": __import__(
            "core.nervous_system.contracts.enums", fromlist=["RuntimeEnvironment"]
        ).RuntimeEnvironment.QA_PAPER,
        "account_alias": "paper",
        "decision_kind": DecisionKind.ENTRY,
        "symbol": OCC,
        "underlying": "AMD",
        "side": OrderSide.BUY,
        "quantity": D("3"),
        "risk_reducing": False,
        "broker_position_key": None,
        "quote": _quote(),
        "degraded_reason": None,
        "maximum_loss": D("1320"),
        "buying_power_required": D("1320"),
        "idempotency_key": "ab" * 32,
        "created_at": NOW,
        "expires_at": NOW + timedelta(minutes=20),
    }
    payload.update(updates)
    return option_order_request(**payload)


def _exit(**updates: Any):
    payload: dict[str, Any] = {
        "decision_kind": DecisionKind.EXIT,
        "side": OrderSide.SELL,
        "risk_reducing": True,
        "broker_position_key": "paper:" + OCC,
        "maximum_loss": D("0"),
        "buying_power_required": D("0"),
    }
    payload.update(updates)
    return _request(**payload)


# --- entries are hard ------------------------------------------------------


def test_an_entry_buys_to_open_at_the_observed_ask() -> None:
    """Executable long-option cost is the ask, not the mid: the mid is a
    mark-to-market basis, and paying it is an assumption nobody filled.
    """

    request = _request()

    assert request.legs[0].position_intent is PositionIntent.BUY_TO_OPEN
    assert request.order_type == "limit"
    assert request.net_limit_price == D("4.40")
    assert request.debit_credit is DebitCredit.DEBIT
    assert request.instrument_family is InstrumentFamily.SINGLE_OPTION


def test_an_entry_leg_carries_the_observed_market() -> None:
    leg = _request().legs[0]

    assert leg.bid == D("4.00")
    assert leg.ask == D("4.40")
    assert leg.quote_at == NOW - timedelta(minutes=1)
    assert leg.quote_degraded_reason is None


def test_an_entry_without_a_quote_is_refused() -> None:
    """Opening risk we cannot price is never acceptable."""

    # Matched on the builder's own message: the leg contract independently
    # refuses this too, so a loose match would pass without ever exercising
    # the guard here.
    with pytest.raises(ValueError, match="option_order_request: an opening"):
        _request(quote=None, degraded_reason="no_quote")


# --- exits are soft, but never quietly -------------------------------------


def test_an_exit_sells_to_close_at_the_top_ladder_rung() -> None:
    """The ladder starts at the mid and walks to the bid; the first rung is
    what the order carries, and the market close is a later fallback.
    """

    request = _exit()

    assert request.legs[0].position_intent is PositionIntent.SELL_TO_CLOSE
    assert request.debit_credit is DebitCredit.CREDIT
    assert request.order_type == "limit"
    assert request.net_limit_price == D("4.20")  # mid of 4.00/4.40


def test_a_quoteless_exit_becomes_a_market_close_carrying_its_reason() -> None:
    """A failed fetch must not trap a position. The exit still goes, and the
    reason travels with it so nobody mistakes it for a priced close.
    """

    request = _exit(quote=None, degraded_reason="no_quote_timestamp")

    assert request.order_type == "market"
    assert request.net_limit_price is None
    assert request.quote_assurance is QuoteAssurance.DEGRADED
    assert request.legs[0].quote_degraded_reason == "no_quote_timestamp"


def test_a_quoteless_exit_without_a_reason_is_refused() -> None:
    with pytest.raises(ValueError, match="option_order_request: an unquoted"):
        _exit(quote=None, degraded_reason=None)


def test_an_exit_carries_its_broker_position_key() -> None:
    assert _exit().broker_position_key == "paper:" + OCC


def test_the_request_hash_is_self_consistent() -> None:
    for request in (_request(), _exit(), _exit(quote=None, degraded_reason="no_quote")):
        assert request.request_hash == request.computed_request_hash()
