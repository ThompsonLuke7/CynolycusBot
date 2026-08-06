"""Closing option orders: market closes, and exits that outlive their quote.

Two rules meet here, and they pull in opposite directions.

*Entries are hard.* Opening a position on a quote we could not read is opening
risk we cannot price, so an opening leg must always carry a real two-sided
market and the time it was observed.

*Exits are soft.* A position we are trying to get out of must not be trapped by
a failed quote fetch. A degraded exit is allowed through, but it is recorded as
degraded — the reason travels with the order rather than being lost, so nobody
later mistakes an unpriced close for a priced one.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest
from pydantic import ValidationError

from core.nervous_system.contracts.enums import (
    DebitCredit,
    DecisionKind,
    InstrumentFamily,
    OptionType,
    OrderSide,
    PositionIntent,
    QuoteAssurance,
    RuntimeEnvironment,
)
from core.nervous_system.contracts.orders import OptionLeg, OrderRequest


NOW = datetime(2026, 8, 4, 18, 20, tzinfo=timezone.utc)
OCC = "AMD260821C00200000"


def _leg(**updates: Any) -> OptionLeg:
    payload: dict[str, Any] = {
        "symbol": OCC,
        "underlying": "AMD",
        "option_type": OptionType.CALL,
        "strike": Decimal("200"),
        "expiration": date(2026, 8, 21),
        "side": OrderSide.SELL,
        "ratio": 1,
        "position_intent": PositionIntent.SELL_TO_CLOSE,
        "quote_at": NOW,
        "bid": Decimal("4.10"),
        "ask": Decimal("4.30"),
    }
    payload.update(updates)
    return OptionLeg(**payload)


def _degraded_leg(**updates: Any) -> OptionLeg:
    payload: dict[str, Any] = {
        "quote_at": None,
        "bid": None,
        "ask": None,
        "quote_degraded_reason": "no_quote_timestamp",
    }
    payload.update(updates)
    return _leg(**payload)


def _request(**updates: Any) -> OrderRequest:
    payload: dict[str, Any] = {
        "decision_id": uuid5(NAMESPACE_URL, "close-test/decision"),
        "policy_decision_id": uuid5(NAMESPACE_URL, "close-test/policy"),
        "environment": RuntimeEnvironment.QA_PAPER,
        "account_alias": "paper",
        "decision_kind": DecisionKind.EXIT,
        "risk_reducing": True,
        "broker_position_key": "paper:AMD",
        "instrument_family": InstrumentFamily.SINGLE_OPTION,
        "legs": (_leg(),),
        "parent_quantity": Decimal("3"),
        "debit_credit": DebitCredit.CREDIT,
        "net_limit_price": Decimal("4.10"),
        "maximum_loss": Decimal("0"),
        "buying_power_required": Decimal("0"),
        "time_in_force": "day",
        "order_type": "limit",
        "idempotency_key": "ab" * 32,
        "created_at": NOW,
        "expires_at": NOW + timedelta(minutes=20),
    }
    payload.update(updates)
    return OrderRequest.create(**payload)


# ---------------------------------------------------------------------------
# Entries stay hard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "intent", [PositionIntent.BUY_TO_OPEN, PositionIntent.SELL_TO_OPEN]
)
def test_an_opening_leg_may_never_omit_its_quote(intent: PositionIntent) -> None:
    """Opening risk we cannot price is never acceptable, whatever the reason."""

    with pytest.raises(ValidationError, match="opening"):
        _degraded_leg(
            position_intent=intent,
            side=(
                OrderSide.BUY if intent is PositionIntent.BUY_TO_OPEN else OrderSide.SELL
            ),
        )


def test_an_opening_leg_may_not_carry_a_degradation_reason() -> None:
    with pytest.raises(ValidationError):
        _leg(
            position_intent=PositionIntent.BUY_TO_OPEN,
            side=OrderSide.BUY,
            quote_degraded_reason="no_quote",
        )


# ---------------------------------------------------------------------------
# Exits may be soft, but never quietly
# ---------------------------------------------------------------------------


def test_a_closing_leg_may_omit_its_quote_when_the_reason_is_recorded() -> None:
    leg = _degraded_leg()

    assert leg.bid is None
    assert leg.ask is None
    assert leg.quote_at is None
    assert leg.quote_degraded_reason == "no_quote_timestamp"


def test_a_degraded_leg_is_reported_as_degraded() -> None:
    assert _degraded_leg().quote_assurance is QuoteAssurance.DEGRADED
    assert _leg().quote_assurance is QuoteAssurance.QUOTED


def test_an_unquoted_leg_without_a_reason_is_refused() -> None:
    """Silence is the one thing that must not be allowed: an unpriced close
    with no explanation is indistinguishable from a bug.
    """

    with pytest.raises(ValidationError, match="reason"):
        _leg(bid=None, ask=None, quote_at=None)


def test_a_quoted_leg_may_not_claim_to_be_degraded() -> None:
    """Mislabelling in the safe direction is still mislabelling; it would make
    the degraded-exit rate meaningless.
    """

    with pytest.raises(ValidationError, match="degrad"):
        _leg(quote_degraded_reason="no_quote")


@pytest.mark.parametrize(
    "partial", [{"bid": None}, {"ask": None}, {"quote_at": None}]
)
def test_a_half_missing_quote_is_refused(partial: dict[str, Any]) -> None:
    """Either we have the whole observed market or we have none of it. A leg
    with a bid but no ask is a market nobody observed.

    Asserted without a degradation reason on purpose: with one present, the
    mislabelling rule would reject it first and this test would pass without
    ever exercising the all-or-nothing rule.
    """

    with pytest.raises(ValidationError, match="together, or none"):
        _leg(**partial)


def test_a_request_reports_degraded_when_any_leg_is() -> None:
    quoted = _request()
    degraded = _request(legs=(_degraded_leg(),), order_type="market", net_limit_price=None)

    assert quoted.quote_assurance is QuoteAssurance.QUOTED
    assert degraded.quote_assurance is QuoteAssurance.DEGRADED


# ---------------------------------------------------------------------------
# Market closes
# ---------------------------------------------------------------------------


def test_a_market_close_of_a_single_option_leg_is_allowed() -> None:
    """Alpaca accepts single-leg option market sells during RTH, and the swing
    position manager already relies on that as its exit fallback when the limit
    ladder does not fill. The contract has to be able to express it.
    """

    request = _request(order_type="market", net_limit_price=None)

    assert request.order_type == "market"
    assert request.net_limit_price is None
    assert request.debit_credit is DebitCredit.CREDIT


def test_a_market_credit_order_that_opens_a_leg_is_still_refused() -> None:
    """The rule exists so a credit *spread* states the credit it expects to
    collect. Opening legs keep that requirement.
    """

    with pytest.raises(ValidationError, match="credit"):
        _request(
            order_type="market",
            net_limit_price=None,
            decision_kind=DecisionKind.ENTRY,
            risk_reducing=False,
            broker_position_key=None,
            legs=(
                _leg(
                    position_intent=PositionIntent.SELL_TO_OPEN,
                    side=OrderSide.SELL,
                ),
            ),
        )


def test_a_limit_close_still_carries_its_ladder_price() -> None:
    """The default exit path stays a limit at or below the bid; the market
    order is only the fallback.
    """

    request = _request(net_limit_price=Decimal("4.08"))

    assert request.order_type == "limit"
    assert request.net_limit_price == Decimal("4.08")


def test_a_limit_close_still_requires_a_positive_limit() -> None:
    with pytest.raises(ValidationError):
        _request(net_limit_price=Decimal("0"))


def test_a_degraded_close_cannot_be_a_limit_order() -> None:
    """With no quote there is no defensible limit price. Inventing one would be
    exactly the fabrication the degraded path exists to avoid.
    """

    with pytest.raises(ValidationError, match="degrad"):
        _request(legs=(_degraded_leg(),), order_type="limit", net_limit_price=Decimal("4.10"))


# ---------------------------------------------------------------------------
# Identity and hashing still hold
# ---------------------------------------------------------------------------


def test_the_request_hash_covers_the_degradation_reason() -> None:
    """Two orders that differ only in why they were unpriced are different
    orders, and the audit trail must be able to tell them apart.
    """

    first = _request(legs=(_degraded_leg(),), order_type="market", net_limit_price=None)
    second = _request(
        legs=(_degraded_leg(quote_degraded_reason="no_quote"),),
        order_type="market",
        net_limit_price=None,
    )

    assert first.request_hash != second.request_hash


def test_a_degraded_request_is_self_consistent() -> None:
    request = _request(legs=(_degraded_leg(),), order_type="market", net_limit_price=None)

    assert request.request_hash == request.computed_request_hash()


def test_an_equity_credit_order_still_requires_a_positive_limit() -> None:
    """`is_pure_close` must not be vacuously true for an order with no option
    legs at all. An equity request has an empty leg tuple, and `all([])` is
    True — which would quietly exempt every equity CREDIT order from the rule.
    """

    with pytest.raises(ValidationError, match="credit"):
        _request(
            instrument_family=InstrumentFamily.EQUITY,
            legs=(),
            equity_symbol="AMD",
            equity_side=OrderSide.SELL,
            order_type="market",
            net_limit_price=None,
            debit_credit=DebitCredit.CREDIT,
        )


def test_an_equity_request_is_never_reported_as_a_pure_close() -> None:
    request = _request(
        instrument_family=InstrumentFamily.EQUITY,
        legs=(),
        equity_symbol="AMD",
        equity_side=OrderSide.SELL,
        order_type="market",
        net_limit_price=None,
        debit_credit=DebitCredit.DEBIT,
    )

    assert request.is_pure_close is False
    assert request.quote_assurance is QuoteAssurance.QUOTED


# ---------------------------------------------------------------------------
# A degraded close still tries a limit when it has a real price
# ---------------------------------------------------------------------------


def test_a_degraded_close_may_be_a_limit_when_its_price_has_a_named_source() -> None:
    """Losing the bid/ask feed does not mean losing every price. A broker
    position mark is a real observation, so the exit should still try to earn
    price improvement before resorting to a market order.
    """

    request = _request(
        legs=(_degraded_leg(),),
        order_type="limit",
        net_limit_price=Decimal("4.05"),
        net_limit_source="broker_position_mark",
    )

    assert request.quote_assurance is QuoteAssurance.DEGRADED
    assert request.order_type == "limit"
    assert request.net_limit_price == Decimal("4.05")
    assert request.net_limit_source == "broker_position_mark"


def test_a_degraded_limit_without_a_named_source_is_refused() -> None:
    """This is the line between a real fallback mark and an invented number.
    An unsourced limit on an unquoted close is a fabrication.
    """

    with pytest.raises(ValidationError, match="source"):
        _request(
            legs=(_degraded_leg(),),
            order_type="limit",
            net_limit_price=Decimal("4.05"),
        )


def test_a_degraded_close_with_no_price_at_all_is_a_market_order() -> None:
    request = _request(
        legs=(_degraded_leg(),), order_type="market", net_limit_price=None
    )

    assert request.order_type == "market"
    assert request.net_limit_price is None
    assert request.net_limit_source is None


def test_a_market_order_may_not_claim_a_limit_source() -> None:
    """A market order has no price, so a price provenance on it is meaningless
    and would pollute the audit trail.
    """

    with pytest.raises(ValidationError, match="source"):
        _request(
            legs=(_degraded_leg(),),
            order_type="market",
            net_limit_price=None,
            net_limit_source="broker_position_mark",
        )


def test_the_limit_source_is_part_of_the_request_identity() -> None:
    first = _request(
        legs=(_degraded_leg(),),
        order_type="limit",
        net_limit_price=Decimal("4.05"),
        net_limit_source="broker_position_mark",
    )
    second = _request(
        legs=(_degraded_leg(),),
        order_type="limit",
        net_limit_price=Decimal("4.05"),
        net_limit_source="last_known_mid",
    )

    assert first.request_hash != second.request_hash
    assert first.request_hash == first.computed_request_hash()
