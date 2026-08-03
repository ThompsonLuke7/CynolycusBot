"""Fill-backed ownership and broker reconciliation tests (Task 16)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest

from core.nervous_system.contracts.enums import (
    AssetClass,
    DebitCredit,
    DecisionKind,
    ExecutionStatus,
    InstrumentFamily,
    OrderSide,
    OwnershipStatus,
    ReconciliationStatus,
    RuntimeEnvironment,
)
from core.nervous_system.contracts.execution import ExecutionEvent
from core.nervous_system.contracts.orders import OrderRequest
from core.nervous_system.contracts.portfolio import OwnershipRecord
from core.nervous_system.contracts.states import PortfolioPosition
from core.nervous_system.portfolio.ownership import (
    OwnershipError,
    assign_fill_ownership,
    broker_position_key,
    net_owned_quantity,
)
from core.nervous_system.portfolio.reconciliation import reconcile_portfolio
from core.nervous_system.tests.test_portfolio_exposure import portfolio_state


UTC = timezone.utc
NOW = datetime(2026, 7, 30, 18, 30, tzinfo=UTC)
DECISION_ID = uuid5(NAMESPACE_URL, "ownership-test/decision")
RECONCILIATION_ID = uuid5(NAMESPACE_URL, "ownership-test/reconciliation")


def build_order(
    *,
    symbol: str = "AMD",
    side: OrderSide = OrderSide.BUY,
    quantity: Decimal = Decimal("100"),
    decision_kind: DecisionKind = DecisionKind.ENTRY,
    risk_reducing: bool = False,
    broker_position_key_value: str | None = None,
    order_request_id: UUID | None = None,
) -> OrderRequest:
    return OrderRequest.create(
        order_request_id=order_request_id or uuid5(NAMESPACE_URL, f"ownership-test/order/{symbol}/{side.value}"),
        decision_id=DECISION_ID,
        policy_decision_id=uuid5(NAMESPACE_URL, "ownership-test/policy"),
        environment=RuntimeEnvironment.QA_PAPER,
        account_alias="paper",
        decision_kind=decision_kind,
        risk_reducing=risk_reducing,
        broker_position_key=broker_position_key_value,
        instrument_family=InstrumentFamily.EQUITY,
        equity_symbol=symbol,
        equity_side=side,
        parent_quantity=quantity,
        debit_credit=DebitCredit.DEBIT,
        net_limit_price=None,
        maximum_loss=Decimal("20000"),
        buying_power_required=Decimal("20000"),
        time_in_force="day",
        order_type="market",
        idempotency_key="f" * 64,
        created_at=NOW,
        expires_at=NOW + timedelta(minutes=20),
    )


def build_event(
    *,
    order: OrderRequest,
    status: ExecutionStatus,
    filled_quantity: Decimal,
    average_fill_price: Decimal | None,
    sequence_no: int = 1,
    observed_at: datetime | None = None,
    event_id: UUID | None = None,
) -> ExecutionEvent:
    return ExecutionEvent.create(
        execution_event_id=event_id
        or uuid5(NAMESPACE_URL, f"ownership-test/event/{order.order_request_id}/{sequence_no}"),
        order_request_id=order.order_request_id,
        event_type=status.value,
        sequence_no=sequence_no,
        status=status,
        observed_at=observed_at or NOW,
        broker_event_at=None,
        client_order_id="cli-1",
        broker_order_id="brk-1",
        broker_parent_order_id=None,
        filled_quantity=filled_quantity,
        average_fill_price=average_fill_price,
        leg_reports=(),
        sanitized_response={},
        previous_event_id=None,
        previous_event_hash=None,
    )


# --------------------------------------------------------------------------
# Step 2: ownership is only ever created by a confirmed fill
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        ExecutionStatus.PLANNED,
        ExecutionStatus.SUBMISSION_PENDING,
        ExecutionStatus.ACCEPTED,
        ExecutionStatus.REJECTED,
        ExecutionStatus.CANCELED,
    ],
)
def test_unfilled_orders_create_no_ownership(status: ExecutionStatus) -> None:
    order = build_order()
    event = build_event(
        order=order, status=status, filled_quantity=Decimal("0"), average_fill_price=None
    )

    with pytest.raises(OwnershipError, match="confirmed fill"):
        assign_fill_ownership(event, order, strategy_id="meta_ranker")


def test_filled_buy_creates_positive_assigned_ownership() -> None:
    order = build_order()
    event = build_event(
        order=order,
        status=ExecutionStatus.FILLED,
        filled_quantity=Decimal("100"),
        average_fill_price=Decimal("200.00"),
    )

    record = assign_fill_ownership(event, order, strategy_id="meta_ranker")

    assert record.ownership_status is OwnershipStatus.ASSIGNED
    assert record.quantity == Decimal("100")
    assert record.account_alias == "paper"
    assert record.broker_position_key == "paper:AMD"
    assert record.decision_record_id == DECISION_ID
    assert record.order_request_id == order.order_request_id
    assert record.source_fill_id == event.execution_event_id
    assert record.effective_at == event.observed_at
    assert record.ended_at is None


def test_partial_fill_allocates_only_the_filled_quantity() -> None:
    order = build_order(quantity=Decimal("100"))
    event = build_event(
        order=order,
        status=ExecutionStatus.PARTIALLY_FILLED,
        filled_quantity=Decimal("30"),
        average_fill_price=Decimal("200.00"),
    )

    record = assign_fill_ownership(event, order, strategy_id="meta_ranker")

    assert record.quantity == Decimal("30")
    assert record.ownership_status is OwnershipStatus.ASSIGNED


def test_sell_fill_creates_negative_ownership_delta() -> None:
    order = build_order(
        side=OrderSide.SELL,
        decision_kind=DecisionKind.EXIT,
        risk_reducing=True,
        broker_position_key_value="paper:AMD",
    )
    event = build_event(
        order=order,
        status=ExecutionStatus.FILLED,
        filled_quantity=Decimal("40"),
        average_fill_price=Decimal("210.00"),
    )

    record = assign_fill_ownership(event, order, strategy_id="meta_ranker")

    assert record.quantity == Decimal("-40")
    assert record.broker_position_key == "paper:AMD"


def test_ownership_identity_is_deterministic_per_fill() -> None:
    order = build_order()
    event = build_event(
        order=order,
        status=ExecutionStatus.FILLED,
        filled_quantity=Decimal("100"),
        average_fill_price=Decimal("200.00"),
    )

    assert assign_fill_ownership(event, order, strategy_id="meta_ranker") == assign_fill_ownership(event, order, strategy_id="meta_ranker")


def test_event_from_another_order_is_rejected() -> None:
    order = build_order()
    other = build_order(symbol="NVDA")
    event = build_event(
        order=other,
        status=ExecutionStatus.FILLED,
        filled_quantity=Decimal("10"),
        average_fill_price=Decimal("100.00"),
    )

    with pytest.raises(OwnershipError, match="another order"):
        assign_fill_ownership(event, order, strategy_id="meta_ranker")


def test_option_ownership_keys_on_the_occ_symbol() -> None:
    assert broker_position_key("paper", "AMD261218C00200000") == (
        "paper:AMD261218C00200000"
    )


def test_ownership_requires_an_explicit_strategy() -> None:
    order = build_order()
    event = build_event(
        order=order,
        status=ExecutionStatus.FILLED,
        filled_quantity=Decimal("100"),
        average_fill_price=Decimal("200.00"),
    )

    with pytest.raises(OwnershipError, match="strategy_id"):
        assign_fill_ownership(event, order, strategy_id="  ")


def test_net_owned_quantity_never_goes_negative_from_overclosing() -> None:
    records = (
        _record(quantity=Decimal("100"), suffix="open"),
        _record(quantity=Decimal("-40"), suffix="trim"),
        _record(quantity=Decimal("-80"), suffix="overclose"),
    )

    assert net_owned_quantity(records) == Decimal("0")


def test_net_owned_quantity_sums_deterministically() -> None:
    records = (
        _record(quantity=Decimal("100"), suffix="open"),
        _record(quantity=Decimal("-40"), suffix="trim"),
    )

    assert net_owned_quantity(records) == Decimal("60")
    assert net_owned_quantity(tuple(reversed(records))) == Decimal("60")


def _record(
    *,
    quantity: Decimal,
    suffix: str,
    key: str = "paper:AMD",
    strategy_id: str | None = "meta_ranker",
    status: OwnershipStatus = OwnershipStatus.ASSIGNED,
) -> OwnershipRecord:
    assigned = status is not OwnershipStatus.UNASSIGNED
    return OwnershipRecord(
        ownership_id=uuid5(NAMESPACE_URL, f"ownership-test/record/{key}/{suffix}"),
        account_alias="paper",
        broker_position_key=key,
        strategy_id=strategy_id if assigned else None,
        decision_record_id=DECISION_ID if assigned else None,
        order_request_id=uuid5(NAMESPACE_URL, "ownership-test/order") if assigned else None,
        source_fill_id=uuid5(NAMESPACE_URL, f"ownership-test/fill/{suffix}") if assigned else None,
        quantity=quantity,
        effective_at=NOW,
        ownership_status=status,
    )


def test_unassigned_ownership_cannot_claim_a_strategy() -> None:
    with pytest.raises(ValueError, match="cannot name a strategy"):
        OwnershipRecord(
            ownership_id=uuid5(NAMESPACE_URL, "ownership-test/bad"),
            account_alias="paper",
            broker_position_key="paper:AMD",
            strategy_id="meta_ranker",
            decision_record_id=None,
            order_request_id=None,
            source_fill_id=None,
            quantity=Decimal("10"),
            effective_at=NOW,
            ownership_status=OwnershipStatus.UNASSIGNED,
        )


def test_assigned_ownership_requires_fill_lineage() -> None:
    with pytest.raises(ValueError, match="requires a source fill"):
        OwnershipRecord(
            ownership_id=uuid5(NAMESPACE_URL, "ownership-test/bad2"),
            account_alias="paper",
            broker_position_key="paper:AMD",
            strategy_id="meta_ranker",
            decision_record_id=DECISION_ID,
            order_request_id=uuid5(NAMESPACE_URL, "ownership-test/order"),
            source_fill_id=None,
            quantity=Decimal("10"),
            effective_at=NOW,
            ownership_status=OwnershipStatus.ASSIGNED,
        )


# --------------------------------------------------------------------------
# Reconciliation
# --------------------------------------------------------------------------


def _position(symbol: str, quantity: float) -> PortfolioPosition:
    return PortfolioPosition(
        broker_position_id=f"pos-{symbol}",
        symbol=symbol,
        underlying=symbol,
        asset_class=AssetClass.EQUITY,
        quantity=quantity,
        market_value=quantity * 200.0,
    )


def _reconcile(positions, records):
    return reconcile_portfolio(
        portfolio_state(positions=positions),
        records,
        reconciliation_id=RECONCILIATION_ID,
    )


def test_exact_agreement_is_matched() -> None:
    result = _reconcile(
        (_position("AMD", 100.0),),
        (_record(quantity=Decimal("100"), suffix="open"),),
    )

    assert len(result.matched) == 1
    line = result.matched[0]
    assert line.status is ReconciliationStatus.MATCHED
    assert line.broker_quantity == Decimal("100")
    assert line.owned_quantity == Decimal("100")
    assert line.strategy_ids == ("meta_ranker",)
    assert result.content_hash == result.computed_content_hash()


def test_partial_attribution_is_reported_as_partial() -> None:
    result = _reconcile(
        (_position("AMD", 100.0),),
        (_record(quantity=Decimal("60"), suffix="open"),),
    )

    assert len(result.partial) == 1
    assert result.partial[0].owned_quantity == Decimal("60")
    assert result.partial[0].broker_quantity == Decimal("100")


def test_manual_broker_position_is_unassigned() -> None:
    result = _reconcile((_position("JPM", 20.0),), ())

    assert len(result.unassigned) == 1
    assert result.unassigned[0].broker_position_key == "paper:JPM"
    assert result.unassigned[0].owned_quantity == Decimal("0")


def test_ownership_without_a_broker_position_is_orphaned() -> None:
    result = _reconcile((), (_record(quantity=Decimal("100"), suffix="open"),))

    assert len(result.orphaned) == 1
    assert result.orphaned[0].broker_quantity == Decimal("0")
    assert result.orphaned[0].owned_quantity == Decimal("100")


def test_ownership_exceeding_broker_quantity_is_a_quantity_mismatch() -> None:
    result = _reconcile(
        (_position("AMD", 50.0),),
        (_record(quantity=Decimal("100"), suffix="open"),),
    )

    assert len(result.quantity_mismatches) == 1
    line = result.quantity_mismatches[0]
    assert line.broker_quantity == Decimal("50")
    assert line.owned_quantity == Decimal("100")


def test_reconciliation_never_mutates_broker_facts() -> None:
    positions = (_position("AMD", 50.0),)
    portfolio = portfolio_state(positions=positions)
    records = (_record(quantity=Decimal("100"), suffix="open"),)

    reconcile_portfolio(portfolio, records, reconciliation_id=RECONCILIATION_ID)

    assert portfolio.positions[0].quantity == 50.0
    assert records[0].quantity == Decimal("100")


def test_direction_disagreement_is_a_quantity_mismatch_not_a_match() -> None:
    result = _reconcile(
        (_position("AMD", -100.0),),
        (_record(quantity=Decimal("100"), suffix="open"),),
    )

    assert result.matched == ()
    assert len(result.quantity_mismatches) == 1


def test_smaller_long_ownership_against_a_short_position_is_never_partial() -> None:
    """Opposite signs are a conflict even when the owned magnitude is smaller."""

    result = _reconcile(
        (_position("AMD", -100.0),),
        (_record(quantity=Decimal("60"), suffix="open"),),
    )

    assert result.partial == ()
    assert len(result.quantity_mismatches) == 1
    line = result.quantity_mismatches[0]
    assert line.broker_quantity == Decimal("-100")
    assert line.owned_quantity == Decimal("60")


def test_reconciliation_is_deterministic_and_order_independent() -> None:
    positions = (_position("AMD", 100.0), _position("JPM", 20.0))
    records = (
        _record(quantity=Decimal("60"), suffix="a"),
        _record(quantity=Decimal("40"), suffix="b"),
    )

    forward = _reconcile(positions, records)
    reverse = _reconcile(tuple(reversed(positions)), tuple(reversed(records)))

    assert forward.content_hash == reverse.content_hash
    assert len(forward.matched) == 1
    assert len(forward.unassigned) == 1


def test_closed_ownership_is_excluded_from_attribution() -> None:
    records = (
        _record(quantity=Decimal("100"), suffix="open"),
        _record(
            quantity=Decimal("100"),
            suffix="closed",
            status=OwnershipStatus.CLOSED,
        ),
    )
    result = _reconcile((_position("AMD", 100.0),), records)

    assert len(result.matched) == 1
    assert result.matched[0].owned_quantity == Decimal("100")
