"""Execution gateway identity, sequence, and refusal tests (Task 21)."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from uuid import NAMESPACE_URL, uuid5

import pytest

from core.nervous_system.contracts.enums import (
    DecisionKind,
    ExecutionStatus,
    OrderSide,
    RuntimeEnvironment,
    SubmissionAttemptStatus,
)
from core.nervous_system.execution.broker import (
    BrokerAmbiguousSubmission,
    BrokerRejected,
    BrokerUnavailable,
)
from core.nervous_system.execution.gateway import (
    CLIENT_ORDER_ID_LENGTH,
    ENVIRONMENT_CODES,
    ExecutionGateway,
    ExecutionOutcome,
    GatewayConflict,
    GatewayPreflightError,
    check_exit_reduces_exposure,
    client_order_id_for,
    order_request_id_for,
)
from core.nervous_system.persistence.repositories.execution import (
    SubmissionConflict,
    is_legal_transition,
)
from core.nervous_system.persistence.uow import UnitOfWork
from core.nervous_system.tests.fixtures.gateway_harness import (
    ACCOUNT,
    NOW,
    FakeBroker,
    FakeJournal,
    broker_order,
    broker_position,
    clock,
    decision_record,
    exit_request,
    expected_client_order_id,
    order_request,
)


D = Decimal


@pytest.fixture
def uow_factory(session_factory):
    return lambda: UnitOfWork(session_factory).__enter__()


def build_gateway(
    *,
    broker=None,
    journal=None,
    uow_factory=None,
    environment=RuntimeEnvironment.QA_PAPER,
    worker_id="worker-a",
    at=NOW,
):
    return ExecutionGateway(
        broker=broker or FakeBroker(),
        journal=journal or FakeJournal(),
        unit_of_work_factory=uow_factory or (lambda: _unavailable()),
        environment=environment,
        account_alias=ACCOUNT,
        worker_id=worker_id,
        clock=clock(at),
    )


def _unavailable():
    raise ConnectionError("postgres is down")


# --------------------------------------------------------------------------
# Deterministic identity
# --------------------------------------------------------------------------


def test_client_order_id_is_exactly_48_characters() -> None:
    request = order_request()
    value = client_order_id_for(request.environment, request.request_hash)

    assert len(value) == CLIENT_ORDER_ID_LENGTH
    assert value.startswith("cyno-qp-")
    assert value.isascii()


@pytest.mark.parametrize(
    ("environment", "code"),
    [
        (RuntimeEnvironment.DEVELOPMENT, "dv"),
        (RuntimeEnvironment.QA_PAPER, "qp"),
        (RuntimeEnvironment.PRODUCTION_LIVE, "pl"),
    ],
)
def test_environment_codes_are_two_characters(environment, code) -> None:
    assert ENVIRONMENT_CODES[environment] == code
    assert len(code) == 2
    value = client_order_id_for(environment, "a" * 64)
    assert len(value) == CLIENT_ORDER_ID_LENGTH


def test_identity_is_stable_across_rebuilds() -> None:
    first, second = order_request(), order_request()

    assert first.request_hash == second.request_hash
    assert first.order_request_id == second.order_request_id
    assert expected_client_order_id(first) == expected_client_order_id(second)


def test_changed_content_produces_a_new_identity() -> None:
    base = order_request()
    changed = order_request(parent_quantity=D("26"))

    assert changed.request_hash != base.request_hash
    assert changed.order_request_id != base.order_request_id
    assert expected_client_order_id(changed) != expected_client_order_id(base)


def test_order_request_id_is_derived_from_decision_and_content() -> None:
    request = order_request()

    assert request.order_request_id == order_request_id_for(
        request.decision_id, request.request_hash
    )


# --------------------------------------------------------------------------
# Preflight refusals (nothing durable, no broker call)
# --------------------------------------------------------------------------


def test_production_live_gateway_cannot_be_constructed() -> None:
    with pytest.raises(GatewayPreflightError, match="PRODUCTION_LIVE"):
        build_gateway(environment=RuntimeEnvironment.PRODUCTION_LIVE)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"account_alias": "other"}, "account alias"),
        ({"environment": RuntimeEnvironment.DEVELOPMENT}, "environment"),
    ],
)
def test_mismatched_request_is_refused_before_the_broker(kwargs, reason) -> None:
    broker = FakeBroker()
    gateway = build_gateway(broker=broker)

    result = gateway.submit(decision=decision_record(), request=order_request(**kwargs))

    assert result.outcome is ExecutionOutcome.REFUSED
    assert reason in result.detail
    assert broker.submit_calls == []


def test_expired_request_is_refused_before_the_broker() -> None:
    broker = FakeBroker()
    gateway = build_gateway(broker=broker, at=NOW + timedelta(hours=1))

    result = gateway.submit(decision=decision_record(), request=order_request())

    assert result.outcome is ExecutionOutcome.REFUSED
    assert "expired" in result.detail
    assert broker.submit_calls == []


def test_a_request_from_another_decision_is_refused() -> None:
    broker = FakeBroker()
    gateway = build_gateway(broker=broker)
    other = decision_record(
        decision_record_id=uuid5(NAMESPACE_URL, "gateway-test/other-decision")
    )

    result = gateway.submit(decision=other, request=order_request())

    assert result.outcome is ExecutionOutcome.REFUSED
    assert broker.submit_calls == []


def test_a_failed_decision_record_cannot_execute() -> None:
    broker = FakeBroker()
    gateway = build_gateway(broker=broker)
    failed = decision_record(
        status="FAILED",
        snapshot_id=None,
        intent_id=None,
        policy_decision_id=None,
        snapshot_hash=None,
        intent_hash=None,
        policy_hash=None,
        raw_strategy_output=None,
        exposure_report=None,
        instrument_candidates=None,
        instrument_selection=None,
        failure_stage="POLICY",
        failure_code="VETO",
        failure_message="rejected",
    )

    result = gateway.submit(decision=failed, request=order_request())

    assert result.outcome is ExecutionOutcome.REFUSED
    assert broker.submit_calls == []


def test_a_non_deterministic_order_id_is_refused() -> None:
    """A caller-supplied request ID must equal the canonical identity."""

    from core.nervous_system.contracts.orders import OrderRequest

    base = order_request()
    payload = base.model_dump()
    payload["order_request_id"] = uuid5(NAMESPACE_URL, "gateway-test/wrong-id")
    tampered = OrderRequest.create(
        **{
            key: value
            for key, value in payload.items()
            if key not in {"request_hash"}
        }
    )
    broker = FakeBroker()

    result = build_gateway(broker=broker).submit(
        decision=decision_record(), request=tampered
    )

    assert result.outcome is ExecutionOutcome.REFUSED
    assert "deterministic identity" in result.detail
    assert broker.submit_calls == []


# --------------------------------------------------------------------------
# PostgreSQL outage: entries fail closed
# --------------------------------------------------------------------------


def test_an_entry_fails_closed_when_postgres_is_down() -> None:
    broker = FakeBroker()
    gateway = build_gateway(broker=broker)

    result = gateway.submit(decision=decision_record(), request=order_request())

    assert result.outcome is ExecutionOutcome.REFUSED
    assert result.reason_code == "POSTGRES_UNAVAILABLE"
    assert broker.submit_calls == [], "an entry must never reach the broker without a DB"


def test_a_risk_reducing_exit_proceeds_when_postgres_is_down() -> None:
    request = exit_request()
    broker = FakeBroker(positions_result=(broker_position("AMD", 100.0),))
    journal = FakeJournal()
    gateway = build_gateway(broker=broker, journal=journal)

    result = gateway.submit(decision=decision_record(), request=request)

    assert result.outcome is ExecutionOutcome.RECONCILIATION_REQUIRED
    assert len(broker.submit_calls) == 1
    assert journal.types() == ["INTENT_TO_SUBMIT", "BROKER_RESPONSE"]


def test_a_fail_operational_exit_still_requires_a_durable_journal() -> None:
    broker = FakeBroker(positions_result=(broker_position("AMD", 100.0),))
    journal = FakeJournal(fail_on=("INTENT_TO_SUBMIT",))
    gateway = build_gateway(broker=broker, journal=journal)

    result = gateway.submit(decision=decision_record(), request=exit_request())

    assert result.outcome is ExecutionOutcome.REFUSED
    assert result.reason_code == "JOURNAL_NOT_DURABLE"
    assert broker.submit_calls == []


def test_a_fail_operational_exit_needs_authoritative_position_state() -> None:
    broker = FakeBroker(positions_result=BrokerUnavailable("no position read"))
    gateway = build_gateway(broker=broker)

    result = gateway.submit(decision=decision_record(), request=exit_request())

    assert result.outcome is ExecutionOutcome.REFUSED
    assert result.reason_code == "BROKER_POSITION_STATE_UNAVAILABLE"


# --------------------------------------------------------------------------
# Exit exposure guards, wired end to end
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("request_kwargs", "held", "reason"),
    [
        ({"parent_quantity": D("500")}, 100.0, "EXIT_EXCEEDS_HELD_QUANTITY"),
        ({}, 0.0, "EXIT_WITHOUT_A_HELD_POSITION"),
        ({"equity_side": OrderSide.BUY}, 100.0, "EXIT_WOULD_INCREASE_A_LONG"),
        ({"equity_side": OrderSide.SELL}, -100.0, "EXIT_WOULD_INCREASE_A_SHORT"),
    ],
)
def test_a_violating_exit_never_reaches_the_broker(
    request_kwargs, held, reason
) -> None:
    """The guard must be wired into submit, not merely available as a function.

    With PostgreSQL down there is no database check left, so this is the only
    thing standing between an oversized "exit" and a real position flip.
    """

    broker = FakeBroker(positions_result=(broker_position("AMD", held),))
    gateway = build_gateway(broker=broker)

    result = gateway.submit(
        decision=decision_record(), request=exit_request(**request_kwargs)
    )

    assert result.outcome is ExecutionOutcome.REFUSED
    assert result.reason_code == reason
    assert broker.submit_calls == [], "a non-reducing exit must never be sent"


def test_a_genuine_exit_is_still_allowed_through() -> None:
    broker = FakeBroker(positions_result=(broker_position("AMD", 100.0),))

    result = build_gateway(broker=broker).submit(
        decision=decision_record(), request=exit_request(parent_quantity=D("25"))
    )

    assert result.outcome is ExecutionOutcome.RECONCILIATION_REQUIRED
    assert len(broker.submit_calls) == 1


# --------------------------------------------------------------------------
# Exit exposure guard units
# --------------------------------------------------------------------------


def test_an_exit_may_not_exceed_the_held_quantity() -> None:
    request = exit_request(parent_quantity=D("500"))

    assert (
        check_exit_reduces_exposure(request, (broker_position("AMD", 100.0),))
        == "EXIT_EXCEEDS_HELD_QUANTITY"
    )


def test_an_exit_without_a_position_is_refused() -> None:
    assert (
        check_exit_reduces_exposure(exit_request(), ())
        == "EXIT_WITHOUT_A_HELD_POSITION"
    )


def test_a_disguised_short_entry_is_refused() -> None:
    """Selling when flat is an opening short, not an exit."""

    assert (
        check_exit_reduces_exposure(exit_request(), (broker_position("AMD", 0.0),))
        == "EXIT_WITHOUT_A_HELD_POSITION"
    )


def test_buying_against_a_long_is_not_an_exit() -> None:
    request = exit_request(equity_side=OrderSide.BUY)

    assert (
        check_exit_reduces_exposure(request, (broker_position("AMD", 100.0),))
        == "EXIT_WOULD_INCREASE_A_LONG"
    )


def test_selling_against_a_short_is_not_an_exit() -> None:
    request = exit_request(equity_side=OrderSide.SELL)

    assert (
        check_exit_reduces_exposure(request, (broker_position("AMD", -100.0),))
        == "EXIT_WOULD_INCREASE_A_SHORT"
    )


def test_covering_a_short_is_a_valid_exit() -> None:
    request = exit_request(equity_side=OrderSide.BUY)

    assert check_exit_reduces_exposure(request, (broker_position("AMD", -100.0),)) is None


def test_an_entry_is_never_treated_as_an_exit() -> None:
    assert (
        check_exit_reduces_exposure(order_request(), (broker_position("AMD", 100.0),))
        == "NOT_RISK_REDUCING"
    )


# --------------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("current", "target", "legal"),
    [
        (SubmissionAttemptStatus.RESERVED, SubmissionAttemptStatus.JOURNALED, True),
        (SubmissionAttemptStatus.JOURNALED, SubmissionAttemptStatus.SUBMITTING, True),
        (SubmissionAttemptStatus.SUBMITTING, SubmissionAttemptStatus.ACCEPTED, True),
        (SubmissionAttemptStatus.SUBMITTING, SubmissionAttemptStatus.AMBIGUOUS, True),
        (SubmissionAttemptStatus.AMBIGUOUS, SubmissionAttemptStatus.ACCEPTED, True),
        (
            SubmissionAttemptStatus.AMBIGUOUS,
            SubmissionAttemptStatus.RECONCILIATION_REQUIRED,
            True,
        ),
        # Backward and impossible moves.
        (SubmissionAttemptStatus.SUBMITTING, SubmissionAttemptStatus.RESERVED, False),
        (SubmissionAttemptStatus.ACCEPTED, SubmissionAttemptStatus.SUBMITTING, False),
        (SubmissionAttemptStatus.ACCEPTED, SubmissionAttemptStatus.REJECTED, False),
        (SubmissionAttemptStatus.REJECTED, SubmissionAttemptStatus.ACCEPTED, False),
        (SubmissionAttemptStatus.RESERVED, SubmissionAttemptStatus.ACCEPTED, False),
    ],
)
def test_only_legal_submission_transitions_are_permitted(current, target, legal) -> None:
    assert is_legal_transition(current, target) is legal
