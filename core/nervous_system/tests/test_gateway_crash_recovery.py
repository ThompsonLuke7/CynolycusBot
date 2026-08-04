"""Crash points, ambiguity, and concurrency (Task 21).

Flow tests use an in-memory unit of work so every crash point is cheap to
exercise. The concurrency primitives the flow depends on -- the lease only one
worker can hold and the compare-and-set a stale worker cannot win -- are also
asserted against the disposable PostgreSQL database, so the fake is never the
only evidence that they work.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from uuid import uuid4

import pytest

from core.nervous_system.contracts.enums import (
    ExecutionStatus,
    RuntimeEnvironment,
    SubmissionAttemptStatus,
)
from core.nervous_system.execution.broker import (
    BrokerAmbiguousSubmission,
    BrokerRejected,
    BrokerUnavailable,
)
from core.nervous_system.execution.gateway import (
    ExecutionGateway,
    ExecutionOutcome,
)
from core.nervous_system.persistence.repositories.execution import SubmissionConflict
from core.nervous_system.persistence.uow import UnitOfWork
from core.nervous_system.tests.fixtures.gateway_harness import (
    ACCOUNT,
    fake_uow_factory,
    NOW,
    FakeBroker,
    FakeJournal,
    broker_order,
    clock,
    decision_record,
    expected_client_order_id,
    order_request,
)


D = Decimal
@pytest.fixture
def uow_factory():
    """In-memory unit of work; DB concurrency is covered separately below."""

    factory, _store = fake_uow_factory()
    return factory


def gateway_for(uow_factory, *, broker=None, journal=None, worker_id="worker-a", at=NOW):
    return ExecutionGateway(
        broker=broker or FakeBroker(),
        journal=journal or FakeJournal(),
        unit_of_work_factory=uow_factory,
        environment=RuntimeEnvironment.QA_PAPER,
        account_alias=ACCOUNT,
        worker_id=worker_id,
        clock=clock(at),
    )


# --------------------------------------------------------------------------
# The exact entry sequence
# --------------------------------------------------------------------------


def test_no_broker_call_before_a_durable_reservation_and_journal(uow_factory) -> None:
    request = order_request()
    _seed(uow_factory, request)
    broker = FakeBroker()
    journal = FakeJournal(fail_on=("INTENT_TO_SUBMIT",))

    result = gateway_for(uow_factory, broker=broker, journal=journal).submit(
        decision=decision_record(), request=request
    )

    assert result.outcome is ExecutionOutcome.REFUSED
    assert result.reason_code == "JOURNAL_NOT_DURABLE"
    assert broker.submit_calls == [], "the broker must not be called without a journal"
    attempt = _attempt(uow_factory, request)
    assert attempt.status is SubmissionAttemptStatus.REJECTED


def test_the_happy_path_journals_intent_then_response(uow_factory) -> None:
    request = order_request()
    _seed(uow_factory, request)
    broker = FakeBroker()
    journal = FakeJournal()

    result = gateway_for(uow_factory, broker=broker, journal=journal).submit(
        decision=decision_record(), request=request
    )

    assert result.outcome is ExecutionOutcome.SUBMITTED
    assert journal.types() == ["INTENT_TO_SUBMIT", "BROKER_RESPONSE"]
    assert len(broker.submit_calls) == 1
    attempt = _attempt(uow_factory, request)
    assert attempt.status is SubmissionAttemptStatus.ACCEPTED
    assert attempt.broker_order_id == "brk-1"
    assert attempt.journal_event_id is not None


def test_the_client_order_id_is_the_deterministic_identity(uow_factory) -> None:
    request = order_request()
    _seed(uow_factory, request)
    broker = FakeBroker()

    gateway_for(uow_factory, broker=broker).submit(
        decision=decision_record(), request=request
    )

    assert _attempt(uow_factory, request).client_order_id == expected_client_order_id(
        request
    )


# --------------------------------------------------------------------------
# Idempotency and conflicts
# --------------------------------------------------------------------------


def test_resubmitting_the_same_request_does_not_call_the_broker_twice(
    uow_factory,
) -> None:
    request = order_request()
    _seed(uow_factory, request)
    broker = FakeBroker()
    gateway = gateway_for(uow_factory, broker=broker)

    first = gateway.submit(decision=decision_record(), request=request)
    second = gateway.submit(decision=decision_record(), request=request)

    assert first.outcome is ExecutionOutcome.SUBMITTED
    assert second.outcome is ExecutionOutcome.DUPLICATE
    assert len(broker.submit_calls) == 1, "one logical order, one broker submission"


def test_the_same_client_id_with_different_content_is_a_hard_conflict(
    uow_factory,
) -> None:
    request = order_request()
    gateway_for(uow_factory).submit(decision=decision_record(), request=request)

    impostor = order_request(parent_quantity=D("999"))
    with uow_factory() as uow:
        with pytest.raises(SubmissionConflict):
            uow.executions.reserve_or_load_attempt(
                request=impostor,
                client_order_id=expected_client_order_id(request),
                reserved_at=NOW,
            )


# --------------------------------------------------------------------------
# Concurrency: lease and fencing token (real PostgreSQL)
# --------------------------------------------------------------------------


def test_only_one_worker_may_own_submitting(uow_factory) -> None:
    request = order_request()
    broker_a, broker_b = FakeBroker(), FakeBroker()

    first = gateway_for(uow_factory, broker=broker_a, worker_id="worker-a").submit(
        decision=decision_record(), request=request
    )
    second = gateway_for(uow_factory, broker=broker_b, worker_id="worker-b").submit(
        decision=decision_record(), request=request
    )

    assert first.outcome is ExecutionOutcome.SUBMITTED
    assert second.outcome is ExecutionOutcome.DUPLICATE
    assert len(broker_a.submit_calls) + len(broker_b.submit_calls) == 1


# --------------------------------------------------------------------------
# Broker failure modes: never resubmit
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        BrokerAmbiguousSubmission("timeout"),
        BrokerUnavailable("429 rate limited"),
        BrokerAmbiguousSubmission("503"),
    ],
)
def test_a_failed_post_is_never_retried(uow_factory, error) -> None:
    request = order_request()
    _seed(uow_factory, request)
    broker = FakeBroker(submit_result=error)

    result = gateway_for(uow_factory, broker=broker).submit(
        decision=decision_record(), request=request
    )

    assert len(broker.submit_calls) == 1, "a lost POST must never be sent again"
    assert broker.lookup_calls == [expected_client_order_id(request)]
    assert result.outcome in {ExecutionOutcome.AMBIGUOUS, ExecutionOutcome.REJECTED}


def test_lookup_finding_the_order_resolves_it_as_submitted(uow_factory) -> None:
    request = order_request()
    _seed(uow_factory, request)
    broker = FakeBroker(
        submit_result=BrokerAmbiguousSubmission("timeout"),
        lookup_result=broker_order(request),
    )

    result = gateway_for(uow_factory, broker=broker).submit(
        decision=decision_record(), request=request
    )

    assert result.outcome is ExecutionOutcome.SUBMITTED
    assert result.reason_code == "RESOLVED_BY_CLIENT_ORDER_ID"
    assert _attempt(uow_factory, request).status is SubmissionAttemptStatus.ACCEPTED


def test_a_failed_lookup_is_ambiguous_never_absent(uow_factory) -> None:
    """Not being able to ask is not the same as the broker saying no."""

    request = order_request()
    _seed(uow_factory, request)
    broker = FakeBroker(
        submit_result=BrokerAmbiguousSubmission("timeout"),
        lookup_result=BrokerUnavailable("lookup failed"),
    )

    result = gateway_for(uow_factory, broker=broker).submit(
        decision=decision_record(), request=request
    )

    assert result.outcome is ExecutionOutcome.AMBIGUOUS
    assert _attempt(uow_factory, request).status is SubmissionAttemptStatus.AMBIGUOUS


def test_a_positive_absence_resolves_as_rejected(uow_factory) -> None:
    request = order_request()
    _seed(uow_factory, request)
    broker = FakeBroker(submit_result=BrokerAmbiguousSubmission("timeout"))

    result = gateway_for(uow_factory, broker=broker).submit(
        decision=decision_record(), request=request
    )

    assert result.outcome is ExecutionOutcome.REJECTED
    assert _attempt(uow_factory, request).status is SubmissionAttemptStatus.REJECTED


def test_a_broker_rejection_is_recorded_and_journalled(uow_factory) -> None:
    request = order_request()
    _seed(uow_factory, request)
    broker = FakeBroker(submit_result=BrokerRejected("422 insufficient buying power"))
    journal = FakeJournal()

    result = gateway_for(uow_factory, broker=broker, journal=journal).submit(
        decision=decision_record(), request=request
    )

    assert result.outcome is ExecutionOutcome.REJECTED
    assert journal.types() == ["INTENT_TO_SUBMIT", "BROKER_REJECTED"]
    assert _attempt(uow_factory, request).status is SubmissionAttemptStatus.REJECTED


def test_a_lost_response_journal_is_ambiguous_not_a_rejection(uow_factory) -> None:
    """The broker answered; failing to record that never means it did not."""

    request = order_request()
    _seed(uow_factory, request)
    broker = FakeBroker()
    journal = FakeJournal(fail_on=("BROKER_RESPONSE",))

    result = gateway_for(uow_factory, broker=broker, journal=journal).submit(
        decision=decision_record(), request=request
    )

    assert result.outcome is ExecutionOutcome.AMBIGUOUS
    assert result.reason_code == "RESPONSE_JOURNAL_NOT_DURABLE"
    assert result.broker_order_id == "brk-1"
    assert len(broker.submit_calls) == 1
    assert _attempt(uow_factory, request).status is SubmissionAttemptStatus.AMBIGUOUS


def test_a_crash_after_reservation_does_not_resubmit_on_restart(uow_factory) -> None:
    """Restart finds the resolved attempt and returns it instead of sending."""

    request = order_request()
    _seed(uow_factory, request)
    broker = FakeBroker()
    gateway_for(uow_factory, broker=broker).submit(
        decision=decision_record(), request=request
    )

    restarted_broker = FakeBroker()
    result = gateway_for(uow_factory, broker=restarted_broker, worker_id="worker-restart").submit(
        decision=decision_record(), request=request
    )

    assert result.outcome is ExecutionOutcome.DUPLICATE
    assert restarted_broker.submit_calls == []


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _seed(uow_factory, request) -> None:
    """The in-memory unit of work needs no foreign-key seeding."""


def _attempt(uow_factory, request):
    with uow_factory() as uow:
        return uow.executions.find_attempt_by_client_order_id(
            environment=request.environment,
            account_alias=request.account_alias,
            client_order_id=expected_client_order_id(request),
        )


# --------------------------------------------------------------------------
# The same concurrency primitives, against real PostgreSQL
# --------------------------------------------------------------------------


@pytest.fixture
def persisted_request(pg_session, complete_decision_chain):
    """A committed decision chain, so attempts satisfy their foreign keys."""

    from core.nervous_system.persistence.repositories.decision import DecisionRepository

    DecisionRepository(pg_session).save_chain(complete_decision_chain)
    pg_session.flush()
    return complete_decision_chain.order_requests[0]


@pytest.fixture
def executions(pg_session):
    from core.nervous_system.persistence.repositories.execution import (
        ExecutionRepository,
    )

    return ExecutionRepository(pg_session)


def test_pg_reserve_is_idempotent_and_detects_conflicts(
    executions, persisted_request
) -> None:
    client_order_id = "cyno-qp-" + persisted_request.request_hash[:40]

    first, created_first = executions.reserve_or_load_attempt(
        request=persisted_request, client_order_id=client_order_id, reserved_at=NOW
    )
    second, created_second = executions.reserve_or_load_attempt(
        request=persisted_request, client_order_id=client_order_id, reserved_at=NOW
    )

    assert created_first is True
    assert created_second is False
    assert first.submission_attempt_id == second.submission_attempt_id

    # A genuinely different order, deliberately reusing the same broker
    # identity. The contract refuses a forged hash, so the impostor is built
    # honestly and only the client order ID is reused.
    from core.nervous_system.contracts.orders import OrderRequest

    fields = persisted_request.model_dump()
    for derived in ("request_hash", "order_request_id"):
        fields.pop(derived)
    fields["parent_quantity"] = D("999")
    impostor = OrderRequest.create(**fields)
    assert impostor.request_hash != persisted_request.request_hash

    with pytest.raises(SubmissionConflict, match="different request content"):
        executions.reserve_or_load_attempt(
            request=impostor, client_order_id=client_order_id, reserved_at=NOW
        )


def test_pg_only_one_worker_wins_the_lease(executions, persisted_request) -> None:
    client_order_id = "cyno-qp-" + persisted_request.request_hash[:40]
    attempt, _ = executions.reserve_or_load_attempt(
        request=persisted_request, client_order_id=client_order_id, reserved_at=NOW
    )

    won = executions.claim_submission(
        submission_attempt_id=attempt.submission_attempt_id,
        owner="worker-a",
        claim_token="token-a",
        lease_until=NOW + timedelta(minutes=1),
        now=NOW,
    )
    stolen = executions.claim_submission(
        submission_attempt_id=attempt.submission_attempt_id,
        owner="worker-b",
        claim_token="token-b",
        lease_until=NOW + timedelta(minutes=1),
        now=NOW,
    )

    assert won is True
    assert stolen is False, "a live lease must not be stealable"


def test_pg_an_expired_lease_may_be_taken_over(executions, persisted_request) -> None:
    client_order_id = "cyno-qp-" + persisted_request.request_hash[:40]
    attempt, _ = executions.reserve_or_load_attempt(
        request=persisted_request, client_order_id=client_order_id, reserved_at=NOW
    )
    executions.claim_submission(
        submission_attempt_id=attempt.submission_attempt_id,
        owner="worker-a",
        claim_token="token-a",
        lease_until=NOW + timedelta(seconds=30),
        now=NOW,
    )

    later = NOW + timedelta(minutes=5)
    taken = executions.claim_submission(
        submission_attempt_id=attempt.submission_attempt_id,
        owner="worker-b",
        claim_token="token-b",
        lease_until=later + timedelta(minutes=1),
        now=later,
    )

    assert taken is True


def test_pg_a_stale_fencing_token_cannot_move_the_attempt(
    executions, persisted_request
) -> None:
    client_order_id = "cyno-qp-" + persisted_request.request_hash[:40]
    attempt, _ = executions.reserve_or_load_attempt(
        request=persisted_request, client_order_id=client_order_id, reserved_at=NOW
    )
    executions.claim_submission(
        submission_attempt_id=attempt.submission_attempt_id,
        owner="worker-b",
        claim_token="token-current",
        lease_until=NOW + timedelta(minutes=1),
        now=NOW,
    )

    moved = executions.transition_attempt(
        submission_attempt_id=attempt.submission_attempt_id,
        expected=SubmissionAttemptStatus.RESERVED,
        target=SubmissionAttemptStatus.JOURNALED,
        claim_token="token-stale",
    )

    assert moved is False, "a worker that lost its lease must not resolve the attempt"


def test_pg_compare_and_set_from_the_wrong_state_does_nothing(
    executions, persisted_request
) -> None:
    client_order_id = "cyno-qp-" + persisted_request.request_hash[:40]
    attempt, _ = executions.reserve_or_load_attempt(
        request=persisted_request, client_order_id=client_order_id, reserved_at=NOW
    )

    moved = executions.transition_attempt(
        submission_attempt_id=attempt.submission_attempt_id,
        expected=SubmissionAttemptStatus.SUBMITTING,
        target=SubmissionAttemptStatus.ACCEPTED,
    )

    assert moved is False


def test_pg_an_illegal_transition_is_refused_outright(
    executions, persisted_request
) -> None:
    client_order_id = "cyno-qp-" + persisted_request.request_hash[:40]
    attempt, _ = executions.reserve_or_load_attempt(
        request=persisted_request, client_order_id=client_order_id, reserved_at=NOW
    )

    with pytest.raises(ValueError, match="illegal submission transition"):
        executions.transition_attempt(
            submission_attempt_id=attempt.submission_attempt_id,
            expected=SubmissionAttemptStatus.RESERVED,
            target=SubmissionAttemptStatus.ACCEPTED,
        )
