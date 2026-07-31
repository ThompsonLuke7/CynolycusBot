from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import select, update

from core.nervous_system.contracts.enums import ExecutionStatus, RuntimeEnvironment
from core.nervous_system.contracts.execution import ExecutionEvent, ExecutionReport
from core.nervous_system.persistence.models import ExecutionEvent as ExecutionEventRow
from core.nervous_system.persistence.models import SubmissionAttempt
from core.nervous_system.persistence.repositories.execution import ExecutionRepository


def _event(order_request_id, *, event_id=None, sequence_no=1, previous_event_id=None, previous_event_hash=None):
    observed_at = datetime(2026, 7, 30, 18, 20, tzinfo=timezone.utc)
    return ExecutionEvent.create(
        execution_event_id=event_id or uuid4(),
        order_request_id=order_request_id,
        event_type="BROKER_UPDATE",
        sequence_no=sequence_no,
        status=ExecutionStatus.ACCEPTED if sequence_no == 1 else ExecutionStatus.FILLED,
        observed_at=observed_at + timedelta(seconds=sequence_no - 1),
        broker_event_at=observed_at + timedelta(seconds=sequence_no - 1),
        client_order_id="task-7-client",
        broker_order_id="broker-task-7",
        broker_parent_order_id=None,
        filled_quantity=Decimal("0") if sequence_no == 1 else Decimal("1"),
        average_fill_price=None if sequence_no == 1 else Decimal("100"),
        leg_reports=({"symbol": "AMD", "status": "FILLED"},),
        sanitized_response={"status": "ok"},
        previous_event_id=previous_event_id,
        previous_event_hash=previous_event_hash,
    )


@pytest.mark.postgres
def test_execution_events_append_and_reconstruct_as_a_valid_report(
    session_factory,
    complete_decision_chain,
):
    chain = complete_decision_chain
    first = _event(chain.order_requests[0].order_request_id)
    second = _event(
        chain.order_requests[0].order_request_id,
        sequence_no=2,
        previous_event_id=first.execution_event_id,
        previous_event_hash=first.event_hash,
    )
    with session_factory() as session:
        repo = ExecutionRepository(session)
        # The FK-safe parent chain is inserted through the decision repository.
        from core.nervous_system.persistence.repositories.decision import DecisionRepository

        DecisionRepository(session).save_chain(chain)
        repo.append_execution_event(first)
        repo.append_execution_event(second)
        report_events = repo.get_events(chain.order_requests[0].order_request_id)
        report = ExecutionReport(
            order_request_id=chain.order_requests[0].order_request_id,
            events=report_events,
            current_status=ExecutionStatus.FILLED,
        )
        session.commit()

    assert [event.sequence_no for event in report.events] == [1, 2]
    assert report.events[-1].event_hash == second.event_hash


@pytest.mark.postgres
def test_execution_append_requires_the_exact_predecessor_id_and_hash(
    session_factory,
    complete_decision_chain,
):
    chain = complete_decision_chain
    first = _event(chain.order_requests[0].order_request_id)
    bad_second = _event(
        chain.order_requests[0].order_request_id,
        sequence_no=2,
        previous_event_id=uuid4(),
        previous_event_hash=first.event_hash,
    )
    with session_factory() as session:
        from core.nervous_system.persistence.repositories.decision import DecisionRepository

        DecisionRepository(session).save_chain(chain)
        repo = ExecutionRepository(session)
        repo.append_execution_event(first)
        with pytest.raises(ValueError, match="predecessor"):
            repo.append_execution_event(bad_second)


@pytest.mark.postgres
def test_client_order_lookup_is_scoped_by_environment_and_account(
    session_factory,
    complete_decision_chain,
):
    chain = complete_decision_chain
    order = chain.order_requests[0]
    now = order.created_at
    client_order_id = f"task7-shared-{uuid4().hex}"
    with session_factory() as session:
        from core.nervous_system.persistence.repositories.decision import DecisionRepository

        DecisionRepository(session).save_chain(chain)
        session.add_all(
            [
                SubmissionAttempt(
                    submission_attempt_id=uuid4(),
                    order_request_id=order.order_request_id,
                    attempt_no=1,
                    environment=RuntimeEnvironment.QA_PAPER.value,
                    account_alias="paper",
                    client_order_id=client_order_id,
                    status="ACCEPTED",
                    reserved_at=now,
                    payload={},
                ),
                SubmissionAttempt(
                    submission_attempt_id=uuid4(),
                    order_request_id=order.order_request_id,
                    attempt_no=2,
                    environment=RuntimeEnvironment.DEVELOPMENT.value,
                    account_alias="dev",
                    client_order_id=client_order_id,
                    status="ACCEPTED",
                    reserved_at=now,
                    payload={},
                ),
            ]
        )
        session.flush()
        repo = ExecutionRepository(session)
        assert repo.find_by_client_order_id(
            RuntimeEnvironment.QA_PAPER, "paper", client_order_id
        ).order_request_id == order.order_request_id
        assert repo.find_by_client_order_id(
            RuntimeEnvironment.QA_PAPER, "other", client_order_id
        ) is None
        session.commit()


@pytest.mark.postgres
def test_execution_payload_tampering_is_rejected_on_reconstruction(
    session_factory,
    complete_decision_chain,
):
    chain = complete_decision_chain
    first = _event(chain.order_requests[0].order_request_id)
    with session_factory() as session:
        from core.nervous_system.persistence.repositories.decision import DecisionRepository

        DecisionRepository(session).save_chain(chain)
        ExecutionRepository(session).append_execution_event(first)
        session.commit()

        session.execute(
            update(ExecutionEventRow)
            .where(ExecutionEventRow.execution_event_id == first.execution_event_id)
            .values(payload={**first.model_dump(mode="json"), "client_order_id": "tampered"})
        )
        session.commit()

    with session_factory() as session:
        with pytest.raises(ValueError, match="event_hash"):
            ExecutionRepository(session).get_events(chain.order_requests[0].order_request_id)
