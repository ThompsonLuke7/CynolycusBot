from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select

from core.nervous_system.contracts.decisions import DecisionOutcome
from core.nervous_system.persistence.models import DecisionRecord as DecisionRecordRow
from core.nervous_system.persistence.models import OutboxEvent
from core.nervous_system.persistence.repositories.decision import (
    CompleteDecisionChain,
    DecisionRepository,
)
from core.nervous_system.persistence.uow import UnitOfWork


class _Session:
    def __init__(self):
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


def test_unit_of_work_uses_one_session_and_rolls_back_uncommitted_exit():
    session = _Session()
    with UnitOfWork(lambda: session) as uow:
        assert uow.session is session
        assert uow.states._session is session
        assert uow.decisions._session is session
        assert uow.executions._session is session
        assert uow.registry._session is session
        assert uow.operations._session is session
    assert session.commits == 0
    assert session.rollbacks == 1
    assert session.closed is True


def test_complete_decision_chain_rejects_mismatched_links(complete_decision_chain):
    chain = complete_decision_chain
    with pytest.raises(ValueError, match="intent"):
        CompleteDecisionChain(
            snapshot=chain.snapshot,
            intent=chain.intent,
            policy_decision=chain.policy_decision,
            record=chain.record.model_copy(update={"intent_id": uuid4()}),
            order_requests=chain.order_requests,
        )


def test_complete_decision_chain_rejects_tampered_snapshot(complete_decision_chain):
    chain = complete_decision_chain
    tampered_snapshot = chain.snapshot.model_copy()
    object.__setattr__(tampered_snapshot, "content_hash", "a" * 64)
    tampered_record = chain.record.model_copy(update={"snapshot_hash": "a" * 64})
    with pytest.raises(ValueError, match="snapshot content_hash"):
        CompleteDecisionChain(
            snapshot=tampered_snapshot,
            intent=chain.intent,
            policy_decision=chain.policy_decision,
            record=tampered_record,
            order_requests=chain.order_requests,
        )


@pytest.mark.postgres
def test_save_chain_and_outbox_commit_as_one_transaction(
    session_factory,
    complete_decision_chain,
):
    chain = complete_decision_chain
    with UnitOfWork(session_factory) as uow:
        uow.decisions.save_chain(chain)
        outbox = uow.operations.enqueue(
            event_type="DecisionRecordCreated",
            aggregate_type="decision_record",
            aggregate_id=str(chain.record.decision_record_id),
            payload={"strategy_id": chain.intent.strategy_id},
        )
        uow.commit()

    with session_factory() as session:
        assert session.scalar(
            select(DecisionRecordRow.decision_record_id).where(
                DecisionRecordRow.decision_record_id == chain.record.decision_record_id
            )
        ) == chain.record.decision_record_id
        assert session.scalar(
            select(OutboxEvent.outbox_event_id).where(
                OutboxEvent.outbox_event_id == outbox.outbox_event_id
            )
        ) == outbox.outbox_event_id


@pytest.mark.postgres
def test_uncommitted_uow_rolls_back_complete_chain(session_factory, complete_decision_chain):
    chain = complete_decision_chain
    with UnitOfWork(session_factory) as uow:
        uow.decisions.save_chain(chain)

    with session_factory() as session:
        assert session.scalar(
            select(DecisionRecordRow.decision_record_id).where(
                DecisionRecordRow.decision_record_id == chain.record.decision_record_id
            )
        ) is None


@pytest.mark.postgres
def test_decision_outcome_is_validated_against_persisted_record(
    session_factory,
    complete_decision_chain,
):
    chain = complete_decision_chain
    with UnitOfWork(session_factory) as uow:
        uow.decisions.save_chain(chain)
        valid = DecisionOutcome.for_decision(
            chain.record,
            outcome_id=uuid4(),
            evaluated_at=chain.record.decision_time + timedelta(hours=1),
            horizon="1d",
            underlying_return=0.02,
            instrument_return=None,
            source_fitness_report_id=None,
            metrics={"mfe": 0.04},
        )
        uow.decisions.append_decision_outcome(valid)
        uow.commit()

    invalid = valid.model_copy(update={"decision_record_id": uuid4()})
    with session_factory() as session:
        with pytest.raises(ValueError, match="decision record"):
            DecisionRepository(session).append_decision_outcome(invalid)
