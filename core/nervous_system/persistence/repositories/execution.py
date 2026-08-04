"""Append-only typed execution-event persistence and broker lookup."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from core.nervous_system.contracts.execution import (
    ExecutionEvent,
    ExecutionReport,
    SubmissionAttemptRecord,
)
from core.nervous_system.contracts.enums import (
    RuntimeEnvironment,
    SubmissionAttemptStatus,
)
from core.nervous_system.contracts.orders import OrderRequest
from core.nervous_system.persistence.models import (
    ExecutionEvent as ExecutionEventRow,
    OrderRequest as OrderRequestRow,
    SubmissionAttempt,
)


def _one_or_none(result: Any) -> Any:
    scalars = result.scalars()
    if hasattr(scalars, "one_or_none"):
        return scalars.one_or_none()
    return scalars.first()


def _order_from_row(row: OrderRequestRow) -> OrderRequest:
    request = OrderRequest.model_validate(row.payload)
    if request.order_request_id != row.order_request_id:
        raise ValueError("order payload ID does not match relational ID")
    if request.decision_id != row.decision_record_id:
        raise ValueError("order payload decision link does not match relational ID")
    if request.policy_decision_id != row.policy_decision_id:
        raise ValueError("order payload policy link does not match relational ID")
    if request.environment.value != row.environment:
        raise ValueError("order payload environment does not match relational column")
    if request.account_alias != row.account_alias:
        raise ValueError("order payload account alias does not match relational column")
    if request.idempotency_key != row.idempotency_key:
        raise ValueError("order payload idempotency key does not match relational column")
    if request.request_hash != row.request_hash:
        raise ValueError("order request hash does not match relational column")
    if request.decision_kind.value != row.decision_kind:
        raise ValueError("order payload decision kind does not match relational column")
    if request.risk_reducing != row.risk_reducing:
        raise ValueError("order payload risk semantics do not match relational column")
    if request.broker_position_key != row.broker_position_key:
        raise ValueError("order payload broker position key does not match relational column")
    if request.order_type != row.order_type:
        raise ValueError("order payload order type does not match relational column")
    if request.parent_quantity != row.parent_quantity:
        raise ValueError("order payload quantity does not match relational column")
    if request.net_limit_price != row.net_limit_price:
        raise ValueError("order payload limit price does not match relational column")
    if request.maximum_loss != row.maximum_loss:
        raise ValueError("order payload maximum loss does not match relational column")
    if request.buying_power_required != row.buying_power_required:
        raise ValueError("order payload buying power does not match relational column")
    if request.created_at != row.created_at or request.expires_at != row.expires_at:
        raise ValueError("order payload timestamps do not match relational columns")
    return request


def _event_from_row(row: ExecutionEventRow) -> ExecutionEvent:
    event = ExecutionEvent.model_validate(row.payload)
    if event.execution_event_id != row.execution_event_id:
        raise ValueError("execution payload ID does not match relational ID")
    if event.order_request_id != row.order_request_id:
        raise ValueError("execution payload order link does not match relational ID")
    if event.status.value != row.status:
        raise ValueError("execution payload status does not match relational column")
    if event.event_type != row.event_type:
        raise ValueError("execution payload event_type does not match relational column")
    if event.client_order_id != row.client_order_id:
        raise ValueError("execution payload client order ID does not match relational column")
    if event.broker_order_id != row.broker_order_id:
        raise ValueError("execution payload broker order ID does not match relational column")
    if event.broker_parent_order_id != row.broker_parent_order_id:
        raise ValueError("execution payload broker parent ID does not match relational column")
    if event.sequence_no != row.sequence_no:
        raise ValueError("execution payload sequence number does not match relational column")
    if event.previous_event_id != row.previous_event_id:
        raise ValueError("execution payload predecessor ID does not match relational column")
    if event.previous_event_hash != row.previous_event_hash:
        raise ValueError("execution payload predecessor hash does not match relational column")
    if event.observed_at != row.observed_at:
        raise ValueError("execution payload observed_at does not match relational column")
    if event.broker_event_at != row.broker_event_at:
        raise ValueError("execution payload broker_event_at does not match relational column")
    if event.filled_quantity != row.filled_quantity:
        raise ValueError("execution payload filled quantity does not match relational column")
    if event.average_fill_price != row.average_fill_price:
        raise ValueError("execution payload average fill price does not match relational column")
    if event.event_hash != row.event_hash:
        raise ValueError("execution event hash does not match relational column")
    return event


class SubmissionConflict(ValueError):
    """The same broker identity was reserved for different request content."""


def _attempt_from_row(row: SubmissionAttempt) -> SubmissionAttemptRecord:
    return SubmissionAttemptRecord(
        submission_attempt_id=row.submission_attempt_id,
        order_request_id=row.order_request_id,
        attempt_no=row.attempt_no,
        environment=RuntimeEnvironment(row.environment),
        account_alias=row.account_alias,
        client_order_id=row.client_order_id,
        status=SubmissionAttemptStatus(row.status),
        request_hash=str(row.payload.get("request_hash", "")),
        reserved_at=row.reserved_at,
        lease_owner=row.lease_owner,
        lease_until=row.lease_until,
        claim_token=row.claim_token,
        journaled_at=row.journaled_at,
        broker_called_at=row.broker_called_at,
        resolved_at=row.resolved_at,
        broker_order_id=row.broker_order_id,
        error_code=row.error_code,
        journal_event_id=row.journal_event_id,
        journal_event_hash=row.journal_event_hash,
        journal_backend=row.journal_backend,
        journal_locator=row.journal_locator,
    )


# Only these transitions are legal; anything else is a backward or impossible
# move and is refused rather than quietly applied.
_LEGAL_TRANSITIONS: dict[SubmissionAttemptStatus, frozenset[SubmissionAttemptStatus]] = {
    SubmissionAttemptStatus.RESERVED: frozenset(
        {SubmissionAttemptStatus.JOURNALED, SubmissionAttemptStatus.REJECTED}
    ),
    SubmissionAttemptStatus.JOURNALED: frozenset(
        {SubmissionAttemptStatus.SUBMITTING, SubmissionAttemptStatus.REJECTED}
    ),
    SubmissionAttemptStatus.SUBMITTING: frozenset(
        {
            SubmissionAttemptStatus.ACCEPTED,
            SubmissionAttemptStatus.REJECTED,
            SubmissionAttemptStatus.AMBIGUOUS,
        }
    ),
    SubmissionAttemptStatus.AMBIGUOUS: frozenset(
        {
            SubmissionAttemptStatus.ACCEPTED,
            SubmissionAttemptStatus.REJECTED,
            SubmissionAttemptStatus.RECONCILIATION_REQUIRED,
        }
    ),
    SubmissionAttemptStatus.ACCEPTED: frozenset(),
    SubmissionAttemptStatus.REJECTED: frozenset(),
    SubmissionAttemptStatus.RECONCILIATION_REQUIRED: frozenset(
        {SubmissionAttemptStatus.ACCEPTED, SubmissionAttemptStatus.REJECTED}
    ),
}


def is_legal_transition(
    current: SubmissionAttemptStatus,
    target: SubmissionAttemptStatus,
) -> bool:
    return target in _LEGAL_TRANSITIONS.get(current, frozenset())


class ExecutionRepository:
    def __init__(self, session: Session):
        self._session = session

    # -- submission attempts -------------------------------------------------

    def reserve_or_load_attempt(
        self,
        *,
        request: OrderRequest,
        client_order_id: str,
        reserved_at: datetime,
        attempt_no: int = 1,
        submission_attempt_id: UUID | None = None,
    ) -> tuple[SubmissionAttemptRecord, bool]:
        """Durably reserve one broker identity, or load the existing one.

        Returns ``(attempt, created)``. The same content under the same broker
        identity loads; different content is a hard conflict, because reusing a
        client order ID for a different order is how a duplicate becomes
        invisible.
        """

        existing = _one_or_none(
            self._session.execute(
                select(SubmissionAttempt).where(
                    SubmissionAttempt.environment == request.environment.value,
                    SubmissionAttempt.account_alias == request.account_alias,
                    SubmissionAttempt.client_order_id == client_order_id,
                )
            )
        )
        if existing is not None:
            record = _attempt_from_row(existing)
            if record.request_hash != request.request_hash:
                raise SubmissionConflict(
                    f"client order ID {client_order_id} already reserved for "
                    "different request content"
                )
            if record.order_request_id != request.order_request_id:
                raise SubmissionConflict(
                    f"client order ID {client_order_id} belongs to another order request"
                )
            return record, False

        row = SubmissionAttempt(
            submission_attempt_id=submission_attempt_id or uuid4(),
            order_request_id=request.order_request_id,
            attempt_no=attempt_no,
            environment=request.environment.value,
            account_alias=request.account_alias,
            client_order_id=client_order_id,
            status=SubmissionAttemptStatus.RESERVED.value,
            reserved_at=reserved_at,
            payload={"request_hash": request.request_hash},
        )
        self._session.add(row)
        self._session.flush()
        return _attempt_from_row(row), True

    def claim_submission(
        self,
        *,
        submission_attempt_id: UUID,
        owner: str,
        claim_token: str,
        lease_until: datetime,
        now: datetime,
    ) -> bool:
        """Take the lease with a fencing token, or report that someone else holds it.

        A unique constraint alone does not stop two workers POSTing, so
        ownership of ``SUBMITTING`` is decided by this conditional update.
        """

        result = self._session.execute(
            update(SubmissionAttempt)
            .where(
                SubmissionAttempt.submission_attempt_id == submission_attempt_id,
                SubmissionAttempt.status.in_(
                    (
                        SubmissionAttemptStatus.RESERVED.value,
                        SubmissionAttemptStatus.JOURNALED.value,
                    )
                ),
                or_(
                    SubmissionAttempt.lease_owner.is_(None),
                    SubmissionAttempt.lease_until.is_(None),
                    SubmissionAttempt.lease_until <= now,
                    SubmissionAttempt.lease_owner == owner,
                ),
            )
            .values(lease_owner=owner, lease_until=lease_until, claim_token=claim_token)
        )
        self._session.flush()
        return bool(result.rowcount)

    def transition_attempt(
        self,
        *,
        submission_attempt_id: UUID,
        expected: SubmissionAttemptStatus,
        target: SubmissionAttemptStatus,
        claim_token: str | None = None,
        **updates: Any,
    ) -> bool:
        """Compare-and-set one legal status transition.

        The claim token fences a stale worker: one that lost its lease cannot
        move an attempt another worker now owns.
        """

        if not is_legal_transition(expected, target):
            raise ValueError(
                f"illegal submission transition {expected.value} -> {target.value}"
            )
        conditions = [
            SubmissionAttempt.submission_attempt_id == submission_attempt_id,
            SubmissionAttempt.status == expected.value,
        ]
        if claim_token is not None:
            conditions.append(SubmissionAttempt.claim_token == claim_token)
        result = self._session.execute(
            update(SubmissionAttempt)
            .where(*conditions)
            .values(status=target.value, **updates)
        )
        self._session.flush()
        return bool(result.rowcount)

    def record_journal_receipt(
        self,
        *,
        submission_attempt_id: UUID,
        event_id: UUID,
        event_hash: str,
        backend: str,
        locator: str,
        journaled_at: datetime,
    ) -> None:
        self._session.execute(
            update(SubmissionAttempt)
            .where(SubmissionAttempt.submission_attempt_id == submission_attempt_id)
            .values(
                journal_event_id=event_id,
                journal_event_hash=event_hash,
                journal_backend=backend,
                journal_locator=locator,
                journaled_at=journaled_at,
            )
        )
        self._session.flush()

    def get_attempt(
        self, submission_attempt_id: UUID
    ) -> SubmissionAttemptRecord | None:
        row = self._session.get(SubmissionAttempt, submission_attempt_id)
        return _attempt_from_row(row) if row is not None else None

    def find_attempt_by_client_order_id(
        self,
        *,
        environment: RuntimeEnvironment,
        account_alias: str,
        client_order_id: str,
    ) -> SubmissionAttemptRecord | None:
        row = _one_or_none(
            self._session.execute(
                select(SubmissionAttempt).where(
                    SubmissionAttempt.environment == environment.value,
                    SubmissionAttempt.account_alias == account_alias,
                    SubmissionAttempt.client_order_id == client_order_id,
                )
            )
        )
        return _attempt_from_row(row) if row is not None else None

    def append_execution_event(self, event: ExecutionEvent) -> None:
        if event.sequence_no == 1:
            existing = self._session.execute(
                select(ExecutionEventRow.execution_event_id)
                .where(ExecutionEventRow.order_request_id == event.order_request_id)
                .limit(1)
            ).scalar_one_or_none()
            if existing is not None:
                raise ValueError("sequence 1 execution event cannot follow an existing event")
        else:
            predecessor = self._session.get(
                ExecutionEventRow, event.previous_event_id
            )
            if predecessor is None:
                raise ValueError("execution event predecessor does not exist")
            if predecessor.order_request_id != event.order_request_id:
                raise ValueError("execution event predecessor belongs to another order")
            if predecessor.sequence_no != event.sequence_no - 1:
                raise ValueError("execution event predecessor sequence is not adjacent")
            if predecessor.event_hash != event.previous_event_hash:
                raise ValueError("execution event predecessor hash does not match")

        self._session.add(
            ExecutionEventRow(
                execution_event_id=event.execution_event_id,
                order_request_id=event.order_request_id,
                status=event.status.value,
                event_type=event.event_type,
                client_order_id=event.client_order_id,
                broker_order_id=event.broker_order_id,
                broker_parent_order_id=event.broker_parent_order_id,
                observed_at=event.observed_at,
                broker_event_at=event.broker_event_at,
                filled_quantity=event.filled_quantity,
                average_fill_price=event.average_fill_price,
                sequence_no=event.sequence_no,
                previous_event_id=event.previous_event_id,
                event_hash=event.event_hash,
                previous_event_hash=event.previous_event_hash,
                payload=event.model_dump(mode="json"),
            )
        )
        self._session.flush()

    def get_events(self, order_request_id: UUID) -> tuple[ExecutionEvent, ...]:
        rows = self._session.execute(
            select(ExecutionEventRow)
            .where(ExecutionEventRow.order_request_id == order_request_id)
            .order_by(ExecutionEventRow.sequence_no.asc())
        ).scalars().all()
        events = tuple(_event_from_row(row) for row in rows)
        if events:
            ExecutionReport(
                order_request_id=order_request_id,
                events=events,
                current_status=events[-1].status,
            )
        return events

    def find_by_client_order_id(
        self,
        environment: RuntimeEnvironment,
        account_alias: str,
        client_order_id: str,
    ) -> OrderRequest | None:
        stmt = (
            select(OrderRequestRow)
            .join(
                SubmissionAttempt,
                SubmissionAttempt.order_request_id == OrderRequestRow.order_request_id,
            )
            .where(
                SubmissionAttempt.environment == environment.value,
                SubmissionAttempt.account_alias == account_alias,
                SubmissionAttempt.client_order_id == client_order_id,
            )
            .limit(1)
        )
        row = _one_or_none(self._session.execute(stmt))
        return None if row is None else _order_from_row(row)


__all__ = ["ExecutionRepository"]
