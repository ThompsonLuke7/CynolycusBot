"""The idempotent execution gateway.

This is the only automated broker write boundary. Its job is not to be clever;
it is to make sure that a crash, a timeout, or a second worker can never turn
one intended order into two real ones.

Three rules drive the design:

1. Nothing reaches the broker before a durable PostgreSQL reservation and a
   durable ``INTENT_TO_SUBMIT`` journal event exist.
2. A broker POST is never retried automatically. A lost response is ambiguous,
   and ambiguity is resolved by asking the broker about the deterministic
   client order ID, never by sending again.
3. Ownership begins only at a confirmed fill. Acceptance is not a fill.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4, uuid5

from sqlalchemy.exc import SQLAlchemyError

from core.nervous_system.contracts.base import ContractModel, UtcDatetime
from core.nervous_system.contracts.decisions import DecisionRecord
from core.nervous_system.contracts.enums import (
    DecisionKind,
    ExecutionStatus,
    OrderSide,
    RuntimeEnvironment,
    SubmissionAttemptStatus,
)
from core.nervous_system.contracts.orders import OrderRequest
from core.nervous_system.persistence.repositories.execution import SubmissionConflict

from .broker import (
    BrokerAdapter,
    BrokerAmbiguousSubmission,
    BrokerAuthenticationError,
    BrokerError,
    BrokerOrder,
    BrokerRejected,
    BrokerUnavailable,
    OrderReplacement,
)
from .journal import (
    CompositeJournalResult,
    ExecutionJournalEvent,
    PostgresPersistenceStatus,
    link_event,
)


# uuid5(NAMESPACE_URL, "https://cynolycus.local/nervous-system/order-request@1")
ORDER_REQUEST_NAMESPACE = UUID("6b6d1a4f-6f9e-5a6b-9c1a-0b2f4d8e7a31")

CLIENT_ORDER_ID_PREFIX = "cyno"
CLIENT_ORDER_ID_LENGTH = 48
REQUEST_HASH_PREFIX_LENGTH = 40

ENVIRONMENT_CODES = {
    RuntimeEnvironment.DEVELOPMENT: "dv",
    RuntimeEnvironment.QA_PAPER: "qp",
    RuntimeEnvironment.PRODUCTION_LIVE: "pl",
}

_ZERO = Decimal("0")


class GatewayError(Exception):
    """Base class for gateway refusals."""


class GatewayPreflightError(GatewayError):
    """The request was refused before anything durable happened."""


class GatewayConflict(GatewayError):
    """The same broker identity was reused for different content."""


class ExecutionOutcome(str, Enum):
    SUBMITTED = "SUBMITTED"
    DUPLICATE = "DUPLICATE"
    REJECTED = "REJECTED"
    AMBIGUOUS = "AMBIGUOUS"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    REFUSED = "REFUSED"


class ExecutionResult(ContractModel):
    outcome: ExecutionOutcome
    order_request_id: UUID
    client_order_id: str
    submission_attempt_id: UUID | None = None
    broker_order_id: str | None = None
    status: ExecutionStatus | None = None
    reason_code: str | None = None
    detail: str | None = None
    journal_event_ids: tuple[UUID, ...] = ()

    @property
    def reached_broker(self) -> bool:
        return self.outcome in {
            ExecutionOutcome.SUBMITTED,
            ExecutionOutcome.REJECTED,
            ExecutionOutcome.AMBIGUOUS,
            ExecutionOutcome.RECONCILIATION_REQUIRED,
        }


def order_request_id_for(decision_id: UUID, request_hash: str) -> UUID:
    """One canonical order request identity per decision and content."""

    return uuid5(ORDER_REQUEST_NAMESPACE, f"{decision_id}|{request_hash}")


def client_order_id_for(
    environment: RuntimeEnvironment,
    request_hash: str,
) -> str:
    """Deterministic 48-character broker identity.

    Recovery reuses this exact value. Adding an attempt suffix would create a
    second broker identity for one logical order, which is precisely how a
    retry becomes a duplicate fill.
    """

    code = ENVIRONMENT_CODES[environment]
    prefix = request_hash[:REQUEST_HASH_PREFIX_LENGTH]
    if len(prefix) != REQUEST_HASH_PREFIX_LENGTH:
        raise GatewayPreflightError("request hash is too short for a client order ID")
    value = f"{CLIENT_ORDER_ID_PREFIX}-{code}-{prefix}"
    if len(value) != CLIENT_ORDER_ID_LENGTH:  # pragma: no cover - arithmetic guard
        raise GatewayPreflightError(
            f"client order ID must be {CLIENT_ORDER_ID_LENGTH} characters"
        )
    return value


class ExecutionGateway:
    """Route one order request to the broker exactly once."""

    def __init__(
        self,
        *,
        broker: BrokerAdapter,
        journal: Any,
        unit_of_work_factory: Callable[[], Any],
        environment: RuntimeEnvironment,
        account_alias: str,
        worker_id: str,
        clock: Callable[[], datetime] | None = None,
        lease_seconds: int = 60,
    ) -> None:
        if environment is RuntimeEnvironment.PRODUCTION_LIVE:
            raise GatewayPreflightError(
                "PRODUCTION_LIVE is refused: the gateway never writes to a live account"
            )
        self._broker = broker
        self._journal = journal
        self._uow_factory = unit_of_work_factory
        self._environment = environment
        self._account_alias = account_alias
        self._worker_id = worker_id
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lease = timedelta(seconds=lease_seconds)

    # -- public surface -----------------------------------------------------

    def submit(
        self,
        *,
        decision: DecisionRecord,
        request: OrderRequest,
    ) -> ExecutionResult:
        client_order_id = client_order_id_for(
            request.environment, request.request_hash
        )
        try:
            self._preflight(decision=decision, request=request)
        except GatewayPreflightError as exc:
            return ExecutionResult(
                outcome=ExecutionOutcome.REFUSED,
                order_request_id=request.order_request_id,
                client_order_id=client_order_id,
                reason_code="PREFLIGHT_REFUSED",
                detail=str(exc),
            )

        try:
            return self._submit_durable(
                decision=decision, request=request, client_order_id=client_order_id
            )
        except _PostgresUnavailable as exc:
            # PostgreSQL is down. An entry always fails closed; a deterministic
            # risk-reducing exit may still proceed on journal evidence alone.
            if not request.risk_reducing:
                return ExecutionResult(
                    outcome=ExecutionOutcome.REFUSED,
                    order_request_id=request.order_request_id,
                    client_order_id=client_order_id,
                    reason_code="POSTGRES_UNAVAILABLE",
                    detail="new entries require PostgreSQL",
                )
            return self._submit_fail_operational_exit(
                request=request, client_order_id=client_order_id, cause=exc
            )

    def cancel(
        self,
        *,
        decision: DecisionRecord,
        broker_order_id: str,
    ) -> ExecutionResult:
        """Cancel is risk-reducing, so it stays available under degradation."""

        self._require_paper()
        try:
            order = self._broker.cancel(broker_order_id)
        except BrokerError as exc:
            return ExecutionResult(
                outcome=ExecutionOutcome.AMBIGUOUS
                if isinstance(exc, BrokerAmbiguousSubmission)
                else ExecutionOutcome.REFUSED,
                order_request_id=decision.decision_record_id,
                client_order_id="",
                broker_order_id=broker_order_id,
                reason_code=type(exc).__name__,
                detail=str(exc),
            )
        return ExecutionResult(
            outcome=ExecutionOutcome.SUBMITTED,
            order_request_id=decision.decision_record_id,
            client_order_id=order.client_order_id,
            broker_order_id=order.broker_order_id,
            status=order.status,
        )

    def replace(
        self,
        *,
        decision: DecisionRecord,
        broker_order_id: str,
        replacement: OrderReplacement,
    ) -> ExecutionResult:
        """Replace only adjusts price/quantity; structure changes are new orders."""

        self._require_paper()
        try:
            order = self._broker.replace(broker_order_id, replacement)
        except BrokerError as exc:
            return ExecutionResult(
                outcome=ExecutionOutcome.AMBIGUOUS
                if isinstance(exc, BrokerAmbiguousSubmission)
                else ExecutionOutcome.REFUSED,
                order_request_id=decision.decision_record_id,
                client_order_id="",
                broker_order_id=broker_order_id,
                reason_code=type(exc).__name__,
                detail=str(exc),
            )
        return ExecutionResult(
            outcome=ExecutionOutcome.SUBMITTED,
            order_request_id=decision.decision_record_id,
            client_order_id=order.client_order_id,
            broker_order_id=order.broker_order_id,
            status=order.status,
        )

    # -- preflight ----------------------------------------------------------

    def _require_paper(self) -> None:
        if self._environment is RuntimeEnvironment.PRODUCTION_LIVE:
            raise GatewayPreflightError("PRODUCTION_LIVE broker writes are refused")

    def _preflight(self, *, decision: DecisionRecord, request: OrderRequest) -> None:
        self._require_paper()
        if request.environment is RuntimeEnvironment.PRODUCTION_LIVE:
            raise GatewayPreflightError("the request is marked PRODUCTION_LIVE")
        if request.environment is not self._environment:
            raise GatewayPreflightError(
                "request environment does not match the gateway environment"
            )
        if request.account_alias != self._account_alias:
            raise GatewayPreflightError(
                "request account alias does not match the gateway account"
            )
        if request.request_hash != request.computed_request_hash():
            raise GatewayPreflightError("order request hash does not match its content")
        if decision.status != "COMPLETE":
            raise GatewayPreflightError("only a complete decision may be executed")
        if decision.decision_record_id != request.decision_id:
            raise GatewayPreflightError("order request belongs to another decision")
        if request.order_request_id != order_request_id_for(
            request.decision_id, request.request_hash
        ):
            raise GatewayPreflightError(
                "order_request_id is not the deterministic identity for this content"
            )
        now = self._clock()
        if now >= request.expires_at:
            raise GatewayPreflightError("the order request has expired")

    # -- durable path -------------------------------------------------------

    def _submit_durable(
        self,
        *,
        decision: DecisionRecord,
        request: OrderRequest,
        client_order_id: str,
    ) -> ExecutionResult:
        now = self._clock()

        # 1. Durable reservation. Nothing may reach the broker before this.
        with self._open_uow() as uow:
            try:
                attempt, created = uow.executions.reserve_or_load_attempt(
                    request=request,
                    client_order_id=client_order_id,
                    reserved_at=now,
                )
            except SubmissionConflict as exc:
                raise GatewayConflict(str(exc)) from exc
            if attempt.is_resolved:
                # Already settled: return what happened, never send again.
                uow.commit()
                return self._result_for_resolved(attempt)
            claimed = uow.executions.claim_submission(
                submission_attempt_id=attempt.submission_attempt_id,
                owner=self._worker_id,
                claim_token=str(uuid4()),
                lease_until=now + self._lease,
                now=now,
            )
            if not claimed:
                uow.commit()
                return ExecutionResult(
                    outcome=ExecutionOutcome.DUPLICATE,
                    order_request_id=request.order_request_id,
                    client_order_id=client_order_id,
                    submission_attempt_id=attempt.submission_attempt_id,
                    reason_code="SUBMISSION_LEASE_HELD",
                    detail="another worker owns this submission",
                )
            attempt = uow.executions.get_attempt(attempt.submission_attempt_id)
            uow.commit()

        token = attempt.claim_token

        # 2. Durable intent journal. Still no broker call.
        intent_event = self._intent_event(request, client_order_id, now)
        journal_result = self._journal.write(intent_event)
        if not journal_result.is_durable:
            with self._open_uow() as uow:
                uow.executions.transition_attempt(
                    submission_attempt_id=attempt.submission_attempt_id,
                    expected=SubmissionAttemptStatus.RESERVED,
                    target=SubmissionAttemptStatus.REJECTED,
                    claim_token=token,
                    error_code="JOURNAL_NOT_DURABLE",
                    resolved_at=self._clock(),
                )
                uow.commit()
            return ExecutionResult(
                outcome=ExecutionOutcome.REFUSED,
                order_request_id=request.order_request_id,
                client_order_id=client_order_id,
                submission_attempt_id=attempt.submission_attempt_id,
                reason_code="JOURNAL_NOT_DURABLE",
                detail="; ".join(journal_result.failures),
            )

        with self._open_uow() as uow:
            uow.executions.record_journal_receipt(
                submission_attempt_id=attempt.submission_attempt_id,
                event_id=intent_event.event_id,
                event_hash=intent_event.event_hash,
                backend=journal_result.receipts[0].backend.value,
                locator=journal_result.receipts[0].locator.uri,
                journaled_at=self._clock(),
            )
            uow.executions.transition_attempt(
                submission_attempt_id=attempt.submission_attempt_id,
                expected=SubmissionAttemptStatus.RESERVED,
                target=SubmissionAttemptStatus.JOURNALED,
                claim_token=token,
            )
            uow.executions.transition_attempt(
                submission_attempt_id=attempt.submission_attempt_id,
                expected=SubmissionAttemptStatus.JOURNALED,
                target=SubmissionAttemptStatus.SUBMITTING,
                claim_token=token,
                broker_called_at=self._clock(),
            )
            uow.commit()

        # 3. Exactly one broker call, outside any database transaction.
        return self._call_broker_once(
            request=request,
            attempt_id=attempt.submission_attempt_id,
            client_order_id=client_order_id,
            token=token,
            intent_event=intent_event,
        )

    def _call_broker_once(
        self,
        *,
        request: OrderRequest,
        attempt_id: UUID,
        client_order_id: str,
        token: str | None,
        intent_event: ExecutionJournalEvent,
    ) -> ExecutionResult:
        try:
            order = self._broker.submit(request)
        except BrokerRejected as exc:
            return self._resolve_rejected(
                request, attempt_id, client_order_id, token, intent_event, exc
            )
        except (BrokerAmbiguousSubmission, BrokerUnavailable, BrokerAuthenticationError) as exc:
            # Never resubmit. Ask the broker whether our deterministic client
            # order ID exists; a failed lookup is ambiguous, not "absent".
            return self._resolve_ambiguous(
                request, attempt_id, client_order_id, token, intent_event, exc
            )

        response_event = self._response_event(
            intent_event, request, client_order_id, order
        )
        response_result = self._journal.write(response_event)
        outcome = ExecutionOutcome.SUBMITTED
        reason: str | None = None
        if not response_result.is_durable:
            # The broker answered but the evidence is not durable. This is not
            # a rejection and must never be resubmitted.
            outcome = ExecutionOutcome.AMBIGUOUS
            reason = "RESPONSE_JOURNAL_NOT_DURABLE"

        try:
            with self._open_uow() as uow:
                uow.executions.transition_attempt(
                    submission_attempt_id=attempt_id,
                    expected=SubmissionAttemptStatus.SUBMITTING,
                    target=(
                        SubmissionAttemptStatus.ACCEPTED
                        if outcome is ExecutionOutcome.SUBMITTED
                        else SubmissionAttemptStatus.AMBIGUOUS
                    ),
                    claim_token=token,
                    broker_order_id=order.broker_order_id,
                    error_code=reason,
                    resolved_at=self._clock(),
                )
                uow.commit()
        except _PostgresUnavailable:
            outcome = ExecutionOutcome.RECONCILIATION_REQUIRED
            reason = "POSTGRES_UNAVAILABLE_AFTER_BROKER"

        return ExecutionResult(
            outcome=outcome,
            order_request_id=request.order_request_id,
            client_order_id=client_order_id,
            submission_attempt_id=attempt_id,
            broker_order_id=order.broker_order_id,
            status=order.status,
            reason_code=reason,
            journal_event_ids=(intent_event.event_id, response_event.event_id),
        )

    def _resolve_rejected(
        self,
        request: OrderRequest,
        attempt_id: UUID,
        client_order_id: str,
        token: str | None,
        intent_event: ExecutionJournalEvent,
        exc: BrokerError,
    ) -> ExecutionResult:
        rejection_event = link_event(
            intent_event,
            event_id=uuid4(),
            event_time=self._clock(),
            observed_at=self._clock(),
            account_id=self._account_alias,
            environment=self._environment,
            event_type="BROKER_REJECTED",
            decision_id=request.decision_id,
            client_order_id=client_order_id,
            broker_order_id=None,
            payload={"error": type(exc).__name__, "detail": str(exc)},
        )
        self._journal.write(rejection_event)
        with self._open_uow() as uow:
            uow.executions.transition_attempt(
                submission_attempt_id=attempt_id,
                expected=SubmissionAttemptStatus.SUBMITTING,
                target=SubmissionAttemptStatus.REJECTED,
                claim_token=token,
                error_code=type(exc).__name__,
                resolved_at=self._clock(),
            )
            uow.commit()
        return ExecutionResult(
            outcome=ExecutionOutcome.REJECTED,
            order_request_id=request.order_request_id,
            client_order_id=client_order_id,
            submission_attempt_id=attempt_id,
            reason_code=type(exc).__name__,
            detail=str(exc),
            journal_event_ids=(intent_event.event_id, rejection_event.event_id),
        )

    def _resolve_ambiguous(
        self,
        request: OrderRequest,
        attempt_id: UUID,
        client_order_id: str,
        token: str | None,
        intent_event: ExecutionJournalEvent,
        exc: BrokerError,
    ) -> ExecutionResult:
        found: BrokerOrder | None = None
        lookup_failed = False
        try:
            found = self._broker.find_by_client_order_id(client_order_id)
        except BrokerError:
            # A failed lookup proves nothing. It is never "not found".
            lookup_failed = True

        if found is not None:
            with self._open_uow() as uow:
                uow.executions.transition_attempt(
                    submission_attempt_id=attempt_id,
                    expected=SubmissionAttemptStatus.SUBMITTING,
                    target=SubmissionAttemptStatus.ACCEPTED,
                    claim_token=token,
                    broker_order_id=found.broker_order_id,
                    resolved_at=self._clock(),
                )
                uow.commit()
            return ExecutionResult(
                outcome=ExecutionOutcome.SUBMITTED,
                order_request_id=request.order_request_id,
                client_order_id=client_order_id,
                submission_attempt_id=attempt_id,
                broker_order_id=found.broker_order_id,
                status=found.status,
                reason_code="RESOLVED_BY_CLIENT_ORDER_ID",
                journal_event_ids=(intent_event.event_id,),
            )

        target = SubmissionAttemptStatus.AMBIGUOUS
        outcome = ExecutionOutcome.AMBIGUOUS
        if not lookup_failed:
            # The broker positively reports no such order, so nothing landed.
            target = SubmissionAttemptStatus.REJECTED
            outcome = ExecutionOutcome.REJECTED
        with self._open_uow() as uow:
            uow.executions.transition_attempt(
                submission_attempt_id=attempt_id,
                expected=SubmissionAttemptStatus.SUBMITTING,
                target=target,
                claim_token=token,
                error_code=type(exc).__name__,
                resolved_at=self._clock(),
            )
            uow.commit()
        return ExecutionResult(
            outcome=outcome,
            order_request_id=request.order_request_id,
            client_order_id=client_order_id,
            submission_attempt_id=attempt_id,
            reason_code=type(exc).__name__,
            detail=str(exc),
            journal_event_ids=(intent_event.event_id,),
        )

    # -- fail-operational exits ---------------------------------------------

    def _submit_fail_operational_exit(
        self,
        *,
        request: OrderRequest,
        client_order_id: str,
        cause: Exception,
    ) -> ExecutionResult:
        """Let a deterministic exit through while PostgreSQL is down.

        The exit is bounded by broker-authoritative position state so it can
        only ever reduce exposure. Anything that could increase it, flip side,
        or open new legs is refused even here.
        """

        try:
            positions = self._broker.positions()
        except BrokerError as exc:
            return ExecutionResult(
                outcome=ExecutionOutcome.REFUSED,
                order_request_id=request.order_request_id,
                client_order_id=client_order_id,
                reason_code="BROKER_POSITION_STATE_UNAVAILABLE",
                detail=str(exc),
            )

        violation = check_exit_reduces_exposure(request, positions)
        if violation is not None:
            return ExecutionResult(
                outcome=ExecutionOutcome.REFUSED,
                order_request_id=request.order_request_id,
                client_order_id=client_order_id,
                reason_code=violation,
                detail="fail-operational exits may only reduce exposure",
            )

        now = self._clock()
        intent_event = self._intent_event(
            request,
            client_order_id,
            now,
            postgres_status=PostgresPersistenceStatus.RECONCILIATION_REQUIRED,
        )
        journal_result = self._journal.write(intent_event)
        if not journal_result.is_durable:
            return ExecutionResult(
                outcome=ExecutionOutcome.REFUSED,
                order_request_id=request.order_request_id,
                client_order_id=client_order_id,
                reason_code="JOURNAL_NOT_DURABLE",
                detail="a fail-operational exit still requires a durable journal",
            )

        try:
            order = self._broker.submit(request)
        except BrokerError as exc:
            return ExecutionResult(
                outcome=ExecutionOutcome.AMBIGUOUS
                if isinstance(exc, BrokerAmbiguousSubmission)
                else ExecutionOutcome.REJECTED,
                order_request_id=request.order_request_id,
                client_order_id=client_order_id,
                reason_code=type(exc).__name__,
                detail=str(exc),
                journal_event_ids=(intent_event.event_id,),
            )

        response_event = self._response_event(
            intent_event,
            request,
            client_order_id,
            order,
            postgres_status=PostgresPersistenceStatus.RECONCILIATION_REQUIRED,
        )
        self._journal.write(response_event)
        return ExecutionResult(
            outcome=ExecutionOutcome.RECONCILIATION_REQUIRED,
            order_request_id=request.order_request_id,
            client_order_id=client_order_id,
            broker_order_id=order.broker_order_id,
            status=order.status,
            reason_code="POSTGRES_UNAVAILABLE",
            detail="exit executed on journal evidence; backfill PostgreSQL",
            journal_event_ids=(intent_event.event_id, response_event.event_id),
        )

    # -- helpers ------------------------------------------------------------

    @contextmanager
    def _open_uow(self) -> Any:
        """Open one transaction, translating any outage into a typed failure.

        Construction and connection are both covered: a UnitOfWork does not
        touch the database until it is entered, so catching only the factory
        call would miss a real outage.
        """

        try:
            unit = self._uow_factory()
        except Exception as exc:
            raise _PostgresUnavailable(str(exc)) from exc
        try:
            with unit as bound:
                yield bound
        except _PostgresUnavailable:
            raise
        except (ConnectionError, TimeoutError, OSError) as exc:
            raise _PostgresUnavailable(str(exc)) from exc
        except SQLAlchemyError as exc:
            raise _PostgresUnavailable(f"{type(exc).__name__}: {exc}") from exc

    def _intent_event(
        self,
        request: OrderRequest,
        client_order_id: str,
        now: datetime,
        *,
        postgres_status: PostgresPersistenceStatus = PostgresPersistenceStatus.PENDING,
    ) -> ExecutionJournalEvent:
        return ExecutionJournalEvent.create(
            event_id=uuid5(ORDER_REQUEST_NAMESPACE, f"intent|{request.order_request_id}"),
            event_time=now,
            observed_at=now,
            account_id=self._account_alias,
            environment=self._environment,
            event_type="INTENT_TO_SUBMIT",
            decision_id=request.decision_id,
            order_request_id=request.order_request_id,
            sequence_no=1,
            client_order_id=client_order_id,
            broker_order_id=None,
            payload={
                "request_hash": request.request_hash,
                "instrument_family": request.instrument_family.value,
                "parent_quantity": str(request.parent_quantity),
                "order_type": request.order_type,
                "risk_reducing": request.risk_reducing,
            },
            postgres_persistence_status=postgres_status,
        )

    def _response_event(
        self,
        intent_event: ExecutionJournalEvent,
        request: OrderRequest,
        client_order_id: str,
        order: BrokerOrder,
        *,
        postgres_status: PostgresPersistenceStatus = PostgresPersistenceStatus.PENDING,
    ) -> ExecutionJournalEvent:
        return link_event(
            intent_event,
            event_id=uuid5(
                ORDER_REQUEST_NAMESPACE, f"response|{request.order_request_id}"
            ),
            event_time=order.observed_at,
            observed_at=order.observed_at,
            account_id=self._account_alias,
            environment=self._environment,
            event_type="BROKER_RESPONSE",
            decision_id=request.decision_id,
            client_order_id=client_order_id,
            broker_order_id=order.broker_order_id,
            payload={
                "status": order.status.value,
                "raw_status": order.raw_status,
                "filled_quantity": str(order.filled_quantity),
                "response": dict(order.raw),
            },
            postgres_persistence_status=postgres_status,
        )

    def _result_for_resolved(self, attempt: Any) -> ExecutionResult:
        mapping = {
            SubmissionAttemptStatus.ACCEPTED: ExecutionOutcome.DUPLICATE,
            SubmissionAttemptStatus.REJECTED: ExecutionOutcome.REJECTED,
            SubmissionAttemptStatus.RECONCILIATION_REQUIRED: (
                ExecutionOutcome.RECONCILIATION_REQUIRED
            ),
        }
        return ExecutionResult(
            outcome=mapping[attempt.status],
            order_request_id=attempt.order_request_id,
            client_order_id=attempt.client_order_id,
            submission_attempt_id=attempt.submission_attempt_id,
            broker_order_id=attempt.broker_order_id,
            reason_code="ALREADY_RESOLVED",
        )


class _PostgresUnavailable(Exception):
    """Raised internally when the database cannot be reached."""


def check_exit_reduces_exposure(
    request: OrderRequest,
    positions: tuple[Any, ...],
) -> str | None:
    """Return a refusal code when an exit would not purely reduce exposure."""

    if not request.risk_reducing or request.decision_kind is DecisionKind.ENTRY:
        return "NOT_RISK_REDUCING"
    if request.legs:
        # Opening new option legs under a database outage is never an exit.
        opening = [
            leg for leg in request.legs if leg.position_intent.value.endswith("TO_OPEN")
        ]
        if opening:
            return "EXIT_OPENS_NEW_LEGS"

    symbol = request.equity_symbol or (request.legs[0].symbol if request.legs else None)
    if symbol is None:
        return "EXIT_HAS_NO_INSTRUMENT"

    held = _ZERO
    for position in positions:
        if position.symbol == symbol:
            held += Decimal(str(position.quantity))
    if held == _ZERO:
        return "EXIT_WITHOUT_A_HELD_POSITION"

    quantity = request.parent_quantity
    if quantity > abs(held):
        return "EXIT_EXCEEDS_HELD_QUANTITY"

    side = request.equity_side or (request.legs[0].side if request.legs else None)
    if held > _ZERO and side is not OrderSide.SELL:
        return "EXIT_WOULD_INCREASE_A_LONG"
    if held < _ZERO and side is not OrderSide.BUY:
        return "EXIT_WOULD_INCREASE_A_SHORT"
    return None


__all__ = [
    "CLIENT_ORDER_ID_LENGTH",
    "ENVIRONMENT_CODES",
    "ORDER_REQUEST_NAMESPACE",
    "ExecutionGateway",
    "ExecutionOutcome",
    "ExecutionResult",
    "GatewayConflict",
    "GatewayError",
    "GatewayPreflightError",
    "check_exit_reduces_exposure",
    "client_order_id_for",
    "order_request_id_for",
]
