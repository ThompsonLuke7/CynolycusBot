"""Fakes for gateway tests: no broker, network, or credentials (Task 21)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from core.nervous_system.contracts.decisions import (
    DecisionRecord,
    HashedDecisionArtifact,
)
from core.nervous_system.contracts.enums import (
    AssetClass,
    DebitCredit,
    DecisionKind,
    ExecutionStatus,
    InstrumentFamily,
    OrderSide,
    RuntimeEnvironment,
)
from core.nervous_system.contracts.orders import OrderRequest
from core.nervous_system.contracts.states import PortfolioPosition
from core.nervous_system.execution.broker import (
    BrokerOrder,
    BrokerPosition,
    BrokerError,
)
from core.nervous_system.execution.gateway import (
    client_order_id_for,
    order_request_id_for,
)
from core.nervous_system.execution.journal import (
    CompositeJournalResult,
    CompositeStatus,
    JournalBackend,
    JournalLocator,
    JournalReceipt,
    JournalWriteStatus,
)


UTC = timezone.utc
NOW = datetime(2026, 8, 3, 18, 30, tzinfo=UTC)
D = Decimal
DECISION_ID = uuid5(NAMESPACE_URL, "gateway-test/decision")
ACCOUNT = "paper"


def clock(value: datetime = NOW):
    return lambda: value


# --------------------------------------------------------------------------
# Contract builders
# --------------------------------------------------------------------------


def decision_record(**overrides: Any) -> DecisionRecord:
    artifacts = tuple(
        HashedDecisionArtifact.from_payload(stage, 1, {"stage": stage, "status": "RUN"})
        for stage in (
            "RAW_STRATEGY_OUTPUT",
            "EXPOSURE_REPORT",
            "INSTRUMENT_CANDIDATES",
            "INSTRUMENT_SELECTION",
        )
    )
    fields: dict[str, Any] = {
        "decision_record_id": DECISION_ID,
        "decision_time": NOW,
        "snapshot_id": uuid5(NAMESPACE_URL, "gateway-test/snapshot"),
        "intent_id": uuid5(NAMESPACE_URL, "gateway-test/intent"),
        "policy_decision_id": uuid5(NAMESPACE_URL, "gateway-test/policy"),
        "source_manifest_hash": "1" * 64,
        "snapshot_hash": "2" * 64,
        "intent_hash": "3" * 64,
        "policy_hash": "4" * 64,
        "raw_strategy_output": artifacts[0],
        "exposure_report": artifacts[1],
        "instrument_candidates": artifacts[2],
        "instrument_selection": artifacts[3],
        "config_hash": "5" * 64,
    }
    fields.update(overrides)
    return DecisionRecord(**fields)


def order_request(**overrides: Any) -> OrderRequest:
    """Build a request whose ID is the deterministic identity for its content."""

    fields: dict[str, Any] = {
        "decision_id": DECISION_ID,
        "policy_decision_id": uuid5(NAMESPACE_URL, "gateway-test/policy"),
        "environment": RuntimeEnvironment.QA_PAPER,
        "account_alias": ACCOUNT,
        "decision_kind": DecisionKind.ENTRY,
        "risk_reducing": False,
        "instrument_family": InstrumentFamily.EQUITY,
        "equity_symbol": "AMD",
        "equity_side": OrderSide.BUY,
        "parent_quantity": D("25"),
        "debit_credit": DebitCredit.DEBIT,
        "net_limit_price": None,
        "maximum_loss": D("5000"),
        "buying_power_required": D("5000"),
        "time_in_force": "day",
        "order_type": "market",
        "idempotency_key": "ab" * 32,
        "created_at": NOW - timedelta(minutes=1),
        "expires_at": NOW + timedelta(minutes=20),
    }
    fields.update(overrides)
    provisional = OrderRequest.create(order_request_id=uuid5(NAMESPACE_URL, "tmp"), **fields)
    # Rebuild with the deterministic identity for this exact content.
    return OrderRequest.create(
        order_request_id=order_request_id_for(
            provisional.decision_id, provisional.request_hash
        ),
        **fields,
    )


def exit_request(**overrides: Any) -> OrderRequest:
    fields: dict[str, Any] = {
        "decision_kind": DecisionKind.EXIT,
        "risk_reducing": True,
        "broker_position_key": "paper:AMD",
        "equity_side": OrderSide.SELL,
        "parent_quantity": D("25"),
    }
    fields.update(overrides)
    return order_request(**fields)


def expected_client_order_id(request: OrderRequest) -> str:
    return client_order_id_for(request.environment, request.request_hash)


def broker_order(request: OrderRequest, **overrides: Any) -> BrokerOrder:
    fields: dict[str, Any] = {
        "broker_order_id": "brk-1",
        "client_order_id": expected_client_order_id(request),
        "status": ExecutionStatus.ACCEPTED,
        "raw_status": "accepted",
        "submitted_at": NOW,
        "updated_at": NOW,
        "filled_quantity": D("0"),
        "observed_at": NOW,
        "raw": {"id": "brk-1", "status": "accepted"},
    }
    fields.update(overrides)
    return BrokerOrder(**fields)


def broker_position(symbol: str = "AMD", quantity: float = 100.0) -> BrokerPosition:
    return BrokerPosition(
        symbol=symbol,
        asset_class=AssetClass.EQUITY,
        quantity=D(str(quantity)),
        observed_at=NOW,
    )


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeBroker:
    """Counts submissions so a duplicate POST is impossible to miss."""

    def __init__(
        self,
        *,
        submit_result: Any = None,
        lookup_result: Any = "__absent__",
        positions_result: Any = (),
    ) -> None:
        self.submit_calls: list[OrderRequest] = []
        self.lookup_calls: list[str] = []
        self._submit_result = submit_result
        self._lookup_result = lookup_result
        self._positions_result = positions_result

    def submit(self, request: OrderRequest) -> BrokerOrder:
        self.submit_calls.append(request)
        result = self._submit_result
        if callable(result):
            result = result(request)
        if isinstance(result, Exception):
            raise result
        return result if result is not None else broker_order(request)

    def find_by_client_order_id(self, client_order_id: str):
        self.lookup_calls.append(client_order_id)
        result = self._lookup_result
        if isinstance(result, Exception):
            raise result
        return None if result == "__absent__" else result

    def positions(self):
        result = self._positions_result
        if isinstance(result, Exception):
            raise result
        return result

    def orders(self, *, status: str = "all"):
        return ()

    def account(self):  # pragma: no cover - unused by these tests
        raise NotImplementedError

    def cancel(self, broker_order_id: str):
        return broker_order(order_request(), broker_order_id=broker_order_id,
                            status=ExecutionStatus.CANCELED, raw_status="canceled")

    def replace(self, broker_order_id: str, replacement):
        return broker_order(order_request(), broker_order_id=broker_order_id,
                            status=ExecutionStatus.ACCEPTED, raw_status="replaced")


class FakeJournal:
    """Records writes and can be told to fail a specific event type."""

    def __init__(self, *, fail_on: tuple[str, ...] = ()) -> None:
        self.events: list[Any] = []
        self.fail_on = fail_on

    def write(self, event) -> CompositeJournalResult:
        self.events.append(event)
        if event.event_type in self.fail_on:
            return CompositeJournalResult(
                status=CompositeStatus.FAILED,
                receipts=(),
                failures=(f"forced failure for {event.event_type}",),
            )
        return CompositeJournalResult(
            status=CompositeStatus.DURABLE,
            receipts=(
                JournalReceipt(
                    backend=JournalBackend.LOCAL,
                    locator=JournalLocator(
                        backend=JournalBackend.LOCAL,
                        object_name=event.object_name,
                        uri=f"/tmp/{event.object_name}",
                    ),
                    content_hash=event.event_hash,
                    status=JournalWriteStatus.WRITTEN,
                ),
            ),
        )

    def types(self) -> list[str]:
        return [event.event_type for event in self.events]

    def iter_events(self, *, account_id: str, after=None):
        return iter(())


class FakeExecutionRepository:
    """In-memory mirror of the submission-attempt repository semantics.

    The concurrency primitives it models are also asserted against real
    PostgreSQL, so this fake speeds up flow tests without becoming the only
    evidence that leases and compare-and-set work.
    """

    def __init__(self, store: dict[str, Any]) -> None:
        self._store = store

    def reserve_or_load_attempt(
        self, *, request, client_order_id, reserved_at, attempt_no=1,
        submission_attempt_id=None,
    ):
        from core.nervous_system.contracts.execution import SubmissionAttemptRecord
        from core.nervous_system.contracts.enums import SubmissionAttemptStatus
        from core.nervous_system.persistence.repositories.execution import (
            SubmissionConflict,
        )

        key = (request.environment.value, request.account_alias, client_order_id)
        existing = self._store.get(key)
        if existing is not None:
            if existing.request_hash != request.request_hash:
                raise SubmissionConflict(
                    f"client order ID {client_order_id} already reserved for "
                    "different request content"
                )
            return existing, False
        record = SubmissionAttemptRecord(
            submission_attempt_id=submission_attempt_id or uuid4(),
            order_request_id=request.order_request_id,
            attempt_no=attempt_no,
            environment=request.environment,
            account_alias=request.account_alias,
            client_order_id=client_order_id,
            status=SubmissionAttemptStatus.RESERVED,
            request_hash=request.request_hash,
            reserved_at=reserved_at,
        )
        self._store[key] = record
        self._store[record.submission_attempt_id] = record
        return record, True

    def claim_submission(
        self, *, submission_attempt_id, owner, claim_token, lease_until, now
    ) -> bool:
        from core.nervous_system.contracts.enums import SubmissionAttemptStatus

        record = self._store.get(submission_attempt_id)
        if record is None or record.status not in {
            SubmissionAttemptStatus.RESERVED,
            SubmissionAttemptStatus.JOURNALED,
        }:
            return False
        held = (
            record.lease_owner is not None
            and record.lease_until is not None
            and now < record.lease_until
            and record.lease_owner != owner
        )
        if held:
            return False
        self._replace(
            record,
            lease_owner=owner,
            lease_until=lease_until,
            claim_token=claim_token,
        )
        return True

    def transition_attempt(
        self, *, submission_attempt_id, expected, target, claim_token=None, **updates
    ) -> bool:
        from core.nervous_system.persistence.repositories.execution import (
            is_legal_transition,
        )

        if not is_legal_transition(expected, target):
            raise ValueError(
                f"illegal submission transition {expected.value} -> {target.value}"
            )
        record = self._store.get(submission_attempt_id)
        if record is None or record.status is not expected:
            return False
        if claim_token is not None and record.claim_token != claim_token:
            return False
        self._replace(record, status=target, **updates)
        return True

    def record_journal_receipt(
        self, *, submission_attempt_id, event_id, event_hash, backend, locator,
        journaled_at,
    ) -> None:
        record = self._store.get(submission_attempt_id)
        if record is not None:
            self._replace(
                record,
                journal_event_id=event_id,
                journal_event_hash=event_hash,
                journal_backend=backend,
                journal_locator=locator,
                journaled_at=journaled_at,
            )

    def get_attempt(self, submission_attempt_id):
        return self._store.get(submission_attempt_id)

    def find_attempt_by_client_order_id(
        self, *, environment, account_alias, client_order_id
    ):
        return self._store.get((environment.value, account_alias, client_order_id))

    def _replace(self, record, **updates) -> None:
        updated = record.model_copy(update=updates)
        key = (
            record.environment.value,
            record.account_alias,
            record.client_order_id,
        )
        self._store[key] = updated
        self._store[record.submission_attempt_id] = updated


class FakeUnitOfWork:
    """Context-managed fake with the same shape as the real UnitOfWork."""

    def __init__(self, store: dict[str, Any], *, fail: bool = False) -> None:
        self._store = store
        self._fail = fail
        self.commits = 0

    def __enter__(self) -> "FakeUnitOfWork":
        if self._fail:
            raise ConnectionError("postgres is down")
        self.executions = FakeExecutionRepository(self._store)
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def commit(self) -> None:
        self.commits += 1


def fake_uow_factory(store: dict[str, Any] | None = None, *, fail: bool = False):
    shared = store if store is not None else {}
    return lambda: FakeUnitOfWork(shared, fail=fail), shared


__all__ = [
    "ACCOUNT",
    "FakeExecutionRepository",
    "FakeUnitOfWork",
    "fake_uow_factory",
    "DECISION_ID",
    "NOW",
    "FakeBroker",
    "FakeJournal",
    "broker_order",
    "broker_position",
    "clock",
    "decision_record",
    "exit_request",
    "expected_client_order_id",
    "order_request",
]
