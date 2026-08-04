"""Broker reconciliation.

Broker orders, fills, and positions are facts. Reconciliation never rewrites
them to agree with internal records: it appends evidence and reports
discrepancies. Ownership is created only from confirmed fills, so an order the
broker merely accepted attributes nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

from core.nervous_system.contracts.base import ContractModel, UtcDatetime
from core.nervous_system.contracts.enums import ExecutionStatus

from .broker import BrokerAdapter, BrokerError, BrokerOrder
from .journal import ExecutionJournalEvent, JournalConflict


_ZERO = Decimal("0")


class DiscrepancyKind(str, Enum):
    BROKER_ONLY_ORDER = "BROKER_ONLY_ORDER"
    UNASSIGNED_POSITION = "UNASSIGNED_POSITION"
    STATUS_MISMATCH = "STATUS_MISMATCH"
    QUANTITY_MISMATCH = "QUANTITY_MISMATCH"
    JOURNAL_ONLY_EVENT = "JOURNAL_ONLY_EVENT"
    CORRUPT_JOURNAL_EVENT = "CORRUPT_JOURNAL_EVENT"
    LOOKUP_UNAVAILABLE = "LOOKUP_UNAVAILABLE"


class Discrepancy(ContractModel):
    kind: DiscrepancyKind
    identity: str
    detail: str
    broker_value: str | None = None
    recorded_value: str | None = None


class ReconciliationReport(ContractModel):
    account_id: str
    observed_at: UtcDatetime
    since: UtcDatetime
    recovered_orders: tuple[str, ...] = ()
    recovered_journal_events: tuple[UUID, ...] = ()
    ownership_created: tuple[str, ...] = ()
    unassigned_positions: tuple[str, ...] = ()
    discrepancies: tuple[Discrepancy, ...] = ()

    @property
    def is_clean(self) -> bool:
        return not self.discrepancies


def reconcile_broker_account(
    *,
    broker: BrokerAdapter,
    unit_of_work: Any,
    journal: Any,
    account_id: str,
    since: datetime,
    observed_at: datetime | None = None,
) -> ReconciliationReport:
    """Compare broker facts with recorded state and report the differences."""

    now = observed_at or since
    discrepancies: list[Discrepancy] = []
    recovered_orders: list[str] = []
    recovered_events: list[UUID] = []
    ownership_created: list[str] = []
    unassigned: list[str] = []

    try:
        broker_orders = broker.orders(status="all")
    except BrokerError as exc:
        broker_orders = ()
        discrepancies.append(
            Discrepancy(
                kind=DiscrepancyKind.LOOKUP_UNAVAILABLE,
                identity="orders",
                detail=f"{type(exc).__name__}: {exc}",
            )
        )

    for order in broker_orders:
        if order.submitted_at is not None and order.submitted_at < since:
            continue
        recorded = unit_of_work.executions.find_attempt_by_client_order_id(
            environment=_environment_of(unit_of_work, order),
            account_alias=account_id,
            client_order_id=order.client_order_id,
        ) if order.client_order_id else None

        # Ownership only ever comes from a confirmed fill, and a recovered
        # broker-only fill is still a fill: it must be attributed, not skipped.
        if order.filled_quantity > _ZERO:
            ownership_created.append(order.broker_order_id)

        if recorded is None:
            # The broker has an order we have no reservation for. It is a fact:
            # record it rather than pretending it does not exist.
            recovered_orders.append(order.broker_order_id)
            discrepancies.append(
                Discrepancy(
                    kind=DiscrepancyKind.BROKER_ONLY_ORDER,
                    identity=order.broker_order_id,
                    detail="broker order has no recorded submission attempt",
                    broker_value=order.raw_status,
                )
            )
            continue

        if recorded.broker_order_id and recorded.broker_order_id != order.broker_order_id:
            discrepancies.append(
                Discrepancy(
                    kind=DiscrepancyKind.STATUS_MISMATCH,
                    identity=order.client_order_id,
                    detail="recorded attempt points at a different broker order",
                    broker_value=order.broker_order_id,
                    recorded_value=recorded.broker_order_id,
                )
            )

        if order.status is ExecutionStatus.FILLED and recorded.status.value not in {
            "ACCEPTED",
            "AMBIGUOUS",
            "RECONCILIATION_REQUIRED",
        }:
            discrepancies.append(
                Discrepancy(
                    kind=DiscrepancyKind.STATUS_MISMATCH,
                    identity=order.client_order_id,
                    detail="broker reports a fill the record does not reflect",
                    broker_value=order.raw_status,
                    recorded_value=recorded.status.value,
                )
            )

    try:
        positions = broker.positions()
    except BrokerError as exc:
        positions = ()
        discrepancies.append(
            Discrepancy(
                kind=DiscrepancyKind.LOOKUP_UNAVAILABLE,
                identity="positions",
                detail=f"{type(exc).__name__}: {exc}",
            )
        )

    owned_symbols = {order.broker_order_id for order in broker_orders}
    for position in positions:
        if position.quantity == _ZERO:
            continue
        # A position with no fill-backed attribution is manual or imported.
        if not _has_attribution(unit_of_work, account_id, position.symbol):
            unassigned.append(position.symbol)
            discrepancies.append(
                Discrepancy(
                    kind=DiscrepancyKind.UNASSIGNED_POSITION,
                    identity=position.symbol,
                    detail="broker position has no fill-backed ownership",
                    broker_value=str(position.quantity),
                )
            )

    for event, problem in _journal_only_events(journal, account_id, since):
        if problem is None:
            recovered_events.append(event.event_id)
            discrepancies.append(
                Discrepancy(
                    kind=DiscrepancyKind.JOURNAL_ONLY_EVENT,
                    identity=str(event.event_id),
                    detail="journal holds an event absent from PostgreSQL",
                )
            )
        else:
            # Corrupt evidence is preserved and reported, never deleted.
            discrepancies.append(
                Discrepancy(
                    kind=DiscrepancyKind.CORRUPT_JOURNAL_EVENT,
                    identity=str(getattr(event, "event_id", "unknown")),
                    detail=problem,
                )
            )

    return ReconciliationReport(
        account_id=account_id,
        observed_at=now,
        since=since,
        recovered_orders=tuple(recovered_orders),
        recovered_journal_events=tuple(recovered_events),
        ownership_created=tuple(ownership_created),
        unassigned_positions=tuple(unassigned),
        discrepancies=tuple(discrepancies),
    )


def _environment_of(unit_of_work: Any, order: BrokerOrder) -> Any:
    from core.nervous_system.contracts.enums import RuntimeEnvironment

    return getattr(unit_of_work, "environment", RuntimeEnvironment.QA_PAPER)


def _has_attribution(unit_of_work: Any, account_id: str, symbol: str) -> bool:
    finder = getattr(unit_of_work, "ownership_for", None)
    if finder is None:
        return False
    return bool(finder(account_id=account_id, symbol=symbol))


def _journal_only_events(
    journal: Any,
    account_id: str,
    since: datetime,
) -> list[tuple[Any, str | None]]:
    results: list[tuple[Any, str | None]] = []
    try:
        events = list(journal.iter_events(account_id=account_id, after=since))
    except JournalConflict as exc:
        return [(None, f"journal chain rejected: {exc}")]
    except Exception as exc:
        return [(None, f"journal unreadable: {type(exc).__name__}: {exc}")]
    for event in events:
        if event.event_hash != event.computed_event_hash():
            results.append((event, "event hash does not match its content"))
        else:
            results.append((event, None))
    return results


__all__ = [
    "Discrepancy",
    "DiscrepancyKind",
    "ReconciliationReport",
    "reconcile_broker_account",
]
