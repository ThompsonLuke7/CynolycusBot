"""Reconciliation runs and the alert projection over an immutable event log."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4, uuid5, NAMESPACE_URL

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.nervous_system.persistence.models import (
    Alert as AlertRow,
    AlertEvent as AlertEventRow,
    ReconciliationItem as ReconciliationItemRow,
    ReconciliationRun as ReconciliationRunRow,
)


CRITICAL = "CRITICAL"
OPEN = "OPEN"


def alert_dedup_key(*, code: str, component: str, entity_id: str | None) -> str:
    """What makes two detections 'the same problem'."""

    return f"{code}|{component}|{entity_id or ''}"


class ObservabilityRepository:
    def __init__(self, session: Session):
        self._session = session

    # -- alerts -------------------------------------------------------------

    def record_alert(
        self,
        *,
        code: str,
        severity: str,
        component: str,
        message: str,
        observed_at: datetime,
        entity_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AlertRow:
        """Append one detection and update the deduplicated projection.

        The projection is what an operator reads: a hundred rows for one stuck
        order is noise. The event log behind it keeps when each occurrence
        actually happened, which is what an incident reconstruction needs.
        """

        key = alert_dedup_key(code=code, component=component, entity_id=entity_id)
        # Locked so two detectors racing on the same problem produce one
        # projection row with a correct count, not two rows or a lost update.
        alert = self._session.execute(
            select(AlertRow).where(AlertRow.dedup_key == key).with_for_update()
        ).scalar_one_or_none()

        if alert is None:
            alert = AlertRow(
                alert_id=uuid4(),
                dedup_key=key,
                code=code,
                severity=severity,
                component=component,
                entity_id=entity_id,
                message=message,
                status=OPEN,
                opened_at=observed_at,
                last_seen_at=observed_at,
                occurrence_count=1,
                details=dict(details or {}),
            )
            self._session.add(alert)
        else:
            alert.occurrence_count += 1
            # A late-arriving observation must not make an active alert look
            # older than it is, so last_seen only ever moves forward.
            if observed_at > alert.last_seen_at:
                alert.last_seen_at = observed_at
                alert.message = message
                alert.severity = severity
        self._session.flush()

        self._session.add(
            AlertEventRow(
                alert_event_id=uuid4(),
                alert_id=alert.alert_id,
                dedup_key=key,
                code=code,
                severity=severity,
                component=component,
                entity_id=entity_id,
                message=message,
                observed_at=observed_at,
                details=dict(details or {}),
                created_at=observed_at,
            )
        )
        self._session.flush()
        return alert

    def alert_events(
        self, *, code: str, component: str = "execution.gateway",
        entity_id: str | None = None, limit: int = 200,
    ) -> tuple[AlertEventRow, ...]:
        key = alert_dedup_key(code=code, component=component, entity_id=entity_id)
        rows = self._session.execute(
            select(AlertEventRow)
            .where(AlertEventRow.dedup_key == key)
            .order_by(AlertEventRow.observed_at, AlertEventRow.created_at)
            .limit(limit)
        ).scalars()
        return tuple(rows)

    def open_critical_alert_count(self) -> int:
        return int(
            self._session.execute(
                select(func.count())
                .select_from(AlertRow)
                .where(AlertRow.status == OPEN, AlertRow.severity == CRITICAL)
            ).scalar_one()
        )

    # -- reconciliation -----------------------------------------------------

    def record_reconciliation_run(
        self,
        *,
        reconciliation_run_id: UUID,
        environment: str,
        account_alias: str,
        observed_at: datetime,
        broker_position_count: int,
        database_position_count: int,
        journal_event_count: int,
        details: dict[str, Any] | None = None,
        status: str | None = None,
    ) -> ReconciliationRunRow:
        """Record one three-way parity check.

        The status is derived from the counts unless a caller states it, so a
        mismatch cannot be recorded as MATCHED by omission.
        """

        derived = (
            "MATCHED"
            if broker_position_count == database_position_count
            else "DISCREPANCY"
        )
        row = ReconciliationRunRow(
            reconciliation_run_id=reconciliation_run_id,
            environment=environment,
            account_alias=account_alias,
            observed_at=observed_at,
            status=status or derived,
            broker_position_count=broker_position_count,
            database_position_count=database_position_count,
            journal_event_count=journal_event_count,
            details=dict(details or {}),
            created_at=observed_at,
        )
        self._session.add(row)
        self._session.flush()
        return row

    def append_reconciliation_item(
        self,
        *,
        reconciliation_run_id: UUID,
        broker_position_key: str,
        discrepancy_code: str,
        ownership_code: str | None = None,
        related_ids: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> ReconciliationItemRow:
        row = ReconciliationItemRow(
            reconciliation_item_id=uuid5(
                NAMESPACE_URL,
                f"{reconciliation_run_id}|{broker_position_key}|{discrepancy_code}",
            ),
            reconciliation_run_id=reconciliation_run_id,
            broker_position_key=broker_position_key,
            discrepancy_code=discrepancy_code,
            ownership_code=ownership_code,
            related_ids=dict(related_ids or {}),
            details=dict(details or {}),
            # Stamped with its run's observation time, never a clock: an item
            # belongs to the moment the parity check was taken.
            created_at=_run_observed_at(self._session, reconciliation_run_id),
        )
        self._session.add(row)
        self._session.flush()
        return row

    def reconciliation_items(
        self, reconciliation_run_id: UUID
    ) -> tuple[ReconciliationItemRow, ...]:
        rows = self._session.execute(
            select(ReconciliationItemRow)
            .where(ReconciliationItemRow.reconciliation_run_id == reconciliation_run_id)
            .order_by(ReconciliationItemRow.broker_position_key)
        ).scalars()
        return tuple(rows)

    def latest_reconciliation(
        self, *, environment: str, account_alias: str
    ) -> ReconciliationRunRow | None:
        """Indexed lookup, not a scan: health calls this on every check."""

        return self._session.execute(
            select(ReconciliationRunRow)
            .where(
                ReconciliationRunRow.environment == environment,
                ReconciliationRunRow.account_alias == account_alias,
            )
            .order_by(ReconciliationRunRow.observed_at.desc())
            .limit(1)
        ).scalars().first()


def _run_observed_at(session: Session, reconciliation_run_id: UUID) -> datetime:
    """An item is stamped with its run's observation time, never a clock."""

    run = session.get(ReconciliationRunRow, reconciliation_run_id)
    if run is None:
        raise ValueError("reconciliation item references an unknown run")
    return run.observed_at


__all__ = ["ObservabilityRepository", "alert_dedup_key"]
