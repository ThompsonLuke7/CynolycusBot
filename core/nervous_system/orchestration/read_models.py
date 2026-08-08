"""Repository-backed projections for the read-only audit surface.

Lists read indexed columns only. A decision record's payload is the whole
immutable chain, and deserializing it for every row of a list makes the cost of
a page grow with the size of history — the one quantity that only ever
increases. Detail is the opposite case: one row, where the full graph is the
point.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from core.nervous_system.orchestration.http import HealthReport
from core.nervous_system.persistence.models import (
    Alert as AlertRow,
    DecisionRecord as DecisionRecordRow,
    JobRun as JobRunRow,
    ReconciliationRun as ReconciliationRunRow,
)
from core.nervous_system.persistence.repositories.observability import (
    ObservabilityRepository,
)


def _utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    aware = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).isoformat()


class AuditStore:
    """The concrete store the audit router reads through."""

    def __init__(
        self,
        session: Session,
        *,
        clock: Callable[[], datetime] | None = None,
        journal_probe: Callable[[], bool] | None = None,
    ) -> None:
        self._session = session
        # Injected so the read path carries no hidden time of its own.
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._journal_probe = journal_probe or (lambda: True)

    # -- lists --------------------------------------------------------------

    def decisions(
        self, *, limit: int, strategy_id: str | None = None, **_: Any
    ) -> list[dict[str, Any]]:
        # Explicit columns, never the ORM entity: selecting the entity would
        # pull the payload JSON for every row.
        query = select(
            DecisionRecordRow.decision_record_id,
            DecisionRecordRow.decision_time,
            DecisionRecordRow.status,
            DecisionRecordRow.failure_stage,
            DecisionRecordRow.snapshot_id,
            DecisionRecordRow.intent_id,
        ).order_by(DecisionRecordRow.decision_time.desc()).limit(limit)
        return [
            {
                "decision_record_id": str(row.decision_record_id),
                "decision_time": _utc(row.decision_time),
                "status": row.status,
                "failure_stage": row.failure_stage,
                "snapshot_id": None if row.snapshot_id is None else str(row.snapshot_id),
                "intent_id": None if row.intent_id is None else str(row.intent_id),
            }
            for row in self._session.execute(query)
        ]

    def alerts(self, *, limit: int, **_: Any) -> list[dict[str, Any]]:
        query = select(
            AlertRow.alert_id,
            AlertRow.code,
            AlertRow.severity,
            AlertRow.component,
            AlertRow.entity_id,
            AlertRow.status,
            AlertRow.occurrence_count,
            AlertRow.opened_at,
            AlertRow.last_seen_at,
        ).order_by(AlertRow.last_seen_at.desc()).limit(limit)
        return [
            {
                "alert_id": str(row.alert_id),
                "code": row.code,
                "severity": row.severity,
                "component": row.component,
                "entity_id": row.entity_id,
                "status": row.status,
                "occurrence_count": row.occurrence_count,
                "opened_at": _utc(row.opened_at),
                "last_seen_at": _utc(row.last_seen_at),
            }
            for row in self._session.execute(query)
        ]

    def reconciliations(self, *, limit: int, **_: Any) -> list[dict[str, Any]]:
        query = select(
            ReconciliationRunRow.reconciliation_run_id,
            ReconciliationRunRow.environment,
            ReconciliationRunRow.account_alias,
            ReconciliationRunRow.observed_at,
            ReconciliationRunRow.status,
            ReconciliationRunRow.broker_position_count,
            ReconciliationRunRow.database_position_count,
            ReconciliationRunRow.journal_event_count,
        ).order_by(ReconciliationRunRow.observed_at.desc()).limit(limit)
        return [
            {
                "reconciliation_run_id": str(row.reconciliation_run_id),
                "environment": row.environment,
                "account_alias": row.account_alias,
                "observed_at": _utc(row.observed_at),
                "status": row.status,
                "broker_position_count": row.broker_position_count,
                "database_position_count": row.database_position_count,
                "journal_event_count": row.journal_event_count,
            }
            for row in self._session.execute(query)
        ]

    # -- detail -------------------------------------------------------------

    def decision(self, decision_record_id: str) -> dict[str, Any] | None:
        """One decision with its full chain, or nothing.

        A malformed identifier is 'not found', not an error: the router turns
        None into a 404, whereas an exception would become a 503 and report a
        healthy system as broken.
        """

        try:
            identifier = UUID(str(decision_record_id))
        except (TypeError, ValueError):
            return None
        row = self._session.get(DecisionRecordRow, identifier)
        if row is None:
            return None
        return {
            "decision_record_id": str(row.decision_record_id),
            "decision_time": _utc(row.decision_time),
            "status": row.status,
            "failure_stage": row.failure_stage,
            "failure_reason": row.failure_reason,
            "content_hash": row.content_hash,
            "snapshot_id": None if row.snapshot_id is None else str(row.snapshot_id),
            "intent_id": None if row.intent_id is None else str(row.intent_id),
            "policy_decision_id": (
                None if row.policy_decision_id is None else str(row.policy_decision_id)
            ),
            "payload": row.payload,
        }

    # -- health -------------------------------------------------------------

    def health(self) -> HealthReport:
        revision = self._session.execute(
            text("select version_num from public.alembic_version")
        ).scalar()
        heartbeat = self._session.execute(
            select(func.max(JobRunRow.heartbeat_at))
        ).scalar()
        reconciliation = self._session.execute(
            select(func.max(ReconciliationRunRow.observed_at))
        ).scalar()
        return HealthReport(
            schema_revision=str(revision) if revision else "unknown",
            database_ok=True,
            journal_ok=bool(self._journal_probe()),
            latest_job_heartbeat=heartbeat,
            latest_reconciliation=reconciliation,
            open_critical_alerts=ObservabilityRepository(
                self._session
            ).open_critical_alert_count(),
            stale_states=(),
            checked_at=self._clock(),
        )


__all__ = ["AuditStore"]
