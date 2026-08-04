"""Typed transactional-outbox enqueue, claim, and fencing operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Any
from uuid import UUID, uuid4

from pydantic_core import to_jsonable_python
from sqlalchemy import or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from core.nervous_system.persistence.models import (
    JobEvent as JobEventRow,
    JobRun as JobRunRow,
    OutboxEvent as OutboxEventRow,
)


class JobStatus(str, Enum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"
    RECOVERED = "RECOVERED"


@dataclass(frozen=True)
class JobRunRecord:
    job_run_id: UUID
    job_type: str
    scheduled_for: datetime
    config_hash: str
    host: str
    revision: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    heartbeat_at: datetime | None
    lease_owner: str | None
    lease_until: datetime | None
    lease_token: str | None
    attempt_no: int
    source_hashes: Mapping[str, Any]
    counts: Mapping[str, Any]
    output_ids: list[Any]
    error: str | None


def _job_record(row: JobRunRow) -> JobRunRecord:
    return JobRunRecord(
        job_run_id=row.job_run_id,
        job_type=row.job_type,
        scheduled_for=row.scheduled_for,
        config_hash=row.config_hash,
        host=row.host,
        revision=row.revision,
        status=row.status,
        started_at=row.started_at,
        finished_at=row.finished_at,
        heartbeat_at=row.heartbeat_at,
        lease_owner=row.lease_owner,
        lease_until=row.lease_until,
        lease_token=row.lease_token,
        attempt_no=row.attempt_no or 0,
        source_hashes=dict(row.source_hashes or {}),
        counts=dict(row.counts or {}),
        output_ids=list(row.output_ids or []),
        error=row.error,
    )


@dataclass(frozen=True)
class OutboxEventRecord:
    outbox_event_id: UUID
    event_type: str
    aggregate_type: str
    aggregate_id: str
    payload: Mapping[str, Any]
    created_at: datetime
    available_at: datetime
    event_hash: str
    claimed_by: str | None = None
    claimed_until: datetime | None = None
    claim_token: str | None = None
    delivered_at: datetime | None = None
    delivery_attempts: int = 0
    last_error: str | None = None


def _aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be timezone-aware UTC")


def _event_hash(
    event_type: str,
    aggregate_type: str,
    aggregate_id: str,
    payload: Mapping[str, Any],
) -> str:
    normalized = to_jsonable_python(dict(payload))
    encoded = json.dumps(
        {
            "aggregate_id": aggregate_id,
            "aggregate_type": aggregate_type,
            "event_type": event_type,
            "payload": normalized,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _event_id(event_hash: str) -> UUID:
    return UUID(hex=event_hash[:32])


def _record(row: OutboxEventRow) -> OutboxEventRecord:
    expected_hash = _event_hash(
        row.event_type,
        row.aggregate_type,
        row.aggregate_id,
        row.payload,
    )
    if row.event_hash != expected_hash or row.outbox_event_id != _event_id(expected_hash):
        raise ValueError("outbox row failed deterministic hash validation")
    return OutboxEventRecord(
        outbox_event_id=row.outbox_event_id,
        event_type=row.event_type,
        aggregate_type=row.aggregate_type,
        aggregate_id=row.aggregate_id,
        payload=row.payload,
        created_at=row.created_at,
        available_at=row.available_at,
        event_hash=row.event_hash,
        claimed_by=row.claimed_by,
        claimed_until=row.claimed_until,
        claim_token=row.claim_token,
        delivered_at=row.delivered_at,
        delivery_attempts=row.delivery_attempts,
        last_error=row.last_error,
    )


class OperationsRepository:
    def __init__(self, session: Session):
        self._session = session

    def enqueue(
        self,
        *,
        event_type: str,
        aggregate_type: str,
        aggregate_id: str | UUID,
        payload: Mapping[str, Any],
        created_at: datetime | None = None,
        available_at: datetime | None = None,
    ) -> OutboxEventRecord:
        aggregate_id = str(aggregate_id)
        event_hash = _event_hash(event_type, aggregate_type, aggregate_id, payload)
        event_id = _event_id(event_hash)
        created = created_at or datetime.now(timezone.utc)
        available = available_at or created
        _aware(created, "created_at")
        _aware(available, "available_at")
        values = {
            "outbox_event_id": event_id,
            "event_type": event_type,
            "aggregate_type": aggregate_type,
            "aggregate_id": aggregate_id,
            "payload": dict(payload),
            "created_at": created,
            "available_at": available,
            "delivery_attempts": 0,
            "event_hash": event_hash,
        }
        self._session.execute(
            pg_insert(OutboxEventRow.__table__)
            .values(**values)
            .on_conflict_do_nothing()
        )
        row = self._session.get(OutboxEventRow, event_id)
        if row is None:
            row = self._session.execute(
                select(OutboxEventRow).where(OutboxEventRow.event_hash == event_hash)
            ).scalars().first()
        if row is None:
            raise RuntimeError("outbox enqueue did not produce or find its deterministic row")
        if row.event_hash != event_hash:
            raise ValueError("outbox conflict contains a non-identical payload")
        return _record(row)

    def claim(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_seconds: int,
        limit: int = 1,
    ) -> tuple[OutboxEventRecord, ...]:
        _aware(now, "now")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if limit <= 0:
            raise ValueError("limit must be positive")
        stmt = (
            select(OutboxEventRow)
            .where(
                OutboxEventRow.delivered_at.is_(None),
                OutboxEventRow.available_at <= now,
                or_(
                    OutboxEventRow.claimed_until.is_(None),
                    OutboxEventRow.claimed_until < now,
                ),
            )
            .order_by(
                OutboxEventRow.available_at.asc(),
                OutboxEventRow.created_at.asc(),
                OutboxEventRow.outbox_event_id.asc(),
            )
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        rows = self._session.execute(stmt).scalars().all()
        claimed_until = now + timedelta(seconds=lease_seconds)
        records: list[OutboxEventRecord] = []
        for row in rows:
            row.claimed_by = worker_id
            row.claimed_until = claimed_until
            row.claim_token = uuid4().hex
            row.delivery_attempts = (row.delivery_attempts or 0) + 1
            records.append(_record(row))
        self._session.flush()
        return tuple(records)

    def claim_pending(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_seconds: int,
        limit: int = 1,
    ) -> tuple[OutboxEventRecord, ...]:
        return self.claim(
            worker_id=worker_id,
            now=now,
            lease_seconds=lease_seconds,
            limit=limit,
        )

    # -- job runs -----------------------------------------------------------

    def claim_job(
        self,
        *,
        job_type: str,
        scheduled_for: datetime,
        config_hash: str,
        owner: str,
        lease_token: str,
        now: datetime,
        lease_seconds: int,
        host: str,
        revision: str,
    ) -> tuple[JobRunRecord, bool]:
        """Take the lease for one scheduled slot, or report who holds it.

        ``(job_type, scheduled_for, config_hash)`` is unique, so the same slot
        can never run twice concurrently. A file lock cannot give this
        guarantee across hosts; the database can.
        """

        _aware(scheduled_for, "scheduled_for")
        _aware(now, "now")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        lease_until = now + timedelta(seconds=lease_seconds)

        existing = self._session.execute(
            select(JobRunRow).where(
                JobRunRow.job_type == job_type,
                JobRunRow.scheduled_for == scheduled_for,
                JobRunRow.config_hash == config_hash,
            )
        ).scalars().first()

        if existing is None:
            row = JobRunRow(
                job_run_id=uuid4(),
                job_type=job_type,
                scheduled_for=scheduled_for,
                config_hash=config_hash,
                host=host,
                revision=revision,
                status=JobStatus.RUNNING.value,
                started_at=now,
                heartbeat_at=now,
                lease_owner=owner,
                lease_until=lease_until,
                lease_token=lease_token,
                attempt_no=1,
            )
            self._session.add(row)
            self._session.flush()
            self._append_job_event(row.job_run_id, JobStatus.RUNNING.value, now, {"claim": "new"})
            return _job_record(row), True

        if existing.status in {JobStatus.SUCCEEDED.value}:
            return _job_record(existing), False

        lease_live = (
            existing.lease_until is not None
            and existing.lease_until > now
            and existing.lease_owner != owner
        )
        if lease_live:
            return _job_record(existing), False

        if existing.lease_until is not None and existing.lease_until <= now:
            # A previous holder died. Record the takeover rather than silently
            # overwriting the row.
            self._append_job_event(
                existing.job_run_id,
                JobStatus.RECOVERED.value,
                now,
                {
                    "previous_owner": existing.lease_owner,
                    "expired_at": existing.lease_until.isoformat(),
                },
            )
        existing.status = JobStatus.RUNNING.value
        existing.host = host
        existing.revision = revision
        existing.lease_owner = owner
        existing.lease_until = lease_until
        existing.lease_token = lease_token
        existing.heartbeat_at = now
        existing.attempt_no = (existing.attempt_no or 0) + 1
        self._session.flush()
        return _job_record(existing), True

    def heartbeat_job(
        self,
        job_run_id: UUID,
        *,
        lease_token: str,
        now: datetime,
        lease_seconds: int,
    ) -> bool:
        """Extend a lease only while this worker still owns it."""

        _aware(now, "now")
        result = self._session.execute(
            update(JobRunRow)
            .where(
                JobRunRow.job_run_id == job_run_id,
                JobRunRow.lease_token == lease_token,
                JobRunRow.status == JobStatus.RUNNING.value,
            )
            .values(
                heartbeat_at=now,
                lease_until=now + timedelta(seconds=lease_seconds),
            )
        )
        self._session.flush()
        return result.rowcount == 1

    def finish_job(
        self,
        job_run_id: UUID,
        *,
        lease_token: str,
        status: str,
        finished_at: datetime,
        source_hashes: Mapping[str, Any] | None = None,
        counts: Mapping[str, Any] | None = None,
        output_ids: list[Any] | None = None,
        exception_summary: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        _aware(finished_at, "finished_at")
        values: dict[str, Any] = {
            "status": status,
            "finished_at": finished_at,
            "source_hashes": dict(source_hashes or {}),
            "counts": dict(counts or {}),
            "output_ids": list(output_ids or []),
        }
        if exception_summary is not None:
            values["exception_summary"] = dict(exception_summary)
        if error is not None:
            values["error"] = error
            values["last_error"] = error
        result = self._session.execute(
            update(JobRunRow)
            .where(
                JobRunRow.job_run_id == job_run_id,
                JobRunRow.lease_token == lease_token,
            )
            .values(**values)
        )
        self._session.flush()
        if result.rowcount == 1:
            self._append_job_event(job_run_id, status, finished_at, {})
        return result.rowcount == 1

    def get_job(self, job_run_id: UUID) -> JobRunRecord | None:
        row = self._session.get(JobRunRow, job_run_id)
        return _job_record(row) if row is not None else None

    def job_events(self, job_run_id: UUID) -> tuple[tuple[str, datetime], ...]:
        rows = self._session.execute(
            select(JobEventRow)
            .where(JobEventRow.job_run_id == job_run_id)
            .order_by(JobEventRow.observed_at.asc())
        ).scalars().all()
        return tuple((row.status, row.observed_at) for row in rows)

    def _append_job_event(
        self,
        job_run_id: UUID,
        status: str,
        observed_at: datetime,
        payload: Mapping[str, Any],
    ) -> None:
        self._session.add(
            JobEventRow(
                job_event_id=uuid4(),
                job_run_id=job_run_id,
                status=status,
                observed_at=observed_at,
                payload=dict(payload),
            )
        )
        self._session.flush()

    def renew_claim(
        self,
        outbox_event_id: UUID,
        *,
        worker_id: str,
        claim_token: str,
        now: datetime,
        lease_seconds: int,
    ) -> bool:
        _aware(now, "now")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        result = self._session.execute(
            update(OutboxEventRow)
            .where(
                OutboxEventRow.outbox_event_id == outbox_event_id,
                OutboxEventRow.claimed_by == worker_id,
                OutboxEventRow.claim_token == claim_token,
                OutboxEventRow.delivered_at.is_(None),
                OutboxEventRow.claimed_until > now,
            )
            .values(claimed_until=now + timedelta(seconds=lease_seconds))
        )
        self._session.flush()
        return result.rowcount == 1

    renew_lease = renew_claim

    def mark_delivered(
        self,
        outbox_event_id: UUID,
        *,
        worker_id: str,
        claim_token: str,
        delivered_at: datetime,
    ) -> bool:
        _aware(delivered_at, "delivered_at")
        result = self._session.execute(
            update(OutboxEventRow)
            .where(
                OutboxEventRow.outbox_event_id == outbox_event_id,
                OutboxEventRow.claimed_by == worker_id,
                OutboxEventRow.claim_token == claim_token,
                OutboxEventRow.delivered_at.is_(None),
            )
            .values(delivered_at=delivered_at)
        )
        self._session.flush()
        return result.rowcount == 1

    def mark_failed(
        self,
        outbox_event_id: UUID,
        *,
        worker_id: str,
        claim_token: str,
        error: str,
        retry_at: datetime | None = None,
    ) -> bool:
        available = retry_at or datetime.now(timezone.utc)
        _aware(available, "retry_at")
        result = self._session.execute(
            update(OutboxEventRow)
            .where(
                OutboxEventRow.outbox_event_id == outbox_event_id,
                OutboxEventRow.claimed_by == worker_id,
                OutboxEventRow.claim_token == claim_token,
                OutboxEventRow.delivered_at.is_(None),
            )
            .values(
                available_at=available,
                claimed_by=None,
                claimed_until=None,
                claim_token=None,
                last_error=error,
            )
        )
        self._session.flush()
        return result.rowcount == 1


__all__ = ["OperationsRepository", "OutboxEventRecord"]
