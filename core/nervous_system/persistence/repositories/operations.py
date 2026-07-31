"""Typed transactional-outbox enqueue, claim, and fencing operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Any
from uuid import UUID, uuid4

from pydantic_core import to_jsonable_python
from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from core.nervous_system.persistence.models import OutboxEvent as OutboxEventRow


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
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


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
        existing = self._session.get(OutboxEventRow, event_id)
        if existing is not None:
            expected = _event_hash(
                existing.event_type,
                existing.aggregate_type,
                existing.aggregate_id,
                existing.payload,
            )
            if existing.event_hash != event_hash or expected != existing.event_hash:
                raise ValueError("existing outbox row failed deterministic hash validation")
            return _record(existing)

        created = created_at or datetime.now(timezone.utc)
        available = available_at or created
        _aware(created, "created_at")
        _aware(available, "available_at")
        row = OutboxEventRow(
            outbox_event_id=event_id,
            event_type=event_type,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            payload=dict(payload),
            created_at=created,
            available_at=available,
            delivery_attempts=0,
            event_hash=event_hash,
        )
        self._session.add(row)
        self._session.flush()
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
