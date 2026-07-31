from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from threading import Barrier
from uuid import UUID, uuid4

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from core.nervous_system.persistence.repositories.operations import OperationsRepository
from core.nervous_system.persistence.models import OutboxEvent
from core.nervous_system.persistence.repositories.registry import (
    ConfigSnapshotRecord,
    ImportItemRecord,
    ImportQuarantineRecord,
    ImportRunRecord,
    LineageEdgeRecord,
    RegistryRepository,
    SourceArtifactRecord,
)


NOW = datetime(2026, 7, 30, 18, 20, tzinfo=timezone.utc)


class _Result:
    def __init__(self, rows=()):
        self._rows = list(rows)

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None


class _Session:
    def __init__(self):
        self.statements = []
        self.added = []
        self.rows = {}
        self.flushes = 0
        self.commits = 0

    def get(self, _model, identifier):
        return self.rows.get(identifier)

    def add(self, value):
        self.added.append(value)

    def flush(self):
        self.flushes += 1

    def execute(self, statement):
        self.statements.append(statement)
        if statement.is_insert:
            values = statement.compile(dialect=postgresql.dialect()).params
            row = OutboxEvent(**values)
            if row.outbox_event_id not in self.rows and not any(
                existing.event_hash == row.event_hash for existing in self.rows.values()
            ):
                self.rows[row.outbox_event_id] = row
        return _Result()


def test_enqueue_hash_and_id_are_deterministic_without_committing():
    first_session = _Session()
    first = OperationsRepository(first_session).enqueue(
        event_type="DecisionRecordCreated",
        aggregate_type="decision_record",
        aggregate_id=UUID("00000000-0000-0000-0000-000000000001"),
        payload={"b": 2, "a": 1},
        created_at=NOW,
        available_at=NOW,
    )
    second_session = _Session()
    second = OperationsRepository(second_session).enqueue(
        event_type="DecisionRecordCreated",
        aggregate_type="decision_record",
        aggregate_id=UUID("00000000-0000-0000-0000-000000000001"),
        payload={"a": 1, "b": 2},
        created_at=NOW + timedelta(hours=3),
        available_at=NOW + timedelta(hours=3),
    )
    assert first.event_hash == second.event_hash
    assert first.outbox_event_id == second.outbox_event_id
    assert isinstance(first.outbox_event_id, UUID)
    assert first_session.commits == 0
    assert second_session.commits == 0


def test_enqueue_rejects_naive_available_at():
    with pytest.raises(ValueError, match="timezone-aware"):
        OperationsRepository(_Session()).enqueue(
            event_type="UTCValidation",
            aggregate_type="test",
            aggregate_id="task7-utc-validation",
            payload={"value": 1},
            created_at=NOW,
            available_at=datetime(2026, 7, 30, 18, 20),
        )


def test_enqueue_rejects_non_utc_available_at():
    with pytest.raises(ValueError, match="UTC"):
        OperationsRepository(_Session()).enqueue(
            event_type="UTCValidationOffset",
            aggregate_type="test",
            aggregate_id="task7-utc-validation-offset",
            payload={"value": 1},
            created_at=NOW,
            available_at=datetime(
                2026, 7, 30, 18, 20, tzinfo=timezone(timedelta(hours=-4))
            ),
        )


def _canonical_config_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def test_config_snapshot_accepts_hash_matching_canonical_payload():
    payload = {"nested": {"b": 2, "a": 1}, "enabled": True}
    snapshot = ConfigSnapshotRecord(
        config_version="task7-test@1",
        content_hash=_canonical_config_hash(payload),
        payload=payload,
        created_at=NOW,
    )

    saved = RegistryRepository(_Session()).save_config_snapshot(snapshot)

    assert saved == snapshot


def test_config_snapshot_rejects_hash_mismatch_before_write():
    session = _Session()
    snapshot = ConfigSnapshotRecord(
        config_version="task7-test@1",
        content_hash="a" * 64,
        payload={"enabled": True},
        created_at=NOW,
    )

    with pytest.raises(ValueError, match="content_hash"):
        RegistryRepository(session).save_config_snapshot(snapshot)
    assert session.added == []


def test_claim_statement_uses_postgresql_skip_locked_and_deterministic_order():
    session = _Session()
    OperationsRepository(session).claim(
        worker_id="worker-a",
        now=NOW,
        lease_seconds=30,
        limit=2,
    )
    assert len(session.statements) == 1
    sql = str(session.statements[0].compile(dialect=postgresql.dialect()))
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "available_at" in sql
    assert "claimed_until" in sql
    assert "delivered_at" in sql
    assert "outbox_event_id" in sql


def test_registry_records_are_typed_and_preserve_raw_quarantine_evidence_offline():
    session = _Session()
    repo = RegistryRepository(session)
    source = repo.save_source_artifact(
        SourceArtifactRecord(
            source_id=UUID("00000000-0000-0000-0000-000000000001"),
            uri="Data/inference/raw.jsonl",
            sha256="a" * 64,
            byte_size=12,
            source_kind="test",
            discovered_at=NOW,
            metadata={"encoding": "utf-8"},
        )
    )
    run = repo.save_import_run(
        ImportRunRecord(
            import_run_id=UUID("00000000-0000-0000-0000-000000000002"),
            importer_version="task-7@1",
            started_at=NOW,
            finished_at=None,
            status="RUNNING",
            counts={},
        )
    )
    item = repo.save_import_item(
        ImportItemRecord(
            import_item_id=UUID("00000000-0000-0000-0000-000000000003"),
            import_run_id=run.import_run_id,
            source_id=source.source_id,
            importer_version="task-7@1",
            record_locator="line:1",
            normalized_hash="b" * 64,
            target_type="raw",
            target_id=None,
            status="QUARANTINED",
            warnings={"warning": "kept"},
        )
    )
    quarantine = repo.save_import_quarantine(
        ImportQuarantineRecord(
            quarantine_id=UUID("00000000-0000-0000-0000-000000000004"),
            import_run_id=run.import_run_id,
            source_id=source.source_id,
            record_locator=item.record_locator,
            raw_payload={"not": "rewritten"},
            raw_text='{"not": "rewritten"}\n',
            error_code="BAD_ROW",
            error_message="preserve this exact evidence",
            created_at=NOW,
        )
    )
    edge = repo.save_lineage_edge(
        LineageEdgeRecord(
            lineage_edge_id=UUID("00000000-0000-0000-0000-000000000005"),
            source_id=source.source_id,
            target_type="raw",
            target_id="target-1",
            relationship="IMPORTED_AS",
            created_at=NOW,
        )
    )
    assert quarantine.raw_text == '{"not": "rewritten"}\n'
    assert edge.source_id == source.source_id
    assert len(session.added) == 5
    assert session.commits == 0


@pytest.mark.postgres
def test_concurrent_identical_enqueue_is_atomic_and_returns_equivalent_records(session_factory):
    barrier = Barrier(2)
    first_session = session_factory()
    second_session = session_factory()
    tracked_sessions = {first_session, second_session}
    flushed_sessions: set[Session] = set()

    def synchronize_first_flush(session, _flush_context, _instances):
        if session in tracked_sessions and session not in flushed_sessions:
            flushed_sessions.add(session)
            barrier.wait(timeout=10)

    event.listen(Session, "before_flush", synchronize_first_flush)
    event_payload = {
        "event_type": "ConcurrentEnqueue",
        "aggregate_type": "test",
        "aggregate_id": f"task7-concurrent-identical-{uuid4().hex}",
        "payload": {"value": 1},
        "created_at": NOW,
        "available_at": NOW + timedelta(minutes=1),
    }

    def enqueue_once(session):
        try:
            record = OperationsRepository(session).enqueue(**event_payload)
            session.commit()
            return record
        finally:
            session.close()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            records = tuple(executor.map(enqueue_once, (first_session, second_session)))
    finally:
        event.remove(Session, "before_flush", synchronize_first_flush)

    assert records[0].outbox_event_id == records[1].outbox_event_id
    assert records[0].event_hash == records[1].event_hash
    with session_factory() as session:
        assert session.scalar(
            select(func.count()).where(
                OutboxEvent.event_hash == records[0].event_hash
            )
        ) == 1


@pytest.mark.postgres
def test_enqueue_is_idempotent_and_future_events_are_not_claimed(
    session_factory,
):
    run_key = uuid4().hex
    active_id = f"task7-{run_key}-active"
    future_id = f"task7-{run_key}-future"
    with session_factory() as session:
        repo = OperationsRepository(session)
        active = repo.enqueue(
            event_type="TestEvent",
            aggregate_type="test",
            aggregate_id=active_id,
            payload={"value": 1},
            created_at=NOW,
            available_at=NOW,
        )
        duplicate = repo.enqueue(
            event_type="TestEvent",
            aggregate_type="test",
            aggregate_id=active_id,
            payload={"value": 1},
            created_at=NOW + timedelta(hours=1),
            available_at=NOW + timedelta(hours=1),
        )
        future = repo.enqueue(
            event_type="TestEvent",
            aggregate_type="test",
            aggregate_id=future_id,
            payload={"value": 2},
            created_at=NOW,
            available_at=NOW + timedelta(minutes=5),
        )
        session.commit()

    assert duplicate.outbox_event_id == active.outbox_event_id
    with session_factory() as session:
        claimed = OperationsRepository(session).claim(
            worker_id="worker-a", now=NOW, lease_seconds=30, limit=10
        )
        session.commit()
    assert [item.aggregate_id for item in claimed] == [active_id]
    assert future.outbox_event_id not in {item.outbox_event_id for item in claimed}


@pytest.mark.postgres
def test_claims_are_disjoint_and_finalization_is_fenced(session_factory):
    run_key = uuid4().hex
    with session_factory() as session:
        repo = OperationsRepository(session)
        for index in range(3):
            repo.enqueue(
                event_type="ClaimEvent",
                aggregate_type="test",
                aggregate_id=f"task7-{run_key}-claim-{index}",
                payload={"index": index},
                created_at=NOW,
                available_at=NOW,
            )
        session.commit()

    with session_factory() as first_session:
        first_claim = OperationsRepository(first_session).claim(
            worker_id="worker-a", now=NOW, lease_seconds=30, limit=2
        )
        first_session.commit()
    with session_factory() as second_session:
        second_claim = OperationsRepository(second_session).claim(
            worker_id="worker-b", now=NOW, lease_seconds=30, limit=2
        )
        second_session.commit()

    first_ids = {item.outbox_event_id for item in first_claim}
    second_ids = {item.outbox_event_id for item in second_claim}
    assert len(first_ids) == 2
    assert len(second_ids) == 1
    assert first_ids.isdisjoint(second_ids)

    claim = first_claim[0]
    with session_factory() as session:
        repo = OperationsRepository(session)
        assert repo.renew_claim(
            claim.outbox_event_id,
            worker_id="worker-b",
            claim_token=claim.claim_token,
            now=NOW,
            lease_seconds=30,
        ) is False
        assert repo.mark_delivered(
            claim.outbox_event_id,
            worker_id="worker-b",
            claim_token=claim.claim_token,
            delivered_at=NOW,
        ) is False
        assert repo.renew_claim(
            claim.outbox_event_id,
            worker_id="worker-a",
            claim_token=claim.claim_token,
            now=NOW,
            lease_seconds=30,
        ) is True
        session.commit()

    with session_factory() as session:
        repo = OperationsRepository(session)
        assert repo.mark_failed(
            claim.outbox_event_id,
            worker_id="worker-a",
            claim_token=claim.claim_token,
            error="retryable",
            retry_at=NOW + timedelta(minutes=1),
        ) is True
        session.commit()

    with session_factory() as session:
        retry = OperationsRepository(session).claim(
            worker_id="worker-c",
            now=NOW + timedelta(minutes=1),
            lease_seconds=30,
            limit=10,
        )
        session.commit()
    assert claim.outbox_event_id in {item.outbox_event_id for item in retry}
