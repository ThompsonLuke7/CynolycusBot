"""Typed source, import, quarantine, and lineage registry operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.nervous_system.persistence.models import (
    ConfigSnapshot as ConfigSnapshotRow,
    ImportItem as ImportItemRow,
    ImportQuarantine as ImportQuarantineRow,
    ImportRun as ImportRunRow,
    LineageEdge as LineageEdgeRow,
    SourceArtifact as SourceArtifactRow,
)


@dataclass(frozen=True)
class SourceArtifactRecord:
    uri: str
    sha256: str
    byte_size: int
    source_kind: str
    discovered_at: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True)
class ImportRunRecord:
    importer_version: str
    started_at: datetime
    finished_at: datetime | None
    status: str
    counts: Mapping[str, Any]
    import_run_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True)
class ImportItemRecord:
    import_run_id: UUID
    source_id: UUID
    importer_version: str
    record_locator: str
    normalized_hash: str
    target_type: str
    target_id: str | None
    status: str
    warnings: Mapping[str, Any]
    import_item_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True)
class ImportQuarantineRecord:
    import_run_id: UUID
    source_id: UUID
    record_locator: str
    raw_payload: Mapping[str, Any] | None
    raw_text: str | None
    error_code: str
    error_message: str
    created_at: datetime
    quarantine_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True)
class LineageEdgeRecord:
    source_id: UUID
    target_type: str
    target_id: str
    relationship: str
    created_at: datetime
    lineage_edge_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True)
class ConfigSnapshotRecord:
    config_version: str
    content_hash: str
    payload: Mapping[str, Any]
    created_at: datetime
    config_snapshot_id: UUID = field(default_factory=uuid4)


def _validate_hash(value: str, field_name: str) -> None:
    if re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{field_name} must be a 64-character SHA-256 hex string")


def _validate_time(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _one_or_none(result: Any) -> Any:
    scalars = result.scalars()
    if hasattr(scalars, "one_or_none"):
        return scalars.one_or_none()
    return scalars.first()


class RegistryRepository:
    def __init__(self, session: Session):
        self._session = session

    def save_source_artifact(self, artifact: SourceArtifactRecord) -> SourceArtifactRecord:
        _validate_hash(artifact.sha256, "sha256")
        if artifact.byte_size < 0:
            raise ValueError("source artifact byte_size must be nonnegative")
        _validate_time(artifact.discovered_at, "discovered_at")
        self._session.add(
            SourceArtifactRow(
                source_id=artifact.source_id,
                uri=artifact.uri,
                sha256=artifact.sha256,
                byte_size=artifact.byte_size,
                source_kind=artifact.source_kind,
                discovered_at=artifact.discovered_at,
                metadata_json=dict(artifact.metadata),
            )
        )
        self._session.flush()
        return artifact

    register_source_artifact = save_source_artifact

    def get_source_artifact(self, uri: str, sha256: str) -> SourceArtifactRecord | None:
        _validate_hash(sha256, "sha256")
        row = _one_or_none(
            self._session.execute(
                select(SourceArtifactRow).where(
                    SourceArtifactRow.uri == uri,
                    SourceArtifactRow.sha256 == sha256,
                )
            )
        )
        if row is None:
            return None
        return SourceArtifactRecord(
            source_id=row.source_id,
            uri=row.uri,
            sha256=row.sha256,
            byte_size=row.byte_size,
            source_kind=row.source_kind,
            discovered_at=row.discovered_at,
            metadata=row.metadata_json,
        )

    def save_import_run(self, run: ImportRunRecord) -> ImportRunRecord:
        _validate_time(run.started_at, "started_at")
        if run.finished_at is not None:
            _validate_time(run.finished_at, "finished_at")
            if run.finished_at < run.started_at:
                raise ValueError("finished_at must not precede started_at")
        self._session.add(
            ImportRunRow(
                import_run_id=run.import_run_id,
                importer_version=run.importer_version,
                started_at=run.started_at,
                finished_at=run.finished_at,
                status=run.status,
                counts=dict(run.counts),
            )
        )
        self._session.flush()
        return run

    def save_import_item(self, item: ImportItemRecord) -> ImportItemRecord:
        _validate_hash(item.normalized_hash, "normalized_hash")
        self._session.add(
            ImportItemRow(
                import_item_id=item.import_item_id,
                import_run_id=item.import_run_id,
                source_id=item.source_id,
                importer_version=item.importer_version,
                record_locator=item.record_locator,
                normalized_hash=item.normalized_hash,
                target_type=item.target_type,
                target_id=item.target_id,
                status=item.status,
                warnings=dict(item.warnings),
            )
        )
        self._session.flush()
        return item

    def save_import_quarantine(
        self, quarantine: ImportQuarantineRecord
    ) -> ImportQuarantineRecord:
        _validate_time(quarantine.created_at, "created_at")
        self._session.add(
            ImportQuarantineRow(
                quarantine_id=quarantine.quarantine_id,
                import_run_id=quarantine.import_run_id,
                source_id=quarantine.source_id,
                record_locator=quarantine.record_locator,
                # Do not normalize or derive raw evidence.  The importer owns
                # the exact bytes/text represented by these values.
                raw_payload=None if quarantine.raw_payload is None else dict(quarantine.raw_payload),
                raw_text=quarantine.raw_text,
                error_code=quarantine.error_code,
                error_message=quarantine.error_message,
                created_at=quarantine.created_at,
            )
        )
        self._session.flush()
        return quarantine

    def save_lineage_edge(self, edge: LineageEdgeRecord) -> LineageEdgeRecord:
        _validate_time(edge.created_at, "created_at")
        self._session.add(
            LineageEdgeRow(
                lineage_edge_id=edge.lineage_edge_id,
                source_id=edge.source_id,
                target_type=edge.target_type,
                target_id=edge.target_id,
                relationship=edge.relationship,
                created_at=edge.created_at,
            )
        )
        self._session.flush()
        return edge

    def save_config_snapshot(self, snapshot: ConfigSnapshotRecord) -> ConfigSnapshotRecord:
        _validate_hash(snapshot.content_hash, "content_hash")
        _validate_time(snapshot.created_at, "created_at")
        self._session.add(
            ConfigSnapshotRow(
                config_snapshot_id=snapshot.config_snapshot_id,
                config_version=snapshot.config_version,
                content_hash=snapshot.content_hash,
                payload=dict(snapshot.payload),
                created_at=snapshot.created_at,
            )
        )
        self._session.flush()
        return snapshot

    def save_quarantine(self, quarantine: ImportQuarantineRecord) -> ImportQuarantineRecord:
        return self.save_import_quarantine(quarantine)

    def save_lineage(self, edge: LineageEdgeRecord) -> LineageEdgeRecord:
        return self.save_lineage_edge(edge)


__all__ = [
    "ConfigSnapshotRecord",
    "ImportItemRecord",
    "ImportQuarantineRecord",
    "ImportRunRecord",
    "LineageEdgeRecord",
    "RegistryRepository",
    "SourceArtifactRecord",
]
