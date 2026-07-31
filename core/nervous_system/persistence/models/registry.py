"""ORM mappings for immutable source registration and historical imports."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import BigInteger, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import (
    Base,
    JSONB,
    SCHEMA,
    jsonb_column,
    jsonb_default,
    utc_timestamp,
    uuid_primary_key,
)


class SourceArtifact(Base):
    """Immutable identity and byte hash for a discovered source artifact."""

    __tablename__ = "source_artifacts"

    source_id: Mapped[UUID] = uuid_primary_key()
    uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    byte_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    source_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    discovered_at: Mapped[datetime] = utc_timestamp()
    # ``metadata`` is reserved by SQLAlchemy's declarative API; retain the
    # exact database column name under a safe Python attribute.
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default=jsonb_default(),
    )

    __table_args__ = (
        UniqueConstraint(
            "uri",
            "sha256",
            name="uq_ns_source_artifacts_uri_sha256",
        ),
    )


class ImportRun(Base):
    """One versioned, repeatable historical import attempt."""

    __tablename__ = "import_runs"

    import_run_id: Mapped[UUID] = uuid_primary_key()
    importer_version: Mapped[str] = mapped_column(String(64), nullable=False)
    started_at: Mapped[datetime] = utc_timestamp()
    finished_at: Mapped[datetime | None] = utc_timestamp(nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    counts: Mapped[dict[str, Any]] = jsonb_column()


class ImportItem(Base):
    """Normalized row identity, including the importer version in its key."""

    __tablename__ = "import_items"

    import_item_id: Mapped[UUID] = uuid_primary_key()
    import_run_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.import_runs.import_run_id",
            name="fk_ns_import_items_import_run",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    source_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.source_artifacts.source_id",
            name="fk_ns_import_items_source",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    importer_version: Mapped[str] = mapped_column(String(64), nullable=False)
    record_locator: Mapped[str] = mapped_column(String(512), nullable=False)
    normalized_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    target_type: Mapped[str] = mapped_column(String(64), nullable=False)
    target_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    warnings: Mapped[dict[str, Any]] = jsonb_column()

    __table_args__ = (
        UniqueConstraint(
            "source_id",
            "record_locator",
            "importer_version",
            "normalized_hash",
            name="uq_ns_import_items_identity",
        ),
    )


class ImportQuarantine(Base):
    """Immutable evidence for invalid rows, including malformed raw text."""

    __tablename__ = "import_quarantine"

    quarantine_id: Mapped[UUID] = uuid_primary_key()
    import_run_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.import_runs.import_run_id",
            name="fk_ns_import_quarantine_import_run",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    source_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.source_artifacts.source_id",
            name="fk_ns_import_quarantine_source",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    record_locator: Mapped[str] = mapped_column(String(512), nullable=False)
    raw_payload: Mapped[dict[str, Any] | None] = jsonb_column(nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str] = mapped_column(String(64), nullable=False)
    error_message: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = utc_timestamp()


class LineageEdge(Base):
    """Source-to-derived relationship retained for audit and replay."""

    __tablename__ = "lineage_edges"

    lineage_edge_id: Mapped[UUID] = uuid_primary_key()
    source_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.source_artifacts.source_id",
            name="fk_ns_lineage_edges_source",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    target_type: Mapped[str] = mapped_column(String(64), nullable=False)
    target_id: Mapped[str] = mapped_column(String(256), nullable=False)
    relationship: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = utc_timestamp()


class ConfigSnapshot(Base):
    """Immutable configuration payload used by a producer or decision."""

    __tablename__ = "config_snapshots"

    config_snapshot_id: Mapped[UUID] = uuid_primary_key()
    config_version: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = jsonb_column()
    created_at: Mapped[datetime] = utc_timestamp()

    __table_args__ = (
        UniqueConstraint(
            "content_hash",
            name="uq_ns_config_snapshots_content_hash",
        ),
    )


__all__ = [
    "ConfigSnapshot",
    "ImportItem",
    "ImportQuarantine",
    "ImportRun",
    "LineageEdge",
    "SourceArtifact",
]
