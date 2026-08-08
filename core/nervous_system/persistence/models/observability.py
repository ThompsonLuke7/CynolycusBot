"""ORM mappings for reconciliation runs/items and immutable alert events."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, SCHEMA, jsonb_column, utc_timestamp, uuid_primary_key


class ReconciliationRun(Base):
    """One three-way parity check between broker, database, and journal."""

    __tablename__ = "reconciliation_runs"
    __table_args__ = (
        CheckConstraint(
            "status in ('MATCHED', 'DISCREPANCY', 'FAILED')",
            name="ck_ns_reconciliation_runs_status",
        ),
        CheckConstraint(
            "broker_position_count >= 0 and database_position_count >= 0 "
            "and journal_event_count >= 0",
            name="ck_ns_reconciliation_runs_counts",
        ),
        # Health reads the latest run on every check.
        Index(
            "ix_ns_reconciliation_runs_latest",
            "environment",
            "account_alias",
            text("observed_at DESC"),
        ),
        {"schema": SCHEMA},
    )

    reconciliation_run_id: Mapped[UUID] = uuid_primary_key()
    environment: Mapped[str] = mapped_column(String(32), nullable=False)
    account_alias: Mapped[str] = mapped_column(String(64), nullable=False)
    observed_at: Mapped[datetime] = utc_timestamp()
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    broker_position_count: Mapped[int] = mapped_column(Integer, nullable=False)
    database_position_count: Mapped[int] = mapped_column(Integer, nullable=False)
    journal_event_count: Mapped[int] = mapped_column(Integer, nullable=False)
    details: Mapped[dict[str, Any]] = jsonb_column()
    created_at: Mapped[datetime] = utc_timestamp()


class ReconciliationItem(Base):
    """One discrepancy inside a reconciliation run."""

    __tablename__ = "reconciliation_items"
    __table_args__ = (
        Index("ix_ns_reconciliation_items_run", "reconciliation_run_id"),
        {"schema": SCHEMA},
    )

    reconciliation_item_id: Mapped[UUID] = uuid_primary_key()
    reconciliation_run_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.reconciliation_runs.reconciliation_run_id",
            name="fk_ns_reconciliation_items_run",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    broker_position_key: Mapped[str] = mapped_column(String(256), nullable=False)
    discrepancy_code: Mapped[str] = mapped_column(String(64), nullable=False)
    ownership_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    related_ids: Mapped[dict[str, Any]] = jsonb_column()
    details: Mapped[dict[str, Any]] = jsonb_column()
    created_at: Mapped[datetime] = utc_timestamp()


class AlertEvent(Base):
    """One detection. Immutable: the projection is rebuilt, never these."""

    __tablename__ = "alert_events"
    __table_args__ = (
        Index("ix_ns_alert_events_history", "dedup_key", "observed_at"),
        {"schema": SCHEMA},
    )

    alert_event_id: Mapped[UUID] = uuid_primary_key()
    alert_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.alerts.alert_id",
            name="fk_ns_alert_events_alert",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    dedup_key: Mapped[str] = mapped_column(String(256), nullable=False)
    code: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    component: Mapped[str] = mapped_column(String(128), nullable=False)
    entity_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    observed_at: Mapped[datetime] = utc_timestamp()
    details: Mapped[dict[str, Any]] = jsonb_column()
    created_at: Mapped[datetime] = utc_timestamp()


__all__ = ["AlertEvent", "ReconciliationItem", "ReconciliationRun"]
