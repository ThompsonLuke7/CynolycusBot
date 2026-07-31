"""ORM mappings for ownership, jobs, outbox delivery, and alerts."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import (
    Base,
    DecimalType,
    JSONB,
    SCHEMA,
    jsonb_column,
    utc_timestamp,
    uuid_primary_key,
)


class PortfolioOwnership(Base):
    """Fill-backed attribution of one broker position component."""

    __tablename__ = "portfolio_ownership"

    ownership_id: Mapped[UUID] = uuid_primary_key()
    account_alias: Mapped[str] = mapped_column(String(64), nullable=False)
    broker_position_key: Mapped[str] = mapped_column(String(256), nullable=False)
    strategy_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ownership_status: Mapped[str] = mapped_column(String(32), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(DecimalType, nullable=False)
    source_fill_ids: Mapped[list[Any]] = jsonb_column(default="[]")
    decision_record_id: Mapped[UUID | None] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.decision_records.decision_record_id",
            name="fk_ns_portfolio_ownership_decision_record",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )
    order_request_id: Mapped[UUID | None] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.order_requests.order_request_id",
            name="fk_ns_portfolio_ownership_order_request",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )
    effective_at: Mapped[datetime] = utc_timestamp()
    ended_at: Mapped[datetime | None] = utc_timestamp(nullable=True)


class JobRun(Base):
    """Durable orchestration job lifecycle record."""

    __tablename__ = "job_runs"

    job_run_id: Mapped[UUID] = uuid_primary_key()
    job_type: Mapped[str] = mapped_column(String(128), nullable=False)
    scheduled_for: Mapped[datetime] = utc_timestamp()
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    host: Mapped[str] = mapped_column(String(128), nullable=False)
    revision: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = utc_timestamp()
    finished_at: Mapped[datetime | None] = utc_timestamp(nullable=True)
    heartbeat_at: Mapped[datetime | None] = utc_timestamp(nullable=True)
    lease_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_until: Mapped[datetime | None] = utc_timestamp(nullable=True)
    lease_token: Mapped[str | None] = mapped_column(String(128), nullable=True)
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    source_hashes: Mapped[dict[str, Any]] = jsonb_column()
    counts: Mapped[dict[str, Any]] = jsonb_column()
    dependency_ids: Mapped[list[Any]] = jsonb_column(default="[]")
    input_ids: Mapped[list[Any]] = jsonb_column(default="[]")
    output_ids: Mapped[list[Any]] = jsonb_column(default="[]")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    exception_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "job_type",
            "scheduled_for",
            "config_hash",
            name="uq_ns_job_runs_idempotency",
        ),
        CheckConstraint(
            "attempt_no >= 0",
            name="ck_ns_job_runs_nonnegative_attempts",
        ),
    )


class JobEvent(Base):
    """Append-only status event for one orchestration job."""

    __tablename__ = "job_events"

    job_event_id: Mapped[UUID] = uuid_primary_key()
    job_run_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.job_runs.job_run_id",
            name="fk_ns_job_events_job_run",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    observed_at: Mapped[datetime] = utc_timestamp()
    payload: Mapped[dict[str, Any]] = jsonb_column()


class OutboxEvent(Base):
    """Transactional event awaiting idempotent external dispatch."""

    __tablename__ = "outbox_events"

    outbox_event_id: Mapped[UUID] = uuid_primary_key()
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    aggregate_type: Mapped[str] = mapped_column(String(128), nullable=False)
    aggregate_id: Mapped[str] = mapped_column(String(256), nullable=False)
    payload: Mapped[dict[str, Any]] = jsonb_column()
    created_at: Mapped[datetime] = utc_timestamp()
    available_at: Mapped[datetime] = utc_timestamp()
    claimed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    claimed_until: Mapped[datetime | None] = utc_timestamp(nullable=True)
    delivered_at: Mapped[datetime | None] = utc_timestamp(nullable=True)
    claim_token: Mapped[str | None] = mapped_column(String(128), nullable=True)
    delivery_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default="0",
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    event_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "event_hash",
            name="uq_ns_outbox_events_event_hash",
        ),
        CheckConstraint(
            "delivery_attempts >= 0",
            name="ck_ns_outbox_nonnegative_attempts",
        ),
    )


class Alert(Base):
    """Deduplicated operational alert state."""

    __tablename__ = "alerts"

    alert_id: Mapped[UUID] = uuid_primary_key()
    dedup_key: Mapped[str] = mapped_column(String(256), nullable=False)
    code: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    component: Mapped[str] = mapped_column(String(128), nullable=False)
    entity_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    opened_at: Mapped[datetime] = utc_timestamp()
    last_seen_at: Mapped[datetime] = utc_timestamp()
    occurrence_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default="1",
    )
    acknowledged_at: Mapped[datetime | None] = utc_timestamp(nullable=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    resolved_at: Mapped[datetime | None] = utc_timestamp(nullable=True)
    details: Mapped[dict[str, Any]] = jsonb_column()

    __table_args__ = (
        UniqueConstraint(
            "dedup_key",
            name="uq_ns_alerts_dedup_key",
        ),
        CheckConstraint(
            "occurrence_count > 0",
            name="ck_ns_alerts_positive_occurrences",
        ),
    )


Index(
    "ix_ns_outbox_events_pending_delivery",
    OutboxEvent.delivered_at,
    OutboxEvent.available_at,
    OutboxEvent.claimed_until,
    OutboxEvent.created_at,
    OutboxEvent.outbox_event_id,
    postgresql_where=OutboxEvent.delivered_at.is_(None),
)


__all__ = ["Alert", "JobEvent", "JobRun", "OutboxEvent", "PortfolioOwnership"]
