"""ORM mappings for state envelopes, snapshots, and broker observations."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import CheckConstraint, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import (
    Base,
    JSONB,
    SCHEMA,
    TimestampType,
    UUIDType,
    jsonb_column,
    utc_timestamp,
    uuid_primary_key,
)


class StateRecord(Base):
    """Relational state envelope plus its complete validated contract payload."""

    __tablename__ = "state_records"

    state_id: Mapped[UUID] = uuid_primary_key()
    state_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(128), nullable=False)
    as_of: Mapped[datetime] = utc_timestamp()
    available_at: Mapped[datetime] = utc_timestamp()
    generated_at: Mapped[datetime] = utc_timestamp()
    valid_until: Mapped[datetime] = utc_timestamp()
    schema_version: Mapped[int] = mapped_column(nullable=False)
    producer: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(128), nullable=False)
    config_version: Mapped[str] = mapped_column(String(128), nullable=False)
    quality_severity: Mapped[str] = mapped_column(String(16), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = jsonb_column()
    created_at: Mapped[datetime] = utc_timestamp()

    __table_args__ = (
        UniqueConstraint("content_hash", name="uq_ns_state_records_content_hash"),
        CheckConstraint(
            "valid_until > available_at",
            name="ck_ns_state_records_valid_window",
        ),
    )


Index(
    "ix_ns_state_records_type_entity_available",
    StateRecord.state_type,
    StateRecord.entity_id,
    StateRecord.available_at.desc(),
)
Index(
    "ix_ns_state_records_type_entity_asof",
    StateRecord.state_type,
    StateRecord.entity_id,
    StateRecord.as_of.desc(),
)
Index("ix_ns_state_records_valid_until", StateRecord.valid_until)
Index("ix_ns_state_records_content_hash", StateRecord.content_hash)


class ContextSnapshot(Base):
    """Immutable context selected for one strategy decision."""

    __tablename__ = "context_snapshots"

    snapshot_id: Mapped[UUID] = uuid_primary_key()
    decision_time: Mapped[datetime] = utc_timestamp()
    strategy_id: Mapped[str] = mapped_column(String(128), nullable=False)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False)
    freshness_profile: Mapped[str] = mapped_column(String(64), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = jsonb_column()
    created_at: Mapped[datetime] = utc_timestamp()


class PortfolioObservation(Base):
    """Timestamped broker account observation supplied by an adapter."""

    __tablename__ = "portfolio_observations"

    observation_id: Mapped[UUID] = uuid_primary_key()
    account_alias: Mapped[str] = mapped_column(String(64), nullable=False)
    broker_observed_at: Mapped[datetime] = utc_timestamp()
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = jsonb_column()
    created_at: Mapped[datetime] = utc_timestamp()


__all__ = ["ContextSnapshot", "PortfolioObservation", "StateRecord"]
