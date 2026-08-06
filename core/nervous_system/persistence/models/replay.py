"""ORM mappings for replay runs, replay decisions, and source-fitness reports."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, SCHEMA, jsonb_column, utc_timestamp, uuid_primary_key


class ReplayRun(Base):
    """One reproducible replay, pinned to the exact sources it saw."""

    __tablename__ = "replay_runs"
    __table_args__ = (
        UniqueConstraint(
            "source_manifest_hash",
            "schedule_hash",
            "config_hash",
            "model_hash",
            "deterministic_seed",
            name="uq_ns_replay_runs_identity",
        ),
        CheckConstraint(
            "status in ('RUNNING', 'COMPLETE', 'FAILED', 'ABORTED')",
            name="ck_ns_replay_runs_status",
        ),
        {"schema": SCHEMA},
    )

    replay_run_id: Mapped[UUID] = uuid_primary_key()
    source_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    schedule_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    model_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # Persisted so a repeat run reproduces exactly rather than approximately.
    deterministic_seed: Mapped[int] = mapped_column(BigInteger, nullable=False)
    execution_assumptions: Mapped[dict[str, Any]] = jsonb_column()
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    limitations: Mapped[dict[str, Any]] = jsonb_column(default="[]")
    started_at: Mapped[datetime] = utc_timestamp()
    completed_at: Mapped[datetime | None] = utc_timestamp(nullable=True)
    created_at: Mapped[datetime] = utc_timestamp()


class ReplayDecision(Base):
    """One decision produced inside a replay, in schedule order."""

    __tablename__ = "replay_decisions"
    __table_args__ = (
        UniqueConstraint(
            "replay_run_id", "sequence_no", name="uq_ns_replay_decisions_sequence"
        ),
        UniqueConstraint(
            "replay_run_id",
            "decision_record_id",
            name="uq_ns_replay_decisions_identity",
        ),
        CheckConstraint("sequence_no > 0", name="ck_ns_replay_decisions_sequence"),
        CheckConstraint(
            "decision_bar <= decision_time",
            name="ck_ns_replay_decisions_causal_bar",
        ),
        {"schema": SCHEMA},
    )

    replay_decision_id: Mapped[UUID] = uuid_primary_key()
    replay_run_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.replay_runs.replay_run_id",
            name="fk_ns_replay_decisions_run",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False)
    decision_record_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.decision_records.decision_record_id",
            name="fk_ns_replay_decisions_decision_record",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    decision_time: Mapped[datetime] = utc_timestamp()
    decision_bar: Mapped[datetime] = utc_timestamp()
    lineage: Mapped[dict[str, Any]] = jsonb_column()
    created_at: Mapped[datetime] = utc_timestamp()


class SourceFitnessReport(Base):
    """A source's fitness verdict, with the bar it was judged against."""

    __tablename__ = "source_fitness_reports"
    __table_args__ = (
        CheckConstraint(
            "status in ('FIT_FOR_OPTION_PNL', 'SOURCE_FITNESS_INSUFFICIENT_SAMPLE', "
            "'SOURCE_UNFIT_FOR_OPTION_PNL')",
            name="ck_ns_source_fitness_reports_status",
        ),
        {"schema": SCHEMA},
    )

    source_fitness_report_id: Mapped[UUID] = uuid_primary_key()
    replay_run_id: Mapped[UUID | None] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.replay_runs.replay_run_id",
            name="fk_ns_source_fitness_reports_run",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(48), nullable=False)
    # A verdict without the bar it was judged against is not interpretable.
    thresholds_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    feed: Mapped[str] = mapped_column(String(64), nullable=False)
    tier: Mapped[str] = mapped_column(String(64), nullable=False)
    side_metrics: Mapped[dict[str, Any]] = jsonb_column(default="[]")
    reason_codes: Mapped[dict[str, Any]] = jsonb_column(default="[]")
    warnings: Mapped[dict[str, Any]] = jsonb_column(default="[]")
    evaluated_at: Mapped[datetime] = utc_timestamp()
    created_at: Mapped[datetime] = utc_timestamp()


__all__ = ["ReplayDecision", "ReplayRun", "SourceFitnessReport"]
