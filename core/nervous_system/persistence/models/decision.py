"""ORM mappings for intents, policy decisions, and decision records."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    false,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import (
    Base,
    DecimalType,
    SCHEMA,
    jsonb_column,
    utc_timestamp,
    uuid_primary_key,
)


class TradeIntent(Base):
    """Immutable intent linked to the context snapshot used to create it."""

    __tablename__ = "trade_intents"

    intent_id: Mapped[UUID] = uuid_primary_key()
    strategy_id: Mapped[str] = mapped_column(String(128), nullable=False)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False)
    decision_time: Mapped[datetime] = utc_timestamp()
    snapshot_id: Mapped[UUID | None] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.context_snapshots.snapshot_id",
            name="fk_ns_trade_intents_snapshot",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = jsonb_column()
    created_at: Mapped[datetime] = utc_timestamp()


class PolicyDecision(Base):
    """Immutable result of deterministic policy evaluation."""

    __tablename__ = "policy_decisions"

    policy_decision_id: Mapped[UUID] = uuid_primary_key()
    intent_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.trade_intents.intent_id",
            name="fk_ns_policy_decisions_intent",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    snapshot_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.context_snapshots.snapshot_id",
            name="fk_ns_policy_decisions_snapshot",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    final_risk_budget: Mapped[Decimal] = mapped_column(DecimalType, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = jsonb_column()
    created_at: Mapped[datetime] = utc_timestamp()

    __table_args__ = (
        CheckConstraint(
            "final_risk_budget >= 0",
            name="ck_ns_policy_decisions_nonnegative_budget",
        ),
    )


class PolicyModifier(Base):
    """One ordered, auditable policy budget transformation."""

    __tablename__ = "policy_modifiers"

    modifier_id: Mapped[UUID] = uuid_primary_key()
    policy_decision_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.policy_decisions.policy_decision_id",
            name="fk_ns_policy_modifiers_policy_decision",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False)
    rule_id: Mapped[str] = mapped_column(String(128), nullable=False)
    operation: Mapped[str] = mapped_column(String(32), nullable=False)
    configured_value: Mapped[Decimal] = mapped_column(DecimalType, nullable=False)
    budget_before: Mapped[Decimal] = mapped_column(DecimalType, nullable=False)
    budget_after: Mapped[Decimal] = mapped_column(DecimalType, nullable=False)
    reason_code: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = jsonb_column()

    __table_args__ = (
        UniqueConstraint(
            "policy_decision_id",
            "sequence_no",
            name="uq_ns_policy_modifiers_decision_sequence",
        ),
        CheckConstraint(
            "configured_value >= 0 AND budget_before >= 0 AND budget_after >= 0",
            name="ck_ns_policy_modifiers_nonnegative_values",
        ),
    )


class DecisionRecord(Base):
    """Complete immutable decision chain, persisted before any order request."""

    __tablename__ = "decision_records"

    decision_record_id: Mapped[UUID] = uuid_primary_key()
    decision_time: Mapped[datetime] = utc_timestamp()
    snapshot_id: Mapped[UUID | None] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.context_snapshots.snapshot_id",
            name="fk_ns_decision_records_snapshot",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )
    intent_id: Mapped[UUID | None] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.trade_intents.intent_id",
            name="fk_ns_decision_records_intent",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )
    policy_decision_id: Mapped[UUID | None] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.policy_decisions.policy_decision_id",
            name="fk_ns_decision_records_policy_decision",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    failure_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = jsonb_column()
    created_at: Mapped[datetime] = utc_timestamp()

    __table_args__ = (
        UniqueConstraint(
            "content_hash",
            name="uq_ns_decision_records_content_hash",
        ),
        CheckConstraint(
            "((status = 'COMPLETE' AND snapshot_id IS NOT NULL "
            "AND intent_id IS NOT NULL AND policy_decision_id IS NOT NULL) OR "
            "(status = 'FAILED' AND failure_stage IS NOT NULL "
            "AND failure_reason IS NOT NULL))",
            name="ck_ns_decision_records_status_requirements",
        ),
    )


class DecisionOutcome(Base):
    """Hindsight evaluation linked to an immutable decision record."""

    __tablename__ = "decision_outcomes"

    outcome_id: Mapped[UUID] = uuid_primary_key()
    decision_record_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.decision_records.decision_record_id",
            name="fk_ns_decision_outcomes_decision_record",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    evaluated_at: Mapped[datetime] = utc_timestamp()
    horizon: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = jsonb_column()
    created_at: Mapped[datetime] = utc_timestamp()
    replay_run_id: Mapped[UUID | None] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.replay_runs.replay_run_id",
            name="fk_ns_decision_outcomes_replay_run",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )
    source_fitness_report_id: Mapped[UUID | None] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.source_fitness_reports.source_fitness_report_id",
            name="fk_ns_decision_outcomes_source_fitness_report",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )
    horizon_kind: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="BARS"
    )
    target_window_start: Mapped[datetime | None] = utc_timestamp(nullable=True)
    target_window_end: Mapped[datetime | None] = utc_timestamp(nullable=True)
    mark_basis: Mapped[str | None] = mapped_column(String(32), nullable=True)
    fill_basis: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source_observation_hashes: Mapped[dict[str, Any]] = jsonb_column(default="[]")
    # Revisions append; they never update a prior outcome. A maturing horizon
    # is PENDING, never zero -- a zero would read as a real flat result.
    revision_number: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="1"
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="PENDING"
    )
    option_pnl_eligible: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=false()
    )

    __table_args__ = (
        UniqueConstraint(
            "decision_record_id",
            "horizon",
            "revision_number",
            name="uq_ns_decision_outcomes_revision",
        ),
        CheckConstraint(
            "revision_number > 0", name="ck_ns_decision_outcomes_revision_positive"
        ),
        CheckConstraint(
            "status in ('PENDING', 'FINAL', 'SUPERSEDED', 'UNAVAILABLE')",
            name="ck_ns_decision_outcomes_status",
        ),
        CheckConstraint(
            "target_window_start is null or target_window_end is null "
            "or target_window_start <= target_window_end",
            name="ck_ns_decision_outcomes_target_window",
        ),
        {"schema": SCHEMA},
    )


__all__ = [
    "DecisionOutcome",
    "DecisionRecord",
    "PolicyDecision",
    "PolicyModifier",
    "TradeIntent",
]
