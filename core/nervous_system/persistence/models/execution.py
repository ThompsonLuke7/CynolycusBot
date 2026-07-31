"""ORM mappings for planned orders, submissions, and execution events."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import (
    Base,
    DecimalType,
    SCHEMA,
    UUIDType,
    jsonb_column,
    utc_timestamp,
    uuid_primary_key,
)


class OrderRequest(Base):
    """Immutable order request.

    The contract's ``OrderRequest.decision_id`` is persisted as the nullable
    ``decision_record_id`` foreign key because orders are inserted only after
    the complete decision record has been allocated and persisted.
    """

    __tablename__ = "order_requests"

    order_request_id: Mapped[UUID] = uuid_primary_key()
    decision_record_id: Mapped[UUID | None] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.decision_records.decision_record_id",
            name="fk_ns_order_requests_decision_record",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )
    policy_decision_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.policy_decisions.policy_decision_id",
            name="fk_ns_order_requests_policy_decision",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    environment: Mapped[str] = mapped_column(String(32), nullable=False)
    account_alias: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(256), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    decision_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    risk_reducing: Mapped[bool] = mapped_column(Boolean, nullable=False)
    order_type: Mapped[str] = mapped_column(String(32), nullable=False)
    broker_position_key: Mapped[str | None] = mapped_column(String(256), nullable=True)
    parent_quantity: Mapped[Decimal] = mapped_column(DecimalType, nullable=False)
    net_limit_price: Mapped[Decimal | None] = mapped_column(DecimalType, nullable=True)
    maximum_loss: Mapped[Decimal] = mapped_column(DecimalType, nullable=False)
    buying_power_required: Mapped[Decimal] = mapped_column(DecimalType, nullable=False)
    payload: Mapped[dict[str, Any]] = jsonb_column()
    created_at: Mapped[datetime] = utc_timestamp()
    expires_at: Mapped[datetime] = utc_timestamp()

    __table_args__ = (
        UniqueConstraint(
            "environment",
            "account_alias",
            "idempotency_key",
            name="uq_ns_order_requests_idempotency",
        ),
        CheckConstraint(
            "expires_at > created_at",
            name="ck_ns_order_requests_expiry",
        ),
        CheckConstraint(
            "parent_quantity > 0 AND maximum_loss >= 0 "
            "AND buying_power_required >= 0",
            name="ck_ns_order_requests_nonnegative_values",
        ),
        CheckConstraint(
            "((order_type = 'limit' AND net_limit_price IS NOT NULL AND net_limit_price > 0) OR "
            "(order_type = 'market' AND net_limit_price IS NULL))",
            name="ck_ns_order_requests_limit_price_semantics",
        ),
    )


class OrderLeg(Base):
    """One ordered leg of an approved option structure."""

    __tablename__ = "order_legs"

    order_leg_id: Mapped[UUID] = uuid_primary_key()
    order_request_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.order_requests.order_request_id",
            name="fk_ns_order_legs_order_request",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    position_intent: Mapped[str] = mapped_column(String(32), nullable=False)
    ratio: Mapped[int] = mapped_column(Integer, nullable=False)
    payload: Mapped[dict[str, Any]] = jsonb_column()

    __table_args__ = (
        UniqueConstraint(
            "order_request_id",
            "sequence_no",
            name="uq_ns_order_legs_request_sequence",
        ),
        CheckConstraint(
            "sequence_no >= 1 AND ratio > 0",
            name="ck_ns_order_legs_positive_sequence_ratio",
        ),
    )


class SubmissionAttempt(Base):
    """Durable reservation and broker submission attempt."""

    __tablename__ = "submission_attempts"

    submission_attempt_id: Mapped[UUID] = uuid_primary_key()
    order_request_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.order_requests.order_request_id",
            name="fk_ns_submission_attempts_order_request",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False)
    environment: Mapped[str] = mapped_column(String(32), nullable=False)
    account_alias: Mapped[str] = mapped_column(String(64), nullable=False)
    client_order_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    reserved_at: Mapped[datetime] = utc_timestamp()
    lease_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_until: Mapped[datetime | None] = utc_timestamp(nullable=True)
    claim_token: Mapped[str | None] = mapped_column(String(128), nullable=True)
    journaled_at: Mapped[datetime | None] = utc_timestamp(nullable=True)
    broker_called_at: Mapped[datetime | None] = utc_timestamp(nullable=True)
    resolved_at: Mapped[datetime | None] = utc_timestamp(nullable=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    journal_event_id: Mapped[UUID | None] = mapped_column(UUIDType, nullable=True)
    journal_event_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    journal_backend: Mapped[str | None] = mapped_column(String(32), nullable=True)
    journal_locator: Mapped[str | None] = mapped_column(String(512), nullable=True)
    journal_generation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[dict[str, Any]] = jsonb_column()

    __table_args__ = (
        UniqueConstraint(
            "environment",
            "account_alias",
            "client_order_id",
            name="uq_ns_submission_attempts_client_order",
        ),
        UniqueConstraint(
            "order_request_id",
            "attempt_no",
            name="uq_ns_submission_attempts_request_attempt",
        ),
        CheckConstraint(
            "attempt_no >= 1",
            name="ck_ns_submission_attempts_positive_attempt",
        ),
    )


class ExecutionEvent(Base):
    """Append-only broker execution event with relational fill facts."""

    __tablename__ = "execution_events"

    execution_event_id: Mapped[UUID] = uuid_primary_key()
    order_request_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.order_requests.order_request_id",
            name="fk_ns_execution_events_order_request",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    client_order_id: Mapped[str] = mapped_column(String(128), nullable=False)
    broker_order_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    broker_parent_order_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    observed_at: Mapped[datetime] = utc_timestamp()
    broker_event_at: Mapped[datetime | None] = utc_timestamp(nullable=True)
    filled_quantity: Mapped[Decimal] = mapped_column(DecimalType, nullable=False)
    average_fill_price: Mapped[Decimal | None] = mapped_column(DecimalType, nullable=True)
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False)
    previous_event_id: Mapped[UUID | None] = mapped_column(
        ForeignKey(
            f"{SCHEMA}.execution_events.execution_event_id",
            name="fk_ns_execution_events_previous_event",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )
    event_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    previous_event_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    journal_event_id: Mapped[UUID | None] = mapped_column(UUIDType, nullable=True)
    journal_event_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    journal_backend: Mapped[str | None] = mapped_column(String(32), nullable=True)
    journal_locator: Mapped[str | None] = mapped_column(String(512), nullable=True)
    journal_generation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[dict[str, Any]] = jsonb_column()

    __table_args__ = (
        UniqueConstraint(
            "event_hash",
            name="uq_ns_execution_events_event_hash",
        ),
        UniqueConstraint(
            "order_request_id",
            "sequence_no",
            name="uq_ns_execution_events_order_sequence",
        ),
        CheckConstraint(
            "(broker_event_at IS NULL OR broker_event_at <= observed_at) "
            "AND filled_quantity >= 0 "
            "AND (average_fill_price IS NULL OR average_fill_price >= 0)",
            name="ck_ns_execution_events_observed_order",
        ),
        CheckConstraint(
            "((sequence_no = 1 AND previous_event_id IS NULL "
            "AND previous_event_hash IS NULL) OR "
            "(sequence_no > 1 AND previous_event_id IS NOT NULL "
            "AND previous_event_hash IS NOT NULL))",
            name="ck_ns_execution_events_sequence_chain",
        ),
    )


__all__ = ["ExecutionEvent", "OrderLeg", "OrderRequest", "SubmissionAttempt"]
