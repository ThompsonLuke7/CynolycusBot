"""Create decision, execution, and operational coordination tables."""

from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0002_decision_execution"
down_revision: str = "0001_state_registry"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

SCHEMA = "nervous_system"
TABLE_NAMES = (
    "trade_intents",
    "policy_decisions",
    "policy_modifiers",
    "decision_records",
    "order_requests",
    "order_legs",
    "submission_attempts",
    "execution_events",
    "decision_outcomes",
    "portfolio_ownership",
    "job_runs",
    "job_events",
    "outbox_events",
    "alerts",
)


def _jsonb_default(value: str = "{}") -> sa.TextClause:
    return sa.text(f"'{value}'::jsonb")


def _uuid() -> postgresql.UUID:
    return postgresql.UUID(as_uuid=True)


def _timestamp() -> sa.DateTime:
    return sa.DateTime(timezone=True)


def _numeric() -> sa.Numeric:
    return sa.Numeric(20, 8)


def _jsonb() -> postgresql.JSONB:
    return postgresql.JSONB()


def upgrade() -> None:
    op.create_table(
        "trade_intents",
        sa.Column("intent_id", _uuid(), nullable=False),
        sa.Column("strategy_id", sa.String(128), nullable=False),
        sa.Column("ticker", sa.String(32), nullable=False),
        sa.Column("decision_time", _timestamp(), nullable=False),
        sa.Column("snapshot_id", _uuid(), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("payload", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.Column("created_at", _timestamp(), nullable=False),
        sa.PrimaryKeyConstraint("intent_id", name="pk_ns_trade_intents"),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            [f"{SCHEMA}.context_snapshots.snapshot_id"],
            name="fk_ns_trade_intents_snapshot",
            ondelete="RESTRICT",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "policy_decisions",
        sa.Column("policy_decision_id", _uuid(), nullable=False),
        sa.Column("intent_id", _uuid(), nullable=False),
        sa.Column("snapshot_id", _uuid(), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("final_risk_budget", _numeric(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("payload", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.Column("created_at", _timestamp(), nullable=False),
        sa.PrimaryKeyConstraint("policy_decision_id", name="pk_ns_policy_decisions"),
        sa.ForeignKeyConstraint(
            ["intent_id"],
            [f"{SCHEMA}.trade_intents.intent_id"],
            name="fk_ns_policy_decisions_intent",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            [f"{SCHEMA}.context_snapshots.snapshot_id"],
            name="fk_ns_policy_decisions_snapshot",
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            "final_risk_budget >= 0",
            name="ck_ns_policy_decisions_nonnegative_budget",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "policy_modifiers",
        sa.Column("modifier_id", _uuid(), nullable=False),
        sa.Column("policy_decision_id", _uuid(), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("rule_id", sa.String(128), nullable=False),
        sa.Column("operation", sa.String(32), nullable=False),
        sa.Column("configured_value", _numeric(), nullable=False),
        sa.Column("budget_before", _numeric(), nullable=False),
        sa.Column("budget_after", _numeric(), nullable=False),
        sa.Column("reason_code", sa.String(64), nullable=False),
        sa.Column("payload", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.PrimaryKeyConstraint("modifier_id", name="pk_ns_policy_modifiers"),
        sa.ForeignKeyConstraint(
            ["policy_decision_id"],
            [f"{SCHEMA}.policy_decisions.policy_decision_id"],
            name="fk_ns_policy_modifiers_policy_decision",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "policy_decision_id",
            "sequence_no",
            name="uq_ns_policy_modifiers_decision_sequence",
        ),
        sa.CheckConstraint(
            "configured_value >= 0 AND budget_before >= 0 AND budget_after >= 0",
            name="ck_ns_policy_modifiers_nonnegative_values",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "decision_records",
        sa.Column("decision_record_id", _uuid(), nullable=False),
        sa.Column("decision_time", _timestamp(), nullable=False),
        sa.Column("snapshot_id", _uuid(), nullable=True),
        sa.Column("intent_id", _uuid(), nullable=True),
        sa.Column("policy_decision_id", _uuid(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("failure_stage", sa.String(64), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("payload", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.Column("created_at", _timestamp(), nullable=False),
        sa.PrimaryKeyConstraint("decision_record_id", name="pk_ns_decision_records"),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            [f"{SCHEMA}.context_snapshots.snapshot_id"],
            name="fk_ns_decision_records_snapshot",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["intent_id"],
            [f"{SCHEMA}.trade_intents.intent_id"],
            name="fk_ns_decision_records_intent",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["policy_decision_id"],
            [f"{SCHEMA}.policy_decisions.policy_decision_id"],
            name="fk_ns_decision_records_policy_decision",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "content_hash",
            name="uq_ns_decision_records_content_hash",
        ),
        sa.CheckConstraint(
            "((status = 'COMPLETE' AND snapshot_id IS NOT NULL "
            "AND intent_id IS NOT NULL AND policy_decision_id IS NOT NULL) OR "
            "(status = 'FAILED' AND failure_stage IS NOT NULL "
            "AND failure_reason IS NOT NULL))",
            name="ck_ns_decision_records_status_requirements",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "order_requests",
        sa.Column("order_request_id", _uuid(), nullable=False),
        sa.Column("decision_record_id", _uuid(), nullable=True),
        sa.Column("policy_decision_id", _uuid(), nullable=False),
        sa.Column("environment", sa.String(32), nullable=False),
        sa.Column("account_alias", sa.String(64), nullable=False),
        sa.Column("idempotency_key", sa.String(256), nullable=False),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("decision_kind", sa.String(32), nullable=False),
        sa.Column("risk_reducing", sa.Boolean(), nullable=False),
        sa.Column("order_type", sa.String(32), nullable=False),
        sa.Column("broker_position_key", sa.String(256), nullable=True),
        sa.Column("parent_quantity", _numeric(), nullable=False),
        sa.Column("net_limit_price", _numeric(), nullable=True),
        sa.Column("maximum_loss", _numeric(), nullable=False),
        sa.Column("buying_power_required", _numeric(), nullable=False),
        sa.Column("payload", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.Column("created_at", _timestamp(), nullable=False),
        sa.Column("expires_at", _timestamp(), nullable=False),
        sa.PrimaryKeyConstraint("order_request_id", name="pk_ns_order_requests"),
        sa.ForeignKeyConstraint(
            ["decision_record_id"],
            [f"{SCHEMA}.decision_records.decision_record_id"],
            name="fk_ns_order_requests_decision_record",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["policy_decision_id"],
            [f"{SCHEMA}.policy_decisions.policy_decision_id"],
            name="fk_ns_order_requests_policy_decision",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "environment",
            "account_alias",
            "idempotency_key",
            name="uq_ns_order_requests_idempotency",
        ),
        sa.CheckConstraint(
            "expires_at > created_at",
            name="ck_ns_order_requests_expiry",
        ),
        sa.CheckConstraint(
            "parent_quantity > 0 AND maximum_loss >= 0 "
            "AND buying_power_required >= 0",
            name="ck_ns_order_requests_nonnegative_values",
        ),
        sa.CheckConstraint(
            "((order_type = 'LIMIT' AND net_limit_price IS NOT NULL "
            "AND net_limit_price > 0) OR "
            "(order_type = 'MARKET' AND net_limit_price IS NULL))",
            name="ck_ns_order_requests_limit_price_semantics",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "order_legs",
        sa.Column("order_leg_id", _uuid(), nullable=False),
        sa.Column("order_request_id", _uuid(), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(64), nullable=False),
        sa.Column("side", sa.String(16), nullable=False),
        sa.Column("position_intent", sa.String(32), nullable=False),
        sa.Column("ratio", sa.Integer(), nullable=False),
        sa.Column("payload", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.PrimaryKeyConstraint("order_leg_id", name="pk_ns_order_legs"),
        sa.ForeignKeyConstraint(
            ["order_request_id"],
            [f"{SCHEMA}.order_requests.order_request_id"],
            name="fk_ns_order_legs_order_request",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "order_request_id",
            "sequence_no",
            name="uq_ns_order_legs_request_sequence",
        ),
        sa.CheckConstraint(
            "sequence_no >= 1 AND ratio > 0",
            name="ck_ns_order_legs_positive_sequence_ratio",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "submission_attempts",
        sa.Column("submission_attempt_id", _uuid(), nullable=False),
        sa.Column("order_request_id", _uuid(), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("environment", sa.String(32), nullable=False),
        sa.Column("account_alias", sa.String(64), nullable=False),
        sa.Column("client_order_id", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("reserved_at", _timestamp(), nullable=False),
        sa.Column("lease_owner", sa.String(128), nullable=True),
        sa.Column("lease_until", _timestamp(), nullable=True),
        sa.Column("claim_token", sa.String(128), nullable=True),
        sa.Column("journaled_at", _timestamp(), nullable=True),
        sa.Column("broker_called_at", _timestamp(), nullable=True),
        sa.Column("resolved_at", _timestamp(), nullable=True),
        sa.Column("broker_order_id", sa.String(128), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("journal_event_id", _uuid(), nullable=True),
        sa.Column("journal_event_hash", sa.String(64), nullable=True),
        sa.Column("journal_backend", sa.String(32), nullable=True),
        sa.Column("journal_locator", sa.String(512), nullable=True),
        sa.Column("journal_generation", sa.Integer(), nullable=True),
        sa.Column("payload", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.PrimaryKeyConstraint("submission_attempt_id", name="pk_ns_submission_attempts"),
        sa.ForeignKeyConstraint(
            ["order_request_id"],
            [f"{SCHEMA}.order_requests.order_request_id"],
            name="fk_ns_submission_attempts_order_request",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "environment",
            "account_alias",
            "client_order_id",
            name="uq_ns_submission_attempts_client_order",
        ),
        sa.UniqueConstraint(
            "order_request_id",
            "attempt_no",
            name="uq_ns_submission_attempts_request_attempt",
        ),
        sa.CheckConstraint(
            "attempt_no >= 1",
            name="ck_ns_submission_attempts_positive_attempt",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "execution_events",
        sa.Column("execution_event_id", _uuid(), nullable=False),
        sa.Column("order_request_id", _uuid(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("client_order_id", sa.String(128), nullable=False),
        sa.Column("broker_order_id", sa.String(128), nullable=True),
        sa.Column("broker_parent_order_id", sa.String(128), nullable=True),
        sa.Column("observed_at", _timestamp(), nullable=False),
        sa.Column("broker_event_at", _timestamp(), nullable=True),
        sa.Column("filled_quantity", _numeric(), nullable=False),
        sa.Column("average_fill_price", _numeric(), nullable=True),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("previous_event_id", _uuid(), nullable=True),
        sa.Column("event_hash", sa.String(64), nullable=False),
        sa.Column("previous_event_hash", sa.String(64), nullable=True),
        sa.Column("journal_event_id", _uuid(), nullable=True),
        sa.Column("journal_event_hash", sa.String(64), nullable=True),
        sa.Column("journal_backend", sa.String(32), nullable=True),
        sa.Column("journal_locator", sa.String(512), nullable=True),
        sa.Column("journal_generation", sa.Integer(), nullable=True),
        sa.Column("payload", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.PrimaryKeyConstraint("execution_event_id", name="pk_ns_execution_events"),
        sa.ForeignKeyConstraint(
            ["order_request_id"],
            [f"{SCHEMA}.order_requests.order_request_id"],
            name="fk_ns_execution_events_order_request",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["previous_event_id"],
            [f"{SCHEMA}.execution_events.execution_event_id"],
            name="fk_ns_execution_events_previous_event",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "event_hash",
            name="uq_ns_execution_events_event_hash",
        ),
        sa.UniqueConstraint(
            "order_request_id",
            "sequence_no",
            name="uq_ns_execution_events_order_sequence",
        ),
        sa.CheckConstraint(
            "(broker_event_at IS NULL OR broker_event_at <= observed_at) "
            "AND filled_quantity >= 0 "
            "AND (average_fill_price IS NULL OR average_fill_price >= 0)",
            name="ck_ns_execution_events_observed_order",
        ),
        sa.CheckConstraint(
            "((sequence_no = 1 AND previous_event_id IS NULL "
            "AND previous_event_hash IS NULL) OR "
            "(sequence_no > 1 AND previous_event_id IS NOT NULL "
            "AND previous_event_hash IS NOT NULL))",
            name="ck_ns_execution_events_sequence_chain",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "decision_outcomes",
        sa.Column("outcome_id", _uuid(), nullable=False),
        sa.Column("decision_record_id", _uuid(), nullable=False),
        sa.Column("evaluated_at", _timestamp(), nullable=False),
        sa.Column("horizon", sa.String(64), nullable=False),
        sa.Column("payload", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.Column("created_at", _timestamp(), nullable=False),
        sa.PrimaryKeyConstraint("outcome_id", name="pk_ns_decision_outcomes"),
        sa.ForeignKeyConstraint(
            ["decision_record_id"],
            [f"{SCHEMA}.decision_records.decision_record_id"],
            name="fk_ns_decision_outcomes_decision_record",
            ondelete="RESTRICT",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "portfolio_ownership",
        sa.Column("ownership_id", _uuid(), nullable=False),
        sa.Column("account_alias", sa.String(64), nullable=False),
        sa.Column("broker_position_key", sa.String(256), nullable=False),
        sa.Column("strategy_id", sa.String(128), nullable=True),
        sa.Column("ownership_status", sa.String(32), nullable=False),
        sa.Column("quantity", _numeric(), nullable=False),
        sa.Column("source_fill_ids", _jsonb(), nullable=False, server_default=_jsonb_default("[]")),
        sa.Column("decision_record_id", _uuid(), nullable=True),
        sa.Column("order_request_id", _uuid(), nullable=True),
        sa.Column("effective_at", _timestamp(), nullable=False),
        sa.Column("ended_at", _timestamp(), nullable=True),
        sa.PrimaryKeyConstraint("ownership_id", name="pk_ns_portfolio_ownership"),
        sa.ForeignKeyConstraint(
            ["decision_record_id"],
            [f"{SCHEMA}.decision_records.decision_record_id"],
            name="fk_ns_portfolio_ownership_decision_record",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["order_request_id"],
            [f"{SCHEMA}.order_requests.order_request_id"],
            name="fk_ns_portfolio_ownership_order_request",
            ondelete="RESTRICT",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "job_runs",
        sa.Column("job_run_id", _uuid(), nullable=False),
        sa.Column("job_type", sa.String(128), nullable=False),
        sa.Column("scheduled_for", _timestamp(), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("host", sa.String(128), nullable=False),
        sa.Column("revision", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("started_at", _timestamp(), nullable=False),
        sa.Column("finished_at", _timestamp(), nullable=True),
        sa.Column("heartbeat_at", _timestamp(), nullable=True),
        sa.Column("lease_owner", sa.String(128), nullable=True),
        sa.Column("lease_until", _timestamp(), nullable=True),
        sa.Column("lease_token", sa.String(128), nullable=True),
        sa.Column("attempt_no", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source_hashes", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.Column("counts", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.Column("dependency_ids", _jsonb(), nullable=False, server_default=_jsonb_default("[]")),
        sa.Column("input_ids", _jsonb(), nullable=False, server_default=_jsonb_default("[]")),
        sa.Column("output_ids", _jsonb(), nullable=False, server_default=_jsonb_default("[]")),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("exception_summary", _jsonb(), nullable=True),
        sa.PrimaryKeyConstraint("job_run_id", name="pk_ns_job_runs"),
        sa.UniqueConstraint(
            "job_type",
            "scheduled_for",
            "config_hash",
            name="uq_ns_job_runs_idempotency",
        ),
        sa.CheckConstraint(
            "attempt_no >= 0",
            name="ck_ns_job_runs_nonnegative_attempts",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "job_events",
        sa.Column("job_event_id", _uuid(), nullable=False),
        sa.Column("job_run_id", _uuid(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("observed_at", _timestamp(), nullable=False),
        sa.Column("payload", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.PrimaryKeyConstraint("job_event_id", name="pk_ns_job_events"),
        sa.ForeignKeyConstraint(
            ["job_run_id"],
            [f"{SCHEMA}.job_runs.job_run_id"],
            name="fk_ns_job_events_job_run",
            ondelete="RESTRICT",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "outbox_events",
        sa.Column("outbox_event_id", _uuid(), nullable=False),
        sa.Column("event_type", sa.String(128), nullable=False),
        sa.Column("aggregate_type", sa.String(128), nullable=False),
        sa.Column("aggregate_id", sa.String(256), nullable=False),
        sa.Column("payload", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.Column("created_at", _timestamp(), nullable=False),
        sa.Column("available_at", _timestamp(), nullable=False),
        sa.Column("claimed_by", sa.String(128), nullable=True),
        sa.Column("claimed_until", _timestamp(), nullable=True),
        sa.Column("delivered_at", _timestamp(), nullable=True),
        sa.Column("claim_token", sa.String(128), nullable=True),
        sa.Column("delivery_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("event_hash", sa.String(64), nullable=False),
        sa.PrimaryKeyConstraint("outbox_event_id", name="pk_ns_outbox_events"),
        sa.UniqueConstraint(
            "event_hash",
            name="uq_ns_outbox_events_event_hash",
        ),
        sa.CheckConstraint(
            "delivery_attempts >= 0",
            name="ck_ns_outbox_nonnegative_attempts",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_ns_outbox_events_pending_delivery",
        "outbox_events",
        [
            "delivered_at",
            "available_at",
            "claimed_until",
            "created_at",
            "outbox_event_id",
        ],
        unique=False,
        schema=SCHEMA,
        postgresql_where=sa.text("delivered_at IS NULL"),
    )

    op.create_table(
        "alerts",
        sa.Column("alert_id", _uuid(), nullable=False),
        sa.Column("dedup_key", sa.String(256), nullable=False),
        sa.Column("code", sa.String(64), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("component", sa.String(128), nullable=False),
        sa.Column("entity_id", sa.String(256), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("opened_at", _timestamp(), nullable=False),
        sa.Column("last_seen_at", _timestamp(), nullable=False),
        sa.Column("occurrence_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("acknowledged_at", _timestamp(), nullable=True),
        sa.Column("acknowledged_by", sa.String(128), nullable=True),
        sa.Column("resolved_at", _timestamp(), nullable=True),
        sa.Column("details", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.PrimaryKeyConstraint("alert_id", name="pk_ns_alerts"),
        sa.UniqueConstraint("dedup_key", name="uq_ns_alerts_dedup_key"),
        sa.CheckConstraint(
            "occurrence_count > 0",
            name="ck_ns_alerts_positive_occurrences",
        ),
        schema=SCHEMA,
    )


def downgrade() -> None:
    for table_name in reversed(TABLE_NAMES):
        op.drop_table(table_name, schema=SCHEMA)
