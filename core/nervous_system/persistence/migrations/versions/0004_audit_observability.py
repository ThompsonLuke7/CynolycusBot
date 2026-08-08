"""Reconciliation runs/items and append-only alert events.

`alerts` (from 0002) stays the deduplicated projection an operator reads.
`alert_events` is the immutable history behind it: the projection can be
rebuilt from the events, but the events are never rewritten, because when each
occurrence actually happened is exactly what you need when reconstructing an
incident.
"""

from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0004_audit_observability"
down_revision: str = "0003_replay_fitness"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

SCHEMA = "nervous_system"
TABLE_NAMES = ("reconciliation_runs", "reconciliation_items", "alert_events")


def _uuid() -> postgresql.UUID:
    return postgresql.UUID(as_uuid=True)


def _timestamp() -> sa.DateTime:
    return sa.DateTime(timezone=True)


def _jsonb() -> postgresql.JSONB:
    return postgresql.JSONB()


def _jsonb_default(value: str = "{}") -> sa.TextClause:
    return sa.text(f"'{value}'::jsonb")


def upgrade() -> None:
    op.create_table(
        "reconciliation_runs",
        sa.Column("reconciliation_run_id", _uuid(), nullable=False),
        sa.Column("environment", sa.String(32), nullable=False),
        sa.Column("account_alias", sa.String(64), nullable=False),
        sa.Column("observed_at", _timestamp(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("broker_position_count", sa.Integer(), nullable=False),
        sa.Column("database_position_count", sa.Integer(), nullable=False),
        sa.Column("journal_event_count", sa.Integer(), nullable=False),
        sa.Column("details", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.Column("created_at", _timestamp(), nullable=False),
        sa.PrimaryKeyConstraint(
            "reconciliation_run_id", name="pk_ns_reconciliation_runs"
        ),
        sa.CheckConstraint(
            "status in ('MATCHED', 'DISCREPANCY', 'FAILED')",
            name="ck_ns_reconciliation_runs_status",
        ),
        sa.CheckConstraint(
            "broker_position_count >= 0 and database_position_count >= 0 "
            "and journal_event_count >= 0",
            name="ck_ns_reconciliation_runs_counts",
        ),
        schema=SCHEMA,
    )
    # Health reads the latest run on every check; this keeps that a lookup
    # rather than a scan of every run ever recorded.
    op.create_index(
        "ix_ns_reconciliation_runs_latest",
        "reconciliation_runs",
        ["environment", "account_alias", sa.text("observed_at DESC")],
        schema=SCHEMA,
    )

    op.create_table(
        "reconciliation_items",
        sa.Column("reconciliation_item_id", _uuid(), nullable=False),
        sa.Column("reconciliation_run_id", _uuid(), nullable=False),
        sa.Column("broker_position_key", sa.String(256), nullable=False),
        sa.Column("discrepancy_code", sa.String(64), nullable=False),
        sa.Column("ownership_code", sa.String(64), nullable=True),
        sa.Column("related_ids", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.Column("details", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.Column("created_at", _timestamp(), nullable=False),
        sa.PrimaryKeyConstraint(
            "reconciliation_item_id", name="pk_ns_reconciliation_items"
        ),
        sa.ForeignKeyConstraint(
            ["reconciliation_run_id"],
            [f"{SCHEMA}.reconciliation_runs.reconciliation_run_id"],
            name="fk_ns_reconciliation_items_run",
            ondelete="RESTRICT",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_ns_reconciliation_items_run",
        "reconciliation_items",
        ["reconciliation_run_id"],
        schema=SCHEMA,
    )

    op.create_table(
        "alert_events",
        sa.Column("alert_event_id", _uuid(), nullable=False),
        sa.Column("alert_id", _uuid(), nullable=False),
        sa.Column("dedup_key", sa.String(256), nullable=False),
        sa.Column("code", sa.String(64), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("component", sa.String(128), nullable=False),
        sa.Column("entity_id", sa.String(256), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("observed_at", _timestamp(), nullable=False),
        sa.Column("details", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.Column("created_at", _timestamp(), nullable=False),
        sa.PrimaryKeyConstraint("alert_event_id", name="pk_ns_alert_events"),
        sa.ForeignKeyConstraint(
            ["alert_id"],
            [f"{SCHEMA}.alerts.alert_id"],
            name="fk_ns_alert_events_alert",
            ondelete="RESTRICT",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_ns_alert_events_history",
        "alert_events",
        ["dedup_key", "observed_at"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_index("ix_ns_alert_events_history", "alert_events", schema=SCHEMA)
    op.drop_index("ix_ns_reconciliation_items_run", "reconciliation_items", schema=SCHEMA)
    op.drop_index("ix_ns_reconciliation_runs_latest", "reconciliation_runs", schema=SCHEMA)
    for table_name in reversed(TABLE_NAMES):
        op.drop_table(table_name, schema=SCHEMA)
