"""Create replay run/decision tables, source-fitness reports, and extend outcomes.

Outcomes become append-only *revisions*: a maturing horizon is PENDING, never
zero, and a later evaluation appends a new revision rather than updating the
earlier one. An updated outcome would silently rewrite the record of what we
believed at the time, which is the one thing an audit trail may not do.
"""

from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0003_replay_fitness"
down_revision: str = "0002_decision_execution"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

SCHEMA = "nervous_system"
TABLE_NAMES = ("replay_runs", "replay_decisions", "source_fitness_reports")

OUTCOME_COLUMNS = (
    "replay_run_id",
    "source_fitness_report_id",
    "horizon_kind",
    "target_window_start",
    "target_window_end",
    "mark_basis",
    "fill_basis",
    "source_observation_hashes",
    "revision_number",
    "status",
    "option_pnl_eligible",
)


def _jsonb_default(value: str = "{}") -> sa.TextClause:
    return sa.text(f"'{value}'::jsonb")


def _uuid() -> postgresql.UUID:
    return postgresql.UUID(as_uuid=True)


def _timestamp() -> sa.DateTime:
    return sa.DateTime(timezone=True)


def _jsonb() -> postgresql.JSONB:
    return postgresql.JSONB()


def upgrade() -> None:
    op.create_table(
        "replay_runs",
        sa.Column("replay_run_id", _uuid(), nullable=False),
        # The manifest is immutable: a run is only reproducible against the
        # exact source set it saw.
        sa.Column("source_manifest_hash", sa.String(64), nullable=False),
        sa.Column("schedule_hash", sa.String(64), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("model_hash", sa.String(64), nullable=False),
        # Persisted so a repeat run reproduces exactly rather than approximately.
        sa.Column("deterministic_seed", sa.BigInteger(), nullable=False),
        sa.Column("execution_assumptions", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("limitations", _jsonb(), nullable=False, server_default=_jsonb_default("[]")),
        sa.Column("started_at", _timestamp(), nullable=False),
        sa.Column("completed_at", _timestamp(), nullable=True),
        sa.Column("created_at", _timestamp(), nullable=False),
        sa.PrimaryKeyConstraint("replay_run_id", name="pk_ns_replay_runs"),
        sa.UniqueConstraint(
            "source_manifest_hash",
            "schedule_hash",
            "config_hash",
            "model_hash",
            "deterministic_seed",
            name="uq_ns_replay_runs_identity",
        ),
        sa.CheckConstraint(
            "status in ('RUNNING', 'COMPLETE', 'FAILED', 'ABORTED')",
            name="ck_ns_replay_runs_status",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "replay_decisions",
        sa.Column("replay_decision_id", _uuid(), nullable=False),
        sa.Column("replay_run_id", _uuid(), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("decision_record_id", _uuid(), nullable=False),
        sa.Column("snapshot_hash", sa.String(64), nullable=False),
        sa.Column("decision_time", _timestamp(), nullable=False),
        sa.Column("decision_bar", _timestamp(), nullable=False),
        sa.Column("lineage", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.Column("created_at", _timestamp(), nullable=False),
        sa.PrimaryKeyConstraint("replay_decision_id", name="pk_ns_replay_decisions"),
        sa.ForeignKeyConstraint(
            ["replay_run_id"],
            [f"{SCHEMA}.replay_runs.replay_run_id"],
            name="fk_ns_replay_decisions_run",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["decision_record_id"],
            [f"{SCHEMA}.decision_records.decision_record_id"],
            name="fk_ns_replay_decisions_decision_record",
            ondelete="RESTRICT",
        ),
        # The replay clock advances only through scheduled decision points, so
        # a sequence number is unique within a run by construction.
        sa.UniqueConstraint(
            "replay_run_id", "sequence_no", name="uq_ns_replay_decisions_sequence"
        ),
        sa.UniqueConstraint(
            "replay_run_id",
            "decision_record_id",
            name="uq_ns_replay_decisions_identity",
        ),
        sa.CheckConstraint("sequence_no > 0", name="ck_ns_replay_decisions_sequence"),
        sa.CheckConstraint(
            "decision_bar <= decision_time",
            name="ck_ns_replay_decisions_causal_bar",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "source_fitness_reports",
        sa.Column("source_fitness_report_id", _uuid(), nullable=False),
        sa.Column("replay_run_id", _uuid(), nullable=True),
        sa.Column("status", sa.String(48), nullable=False),
        sa.Column("thresholds_hash", sa.String(64), nullable=False),
        sa.Column("source", sa.String(64), nullable=False),
        sa.Column("feed", sa.String(64), nullable=False),
        sa.Column("tier", sa.String(64), nullable=False),
        sa.Column("side_metrics", _jsonb(), nullable=False, server_default=_jsonb_default("[]")),
        sa.Column("reason_codes", _jsonb(), nullable=False, server_default=_jsonb_default("[]")),
        sa.Column("warnings", _jsonb(), nullable=False, server_default=_jsonb_default("[]")),
        sa.Column("evaluated_at", _timestamp(), nullable=False),
        sa.Column("created_at", _timestamp(), nullable=False),
        sa.PrimaryKeyConstraint(
            "source_fitness_report_id", name="pk_ns_source_fitness_reports"
        ),
        sa.ForeignKeyConstraint(
            ["replay_run_id"],
            [f"{SCHEMA}.replay_runs.replay_run_id"],
            name="fk_ns_source_fitness_reports_run",
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            "status in ('FIT_FOR_OPTION_PNL', 'SOURCE_FITNESS_INSUFFICIENT_SAMPLE', "
            "'SOURCE_UNFIT_FOR_OPTION_PNL')",
            name="ck_ns_source_fitness_reports_status",
        ),
        schema=SCHEMA,
    )

    op.add_column(
        "decision_outcomes",
        sa.Column("replay_run_id", _uuid(), nullable=True),
        schema=SCHEMA,
    )
    op.add_column(
        "decision_outcomes",
        sa.Column("source_fitness_report_id", _uuid(), nullable=True),
        schema=SCHEMA,
    )
    op.add_column(
        "decision_outcomes",
        sa.Column("horizon_kind", sa.String(32), nullable=False, server_default="BARS"),
        schema=SCHEMA,
    )
    op.add_column(
        "decision_outcomes",
        sa.Column("target_window_start", _timestamp(), nullable=True),
        schema=SCHEMA,
    )
    op.add_column(
        "decision_outcomes",
        sa.Column("target_window_end", _timestamp(), nullable=True),
        schema=SCHEMA,
    )
    op.add_column(
        "decision_outcomes",
        sa.Column("mark_basis", sa.String(32), nullable=True),
        schema=SCHEMA,
    )
    op.add_column(
        "decision_outcomes",
        sa.Column("fill_basis", sa.String(32), nullable=True),
        schema=SCHEMA,
    )
    op.add_column(
        "decision_outcomes",
        sa.Column(
            "source_observation_hashes",
            _jsonb(),
            nullable=False,
            server_default=_jsonb_default("[]"),
        ),
        schema=SCHEMA,
    )
    op.add_column(
        "decision_outcomes",
        sa.Column("revision_number", sa.Integer(), nullable=False, server_default="1"),
        schema=SCHEMA,
    )
    op.add_column(
        "decision_outcomes",
        sa.Column("status", sa.String(32), nullable=False, server_default="PENDING"),
        schema=SCHEMA,
    )
    op.add_column(
        "decision_outcomes",
        sa.Column("option_pnl_eligible", sa.Boolean(), nullable=False, server_default=sa.false()),
        schema=SCHEMA,
    )

    op.create_foreign_key(
        "fk_ns_decision_outcomes_replay_run",
        "decision_outcomes",
        "replay_runs",
        ["replay_run_id"],
        ["replay_run_id"],
        source_schema=SCHEMA,
        referent_schema=SCHEMA,
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_ns_decision_outcomes_source_fitness_report",
        "decision_outcomes",
        "source_fitness_reports",
        ["source_fitness_report_id"],
        ["source_fitness_report_id"],
        source_schema=SCHEMA,
        referent_schema=SCHEMA,
        ondelete="RESTRICT",
    )
    # Revisions append; they never update a prior outcome. One row per
    # (decision, horizon, revision) makes an in-place rewrite impossible.
    op.create_unique_constraint(
        "uq_ns_decision_outcomes_revision",
        "decision_outcomes",
        ["decision_record_id", "horizon", "revision_number"],
        schema=SCHEMA,
    )
    op.create_check_constraint(
        "ck_ns_decision_outcomes_revision_positive",
        "decision_outcomes",
        "revision_number > 0",
        schema=SCHEMA,
    )
    op.create_check_constraint(
        "ck_ns_decision_outcomes_status",
        "decision_outcomes",
        "status in ('PENDING', 'FINAL', 'SUPERSEDED', 'UNAVAILABLE')",
        schema=SCHEMA,
    )
    # A window that ends before it starts would silently invert every horizon
    # measured against it.
    op.create_check_constraint(
        "ck_ns_decision_outcomes_target_window",
        "decision_outcomes",
        "target_window_start is null or target_window_end is null "
        "or target_window_start <= target_window_end",
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_ns_decision_outcomes_target_window",
        "decision_outcomes",
        schema=SCHEMA,
        type_="check",
    )
    op.drop_constraint(
        "ck_ns_decision_outcomes_status",
        "decision_outcomes",
        schema=SCHEMA,
        type_="check",
    )
    op.drop_constraint(
        "ck_ns_decision_outcomes_revision_positive",
        "decision_outcomes",
        schema=SCHEMA,
        type_="check",
    )
    op.drop_constraint(
        "uq_ns_decision_outcomes_revision",
        "decision_outcomes",
        schema=SCHEMA,
        type_="unique",
    )
    op.drop_constraint(
        "fk_ns_decision_outcomes_source_fitness_report",
        "decision_outcomes",
        schema=SCHEMA,
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_ns_decision_outcomes_replay_run",
        "decision_outcomes",
        schema=SCHEMA,
        type_="foreignkey",
    )
    for column_name in reversed(OUTCOME_COLUMNS):
        op.drop_column("decision_outcomes", column_name, schema=SCHEMA)
    for table_name in reversed(TABLE_NAMES):
        op.drop_table(table_name, schema=SCHEMA)
