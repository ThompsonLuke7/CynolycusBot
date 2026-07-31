"""Create state and source-registry tables."""

from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0001_state_registry"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

SCHEMA = "nervous_system"
TABLE_NAMES = (
    "state_records",
    "context_snapshots",
    "portfolio_observations",
    "source_artifacts",
    "import_runs",
    "import_items",
    "import_quarantine",
    "lineage_edges",
    "config_snapshots",
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
    op.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))

    op.create_table(
        "state_records",
        sa.Column("state_id", _uuid(), nullable=False),
        sa.Column("state_type", sa.String(32), nullable=False),
        sa.Column("entity_id", sa.String(128), nullable=False),
        sa.Column("as_of", _timestamp(), nullable=False),
        sa.Column("available_at", _timestamp(), nullable=False),
        sa.Column("generated_at", _timestamp(), nullable=False),
        sa.Column("valid_until", _timestamp(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("producer", sa.String(128), nullable=False),
        sa.Column("model_version", sa.String(128), nullable=False),
        sa.Column("feature_version", sa.String(128), nullable=False),
        sa.Column("config_version", sa.String(128), nullable=False),
        sa.Column("quality_severity", sa.String(16), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("payload", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.Column("created_at", _timestamp(), nullable=False),
        sa.PrimaryKeyConstraint("state_id", name="pk_ns_state_records"),
        sa.UniqueConstraint("content_hash", name="uq_ns_state_records_content_hash"),
        sa.CheckConstraint(
            "valid_until > available_at",
            name="ck_ns_state_records_valid_window",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "context_snapshots",
        sa.Column("snapshot_id", _uuid(), nullable=False),
        sa.Column("decision_time", _timestamp(), nullable=False),
        sa.Column("strategy_id", sa.String(128), nullable=False),
        sa.Column("ticker", sa.String(32), nullable=False),
        sa.Column("freshness_profile", sa.String(64), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("payload", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.Column("created_at", _timestamp(), nullable=False),
        sa.PrimaryKeyConstraint("snapshot_id", name="pk_ns_context_snapshots"),
        schema=SCHEMA,
    )

    op.create_table(
        "portfolio_observations",
        sa.Column("observation_id", _uuid(), nullable=False),
        sa.Column("account_alias", sa.String(64), nullable=False),
        sa.Column("broker_observed_at", _timestamp(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("payload", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.Column("created_at", _timestamp(), nullable=False),
        sa.PrimaryKeyConstraint("observation_id", name="pk_ns_portfolio_observations"),
        schema=SCHEMA,
    )

    op.create_table(
        "source_artifacts",
        sa.Column("source_id", _uuid(), nullable=False),
        sa.Column("uri", sa.String(1024), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("source_kind", sa.String(64), nullable=False),
        sa.Column("discovered_at", _timestamp(), nullable=False),
        sa.Column("metadata", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.PrimaryKeyConstraint("source_id", name="pk_ns_source_artifacts"),
        sa.UniqueConstraint(
            "uri",
            "sha256",
            name="uq_ns_source_artifacts_uri_sha256",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "import_runs",
        sa.Column("import_run_id", _uuid(), nullable=False),
        sa.Column("importer_version", sa.String(64), nullable=False),
        sa.Column("started_at", _timestamp(), nullable=False),
        sa.Column("finished_at", _timestamp(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("counts", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.PrimaryKeyConstraint("import_run_id", name="pk_ns_import_runs"),
        schema=SCHEMA,
    )

    op.create_table(
        "import_items",
        sa.Column("import_item_id", _uuid(), nullable=False),
        sa.Column("import_run_id", _uuid(), nullable=False),
        sa.Column("source_id", _uuid(), nullable=False),
        sa.Column("importer_version", sa.String(64), nullable=False),
        sa.Column("record_locator", sa.String(512), nullable=False),
        sa.Column("normalized_hash", sa.String(64), nullable=False),
        sa.Column("target_type", sa.String(64), nullable=False),
        sa.Column("target_id", sa.String(256), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("warnings", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.PrimaryKeyConstraint("import_item_id", name="pk_ns_import_items"),
        sa.ForeignKeyConstraint(
            ["import_run_id"],
            [f"{SCHEMA}.import_runs.import_run_id"],
            name="fk_ns_import_items_import_run",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            [f"{SCHEMA}.source_artifacts.source_id"],
            name="fk_ns_import_items_source",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "source_id",
            "record_locator",
            "importer_version",
            "normalized_hash",
            name="uq_ns_import_items_identity",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "import_quarantine",
        sa.Column("quarantine_id", _uuid(), nullable=False),
        sa.Column("import_run_id", _uuid(), nullable=False),
        sa.Column("source_id", _uuid(), nullable=False),
        sa.Column("record_locator", sa.String(512), nullable=False),
        sa.Column("raw_payload", _jsonb(), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=False),
        sa.Column("created_at", _timestamp(), nullable=False),
        sa.PrimaryKeyConstraint("quarantine_id", name="pk_ns_import_quarantine"),
        sa.ForeignKeyConstraint(
            ["import_run_id"],
            [f"{SCHEMA}.import_runs.import_run_id"],
            name="fk_ns_import_quarantine_import_run",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            [f"{SCHEMA}.source_artifacts.source_id"],
            name="fk_ns_import_quarantine_source",
            ondelete="RESTRICT",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "lineage_edges",
        sa.Column("lineage_edge_id", _uuid(), nullable=False),
        sa.Column("source_id", _uuid(), nullable=False),
        sa.Column("target_type", sa.String(64), nullable=False),
        sa.Column("target_id", sa.String(256), nullable=False),
        sa.Column("relationship", sa.String(64), nullable=False),
        sa.Column("created_at", _timestamp(), nullable=False),
        sa.PrimaryKeyConstraint("lineage_edge_id", name="pk_ns_lineage_edges"),
        sa.ForeignKeyConstraint(
            ["source_id"],
            [f"{SCHEMA}.source_artifacts.source_id"],
            name="fk_ns_lineage_edges_source",
            ondelete="RESTRICT",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "config_snapshots",
        sa.Column("config_snapshot_id", _uuid(), nullable=False),
        sa.Column("config_version", sa.String(128), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("payload", _jsonb(), nullable=False, server_default=_jsonb_default()),
        sa.Column("created_at", _timestamp(), nullable=False),
        sa.PrimaryKeyConstraint("config_snapshot_id", name="pk_ns_config_snapshots"),
        sa.UniqueConstraint("content_hash", name="uq_ns_config_snapshots_content_hash"),
        schema=SCHEMA,
    )

    op.execute(
        f"CREATE INDEX ix_ns_state_records_type_entity_available "
        f"ON {SCHEMA}.state_records (state_type, entity_id, available_at DESC)"
    )
    op.execute(
        f"CREATE INDEX ix_ns_state_records_type_entity_asof "
        f"ON {SCHEMA}.state_records (state_type, entity_id, as_of DESC)"
    )
    op.create_index(
        "ix_ns_state_records_valid_until",
        "state_records",
        ["valid_until"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_ns_state_records_content_hash",
        "state_records",
        ["content_hash"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_index("ix_ns_state_records_content_hash", table_name="state_records", schema=SCHEMA)
    op.drop_index("ix_ns_state_records_valid_until", table_name="state_records", schema=SCHEMA)
    op.execute(f"DROP INDEX IF EXISTS {SCHEMA}.ix_ns_state_records_type_entity_asof")
    op.execute(f"DROP INDEX IF EXISTS {SCHEMA}.ix_ns_state_records_type_entity_available")
    for table_name in reversed(TABLE_NAMES):
        op.drop_table(table_name, schema=SCHEMA)
    op.execute(sa.text(f"DROP SCHEMA IF EXISTS {SCHEMA}"))
