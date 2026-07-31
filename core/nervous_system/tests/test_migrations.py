"""Static and PostgreSQL integration coverage for Task 6 migrations."""

from __future__ import annotations

from io import StringIO
import importlib
from pathlib import Path
import re
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from pydantic import ValidationError
from sqlalchemy import DateTime, Integer, Numeric, String, Text, inspect
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable
from sqlalchemy.types import BigInteger, Boolean
from sqlalchemy import JSON


REPO_ROOT = Path(__file__).resolve().parents[3]
ALEMBIC_INI = REPO_ROOT / "core/nervous_system/persistence/alembic.ini"
SCHEMA = "nervous_system"

EXPECTED_TABLES = {
    "state_records",
    "context_snapshots",
    "trade_intents",
    "policy_decisions",
    "policy_modifiers",
    "order_requests",
    "order_legs",
    "submission_attempts",
    "execution_events",
    "decision_records",
    "decision_outcomes",
    "portfolio_observations",
    "portfolio_ownership",
    "source_artifacts",
    "import_runs",
    "import_items",
    "import_quarantine",
    "lineage_edges",
    "config_snapshots",
    "job_runs",
    "job_events",
    "outbox_events",
    "alerts",
}

EXPECTED_PRIMARY_KEYS = {
    "state_records": "state_id",
    "context_snapshots": "snapshot_id",
    "trade_intents": "intent_id",
    "policy_decisions": "policy_decision_id",
    "policy_modifiers": "modifier_id",
    "order_requests": "order_request_id",
    "order_legs": "order_leg_id",
    "submission_attempts": "submission_attempt_id",
    "execution_events": "execution_event_id",
    "decision_records": "decision_record_id",
    "decision_outcomes": "outcome_id",
    "portfolio_observations": "observation_id",
    "portfolio_ownership": "ownership_id",
    "source_artifacts": "source_id",
    "import_runs": "import_run_id",
    "import_items": "import_item_id",
    "import_quarantine": "quarantine_id",
    "lineage_edges": "lineage_edge_id",
    "config_snapshots": "config_snapshot_id",
    "job_runs": "job_run_id",
    "job_events": "job_event_id",
    "outbox_events": "outbox_event_id",
    "alerts": "alert_id",
}

EXPECTED_TABLE_COLUMNS = {
    "state_records": {
        "state_id", "state_type", "entity_id", "as_of", "available_at",
        "generated_at", "valid_until", "schema_version", "producer",
        "model_version", "feature_version", "config_version", "quality_severity",
        "content_hash", "payload", "created_at",
    },
    "context_snapshots": {
        "snapshot_id", "decision_time", "strategy_id", "ticker",
        "freshness_profile", "content_hash", "payload", "created_at",
    },
    "portfolio_observations": {
        "observation_id", "account_alias", "broker_observed_at",
        "content_hash", "payload", "created_at",
    },
    "source_artifacts": {
        "source_id", "uri", "sha256", "byte_size", "source_kind",
        "discovered_at", "metadata",
    },
    "import_runs": {
        "import_run_id", "importer_version", "started_at", "finished_at",
        "status", "counts",
    },
    "import_items": {
        "import_item_id", "import_run_id", "source_id", "importer_version",
        "record_locator", "normalized_hash", "target_type", "target_id",
        "status", "warnings",
    },
    "import_quarantine": {
        "quarantine_id", "import_run_id", "source_id", "record_locator",
        "raw_payload", "raw_text", "error_code", "error_message", "created_at",
    },
    "lineage_edges": {
        "lineage_edge_id", "source_id", "target_type", "target_id",
        "relationship", "created_at",
    },
    "config_snapshots": {
        "config_snapshot_id", "config_version", "content_hash", "payload",
        "created_at",
    },
    "trade_intents": {
        "intent_id", "strategy_id", "ticker", "decision_time", "snapshot_id",
        "content_hash", "payload", "created_at",
    },
    "policy_decisions": {
        "policy_decision_id", "intent_id", "snapshot_id", "action",
        "final_risk_budget", "content_hash", "payload", "created_at",
    },
    "policy_modifiers": {
        "modifier_id", "policy_decision_id", "sequence_no", "rule_id",
        "operation", "configured_value", "budget_before", "budget_after",
        "reason_code", "payload",
    },
    "decision_records": {
        "decision_record_id", "decision_time", "snapshot_id", "intent_id",
        "policy_decision_id", "status", "failure_stage", "failure_reason",
        "content_hash", "payload", "created_at",
    },
    "decision_outcomes": {
        "outcome_id", "decision_record_id", "evaluated_at", "horizon",
        "payload", "created_at",
    },
    "order_requests": {
        "order_request_id", "decision_record_id", "policy_decision_id",
        "environment", "account_alias", "idempotency_key", "request_hash",
        "status", "decision_kind", "risk_reducing", "order_type",
        "broker_position_key", "parent_quantity", "net_limit_price", "maximum_loss",
        "buying_power_required", "payload", "created_at", "expires_at",
    },
    "order_legs": {
        "order_leg_id", "order_request_id", "sequence_no", "symbol", "side",
        "position_intent", "ratio", "payload",
    },
    "submission_attempts": {
        "submission_attempt_id", "order_request_id", "attempt_no", "environment",
        "account_alias", "client_order_id", "status", "reserved_at",
        "journaled_at", "broker_called_at", "resolved_at", "broker_order_id",
        "error_code", "journal_event_id", "journal_event_hash", "journal_backend",
        "journal_locator", "journal_generation", "lease_owner", "lease_until",
        "claim_token", "payload",
    },
    "execution_events": {
        "execution_event_id", "order_request_id", "status", "client_order_id",
        "event_type", "broker_order_id", "broker_parent_order_id", "observed_at", "broker_event_at", "filled_quantity",
        "average_fill_price", "sequence_no", "previous_event_id", "event_hash",
        "previous_event_hash", "journal_event_id", "journal_event_hash",
        "journal_backend", "journal_locator", "journal_generation", "payload",
    },
    "portfolio_ownership": {
        "ownership_id", "account_alias", "broker_position_key", "strategy_id",
        "ownership_status", "quantity", "source_fill_ids", "decision_record_id",
        "order_request_id", "effective_at", "ended_at",
    },
    "job_runs": {
        "job_run_id", "job_type", "scheduled_for", "config_hash", "host", "revision",
        "status", "started_at", "finished_at", "heartbeat_at", "lease_owner",
        "lease_until", "lease_token", "attempt_no", "source_hashes", "counts",
        "dependency_ids", "input_ids", "output_ids", "error", "last_error",
        "exception_summary",
    },
    "job_events": {
        "job_event_id", "job_run_id", "status", "observed_at", "payload",
    },
    "outbox_events": {
        "outbox_event_id", "event_type", "aggregate_type", "aggregate_id",
        "payload", "created_at", "available_at", "claimed_by", "claimed_until",
        "claim_token", "delivered_at", "delivery_attempts", "last_error", "event_hash",
    },
    "alerts": {
        "alert_id", "dedup_key", "code", "severity", "component", "entity_id",
        "message", "status", "opened_at", "last_seen_at", "occurrence_count",
        "acknowledged_at", "acknowledged_by", "resolved_at", "details",
    },
}

EXPECTED_OPTIONAL_COLUMNS = {
    "import_runs": {"finished_at"},
    "import_items": {"target_id"},
    "import_quarantine": {"raw_payload", "raw_text"},
    "trade_intents": {"snapshot_id"},
    "decision_records": {"snapshot_id", "intent_id", "policy_decision_id", "failure_stage", "failure_reason"},
    "order_requests": {"decision_record_id", "broker_position_key", "net_limit_price"},
    "submission_attempts": {
        "journaled_at", "broker_called_at", "resolved_at", "broker_order_id",
        "error_code", "journal_event_id", "journal_event_hash", "journal_backend",
        "journal_locator", "journal_generation", "lease_owner", "lease_until", "claim_token",
    },
    "execution_events": {
        "broker_order_id", "broker_parent_order_id", "broker_event_at", "average_fill_price", "previous_event_id",
        "previous_event_hash", "journal_event_id", "journal_event_hash",
        "journal_backend", "journal_locator", "journal_generation",
    },
    "portfolio_ownership": {"strategy_id", "decision_record_id", "order_request_id", "ended_at"},
    "job_runs": {"finished_at", "heartbeat_at", "lease_owner", "lease_until", "lease_token", "error", "last_error", "exception_summary"},
    "outbox_events": {"claimed_by", "claimed_until", "claim_token", "delivered_at", "last_error"},
    "alerts": {"entity_id", "acknowledged_at", "acknowledged_by", "resolved_at"},
}

UUID_COLUMNS = {
    (table, column)
    for table, columns in {
        "state_records": {"state_id"},
        "context_snapshots": {"snapshot_id"},
        "portfolio_observations": {"observation_id"},
        "source_artifacts": {"source_id"},
        "import_runs": {"import_run_id"},
        "import_items": {"import_item_id", "import_run_id", "source_id"},
        "import_quarantine": {"quarantine_id", "import_run_id", "source_id"},
        "lineage_edges": {"lineage_edge_id", "source_id"},
        "config_snapshots": {"config_snapshot_id"},
        "trade_intents": {"intent_id", "snapshot_id"},
        "policy_decisions": {"policy_decision_id", "intent_id", "snapshot_id"},
        "policy_modifiers": {"modifier_id", "policy_decision_id"},
        "decision_records": {"decision_record_id", "snapshot_id", "intent_id", "policy_decision_id"},
        "decision_outcomes": {"outcome_id", "decision_record_id"},
        "order_requests": {"order_request_id", "decision_record_id", "policy_decision_id"},
        "order_legs": {"order_leg_id", "order_request_id"},
        "submission_attempts": {"submission_attempt_id", "order_request_id", "journal_event_id"},
        "execution_events": {"execution_event_id", "order_request_id", "previous_event_id", "journal_event_id"},
        "portfolio_ownership": {"ownership_id", "decision_record_id", "order_request_id"},
        "job_runs": {"job_run_id"},
        "job_events": {"job_event_id", "job_run_id"},
        "outbox_events": {"outbox_event_id"},
        "alerts": {"alert_id"},
    }.items()
    for column in columns
}

TIMESTAMP_COLUMNS = {
    (table, column)
    for table, columns in {
        "state_records": {"as_of", "available_at", "generated_at", "valid_until", "created_at"},
        "context_snapshots": {"decision_time", "created_at"},
        "portfolio_observations": {"broker_observed_at", "created_at"},
        "source_artifacts": {"discovered_at"},
        "import_runs": {"started_at", "finished_at"},
        "import_quarantine": {"created_at"},
        "lineage_edges": {"created_at"},
        "config_snapshots": {"created_at"},
        "trade_intents": {"decision_time", "created_at"},
        "policy_decisions": {"created_at"},
        "decision_records": {"decision_time", "created_at"},
        "decision_outcomes": {"evaluated_at", "created_at"},
        "order_requests": {"created_at", "expires_at"},
        "submission_attempts": {"reserved_at", "journaled_at", "broker_called_at", "resolved_at", "lease_until"},
        "execution_events": {"observed_at", "broker_event_at"},
        "portfolio_ownership": {"effective_at", "ended_at"},
        "job_runs": {"scheduled_for", "started_at", "finished_at", "heartbeat_at", "lease_until"},
        "job_events": {"observed_at"},
        "outbox_events": {"created_at", "available_at", "claimed_until", "delivered_at"},
        "alerts": {"opened_at", "last_seen_at", "acknowledged_at", "resolved_at"},
    }.items()
    for column in columns
}

JSONB_COLUMNS = {
    (table, column)
    for table, columns in {
        "state_records": {"payload"},
        "context_snapshots": {"payload"},
        "portfolio_observations": {"payload"},
        "source_artifacts": {"metadata"},
        "import_runs": {"counts"},
        "import_items": {"warnings"},
        "import_quarantine": {"raw_payload"},
        "config_snapshots": {"payload"},
        "trade_intents": {"payload"},
        "policy_decisions": {"payload"},
        "policy_modifiers": {"payload"},
        "decision_records": {"payload"},
        "decision_outcomes": {"payload"},
        "order_requests": {"payload"},
        "order_legs": {"payload"},
        "submission_attempts": {"payload"},
        "execution_events": {"payload"},
        "portfolio_ownership": {"source_fill_ids"},
        "job_runs": {"source_hashes", "counts", "dependency_ids", "input_ids", "output_ids", "exception_summary"},
        "job_events": {"payload"},
        "outbox_events": {"payload"},
        "alerts": {"details"},
    }.items()
    for column in columns
}

NUMERIC_COLUMNS = {
    (table, column)
    for table, columns in {
        "policy_decisions": {"final_risk_budget"},
        "policy_modifiers": {"configured_value", "budget_before", "budget_after"},
        "order_requests": {"parent_quantity", "net_limit_price", "maximum_loss", "buying_power_required"},
        "execution_events": {"filled_quantity", "average_fill_price"},
        "portfolio_ownership": {"quantity"},
    }.items()
    for column in columns
}

INTEGER_COLUMNS = {
    (table, column)
    for table, columns in {
        "state_records": {"schema_version"},
        "policy_modifiers": {"sequence_no"},
        "order_legs": {"sequence_no", "ratio"},
        "submission_attempts": {"attempt_no", "journal_generation"},
        "execution_events": {"sequence_no", "journal_generation"},
        "job_runs": {"attempt_no"},
        "outbox_events": {"delivery_attempts"},
        "alerts": {"occurrence_count"},
    }.items()
    for column in columns
}

BIGINT_COLUMNS = {("source_artifacts", "byte_size")}
BOOLEAN_COLUMNS = {("order_requests", "risk_reducing")}
TEXT_COLUMNS = {
    ("import_quarantine", "error_message"),
    ("job_runs", "error"),
    ("job_runs", "last_error"),
    ("decision_records", "failure_reason"),
    ("outbox_events", "last_error"),
    ("alerts", "message"),
}

EXPECTED_UNIQUES = {
    "uq_ns_source_artifacts_uri_sha256": ("source_artifacts", ("uri", "sha256")),
    "uq_ns_state_records_content_hash": ("state_records", ("content_hash",)),
    "uq_ns_config_snapshots_content_hash": ("config_snapshots", ("content_hash",)),
    "uq_ns_import_items_identity": (
        "import_items",
        ("source_id", "record_locator", "importer_version", "normalized_hash"),
    ),
    "uq_ns_order_requests_idempotency": (
        "order_requests",
        ("environment", "account_alias", "idempotency_key"),
    ),
    "uq_ns_submission_attempts_client_order": (
        "submission_attempts",
        ("environment", "account_alias", "client_order_id"),
    ),
    "uq_ns_execution_events_event_hash": ("execution_events", ("event_hash",)),
    "uq_ns_decision_records_content_hash": (
        "decision_records",
        ("content_hash",),
    ),
    "uq_ns_alerts_dedup_key": ("alerts", ("dedup_key",)),
    "uq_ns_policy_modifiers_decision_sequence": (
        "policy_modifiers",
        ("policy_decision_id", "sequence_no"),
    ),
    "uq_ns_order_legs_request_sequence": (
        "order_legs",
        ("order_request_id", "sequence_no"),
    ),
    "uq_ns_submission_attempts_request_attempt": (
        "submission_attempts",
        ("order_request_id", "attempt_no"),
    ),
    "uq_ns_execution_events_order_sequence": (
        "execution_events",
        ("order_request_id", "sequence_no"),
    ),
    "uq_ns_job_runs_idempotency": (
        "job_runs",
        ("job_type", "scheduled_for", "config_hash"),
    ),
    "uq_ns_outbox_events_event_hash": ("outbox_events", ("event_hash",)),
}

EXPECTED_FKS = {
    "fk_ns_import_items_import_run": ("import_items", "import_run_id", "import_runs", "import_run_id"),
    "fk_ns_import_items_source": ("import_items", "source_id", "source_artifacts", "source_id"),
    "fk_ns_import_quarantine_import_run": ("import_quarantine", "import_run_id", "import_runs", "import_run_id"),
    "fk_ns_import_quarantine_source": ("import_quarantine", "source_id", "source_artifacts", "source_id"),
    "fk_ns_lineage_edges_source": ("lineage_edges", "source_id", "source_artifacts", "source_id"),
    "fk_ns_trade_intents_snapshot": ("trade_intents", "snapshot_id", "context_snapshots", "snapshot_id"),
    "fk_ns_policy_decisions_intent": ("policy_decisions", "intent_id", "trade_intents", "intent_id"),
    "fk_ns_policy_decisions_snapshot": ("policy_decisions", "snapshot_id", "context_snapshots", "snapshot_id"),
    "fk_ns_policy_modifiers_policy_decision": ("policy_modifiers", "policy_decision_id", "policy_decisions", "policy_decision_id"),
    "fk_ns_decision_records_snapshot": ("decision_records", "snapshot_id", "context_snapshots", "snapshot_id"),
    "fk_ns_decision_records_intent": ("decision_records", "intent_id", "trade_intents", "intent_id"),
    "fk_ns_decision_records_policy_decision": ("decision_records", "policy_decision_id", "policy_decisions", "policy_decision_id"),
    "fk_ns_order_requests_decision_record": ("order_requests", "decision_record_id", "decision_records", "decision_record_id"),
    "fk_ns_order_requests_policy_decision": ("order_requests", "policy_decision_id", "policy_decisions", "policy_decision_id"),
    "fk_ns_order_legs_order_request": ("order_legs", "order_request_id", "order_requests", "order_request_id"),
    "fk_ns_submission_attempts_order_request": ("submission_attempts", "order_request_id", "order_requests", "order_request_id"),
    "fk_ns_execution_events_order_request": ("execution_events", "order_request_id", "order_requests", "order_request_id"),
    "fk_ns_execution_events_previous_event": ("execution_events", "previous_event_id", "execution_events", "execution_event_id"),
    "fk_ns_decision_outcomes_decision_record": ("decision_outcomes", "decision_record_id", "decision_records", "decision_record_id"),
    "fk_ns_portfolio_ownership_decision_record": ("portfolio_ownership", "decision_record_id", "decision_records", "decision_record_id"),
    "fk_ns_portfolio_ownership_order_request": ("portfolio_ownership", "order_request_id", "order_requests", "order_request_id"),
    "fk_ns_job_events_job_run": ("job_events", "job_run_id", "job_runs", "job_run_id"),
}

EXPECTED_INDEXES = {
    "ix_ns_state_records_type_entity_available": ("state_records", ("state_type", "entity_id", "available_at")),
    "ix_ns_state_records_type_entity_asof": ("state_records", ("state_type", "entity_id", "as_of")),
    "ix_ns_state_records_valid_until": ("state_records", ("valid_until",)),
    "ix_ns_state_records_content_hash": ("state_records", ("content_hash",)),
    "ix_ns_outbox_events_pending_delivery": (
        "outbox_events",
        ("delivered_at", "available_at", "claimed_until", "created_at", "outbox_event_id"),
    ),
}

EXPECTED_CHECKS = {
    "ck_ns_state_records_valid_window",
    "ck_ns_order_requests_expiry",
    "ck_ns_order_requests_nonnegative_values",
    "ck_ns_policy_decisions_nonnegative_budget",
    "ck_ns_policy_modifiers_nonnegative_values",
    "ck_ns_execution_events_observed_order",
    "ck_ns_outbox_nonnegative_attempts",
    "ck_ns_alerts_positive_occurrences",
    "ck_ns_job_runs_nonnegative_attempts",
    "ck_ns_order_legs_positive_sequence_ratio",
    "ck_ns_submission_attempts_positive_attempt",
    "ck_ns_execution_events_sequence_chain",
    "ck_ns_decision_records_status_requirements",
    "ck_ns_order_requests_limit_price_semantics",
}


def _models() -> Any:
    return importlib.import_module("core.nervous_system.persistence.models")


def _all_tables() -> dict[str, Any]:
    return _models().Base.metadata.tables


def _table(name: str) -> Any:
    return _all_tables()[f"{SCHEMA}.{name}"]


def _alembic_config(output_buffer: StringIO | None = None) -> Config:
    cfg = Config(str(ALEMBIC_INI), output_buffer=output_buffer)
    cfg.set_main_option(
        "sqlalchemy.url",
        "postgresql+psycopg://offline:offline@127.0.0.1/cynolycus",
    )
    cfg.attributes["configure_logger"] = False
    return cfg


def _offline_sql(operation: str) -> str:
    output = StringIO()
    cfg = _alembic_config(output)
    if operation == "upgrade":
        command.upgrade(cfg, "head", sql=True)
    else:
        command.downgrade(cfg, "head:base", sql=True)
    return output.getvalue()


def _domain_table_names_from_sql(sql: str, verb: str) -> list[str]:
    pattern = rf"{verb} TABLE (?:IF EXISTS )?\"?{SCHEMA}\"?\.\"?([a-z_]+)\"?"
    return re.findall(pattern, sql)


def _constraint_columns(table: Any, constraint_type: type) -> dict[str, tuple[str, ...]]:
    return {
        constraint.name: tuple(column.name for column in constraint.columns)
        for constraint in table.constraints
        if isinstance(constraint, constraint_type) and constraint.name
    }


def test_orm_metadata_registers_exactly_23_tables() -> None:
    """The model package must register only the approved domain tables."""

    assert importlib.util.find_spec("core.nervous_system.persistence.models.base") is not None
    tables = _all_tables()
    assert {table.name for table in tables.values()} == EXPECTED_TABLES
    assert all(table.schema == SCHEMA for table in tables.values())


def test_each_table_has_native_uuid_primary_key_and_expected_columns() -> None:
    from sqlalchemy import PrimaryKeyConstraint

    tables = _all_tables()
    assert set(EXPECTED_TABLE_COLUMNS) == EXPECTED_TABLES
    for name, expected_columns in EXPECTED_TABLE_COLUMNS.items():
        table = tables[f"{SCHEMA}.{name}"]
        assert {column.name for column in table.columns} == expected_columns
        primary_keys = [constraint for constraint in table.constraints if isinstance(constraint, PrimaryKeyConstraint)]
        assert len(primary_keys) == 1
        assert [column.name for column in primary_keys[0].columns] == [EXPECTED_PRIMARY_KEYS[name]]
        primary_key = table.c[EXPECTED_PRIMARY_KEYS[name]]
        assert isinstance(primary_key.type, postgresql.UUID)
        assert primary_key.type.as_uuid is True
        assert primary_key.nullable is False


def test_every_column_has_expected_type_and_nullability() -> None:
    classified = UUID_COLUMNS | TIMESTAMP_COLUMNS | JSONB_COLUMNS | NUMERIC_COLUMNS | INTEGER_COLUMNS | BIGINT_COLUMNS | BOOLEAN_COLUMNS | TEXT_COLUMNS
    for table_name, expected_columns in EXPECTED_TABLE_COLUMNS.items():
        table = _table(table_name)
        optional = EXPECTED_OPTIONAL_COLUMNS.get(table_name, set())
        for column_name in expected_columns:
            column = table.c[column_name]
            assert column.nullable is (column_name in optional)
            key = (table_name, column_name)
            assert key in classified or isinstance(column.type, String)
            if key in UUID_COLUMNS:
                assert isinstance(column.type, postgresql.UUID)
                assert column.type.as_uuid is True
            elif key in TIMESTAMP_COLUMNS:
                assert isinstance(column.type, DateTime)
                assert column.type.timezone is True
            elif key in JSONB_COLUMNS:
                assert isinstance(column.type, postgresql.JSONB)
            elif key in NUMERIC_COLUMNS:
                assert isinstance(column.type, Numeric)
                assert (column.type.precision, column.type.scale) == (20, 8)
            elif key in INTEGER_COLUMNS:
                assert isinstance(column.type, Integer)
            elif key in BIGINT_COLUMNS:
                assert isinstance(column.type, BigInteger)
            elif key in BOOLEAN_COLUMNS:
                assert isinstance(column.type, Boolean)
            elif key in TEXT_COLUMNS:
                assert isinstance(column.type, Text)
            else:
                assert isinstance(column.type, String)


def test_relational_types_are_postgresql_specific_and_timestamps_are_aware() -> None:
    tables = _all_tables()
    timestamp_columns = {
        column
        for table in tables.values()
        for column in table.columns
        if isinstance(column.type, DateTime)
    }
    assert timestamp_columns
    assert all(column.type.timezone for column in timestamp_columns)
    assert any(isinstance(column.type, Numeric) for table in tables.values() for column in table.columns)
    assert any(isinstance(column.type, postgresql.JSONB) for table in tables.values() for column in table.columns)
    assert any(isinstance(column.type, BigInteger) for table in tables.values() for column in table.columns)
    assert any(isinstance(column.type, Text) for table in tables.values() for column in table.columns)
    assert not any(
        isinstance(column.type, postgresql.ENUM)
        for table in tables.values()
        for column in table.columns
    )
    assert not any(
        isinstance(column.type, Boolean) and column.name in {"state_type", "status", "action"}
        for table in tables.values()
        for column in table.columns
    )


def test_required_json_payloads_are_non_nullable() -> None:
    for table in _all_tables().values():
        for column in table.columns:
            if column.name in {"payload", "metadata", "counts", "warnings", "source_fill_ids", "dependency_ids", "input_ids", "output_ids", "details"}:
                assert isinstance(column.type, postgresql.JSONB)
                if not (table.name == "import_quarantine" and column.name == "raw_payload"):
                    assert column.nullable is False


def test_named_unique_constraints_match_the_audit_contract() -> None:
    from sqlalchemy import UniqueConstraint

    actual: dict[str, tuple[str, ...]] = {}
    for table in _all_tables().values():
        actual.update(_constraint_columns(table, UniqueConstraint))
    for name, (table_name, columns) in EXPECTED_UNIQUES.items():
        assert actual[name] == columns
        assert name in {constraint.name for constraint in _table(table_name).constraints}


def test_named_foreign_keys_are_restrictive_and_match_dependency_order() -> None:
    actual: dict[str, tuple[str, str, str]] = {}
    for table in _all_tables().values():
        for constraint in table.foreign_key_constraints:
            actual[constraint.name] = (
                table.name,
                next(iter(constraint.columns)).name,
                next(iter(constraint.elements)).target_fullname,
            )
            assert constraint.ondelete in (None, "RESTRICT", "NO ACTION")
    for name, (table, column, target_table, target_column) in EXPECTED_FKS.items():
        assert actual[name] == (table, column, f"{SCHEMA}.{target_table}.{target_column}")


def test_journal_receipts_and_recovery_fields_are_explicit_and_nullable() -> None:
    submission = _table("submission_attempts")
    execution = _table("execution_events")
    for table in (submission, execution):
        for column_name in (
            "journal_event_id",
            "journal_event_hash",
            "journal_backend",
            "journal_locator",
            "journal_generation",
        ):
            assert column_name in table.c
            assert table.c[column_name].nullable is True
    for column_name in ("lease_owner", "lease_until", "claim_token"):
        assert column_name in submission.c
        assert submission.c[column_name].nullable is True
    assert execution.c.event_type.nullable is False
    assert execution.c.broker_parent_order_id.nullable is True
    assert execution.c.sequence_no.nullable is False
    assert execution.c.previous_event_id.nullable is True


def test_job_run_idempotency_and_lease_fields_are_explicit() -> None:
    job_runs = _table("job_runs")
    assert "job_name" not in job_runs.c
    for column_name in (
        "job_type",
        "scheduled_for",
        "config_hash",
        "host",
        "revision",
        "source_hashes",
        "counts",
    ):
        assert column_name in job_runs.c
        assert job_runs.c[column_name].nullable is False
    for column_name in (
        "lease_owner",
        "lease_until",
        "lease_token",
        "exception_summary",
    ):
        assert column_name in job_runs.c
        assert job_runs.c[column_name].nullable is True
    assert "attempt_no" in job_runs.c
    assert job_runs.c.attempt_no.nullable is False


def test_order_exit_semantics_and_portfolio_lineage_are_explicit() -> None:
    order_requests = _table("order_requests")
    for column_name in ("decision_kind", "risk_reducing", "order_type"):
        assert column_name in order_requests.c
        assert order_requests.c[column_name].nullable is False
    assert order_requests.c.broker_position_key.nullable is True
    assert order_requests.c.net_limit_price.nullable is True

    ownership = _table("portfolio_ownership")
    assert ownership.c.decision_record_id.nullable is True
    assert ownership.c.order_request_id.nullable is True


def test_order_type_check_matches_lowercase_contract_and_price_semantics() -> None:
    from datetime import datetime, timedelta, timezone
    from decimal import Decimal
    from uuid import uuid4

    from sqlalchemy import CheckConstraint

    from core.nervous_system.contracts.enums import (
        DebitCredit,
        InstrumentFamily,
        OrderSide,
        RuntimeEnvironment,
    )
    from core.nervous_system.contracts.orders import OrderRequest

    created_at = datetime(2026, 7, 30, 14, 30, tzinfo=timezone.utc)
    common_request_fields = {
        "decision_id": uuid4(),
        "policy_decision_id": uuid4(),
        "environment": RuntimeEnvironment.QA_PAPER,
        "account_alias": "paper",
        "instrument_family": InstrumentFamily.EQUITY,
        "equity_symbol": "SPY",
        "equity_side": OrderSide.BUY,
        "parent_quantity": Decimal("1"),
        "debit_credit": DebitCredit.DEBIT,
        "maximum_loss": Decimal("125"),
        "buying_power_required": Decimal("125"),
        "time_in_force": "day",
        "created_at": created_at,
        "expires_at": created_at + timedelta(minutes=5),
    }
    limit_request = OrderRequest.create(
        **common_request_fields,
        net_limit_price=Decimal("1.25"),
        order_type="limit",
        idempotency_key="task-6-lowercase-limit-order",
    )
    market_request = OrderRequest.create(
        **common_request_fields,
        net_limit_price=None,
        order_type="market",
        idempotency_key="task-6-lowercase-market-order",
    )
    with pytest.raises(ValidationError):
        OrderRequest.create(
            **common_request_fields,
            net_limit_price=None,
            order_type="stop",
            idempotency_key="task-6-unsupported-order",
        )

    assert limit_request.order_type == "limit"
    assert limit_request.net_limit_price == Decimal("1.25")
    assert market_request.order_type == "market"
    assert market_request.net_limit_price is None

    checks = {
        constraint.name: str(constraint.sqltext)
        for constraint in _table("order_requests").constraints
        if isinstance(constraint, CheckConstraint)
    }
    check_sql = checks["ck_ns_order_requests_limit_price_semantics"]
    assert f"order_type = '{limit_request.order_type}'" in check_sql
    assert "net_limit_price IS NOT NULL" in check_sql
    assert "net_limit_price > 0" in check_sql
    assert f"order_type = '{market_request.order_type}'" in check_sql
    assert "net_limit_price IS NULL" in check_sql
    assert "order_type = 'LIMIT'" not in check_sql
    assert "order_type = 'MARKET'" not in check_sql

    migration_sql = _offline_sql("upgrade")
    assert f"order_type = '{limit_request.order_type}'" in migration_sql
    assert f"order_type = '{market_request.order_type}'" in migration_sql
    assert "order_type = 'LIMIT'" not in migration_sql
    assert "order_type = 'MARKET'" not in migration_sql


def test_failed_decision_records_can_be_durable_before_the_chain_exists() -> None:
    decision_records = _table("decision_records")
    for column_name in ("snapshot_id", "intent_id", "policy_decision_id"):
        assert decision_records.c[column_name].nullable is True
    assert decision_records.c.status.nullable is False
    assert decision_records.c.failure_stage.nullable is True
    assert decision_records.c.failure_reason.nullable is True

    from sqlalchemy import CheckConstraint

    checks = {
        constraint.name: str(constraint.sqltext)
        for constraint in decision_records.constraints
        if isinstance(constraint, CheckConstraint)
    }
    status_check = checks["ck_ns_decision_records_status_requirements"]
    assert "status = 'COMPLETE'" in status_check
    assert "status = 'FAILED'" in status_check
    assert "failure_stage IS NOT NULL" in status_check
    assert "failure_reason IS NOT NULL" in status_check
    assert "ck_ns_decision_records_status_requirements" in _offline_sql("upgrade")


def test_outbox_fencing_hash_and_pending_delivery_index_are_explicit() -> None:
    outbox = _table("outbox_events")
    assert outbox.c.claim_token.nullable is True
    assert outbox.c.event_hash.nullable is False
    pending_index = next(
        index for index in outbox.indexes if index.name == "ix_ns_outbox_events_pending_delivery"
    )
    assert tuple(column.name for column in pending_index.columns) == (
        "delivered_at", "available_at", "claimed_until", "created_at", "outbox_event_id",
    )
    from sqlalchemy import UniqueConstraint

    assert _constraint_columns(outbox, UniqueConstraint)["uq_ns_outbox_events_event_hash"] == ("event_hash",)


def test_state_indexes_and_named_checks_are_present() -> None:
    from sqlalchemy import CheckConstraint, Index

    indexes = {
        index.name: (index.table.name, tuple(column.name for column in index.columns))
        for table in _all_tables().values()
        for index in table.indexes
        if index.name
    }
    for name, expected in EXPECTED_INDEXES.items():
        assert indexes[name] == expected
    checks = {
        constraint.name
        for table in _all_tables().values()
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint) and constraint.name
    }
    assert EXPECTED_CHECKS <= checks
    assert not any(
        index.dialect_options["postgresql"].get("using") == "gin"
        for table in _all_tables().values()
        for index in table.indexes
    )


def test_execution_and_submission_checks_enforce_recovery_sequence_invariants() -> None:
    from sqlalchemy import CheckConstraint

    checks = {
        constraint.name: str(constraint.sqltext)
        for table in _all_tables().values()
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint) and constraint.name
    }
    assert "sequence_no >= 1" in checks["ck_ns_order_legs_positive_sequence_ratio"]
    assert "ratio > 0" in checks["ck_ns_order_legs_positive_sequence_ratio"]
    assert "attempt_no >= 1" in checks["ck_ns_submission_attempts_positive_attempt"]
    execution_check = checks["ck_ns_execution_events_sequence_chain"]
    assert "sequence_no = 1" in execution_check
    assert "previous_event_id IS NULL" in execution_check
    assert "previous_event_hash IS NULL" in execution_check
    assert "sequence_no > 1" in execution_check
    assert "previous_event_id IS NOT NULL" in execution_check
    assert "previous_event_hash IS NOT NULL" in execution_check

    sql = _offline_sql("upgrade")
    for name in (
        "ck_ns_order_legs_positive_sequence_ratio",
        "ck_ns_submission_attempts_positive_attempt",
        "ck_ns_execution_events_sequence_chain",
    ):
        assert name in sql


def test_metadata_tables_compile_with_the_postgresql_dialect() -> None:
    dialect = postgresql.dialect()
    for table in _all_tables().values():
        ddl = str(CreateTable(table).compile(dialect=dialect))
        assert f"CREATE TABLE {SCHEMA}.{table.name}" in ddl
        for index in table.indexes:
            index_ddl = str(CreateIndex(index).compile(dialect=dialect))
            assert index.name in index_ddl


def test_alembic_revision_chain_and_table_partition() -> None:
    first = importlib.import_module(
        "core.nervous_system.persistence.migrations.versions.0001_state_registry"
    )
    second = importlib.import_module(
        "core.nervous_system.persistence.migrations.versions.0002_decision_execution"
    )
    assert first.revision == "0001_state_registry"
    assert first.down_revision is None
    assert second.revision == "0002_decision_execution"
    assert second.down_revision == first.revision
    assert set(first.TABLE_NAMES) | set(second.TABLE_NAMES) == EXPECTED_TABLES
    assert set(first.TABLE_NAMES).isdisjoint(second.TABLE_NAMES)
    assert len(first.TABLE_NAMES) == 9
    assert len(second.TABLE_NAMES) == 14


def test_offline_upgrade_sql_has_public_version_and_dependency_order() -> None:
    sql = _offline_sql("upgrade")
    assert "nervous_system" in sql
    assert re.search(r"CREATE TABLE\s+(?!nervous_system\.)\"?alembic_version\"?", sql)
    assert "CREATE TABLE nervous_system.alembic_version" not in sql
    assert set(_domain_table_names_from_sql(sql, "CREATE")) == EXPECTED_TABLES
    schema_create = re.search(r"CREATE SCHEMA(?: IF NOT EXISTS)? nervous_system", sql)
    assert schema_create is not None
    assert schema_create.start() < sql.index("CREATE TABLE nervous_system.state_records")
    assert sql.index("CREATE TABLE nervous_system.source_artifacts") < sql.index("CREATE TABLE nervous_system.import_items")
    assert sql.index("CREATE TABLE nervous_system.import_runs") < sql.index("CREATE TABLE nervous_system.import_items")
    assert sql.index("CREATE TABLE nervous_system.decision_records") < sql.index("CREATE TABLE nervous_system.order_requests")
    assert sql.index("CREATE TABLE nervous_system.order_requests") < sql.index("CREATE TABLE nervous_system.order_legs")
    assert sql.index("CREATE TABLE nervous_system.order_requests") < sql.index("CREATE TABLE nervous_system.execution_events")


def test_offline_downgrade_sql_reverses_dependencies_and_drops_schema_last() -> None:
    sql = _offline_sql("downgrade")
    drops = _domain_table_names_from_sql(sql, "DROP")
    assert set(drops) == EXPECTED_TABLES
    assert drops.index("execution_events") < drops.index("order_requests")
    assert drops.index("order_requests") < drops.index("decision_records")
    assert drops.index("decision_records") < drops.index("policy_decisions")
    schema_drop = re.search(r"DROP SCHEMA(?: IF EXISTS)? nervous_system", sql)
    assert schema_drop is not None
    assert schema_drop.start() > sql.rfind("DROP TABLE nervous_system.state_records")


def test_metadata_and_offline_migrations_agree_on_domain_tables() -> None:
    upgrade_sql = _offline_sql("upgrade")
    assert set(_domain_table_names_from_sql(upgrade_sql, "CREATE")) == {
        table.name for table in _all_tables().values()
    }


@pytest.mark.postgres
def test_upgrade_inspect_downgrade_and_reupgrade_complete_schema(postgres_url: str) -> None:
    """Destructive test: ``postgres_url`` must be a dedicated disposable DB."""

    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", postgres_url)
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "head")

    from sqlalchemy import create_engine

    engine = create_engine(postgres_url)
    try:
        database_inspector = inspect(engine)
        assert set(database_inspector.get_table_names(schema=SCHEMA)) == EXPECTED_TABLES
        assert set(database_inspector.get_table_names(schema="public")) >= {"alembic_version"}
        unique_names = {
            constraint["name"]
            for table_name in EXPECTED_TABLES
            for constraint in database_inspector.get_unique_constraints(table_name, schema=SCHEMA)
            if constraint.get("name")
        }
        assert set(EXPECTED_UNIQUES) <= unique_names
        index_names = {
            index["name"]
            for table_name in EXPECTED_TABLES
            for index in database_inspector.get_indexes(table_name, schema=SCHEMA)
            if index.get("name")
        }
        assert set(EXPECTED_INDEXES) <= index_names
    finally:
        engine.dispose()

    command.downgrade(cfg, "base")
    engine = create_engine(postgres_url)
    try:
        assert not inspect(engine).has_schema(SCHEMA)
    finally:
        engine.dispose()

    command.upgrade(cfg, "head")
    engine = create_engine(postgres_url)
    try:
        assert set(inspect(engine).get_table_names(schema=SCHEMA)) == EXPECTED_TABLES
    finally:
        engine.dispose()
