"""Repository-backed projections for the audit surface (Task 25).

Lists read indexed columns only. The payload JSON on a decision record is the
whole immutable chain, and deserializing it for every row of a list is how an
audit page becomes an outage as the table grows — the cost scales with history,
which is exactly the thing that only ever increases.

Detail is the opposite: one row, and there the full graph is the point.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from core.nervous_system.orchestration.read_models import AuditStore


NOW = datetime(2026, 8, 3, 20, 0, tzinfo=timezone.utc)


def _decision(pg_session, **updates):
    from core.nervous_system.persistence.models import DecisionRecord as Row

    decision_id = updates.pop("decision_record_id", None) or uuid4()
    payload = {
        "decision_record_id": decision_id,
        "decision_time": updates.pop("decision_time", NOW),
        "status": "FAILED",
        "failure_stage": "policy",
        "failure_reason": "vetoed",
        "content_hash": uuid4().hex.ljust(64, "0")[:64],
        "payload": {"big": "x" * 2000},
        "created_at": NOW,
    }
    payload.update(updates)
    pg_session.add(Row(**payload))
    pg_session.flush()
    return decision_id


@pytest.fixture
def store(pg_session):
    return AuditStore(pg_session)


# ---------------------------------------------------------------------------
# Lists are projections, not full loads
# ---------------------------------------------------------------------------


def test_a_decision_list_returns_summary_fields_only(store, pg_session) -> None:
    """The heavy payload must not be in a list row: the cost of the page would
    then grow with the size of every decision on it.
    """

    _decision(pg_session)

    rows = store.decisions(limit=10)

    assert rows, "expected at least one row"
    assert "payload" not in rows[0]
    assert set(rows[0]) >= {"decision_record_id", "decision_time", "status"}


def test_a_decision_list_is_newest_first(store, pg_session) -> None:
    older = _decision(pg_session, decision_time=NOW - timedelta(hours=2))
    newest = _decision(pg_session, decision_time=NOW)
    _decision(pg_session, decision_time=NOW - timedelta(hours=1))

    rows = store.decisions(limit=10)

    assert rows[0]["decision_record_id"] == str(newest)
    assert rows[-1]["decision_record_id"] == str(older)


def test_a_decision_list_honours_its_limit(store, pg_session) -> None:
    for _ in range(4):
        _decision(pg_session)

    assert len(store.decisions(limit=2)) == 2


def test_timestamps_are_serialised_as_utc(store, pg_session) -> None:
    _decision(pg_session)

    assert store.decisions(limit=1)[0]["decision_time"].endswith("+00:00")


def test_identifiers_are_serialised_as_strings(store, pg_session) -> None:
    """A UUID object does not survive JSON, and a list route must render."""

    _decision(pg_session)

    assert isinstance(store.decisions(limit=1)[0]["decision_record_id"], str)


# ---------------------------------------------------------------------------
# Detail loads the whole chain
# ---------------------------------------------------------------------------


def test_a_detail_load_includes_the_full_payload(store, pg_session) -> None:
    decision_id = _decision(pg_session)

    detail = store.decision(str(decision_id))

    assert detail["payload"]["big"].startswith("x")


def test_a_missing_detail_is_none_not_an_exception(store) -> None:
    """The router turns this into a 404; raising here would become a 503 and
    report a healthy system as broken.
    """

    assert store.decision(str(uuid4())) is None


def test_a_malformed_identifier_is_none_rather_than_a_crash(store) -> None:
    assert store.decision("not-a-uuid") is None


# ---------------------------------------------------------------------------
# Alerts and reconciliations
# ---------------------------------------------------------------------------


def test_alerts_are_listed_newest_first_with_counts(store, pg_session) -> None:
    from core.nervous_system.persistence.repositories.observability import (
        ObservabilityRepository,
    )

    observability = ObservabilityRepository(pg_session)
    code = f"LIST_{uuid4().hex[:6]}"
    observability.record_alert(
        code=code, severity="CRITICAL", component="c", message="m",
        observed_at=NOW, entity_id="AMD",
    )
    observability.record_alert(
        code=code, severity="CRITICAL", component="c", message="m",
        observed_at=NOW + timedelta(minutes=1), entity_id="AMD",
    )

    rows = [row for row in store.alerts(limit=50) if row["code"] == code]

    assert rows[0]["occurrence_count"] == 2
    assert "details" not in rows[0], "list rows stay light"


def test_reconciliations_are_listed_for_health(store, pg_session) -> None:
    from core.nervous_system.persistence.repositories.observability import (
        ObservabilityRepository,
    )

    ObservabilityRepository(pg_session).record_reconciliation_run(
        reconciliation_run_id=uuid4(), environment="QA_PAPER", account_alias="paper",
        observed_at=NOW, broker_position_count=3, database_position_count=2,
        journal_event_count=9,
    )

    rows = store.reconciliations(limit=10)

    assert rows[0]["status"] == "DISCREPANCY"
    assert rows[0]["broker_position_count"] == 3


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_reports_the_live_schema_revision(store) -> None:
    report = store.health()

    assert report.schema_revision == "0004_audit_observability"
    assert report.database_ok is True


def test_health_counts_open_critical_alerts(store, pg_session) -> None:
    from core.nervous_system.persistence.repositories.observability import (
        ObservabilityRepository,
    )

    ObservabilityRepository(pg_session).record_alert(
        code=f"H_{uuid4().hex[:6]}", severity="CRITICAL", component="c",
        message="m", observed_at=NOW,
    )

    assert store.health().open_critical_alerts >= 1


def test_health_checked_at_is_supplied_not_read_from_a_clock(store) -> None:
    """An injected clock keeps the report reproducible in a test and, more
    importantly, keeps the read path free of hidden time.
    """

    report = AuditStore(store._session, clock=lambda: NOW).health()

    assert report.checked_at == NOW


# ---------------------------------------------------------------------------
# Timestamp normalisation, tested directly
# ---------------------------------------------------------------------------


def test_a_non_utc_timestamp_is_converted_not_just_relabelled() -> None:
    """PostgreSQL hands back UTC, so the round trip through the store cannot
    show this. A value arriving with another offset — from a fixture, an import,
    or a future non-timestamptz column — must still render as UTC, or two rows
    in the same list would be on different clocks.
    """

    from core.nervous_system.orchestration.read_models import _utc

    eastern = datetime(2026, 8, 3, 16, 0, tzinfo=timezone(timedelta(hours=-4)))

    assert _utc(eastern) == "2026-08-03T20:00:00+00:00"


def test_a_naive_timestamp_is_treated_as_utc_rather_than_dropped() -> None:
    from core.nervous_system.orchestration.read_models import _utc

    assert _utc(datetime(2026, 8, 3, 20, 0)) == "2026-08-03T20:00:00+00:00"


def test_a_missing_timestamp_stays_none() -> None:
    from core.nervous_system.orchestration.read_models import _utc

    assert _utc(None) is None
