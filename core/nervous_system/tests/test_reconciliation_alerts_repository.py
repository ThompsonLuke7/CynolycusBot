"""Reconciliation runs and append-only alert events (Task 25).

`alerts` is a *deduplicated projection*: one row per distinct problem, carrying
how many times it has been seen and when it was first and last seen. That is
what an operator wants to look at — a hundred rows for one stuck order is
noise, not information.

`alert_events` is the immutable history behind it. The projection can be
rebuilt from the events; the events are never rewritten. Keeping only the
projection would lose when each occurrence actually happened, which is the part
you need when reconstructing an incident.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from core.nervous_system.persistence.repositories.observability import (
    ObservabilityRepository,
)


NOW = datetime(2026, 8, 3, 20, 0, tzinfo=timezone.utc)


@pytest.fixture
def repo(pg_session):
    return ObservabilityRepository(pg_session)


def _alert(**updates):
    payload = {
        "code": "ORDER_STUCK",
        "severity": "CRITICAL",
        "component": "execution.gateway",
        "entity_id": "AMD",
        "message": "order has been SUBMITTING for 10 minutes",
        "observed_at": NOW,
        "details": {"order_request_id": "abc"},
    }
    payload.update(updates)
    return payload


# ---------------------------------------------------------------------------
# Alerts: one projection row, many events
# ---------------------------------------------------------------------------


def test_a_first_detection_opens_the_projection(repo) -> None:
    alert = repo.record_alert(**_alert())

    assert alert.status == "OPEN"
    assert alert.occurrence_count == 1
    assert alert.opened_at == NOW
    assert alert.last_seen_at == NOW


def test_a_repeat_detection_updates_the_same_row(repo) -> None:
    """A hundred rows for one stuck order is noise, not information."""

    first = repo.record_alert(**_alert())
    second = repo.record_alert(**_alert(observed_at=NOW + timedelta(minutes=5)))

    assert second.alert_id == first.alert_id
    assert second.occurrence_count == 2
    assert second.opened_at == NOW, "first seen never moves"
    assert second.last_seen_at == NOW + timedelta(minutes=5)


def test_a_different_entity_is_a_different_alert(repo) -> None:
    first = repo.record_alert(**_alert(entity_id="AMD"))
    second = repo.record_alert(**_alert(entity_id="NVDA"))

    assert second.alert_id != first.alert_id


def test_a_different_code_is_a_different_alert(repo) -> None:
    first = repo.record_alert(**_alert(code="ORDER_STUCK"))
    second = repo.record_alert(**_alert(code="RECONCILE_MISMATCH"))

    assert second.alert_id != first.alert_id


def test_every_detection_appends_an_event(repo) -> None:
    """The projection says how many; the events say when each one was."""

    repo.record_alert(**_alert())
    repo.record_alert(**_alert(observed_at=NOW + timedelta(minutes=5)))

    events = repo.alert_events(code="ORDER_STUCK", entity_id="AMD")
    assert [event.observed_at for event in events] == [
        NOW, NOW + timedelta(minutes=5)
    ]


def test_alert_events_are_never_rewritten(repo) -> None:
    """Rebuilding the projection is fine; rewriting history is not."""

    repo.record_alert(**_alert(message="first wording"))
    repo.record_alert(**_alert(observed_at=NOW + timedelta(minutes=5),
                               message="second wording"))

    events = repo.alert_events(code="ORDER_STUCK", entity_id="AMD")
    assert [event.message for event in events] == ["first wording", "second wording"]


def test_an_out_of_order_detection_does_not_move_last_seen_backwards(repo) -> None:
    """A late-arriving observation must not make an active alert look older
    than it is.
    """

    repo.record_alert(**_alert(observed_at=NOW + timedelta(minutes=5)))
    alert = repo.record_alert(**_alert(observed_at=NOW))

    assert alert.last_seen_at == NOW + timedelta(minutes=5)
    assert alert.occurrence_count == 2


def test_details_are_recorded_per_event(repo) -> None:
    repo.record_alert(**_alert(details={"attempt": 1}))
    repo.record_alert(**_alert(observed_at=NOW + timedelta(minutes=1),
                               details={"attempt": 2}))

    events = repo.alert_events(code="ORDER_STUCK", entity_id="AMD")
    assert [event.details["attempt"] for event in events] == [1, 2]


def test_open_critical_alerts_can_be_counted_for_health(repo) -> None:
    repo.record_alert(**_alert(severity="CRITICAL"))
    repo.record_alert(**_alert(code="MINOR", severity="WARNING"))

    assert repo.open_critical_alert_count() == 1


# ---------------------------------------------------------------------------
# Reconciliation runs
# ---------------------------------------------------------------------------


def _run(**updates):
    payload = {
        "reconciliation_run_id": uuid4(),
        "environment": "QA_PAPER",
        "account_alias": "paper",
        "observed_at": NOW,
        "broker_position_count": 3,
        "database_position_count": 3,
        "journal_event_count": 12,
    }
    payload.update(updates)
    return payload


def test_a_reconciliation_run_records_three_way_parity(repo) -> None:
    run = repo.record_reconciliation_run(**_run())

    assert run.broker_position_count == 3
    assert run.database_position_count == 3
    assert run.journal_event_count == 12
    assert run.status == "MATCHED"


def test_a_count_mismatch_is_recorded_as_a_discrepancy(repo) -> None:
    run = repo.record_reconciliation_run(**_run(database_position_count=2))

    assert run.status == "DISCREPANCY"


def test_reconciliation_items_carry_their_codes_and_related_ids(repo) -> None:
    run = repo.record_reconciliation_run(**_run(database_position_count=2))

    repo.append_reconciliation_item(
        reconciliation_run_id=run.reconciliation_run_id,
        broker_position_key="paper:AMD",
        discrepancy_code="QUANTITY_MISMATCH",
        ownership_code="UNASSIGNED",
        related_ids={"order_request_id": "abc"},
        details={"broker_qty": "100", "db_qty": "40"},
    )

    items = repo.reconciliation_items(run.reconciliation_run_id)
    assert [item.discrepancy_code for item in items] == ["QUANTITY_MISMATCH"]
    assert items[0].related_ids["order_request_id"] == "abc"


def test_the_latest_reconciliation_is_found_per_environment_and_account(repo) -> None:
    """Health reads this on every check, so it must be an indexed lookup rather
    than a scan of every run ever recorded.

    The newest row is inserted in the *middle* on purpose. If it were first or
    last, an unordered query could return it by luck from whichever end the
    scan happened to start; from the middle, neither end is right.
    """

    repo.record_reconciliation_run(**_run(observed_at=NOW - timedelta(hours=2)))
    newest = repo.record_reconciliation_run(**_run(observed_at=NOW))
    repo.record_reconciliation_run(**_run(observed_at=NOW - timedelta(hours=1)))
    repo.record_reconciliation_run(**_run(account_alias="other", observed_at=NOW))

    found = repo.latest_reconciliation(environment="QA_PAPER", account_alias="paper")
    assert found.reconciliation_run_id == newest.reconciliation_run_id


def test_a_missing_reconciliation_is_none_rather_than_an_error(repo) -> None:
    assert repo.latest_reconciliation(environment="QA_PAPER", account_alias="none") is None


def test_two_detectors_racing_produce_one_alert_with_a_correct_count(
    postgres_engine,
) -> None:
    """Two workers can spot the same problem at once. Without the row lock this
    is a lost update: two projection rows, or a count of one for two events.
    """

    from sqlalchemy.orm import Session

    code = f"RACE_{uuid4().hex[:8]}"
    sessions = [Session(bind=postgres_engine) for _ in range(2)]
    try:
        for index, session in enumerate(sessions):
            ObservabilityRepository(session).record_alert(
                **_alert(code=code, observed_at=NOW + timedelta(minutes=index))
            )
            session.commit()

        reader = Session(bind=postgres_engine)
        try:
            found = ObservabilityRepository(reader).alert_events(
                code=code, entity_id="AMD"
            )
            assert len(found) == 2
            from core.nervous_system.persistence.models import Alert as AlertRow
            from sqlalchemy import select

            rows = reader.execute(
                select(AlertRow).where(AlertRow.code == code)
            ).scalars().all()
            assert len(rows) == 1, "one problem, one projection row"
            assert rows[0].occurrence_count == 2
        finally:
            reader.close()
    finally:
        for session in sessions:
            session.close()
        cleanup = Session(bind=postgres_engine)
        try:
            from sqlalchemy import text

            cleanup.execute(
                text("delete from nervous_system.alert_events where code = :c"),
                {"c": code},
            )
            cleanup.execute(
                text("delete from nervous_system.alerts where code = :c"), {"c": code}
            )
            cleanup.commit()
        finally:
            cleanup.close()


def test_a_second_detector_blocks_while_the_first_holds_the_row(postgres_engine) -> None:
    """A genuine interleave, not two sequential commits: session A holds the
    projection row uncommitted, and B must wait rather than read a stale count
    and overwrite it. Proven by B timing out while A still holds the lock.
    """

    from sqlalchemy import text
    from sqlalchemy.orm import Session

    code = f"LOCK_{uuid4().hex[:8]}"
    a, b = Session(bind=postgres_engine), Session(bind=postgres_engine)
    try:
        ObservabilityRepository(a).record_alert(**_alert(code=code))
        a.commit()

        # A takes the row and holds it, uncommitted.
        ObservabilityRepository(a).record_alert(
            **_alert(code=code, observed_at=NOW + timedelta(minutes=1))
        )

        b.execute(text("set local lock_timeout = '250ms'"))
        with pytest.raises(Exception) as blocked:
            ObservabilityRepository(b).record_alert(
                **_alert(code=code, observed_at=NOW + timedelta(minutes=2))
            )
            b.commit()

        assert "lock" in str(blocked.value).lower()
    finally:
        a.rollback()
        b.rollback()
        a.close()
        b.close()
        cleanup = Session(bind=postgres_engine)
        try:
            from sqlalchemy import text as _text

            cleanup.execute(
                _text("delete from nervous_system.alert_events where code = :c"),
                {"c": code},
            )
            cleanup.execute(
                _text("delete from nervous_system.alerts where code = :c"), {"c": code}
            )
            cleanup.commit()
        finally:
            cleanup.close()


def test_two_concurrent_detectors_do_not_lose_an_occurrence(postgres_engine) -> None:
    """The lost-update case, which only real concurrency can show.

    Both workers read the projection, both increment, both commit. Without the
    row lock each reads count=1 and writes 2, and one occurrence disappears —
    the alert under-reports how often a problem is actually happening, which is
    exactly the signal an operator triages on. With the lock the second waits,
    re-reads, and writes 3.
    """

    import threading

    from sqlalchemy.orm import Session

    code = f"LOST_{uuid4().hex[:8]}"
    seed = Session(bind=postgres_engine)
    try:
        ObservabilityRepository(seed).record_alert(**_alert(code=code))
        seed.commit()
    finally:
        seed.close()

    ready = threading.Barrier(2, timeout=10)
    errors: list[BaseException] = []

    def _detect(offset: int) -> None:
        session = Session(bind=postgres_engine)
        try:
            # Both threads arrive at the read at the same moment.
            ready.wait()
            ObservabilityRepository(session).record_alert(
                **_alert(code=code, observed_at=NOW + timedelta(minutes=offset))
            )
            session.commit()
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)
            session.rollback()
        finally:
            session.close()

    threads = [threading.Thread(target=_detect, args=(index,)) for index in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    reader = Session(bind=postgres_engine)
    try:
        assert not errors, f"a detector failed: {errors[0]!r}"
        from sqlalchemy import select

        from core.nervous_system.persistence.models import Alert as AlertRow

        alert = reader.execute(
            select(AlertRow).where(AlertRow.code == code)
        ).scalar_one()
        assert alert.occurrence_count == 3, "an occurrence was lost"
    finally:
        reader.close()
        cleanup = Session(bind=postgres_engine)
        try:
            from sqlalchemy import text

            cleanup.execute(
                text("delete from nervous_system.alert_events where code = :c"),
                {"c": code},
            )
            cleanup.execute(
                text("delete from nervous_system.alerts where code = :c"), {"c": code}
            )
            cleanup.commit()
        finally:
            cleanup.close()
