"""Transactional outbox, job leases, and the 4H loop guard (Task 22).

The SKIP LOCKED and lease tests run against PostgreSQL only; they are
meaningless on SQLite, which has no row-level locking.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import pytest

from core.nervous_system.orchestration.jobs import (
    Stage,
    StageStatus,
    redact_exception,
    run_stages,
)
from core.nervous_system.orchestration.outbox import OutboxDispatcher
from core.nervous_system.persistence.repositories.operations import (
    JobStatus,
    OperationsRepository,
)
from core.nervous_system.persistence.uow import UnitOfWork


UTC = timezone.utc
NOW = datetime(2026, 8, 3, 18, 30, tzinfo=UTC)


def clock(value: datetime = NOW):
    return lambda: value


# --------------------------------------------------------------------------
# Stage ordering and postconditions (no database required)
# --------------------------------------------------------------------------


def ok_stage(name: str, **kw) -> Stage:
    return Stage(name=name, run=lambda: 0, **kw)


def failing_stage(name: str, code: int = 1) -> Stage:
    return Stage(name=name, run=lambda: code)


def test_stages_run_in_order_when_all_succeed() -> None:
    calls: list[str] = []
    stages = [
        Stage(name=name, run=lambda n=name: (calls.append(n), 0)[1])
        for name in ("bars", "feeds", "matrix", "runner")
    ]

    results = run_stages(stages)

    assert calls == ["bars", "feeds", "matrix", "runner"]
    assert all(item.status is StageStatus.OK for item in results)


def test_a_failed_stage_prevents_every_later_stage() -> None:
    """The old loop ran the runner anyway; that is what this forbids."""

    calls: list[str] = []
    stages = [
        ok_stage("bars"),
        failing_stage("feeds"),
        Stage(name="matrix", run=lambda: (calls.append("matrix"), 0)[1]),
        Stage(name="runner", run=lambda: (calls.append("runner"), 0)[1]),
    ]

    results = run_stages(stages)

    assert calls == [], "nothing downstream of a failure may execute"
    assert results[1].status is StageStatus.FAILED
    assert results[2].status is StageStatus.NOT_RUN
    assert results[3].status is StageStatus.NOT_RUN
    assert "blocked by feeds" in results[3].reason


def test_a_nonzero_runner_status_is_a_failure() -> None:
    results = run_stages([failing_stage("runner", code=3)])

    assert results[0].status is StageStatus.FAILED
    assert results[0].exit_code == 3


def test_exit_zero_is_not_enough_without_a_postcondition() -> None:
    """update_meta_matrix.py returns zero for a no-op, so output is verified."""

    stage = Stage(
        name="matrix",
        run=lambda: 0,
        verify=lambda: (False, "matrix is 9 hours old", {"matrix_age_sec": 32400}),
    )

    results = run_stages([stage, ok_stage("runner")])

    assert results[0].status is StageStatus.FAILED
    assert "postcondition failed" in results[0].reason
    assert results[0].counts == {"matrix_age_sec": 32400}
    assert results[1].status is StageStatus.NOT_RUN


def test_a_verified_postcondition_passes_the_counts_through() -> None:
    stage = Stage(
        name="matrix", run=lambda: 0, verify=lambda: (True, "fresh", {"rows": 1200})
    )

    results = run_stages([stage])

    assert results[0].status is StageStatus.OK
    assert results[0].counts == {"rows": 1200}


def test_skipping_without_a_certificate_is_refused() -> None:
    stage = Stage(name="matrix", run=lambda: 0, skip=True)

    results = run_stages([stage, ok_stage("runner")])

    assert results[0].status is StageStatus.FAILED
    assert "skip refused" in results[0].reason
    assert results[1].status is StageStatus.NOT_RUN


def test_skipping_with_a_freshness_certificate_is_allowed() -> None:
    stage = Stage(
        name="matrix",
        run=lambda: 0,
        skip=True,
        skip_certificate=lambda: (True, "matrix is 120s old"),
    )

    results = run_stages([stage, ok_stage("runner")])

    assert results[0].status is StageStatus.SKIPPED_CERTIFIED
    assert results[0].ok
    assert results[1].status is StageStatus.OK


def test_an_optional_stage_failure_does_not_block() -> None:
    stages = [
        Stage(name="themes", run=lambda: 1, required=False),
        ok_stage("runner"),
    ]

    results = run_stages(stages)

    assert results[0].status is StageStatus.FAILED
    assert results[1].status is StageStatus.OK


def test_a_raising_stage_is_recorded_redacted() -> None:
    stage = Stage(name="feeds", run=lambda: (_ for _ in ()).throw(RuntimeError("api_key=SECRET")))

    results = run_stages([stage])

    assert results[0].status is StageStatus.FAILED
    assert "SECRET" not in results[0].reason


def test_exception_summaries_redact_credentials() -> None:
    assert "SECRET" not in str(redact_exception(ValueError("token=SECRET")))
    assert redact_exception(ValueError("plain failure"))["message"] == "plain failure"


# --------------------------------------------------------------------------
# 4H loop wiring
# --------------------------------------------------------------------------


def loop_args(**overrides) -> argparse.Namespace:
    from signals.meta_context.meta_ranker.run_4h_loop import build_parser

    argv: list[str] = []
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value is not False and value is not None:
            argv.extend([flag, str(value)])
    return build_parser().parse_args(argv)


def test_the_loop_verifies_the_matrix_the_runner_actually_reads() -> None:
    """The freshness check must point at the real artifact.

    A wrong path would fail the postcondition forever and silently stop the
    live loop from ever trading again.
    """

    from signals.meta_context.meta_ranker.live_runner import DEFAULT_MATRIX
    from signals.meta_context.meta_ranker.run_4h_loop import MATRIX_PATH
    from signals.meta_context.meta_ranker.update_meta_matrix import MATRIX as WRITTEN

    assert MATRIX_PATH == WRITTEN, "the loop must verify the file the matrix job writes"
    assert MATRIX_PATH == DEFAULT_MATRIX, "and the file the runner scores from"


def test_a_missing_matrix_fails_the_postcondition() -> None:
    from signals.meta_context.meta_ranker import run_4h_loop

    fresh, detail, counts = run_4h_loop._matrix_is_fresh()
    assert isinstance(fresh, bool)
    if not run_4h_loop.MATRIX_PATH.exists():
        assert "does not exist" in detail


def test_the_loop_rejects_the_legacy_live_flag() -> None:
    from signals.meta_context.meta_ranker.run_4h_loop import main

    assert main(["--mode", "equity", "--live"]) == 2


def test_the_runner_does_not_execute_after_a_failed_stage() -> None:
    from signals.meta_context.meta_ranker.run_4h_loop import run_once

    calls: list[str] = []
    result = run_once(
        loop_args(mode="equity", submit=True),
        runner=lambda: (calls.append("runner"), 0)[1],
        guard_factory=_ok_guard,
        bars=lambda: 1,
        feeds=lambda: 0,
        matrix=lambda: 0,
        matrix_freshness=lambda: (True, "fresh", {}),
    )

    assert calls == [], "a failed bars stage must stop the runner"
    assert result.runner_ok is False
    assert result.submitted is False
    assert result.stage("runner").status is StageStatus.NOT_RUN
    assert result.exit_code == 1


def test_a_blocked_guard_stops_the_pass() -> None:
    from signals.meta_context.meta_ranker.run_4h_loop import run_once

    calls: list[str] = []
    result = run_once(
        loop_args(mode="equity", submit=True),
        runner=lambda: (calls.append("runner"), 0)[1],
        guard_factory=_blocked_guard,
    )

    assert calls == []
    assert result.runner_ok is False
    assert "guard blocked" in result.blocked_reason


def test_a_stale_matrix_stops_the_runner() -> None:
    from signals.meta_context.meta_ranker.run_4h_loop import run_once

    calls: list[str] = []
    result = run_once(
        loop_args(mode="equity", submit=True),
        runner=lambda: (calls.append("runner"), 0)[1],
        guard_factory=_ok_guard,
        bars=lambda: 0,
        feeds=lambda: 0,
        matrix=lambda: 0,
        matrix_freshness=lambda: (False, "matrix is 9h old", {"matrix_age_sec": 32400}),
    )

    assert calls == [], "the runner must not score a stale matrix"
    assert result.stage("matrix").status is StageStatus.FAILED


def test_a_clean_pass_runs_the_runner() -> None:
    from signals.meta_context.meta_ranker.run_4h_loop import run_once

    calls: list[str] = []
    result = run_once(
        loop_args(mode="equity", submit=True),
        runner=lambda: (calls.append("runner"), 0)[1],
        guard_factory=_ok_guard,
        bars=lambda: 0,
        feeds=lambda: 0,
        matrix=lambda: 0,
        matrix_freshness=lambda: (True, "fresh", {}),
    )

    assert calls == ["runner"]
    assert result.runner_ok is True
    assert result.submitted is True
    assert result.exit_code == 0


class _Guard:
    def __init__(self, ok: bool, reason: str = "") -> None:
        self.ok = ok
        self.reason = reason

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def _ok_guard():
    return _Guard(True)


def _blocked_guard():
    return _Guard(False, "live window")


# --------------------------------------------------------------------------
# Outbox and job leases (PostgreSQL only)
# --------------------------------------------------------------------------


@pytest.fixture
def operations(pg_session):
    return OperationsRepository(pg_session)


@pytest.fixture
def clean_outbox(session_factory):
    """Committed outbox rows outlive a rolled-back session, so wipe explicitly.

    Without this, a leftover row claimed until NOW+60s is invisible to a later
    run using the same frozen clock, and the disjoint-claim test starves.
    """

    from sqlalchemy import text

    def wipe() -> None:
        with UnitOfWork(session_factory) as uow:
            uow.session.execute(text("DELETE FROM nervous_system.outbox_events"))
            uow.commit()

    wipe()
    yield
    wipe()


def test_enqueue_is_deterministic_and_idempotent(operations) -> None:
    first = operations.enqueue(
        event_type="DecisionRecorded",
        aggregate_type="DecisionRecord",
        aggregate_id="abc",
        payload={"n": 1},
        created_at=NOW,
    )
    second = operations.enqueue(
        event_type="DecisionRecorded",
        aggregate_type="DecisionRecord",
        aggregate_id="abc",
        payload={"n": 1},
        created_at=NOW,
    )

    assert first.outbox_event_id == second.outbox_event_id
    assert first.event_hash == second.event_hash


def test_a_future_row_is_not_claimed(operations) -> None:
    operations.enqueue(
        event_type="Later",
        aggregate_type="DecisionRecord",
        aggregate_id="future",
        payload={},
        created_at=NOW,
        available_at=NOW + timedelta(hours=1),
    )

    claimed = operations.claim(worker_id="w1", now=NOW, lease_seconds=60, limit=10)

    assert all(item.aggregate_id != "future" for item in claimed)


def test_two_workers_claim_disjoint_rows(session_factory, clean_outbox) -> None:
    """SKIP LOCKED means concurrent workers never take the same row."""

    with UnitOfWork(session_factory) as seed:
        for index in range(4):
            seed.operations.enqueue(
                event_type="Fanout",
                aggregate_type="DecisionRecord",
                aggregate_id=f"disjoint-{index}",
                payload={"i": index},
                created_at=NOW,
            )
        seed.commit()

    with UnitOfWork(session_factory) as first, UnitOfWork(session_factory) as second:
        a = first.operations.claim(worker_id="w1", now=NOW, lease_seconds=60, limit=2)
        b = second.operations.claim(worker_id="w2", now=NOW, lease_seconds=60, limit=2)
        ids_a = {item.outbox_event_id for item in a}
        ids_b = {item.outbox_event_id for item in b}
        first.commit()
        second.commit()

    assert ids_a and ids_b
    assert ids_a.isdisjoint(ids_b)


def test_a_stale_token_cannot_finalize(operations) -> None:
    event = operations.enqueue(
        event_type="Fenced",
        aggregate_type="DecisionRecord",
        aggregate_id="fenced",
        payload={},
        created_at=NOW,
    )
    claimed = operations.claim(worker_id="w1", now=NOW, lease_seconds=60, limit=10)
    target = next(item for item in claimed if item.outbox_event_id == event.outbox_event_id)

    assert not operations.mark_delivered(
        target.outbox_event_id,
        worker_id="w1",
        claim_token="stale-token",
        delivered_at=NOW,
    )
    assert operations.mark_delivered(
        target.outbox_event_id,
        worker_id="w1",
        claim_token=target.claim_token,
        delivered_at=NOW,
    )


def test_a_failed_delivery_is_released_for_retry(operations) -> None:
    event = operations.enqueue(
        event_type="Retryable",
        aggregate_type="DecisionRecord",
        aggregate_id="retry",
        payload={},
        created_at=NOW,
    )
    claimed = operations.claim(worker_id="w1", now=NOW, lease_seconds=60, limit=10)
    target = next(item for item in claimed if item.outbox_event_id == event.outbox_event_id)

    assert operations.mark_failed(
        target.outbox_event_id,
        worker_id="w1",
        claim_token=target.claim_token,
        error="handler blew up",
        retry_at=NOW,
    )
    again = operations.claim(worker_id="w2", now=NOW + timedelta(seconds=1), lease_seconds=60, limit=10)
    assert any(item.outbox_event_id == event.outbox_event_id for item in again)


def test_job_slot_uniqueness_and_lease(operations) -> None:
    kwargs = dict(
        job_type="meta_4h_loop",
        scheduled_for=NOW,
        config_hash="c" * 64,
        now=NOW,
        lease_seconds=300,
        host="host-a",
        revision="rev-1",
    )
    first, claimed_first = operations.claim_job(owner="w1", lease_token="t1", **kwargs)
    second, claimed_second = operations.claim_job(owner="w2", lease_token="t2", **kwargs)

    assert claimed_first is True
    assert claimed_second is False, "one scheduled slot, one runner"
    assert first.job_run_id == second.job_run_id
    assert first.host == "host-a"
    assert first.revision == "rev-1"


def test_an_expired_job_lease_is_recovered_with_an_event(operations) -> None:
    kwargs = dict(
        job_type="meta_4h_loop",
        scheduled_for=NOW,
        config_hash="d" * 64,
        host="host-a",
        revision="rev-1",
    )
    first, _ = operations.claim_job(
        owner="w1", lease_token="t1", now=NOW, lease_seconds=1, **kwargs
    )
    later = NOW + timedelta(minutes=10)
    second, claimed = operations.claim_job(
        owner="w2", lease_token="t2", now=later, lease_seconds=300, **kwargs
    )

    assert claimed is True
    assert second.job_run_id == first.job_run_id
    assert second.attempt_no == 2
    statuses = [status for status, _ in operations.job_events(first.job_run_id)]
    assert JobStatus.RECOVERED.value in statuses, "a takeover is recorded, not silent"


def test_a_heartbeat_requires_the_current_token(operations) -> None:
    record, _ = operations.claim_job(
        job_type="meta_4h_loop",
        scheduled_for=NOW,
        config_hash="e" * 64,
        owner="w1",
        lease_token="t1",
        now=NOW,
        lease_seconds=300,
        host="host-a",
        revision="rev-1",
    )

    assert operations.heartbeat_job(
        record.job_run_id, lease_token="t1", now=NOW, lease_seconds=300
    )
    assert not operations.heartbeat_job(
        record.job_run_id, lease_token="stale", now=NOW, lease_seconds=300
    )


def test_finishing_records_counts_and_requires_the_token(operations) -> None:
    record, _ = operations.claim_job(
        job_type="meta_4h_loop",
        scheduled_for=NOW,
        config_hash="f" * 64,
        owner="w1",
        lease_token="t1",
        now=NOW,
        lease_seconds=300,
        host="host-a",
        revision="rev-1",
    )

    assert not operations.finish_job(
        record.job_run_id, lease_token="stale", status="SUCCEEDED", finished_at=NOW
    )
    assert operations.finish_job(
        record.job_run_id,
        lease_token="t1",
        status="SUCCEEDED",
        finished_at=NOW,
        counts={"matrix": {"rows": 10}},
    )
    stored = operations.get_job(record.job_run_id)
    assert stored.status == "SUCCEEDED"
    assert stored.counts == {"matrix": {"rows": 10}}
