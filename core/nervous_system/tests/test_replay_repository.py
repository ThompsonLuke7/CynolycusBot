"""Replay persistence, and the append-only outcome guarantee (Task 24).

The schema constraint alone is not the guarantee — it only rejects a bad write
if some code tries one. These exercise the writer against real PostgreSQL.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest

from core.nervous_system.contracts.replay import (
    MarkType,
    SideFitnessMetrics,
    SourceFitnessStatus,
    SourceFitnessThresholds,
)
from core.nervous_system.persistence.repositories.replay import (
    ReplayRepository,
    option_pnl_is_permitted,
)
from core.nervous_system.replay.fitness import evaluate_source_fitness


NOW = datetime(2026, 8, 3, 20, 0, tzinfo=timezone.utc)
BAR = NOW - timedelta(minutes=5)


def _run_kwargs(**updates):
    payload = {
        "replay_run_id": uuid4(),
        "source_manifest_hash": "a" * 64,
        "schedule_hash": "b" * 64,
        "config_hash": "c" * 64,
        "model_hash": "d" * 64,
        "deterministic_seed": 4242,
        "execution_assumptions": {"fill": "next_bar_open"},
        "started_at": NOW,
    }
    payload.update(updates)
    return payload


@pytest.fixture
def repo(pg_session):
    return ReplayRepository(pg_session)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def test_a_run_is_persisted_with_its_seed(repo) -> None:
    row = repo.start_run(**_run_kwargs())

    assert row.deterministic_seed == 4242
    assert row.status == "RUNNING"


def test_re_running_the_same_configuration_converges_on_one_run(repo) -> None:
    """Otherwise repeated runs accumulate near-duplicates nobody can tell apart."""

    first = repo.start_run(**_run_kwargs())
    second = repo.start_run(**_run_kwargs(replay_run_id=uuid4()))

    assert second.replay_run_id == first.replay_run_id


def test_a_different_seed_is_a_different_run(repo) -> None:
    first = repo.start_run(**_run_kwargs())
    second = repo.start_run(**_run_kwargs(replay_run_id=uuid4(), deterministic_seed=99))

    assert second.replay_run_id != first.replay_run_id


def test_finishing_records_limitations_even_on_success(repo) -> None:
    """A run that completed with known gaps is not the same as one with none."""

    run = repo.start_run(**_run_kwargs())

    finished = repo.finish_run(
        run.replay_run_id, status="COMPLETE", completed_at=NOW,
        limitations=["option quotes missing for 3 sessions"],
    )

    assert finished.limitations == ["option quotes missing for 3 sessions"]


def test_a_finished_run_cannot_be_finished_again(repo) -> None:
    run = repo.start_run(**_run_kwargs())
    repo.finish_run(run.replay_run_id, status="COMPLETE", completed_at=NOW)

    with pytest.raises(ValueError, match="already"):
        repo.finish_run(run.replay_run_id, status="FAILED", completed_at=NOW)


# ---------------------------------------------------------------------------
# Fitness verdicts
# ---------------------------------------------------------------------------


def _metrics(**updates):
    payload = {
        "option_type": "CALL",
        "mark_type": MarkType.QUOTE_BID_ASK,
        "matched_positions": 40,
        "sessions": 15,
        "valid_quote_fraction": Decimal("0.99"),
        "identical_mark_fraction": Decimal("0.01"),
        "pearson": Decimal("0.91"),
        "spearman": Decimal("0.88"),
        "max_quote_age_seconds": Decimal("30"),
        "entitlement_verified": True,
    }
    payload.update(updates)
    return SideFitnessMetrics(**payload)


def _report(**updates):
    return evaluate_source_fitness(
        sides=updates.pop("sides", (_metrics(), _metrics(option_type="PUT"))),
        thresholds=SourceFitnessThresholds(),
        source="alpaca", feed="opra", tier="indicative",
    )


def test_a_fitness_verdict_persists_with_the_bar_it_was_judged_against(repo) -> None:
    report = _report()

    row = repo.save_fitness_report(
        report, source_fitness_report_id=uuid4(), evaluated_at=NOW
    )

    assert row.status == SourceFitnessStatus.FIT_FOR_OPTION_PNL.value
    assert row.thresholds_hash == report.thresholds_hash
    assert len(row.side_metrics) == 2


def test_an_unfit_verdict_persists_its_reasons(repo) -> None:
    report = _report(sides=(_metrics(pearson=Decimal("0.09")), _metrics(option_type="PUT")))

    row = repo.save_fitness_report(
        report, source_fitness_report_id=uuid4(), evaluated_at=NOW
    )

    assert row.status == SourceFitnessStatus.SOURCE_UNFIT_FOR_OPTION_PNL.value
    assert "LOW_DERIVATIVE_CORRELATION" in row.reason_codes


# ---------------------------------------------------------------------------
# Outcomes append; they never rewrite
# ---------------------------------------------------------------------------


def _decision(pg_session):
    """One decision record to hang outcomes off.

    Recorded as FAILED because a COMPLETE record requires the whole snapshot /
    intent / policy chain, and these tests are about outcome revisions rather
    than about rebuilding that chain.
    """

    from core.nervous_system.persistence.models import DecisionRecord as Row

    decision_id = uuid4()
    pg_session.add(
        Row(
            decision_record_id=decision_id,
            decision_time=NOW,
            status="FAILED",
            failure_stage="test",
            failure_reason="fixture",
            content_hash=uuid5(NAMESPACE_URL, str(decision_id)).hex.ljust(64, "0")[:64],
            payload={},
            created_at=NOW,
        )
    )
    pg_session.flush()
    return decision_id


def test_a_maturing_outcome_is_pending_not_zero(repo, pg_session) -> None:
    decision_id = _decision(pg_session)

    row = repo.append_outcome_revision(
        outcome_id=uuid4(), decision_record_id=decision_id, horizon="53x4h",
        evaluated_at=NOW, status="PENDING", payload={"realized_pnl": None},
    )

    assert row.status == "PENDING"
    assert row.revision_number == 1
    assert row.payload["realized_pnl"] is None


def test_a_later_evaluation_appends_a_revision(repo, pg_session) -> None:
    decision_id = _decision(pg_session)
    repo.append_outcome_revision(
        outcome_id=uuid4(), decision_record_id=decision_id, horizon="53x4h",
        evaluated_at=NOW, status="PENDING", payload={"realized_pnl": None},
    )

    second = repo.append_outcome_revision(
        outcome_id=uuid4(), decision_record_id=decision_id, horizon="53x4h",
        evaluated_at=NOW + timedelta(days=1), status="FINAL",
        payload={"realized_pnl": "998.00"},
    )

    revisions = repo.outcome_revisions(decision_id, "53x4h")
    assert [r.revision_number for r in revisions] == [1, 2]
    assert [r.status for r in revisions] == ["PENDING", "FINAL"]
    assert second.revision_number == 2


def test_the_earlier_revision_is_left_exactly_as_it_was(repo, pg_session) -> None:
    """This is the whole guarantee: what we believed at the time survives."""

    decision_id = _decision(pg_session)
    repo.append_outcome_revision(
        outcome_id=uuid4(), decision_record_id=decision_id, horizon="53x4h",
        evaluated_at=NOW, status="PENDING", payload={"realized_pnl": None},
    )
    repo.append_outcome_revision(
        outcome_id=uuid4(), decision_record_id=decision_id, horizon="53x4h",
        evaluated_at=NOW + timedelta(days=1), status="FINAL",
        payload={"realized_pnl": "998.00"},
    )

    first = repo.outcome_revisions(decision_id, "53x4h")[0]
    assert first.status == "PENDING"
    assert first.payload["realized_pnl"] is None


def test_a_settled_outcome_is_not_revised_again(repo, pg_session) -> None:
    """Re-measuring something already final changes history rather than
    extending it.
    """

    decision_id = _decision(pg_session)
    repo.append_outcome_revision(
        outcome_id=uuid4(), decision_record_id=decision_id, horizon="53x4h",
        evaluated_at=NOW, status="FINAL", payload={"realized_pnl": "1.00"},
    )

    with pytest.raises(ValueError, match="already FINAL"):
        repo.append_outcome_revision(
            outcome_id=uuid4(), decision_record_id=decision_id, horizon="53x4h",
            evaluated_at=NOW + timedelta(days=1), status="FINAL",
            payload={"realized_pnl": "2.00"},
        )


def test_different_horizons_revise_independently(repo, pg_session) -> None:
    decision_id = _decision(pg_session)
    repo.append_outcome_revision(
        outcome_id=uuid4(), decision_record_id=decision_id, horizon="1d",
        evaluated_at=NOW, status="FINAL", payload={},
    )

    row = repo.append_outcome_revision(
        outcome_id=uuid4(), decision_record_id=decision_id, horizon="53x4h",
        evaluated_at=NOW, status="PENDING", payload={},
    )

    assert row.revision_number == 1


def test_an_outcome_is_not_option_eligible_by_default(repo, pg_session) -> None:
    decision_id = _decision(pg_session)

    row = repo.append_outcome_revision(
        outcome_id=uuid4(), decision_record_id=decision_id, horizon="53x4h",
        evaluated_at=NOW, status="PENDING", payload={},
    )

    assert row.option_pnl_eligible is False


def test_an_unfit_source_leaves_underlying_only_outcomes_intact(repo, pg_session) -> None:
    """The underlying result is still real when option marks are not. Refusing
    to record it would throw away good evidence along with bad.
    """

    unfit = _report(sides=(_metrics(mark_type=MarkType.TRADE_PRINT), _metrics(option_type="PUT")))
    decision_id = _decision(pg_session)

    row = repo.append_outcome_revision(
        outcome_id=uuid4(), decision_record_id=decision_id, horizon="53x4h",
        evaluated_at=NOW, status="FINAL",
        payload={"underlying_movement": "1000.00"},
        option_pnl_eligible=option_pnl_is_permitted(unfit),
    )

    assert row.option_pnl_eligible is False
    assert row.payload["underlying_movement"] == "1000.00"


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def test_replay_decisions_come_back_in_schedule_order(repo, pg_session) -> None:
    run = repo.start_run(**_run_kwargs())
    for sequence in (3, 1, 2):
        repo.append_decision(
            replay_decision_id=uuid4(), replay_run_id=run.replay_run_id,
            sequence_no=sequence, decision_record_id=_decision(pg_session),
            snapshot_hash="e" * 64, decision_time=NOW, decision_bar=BAR, lineage={},
        )

    assert [d.sequence_no for d in repo.run_decisions(run.replay_run_id)] == [1, 2, 3]


def test_a_decision_taken_before_its_bar_is_refused(repo, pg_session) -> None:
    run = repo.start_run(**_run_kwargs())

    with pytest.raises(ValueError, match="before the bar"):
        repo.append_decision(
            replay_decision_id=uuid4(), replay_run_id=run.replay_run_id,
            sequence_no=1, decision_record_id=_decision(pg_session),
            snapshot_hash="e" * 64,
            decision_time=BAR, decision_bar=NOW, lineage={},
        )
