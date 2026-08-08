"""Typed persistence for replay runs, fitness verdicts, and outcome revisions.

The one rule that shapes this module: outcomes are append-only. A later
evaluation adds a revision; it never rewrites an earlier one, because
rewriting what we believed at the time is the thing an audit trail may not do.
A maturing horizon is PENDING, never zero.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.nervous_system.contracts.replay import (
    SourceFitnessReport,
    SourceFitnessStatus,
)
from core.nervous_system.persistence.models import (
    DecisionOutcome as DecisionOutcomeRow,
    ReplayDecision as ReplayDecisionRow,
    ReplayRun as ReplayRunRow,
    SourceFitnessReport as SourceFitnessReportRow,
)


TERMINAL_OUTCOME_STATUSES = frozenset({"FINAL", "UNAVAILABLE"})


class ReplayRepository:
    def __init__(self, session: Session):
        self._session = session

    # -- runs ---------------------------------------------------------------

    def start_run(
        self,
        *,
        replay_run_id: UUID,
        source_manifest_hash: str,
        schedule_hash: str,
        config_hash: str,
        model_hash: str,
        deterministic_seed: int,
        execution_assumptions: dict[str, Any],
        started_at: datetime,
    ) -> ReplayRunRow:
        """Open a run, or return the existing one with the same identity.

        Identity includes the seed, so re-running the same configuration
        converges on one row instead of accumulating near-duplicates that
        cannot be told apart afterwards.
        """

        existing = self._session.execute(
            select(ReplayRunRow).where(
                ReplayRunRow.source_manifest_hash == source_manifest_hash,
                ReplayRunRow.schedule_hash == schedule_hash,
                ReplayRunRow.config_hash == config_hash,
                ReplayRunRow.model_hash == model_hash,
                ReplayRunRow.deterministic_seed == deterministic_seed,
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing
        row = ReplayRunRow(
            replay_run_id=replay_run_id,
            source_manifest_hash=source_manifest_hash,
            schedule_hash=schedule_hash,
            config_hash=config_hash,
            model_hash=model_hash,
            deterministic_seed=deterministic_seed,
            execution_assumptions=execution_assumptions,
            status="RUNNING",
            limitations=[],
            started_at=started_at,
            created_at=started_at,
        )
        self._session.add(row)
        self._session.flush()
        return row

    def finish_run(
        self,
        replay_run_id: UUID,
        *,
        status: str,
        completed_at: datetime,
        limitations: list[str] | None = None,
    ) -> ReplayRunRow:
        row = self._session.get(ReplayRunRow, replay_run_id)
        if row is None:
            raise ValueError("cannot finish an unknown replay run")
        if row.status != "RUNNING":
            raise ValueError(f"replay run is already {row.status}")
        row.status = status
        row.completed_at = completed_at
        # Limitations are recorded even on success: a run that completed with
        # known gaps is not the same as one that had none.
        row.limitations = list(limitations or [])
        self._session.flush()
        return row

    # -- decisions ----------------------------------------------------------

    def append_decision(
        self,
        *,
        replay_decision_id: UUID,
        replay_run_id: UUID,
        sequence_no: int,
        decision_record_id: UUID,
        snapshot_hash: str,
        decision_time: datetime,
        decision_bar: datetime,
        lineage: dict[str, Any],
    ) -> ReplayDecisionRow:
        if decision_bar > decision_time:
            raise ValueError("a decision cannot be taken before the bar it acted on")
        row = ReplayDecisionRow(
            replay_decision_id=replay_decision_id,
            replay_run_id=replay_run_id,
            sequence_no=sequence_no,
            decision_record_id=decision_record_id,
            snapshot_hash=snapshot_hash,
            decision_time=decision_time,
            decision_bar=decision_bar,
            lineage=lineage,
            created_at=decision_time,
        )
        self._session.add(row)
        self._session.flush()
        return row

    def run_decisions(self, replay_run_id: UUID) -> tuple[ReplayDecisionRow, ...]:
        rows = self._session.execute(
            select(ReplayDecisionRow)
            .where(ReplayDecisionRow.replay_run_id == replay_run_id)
            .order_by(ReplayDecisionRow.sequence_no)
        ).scalars()
        return tuple(rows)

    # -- fitness ------------------------------------------------------------

    def save_fitness_report(
        self,
        report: SourceFitnessReport,
        *,
        source_fitness_report_id: UUID,
        evaluated_at: datetime,
        replay_run_id: UUID | None = None,
    ) -> SourceFitnessReportRow:
        row = SourceFitnessReportRow(
            source_fitness_report_id=source_fitness_report_id,
            replay_run_id=replay_run_id,
            status=report.status.value,
            thresholds_hash=report.thresholds_hash,
            source=report.source,
            feed=report.feed,
            tier=report.tier,
            side_metrics=[side.model_dump(mode="json") for side in report.sides],
            reason_codes=[reason.value for reason in report.reasons],
            warnings=[reason.value for reason in report.warnings],
            evaluated_at=evaluated_at,
            created_at=evaluated_at,
        )
        self._session.add(row)
        self._session.flush()
        return row

    # -- outcomes -----------------------------------------------------------

    def latest_outcome(
        self, decision_record_id: UUID, horizon: str
    ) -> DecisionOutcomeRow | None:
        return self._session.execute(
            select(DecisionOutcomeRow)
            .where(
                DecisionOutcomeRow.decision_record_id == decision_record_id,
                DecisionOutcomeRow.horizon == horizon,
            )
            .order_by(DecisionOutcomeRow.revision_number.desc())
        ).scalars().first()

    def append_outcome_revision(
        self,
        *,
        outcome_id: UUID,
        decision_record_id: UUID,
        horizon: str,
        evaluated_at: datetime,
        status: str,
        payload: dict[str, Any],
        option_pnl_eligible: bool = False,
        source_fitness_report_id: UUID | None = None,
        replay_run_id: UUID | None = None,
        horizon_kind: str = "BARS",
        target_window_start: datetime | None = None,
        target_window_end: datetime | None = None,
        mark_basis: str | None = None,
        fill_basis: str | None = None,
        source_observation_hashes: list[str] | None = None,
    ) -> DecisionOutcomeRow:
        """Append the next revision for one (decision, horizon).

        Never updates a prior row. A settled outcome is not revised again:
        re-measuring something already final would change history rather than
        extend it.
        """

        previous = self.latest_outcome(decision_record_id, horizon)
        if previous is not None and previous.status in TERMINAL_OUTCOME_STATUSES:
            raise ValueError(
                f"outcome for {decision_record_id} @{horizon} is already "
                f"{previous.status}; a settled outcome is not revised"
            )
        row = DecisionOutcomeRow(
            outcome_id=outcome_id,
            decision_record_id=decision_record_id,
            evaluated_at=evaluated_at,
            horizon=horizon,
            payload=payload,
            created_at=evaluated_at,
            replay_run_id=replay_run_id,
            source_fitness_report_id=source_fitness_report_id,
            horizon_kind=horizon_kind,
            target_window_start=target_window_start,
            target_window_end=target_window_end,
            mark_basis=mark_basis,
            fill_basis=fill_basis,
            source_observation_hashes=source_observation_hashes or [],
            revision_number=1 if previous is None else previous.revision_number + 1,
            status=status,
            # Only an affirmatively fit source may produce option P&L.
            option_pnl_eligible=bool(option_pnl_eligible),
        )
        self._session.add(row)
        self._session.flush()
        return row

    def outcome_revisions(
        self, decision_record_id: UUID, horizon: str
    ) -> tuple[DecisionOutcomeRow, ...]:
        rows = self._session.execute(
            select(DecisionOutcomeRow)
            .where(
                DecisionOutcomeRow.decision_record_id == decision_record_id,
                DecisionOutcomeRow.horizon == horizon,
            )
            .order_by(DecisionOutcomeRow.revision_number)
        ).scalars()
        return tuple(rows)


def option_pnl_is_permitted(report: SourceFitnessReport) -> bool:
    """Underlying-only outcomes survive an unfit source; option P&L does not."""

    return report.status is SourceFitnessStatus.FIT_FOR_OPTION_PNL


__all__ = ["ReplayRepository", "TERMINAL_OUTCOME_STATUSES", "option_pnl_is_permitted"]
