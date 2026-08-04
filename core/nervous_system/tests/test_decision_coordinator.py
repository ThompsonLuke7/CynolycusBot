"""Decision coordinator mode gating and stage artifacts (Task 22)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.nervous_system.contracts.enums import (
    PolicyAction,
    PolicyMode,
    RuntimeEnvironment,
)
from core.nervous_system.orchestration.coordinator import (
    STAGES,
    CoordinatorRefusal,
    DecisionCoordinator,
    downstream_not_run,
    not_run,
)
from core.nervous_system.tests.fixtures.gateway_harness import (
    NOW,
    clock,
    order_request,
)
from core.nervous_system.tests.test_policy_engine import (
    build_intent,
    build_snapshot,
)


class ExplodingGatewayFactory:
    """Any construction at all is a failure for the refusal paths."""

    def __init__(self) -> None:
        self.constructions = 0

    def __call__(self):
        self.constructions += 1
        raise AssertionError("no gateway may be constructed on this path")


class RecordingGateway:
    def __init__(self) -> None:
        self.submits: list = []

    def submit(self, *, decision, request):
        self.submits.append(request)
        return "submitted"


def coordinator(
    *,
    environment=RuntimeEnvironment.QA_PAPER,
    gateway_factory=None,
):
    return DecisionCoordinator(
        environment=environment,
        unit_of_work_factory=lambda: None,
        clock=clock(NOW),
        gateway_factory=gateway_factory,
    )


def an_intent():
    snapshot = build_snapshot()
    return build_intent(snapshot=snapshot)


class FakePolicy:
    def __init__(self, action: PolicyAction) -> None:
        self.action = action


class FakeSelection:
    def __init__(self, outcome: str) -> None:
        self.outcome = type("O", (), {"value": outcome})()


# --------------------------------------------------------------------------
# Mode gating
# --------------------------------------------------------------------------


def test_production_live_is_vetoed_before_any_gateway_construction() -> None:
    factory = ExplodingGatewayFactory()
    coord = coordinator(
        environment=RuntimeEnvironment.PRODUCTION_LIVE, gateway_factory=factory
    )

    outcome = coord.process_intent(
        an_intent(), policy_mode=PolicyMode.ENFORCE, submit=True
    )

    assert outcome.refusal is CoordinatorRefusal.PRODUCTION_LIVE
    assert outcome.gateway_invoked is False
    assert factory.constructions == 0


def test_off_mode_refuses_submit_rather_than_downgrading() -> None:
    factory = ExplodingGatewayFactory()
    coord = coordinator(gateway_factory=factory)

    outcome = coord.process_intent(
        an_intent(), policy_mode=PolicyMode.OFF, submit=True
    )

    assert outcome.refusal is CoordinatorRefusal.OFF_MODE_SUBMIT
    assert outcome.submitted is False
    assert factory.constructions == 0


def test_off_mode_without_submit_plans_without_a_gateway() -> None:
    factory = ExplodingGatewayFactory()
    coord = coordinator(gateway_factory=factory)

    outcome = coord.process_intent(
        an_intent(), policy_mode=PolicyMode.OFF, submit=False
    )

    assert outcome.refusal is CoordinatorRefusal.DRY_RUN
    assert factory.constructions == 0


def test_a_dry_run_completes_planning_and_stops_before_the_gateway() -> None:
    factory = ExplodingGatewayFactory()
    coord = coordinator(gateway_factory=factory)

    outcome = coord.process_intent(
        an_intent(), policy_mode=PolicyMode.ENFORCE, submit=False
    )

    assert outcome.refusal is CoordinatorRefusal.DRY_RUN
    assert outcome.submitted is False
    assert factory.constructions == 0


@pytest.mark.parametrize("mode", [PolicyMode.SHADOW, PolicyMode.ENFORCE])
def test_submitting_modes_reach_the_gateway_in_qa_paper(mode) -> None:
    gateway = RecordingGateway()
    coord = coordinator(gateway_factory=lambda: gateway)
    request = order_request()

    outcome = coord.process_intent(
        an_intent(),
        policy_mode=mode,
        submit=True,
        policy_decision=FakePolicy(PolicyAction.APPROVE),
        selection=FakeSelection("SELECTED_OPTION"),
        order_request=request,
    )

    assert outcome.submitted is True
    assert outcome.gateway_invoked is True
    assert gateway.submits == [request]


@pytest.mark.parametrize("action", [PolicyAction.REJECT, PolicyAction.DEFER])
def test_a_policy_veto_stops_before_the_gateway(action) -> None:
    factory = ExplodingGatewayFactory()
    coord = coordinator(gateway_factory=factory)

    outcome = coord.process_intent(
        an_intent(),
        policy_mode=PolicyMode.ENFORCE,
        submit=True,
        policy_decision=FakePolicy(action),
        order_request=order_request(),
    )

    assert outcome.refusal is CoordinatorRefusal.POLICY_VETO
    assert factory.constructions == 0


def test_no_eligible_instrument_stops_before_the_gateway() -> None:
    factory = ExplodingGatewayFactory()
    coord = coordinator(gateway_factory=factory)

    outcome = coord.process_intent(
        an_intent(),
        policy_mode=PolicyMode.ENFORCE,
        submit=True,
        policy_decision=FakePolicy(PolicyAction.APPROVE),
        selection=FakeSelection("NO_ELIGIBLE_INSTRUMENT"),
        order_request=order_request(),
    )

    assert outcome.refusal is CoordinatorRefusal.NO_ELIGIBLE_INSTRUMENT
    assert factory.constructions == 0


def test_may_submit_is_decidable_before_any_work() -> None:
    coord = coordinator()

    assert coord.may_submit(policy_mode=PolicyMode.ENFORCE, submit=True) == (True, None)
    assert coord.may_submit(policy_mode=PolicyMode.ENFORCE, submit=False)[0] is False
    assert coord.may_submit(policy_mode=PolicyMode.OFF, submit=True) == (
        False,
        CoordinatorRefusal.OFF_MODE_SUBMIT,
    )
    live = coordinator(environment=RuntimeEnvironment.PRODUCTION_LIVE)
    assert live.may_submit(policy_mode=PolicyMode.ENFORCE, submit=True) == (
        False,
        CoordinatorRefusal.PRODUCTION_LIVE,
    )


# --------------------------------------------------------------------------
# NOT_RUN artifacts
# --------------------------------------------------------------------------


def test_a_blocked_stage_and_everything_after_it_are_not_run() -> None:
    artifacts = {
        "RAW_STRATEGY_OUTPUT": not_run("RAW_STRATEGY_OUTPUT", "seed"),
    }
    downstream_not_run(artifacts, "INSTRUMENT_CANDIDATES", "policy vetoed")

    assert artifacts["INSTRUMENT_CANDIDATES"].payload["status"] == "NOT_RUN"
    assert artifacts["INSTRUMENT_SELECTION"].payload["status"] == "NOT_RUN"
    assert "EXPOSURE_REPORT" not in artifacts, "earlier stages keep their real result"


def test_a_not_run_artifact_carries_its_stage_and_reason() -> None:
    artifact = not_run("EXPOSURE_REPORT", "policy REJECT")

    assert artifact.payload["status"] == "NOT_RUN"
    assert artifact.payload["blocking_stage"] == "EXPOSURE_REPORT"
    assert artifact.payload["reason"] == "policy REJECT"


def test_the_stage_order_is_fixed() -> None:
    assert STAGES == (
        "RAW_STRATEGY_OUTPUT",
        "EXPOSURE_REPORT",
        "INSTRUMENT_CANDIDATES",
        "INSTRUMENT_SELECTION",
    )


# --------------------------------------------------------------------------
# The atomic planning transaction (real PostgreSQL)
# --------------------------------------------------------------------------


def _chain_pieces():
    """Build a self-consistent snapshot/intent/policy set with a fresh identity.

    The disposable database keeps committed rows between runs, so a fixed
    intent id would make the second run converge on an existing decision and
    mask what these tests are checking.
    """

    import uuid as _uuid

    from core.nervous_system.policy.engine import evaluate_policy
    from core.nervous_system.tests.test_policy_engine import build_config

    snapshot = build_snapshot()
    intent = build_intent(snapshot=snapshot, intent_id=_uuid.uuid4())
    policy = evaluate_policy(intent, snapshot, build_config())
    return snapshot, intent, policy


def test_planning_persists_the_whole_chain_and_one_outbox_event(
    session_factory, pg_session
) -> None:
    from sqlalchemy import text

    from core.nervous_system.persistence.uow import UnitOfWork

    snapshot, intent, policy = _chain_pieces()
    assert policy.action is PolicyAction.APPROVE

    gateway = RecordingGateway()
    coord = DecisionCoordinator(
        environment=RuntimeEnvironment.QA_PAPER,
        unit_of_work_factory=lambda: UnitOfWork(session_factory),
        clock=clock(NOW),
        gateway_factory=lambda: gateway,
    )

    outcome = coord.process_intent(
        intent,
        policy_mode=PolicyMode.ENFORCE,
        submit=False,
        snapshot=snapshot,
        policy_decision=policy,
    )

    assert outcome.decision is not None, "a dry run still persists the full chain"
    assert outcome.decision.status == "COMPLETE"
    assert outcome.submitted is False
    assert gateway.submits == [], "planning must not reach the gateway"

    with UnitOfWork(session_factory) as uow:
        stored = uow.session.execute(
            text(
                "SELECT status FROM nervous_system.decision_records "
                "WHERE decision_record_id = :id"
            ),
            {"id": str(outcome.decision.decision_record_id)},
        ).first()
        events = uow.session.execute(
            text(
                "SELECT count(*) FROM nervous_system.outbox_events "
                "WHERE aggregate_id = :id"
            ),
            {"id": str(outcome.decision.decision_record_id)},
        ).scalar()
    assert stored is not None and stored[0] == "COMPLETE"
    assert events == 1, "the decision and its outbox event commit together"


def test_replanning_the_same_intent_converges(session_factory) -> None:
    """A retry after a crash must return the existing decision, not explode.

    The chain is inserted with plain writes, so without a convergence check a
    replay raises a unique-key violation and the intent is stuck forever.
    """

    from sqlalchemy import text

    from core.nervous_system.persistence.uow import UnitOfWork

    snapshot, intent, policy = _chain_pieces()
    coord = DecisionCoordinator(
        environment=RuntimeEnvironment.QA_PAPER,
        unit_of_work_factory=lambda: UnitOfWork(session_factory),
        clock=clock(NOW),
    )
    kwargs = dict(
        policy_mode=PolicyMode.ENFORCE,
        submit=False,
        snapshot=snapshot,
        policy_decision=policy,
    )

    first = coord.process_intent(intent, **kwargs)
    second = coord.process_intent(intent, **kwargs)

    assert first.decision is not None and second.decision is not None
    assert first.decision.decision_record_id == second.decision.decision_record_id

    with UnitOfWork(session_factory) as uow:
        decisions = uow.session.execute(
            text(
                "SELECT count(*) FROM nervous_system.decision_records "
                "WHERE decision_record_id = :id"
            ),
            {"id": str(first.decision.decision_record_id)},
        ).scalar()
    assert decisions == 1, "one logical decision, one row"


def test_an_early_failure_records_no_dangling_lineage(session_factory) -> None:
    import uuid as _uuid

    from core.nervous_system.persistence.uow import UnitOfWork

    # A unique message keeps this independent of anything already in the
    # disposable database, while still exercising convergence within the run.
    message = f"market state unavailable {_uuid.uuid4()}"
    coord = DecisionCoordinator(
        environment=RuntimeEnvironment.QA_PAPER,
        unit_of_work_factory=lambda: UnitOfWork(session_factory),
        clock=clock(NOW),
    )

    record = coord.record_failure(
        failure_stage="SNAPSHOT",
        failure_code="REQUIRED_STATE_MISSING",
        failure_message=message,
    )

    assert record.status == "FAILED"
    assert record.snapshot_id is None
    assert record.intent_id is None
    assert record.policy_decision_id is None

    with UnitOfWork(session_factory) as uow:
        stored = uow.decisions.get_decision_record(record.decision_record_id)
    assert stored is not None
    assert stored.failure_stage == "SNAPSHOT"

    # Retrying the same failure converges rather than colliding on the
    # unique content hash.
    again = coord.record_failure(
        failure_stage="SNAPSHOT",
        failure_code="REQUIRED_STATE_MISSING",
        failure_message=message,
    )
    assert again.decision_record_id == record.decision_record_id
