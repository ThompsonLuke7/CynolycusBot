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
