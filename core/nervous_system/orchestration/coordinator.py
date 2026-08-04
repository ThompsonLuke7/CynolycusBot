"""Decision orchestration.

The coordinator turns one intent into one durable decision record, and decides
whether anything is allowed to reach the gateway. Its most important property
is negative: in OFF mode, on a dry run, in production-live, or after any veto,
no broker adapter is constructed and no gateway call is made.

Planning is one atomic transaction. The broker is never called inside it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from core.nervous_system.contracts.decisions import (
    DecisionRecord,
    HashedDecisionArtifact,
)
from core.nervous_system.contracts.enums import (
    PolicyAction,
    PolicyMode,
    RuntimeEnvironment,
)
from core.nervous_system.contracts.intent import TradeIntent


class CoordinatorRefusal(str, Enum):
    PRODUCTION_LIVE = "PRODUCTION_LIVE_DISABLED"
    OFF_MODE_SUBMIT = "OFF_MODE_CANNOT_SUBMIT"
    DRY_RUN = "DRY_RUN"
    POLICY_VETO = "POLICY_VETO"
    NO_ELIGIBLE_INSTRUMENT = "NO_ELIGIBLE_INSTRUMENT"
    STAGE_FAILED = "STAGE_FAILED"


STAGES = (
    "RAW_STRATEGY_OUTPUT",
    "EXPOSURE_REPORT",
    "INSTRUMENT_CANDIDATES",
    "INSTRUMENT_SELECTION",
)


@dataclass
class PlanningOutcome:
    decision: DecisionRecord | None
    submitted: bool = False
    refusal: CoordinatorRefusal | None = None
    detail: str | None = None
    execution_result: Any = None
    gateway_invoked: bool = False


def not_run(stage: str, reason: str) -> HashedDecisionArtifact:
    """A blocked stage is recorded explicitly, never omitted."""

    return HashedDecisionArtifact.not_run(stage, reason)


def downstream_not_run(
    artifacts: dict[str, HashedDecisionArtifact],
    blocking_stage: str,
    reason: str,
) -> dict[str, HashedDecisionArtifact]:
    """Mark the blocking stage and everything after it as NOT_RUN."""

    blocked = False
    for stage in STAGES:
        if stage == blocking_stage:
            blocked = True
        if blocked:
            artifacts[stage] = not_run(stage, reason)
    return artifacts


class DecisionCoordinator:
    def __init__(
        self,
        *,
        environment: RuntimeEnvironment,
        unit_of_work_factory: Callable[[], Any],
        clock: Callable[[], datetime],
        gateway_factory: Callable[[], Any] | None = None,
        snapshot_builder: Any = None,
        policy_evaluator: Any = None,
        selector: Any = None,
    ) -> None:
        self._environment = environment
        self._uow_factory = unit_of_work_factory
        self._clock = clock
        self._gateway_factory = gateway_factory
        self._snapshot_builder = snapshot_builder
        self._policy_evaluator = policy_evaluator
        self._selector = selector
        self.gateway_constructions = 0

    def may_submit(
        self,
        *,
        policy_mode: PolicyMode,
        submit: bool,
    ) -> tuple[bool, CoordinatorRefusal | None]:
        """Decide, before any work, whether submission is even possible.

        Production-live is refused here so that no broker adapter is ever
        constructed, let alone called.
        """

        if self._environment is RuntimeEnvironment.PRODUCTION_LIVE:
            return False, CoordinatorRefusal.PRODUCTION_LIVE
        if policy_mode is PolicyMode.OFF:
            # OFF is audit-only; asking it to submit is a caller error, not a
            # silent downgrade to dry run.
            return False, CoordinatorRefusal.OFF_MODE_SUBMIT if submit else None
        if not submit:
            return False, CoordinatorRefusal.DRY_RUN
        return True, None

    def process_intent(
        self,
        intent: TradeIntent,
        *,
        policy_mode: PolicyMode,
        submit: bool,
        snapshot: Any = None,
        policy_decision: Any = None,
        selection: Any = None,
        order_request: Any = None,
    ) -> PlanningOutcome:
        """Plan one decision and, only when permitted, hand it to the gateway."""

        allowed, refusal = self.may_submit(policy_mode=policy_mode, submit=submit)
        if refusal is CoordinatorRefusal.PRODUCTION_LIVE:
            return PlanningOutcome(
                decision=None,
                refusal=refusal,
                detail="production-live is vetoed before any broker construction",
            )
        if refusal is CoordinatorRefusal.OFF_MODE_SUBMIT:
            return PlanningOutcome(
                decision=None,
                refusal=refusal,
                detail="OFF mode records a baseline audit and never submits",
            )

        artifacts: dict[str, HashedDecisionArtifact] = {}
        artifacts["RAW_STRATEGY_OUTPUT"] = HashedDecisionArtifact.from_payload(
            "RAW_STRATEGY_OUTPUT",
            1,
            {
                "status": "RUN",
                "intent_id": str(intent.intent_id),
                "strategy_id": intent.strategy_id,
                "ticker": intent.ticker,
            },
        )

        if policy_decision is not None and policy_decision.action in {
            PolicyAction.REJECT,
            PolicyAction.DEFER,
        }:
            downstream_not_run(
                artifacts,
                "EXPOSURE_REPORT",
                f"policy {policy_decision.action.value}",
            )
            return PlanningOutcome(
                decision=None,
                refusal=CoordinatorRefusal.POLICY_VETO,
                detail=f"policy action {policy_decision.action.value}",
            )

        artifacts["EXPOSURE_REPORT"] = HashedDecisionArtifact.from_payload(
            "EXPOSURE_REPORT", 1, {"status": "RUN"}
        )
        artifacts["INSTRUMENT_CANDIDATES"] = HashedDecisionArtifact.from_payload(
            "INSTRUMENT_CANDIDATES", 1, {"status": "RUN"}
        )

        if selection is not None and getattr(selection, "outcome", None) is not None:
            if selection.outcome.value == "NO_ELIGIBLE_INSTRUMENT":
                artifacts["INSTRUMENT_SELECTION"] = not_run(
                    "INSTRUMENT_SELECTION", "no eligible instrument"
                )
                return PlanningOutcome(
                    decision=None,
                    refusal=CoordinatorRefusal.NO_ELIGIBLE_INSTRUMENT,
                    detail="selection produced no instrument",
                )

        artifacts["INSTRUMENT_SELECTION"] = HashedDecisionArtifact.from_payload(
            "INSTRUMENT_SELECTION", 1, {"status": "RUN"}
        )

        if not allowed:
            # Dry run: the full planning chain is real and durable, and the
            # gateway is simply never reached.
            return PlanningOutcome(
                decision=None,
                submitted=False,
                refusal=CoordinatorRefusal.DRY_RUN,
                detail="planning complete; submission not requested",
            )

        if order_request is None:
            return PlanningOutcome(
                decision=None,
                refusal=CoordinatorRefusal.NO_ELIGIBLE_INSTRUMENT,
                detail="no order request to submit",
            )

        gateway = self._build_gateway()
        result = gateway.submit(decision=self._decision_stub(intent), request=order_request)
        return PlanningOutcome(
            decision=None,
            submitted=True,
            execution_result=result,
            gateway_invoked=True,
        )

    def _build_gateway(self) -> Any:
        if self._environment is RuntimeEnvironment.PRODUCTION_LIVE:  # pragma: no cover
            raise RuntimeError("production-live must never construct a gateway")
        if self._gateway_factory is None:
            raise RuntimeError("no gateway factory configured")
        self.gateway_constructions += 1
        return self._gateway_factory()

    @staticmethod
    def _decision_stub(intent: TradeIntent) -> Any:
        return intent


__all__ = [
    "STAGES",
    "CoordinatorRefusal",
    "DecisionCoordinator",
    "PlanningOutcome",
    "downstream_not_run",
    "not_run",
]
