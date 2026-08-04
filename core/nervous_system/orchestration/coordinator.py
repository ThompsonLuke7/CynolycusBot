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
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from core.nervous_system.contracts.base import content_hash
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

from .events import AggregateType, EventType


class CoordinatorRefusal(str, Enum):
    PRODUCTION_LIVE = "PRODUCTION_LIVE_DISABLED"
    OFF_MODE_SUBMIT = "OFF_MODE_CANNOT_SUBMIT"
    DRY_RUN = "DRY_RUN"
    POLICY_VETO = "POLICY_VETO"
    NO_ELIGIBLE_INSTRUMENT = "NO_ELIGIBLE_INSTRUMENT"
    STAGE_FAILED = "STAGE_FAILED"


# uuid5(NAMESPACE_URL, "https://cynolycus.local/nervous-system/decision-record@1")
_DECISION_NAMESPACE = uuid5(NAMESPACE_URL, "cynolycus/nervous-system/decision-record@1")

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


def build_decision_record(
    *,
    decision_record_id: UUID,
    decision_time: datetime,
    snapshot: Any,
    intent: TradeIntent,
    policy_decision: Any,
    artifacts: Mapping[str, HashedDecisionArtifact],
    order_requests: Sequence[Any] = (),
    source_manifest_hash: str,
    config_hash: str,
) -> DecisionRecord:
    """Assemble the complete record, deriving every lineage hash from content."""

    return DecisionRecord(
        decision_record_id=decision_record_id,
        decision_time=decision_time,
        status="COMPLETE",
        snapshot_id=snapshot.snapshot_id,
        intent_id=intent.intent_id,
        policy_decision_id=policy_decision.policy_decision_id,
        order_request_ids=tuple(item.order_request_id for item in order_requests),
        source_manifest_hash=source_manifest_hash,
        snapshot_hash=snapshot.content_hash,
        intent_hash=content_hash(intent, exclude={"intent_id"}),
        policy_hash=content_hash(policy_decision, exclude={"policy_decision_id"}),
        raw_strategy_output=artifacts["RAW_STRATEGY_OUTPUT"],
        exposure_report=artifacts["EXPOSURE_REPORT"],
        instrument_candidates=artifacts["INSTRUMENT_CANDIDATES"],
        instrument_selection=artifacts["INSTRUMENT_SELECTION"],
        order_hashes=tuple(item.request_hash for item in order_requests),
        config_hash=config_hash,
    )


def failed_decision_record(
    *,
    decision_record_id: UUID,
    decision_time: datetime,
    failure_stage: str,
    failure_code: str,
    failure_message: str,
    source_manifest_hash: str | None = None,
) -> DecisionRecord:
    """Record an early failure durably without inventing lineage.

    A failure before a snapshot or policy exists must not manufacture IDs for
    rows that were never written, so every link stays null.
    """

    return DecisionRecord(
        decision_record_id=decision_record_id,
        decision_time=decision_time,
        status="FAILED",
        snapshot_id=None,
        intent_id=None,
        policy_decision_id=None,
        source_manifest_hash=source_manifest_hash,
        snapshot_hash=None,
        intent_hash=None,
        policy_hash=None,
        failure_stage=failure_stage,
        failure_code=failure_code,
        failure_message=failure_message,
    )


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

        # Everything planned. Persist the whole chain and its outbox event in
        # one transaction, before the broker is considered.
        decision = None
        if snapshot is not None and policy_decision is not None:
            decision = self._persist_planning(
                snapshot=snapshot,
                intent=intent,
                policy_decision=policy_decision,
                artifacts=artifacts,
                order_requests=(order_request,) if order_request is not None else (),
            )

        if not allowed:
            # Dry run: the full planning chain is real and durable, and the
            # gateway is simply never reached.
            return PlanningOutcome(
                decision=decision,
                submitted=False,
                refusal=CoordinatorRefusal.DRY_RUN,
                detail="planning complete; submission not requested",
            )

        if order_request is None:
            return PlanningOutcome(
                decision=decision,
                refusal=CoordinatorRefusal.NO_ELIGIBLE_INSTRUMENT,
                detail="no order request to submit",
            )

        gateway = self._build_gateway()
        result = gateway.submit(decision=decision, request=order_request)
        return PlanningOutcome(
            decision=decision,
            submitted=True,
            execution_result=result,
            gateway_invoked=True,
        )

    def _persist_planning(
        self,
        *,
        snapshot: Any,
        intent: TradeIntent,
        policy_decision: Any,
        artifacts: Mapping[str, HashedDecisionArtifact],
        order_requests: Sequence[Any],
    ) -> DecisionRecord:
        """Commit snapshot, intent, policy, artifacts, orders, and outbox once.

        The broker is never called from inside this transaction: a broker call
        holding a database transaction open would make a slow or hung request
        block every other writer, and a rollback would erase the record that
        the call happened.
        """

        from core.nervous_system.persistence.repositories.decision import (
            CompleteDecisionChain,
        )

        decision_id = self._decision_id(intent, policy_decision)
        record = build_decision_record(
            decision_record_id=decision_id,
            decision_time=intent.created_at,
            snapshot=snapshot,
            intent=intent,
            policy_decision=policy_decision,
            artifacts=artifacts,
            order_requests=order_requests,
            source_manifest_hash=snapshot.content_hash,
            config_hash=policy_decision.config_version.encode("utf-8").hex().ljust(64, "0")[:64],
        )
        chain = CompleteDecisionChain(
            snapshot=snapshot,
            intent=intent,
            policy_decision=policy_decision,
            record=record,
            order_requests=tuple(order_requests),
        )
        with self._uow_factory() as uow:
            # Replanning the same intent and policy must converge, not explode
            # on a unique key. A crash between the commit and the caller's
            # return would otherwise leave the retry permanently stuck.
            existing = uow.decisions.get_decision_record(decision_id)
            if existing is not None:
                uow.commit()
                return existing
            uow.decisions.save_chain(chain)
            uow.operations.enqueue(
                event_type=EventType.DECISION_RECORDED.value,
                aggregate_type=AggregateType.DECISION.value,
                aggregate_id=str(record.decision_record_id),
                payload={
                    "intent_id": str(intent.intent_id),
                    "policy_action": policy_decision.action.value,
                    "order_request_ids": [
                        str(item.order_request_id) for item in order_requests
                    ],
                },
                created_at=self._clock(),
            )
            uow.commit()
        return record

    def record_failure(
        self,
        *,
        failure_stage: str,
        failure_code: str,
        failure_message: str,
        decision_time: datetime | None = None,
    ) -> DecisionRecord:
        """Persist a durable FAILED record for a failure before planning."""

        when = decision_time or self._clock()
        # Identity is derived from content so that retrying the same failure
        # converges. The decision_records content hash is unique, so a random
        # ID would collide on a replay instead of returning the existing row.
        record = failed_decision_record(
            decision_record_id=uuid5(
                _DECISION_NAMESPACE,
                f"failed|{failure_stage}|{failure_code}|{failure_message}|{when.isoformat()}",
            ),
            decision_time=when,
            failure_stage=failure_stage,
            failure_code=failure_code,
            failure_message=failure_message,
        )
        with self._uow_factory() as uow:
            existing = uow.decisions.get_decision_record(record.decision_record_id)
            if existing is not None:
                uow.commit()
                return existing
            uow.decisions.save_decision_record(record)
            uow.commit()
        return record

    @staticmethod
    def _decision_id(intent: TradeIntent, policy_decision: Any) -> UUID:
        return uuid5(
            _DECISION_NAMESPACE,
            f"{intent.intent_id}|{policy_decision.policy_decision_id}",
        )

    def _build_gateway(self) -> Any:
        if self._environment is RuntimeEnvironment.PRODUCTION_LIVE:  # pragma: no cover
            raise RuntimeError("production-live must never construct a gateway")
        if self._gateway_factory is None:
            raise RuntimeError("no gateway factory configured")
        self.gateway_constructions += 1
        return self._gateway_factory()

__all__ = [
    "STAGES",
    "CoordinatorRefusal",
    "DecisionCoordinator",
    "PlanningOutcome",
    "build_decision_record",
    "downstream_not_run",
    "failed_decision_record",
    "not_run",
]
