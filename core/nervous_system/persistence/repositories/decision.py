"""Typed persistence for the immutable decision chain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.nervous_system.contracts.base import content_hash
from core.nervous_system.contracts.context import ContextSnapshot
from core.nervous_system.contracts.decisions import DecisionOutcome, DecisionRecord
from core.nervous_system.contracts.intent import TradeIntent
from core.nervous_system.contracts.orders import OrderRequest
from core.nervous_system.contracts.policy import PolicyDecision
from core.nervous_system.persistence.models import (
    ContextSnapshot as ContextSnapshotRow,
    DecisionOutcome as DecisionOutcomeRow,
    DecisionRecord as DecisionRecordRow,
    OrderLeg,
    OrderRequest as OrderRequestRow,
    PolicyDecision as PolicyDecisionRow,
    PolicyModifier as PolicyModifierRow,
    TradeIntent as TradeIntentRow,
)


def _hash_without_identity(contract: Any, identity_field: str) -> str:
    return content_hash(contract, exclude={identity_field})


@dataclass(frozen=True)
class CompleteDecisionChain:
    """All typed inputs required to persist one complete decision atomically."""

    snapshot: ContextSnapshot
    intent: TradeIntent
    policy_decision: PolicyDecision
    record: DecisionRecord
    order_requests: tuple[OrderRequest, ...] = ()

    def __post_init__(self) -> None:
        if self.snapshot.content_hash != self.snapshot.computed_content_hash():
            raise ValueError("decision chain snapshot content_hash does not match snapshot")
        if self.intent.snapshot_id != self.snapshot.snapshot_id:
            raise ValueError("decision chain intent does not reference its snapshot")
        if self.policy_decision.intent_id != self.intent.intent_id:
            raise ValueError("decision chain policy does not reference its intent")
        if self.policy_decision.snapshot_id != self.snapshot.snapshot_id:
            raise ValueError("decision chain policy does not reference its snapshot")
        if self.record.snapshot_id != self.snapshot.snapshot_id:
            raise ValueError("decision chain record does not reference its snapshot")
        if self.record.intent_id != self.intent.intent_id:
            raise ValueError("decision chain record does not reference its intent")
        if self.record.policy_decision_id != self.policy_decision.policy_decision_id:
            raise ValueError("decision chain record does not reference its policy")
        order_ids = tuple(order.order_request_id for order in self.order_requests)
        order_hashes = tuple(order.request_hash for order in self.order_requests)
        if self.record.order_request_ids != order_ids:
            raise ValueError("decision chain record order IDs do not match requests")
        if self.record.order_hashes != order_hashes:
            raise ValueError("decision chain record order hashes do not match requests")
        if self.record.snapshot_hash != self.snapshot.content_hash:
            raise ValueError("decision chain record snapshot hash does not match snapshot")
        if self.record.intent_hash != _hash_without_identity(self.intent, "intent_id"):
            raise ValueError("decision chain record intent hash does not match intent")
        if self.record.policy_hash != _hash_without_identity(
            self.policy_decision, "policy_decision_id"
        ):
            raise ValueError("decision chain record policy hash does not match policy")
        for order in self.order_requests:
            if order.decision_id != self.record.decision_record_id:
                raise ValueError("decision chain order does not reference its record")
            if order.policy_decision_id != self.policy_decision.policy_decision_id:
                raise ValueError("decision chain order does not reference its policy")

    @property
    def policy(self) -> PolicyDecision:
        return self.policy_decision

    @property
    def orders(self) -> tuple[OrderRequest, ...]:
        return self.order_requests


def _record_from_row(row: DecisionRecordRow) -> DecisionRecord:
    record = DecisionRecord.model_validate(row.payload)
    if record.decision_record_id != row.decision_record_id:
        raise ValueError("decision payload ID does not match relational ID")
    if record.decision_time != row.decision_time:
        raise ValueError("decision payload time does not match relational column")
    expected_hash = _hash_without_identity(record, "decision_record_id")
    if row.content_hash != expected_hash:
        raise ValueError("decision record content_hash does not match payload")
    if row.status != "COMPLETE":
        raise ValueError("only complete decision records can have typed outcomes")
    if row.snapshot_id != record.snapshot_id:
        raise ValueError("decision record snapshot link does not match payload")
    if row.intent_id != record.intent_id:
        raise ValueError("decision record intent link does not match payload")
    if row.policy_decision_id != record.policy_decision_id:
        raise ValueError("decision record policy link does not match payload")
    return record


class DecisionRepository:
    def __init__(self, session: Session):
        self._session = session

    def save_trade_intent(self, intent: TradeIntent) -> None:
        self._session.add(
            TradeIntentRow(
                intent_id=intent.intent_id,
                strategy_id=intent.strategy_id,
                ticker=intent.ticker,
                decision_time=intent.created_at,
                snapshot_id=intent.snapshot_id,
                content_hash=_hash_without_identity(intent, "intent_id"),
                payload=intent.model_dump(mode="json"),
                created_at=intent.created_at,
            )
        )
        self._session.flush()

    def save_policy_decision(self, decision: PolicyDecision) -> None:
        self._session.add(
            PolicyDecisionRow(
                policy_decision_id=decision.policy_decision_id,
                intent_id=decision.intent_id,
                snapshot_id=decision.snapshot_id,
                action=decision.action.value,
                final_risk_budget=decision.final_risk_budget,
                content_hash=_hash_without_identity(decision, "policy_decision_id"),
                payload=decision.model_dump(mode="json"),
                created_at=decision.created_at,
            )
        )
        self._session.flush()
        for sequence_no, modifier in enumerate(decision.modifiers, start=1):
            self._session.add(
                PolicyModifierRow(
                    modifier_id=uuid4(),
                    policy_decision_id=decision.policy_decision_id,
                    sequence_no=sequence_no,
                    rule_id=modifier.rule_id,
                    operation=modifier.operation.value,
                    configured_value=modifier.configured_value,
                    budget_before=modifier.budget_before,
                    budget_after=modifier.budget_after,
                    reason_code=modifier.reason_code,
                    payload=modifier.model_dump(mode="json"),
                )
            )
        self._session.flush()

    def save_order_request(self, request: OrderRequest) -> None:
        row = OrderRequestRow(
            order_request_id=request.order_request_id,
            decision_record_id=request.decision_id,
            policy_decision_id=request.policy_decision_id,
            environment=request.environment.value,
            account_alias=request.account_alias,
            idempotency_key=request.idempotency_key,
            request_hash=request.request_hash,
            status="PLANNED",
            decision_kind=request.decision_kind.value,
            risk_reducing=request.risk_reducing,
            order_type=request.order_type,
            broker_position_key=request.broker_position_key,
            parent_quantity=request.parent_quantity,
            net_limit_price=request.net_limit_price,
            maximum_loss=request.maximum_loss,
            buying_power_required=request.buying_power_required,
            payload=request.model_dump(mode="json"),
            created_at=request.created_at,
            expires_at=request.expires_at,
        )
        self._session.add(row)
        self._session.flush()
        for sequence_no, leg in enumerate(request.legs, start=1):
            self._session.add(
                OrderLeg(
                    order_leg_id=uuid4(),
                    order_request_id=request.order_request_id,
                    sequence_no=sequence_no,
                    symbol=leg.symbol,
                    side=leg.side.value,
                    position_intent=leg.position_intent.value,
                    ratio=leg.ratio,
                    payload=leg.model_dump(mode="json"),
                )
            )
        self._session.flush()

    def save_decision_record(self, record: DecisionRecord) -> None:
        self._session.add(
            DecisionRecordRow(
                decision_record_id=record.decision_record_id,
                decision_time=record.decision_time,
                snapshot_id=record.snapshot_id,
                intent_id=record.intent_id,
                policy_decision_id=record.policy_decision_id,
                status="COMPLETE",
                failure_stage=None,
                failure_reason=None,
                content_hash=_hash_without_identity(record, "decision_record_id"),
                payload=record.model_dump(mode="json"),
                created_at=record.decision_time,
            )
        )
        self._session.flush()

    def save_chain(self, chain: CompleteDecisionChain) -> None:
        """Insert the complete chain in FK-safe order without committing."""

        # Re-run validation at the repository boundary in case a caller built
        # the dataclass through a deserializer or a future mutable adapter.
        CompleteDecisionChain(
            snapshot=chain.snapshot,
            intent=chain.intent,
            policy_decision=chain.policy_decision,
            record=chain.record,
            order_requests=chain.order_requests,
        )
        self._session.add(
            ContextSnapshotRow(
                snapshot_id=chain.snapshot.snapshot_id,
                decision_time=chain.snapshot.decision_time,
                strategy_id=chain.snapshot.strategy_id,
                ticker=chain.snapshot.ticker,
                freshness_profile=chain.snapshot.freshness_profile,
                content_hash=chain.snapshot.content_hash,
                payload=chain.snapshot.model_dump(mode="json"),
                created_at=chain.snapshot.decision_time,
            )
        )
        self._session.flush()
        self.save_trade_intent(chain.intent)
        self.save_policy_decision(chain.policy_decision)
        self.save_decision_record(chain.record)
        for order in chain.order_requests:
            self.save_order_request(order)

    def append_decision_outcome(self, outcome: DecisionOutcome) -> None:
        row = self._session.get(DecisionRecordRow, outcome.decision_record_id)
        if row is None:
            raise ValueError("outcome references an unknown decision record")
        record = _record_from_row(row)
        outcome.validate_against(record)
        self._session.add(
            DecisionOutcomeRow(
                outcome_id=outcome.outcome_id,
                decision_record_id=outcome.decision_record_id,
                evaluated_at=outcome.evaluated_at,
                horizon=outcome.horizon,
                payload=outcome.model_dump(mode="json"),
                created_at=outcome.evaluated_at,
            )
        )
        self._session.flush()


__all__ = ["CompleteDecisionChain", "DecisionRepository"]
