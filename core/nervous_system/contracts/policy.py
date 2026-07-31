from __future__ import annotations

from decimal import Decimal
from uuid import UUID

from pydantic import model_validator

from .base import ContractModel, FiniteFloat, NonNegativeDecimal, UtcDatetime
from .enums import (
    Direction,
    InstrumentFamily,
    ModifierOperation,
    PolicyAction,
    PolicyMode,
    RuntimeEnvironment,
)


class PolicyModifier(ContractModel):
    rule_id: str
    rule_version: str
    operation: ModifierOperation
    input_value: str
    configured_condition: str
    configured_value: NonNegativeDecimal
    budget_before: NonNegativeDecimal
    budget_after: NonNegativeDecimal
    reason_code: str
    source_state_id: UUID | None
    config_version: str

    @model_validator(mode="after")
    def validate_budget_operation(self) -> PolicyModifier:
        if self.operation is ModifierOperation.MULTIPLY:
            if self.configured_value > Decimal("1.0"):
                raise ValueError("MULTIPLY modifiers must not increase risk")
            expected = self.budget_before * self.configured_value
        else:
            expected = min(self.budget_before, self.configured_value)
        if self.budget_after != expected:
            raise ValueError("modifier budget_after does not match its operation")
        return self


class PolicyDecision(ContractModel):
    policy_decision_id: UUID
    intent_id: UUID
    snapshot_id: UUID
    environment: RuntimeEnvironment
    mode: PolicyMode
    action: PolicyAction
    approved_direction: Direction
    base_risk_budget: NonNegativeDecimal
    final_risk_budget: NonNegativeDecimal
    allowed_instruments: frozenset[InstrumentFamily]
    hard_vetoes: tuple[str, ...]
    modifiers: tuple[PolicyModifier, ...]
    stop_adjustment: FiniteFloat | None
    target_adjustment: FiniteFloat | None
    holding_period_adjustment: int | None
    hedge_requirement: str | None
    reason_codes: tuple[str, ...]
    policy_version: str
    config_version: str
    created_at: UtcDatetime
    expires_at: UtcDatetime

    @model_validator(mode="after")
    def validate_policy(self) -> PolicyDecision:
        if self.expires_at <= self.created_at:
            raise ValueError("policy expires_at must be after created_at")
        if self.final_risk_budget > self.base_risk_budget:
            raise ValueError("final_risk_budget must not exceed base_risk_budget")
        if self.action in {PolicyAction.APPROVE, PolicyAction.APPROVE_REDUCED} and self.hard_vetoes:
            raise ValueError("approved actions cannot contain hard vetoes")
        prior = self.base_risk_budget
        for modifier in self.modifiers:
            if modifier.budget_before != prior:
                raise ValueError("policy modifier budgets must form an ordered chain")
            prior = modifier.budget_after
        if self.action is PolicyAction.REJECT:
            if self.final_risk_budget != Decimal("0"):
                raise ValueError("REJECT must have zero final_risk_budget")
            return self
        if self.final_risk_budget != prior:
            raise ValueError("final_risk_budget must equal the last modifier budget")
        return self
