"""The deterministic policy evaluator.

``evaluate_policy`` is a pure function of ``(intent, snapshot, config)``.  Given
the same three inputs it always produces the same ``PolicyDecision``, including
its identity, so a replayed decision is byte-comparable with the original.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from uuid import UUID, uuid5

from core.nervous_system.config.policy import PolicyConfig
from core.nervous_system.contracts.context import ContextSnapshot
from core.nervous_system.contracts.enums import (
    Direction,
    InstrumentFamily,
    ModifierOperation,
    PolicyAction,
    PolicyMode,
)
from core.nervous_system.contracts.intent import TradeIntent
from core.nervous_system.contracts.policy import PolicyDecision, PolicyModifier

from .permissions import environment_vetoes, permitted_instruments
from .reason_codes import ReasonCode
from .rules import (
    HARD_RULES,
    ModifierSpec,
    context_modifier_specs,
    is_risk_reducing,
    max_position_cap_spec,
    money_quantum_cap_spec,
)


# uuid5(NAMESPACE_URL, "https://cynolycus.local/nervous-system/policy-decision@1")
POLICY_DECISION_NAMESPACE = UUID("c5239c33-2411-54f8-b9ae-34efa823a134")

_ZERO = Decimal("0")


class ExecutionBasis(str, Enum):
    """What orchestration may execute from a policy decision.

    The ``PolicyDecision`` contract is identical in every mode; only the basis
    changes.  Hard vetoes bind in every mode: ``BASELINE_INTENT`` selects the
    sizing basis, it never resurrects a rejected decision.
    """

    AUDIT_ONLY = "AUDIT_ONLY"
    BASELINE_INTENT = "BASELINE_INTENT"
    POLICY_FINAL = "POLICY_FINAL"


_EXECUTABLE_ACTIONS = frozenset(
    {PolicyAction.APPROVE, PolicyAction.APPROVE_REDUCED, PolicyAction.EXIT}
)


def execution_basis(decision: PolicyDecision) -> ExecutionBasis:
    """Derive the execution basis from the mode recorded on the decision."""

    if decision.mode is PolicyMode.OFF:
        return ExecutionBasis.AUDIT_ONLY
    if decision.mode is PolicyMode.SHADOW:
        return ExecutionBasis.BASELINE_INTENT
    return ExecutionBasis.POLICY_FINAL


def is_executable(decision: PolicyDecision) -> bool:
    """Whether orchestration may submit anything for this decision.

    Orchestration must gate on this rather than on ``action`` alone: an OFF
    decision records a baseline that was never evaluated against the hard
    rules, so it is audit-only regardless of the action it carries.
    """

    return (
        execution_basis(decision) is not ExecutionBasis.AUDIT_ONLY
        and decision.action in _EXECUTABLE_ACTIONS
        and not decision.hard_vetoes
    )


def evaluate_policy(
    intent: TradeIntent,
    snapshot: ContextSnapshot,
    config: PolicyConfig,
) -> PolicyDecision:
    """Evaluate the fixed rule order and return one auditable decision."""

    base_budget = intent.position_size_requested

    # Rule 1 first and alone: production-live is denied even in OFF mode.
    environment_denials = environment_vetoes(intent, snapshot, config)
    if environment_denials:
        return _build(
            intent,
            snapshot,
            config,
            action=PolicyAction.REJECT,
            final_budget=_ZERO,
            modifiers=(),
            allowed_instruments=frozenset(),
            hard_vetoes=environment_denials,
            informational=(),
        )

    if config.mode is PolicyMode.OFF:
        return _build(
            intent,
            snapshot,
            config,
            action=PolicyAction.DEFER,
            final_budget=base_budget,
            modifiers=(),
            allowed_instruments=frozenset(intent.instrument_preferences),
            hard_vetoes=(),
            informational=(ReasonCode.POLICY_OFF_AUDIT_ONLY,),
        )

    risk_reducing = is_risk_reducing(intent)

    hard_vetoes: list[ReasonCode] = []
    for rule in HARD_RULES:
        if risk_reducing and not rule.applies_to_risk_reducing:
            continue
        hard_vetoes.extend(rule.evaluate(intent, snapshot, config))
    ordered_vetoes = tuple(dict.fromkeys(hard_vetoes))

    if ordered_vetoes:
        return _build(
            intent,
            snapshot,
            config,
            action=PolicyAction.REJECT,
            final_budget=_ZERO,
            modifiers=(),
            allowed_instruments=frozenset(),
            hard_vetoes=ordered_vetoes,
            informational=(),
        )

    if risk_reducing:
        # A risk-reducing exit keeps the strategy's requested size: contextual
        # modulation and entry caps must never trap an open position.
        return _build(
            intent,
            snapshot,
            config,
            action=PolicyAction.EXIT,
            final_budget=base_budget,
            modifiers=(),
            allowed_instruments=frozenset(intent.instrument_preferences),
            hard_vetoes=(),
            informational=(ReasonCode.EXIT_RISK_REDUCING_PERMITTED,),
        )

    context_specs, context_notes = context_modifier_specs(snapshot, config)
    modifiers, budget = _apply(context_specs, base_budget, config)

    cap_modifier, budget = _apply_one(max_position_cap_spec(budget, config), budget, config)
    quantum_modifier, budget = _apply_one(
        money_quantum_cap_spec(budget, config), budget, config
    )
    modifiers = modifiers + (cap_modifier, quantum_modifier)

    allowed = permitted_instruments(intent, config)

    if budget < config.minimum_order_notional:
        return _build(
            intent,
            snapshot,
            config,
            action=PolicyAction.REJECT,
            final_budget=_ZERO,
            modifiers=modifiers,
            allowed_instruments=frozenset(),
            hard_vetoes=(ReasonCode.SIZE_BELOW_MINIMUM_EXECUTABLE,),
            informational=context_notes + tuple(m.reason_code for m in context_specs),
        )

    if budget == base_budget:
        action = PolicyAction.APPROVE
        outcome = ReasonCode.POLICY_APPROVED_AS_REQUESTED
    else:
        action = PolicyAction.APPROVE_REDUCED
        outcome = ReasonCode.POLICY_APPROVED_REDUCED

    return _build(
        intent,
        snapshot,
        config,
        action=action,
        final_budget=budget,
        modifiers=modifiers,
        allowed_instruments=allowed,
        hard_vetoes=(),
        informational=(
            context_notes
            + tuple(spec.reason_code for spec in context_specs)
            + (outcome,)
        ),
    )


def _apply(
    specs: tuple[ModifierSpec, ...],
    budget: Decimal,
    config: PolicyConfig,
) -> tuple[tuple[PolicyModifier, ...], Decimal]:
    modifiers: list[PolicyModifier] = []
    for spec in specs:
        modifier, budget = _apply_one(spec, budget, config)
        modifiers.append(modifier)
    return tuple(modifiers), budget


def _apply_one(
    spec: ModifierSpec,
    budget: Decimal,
    config: PolicyConfig,
) -> tuple[PolicyModifier, Decimal]:
    if spec.operation is ModifierOperation.MULTIPLY:
        after = budget * spec.configured_value
    else:
        # min() with the configured value preferred on ties, so the money
        # quantum normalises the Decimal scale instead of carrying the wider
        # scale of the multiplier chain into downstream sizing.
        after = spec.configured_value if spec.configured_value <= budget else budget
    modifier = PolicyModifier(
        rule_id=spec.rule_id,
        rule_version=spec.rule_version,
        operation=spec.operation,
        input_value=spec.input_value,
        configured_condition=spec.configured_condition,
        configured_value=spec.configured_value,
        budget_before=budget,
        budget_after=after,
        reason_code=spec.reason_code.value,
        source_state_id=spec.source_state_id,
        config_version=config.config_version,
    )
    return modifier, after


def _decision_identity(
    intent: TradeIntent,
    snapshot: ContextSnapshot,
    config: PolicyConfig,
) -> UUID:
    return uuid5(
        POLICY_DECISION_NAMESPACE,
        "|".join(
            (
                str(intent.intent_id),
                snapshot.content_hash,
                config.content_hash,
                config.mode.value,
            )
        ),
    )


def _build(
    intent: TradeIntent,
    snapshot: ContextSnapshot,
    config: PolicyConfig,
    *,
    action: PolicyAction,
    final_budget: Decimal,
    modifiers: tuple[PolicyModifier, ...],
    allowed_instruments: frozenset[InstrumentFamily],
    hard_vetoes: tuple[ReasonCode, ...],
    informational: tuple[ReasonCode, ...],
) -> PolicyDecision:
    reason_codes = tuple(
        code.value for code in dict.fromkeys(hard_vetoes + informational)
    )
    return PolicyDecision(
        policy_decision_id=_decision_identity(intent, snapshot, config),
        intent_id=intent.intent_id,
        snapshot_id=snapshot.snapshot_id,
        environment=config.environment,
        mode=config.mode,
        action=action,
        approved_direction=(
            Direction.NEUTRAL if action is PolicyAction.REJECT else intent.direction
        ),
        base_risk_budget=intent.position_size_requested,
        final_risk_budget=final_budget,
        allowed_instruments=allowed_instruments,
        hard_vetoes=tuple(code.value for code in hard_vetoes),
        modifiers=modifiers,
        stop_adjustment=None,
        target_adjustment=None,
        holding_period_adjustment=None,
        hedge_requirement=None,
        reason_codes=reason_codes,
        policy_version=config.policy_version,
        config_version=config.config_version,
        created_at=intent.created_at,
        expires_at=intent.created_at + config.entry_window,
    )


__all__ = [
    "POLICY_DECISION_NAMESPACE",
    "ExecutionBasis",
    "evaluate_policy",
    "execution_basis",
    "is_executable",
]
