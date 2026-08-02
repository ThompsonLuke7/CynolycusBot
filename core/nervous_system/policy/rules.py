"""Pure policy rules in their fixed evaluation order.

Every rule receives only contracts and configuration.  No rule reads a clock,
a file, a database, a broker, or the process environment.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from uuid import UUID

from core.nervous_system.config.policy import PolicyConfig
from core.nervous_system.contracts.context import ContextSnapshot
from core.nervous_system.contracts.enums import (
    DataQualitySeverity,
    DecisionKind,
    ModifierOperation,
    RuntimeEnvironment,
)
from core.nervous_system.contracts.intent import TradeIntent

from .permissions import environment_vetoes, instrument_vetoes
from .reason_codes import ReasonCode


RULES_VERSION = "1"

_SEVERITY_ORDER = (
    DataQualitySeverity.INFO,
    DataQualitySeverity.WARNING,
    DataQualitySeverity.ERROR,
    DataQualitySeverity.CRITICAL,
)
_DEGRADED_SNAPSHOT_STATUSES = {
    "STALE": ReasonCode.SNAPSHOT_REQUIRED_STATE_STALE,
    "MISSING": ReasonCode.SNAPSHOT_REQUIRED_STATE_MISSING,
    "INVALID": ReasonCode.SNAPSHOT_INVALID,
}


def is_risk_reducing(intent: TradeIntent) -> bool:
    """Only an explicit EXIT closes exposure and earns the narrow permission."""

    return intent.decision_kind is DecisionKind.EXIT


def _decimal(value: float | int | Decimal) -> Decimal:
    """Convert a contract float to Decimal without inheriting binary noise."""

    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _dedupe(codes: list[ReasonCode]) -> tuple[ReasonCode, ...]:
    return tuple(dict.fromkeys(codes))


# ---------------------------------------------------------------------------
# Rule 2: snapshot validity and freshness
# ---------------------------------------------------------------------------


def snapshot_vetoes(
    intent: TradeIntent,
    snapshot: ContextSnapshot,
    config: PolicyConfig,
) -> tuple[ReasonCode, ...]:
    vetoes: list[ReasonCode] = []
    if not snapshot.valid:
        vetoes.append(ReasonCode.SNAPSHOT_INVALID)
    if snapshot.freshness_profile != config.required_snapshot_profile:
        vetoes.append(ReasonCode.SNAPSHOT_PROFILE_MISMATCH)
    if intent.snapshot_id != snapshot.snapshot_id:
        vetoes.append(ReasonCode.SNAPSHOT_LINEAGE_MISMATCH)
    if snapshot.decision_time > intent.created_at:
        vetoes.append(ReasonCode.SNAPSHOT_DECISION_TIME_AFTER_INTENT)
    if snapshot.stale_inputs:
        vetoes.append(ReasonCode.SNAPSHOT_REQUIRED_STATE_STALE)
    if snapshot.missing_inputs:
        vetoes.append(ReasonCode.SNAPSHOT_REQUIRED_STATE_MISSING)
    for result in snapshot.requirement_results:
        if not result.required:
            continue
        degraded = _DEGRADED_SNAPSHOT_STATUSES.get(result.status)
        if degraded is not None:
            vetoes.append(degraded)
    return _dedupe(vetoes)


# ---------------------------------------------------------------------------
# Rule 3: operational readiness
# ---------------------------------------------------------------------------


def readiness_vetoes(
    intent: TradeIntent,
    snapshot: ContextSnapshot,
    config: PolicyConfig,
) -> tuple[ReasonCode, ...]:
    readiness = snapshot.readiness_state
    if readiness is None:
        return (ReasonCode.READINESS_STATE_MISSING,)
    vetoes: list[ReasonCode] = []
    if readiness.job not in config.required_readiness_jobs:
        vetoes.append(ReasonCode.READINESS_JOB_NOT_REQUIRED)
    if not readiness.ready:
        vetoes.append(ReasonCode.READINESS_NOT_READY)
    return tuple(vetoes)


# ---------------------------------------------------------------------------
# Rule 5: broker and account constraints
# ---------------------------------------------------------------------------


def broker_vetoes(
    intent: TradeIntent,
    snapshot: ContextSnapshot,
    config: PolicyConfig,
) -> tuple[ReasonCode, ...]:
    vetoes: list[ReasonCode] = []
    if (
        config.environment is RuntimeEnvironment.QA_PAPER
        and config.account_alias not in config.paper_account_aliases
    ):
        vetoes.append(ReasonCode.BROKER_PAPER_ACCOUNT_REQUIRED)

    portfolio = snapshot.portfolio_state
    if portfolio is None:
        vetoes.append(ReasonCode.BROKER_PORTFOLIO_STATE_MISSING)
        return tuple(vetoes)

    if portfolio.account_alias != config.account_alias:
        vetoes.append(ReasonCode.BROKER_ACCOUNT_IDENTITY_MISMATCH)
    # A duplicate order is dangerous in both directions, so this veto is not
    # relaxed for risk-reducing exits.
    if intent.idempotency_key and intent.idempotency_key in portfolio.open_order_ids:
        vetoes.append(ReasonCode.BROKER_DUPLICATE_IDEMPOTENCY_KEY)
    if not is_risk_reducing(intent) and _decimal(
        portfolio.buying_power
    ) < intent.position_size_requested:
        vetoes.append(ReasonCode.BROKER_INSUFFICIENT_BUYING_POWER)
    return tuple(vetoes)


# ---------------------------------------------------------------------------
# Rule 6: hard portfolio limits
# ---------------------------------------------------------------------------


def portfolio_limit_vetoes(
    intent: TradeIntent,
    snapshot: ContextSnapshot,
    config: PolicyConfig,
) -> tuple[ReasonCode, ...]:
    portfolio = snapshot.portfolio_state
    if portfolio is None:
        # Rule 5 already vetoed the missing broker fact; do not double-report.
        return ()
    vetoes: list[ReasonCode] = []
    if portfolio.day_pl is not None and _decimal(portfolio.day_pl) <= -config.max_daily_loss:
        vetoes.append(ReasonCode.PORTFOLIO_MAX_DAILY_LOSS_BREACH)
    gross = sum(
        (
            abs(_decimal(position.market_value))
            for position in portfolio.positions
            if position.market_value is not None
        ),
        Decimal("0"),
    )
    if gross + intent.position_size_requested > config.max_gross_notional:
        vetoes.append(ReasonCode.PORTFOLIO_MAX_GROSS_NOTIONAL_BREACH)
    return tuple(vetoes)


# ---------------------------------------------------------------------------
# Rule 7: liquidity and data-quality limits
# ---------------------------------------------------------------------------


def liquidity_vetoes(
    intent: TradeIntent,
    snapshot: ContextSnapshot,
    config: PolicyConfig,
) -> tuple[ReasonCode, ...]:
    vetoes: list[ReasonCode] = []
    if any(
        issue.severity in config.blocking_data_quality_severities
        for issue in snapshot.data_quality.issues
    ):
        vetoes.append(ReasonCode.DATA_QUALITY_BLOCKING)

    ticker_state = snapshot.ticker_state
    if ticker_state is None:
        vetoes.append(ReasonCode.LIQUIDITY_TICKER_STATE_MISSING)
        return tuple(vetoes)

    observed = ticker_state.metrics.get(config.liquidity_metric)
    if observed is None:
        vetoes.append(ReasonCode.LIQUIDITY_METRIC_UNKNOWN)
    elif _decimal(observed) < config.min_liquidity_value:
        vetoes.append(ReasonCode.LIQUIDITY_BELOW_MINIMUM)
    return tuple(vetoes)


@dataclass(frozen=True)
class HardRule:
    """One ordered hard-veto rule."""

    rule_id: str
    applies_to_risk_reducing: bool
    evaluate: Callable[
        [TradeIntent, ContextSnapshot, PolicyConfig], tuple[ReasonCode, ...]
    ]


HARD_RULES: tuple[HardRule, ...] = (
    HardRule("policy.rule.environment", True, environment_vetoes),
    HardRule("policy.rule.snapshot", False, snapshot_vetoes),
    HardRule("policy.rule.readiness", False, readiness_vetoes),
    HardRule("policy.rule.instrument", False, instrument_vetoes),
    HardRule("policy.rule.broker", True, broker_vetoes),
    HardRule("policy.rule.portfolio_limits", False, portfolio_limit_vetoes),
    HardRule("policy.rule.liquidity", False, liquidity_vetoes),
)


# ---------------------------------------------------------------------------
# Rule 8: contextual size modifiers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModifierSpec:
    """A modifier before it is bound to a running budget."""

    rule_id: str
    operation: ModifierOperation
    input_value: str
    configured_condition: str
    configured_value: Decimal
    reason_code: ReasonCode
    source_state_id: UUID | None

    @property
    def rule_version(self) -> str:
        return f"{self.rule_id}@{RULES_VERSION}"


def context_modifier_specs(
    snapshot: ContextSnapshot,
    config: PolicyConfig,
) -> tuple[tuple[ModifierSpec, ...], tuple[ReasonCode, ...]]:
    """Return the ordered contextual modifiers and the unavailable-context notes."""

    specs: list[ModifierSpec] = []
    notes: list[ReasonCode] = []

    market = snapshot.market_state
    if market is None:
        notes.append(ReasonCode.CONTEXT_MARKET_UNAVAILABLE)
    else:
        specs.append(
            ModifierSpec(
                rule_id="policy.modifier.market_regime",
                operation=ModifierOperation.MULTIPLY,
                input_value=market.regime.value,
                configured_condition=f"market_regime={market.regime.value}",
                configured_value=config.market_regime_multipliers[market.regime],
                reason_code=ReasonCode.CONTEXT_MARKET_REGIME_MODIFIER,
                source_state_id=market.state_id,
            )
        )

    if not snapshot.theme_states:
        notes.append(ReasonCode.CONTEXT_THEME_UNAVAILABLE)
    else:
        # The most conservative theme binds; ties break on theme_id so the
        # result never depends on snapshot ordering.
        theme = min(
            snapshot.theme_states,
            key=lambda state: (
                config.theme_regime_multipliers[state.theme_regime],
                state.theme_id,
            ),
        )
        specs.append(
            ModifierSpec(
                rule_id="policy.modifier.theme_regime",
                operation=ModifierOperation.MULTIPLY,
                input_value=theme.theme_regime.value,
                configured_condition=(
                    f"theme_regime={theme.theme_regime.value};theme_id={theme.theme_id}"
                ),
                configured_value=config.theme_regime_multipliers[theme.theme_regime],
                reason_code=ReasonCode.CONTEXT_THEME_REGIME_MODIFIER,
                source_state_id=theme.state_id,
            )
        )

    dealer = snapshot.dealer_state
    if dealer is None:
        notes.append(ReasonCode.CONTEXT_DEALER_UNAVAILABLE)
    else:
        specs.append(
            ModifierSpec(
                rule_id="policy.modifier.dealer_regime",
                operation=ModifierOperation.MULTIPLY,
                input_value=dealer.dealer_regime.value,
                configured_condition=f"dealer_regime={dealer.dealer_regime.value}",
                configured_value=config.dealer_regime_multipliers[dealer.dealer_regime],
                reason_code=ReasonCode.CONTEXT_DEALER_REGIME_MODIFIER,
                source_state_id=dealer.state_id,
            )
        )

    severity = _worst_non_blocking_severity(snapshot, config)
    if severity is not None:
        specs.append(
            ModifierSpec(
                rule_id="policy.modifier.data_quality",
                operation=ModifierOperation.MULTIPLY,
                input_value=severity.value,
                configured_condition=f"data_quality_severity={severity.value}",
                configured_value=config.data_quality_multipliers[severity],
                reason_code=ReasonCode.CONTEXT_DATA_QUALITY_MODIFIER,
                source_state_id=None,
            )
        )

    return tuple(specs), tuple(notes)


def _worst_non_blocking_severity(
    snapshot: ContextSnapshot,
    config: PolicyConfig,
) -> DataQualitySeverity | None:
    present = {
        issue.severity
        for issue in snapshot.data_quality.issues
        if issue.severity not in config.blocking_data_quality_severities
    }
    for severity in reversed(_SEVERITY_ORDER):
        if severity in present:
            return severity
    return None


# ---------------------------------------------------------------------------
# Rule 9: final caps and minimum executable size
# ---------------------------------------------------------------------------


def max_position_cap_spec(budget: Decimal, config: PolicyConfig) -> ModifierSpec:
    return ModifierSpec(
        rule_id="policy.cap.max_position_notional",
        operation=ModifierOperation.CAP,
        input_value=str(budget),
        configured_condition=f"max_position_notional={config.max_position_notional}",
        configured_value=config.max_position_notional,
        reason_code=ReasonCode.SIZE_CAPPED_TO_MAX_POSITION_NOTIONAL,
        source_state_id=None,
    )


def money_quantum_cap_spec(budget: Decimal, config: PolicyConfig) -> ModifierSpec:
    """Round the budget down to the configured money quantum.

    Expressed as a CAP so the audit trail keeps a single monotone waterfall and
    rounding can never increase risk.
    """

    quantized = budget.quantize(config.money_quantum, rounding=ROUND_DOWN)
    return ModifierSpec(
        rule_id="policy.cap.money_quantum",
        operation=ModifierOperation.CAP,
        input_value=str(budget),
        configured_condition=f"money_quantum={config.money_quantum}",
        configured_value=quantized,
        reason_code=ReasonCode.SIZE_QUANTIZED_TO_MONEY_QUANTUM,
        source_state_id=None,
    )


__all__ = [
    "HARD_RULES",
    "HardRule",
    "ModifierSpec",
    "RULES_VERSION",
    "broker_vetoes",
    "context_modifier_specs",
    "is_risk_reducing",
    "liquidity_vetoes",
    "max_position_cap_spec",
    "money_quantum_cap_spec",
    "portfolio_limit_vetoes",
    "readiness_vetoes",
    "snapshot_vetoes",
]
