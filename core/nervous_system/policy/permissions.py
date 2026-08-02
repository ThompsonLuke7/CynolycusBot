"""Environment, strategy, instrument, and structure permissions.

Permissions are set intersections: widening one list can never restore
something a stricter rule removed.
"""

from __future__ import annotations

from core.nervous_system.config.policy import (
    BOUNDED_STRUCTURE_RISKS,
    PolicyConfig,
    StructureRisk,
)
from core.nervous_system.contracts.context import ContextSnapshot
from core.nervous_system.contracts.enums import (
    InstrumentFamily,
    RuntimeEnvironment,
)
from core.nervous_system.contracts.intent import TradeIntent

from .reason_codes import ReasonCode


_STRUCTURE_VETOES = {
    StructureRisk.NAKED_SHORT: ReasonCode.STRUCTURE_NAKED_SHORT_OPTION,
    StructureRisk.UNCOVERED_RATIO: ReasonCode.STRUCTURE_UNCOVERED_RATIO,
    StructureRisk.UNKNOWN: ReasonCode.STRUCTURE_UNKNOWN_MAXIMUM_LOSS,
}


def environment_vetoes(
    intent: TradeIntent,
    snapshot: ContextSnapshot,
    config: PolicyConfig,
) -> tuple[ReasonCode, ...]:
    """Rule 1: environment and strategy permission."""

    vetoes: list[ReasonCode] = []
    if config.environment is RuntimeEnvironment.PRODUCTION_LIVE:
        vetoes.append(ReasonCode.ENV_PRODUCTION_LIVE_DISABLED_MVP)
    if intent.strategy_id not in config.permitted_strategies:
        vetoes.append(ReasonCode.ENV_STRATEGY_NOT_PERMITTED)
    return tuple(vetoes)


def permitted_instruments(
    intent: TradeIntent,
    config: PolicyConfig,
) -> frozenset[InstrumentFamily]:
    """Return the families that are both requested and permitted with a bounded loss."""

    requested = frozenset(intent.instrument_preferences)
    return frozenset(
        family
        for family in requested & config.allowed_instruments
        if config.structure_risk_for(family) in BOUNDED_STRUCTURE_RISKS
    )


def instrument_vetoes(
    intent: TradeIntent,
    snapshot: ContextSnapshot,
    config: PolicyConfig,
) -> tuple[ReasonCode, ...]:
    """Rule 4: instrument and structure permission."""

    requested = frozenset(intent.instrument_preferences)
    if not requested:
        return (ReasonCode.INSTRUMENT_PREFERENCE_MISSING,)

    allowed = requested & config.allowed_instruments
    if not allowed:
        return (ReasonCode.INSTRUMENT_FAMILY_NOT_PERMITTED,)

    vetoes: list[ReasonCode] = []
    for family in sorted(allowed, key=lambda item: item.value):
        risk = config.structure_risk_for(family)
        veto = _STRUCTURE_VETOES.get(risk)
        if veto is not None and veto not in vetoes:
            vetoes.append(veto)
    if not permitted_instruments(intent, config) and not vetoes:
        vetoes.append(ReasonCode.INSTRUMENT_FAMILY_NOT_PERMITTED)
    return tuple(vetoes)


__all__ = [
    "environment_vetoes",
    "instrument_vetoes",
    "permitted_instruments",
]
