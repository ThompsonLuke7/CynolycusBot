from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from strategies.intraday_structure.config import IntradayStructureConfig
from strategies.intraday_structure.features import FeatureSnapshot
from strategies.intraday_structure.models import (
    Bar,
    MarketContext,
    OptionsContext,
    SetupRecord,
    SetupState,
    SetupType,
    StructuralLevel,
)


@dataclass(frozen=True)
class DetectionContext:
    bar: Bar
    bars: Sequence[Bar]
    features: FeatureSnapshot
    levels: Sequence[StructuralLevel]
    market: MarketContext
    options: OptionsContext
    config: IntradayStructureConfig


@dataclass(frozen=True)
class DetectionDecision:
    state: SetupState | None = None
    phase: str | None = None
    reason: str = "no_change"
    evidence: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    confidence: float | None = None
    pivot: float | None = None
    invalidation: float | None = None
    metadata: dict = field(default_factory=dict)


class SetupDetector(Protocol):
    setup_type: SetupType

    def evaluate(self, setup: SetupRecord, ctx: DetectionContext) -> DetectionDecision: ...


def is_long(setup: SetupRecord) -> bool:
    return setup.direction.value == "long"


def directional(value: float, setup: SetupRecord) -> float:
    return value if is_long(setup) else -value


def market_supports(setup: SetupRecord, ctx: DetectionContext) -> bool:
    score = ctx.market.market_alignment_score
    directional_score = score if is_long(setup) else 1.0 - score
    relative_strength = directional(ctx.features.get("relative_strength_vs_spy"), setup)
    return directional_score >= 0.30 or relative_strength >= ctx.config.exceptional_relative_strength
