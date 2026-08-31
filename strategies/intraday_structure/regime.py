"""Rule-based context regime, and the abstention record that carries it.

WHY. ``engine._apply_decision`` already refuses roughly half the setups that
clear their detector -- 684 ``risk_or_target_plan_unavailable``, 327
``reward_risk_below_threshold``, 19 ``runway_below_threshold`` across the first
27 live sessions -- and every one of those refusals was a bare ``return`` that
appended to ``setup.warnings`` and vanished at the next overwrite.  The decision
to stand down was already being made correctly; it was simply never recorded, so
nobody could ask whether standing down was right.

There is deliberately NO model here.  The regime is a handful of thresholds over
features the engine already computes, and it is the baseline any later model has
to beat.  It is recorded on every abstention and every signal; whether it is
allowed to VETO is a separate, default-off config switch, so turning it on is a
measurable change rather than a silent one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from strategies.intraday_structure.config import RegimePolicy
from strategies.intraday_structure.levels import nearest_resistance, nearest_support
from strategies.intraday_structure.models import SetupRecord, StructuralLevel
from strategies.intraday_structure.state_store import append_jsonl


ABSTENTION_SCHEMA_VERSION = "intraday_structure_abstention_v1"

TRENDING_UP = "TRENDING_UP"
TRENDING_DOWN = "TRENDING_DOWN"
BALANCED = "BALANCED"
COMPRESSED = "COMPRESSED"


@dataclass(frozen=True)
class RegimeAssessment:
    """A regime label plus every number that produced it."""

    regime: str
    evidence: tuple[str, ...]
    atr_contraction: float
    room_to_support_atr: float | None
    room_to_resistance_atr: float | None
    trapped_between_levels: bool
    failed_break_count: int
    trend_strength: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_context(
    *,
    spot: float,
    atr: float,
    features: dict[str, float],
    levels: Sequence[StructuralLevel],
    policy: RegimePolicy,
) -> RegimeAssessment:
    """Label the tape from features and levels the engine has already computed.

    Order matters: compression is checked first because a coiled, level-boxed
    tape is the condition the caller most needs to hear about, and a weak trend
    reading inside one is not informative.
    """
    scale = max(abs(atr), abs(spot) * 1e-6)
    # Only levels several mechanisms agree on can box price in; see
    # RegimePolicy.trapped_min_strength.
    walls = [level for level in levels if level.strength >= policy.trapped_min_strength]
    support = nearest_support(walls, spot)
    resistance = nearest_resistance(walls, spot)
    room_down = (spot - support.price) / scale if support else None
    room_up = (resistance.price - spot) / scale if resistance else None
    trapped = (
        policy.trapped_room_atr is not None
        and room_down is not None and room_up is not None
        and room_down <= policy.trapped_room_atr and room_up <= policy.trapped_room_atr
    )
    # Rejections already counted on the surrounding liquidity zones; no need to
    # recompute what levels.py derived from add_liquidity_zone_features().
    failed_breaks = sum(level.rejection_count for level in (support, resistance) if level is not None)
    contraction = float(features.get("atr_contraction", 1.0) or 1.0)
    trend = float(features.get("trend_strength", 0.0) or 0.0)
    vwap_distance = float(features.get("distance_to_vwap_atr", 0.0) or 0.0)

    evidence: list[str] = []
    if contraction <= policy.compression_atr_ratio:
        evidence.append("range_contraction")
    if trapped:
        evidence.append("trapped_between_levels")
    if failed_breaks >= policy.failed_break_count:
        evidence.append("repeated_failed_breaks")

    if evidence:
        return RegimeAssessment(
            COMPRESSED, tuple(evidence), contraction, room_down, room_up,
            trapped, failed_breaks, trend,
        )
    if trend >= policy.trend_strength and vwap_distance >= 0:
        regime, evidence = TRENDING_UP, ["trend_up", "above_vwap"]
    elif trend <= -policy.trend_strength and vwap_distance <= 0:
        regime, evidence = TRENDING_DOWN, ["trend_down", "below_vwap"]
    else:
        regime, evidence = BALANCED, ["no_directional_conviction"]
    return RegimeAssessment(
        regime, tuple(evidence), contraction, room_down, room_up,
        trapped, failed_breaks, trend,
    )


def regime_conflicts(regime: str, direction: str) -> bool:
    """Does the regime argue against taking ``direction`` at all?

    A compressed tape argues against both sides; a trending tape argues against
    fading it.  Only consulted when ``RegimePolicy.veto_enabled`` is on.
    """
    if regime == COMPRESSED:
        return True
    if regime == TRENDING_UP:
        return direction == "short"
    if regime == TRENDING_DOWN:
        return direction == "long"
    return False


@dataclass(frozen=True)
class AbstentionRecord:
    """One recorded decision NOT to take a setup that cleared its detector."""

    schema_version: str
    engine_version: str
    setup_id: str
    ticker: str
    direction: str
    setup_type: str
    timestamp: str
    spot: float | None
    no_trade_reason: str
    context_regime: str
    regime_evidence: list[str]
    proposed_invalidation: float | None
    proposed_target: float | None
    runway_score: float | None
    reward_risk: float | None
    min_runway_score: float
    min_reward_risk: float
    atr: float | None
    atr_contraction: float | None
    room_to_support_atr: float | None
    room_to_resistance_atr: float | None
    trapped_between_levels: bool
    failed_break_count: int
    candidate_sources: list[str]
    candidate_score: float
    confidence: float
    market_alignment_score: float
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_abstention_record(
    setup: SetupRecord,
    *,
    reason: str,
    regime: RegimeAssessment,
    timestamp: datetime,
    spot: float | None,
    atr: float | None,
    engine_version: str,
    min_runway_score: float,
    min_reward_risk: float,
    runway_score: float | None = None,
    reward_risk: float | None = None,
    proposed_invalidation: float | None = None,
    proposed_target: float | None = None,
) -> AbstentionRecord:
    return AbstentionRecord(
        schema_version=ABSTENTION_SCHEMA_VERSION,
        engine_version=engine_version,
        setup_id=setup.setup_id,
        ticker=setup.ticker,
        direction=setup.direction.value,
        setup_type=setup.setup_type.value,
        timestamp=timestamp.isoformat(),
        spot=spot,
        no_trade_reason=reason,
        context_regime=regime.regime,
        regime_evidence=list(regime.evidence),
        proposed_invalidation=proposed_invalidation,
        proposed_target=proposed_target,
        runway_score=runway_score,
        reward_risk=reward_risk,
        min_runway_score=min_runway_score,
        min_reward_risk=min_reward_risk,
        atr=atr,
        atr_contraction=regime.atr_contraction,
        room_to_support_atr=regime.room_to_support_atr,
        room_to_resistance_atr=regime.room_to_resistance_atr,
        trapped_between_levels=regime.trapped_between_levels,
        failed_break_count=regime.failed_break_count,
        candidate_sources=list(setup.candidate.sources),
        candidate_score=float(setup.candidate.score),
        confidence=float(setup.confidence),
        market_alignment_score=float(setup.market_alignment_score),
        evidence=list(setup.evidence),
    )


def abstention_sink(path: str | Path) -> Callable[[AbstentionRecord], None]:
    return lambda record: append_jsonl(path, record.to_dict())
