from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from strategies.intraday_structure.levels import nearest_level
from strategies.intraday_structure.models import MarketContext, OptionsContext, StructuralLevel


@dataclass(frozen=True)
class RunwayResult:
    runway_score: float
    next_target: float | None
    intermediate_obstacles: tuple[dict, ...]
    explanation: tuple[str, ...]
    components: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def score_runway(
    *,
    spot: float,
    direction: str,
    atr: float,
    levels: Sequence[StructuralLevel],
    trend_strength: float,
    market: MarketContext,
    options: OptionsContext,
    beyond: float | None = None,
) -> RunwayResult:
    """Transparent 0..1 runway score; every component is returned for audit."""
    target = nearest_level(levels, spot, direction, beyond=beyond)
    if target is None:
        return RunwayResult(0.25, None, (), ("no_causal_target_available",), {
            "distance": 0.25, "congestion": 0.25, "level_strength": 0.25,
            "trend": _trend_component(direction, trend_strength),
            "market": market.market_alignment_score, "options": 0.5,
        })
    signed_distance = (target.price - spot) if direction == "long" else (spot - target.price)
    distance_atr = max(0.0, signed_distance / max(atr, spot * 1e-6))
    obstacles = [
        level for level in levels
        if ((spot < level.price < target.price) if direction == "long" else (target.price < level.price < spot))
    ]
    congestion_penalty = min(1.0, sum(level.strength for level in obstacles) / 2.5)
    distance_component = float(np.clip(distance_atr / 2.0, 0.0, 1.0))
    congestion_component = 1.0 - congestion_penalty
    strength_component = 1.0 - float(np.clip(target.strength, 0.0, 1.0)) * 0.45
    trend_component = _trend_component(direction, trend_strength)
    market_component = market.market_alignment_score if direction == "long" else 1.0 - market.market_alignment_score
    options_component = _options_component(direction, spot, target.price, options)
    components = {
        "distance": distance_component,
        "congestion": congestion_component,
        "level_strength": strength_component,
        "trend": trend_component,
        "market": market_component,
        "options": options_component,
    }
    weights = {"distance": 0.25, "congestion": 0.22, "level_strength": 0.13, "trend": 0.18, "market": 0.14, "options": 0.08}
    score = float(np.clip(sum(components[k] * weights[k] for k in weights), 0.0, 1.0))
    explanation = [f"target_{target.level_type}", f"distance_{distance_atr:.2f}_atr"]
    explanation.append("clear_runway" if congestion_component >= 0.7 else "structural_congestion")
    if options.source == "none":
        explanation.append("options_unavailable_neutral_weight")
    return RunwayResult(
        runway_score=score, next_target=target.price,
        intermediate_obstacles=tuple(level.to_dict() for level in obstacles),
        explanation=tuple(explanation), components=components,
    )


def _trend_component(direction: str, strength: float) -> float:
    signed = strength if direction == "long" else -strength
    return float(np.clip(0.5 + 0.25 * signed, 0.0, 1.0))


def _options_component(direction: str, spot: float, target: float, options: OptionsContext) -> float:
    if options.source == "none":
        return 0.5
    wall = options.call_wall if direction == "long" else options.put_wall
    if wall is None:
        return 0.5
    blocks = wall < target if direction == "long" else wall > target
    if blocks:
        return 0.15
    room = abs(wall - spot) / max(abs(target - spot), spot * 1e-6)
    return float(np.clip(0.45 + 0.15 * room, 0.0, 1.0))
