"""Explainable dealer-level qualification for already-confirmed price structure.

The plate score does not create a trade, infer dealer inventory, or override the
one-minute detectors. It labels a confirmed setup when its next destination is
a strong, reachable, as-of dealer-derived level and the estimated imbalance is
not materially opposed to the setup direction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from strategies.intraday_structure.config import DealerPlatePolicy
from strategies.intraday_structure.models import Direction, OptionsContext, StructuralLevel


@dataclass(frozen=True)
class DealerPlateResult:
    score: float
    qualified: bool
    target: float | None
    target_type: str | None
    target_strength: float
    distance_atr: float | None
    components: dict[str, float]
    evidence: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_dealer_plate(
    *,
    direction: Direction,
    spot: float,
    atr: float,
    options: OptionsContext,
    policy: DealerPlatePolicy,
    levels: Sequence[StructuralLevel] | None = None,
) -> DealerPlateResult:
    if not policy.enabled:
        return _unavailable("dealer_plate_disabled")
    if options.source == "none" or not options.levels:
        return _unavailable("dealer_levels_unavailable")
    candidates = _destination_levels(direction, spot, levels or options.levels)
    if not candidates:
        return _unavailable("no_directional_dealer_target")
    target = candidates[0]
    distance_atr = abs(target.price - spot) / max(atr, spot * 1e-6)
    distance_component = _distance_component(distance_atr, policy)
    strength_component = float(np.clip(target.strength, 0.0, 1.0))
    imbalance_component = _imbalance_component(direction, options.dealer_imbalance)
    freshness_component = 0.35 if any("stale" in warning for warning in options.warnings) else 1.0
    source_component = float(np.clip(options.dealer_strength_score, 0.0, 1.0)) if options.dealer_strength_score > 0 else strength_component
    components = {
        "target_strength": strength_component,
        "target_distance": distance_component,
        "directional_imbalance": imbalance_component,
        "snapshot_freshness": freshness_component,
        "map_strength": source_component,
    }
    weights = {
        "target_strength": 0.34,
        "target_distance": 0.22,
        "directional_imbalance": 0.18,
        "snapshot_freshness": 0.12,
        "map_strength": 0.14,
    }
    score = float(np.clip(sum(components[key] * weights[key] for key in weights), 0.0, 1.0))
    qualified = (
        score >= policy.min_score
        and strength_component >= policy.min_target_strength
        and policy.min_target_distance_atr <= distance_atr <= policy.max_target_distance_atr
        and freshness_component >= 1.0
    )
    evidence = [f"dealer_target_{target.level_type}", f"dealer_target_distance_{distance_atr:.2f}_atr"]
    if qualified:
        evidence.append("favorable_dealer_swing_plate")
    elif strength_component < policy.min_target_strength:
        evidence.append("dealer_target_strength_below_threshold")
    elif not policy.min_target_distance_atr <= distance_atr <= policy.max_target_distance_atr:
        evidence.append("dealer_target_distance_outside_policy")
    else:
        evidence.append("dealer_plate_score_below_threshold")
    warnings = [warning for warning in options.warnings if "stale" in warning or "estimated" in warning]
    return DealerPlateResult(
        score=score, qualified=qualified, target=target.price, target_type=target.level_type,
        target_strength=strength_component, distance_atr=distance_atr,
        components=components, evidence=tuple(evidence), warnings=tuple(warnings),
    )


def _destination_levels(direction: Direction, spot: float, levels: Sequence[StructuralLevel]) -> list[StructuralLevel]:
    if direction == Direction.LONG:
        rows = [level for level in levels if level.price > spot and _is_upside_destination(level)]
        return sorted(rows, key=lambda level: level.price)
    rows = [level for level in levels if level.price < spot and _is_downside_destination(level)]
    return sorted(rows, key=lambda level: level.price, reverse=True)


def _is_upside_destination(level: StructuralLevel) -> bool:
    return any(token in level.level_type for token in ("magnet_above", "call_wall", "ceiling", "gamma_flip"))


def _is_downside_destination(level: StructuralLevel) -> bool:
    return any(token in level.level_type for token in ("magnet_below", "put_wall", "floor", "gamma_flip"))


def _distance_component(distance_atr: float, policy: DealerPlatePolicy) -> float:
    if distance_atr < policy.min_target_distance_atr or distance_atr > policy.max_target_distance_atr:
        return 0.0
    ideal = min(3.0, policy.max_target_distance_atr)
    if distance_atr <= ideal:
        return float(np.clip((distance_atr - policy.min_target_distance_atr) / max(ideal - policy.min_target_distance_atr, 1e-6), 0.0, 1.0))
    return float(np.clip(1.0 - (distance_atr - ideal) / max(policy.max_target_distance_atr - ideal, 1e-6), 0.0, 1.0))


def _imbalance_component(direction: Direction, imbalance: float | None) -> float:
    if imbalance is None or not np.isfinite(imbalance):
        return 0.5
    signed = float(np.clip(imbalance, -1.0, 1.0))
    if direction == Direction.SHORT:
        signed *= -1.0
    return float(np.clip(0.5 + 0.5 * signed, 0.0, 1.0))


def _unavailable(reason: str) -> DealerPlateResult:
    return DealerPlateResult(
        score=0.0, qualified=False, target=None, target_type=None, target_strength=0.0,
        distance_atr=None, components={}, evidence=(), warnings=(reason,),
    )
