from __future__ import annotations

from strategies.intraday_structure.detectors.base import DetectionContext, DetectionDecision, is_long
from strategies.intraday_structure.levels import nearest_support, nearest_resistance
from strategies.intraday_structure.models import SetupRecord, SetupState, SetupType


class StructuralRejectionDetector:
    setup_type = SetupType.STRUCTURAL_REJECTION

    def evaluate(self, setup: SetupRecord, ctx: DetectionContext) -> DetectionDecision:
        f, t = ctx.features, ctx.config.detector
        level = nearest_support(ctx.levels, ctx.bar.close) if is_long(setup) else nearest_resistance(ctx.levels, ctx.bar.close)
        pivot = setup.pivot or (level.price if level else None)
        if pivot is None:
            return DetectionDecision(warnings=("no_structural_rejection_level",))
        distance = abs(ctx.bar.close - pivot) / max(f.get("atr"), 1e-9)
        wick = f.get("lower_wick_ratio") if is_long(setup) else f.get("upper_wick_ratio")
        closing_away = ctx.bar.close > pivot if is_long(setup) else ctx.bar.close < pivot
        if setup.state == SetupState.WATCHING and distance <= t.rejection_distance_atr and wick >= t.long_wick_ratio * 0.75:
            evidence = ["structural_level_test", "rejection_wick"]
            if level and "options_" in level.level_type:
                evidence.append("options_derived_level")
            return DetectionDecision(
                SetupState.SETUP_DETECTED, "LEVEL_TEST", "price rejected a structural level",
                tuple(evidence), confidence=0.48, pivot=pivot,
                invalidation=pivot - t.retest_tolerance_atr * f.get("atr") if is_long(setup) else pivot + t.retest_tolerance_atr * f.get("atr"),
            )
        if setup.state == SetupState.SETUP_DETECTED:
            if closing_away and distance >= 0.10:
                return DetectionDecision(SetupState.ARMED, "RETURN_FROM_LEVEL", "price moved away from rejected level", ("failed_auction", "returned_from_level"), confidence=0.58, pivot=pivot)
        if setup.state == SetupState.ARMED:
            momentum = f.get("ret_1") > 0 if is_long(setup) else f.get("ret_1") < 0
            if closing_away and momentum:
                return DetectionDecision(
                    SetupState.CONFIRMED, "REJECTION_CONFIRMED", "direction away from structural level confirmed",
                    ("level_rejection_confirmed", "momentum_away_from_level"), confidence=0.65, pivot=pivot,
                )
            invalid = ctx.bar.close < setup.invalidation if is_long(setup) and setup.invalidation is not None else ctx.bar.close > setup.invalidation if setup.invalidation is not None else False
            if invalid:
                return DetectionDecision(SetupState.INVALIDATED, "LEVEL_FAILED", "rejected level failed", ("structural_level_broken",))
        return DetectionDecision()
