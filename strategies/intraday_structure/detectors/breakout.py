from __future__ import annotations

from strategies.intraday_structure.detectors.base import DetectionContext, DetectionDecision, is_long, market_supports
from strategies.intraday_structure.levels import nearest_support, nearest_resistance
from strategies.intraday_structure.models import SetupRecord, SetupState, SetupType


class BreakoutContinuationDetector:
    setup_type = SetupType.BREAKOUT

    def evaluate(self, setup: SetupRecord, ctx: DetectionContext) -> DetectionDecision:
        f, t = ctx.features, ctx.config.detector
        pivot = setup.pivot or setup.candidate.pivot
        if pivot is None:
            level = nearest_resistance(ctx.levels, ctx.bar.close) if is_long(setup) else nearest_support(ctx.levels, ctx.bar.close)
            pivot = level.price if level else None
        if pivot is None:
            return DetectionDecision(warnings=("no_breakout_pivot",))
        atr = f.get("atr")
        signed_distance = (pivot - ctx.bar.close) if is_long(setup) else (ctx.bar.close - pivot)
        broken = ctx.bar.close >= pivot + t.pivot_break_buffer_atr * atr if is_long(setup) else ctx.bar.close <= pivot - t.pivot_break_buffer_atr * atr
        holds = ctx.bar.close >= pivot - t.retest_tolerance_atr * atr if is_long(setup) else ctx.bar.close <= pivot + t.retest_tolerance_atr * atr
        if setup.state == SetupState.WATCHING and -0.25 * atr <= signed_distance <= 0.75 * atr:
            return DetectionDecision(
                SetupState.SETUP_DETECTED, "APPROACHING", "price approaching structural pivot",
                ("pivot_approach",), confidence=0.40, pivot=pivot,
            )
        if setup.state == SetupState.SETUP_DETECTED and broken:
            strength = abs(ctx.bar.close - pivot) / max(atr, 1e-9)
            return DetectionDecision(
                SetupState.ARMED, "BROKEN", "pivot closed through break buffer",
                ("pivot_broken", "volume_confirmation") if f.get("relative_volume_1m") >= t.min_relative_volume else ("pivot_broken",),
                confidence=min(0.68, 0.50 + 0.12 * strength), pivot=pivot,
                invalidation=pivot - t.retest_tolerance_atr * atr if is_long(setup) else pivot + t.retest_tolerance_atr * atr,
                metadata={"hold_count": 0, "retest_count": 0, "failed_break_count": 0},
            )
        if setup.state == SetupState.ARMED:
            if not holds:
                fails = int(setup.metadata.get("failed_break_count", 0)) + 1
                if fails >= t.max_failed_breaks:
                    return DetectionDecision(SetupState.INVALIDATED, "FAILED_BREAKOUT", "pivot could not hold", ("failed_breakout",), metadata={"failed_break_count": fails})
                return DetectionDecision(phase="RETESTING", reason="pivot retest lost temporarily", evidence=("retest_in_progress",), metadata={"failed_break_count": fails, "hold_count": 0})
            hold_count = int(setup.metadata.get("hold_count", 0)) + 1
            touched = ctx.bar.low <= pivot + t.retest_tolerance_atr * atr if is_long(setup) else ctx.bar.high >= pivot - t.retest_tolerance_atr * atr
            retests = int(setup.metadata.get("retest_count", 0)) + int(touched)
            evidence = ["breakout_hold"]
            if touched:
                evidence.append("retest_held")
            if (hold_count >= t.hold_bars or retests >= 1) and market_supports(setup, ctx):
                above_vwap = f.get("distance_to_vwap_atr") >= 0 if is_long(setup) else f.get("distance_to_vwap_atr") <= 0
                if above_vwap:
                    evidence.append("vwap_aligned")
                return DetectionDecision(
                    SetupState.CONFIRMED, "HOLD_CONFIRMED", "break-and-hold confirmed",
                    tuple(evidence), confidence=0.70, pivot=pivot,
                    metadata={"hold_count": hold_count, "retest_count": retests},
                )
            return DetectionDecision(phase="RETESTING", reason="waiting for pivot hold", evidence=tuple(evidence), metadata={"hold_count": hold_count, "retest_count": retests})
        return DetectionDecision()
