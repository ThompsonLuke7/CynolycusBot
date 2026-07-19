from __future__ import annotations

from strategies.intraday_structure.detectors.base import DetectionContext, DetectionDecision, market_supports
from strategies.intraday_structure.models import Direction, SetupRecord, SetupState, SetupType


class VReversalDetector:
    setup_type = SetupType.V_REVERSAL

    def evaluate(self, setup: SetupRecord, ctx: DetectionContext) -> DetectionDecision:
        if setup.direction != Direction.LONG:
            return DetectionDecision(warnings=("v_reversal_v1_long_only",))
        f = ctx.features
        t = ctx.config.detector
        if setup.state == SetupState.WATCHING:
            selloff = f.get("ret_3") <= t.selloff_return_3
            expanded = f.get("range_expansion") >= t.range_expansion
            abnormal_volume = f.get("relative_volume_1m") >= t.min_relative_volume
            if selloff and expanded and abnormal_volume:
                return DetectionDecision(
                    SetupState.SETUP_DETECTED, "SELL_OFF", "accelerating downside expansion",
                    ("accelerating_selloff", "range_expansion", "abnormal_relative_volume"),
                    confidence=0.42,
                    invalidation=ctx.bar.low - t.retest_tolerance_atr * f.get("atr"),
                    metadata={"selloff_low": ctx.bar.low},
                )
        elif setup.state == SetupState.SETUP_DETECTED:
            capitulation = f.get("relative_volume_1m") >= t.capitulation_volume
            rejection = f.get("lower_wick_ratio") >= t.long_wick_ratio and f.get("close_location_value") >= 0.55
            decelerating = f.get("downside_momentum_deceleration") > 0
            if capitulation and (rejection or decelerating):
                return DetectionDecision(
                    SetupState.ARMED, "CAPITULATION", "capitulation bar rejected its low",
                    ("capitulation_volume", "failed_breakdown" if rejection else "selling_deceleration"),
                    confidence=0.56,
                    pivot=f.get("micro_swing_high", ctx.bar.high),
                    invalidation=ctx.bar.low - 0.10 * f.get("atr"),
                    metadata={"capitulation_low": ctx.bar.low, "bars_since_capitulation": 0},
                )
        elif setup.state == SetupState.ARMED:
            cap_low = float(setup.metadata.get("capitulation_low", ctx.bar.low))
            if ctx.bar.close < cap_low - t.retest_tolerance_atr * f.get("atr"):
                return DetectionDecision(SetupState.INVALIDATED, "FAILED", "capitulation low failed", ("reversal_failed",))
            higher_low = f.get("micro_higher_low") > 0
            micro_break = ctx.bar.close > float(setup.pivot or f.get("micro_swing_high", ctx.bar.high))
            vwap_reclaim = f.get("distance_to_vwap_atr") >= -0.10
            if (higher_low and micro_break or micro_break and vwap_reclaim) and market_supports(setup, ctx):
                evidence = ["micro_higher_low" if higher_low else "selling_deceleration", "micro_swing_high_broken"]
                if vwap_reclaim:
                    evidence.append("vwap_approach_or_reclaim")
                return DetectionDecision(
                    SetupState.CONFIRMED, "CONFIRMED_REVERSAL", "higher-low reversal confirmed",
                    tuple(evidence), confidence=0.68,
                    pivot=setup.pivot or f.get("micro_swing_high"), invalidation=cap_low,
                )
            if higher_low:
                return DetectionDecision(phase="STABILIZATION", reason="higher low forming", evidence=("micro_higher_low",), confidence=0.60)
        return DetectionDecision()
