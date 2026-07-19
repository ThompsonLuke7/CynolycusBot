from __future__ import annotations

from strategies.intraday_structure.detectors.base import DetectionContext, DetectionDecision, directional, is_long, market_supports
from strategies.intraday_structure.models import SetupRecord, SetupState, SetupType


class TrendPullbackDetector:
    setup_type = SetupType.TREND_PULLBACK

    def evaluate(self, setup: SetupRecord, ctx: DetectionContext) -> DetectionDecision:
        f, t = ctx.features, ctx.config.detector
        trend = directional(f.get("trend_strength"), setup)
        pullback = abs(f.get("distance_to_ema9_atr")) <= t.pullback_max_atr or abs(f.get("distance_to_vwap_atr")) <= t.pullback_max_atr
        controlled_volume = f.get("relative_volume_1m") <= max(1.15, t.min_relative_volume)
        countertrend = f.get("ret_1") <= 0 if is_long(setup) else f.get("ret_1") >= 0
        if setup.state == SetupState.WATCHING and trend >= t.trend_strength:
            return DetectionDecision(SetupState.SETUP_DETECTED, "ESTABLISHED_TREND", "trend established", ("established_trend",), confidence=0.38)
        if setup.state == SetupState.SETUP_DETECTED and pullback and controlled_volume and countertrend:
            invalidation = min(f.get("ema_20"), f.get("session_vwap")) - t.retest_tolerance_atr * f.get("atr") if is_long(setup) else max(f.get("ema_20"), f.get("session_vwap")) + t.retest_tolerance_atr * f.get("atr")
            return DetectionDecision(
                SetupState.ARMED, "CONTROLLED_PULLBACK", "controlled pullback reached dynamic support",
                ("controlled_pullback", "reduced_countertrend_volume", "dynamic_level_hold"),
                confidence=0.55, pivot=f.get("ema_9"), invalidation=invalidation,
            )
        if setup.state == SetupState.ARMED:
            reaccelerating = f.get("ret_1") > 0 and ctx.bar.close > f.get("ema_9") if is_long(setup) else f.get("ret_1") < 0 and ctx.bar.close < f.get("ema_9")
            if reaccelerating and trend >= t.trend_strength * 0.6 and market_supports(setup, ctx):
                return DetectionDecision(
                    SetupState.CONFIRMED, "REACCELERATION", "trend resumed after controlled pullback",
                    ("trend_reacceleration", "ema_reclaim", "market_context_supportive"),
                    confidence=0.67,
                )
            invalid = ctx.bar.close < setup.invalidation if is_long(setup) and setup.invalidation is not None else ctx.bar.close > setup.invalidation if setup.invalidation is not None else False
            if invalid:
                return DetectionDecision(SetupState.INVALIDATED, "PULLBACK_FAILED", "pullback lost invalidation structure", ("trend_structure_lost",))
        return DetectionDecision()
