from __future__ import annotations

from strategies.intraday_structure.detectors.base import DetectionContext, DetectionDecision, directional, is_long, market_supports
from strategies.intraday_structure.models import SetupRecord, SetupState, SetupType


class VwapReclaimDetector:
    setup_type = SetupType.VWAP_RECLAIM

    def evaluate(self, setup: SetupRecord, ctx: DetectionContext) -> DetectionDecision:
        f, t = ctx.features, ctx.config.detector
        distance = directional(f.get("distance_to_vwap_atr"), setup)
        duration = int(f.get("above_vwap_duration") if is_long(setup) else f.get("below_vwap_duration"))
        if setup.state == SetupState.WATCHING and distance < -0.10:
            return DetectionDecision(SetupState.SETUP_DETECTED, "BELOW_VWAP" if is_long(setup) else "ABOVE_VWAP", "price is on the countertrend side of VWAP", ("vwap_reclaim_candidate",), confidence=0.35)
        if setup.state == SetupState.SETUP_DETECTED and distance >= 0:
            return DetectionDecision(
                SetupState.ARMED, "VWAP_RECLAIM", "price crossed session VWAP",
                ("vwap_reclaimed",), confidence=0.52,
                pivot=f.get("session_vwap"),
                invalidation=f.get("session_vwap") - 0.25 * f.get("atr") if is_long(setup) else f.get("session_vwap") + 0.25 * f.get("atr"),
                metadata={"vwap_hold_count": duration},
            )
        if setup.state == SetupState.ARMED:
            if distance < -t.retest_tolerance_atr:
                return DetectionDecision(SetupState.INVALIDATED, "FAILED_RECLAIM", "VWAP reclaim failed", ("vwap_lost",))
            rs = directional(f.get("relative_strength_vs_spy"), setup)
            volume_ok = f.get("relative_volume_1m") >= t.min_relative_volume or f.get("relative_volume_5m") >= 1.0
            if duration >= t.vwap_hold_bars and rs >= 0 and volume_ok and market_supports(setup, ctx):
                return DetectionDecision(
                    SetupState.CONFIRMED, "HOLD_CONFIRMED", "VWAP held with improving relative strength",
                    ("vwap_hold", "relative_strength_improving", "volume_confirmation"),
                    confidence=0.66, pivot=f.get("session_vwap"),
                    metadata={"vwap_hold_count": duration},
                )
            return DetectionDecision(phase="HOLDING", reason="waiting for VWAP hold", evidence=("above_vwap",), metadata={"vwap_hold_count": duration})
        return DetectionDecision()
