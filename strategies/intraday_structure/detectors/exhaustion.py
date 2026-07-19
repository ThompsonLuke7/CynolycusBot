from __future__ import annotations

from strategies.intraday_structure.detectors.base import DetectionContext, DetectionDecision, is_long
from strategies.intraday_structure.models import SetupRecord, SetupState, SetupType


class ExhaustionDetector:
    """Overlay applied to confirmed/running setups; it never invents a trade."""

    setup_type = SetupType.EXHAUSTION

    def evaluate(self, setup: SetupRecord, ctx: DetectionContext) -> DetectionDecision:
        if setup.state not in {SetupState.RUNNING, SetupState.TARGET_REACHED, SetupState.EXTENDED}:
            return DetectionDecision()
        f, t = ctx.features, ctx.config.detector
        momentum = f.get("ret_1") if is_long(setup) else -f.get("ret_1")
        rejection = f.get("upper_wick_ratio") if is_long(setup) else f.get("lower_wick_ratio")
        vwap_lost = f.get("distance_to_vwap_atr") < -t.retest_tolerance_atr if is_long(setup) else f.get("distance_to_vwap_atr") > t.retest_tolerance_atr
        repeated_failure = int(setup.metadata.get("target_failure_count", 0)) >= t.max_failed_breaks
        divergence = f.get("momentum_divergence") > 0 and momentum <= t.exhaustion_momentum
        climax = f.get("relative_volume_1m") >= t.capitulation_volume and rejection >= t.long_wick_ratio
        if vwap_lost or repeated_failure or divergence and climax:
            evidence = []
            if vwap_lost:
                evidence.append("vwap_or_microstructure_lost")
            if repeated_failure:
                evidence.append("repeated_target_failure")
            if divergence:
                evidence.append("momentum_divergence")
            if climax:
                evidence.append("volume_climax_rejection")
            return DetectionDecision(
                SetupState.EXHAUSTED, "EXHAUSTED", "extension conditions deteriorated",
                tuple(evidence), confidence=max(setup.confidence, 0.65),
            )
        return DetectionDecision()
