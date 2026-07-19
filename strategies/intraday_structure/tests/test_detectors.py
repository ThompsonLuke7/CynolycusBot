from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from strategies.intraday_structure.config import IntradayStructureConfig
from strategies.intraday_structure.detectors import (
    BreakoutContinuationDetector,
    ExhaustionDetector,
    StructuralRejectionDetector,
    TrendPullbackDetector,
    VReversalDetector,
    VwapReclaimDetector,
)
from strategies.intraday_structure.detectors.base import DetectionContext
from strategies.intraday_structure.features import FeatureSnapshot
from strategies.intraday_structure.models import (
    Bar,
    Candidate,
    Direction,
    MarketContext,
    OptionsContext,
    SetupRecord,
    SetupState,
    SetupType,
    StructuralLevel,
)


NOW = datetime(2026, 7, 17, 15, 0, tzinfo=timezone.utc)


def _setup(kind: SetupType, state: SetupState, *, direction: Direction = Direction.LONG, pivot: float | None = None) -> SetupRecord:
    candidate = Candidate("XYZ", NOW, direction, ("test",), score=0.8, pivot=pivot)
    return SetupRecord(f"XYZ:{direction.value}:{kind.value}", "XYZ", kind, direction, candidate, state=state, pivot=pivot, invalidation=98.0)


def _ctx(**updates) -> DetectionContext:
    values = {
        "atr": 1.0, "ret_1": 0.01, "ret_3": 0.02, "range_expansion": 1.0,
        "relative_volume_1m": 1.5, "relative_volume_5m": 1.2, "lower_wick_ratio": 0.1,
        "upper_wick_ratio": 0.1, "close_location_value": 0.7,
        "downside_momentum_deceleration": 0.01, "micro_higher_low": 1,
        "micro_swing_high": 100.5, "micro_swing_low": 99.0,
        "distance_to_vwap_atr": 0.2, "above_vwap_duration": 3, "below_vwap_duration": 0,
        "session_vwap": 100.0, "relative_strength_vs_spy": 0.02,
        "trend_strength": 0.8, "distance_to_ema9_atr": 0.1, "distance_to_ema20_atr": 0.2,
        "ema_9": 100.0, "ema_20": 99.5, "momentum_divergence": 0,
    }
    values.update(updates)
    close = float(values.pop("close", 100.8))
    bar = Bar("XYZ", NOW, 100.0, max(101.0, close + 0.2), min(99.5, close - 0.2), close, 2000)
    market = MarketContext(NOW, market_alignment_score=float(values.pop("market_alignment", 0.75)))
    levels = [
        StructuralLevel(100.0, "put_wall", 0.8, directionality="support"),
        StructuralLevel(102.0, "call_wall", 0.8, directionality="resistance"),
    ]
    return DetectionContext(bar, [bar], FeatureSnapshot(NOW, values), levels, market, OptionsContext(source="test", put_wall=100, call_wall=102), IntradayStructureConfig())


def test_v_reversal_success_and_failure() -> None:
    detector = VReversalDetector()
    detected = detector.evaluate(_setup(SetupType.V_REVERSAL, SetupState.WATCHING), _ctx(ret_3=-0.02, range_expansion=1.8, relative_volume_1m=1.8))
    assert detected.state == SetupState.SETUP_DETECTED
    armed = detector.evaluate(_setup(SetupType.V_REVERSAL, SetupState.SETUP_DETECTED), _ctx(relative_volume_1m=2.5, lower_wick_ratio=0.6))
    assert armed.state == SetupState.ARMED
    setup = _setup(SetupType.V_REVERSAL, SetupState.ARMED, pivot=100.5)
    setup.metadata["capitulation_low"] = 99.0
    assert detector.evaluate(setup, _ctx(close=100.8)).state == SetupState.CONFIRMED
    assert detector.evaluate(setup, _ctx(close=98.5)).state == SetupState.INVALIDATED


def test_breakout_hold_and_false_breakout() -> None:
    detector = BreakoutContinuationDetector()
    setup = _setup(SetupType.BREAKOUT, SetupState.ARMED, pivot=100.0)
    setup.metadata.update(hold_count=1, retest_count=0, failed_break_count=0)
    assert detector.evaluate(setup, _ctx(close=100.8)).state == SetupState.CONFIRMED
    setup.metadata["failed_break_count"] = 1
    assert detector.evaluate(setup, _ctx(close=99.0)).state == SetupState.INVALIDATED


def test_vwap_reclaim_and_market_conflict() -> None:
    detector = VwapReclaimDetector()
    setup = _setup(SetupType.VWAP_RECLAIM, SetupState.ARMED)
    assert detector.evaluate(setup, _ctx()).state == SetupState.CONFIRMED
    conflict = detector.evaluate(setup, _ctx(market_alignment=0.1, relative_strength_vs_spy=0.0))
    assert conflict.state is None


def test_gamma_structural_rejection() -> None:
    detector = StructuralRejectionDetector()
    setup = _setup(SetupType.STRUCTURAL_REJECTION, SetupState.ARMED, pivot=100.0)
    assert detector.evaluate(setup, _ctx(close=100.8)).state == SetupState.CONFIRMED


def test_trend_pullback_reacceleration() -> None:
    detector = TrendPullbackDetector()
    setup = _setup(SetupType.TREND_PULLBACK, SetupState.ARMED)
    assert detector.evaluate(setup, _ctx(close=100.8, ret_1=0.01, trend_strength=0.8)).state == SetupState.CONFIRMED


def test_exhaustion_overlay_detects_target_extension_failure() -> None:
    setup = _setup(SetupType.BREAKOUT, SetupState.RUNNING)
    setup.metadata["target_failure_count"] = 2
    result = ExhaustionDetector().evaluate(setup, _ctx())
    assert result.state == SetupState.EXHAUSTED
    assert "repeated_target_failure" in result.evidence
