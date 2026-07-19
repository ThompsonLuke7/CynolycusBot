from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd

from strategies.intraday_structure.config import IntradayStructureConfig
from strategies.intraday_structure.detectors.base import DetectionContext
from strategies.intraday_structure.features import FeatureSnapshot
from strategies.intraday_structure.labels import LabelConfig, build_event_labels
from strategies.intraday_structure.models import (
    Bar, Candidate, Direction, MarketContext, OptionsContext, SetupRecord, SetupState, SetupType, StructuralLevel,
)
from strategies.intraday_structure.target_manager import evaluate_extension


NOW = datetime(2026, 7, 17, 15, tzinfo=timezone.utc)


def _target_context(*, trend=1.0, runway_target=103.0) -> DetectionContext:
    config = IntradayStructureConfig()
    bar = Bar("XYZ", NOW, 101, 101.5, 100.8, 101.2, 3000)
    features = FeatureSnapshot(NOW, {"atr": 1, "relative_volume_5m": 1.2, "trend_strength": trend, "micro_swing_low": 100, "micro_swing_high": 102})
    levels = [StructuralLevel(runway_target, "next_resistance", 0.3, directionality="resistance")]
    market = MarketContext(NOW, market_alignment_score=0.8)
    return DetectionContext(bar, [bar], features, levels, market, OptionsContext(), config)


def test_target_extension_and_extension_failure() -> None:
    candidate = Candidate("XYZ", NOW, Direction.LONG, ("test",), score=0.8)
    setup = SetupRecord("x", "XYZ", SetupType.BREAKOUT, Direction.LONG, candidate, state=SetupState.TARGET_REACHED, targets=[101], entry_price=100, invalidation=99)
    extended = evaluate_extension(setup, _target_context())
    assert extended.state == SetupState.EXTENDED
    assert setup.active_target == 103.0
    setup.state = SetupState.TARGET_REACHED
    setup.metadata["target_failure_count"] = 1
    failed = evaluate_extension(setup, _target_context(trend=-1.0, runway_target=104))
    assert failed.state == SetupState.EXHAUSTED


def test_labels_use_conservative_same_bar_collision_and_overlap_suppression() -> None:
    start = datetime(2026, 7, 17, 13, 30, tzinfo=timezone.utc)
    bars = pd.DataFrame([
        {"symbol": "XYZ", "timestamp": start + timedelta(minutes=i), "open": 100, "high": 102 if i == 1 else 100.5, "low": 98 if i == 1 else 99.5, "close": 100, "volume": 1000}
        for i in range(10)
    ])
    event = {"ticker": "XYZ", "timestamp": start, "setup_type": "breakout_continuation", "direction": "long", "entry_price": 100, "invalidation": 99, "targets": [101], "pivot": 100}
    labels = build_event_labels(bars, [event, {**event, "timestamp": start + timedelta(minutes=2)}], config=LabelConfig(forward_bars=5, overlap_cooldown_bars=5))
    assert len(labels) == 1
    assert not bool(labels.iloc[0]["target_before_invalidation"])
    assert bool(labels.iloc[0]["breakout_failed"])
