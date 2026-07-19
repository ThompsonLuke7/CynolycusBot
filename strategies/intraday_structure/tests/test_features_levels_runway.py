from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from strategies.intraday_structure.features import compute_features
from strategies.intraday_structure.levels import cluster_levels
from strategies.intraday_structure.models import Bar, MarketContext, OptionsContext, StructuralLevel
from strategies.intraday_structure.runway import score_runway


def _bars(count: int, *, future_bump: float = 0.0) -> list[Bar]:
    start = datetime(2026, 7, 17, 13, 30, tzinfo=timezone.utc)
    rows = []
    for i in range(count):
        close = 100.0 + 0.02 * i + (future_bump if i == count - 1 else 0.0)
        rows.append(Bar("XYZ", start + timedelta(minutes=i), close - 0.01, close + 0.08, close - 0.08, close, 1000 + 10 * i))
    return rows


def test_feature_prefix_is_unchanged_by_future_bar() -> None:
    prefix = _bars(30)
    before = compute_features(prefix).to_dict()
    extended = prefix + [Bar("XYZ", prefix[-1].timestamp + timedelta(minutes=1), 200, 205, 195, 202, 1_000_000)]
    after_same_prefix = compute_features(extended[:-1]).to_dict()
    assert before == after_same_prefix
    assert compute_features(extended).get("relative_volume_1m") > before["relative_volume_1m"]


def test_cluster_levels_deduplicates_with_atr_threshold() -> None:
    levels = [
        StructuralLevel(100.00, "pdh", 0.8),
        StructuralLevel(100.08, "call_wall", 0.9),
        StructuralLevel(102.00, "swing_high", 0.5),
    ]
    clustered = cluster_levels(levels, spot=99.0, atr=1.0, atr_threshold=0.2, pct_threshold=0.001)
    assert len(clustered) == 2
    assert clustered[0].metadata["cluster_size"] == 2
    assert set(clustered[0].metadata["sources"]) == {"call_wall", "pdh"}


def test_runway_score_is_transparent_and_congestion_lowers_score() -> None:
    market = MarketContext(datetime.now(timezone.utc), market_alignment_score=0.75)
    clear = [StructuralLevel(103.0, "target", 0.5, directionality="resistance")]
    congested = [
        StructuralLevel(101.0, "obstacle_a", 0.9, directionality="both"),
        StructuralLevel(102.0, "obstacle_b", 0.9, directionality="both"),
        StructuralLevel(103.0, "target", 0.5, directionality="resistance"),
    ]
    clear_score = score_runway(spot=100, direction="long", atr=1, levels=clear, trend_strength=0.8, market=market, options=OptionsContext())
    congested_score = score_runway(spot=100, direction="long", atr=1, levels=congested, trend_strength=0.8, market=market, options=OptionsContext())
    assert clear_score.runway_score > congested_score.runway_score
    assert set(clear_score.components) == {"distance", "congestion", "level_strength", "trend", "market", "options"}
    assert clear_score.next_target == 103.0
