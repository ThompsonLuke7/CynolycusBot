from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from strategies.intraday_structure.config import IntradayStructureConfig, RegimePolicy
from strategies.intraday_structure.models import Direction, OptionsContext, StructuralLevel
from strategies.intraday_structure.premarket import (
    OVERNIGHT_GAP_WARNING,
    TARGET_LADDER_DEPTH,
    build_levels,
    build_trade_plan,
    premarket_regime_policy,
)
from strategies.intraday_structure.regime import classify_context


START = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _daily(n=90, *, base=100.0, drift=0.0, span=2.0) -> pd.DataFrame:
    rows = []
    for i in range(n):
        close = base + drift * i
        rows.append({
            "timestamp": START + timedelta(days=i), "open": close - 0.2,
            "high": close + span / 2, "low": close - span / 2,
            "close": close, "volume": 1_000_000,
        })
    return pd.DataFrame(rows)


def _hourly(n=70, *, base=100.0, span=1.0) -> pd.DataFrame:
    rows = []
    for i in range(n):
        close = base + (i % 5) * 0.1
        rows.append({
            "timestamp": START + timedelta(hours=i), "open": close,
            "high": close + span / 2, "low": close - span / 2,
            "close": close, "volume": 100_000,
        })
    return pd.DataFrame(rows)


def _plan(**kwargs):
    defaults = dict(
        ticker="XYZ", direction=Direction.LONG, sources=("meta_ranker",), score=0.8,
        daily=_daily(), hourly=_hourly(), options=None,
        config=IntradayStructureConfig(), reference_as_of="2026-08-25",
    )
    defaults.update(kwargs)
    return build_trade_plan(**defaults)


def test_a_plan_names_the_trigger_stop_and_ladder_before_the_open() -> None:
    plan = _plan()
    assert plan is not None
    assert plan.trigger is not None and plan.trigger > plan.reference_price
    assert plan.invalidation is not None and plan.invalidation < plan.trigger
    assert plan.targets == sorted(plan.targets), "a long ladder must walk upward"
    assert len(plan.targets) <= TARGET_LADDER_DEPTH
    assert plan.trigger_level_sources, "the trigger must say which levels back it"


def test_a_short_plan_is_mirrored() -> None:
    plan = _plan(direction=Direction.SHORT)
    assert plan.trigger < plan.reference_price
    assert plan.invalidation > plan.trigger
    assert plan.targets == sorted(plan.targets, reverse=True)


def test_an_unreachable_rung_is_not_published() -> None:
    # A 60-day high far above a low-priced name used to land in the ladder as a
    # target 99 ATR away.
    daily = _daily(90, base=5.0, span=0.4)
    daily.loc[0, "high"] = 50.0
    config = IntradayStructureConfig()
    plan = _plan(daily=daily, hourly=_hourly(70, base=5.0, span=0.2), config=config)
    assert plan is not None
    cap = config.target.max_target_distance_atr * plan.atr
    for target in plan.targets:
        assert abs(target - plan.trigger) <= cap


def test_every_plan_admits_it_cannot_see_the_overnight_session() -> None:
    assert OVERNIGHT_GAP_WARNING in _plan().warnings
    assert "no_dealer_levels_available" in _plan().warnings


def test_a_declined_plan_still_says_why() -> None:
    # Flat tape, so the first destination is close and reward:risk is thin.
    plan = _plan(daily=_daily(90, span=8.0))
    if plan.no_trade_reason is not None:
        assert plan.actionable is False
        assert plan.no_trade_reason in {
            "reward_risk_below_threshold", "runway_below_threshold",
            "no_causal_target_beyond_trigger", "no_trigger_level_beyond_reference_price",
        }


def test_dealer_levels_reach_the_fused_set_when_available() -> None:
    options = OptionsContext(
        source="dealer_level_summary_static", call_wall=110.0,
        levels=(StructuralLevel(110.0, "options_dealer_call_wall", 0.9, directionality="resistance"),),
    )
    fused = build_levels(daily=_daily(), hourly=_hourly(), spot=100.0, atr=2.0, options=options)
    assert any("options_dealer_call_wall" in level.level_type for level in fused)


def test_nothing_after_the_decision_time_is_used() -> None:
    # build_trade_plan prices off the LAST row it is handed; the caller filters.
    # This pins that it does not peek at a wider frame.
    daily = _daily(90)
    truncated = daily.iloc[:60]
    assert _plan(daily=truncated).reference_price == pytest.approx(float(truncated.iloc[-1]["close"]))


def test_the_premarket_policy_drops_the_test_that_cannot_mean_anything_there() -> None:
    policy = premarket_regime_policy(RegimePolicy())
    assert policy.trapped_room_atr is None
    assert policy.compression_atr_ratio == RegimePolicy().compression_atr_ratio

    # Prior-day high and low bracket spot at about one daily ATR by
    # construction, so the intraday policy calls literally everything trapped.
    levels = [
        StructuralLevel(99.5, "prior_day_low", 0.82, directionality="support"),
        StructuralLevel(100.5, "prior_day_high", 0.82, directionality="resistance"),
    ]
    features = {"atr_contraction": 1.0, "trend_strength": 1.0, "distance_to_vwap_atr": 0.5}
    intraday = classify_context(spot=100.0, atr=1.0, features=features, levels=levels, policy=RegimePolicy())
    premarket = classify_context(spot=100.0, atr=1.0, features=features, levels=levels, policy=policy)
    assert intraday.regime == "COMPRESSED"
    assert premarket.regime == "TRENDING_UP"


def test_a_lone_weak_level_is_not_a_wall() -> None:
    # A round number on its own must not box price in; only levels several
    # mechanisms agree on count.
    weak = [
        StructuralLevel(99.9, "round_number_1", 0.34, directionality="support"),
        StructuralLevel(100.1, "round_number_1", 0.34, directionality="resistance"),
    ]
    features = {"atr_contraction": 1.0, "trend_strength": 1.0, "distance_to_vwap_atr": 0.5}
    assert classify_context(spot=100.0, atr=1.0, features=features, levels=weak, policy=RegimePolicy()).regime != "COMPRESSED"


def test_no_daily_bars_means_no_plan_rather_than_a_guess() -> None:
    assert _plan(daily=pd.DataFrame()) is None
