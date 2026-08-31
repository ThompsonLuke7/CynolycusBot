from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from strategies.intraday_structure.config import IntradayStructureConfig, RegimePolicy
from strategies.intraday_structure.detectors.base import DetectionContext, DetectionDecision
from strategies.intraday_structure.engine import IntradayStructureEngine
from strategies.intraday_structure.features import FeatureSnapshot, compute_features
from strategies.intraday_structure.models import (
    Bar, Candidate, Direction, MarketContext, OptionsContext, SetupRecord, SetupState,
    SetupType, StructuralLevel,
)
from strategies.intraday_structure.regime import (
    BALANCED, COMPRESSED, TRENDING_DOWN, TRENDING_UP, classify_context, regime_conflicts,
)


NOW = datetime(2026, 8, 25, 15, 0, tzinfo=timezone.utc)


def _classify(levels, *, policy=None, **feature_overrides):
    features = {"atr_contraction": 1.0, "trend_strength": 0.0, "distance_to_vwap_atr": 0.0}
    features.update(feature_overrides)
    return classify_context(
        spot=100.0, atr=1.0, features=features, levels=levels,
        policy=policy or RegimePolicy(),
    )


def test_a_coiling_tape_is_compressed() -> None:
    assessment = _classify([], atr_contraction=0.60, trend_strength=1.5)
    assert assessment.regime == COMPRESSED
    assert "range_contraction" in assessment.evidence
    # Compression wins over a strong trend reading: a coiled tape is what the
    # caller needs to hear about, and trend inside one is not informative.
    assert assessment.trend_strength == 1.5


def test_price_boxed_between_nearby_levels_is_compressed() -> None:
    levels = [
        StructuralLevel(99.5, "prior_day_low", 0.8, directionality="support"),
        StructuralLevel(100.5, "prior_day_high", 0.8, directionality="resistance"),
    ]
    assessment = _classify(levels)
    assert assessment.regime == COMPRESSED
    assert assessment.trapped_between_levels is True
    assert assessment.room_to_support_atr == pytest.approx(0.5)
    assert assessment.room_to_resistance_atr == pytest.approx(0.5)


def test_wide_levels_are_not_a_trap() -> None:
    levels = [
        StructuralLevel(95.0, "prior_day_low", 0.8, directionality="support"),
        StructuralLevel(105.0, "prior_day_high", 0.8, directionality="resistance"),
    ]
    assert _classify(levels, trend_strength=0.8, distance_to_vwap_atr=0.4).regime == TRENDING_UP
    assert _classify(levels, trend_strength=-0.8, distance_to_vwap_atr=-0.4).regime == TRENDING_DOWN
    assert _classify(levels).regime == BALANCED


def test_repeatedly_rejected_levels_read_as_compressed() -> None:
    levels = [
        StructuralLevel(95.0, "liquidity_support_zone", 0.6, directionality="support", rejection_count=2),
        StructuralLevel(105.0, "liquidity_resistance_zone", 0.6, directionality="resistance", rejection_count=2),
    ]
    assessment = _classify(levels, trend_strength=0.8, distance_to_vwap_atr=0.4)
    assert assessment.regime == COMPRESSED
    assert assessment.failed_break_count == 4
    assert "repeated_failed_breaks" in assessment.evidence


def test_a_trend_argues_against_fading_it_but_not_against_joining_it() -> None:
    assert regime_conflicts(TRENDING_UP, "short") is True
    assert regime_conflicts(TRENDING_UP, "long") is False
    assert regime_conflicts(COMPRESSED, "long") is True
    assert regime_conflicts(COMPRESSED, "short") is True
    assert regime_conflicts(BALANCED, "long") is False


def test_atr_contraction_is_computed_and_finite() -> None:
    quiet = [Bar("XYZ", NOW + timedelta(minutes=i), 100, 100.2, 99.8, 100, 1000) for i in range(70)]
    features = compute_features(quiet)
    assert features.get("atr_contraction") == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------------------
# Engine: the refusal must be emitted, not swallowed.
# --------------------------------------------------------------------------

def _ctx(config: IntradayStructureConfig, *, atr=1.0) -> DetectionContext:
    bar = Bar("XYZ", NOW, 100, 100.5, 99.5, 100.0, 5000)
    features = FeatureSnapshot(NOW, {
        "atr": atr, "atr_contraction": 1.0, "trend_strength": 0.0,
        "distance_to_vwap_atr": 0.0, "relative_volume_5m": 1.2,
        "micro_swing_low": 99.0, "micro_swing_high": 101.0,
    })
    # One resistance far enough to be a legal target but too close in R terms.
    levels = [StructuralLevel(100.2, "next_resistance", 0.3, directionality="resistance")]
    return DetectionContext(bar, [bar], features, levels, MarketContext(NOW, market_alignment_score=0.8), OptionsContext(), config)


def _setup() -> SetupRecord:
    candidate = Candidate("XYZ", NOW, Direction.LONG, ("meta_ranker",), score=0.8)
    return SetupRecord(
        setup_id="XYZ:long:breakout_continuation", ticker="XYZ",
        setup_type=SetupType.BREAKOUT, direction=Direction.LONG,
        candidate=candidate, state=SetupState.ARMED, invalidation=99.0,
    )


def test_a_refused_confirmation_is_recorded_rather_than_dropped() -> None:
    records: list = []
    config = IntradayStructureConfig(enabled=True, min_average_dollar_volume=0.0)
    engine = IntradayStructureEngine(config, abstention_sink=records.append)
    setup = _setup()
    engine._apply_decision(setup, DetectionDecision(SetupState.CONFIRMED, "HOLD", "confirmed"), _ctx(config))

    assert setup.state == SetupState.ARMED, "an abstention must not advance the setup"
    assert len(records) == 1
    record = records[0]
    assert record.no_trade_reason == "reward_risk_below_threshold"
    assert record.reward_risk is not None and record.reward_risk < config.target.min_reward_risk
    assert record.min_reward_risk == config.target.min_reward_risk
    assert record.context_regime in {TRENDING_UP, TRENDING_DOWN, BALANCED, COMPRESSED}
    assert record.candidate_sources == ["meta_ranker"]
    assert setup.metadata["no_trade_reason"] == "reward_risk_below_threshold"


def test_the_regime_veto_is_off_by_default_and_measurable_when_on() -> None:
    ctx_config = IntradayStructureConfig(enabled=True, min_average_dollar_volume=0.0)
    # A compressed tape with an otherwise acceptable plan.
    bar = Bar("XYZ", NOW, 100, 100.5, 99.5, 100.0, 5000)
    features = FeatureSnapshot(NOW, {
        "atr": 1.0, "atr_contraction": 0.5, "trend_strength": 0.0,
        "distance_to_vwap_atr": 0.0, "relative_volume_5m": 1.2,
        "micro_swing_low": 99.0, "micro_swing_high": 101.0,
    })
    levels = [StructuralLevel(105.0, "next_resistance", 0.3, directionality="resistance")]

    def run(veto: bool):
        config = IntradayStructureConfig(
            enabled=True, min_average_dollar_volume=0.0,
            regime=RegimePolicy(veto_enabled=veto),
        )
        ctx = DetectionContext(bar, [bar], features, levels, MarketContext(NOW, market_alignment_score=0.8), OptionsContext(), config)
        records: list = []
        engine = IntradayStructureEngine(config, abstention_sink=records.append)
        setup = _setup()
        engine._apply_decision(setup, DetectionDecision(SetupState.CONFIRMED, "HOLD", "confirmed"), ctx)
        return setup, records

    setup_off, records_off = run(False)
    assert setup_off.state == SetupState.CONFIRMED
    assert records_off == []
    # ...but the label is recorded either way, which is the point.
    assert setup_off.metadata["context_regime"] == COMPRESSED

    setup_on, records_on = run(True)
    assert setup_on.state == SetupState.ARMED
    assert records_on[0].no_trade_reason == "regime_conflict_compressed"


def test_a_failing_abstention_sink_cannot_break_the_engine() -> None:
    def explode(_record):
        raise OSError("disk full")

    config = IntradayStructureConfig(enabled=True, min_average_dollar_volume=0.0)
    engine = IntradayStructureEngine(config, abstention_sink=explode)
    setup = _setup()
    engine._apply_decision(setup, DetectionDecision(SetupState.CONFIRMED, "HOLD", "confirmed"), _ctx(config))
    assert setup.state == SetupState.ARMED


def test_the_report_refuses_to_blend_rows_priced_under_different_assumptions() -> None:
    from strategies.intraday_structure.reporting import build_report, render_report

    rows = [
        {"net_return": 0.01, "cost_spread_bps": 8.0, "cost_slippage_bps": 4.0, "cost_commission_per_share": 0.0},
        {"net_return": -0.01, "cost_spread_bps": 20.0, "cost_slippage_bps": 4.0, "cost_commission_per_share": 0.0},
    ]
    text = render_report(build_report(rows))
    assert "more than one cost assumption" in text

    single = render_report(build_report(rows[:1]))
    assert "more than one cost assumption" not in single


def test_a_thin_bucket_is_shown_but_flagged() -> None:
    from strategies.intraday_structure.reporting import MIN_REPORTABLE_N, build_report, render_report

    rows = [{"net_return": 0.05, "setup_type": "breakout_continuation",
             "cost_spread_bps": 8.0, "cost_slippage_bps": 4.0, "cost_commission_per_share": 0.0}] * 3
    text = render_report(build_report(rows))
    assert "breakout_continuation" in text, "a thin bucket must still be shown"
    assert f"fewer than {MIN_REPORTABLE_N} setups" in text
