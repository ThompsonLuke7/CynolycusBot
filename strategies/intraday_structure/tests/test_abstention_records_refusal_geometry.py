"""`invalidation_wider_than_max_atr` must record the width it refused on.

98.8% of this engine's abstentions are that one reason, and until 2026-09-20
every one of them was written with `proposed_invalidation: null` and no width at
all -- `build_target_plan` computed the stop, compared it to the cap, and threw
both numbers away. Measured across 30,000 sampled events: `proposed_invalidation`,
`proposed_target` and `reward_risk` were 0% populated.

That blocked a real question ("widen the cap and size down instead of refusing"),
because the distribution of refused widths was never written down. These tests
pin the fields so it cannot silently regress to null.
"""
from __future__ import annotations

from datetime import datetime, timezone

from strategies.intraday_structure.config import IntradayStructureConfig
from strategies.intraday_structure.detectors.base import DetectionContext
from strategies.intraday_structure.features import FeatureSnapshot
from strategies.intraday_structure.models import (
    Bar, Candidate, Direction, MarketContext, OptionsContext, SetupRecord, SetupState,
    SetupType, StructuralLevel,
)
from strategies.intraday_structure.regime import ABSTENTION_SCHEMA_VERSION, build_abstention_record
from strategies.intraday_structure.target_manager import (
    INVALIDATION_TOO_WIDE, NO_CAUSAL_TARGET, build_target_plan,
)

NOW = datetime(2026, 9, 20, 15, tzinfo=timezone.utc)


def _ctx(*, atr=1.0, target=None) -> DetectionContext:
    bar = Bar("XYZ", NOW, 101, 101.5, 100.8, 101.0, 3000)
    features = FeatureSnapshot(NOW, {"atr": atr, "relative_volume_5m": 1.2,
                                     "trend_strength": 1.0, "micro_swing_low": 100,
                                     "micro_swing_high": 102})
    levels = [StructuralLevel(target, "next_resistance", 0.3, directionality="resistance")] if target else []
    return DetectionContext(bar, [bar], features, levels, MarketContext(NOW, market_alignment_score=0.8),
                            OptionsContext(), IntradayStructureConfig())


def _setup(invalidation: float) -> SetupRecord:
    cand = Candidate("XYZ", NOW, Direction.LONG, ("test",), score=0.8)
    return SetupRecord("x", "XYZ", SetupType.BREAKOUT, Direction.LONG, cand,
                       state=SetupState.ARMED, invalidation=invalidation)


def test_too_wide_refusal_carries_the_geometry_it_refused_on():
    ctx = _ctx(atr=1.0)
    cap = ctx.config.target.max_invalidation_atr
    # close 101, stop at 97 -> 4.0 ATR of risk against a 2.0 cap
    out = build_target_plan(_setup(97.0), ctx)
    assert out.plan is None and out.reason == INVALIDATION_TOO_WIDE
    assert out.proposed_invalidation == 97.0
    assert out.proposed_risk_atr == 4.0
    assert out.max_invalidation_atr == cap
    assert out.proposed_risk_atr > out.max_invalidation_atr


def test_no_causal_target_also_carries_it():
    """An affordable stop with nowhere to go: the width is still worth recording,
    because it separates 'too expensive' from 'nothing to aim at'."""
    out = build_target_plan(_setup(100.0), _ctx(atr=1.0, target=None))
    assert out.plan is None and out.reason == NO_CAUSAL_TARGET
    assert out.proposed_risk_atr == 1.0
    assert out.proposed_invalidation == 100.0


def test_a_granted_plan_is_unaffected():
    out = build_target_plan(_setup(100.0), _ctx(atr=1.0, target=103.0))
    assert out.reason is None and out.plan is not None
    assert out.plan.invalidation == 100.0


def test_record_persists_the_new_fields_and_bumps_the_schema():
    from strategies.intraday_structure.regime import RegimeAssessment

    ctx = _ctx(atr=1.0)
    out = build_target_plan(_setup(97.0), ctx)
    regime = RegimeAssessment(
        regime="TRENDING_UP", evidence=["x"], atr_contraction=1.0,
        trend_strength=1.0, room_to_support_atr=2.4, room_to_resistance_atr=2.6,
        trapped_between_levels=False, failed_break_count=0,
    )
    rec = build_abstention_record(
        _setup(97.0), reason=out.reason, regime=regime, timestamp=NOW, spot=101.0,
        atr=1.0, engine_version="test", min_runway_score=0.45, min_reward_risk=1.25,
        proposed_invalidation=out.proposed_invalidation,
        proposed_risk_atr=out.proposed_risk_atr,
        max_invalidation_atr=out.max_invalidation_atr,
    ).to_dict()
    assert rec["proposed_risk_atr"] == 4.0
    assert rec["max_invalidation_atr"] == ctx.config.target.max_invalidation_atr
    assert rec["proposed_invalidation"] == 97.0
    # v1 readers must not silently treat the new fields as present.
    assert rec["schema_version"] == ABSTENTION_SCHEMA_VERSION == "intraday_structure_abstention_v2"


# --------------------------------------------------------------------------
# Engine wiring: the numbers have to survive the trip to the sink. This is the
# step that was actually broken -- build_target_plan knew the width, and
# _abstain passed `plan.invalidation if plan else None`, which is None on
# exactly the refusal the width describes.
# --------------------------------------------------------------------------

def test_engine_emits_the_refusal_geometry_on_the_too_wide_path():
    from strategies.intraday_structure.detectors.base import DetectionDecision
    from strategies.intraday_structure.engine import IntradayStructureEngine
    from strategies.intraday_structure.models import MarketContext as MC

    records: list = []
    config = IntradayStructureConfig(enabled=True, min_average_dollar_volume=0.0)
    engine = IntradayStructureEngine(config, abstention_sink=records.append)

    bar = Bar("XYZ", NOW, 100, 100.5, 99.5, 100.0, 5000)
    features = FeatureSnapshot(NOW, {
        "atr": 1.0, "atr_contraction": 1.0, "trend_strength": 0.0,
        "distance_to_vwap_atr": 0.0, "relative_volume_5m": 1.2,
        "micro_swing_low": 99.0, "micro_swing_high": 101.0,
    })
    levels = [StructuralLevel(103.0, "next_resistance", 0.3, directionality="resistance")]
    ctx = DetectionContext(bar, [bar], features, levels,
                           MC(NOW, market_alignment_score=0.8), OptionsContext(), config)

    cand = Candidate("XYZ", NOW, Direction.LONG, ("meta_ranker",), score=0.8)
    # stop at 95 against a close of 100 with ATR 1.0 -> 5.0 ATR, over the 2.0 cap
    setup = SetupRecord("XYZ:long:breakout_continuation", "XYZ", SetupType.BREAKOUT,
                        Direction.LONG, cand, state=SetupState.ARMED, invalidation=95.0)

    engine._apply_decision(setup, DetectionDecision(SetupState.CONFIRMED, "HOLD", "confirmed"), ctx)

    assert len(records) == 1
    rec = records[0]
    assert rec.no_trade_reason == INVALIDATION_TOO_WIDE
    assert rec.proposed_invalidation == 95.0, "the refused stop must reach the ledger"
    assert rec.proposed_risk_atr == 5.0
    assert rec.max_invalidation_atr == config.target.max_invalidation_atr
    assert rec.proposed_risk_atr > rec.max_invalidation_atr
