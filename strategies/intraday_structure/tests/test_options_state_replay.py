from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from strategies.intraday_structure.config import IntradayStructureConfig
from strategies.intraday_structure.candidate_sources import DealerRankingCandidateFeed, LiquidityCandidateFeed
from strategies.intraday_structure.dealer_plate import evaluate_dealer_plate
from strategies.intraday_structure.engine import IntradayStructureEngine
from strategies.intraday_structure.models import Bar, Candidate, Direction, OptionsContext, PriceUpdate, SetupState, StructuralLevel
from strategies.intraday_structure.options import (
    DealerLevelSummaryOptionsProvider,
    DealerSnapshotOptionsProvider,
    LiveOptionsFlowProvider,
    OptionFlowPrint,
)
from strategies.intraday_structure.replay import EventReplay
from strategies.intraday_structure.state_store import JsonStateStore


def test_static_options_provider_rejects_future_snapshot(tmp_path) -> None:
    out = tmp_path / "XYZ"
    out.mkdir()
    (out / "live_gamma_levels.json").write_text(json.dumps({
        "timestamp": "2026-07-18T15:00:00+00:00", "total_gex": -1000,
        "call_wall": 105, "put_wall": 95, "gamma_flip": 100,
    }))
    provider = DealerSnapshotOptionsProvider(tmp_path)
    rejected = provider.context("XYZ", datetime(2026, 7, 17, 15, tzinfo=timezone.utc), 100)
    accepted = provider.context("XYZ", datetime(2026, 7, 19, 15, tzinfo=timezone.utc), 100)
    assert rejected.source == "none"
    assert "future_dealer_snapshot_rejected" in rejected.warnings
    assert accepted.call_wall == 105
    assert accepted.gamma_regime == "negative"


def test_broad_dealer_summary_is_asof_and_strength_weighted() -> None:
    captured = "2026-07-16T19:45:00+00:00"
    frame = pd.DataFrame([
        {
            "captured_at": captured, "symbol": "XYZ", "scope": "daily_week", "spot": 100,
            "total_gex": -2_000_000, "total_abs_gex": 1_000_000_000, "call_wall": 103,
            "put_wall": 96, "gamma_flip": 99, "next_magnet_above": 102, "next_magnet_below": 97,
            "magnet_strength": 500_000_000, "call_gex_total": 600_000_000, "put_gex_total": -300_000_000,
            "wall_dominance": 0.9, "gamma_density": 900_000_000, "gex_concentration_index": 0.8,
            "dealer_imbalance": 0.4,
        },
        {
            "captured_at": captured, "symbol": "WEAK", "scope": "daily_week", "spot": 100,
            "total_gex": 2_000, "total_abs_gex": 10_000, "call_wall": 103,
            "put_wall": 96, "gamma_flip": 99, "next_magnet_above": 102, "next_magnet_below": 97,
            "magnet_strength": 50, "call_gex_total": 100, "put_gex_total": -100,
            "wall_dominance": 0.1, "gamma_density": 100, "gex_concentration_index": 0.1,
            "dealer_imbalance": 0.0,
        },
    ])
    provider = DealerLevelSummaryOptionsProvider.from_frame(frame)
    future = provider.context("XYZ", datetime(2026, 7, 16, 19, tzinfo=timezone.utc), 100)
    accepted = provider.context("XYZ", datetime(2026, 7, 17, 14, tzinfo=timezone.utc), 100)
    weak = provider.context("WEAK", datetime(2026, 7, 17, 14, tzinfo=timezone.utc), 100)
    assert future.source == "none"
    assert "future_dealer_summary_rejected" in future.warnings
    assert accepted.source == "dealer_level_summary_static"
    assert accepted.call_wall == 103
    assert accepted.dealer_strength_score > weak.dealer_strength_score
    assert any("magnet_above" in level.level_type for level in accepted.levels)


def test_dealer_plate_requires_strong_reachable_asof_target() -> None:
    options = OptionsContext(
        source="dealer_level_summary_static", dealer_imbalance=0.5, dealer_strength_score=0.9,
        levels=(StructuralLevel(102, "options_dealer_call_wall", 0.9, directionality="resistance"),),
    )
    result = evaluate_dealer_plate(
        direction=Direction.LONG, spot=100, atr=1, options=options,
        policy=IntradayStructureConfig().dealer_plate,
    )
    assert result.qualified
    assert result.target == 102
    assert "favorable_dealer_swing_plate" in result.evidence
    weak = evaluate_dealer_plate(
        direction=Direction.LONG, spot=100, atr=1,
        options=OptionsContext(source="dealer_level_summary_static", levels=(StructuralLevel(100.2, "options_dealer_call_wall", 0.2, directionality="resistance"),)),
        policy=IntradayStructureConfig().dealer_plate,
    )
    assert not weak.qualified


def test_dealer_ranking_feed_keeps_structural_and_change_candidates(tmp_path) -> None:
    captured = datetime(2026, 7, 16, 19, 45, tzinfo=timezone.utc)
    path = tmp_path / "rankings.parquet"
    pd.DataFrame([
        {"symbol": "TOP", "captured_at": captured, "dealer_swing_rank": 2, "dealer_change_intensity_rank": 90},
        {"symbol": "CHANGER", "captured_at": captured, "dealer_swing_rank": 500, "dealer_change_intensity_rank": 3, "dealer_change_direction": "neutral"},
        {"symbol": "SKIP", "captured_at": captured, "dealer_swing_rank": 500, "dealer_change_intensity_rank": 90},
    ]).to_parquet(path, index=False)
    feed = DealerRankingCandidateFeed(path, top_structural=5, top_change=5, max_age_hours=30)
    candidates = feed.poll(now=captured + timedelta(hours=10))
    assert {candidate.ticker for candidate in candidates} == {"TOP", "CHANGER"}
    changer = next(candidate for candidate in candidates if candidate.ticker == "CHANGER")
    assert changer.available_at == captured
    assert not feed.poll(now=captured + timedelta(hours=10))


def test_liquidity_feed_is_bounded_eligible_and_reseeds_next_session(tmp_path) -> None:
    path = tmp_path / "shared_universe.csv"
    pd.DataFrame([
        {"ticker": "LOW", "is_eligible": True, "avg_dollar_volume_20d": 10_000_000},
        {"ticker": "HIGH", "is_eligible": True, "avg_dollar_volume_20d": 30_000_000},
        {"ticker": "MID", "is_eligible": True, "avg_dollar_volume_20d": 20_000_000},
        {"ticker": "INELIGIBLE", "is_eligible": False, "avg_dollar_volume_20d": 99_000_000},
    ]).to_csv(path, index=False)
    feed = LiquidityCandidateFeed(path, top_n=2)
    now = datetime(2026, 7, 17, 14, tzinfo=timezone.utc)
    candidates = feed.poll(now=now)
    assert [candidate.ticker for candidate in candidates] == ["HIGH", "MID"]
    assert all(candidate.average_dollar_volume >= 20_000_000 for candidate in candidates)
    assert not feed.poll(now=now + timedelta(hours=1))
    assert {candidate.ticker for candidate in feed.poll(now=now + timedelta(days=1))} == {"HIGH", "MID"}


def test_newer_liquidity_seed_preserves_dealer_context(tmp_path) -> None:
    config = replace(IntradayStructureConfig(), state_path=str(tmp_path / "state.json"))
    engine = IntradayStructureEngine(config)
    dealer_time = datetime(2026, 7, 16, 20, tzinfo=timezone.utc)
    liquidity_time = dealer_time + timedelta(hours=17)
    assert engine.register_candidate(Candidate(
        "MU", dealer_time, Direction.LONG, ("dealer_level_map",), score=0.9,
        metadata={"dealer_swing_rank": 1},
    ))
    assert engine.register_candidate(Candidate(
        "MU", liquidity_time, Direction.LONG, ("high_liquidity_universe",), score=0.5,
        average_dollar_volume=100_000_000,
    ))
    merged = engine.candidates[("MU", Direction.LONG)]
    assert merged.timestamp == liquidity_time
    assert merged.score == 0.9
    assert set(merged.sources) == {"dealer_level_map", "high_liquidity_universe"}
    assert merged.metadata["dealer_swing_rank"] == 1


def test_live_flow_provider_aggregates_only_asof_prints() -> None:
    provider = LiveOptionsFlowProvider(window_minutes=10)
    ts = datetime(2026, 7, 17, 15, tzinfo=timezone.utc)
    provider.ingest(OptionFlowPrint(ts, "XYZ", 100, "2026-07-17", "C", 10, 1000, 1.0, bid=0.9, ask=1.0, delta=0.5))
    provider.ingest(OptionFlowPrint(ts + timedelta(minutes=2), "XYZ", 100, "2026-07-17", "P", 5, 500, 1.0, bid=1.0, ask=1.1, delta=-0.5))
    context = provider.context("XYZ", ts, 100)
    assert context.source == "live_options_flow"
    assert context.net_delta_premium > 0


def test_duplicate_suppression_and_restart_recovery(tmp_path) -> None:
    config = replace(IntradayStructureConfig(), state_path=str(tmp_path / "state.json"))
    store = JsonStateStore(config.state_path)
    engine = IntradayStructureEngine(config, state_store=store)
    candidate = Candidate("XYZ", datetime(2026, 7, 17, 13, 30, tzinfo=timezone.utc), Direction.LONG, ("manual",), score=0.8)
    assert engine.register_candidate(candidate)
    assert not engine.register_candidate(candidate)
    assert len(engine.setups) == 5
    restored = IntradayStructureEngine(config, state_store=store)
    assert restored.restore_from_store()
    assert len(restored.setups) == 5
    assert list(restored.candidates.values())[0].ticker == "XYZ"


def test_candidate_eligibility_and_fast_price_update(tmp_path) -> None:
    config = replace(
        IntradayStructureConfig(), supported_tickers=("XYZ",),
        state_path=str(tmp_path / "state.json"),
    )
    engine = IntradayStructureEngine(config)
    now = datetime(2026, 7, 17, 15, tzinfo=timezone.utc)
    assert not engine.register_candidate(Candidate("ABC", now, Direction.LONG, ("test",), average_dollar_volume=10_000_000))
    assert not engine.register_candidate(Candidate("XYZ", now, Direction.LONG, ("test",), average_dollar_volume=1_000))
    assert engine.register_candidate(Candidate("XYZ", now, Direction.LONG, ("test",), average_dollar_volume=10_000_000))
    setup = next(iter(engine.setups.values()))
    setup.state = SetupState.RUNNING
    setup.invalidation = 99.0
    setup.targets = [101.0]
    setup.spot = 100.0
    setup.updated_at = now
    transitions = engine.on_price_update(PriceUpdate("XYZ", now + timedelta(seconds=5), 101.1, event_type="trade"))
    assert transitions[-1].to_state == SetupState.TARGET_REACHED


def test_replay_is_deterministic_and_uses_one_minute_bars() -> None:
    start = datetime(2026, 7, 17, 13, 30, tzinfo=timezone.utc)
    rows = []
    for i in range(50):
        close = 99.7 + 0.002 * i if i < 25 else 100.25 + 0.01 * (i - 25)
        for symbol in ("QQQ", "SPY"):
            rows.append({"symbol": symbol, "timestamp": start + timedelta(minutes=i), "open": close - 0.01, "high": close + 0.05, "low": close - 0.05, "close": close, "volume": 5000})
        rows.append({"symbol": "XYZ", "timestamp": start + timedelta(minutes=i), "open": close - 0.02, "high": close + 0.10, "low": close - 0.08, "close": close, "volume": 3000 if i < 25 else 8000})
    bars = pd.DataFrame(rows)
    candidate = Candidate(
        "XYZ", start, Direction.LONG, ("test",), score=0.9, pivot=100.0,
        metadata={"structural_levels": [{"price": 102.0, "level_type": "target", "strength": 0.5, "directionality": "resistance"}]},
    )
    base = IntradayStructureConfig()
    config = replace(base, target=replace(base.target, min_runway_score=0.0, min_reward_risk=0.0))
    one = EventReplay(config).run(bars, [candidate])
    two = EventReplay(config).run(bars, [candidate])
    pd.testing.assert_frame_equal(one.transitions, two.transitions)
    assert one.metrics == two.metrics
    five_minute = bars[bars["symbol"] == "XYZ"].iloc[::5].copy()
    with pytest.raises(ValueError, match="true 1-minute"):
        EventReplay(config).run(five_minute, [candidate])
