from __future__ import annotations

import json
import queue
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd

from strategies.intraday_structure.candidate_sources import (
    CatalystCandidateFeed,
    OpeningMomentumCandidateFeed,
)
from strategies.intraday_structure.config import (
    CatalystDiscoveryPolicy,
    CandidateCapacityPolicy,
    EvidencePolicy,
    IntradayStructureConfig,
    OpeningDiscoveryPolicy,
)
from strategies.intraday_structure.engine import IntradayStructureEngine
from strategies.intraday_structure.evidence import (
    CandidateOutcomeTracker,
    DecisionEventLedger,
    render_decision_summary,
    summarize_decision_events,
)
from strategies.intraday_structure.models import Bar, Candidate, Direction
from strategies.intraday_structure.runner import IntradayStructureRunner


OPEN = datetime(2026, 8, 27, 13, 30, tzinfo=timezone.utc)


def _universe(path, rows) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_opening_feed_promotes_a_gap_volume_leader_with_separate_clocks(tmp_path) -> None:
    universe = tmp_path / "universe.csv"
    _universe(universe, [
        {"ticker": "CRM", "is_eligible": True, "px": 100.0, "avg_dollar_volume_20d": 10_000_000},
        {"ticker": "SPY", "is_eligible": True, "px": 500.0, "avg_dollar_volume_20d": 1_000_000_000},
    ])
    feed = OpeningMomentumCandidateFeed(OpeningDiscoveryPolicy(
        universe_path=str(universe), top_n=10, min_volume_pace=1.0,
    ))
    observed = OPEN + timedelta(minutes=15)  # delayed feed: do not hide this clock
    got = feed.observe(Bar("CRM", OPEN, 105.0, 106.0, 104.8, 105.5, 20_000), observed_at=observed)
    assert len(got) == 1
    candidate = got[0]
    assert candidate.sources == ("opening_momentum",)
    assert candidate.direction == Direction.LONG
    assert candidate.timestamp == OPEN
    assert candidate.available_at == observed
    assert candidate.metadata["discovery_trigger"] == "gap_volume_leader"
    assert candidate.metadata["market_data_lag_seconds"] == 900.0
    assert feed.observe(Bar("CRM", OPEN + timedelta(minutes=1), 105.5, 107, 105, 106.5, 20_000)) == []


def test_opening_feed_promotes_a_non_gap_acceleration_leader(tmp_path) -> None:
    universe = tmp_path / "universe.csv"
    _universe(universe, [
        {"ticker": "PTEN", "is_eligible": True, "px": 10.0, "avg_dollar_volume_20d": 5_000_000},
    ])
    feed = OpeningMomentumCandidateFeed(OpeningDiscoveryPolicy(
        universe_path=str(universe), top_n=10, min_volume_pace=0.5,
        min_session_range_pct=0.005, min_relative_strength_pct=0.001,
    ))
    closes = [9.98, 10.01, 10.04, 10.10]
    got = []
    for index, close in enumerate(closes):
        got.extend(feed.observe(Bar(
            "PTEN", OPEN + timedelta(minutes=index),
            9.98 if index == 0 else closes[index - 1], close + 0.02,
            min(9.96, close - 0.02), close, 100_000,
        ), observed_at=OPEN + timedelta(minutes=index, seconds=5)))
    assert len(got) == 1
    assert got[0].metadata["discovery_trigger"] == "rth_acceleration_leader"


def test_promoted_candidate_can_be_warmed_with_only_pre_signal_bars(tmp_path) -> None:
    universe = tmp_path / "universe.csv"
    _universe(universe, [
        {"ticker": "CRM", "is_eligible": True, "px": 100.0, "avg_dollar_volume_20d": 10_000_000},
    ])
    feed = OpeningMomentumCandidateFeed(OpeningDiscoveryPolicy(
        universe_path=str(universe), top_n=10, min_volume_pace=0.5,
    ))
    for minute in range(19):
        assert feed.observe(Bar(
            "CRM", OPEN + timedelta(minutes=minute), 100, 100.1, 99.9, 100, 5_000,
        ), observed_at=OPEN + timedelta(minutes=minute, seconds=5)) == []
    signal_bar = Bar("CRM", OPEN + timedelta(minutes=19), 105, 106, 104.8, 105.5, 50_000)
    candidate = feed.observe(signal_bar, observed_at=OPEN + timedelta(minutes=19, seconds=5))[0]
    warmup = feed.history("CRM", before=candidate.timestamp)
    assert len(warmup) == 19
    assert all(bar.timestamp < candidate.timestamp for bar in warmup)

    engine = IntradayStructureEngine(replace(
        IntradayStructureConfig(), min_average_dollar_volume=0,
    ))
    assert engine.seed_history(warmup) == 19
    assert engine.register_candidate(candidate)
    engine.on_bar(signal_bar)
    assert len(engine.histories["CRM"]) == 20


def test_catalyst_feed_requires_material_relevant_non_recap_news(tmp_path) -> None:
    universe = tmp_path / "universe.csv"
    _universe(universe, [
        {"ticker": ticker, "is_eligible": True, "px": 100.0, "avg_dollar_volume_20d": 20_000_000}
        for ticker in ("CRWD", "NOW", "HUT")
    ])
    ledger = tmp_path / "catalysts.parquet"
    scored = OPEN + timedelta(minutes=1)
    pd.DataFrame([
        {
            "record_id": "good", "content_hash": "good", "ticker": "CRWD",
            "timestamp": OPEN, "scored_at": scored,
            "headline": "CRWD Q2 Earnings Beat Estimates on ARR Strength",
            "source": "test", "catalyst_family": "company_news",
            "catalyst_subtype": "general_company_news", "catalyst_score": 0.80,
            "information_direction": "other",
        },
        {
            "record_id": "ambiguous", "content_hash": "ambiguous", "ticker": "NOW",
            "timestamp": OPEN, "scored_at": scored,
            "headline": "Here Is My Top Space Stock to Buy Right Now",
            "source": "test", "catalyst_family": "company_news",
            "catalyst_subtype": "general_company_news", "catalyst_score": 0.82,
            "information_direction": "other",
        },
        {
            "record_id": "recap", "content_hash": "recap", "ticker": "HUT",
            "timestamp": OPEN, "scored_at": scored,
            "headline": "Hut 8 Shares Gap Up - Time to Buy?",
            "source": "test", "catalyst_family": "company_news",
            "catalyst_subtype": "general_company_news", "catalyst_score": 0.75,
            "information_direction": "price_recap",
        },
    ]).to_parquet(ledger, index=False)
    feed = CatalystCandidateFeed(
        CatalystDiscoveryPolicy(
            ledger_path=str(ledger), universe_path=str(universe), refresh_seconds=0,
        ),
        allowed_symbols={"CRWD", "NOW", "HUT"},
    )
    got = feed.poll(now=scored + timedelta(minutes=1))
    assert [candidate.ticker for candidate in got] == ["CRWD"]
    assert got[0].available_at == scored
    rejected = {row["ticker"]: row["reason"] for row in feed.last_decisions if row["event_type"] == "catalyst_rejected"}
    assert rejected["NOW"] == "ambiguous_ticker_subject_not_explicit"
    assert rejected["HUT"] == "backward_looking_price_recap"


def test_capacity_reserves_opening_and_catalyst_candidates() -> None:
    events = []
    capacity = CandidateCapacityPolicy(opening_reserve=1, catalyst_reserve=1)
    config = replace(
        IntradayStructureConfig(), candidate_limit=3, min_average_dollar_volume=0,
        candidate_capacity=capacity,
    )
    engine = IntradayStructureEngine(config, candidate_event_sink=events.append)
    for index in range(3):
        engine.register_candidate(Candidate(
            f"BASE{index}", OPEN + timedelta(seconds=index), Direction.LONG,
            ("high_liquidity_universe",), score=0.99 - index * 0.01,
        ))
    assert engine.register_candidate(Candidate(
        "OPEN", OPEN + timedelta(minutes=1), Direction.LONG,
        ("opening_momentum",), score=0.60,
    ))
    assert engine.register_candidate(Candidate(
        "NEWS", OPEN + timedelta(minutes=2), Direction.LONG,
        ("validated_catalyst",), score=0.62,
    ))
    retained = {candidate.ticker for candidate in engine.candidates.values()}
    assert {"OPEN", "NEWS"}.issubset(retained)
    assert len(retained) == 3
    assert any(row["event_type"] == "candidate_evicted" for row in events)


def test_candidate_batch_can_persist_once_instead_of_once_per_symbol() -> None:
    saves = []
    store = type("Store", (), {
        "save": lambda self, state: saves.append(len(state["candidates"])),
        "load": lambda self: None,
    })()
    engine = IntradayStructureEngine(
        replace(IntradayStructureConfig(), candidate_limit=10), state_store=store,
    )
    for index in range(3):
        assert engine.register_candidate(Candidate(
            f"BATCH{index}", OPEN, Direction.LONG, ("high_liquidity_universe",), score=0.5,
        ), persist=False)
    assert saves == []
    engine.persist()
    assert saves == [3]


def test_candidate_outcomes_use_next_bar_and_emit_fixed_horizon_mfe_mae(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    ledger = DecisionEventLedger(path)
    tracker = CandidateOutcomeTracker(ledger, horizons_minutes=(5,))
    candidate = Candidate(
        "CRM", OPEN, Direction.LONG, ("opening_momentum",), score=0.8,
        available_at=OPEN + timedelta(seconds=15),
    )
    tracker.track(candidate)
    tracker.on_bar(Bar("CRM", OPEN, 99, 150, 1, 120, 1_000))  # signal bar must be ignored
    for minute in range(1, 7):
        tracker.on_bar(Bar(
            "CRM", OPEN + timedelta(minutes=minute), 100, 101 + minute,
            99 - minute / 10, 100 + minute / 2, 1_000,
        ))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    payload = rows[0]["payload"]
    assert payload["entry_time"] == (OPEN + timedelta(minutes=1)).isoformat()
    assert payload["entry_price"] == 100
    assert payload["horizon_minutes"] == 5
    assert payload["mfe_return"] > 0
    assert payload["mae_return"] < 0
    assert payload["model_event_time_only"] is True
    assert payload["candidate_availability_lag_seconds"] == 15.0


def test_runner_wires_the_funnel_and_finalizes_archive_manifest(tmp_path) -> None:
    config = replace(
        IntradayStructureConfig(enabled=True, min_average_dollar_volume=0),
        opening_discovery=OpeningDiscoveryPolicy(enabled=False),
        catalyst_discovery=CatalystDiscoveryPolicy(enabled=False),
        evidence=EvidencePolicy(
            enabled=True, event_path=str(tmp_path / "decision_events.jsonl"),
            outcome_horizons_minutes=(5,),
        ),
        state_path=str(tmp_path / "state.json"),
        transition_log_path=str(tmp_path / "transitions.jsonl"),
        signal_path=str(tmp_path / "signals.json"),
        ledger_path=str(tmp_path / "closed.jsonl"),
        abstention_path=str(tmp_path / "abstentions.jsonl"),
        bar_archive_root=str(tmp_path / "archive"),
    )
    runner = IntradayStructureRunner(config, queue.Queue())
    assert runner._register(Candidate("CRM", OPEN, Direction.LONG, ("opening_momentum",), score=0.8))
    events = [json.loads(line) for line in (tmp_path / "decision_events.jsonl").read_text().splitlines()]
    assert events[-1]["event_type"] == "candidate_registered"

    payload = {
        "symbol": "CRM", "timestamp": OPEN.isoformat(),
        "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1_000,
    }
    runner._record_archive(payload)
    next_day = dict(payload, timestamp=(OPEN + timedelta(days=1)).isoformat())
    runner._record_archive(next_day)
    manifest = tmp_path / "archive" / "manifest_20260827.json"
    assert manifest.exists()
    manifest_row = json.loads(manifest.read_text())
    assert manifest_row["bar_count"] == 1
    assert manifest_row["engine_version"] == config.version


def test_funnel_report_groups_candidate_outcomes_without_calling_them_fills() -> None:
    rows = [
        {
            "event_type": "candidate_registered",
            "payload": {"candidate": {"sources": ["opening_momentum"]}},
        },
        {
            "event_type": "setup_transition",
            "payload": {"to_state": "CONFIRMED"},
        },
        {
            "event_type": "candidate_fixed_horizon_outcome",
            "payload": {
                "candidate_sources": ["opening_momentum"], "horizon_minutes": 15,
                "signed_return": 0.02, "mfe_return": 0.03, "mae_return": -0.01,
            },
        },
    ]
    summary = summarize_decision_events(rows)
    assert summary["funnel"]["candidate_registered"] == 1
    assert summary["funnel"]["setup_confirmed"] == 1
    assert summary["fixed_horizon_outcomes"][0]["mean_signed_return"] == 0.02
    assert summary["fixed_horizon_outcomes"][0]["underpowered"] is True
    assert "not broker fills" in render_decision_summary(summary)
