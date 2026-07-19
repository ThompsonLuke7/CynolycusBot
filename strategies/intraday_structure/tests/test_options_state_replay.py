from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from strategies.intraday_structure.config import IntradayStructureConfig
from strategies.intraday_structure.engine import IntradayStructureEngine
from strategies.intraday_structure.models import Bar, Candidate, Direction, PriceUpdate, SetupState
from strategies.intraday_structure.options import (
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
