"""Regression: a closed setup must be able to come back.

The revival block used to sit inside ``register_candidate``'s ``current is None``
branch, so it was only reachable when the candidate key was ABSENT. A candidate
refreshed more often than its 1,440-minute TTL -- which is exactly what
``LiquidityCandidateFeed`` does with the same top-ADV names every session -- kept
its CLOSED setups dark forever, and ``_cooldown_complete`` was dead code.

Measured against live state on 2026-08-26: 204 of 630 setups sitting under a
still-live candidate were CLOSED and unrevivable. They looked healthy because
``_evaluate_candidate`` bumps ``updated_at`` before it checks for CLOSED.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from strategies.intraday_structure.config import IntradayStructureConfig
from strategies.intraday_structure.engine import IntradayStructureEngine
from strategies.intraday_structure.models import Bar, Candidate, Direction, SetupState, SetupType


NOW = datetime(2026, 8, 26, 13, 30, tzinfo=timezone.utc)


def _engine(**overrides) -> IntradayStructureEngine:
    config = IntradayStructureConfig(
        enabled=True, min_average_dollar_volume=0.0, failure_cooldown_bars=5, **overrides,
    )
    return IntradayStructureEngine(config)


def _candidate(offset_minutes: int = 0, **kwargs) -> Candidate:
    defaults = dict(
        ticker="XYZ", timestamp=NOW + timedelta(minutes=offset_minutes),
        direction=Direction.LONG, sources=("high_liquidity_universe",), score=0.5,
    )
    defaults.update(kwargs)
    return Candidate(**defaults)


def test_a_closed_setup_revives_when_its_candidate_is_re_registered() -> None:
    engine = _engine()
    engine.register_candidate(_candidate())
    setup_id = "XYZ:long:breakout_continuation"
    engine.setups[setup_id].state = SetupState.CLOSED
    engine._bar_counts["XYZ"] = 100  # cooldown well past

    # The candidate key is STILL live -- this is precisely the case that used to
    # return early before reaching the revival block.
    assert ("XYZ", Direction.LONG) in engine.candidates
    engine.register_candidate(_candidate(offset_minutes=390))

    assert engine.setups[setup_id].state == SetupState.WATCHING
    assert engine.setups[setup_id].bars_alive == 0


def test_the_cooldown_still_holds_a_setup_back() -> None:
    engine = _engine()
    engine.register_candidate(_candidate())
    setup_id = "XYZ:long:breakout_continuation"
    engine.setups[setup_id].state = SetupState.CLOSED
    engine.setups[setup_id].metadata["terminal_bar_count"] = 100
    engine._bar_counts["XYZ"] = 102  # only 2 of 5 cooldown bars elapsed

    engine.register_candidate(_candidate(offset_minutes=390))
    assert engine.setups[setup_id].state == SetupState.CLOSED, "cooldown must still bite"

    engine._bar_counts["XYZ"] = 106
    engine.register_candidate(_candidate(offset_minutes=780))
    assert engine.setups[setup_id].state == SetupState.WATCHING


def test_a_live_setup_is_refreshed_not_reset() -> None:
    engine = _engine()
    engine.register_candidate(_candidate())
    setup = engine.setups["XYZ:long:breakout_continuation"]
    setup.state = SetupState.ARMED
    setup.bars_alive = 42

    engine.register_candidate(_candidate(offset_minutes=10, sources=("meta_ranker",), score=0.9))
    survivor = engine.setups["XYZ:long:breakout_continuation"]
    assert survivor.state == SetupState.ARMED, "an in-flight setup must not be reset"
    assert survivor.bars_alive == 42
    assert "meta_ranker" in survivor.candidate.sources


def test_an_older_registration_does_not_roll_availability_back() -> None:
    engine = _engine()
    engine.register_candidate(_candidate(offset_minutes=60, sources=("high_liquidity_universe",)))
    before = engine.candidates[("XYZ", Direction.LONG)].available_at

    engine.register_candidate(_candidate(offset_minutes=0, sources=("dealer_ranker",), score=0.9))
    merged = engine.candidates[("XYZ", Direction.LONG)]
    assert merged.available_at == before, "an older seed must not roll availability back"
    # ...but its context is still absorbed.
    assert set(merged.sources) == {"high_liquidity_universe", "dealer_ranker"}
    assert merged.score == 0.9


def test_the_broad_watchlist_no_longer_goes_dark_across_sessions() -> None:
    """End-to-end: the LiquidityCandidateFeed pattern -- same name, every session."""
    engine = _engine()
    revivals = 0
    for session in range(5):
        engine.register_candidate(_candidate(offset_minutes=session * 390))
        setup = engine.setups["XYZ:long:breakout_continuation"]
        if setup.state == SetupState.WATCHING:
            revivals += 1
        # Simulate the setup running its course and closing during the session.
        setup.state = SetupState.CLOSED
        engine._bar_counts["XYZ"] += 50
    assert revivals == 5, f"expected a fresh setup every session, got {revivals}"


def test_registering_an_unchanged_candidate_twice_is_not_reported_as_a_change() -> None:
    engine = _engine()
    assert engine.register_candidate(_candidate()) is True
    assert engine.register_candidate(_candidate()) is False, "an identical re-register is a no-op"


def test_revival_produces_a_fresh_ledger_lifecycle() -> None:
    rows: list = []
    config = IntradayStructureConfig(enabled=True, min_average_dollar_volume=0.0, failure_cooldown_bars=1)
    engine = IntradayStructureEngine(config, ledger_sink=rows.append)
    engine.register_candidate(_candidate())
    setup_id = "XYZ:long:breakout_continuation"

    setup = engine.setups[setup_id]
    setup.state = SetupState.RUNNING
    setup.entry_price = 100.0
    setup.entry_time = NOW
    setup.metadata.update({"initial_invalidation": 99.0, "exit_price": 99.0})
    bar = Bar("XYZ", NOW + timedelta(minutes=5), 99.5, 99.6, 98.9, 99.0, 1000)
    engine._transition(setup, SetupState.INVALIDATED, bar, "invalidation touched", ("stop",))
    assert len(rows) == 1

    setup.state = SetupState.CLOSED
    engine._bar_counts["XYZ"] = 100
    engine.register_candidate(_candidate(offset_minutes=390))
    revived = engine.setups[setup_id]
    # A revived setup is a NEW lifecycle: no ledger guard, no inherited entry.
    assert revived.entry_price is None
    assert "ledgered_at" not in revived.metadata
