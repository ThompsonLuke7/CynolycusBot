"""The manual watchlist has to survive past its own TTL, and cover both sides."""

from __future__ import annotations

import queue
from dataclasses import replace as dc_replace
from datetime import datetime, timedelta, timezone

from strategies.intraday_structure.candidate_sources import ManualCandidateFeed, manual_candidates
from strategies.intraday_structure.config import IntradayStructureConfig, load_config
from strategies.intraday_structure.models import Direction


NOON_ET = datetime(2026, 8, 28, 16, 0, tzinfo=timezone.utc)  # 12:00 ET


def test_a_watchlist_symbol_is_seeded_on_both_sides() -> None:
    # Long-only silently excluded every rejection setup: price turning down at
    # resistance is a SHORT, which is exactly what a call-wall study needs.
    got = manual_candidates(["SPY"], timestamp=NOON_ET)
    assert {c.direction for c in got} == {Direction.LONG, Direction.SHORT}


def test_the_feed_reseeds_once_per_session_not_once_per_process() -> None:
    feed = ManualCandidateFeed(["SPY", "QQQ"])
    first = feed.poll(now=NOON_ET)
    assert len(first) == 4  # 2 symbols x 2 directions

    assert feed.poll(now=NOON_ET + timedelta(hours=1)) == [], "same session, no re-seed"

    # The old behaviour registered once per PROCESS; with a 1,440-minute TTL the
    # watchlist was empty on any server running longer than a day.
    next_session = feed.poll(now=NOON_ET + timedelta(days=1))
    assert len(next_session) == 4
    assert next_session[0].available_at > first[0].available_at


def test_the_session_boundary_is_eastern_not_utc() -> None:
    feed = ManualCandidateFeed(["SPY"])
    # 2026-08-29 01:00 UTC is still 2026-08-28 in ET.
    feed.poll(now=datetime(2026, 8, 28, 20, 0, tzinfo=timezone.utc))
    assert feed.poll(now=datetime(2026, 8, 29, 1, 0, tzinfo=timezone.utc)) == []
    assert feed.poll(now=datetime(2026, 8, 29, 16, 0, tzinfo=timezone.utc)) != []


def test_an_empty_watchlist_is_silent() -> None:
    assert ManualCandidateFeed([]).poll(now=NOON_ET) == []
    assert ManualCandidateFeed(["", "  "]).poll(now=NOON_ET) == []


def test_symbols_are_normalised() -> None:
    feed = ManualCandidateFeed([" spy ", "qqq"])
    assert {c.ticker for c in feed.poll(now=NOON_ET)} == {"SPY", "QQQ"}


def test_the_shipped_config_watches_the_five_study_a_etfs() -> None:
    config = load_config()
    assert set(config.manual_watchlist) == {"SPY", "QQQ", "IWM", "GLD", "SLV"}


def test_the_runner_wires_the_feed_and_it_survives_a_second_poll(tmp_path) -> None:
    from strategies.intraday_structure.runner import IntradayStructureRunner

    config = dc_replace(
        IntradayStructureConfig(enabled=True, min_average_dollar_volume=0.0),
        manual_watchlist=("GLD", "SLV"),
        state_path=str(tmp_path / "s.json"),
        transition_log_path=str(tmp_path / "t.jsonl"),
        signal_path=str(tmp_path / "sig.json"),
        ledger_path=str(tmp_path / "l.jsonl"),
        abstention_path=str(tmp_path / "a.jsonl"),
        bar_archive_root=str(tmp_path / "archive"),
    )
    runner = IntradayStructureRunner(config, queue.Queue())
    for candidate in runner.manual_feed.poll(now=NOON_ET):
        runner.engine.register_candidate(candidate)

    keys = set(runner.engine.candidates)
    assert ("GLD", Direction.LONG) in keys and ("GLD", Direction.SHORT) in keys
    assert ("SLV", Direction.SHORT) in keys
    # A short candidate must actually produce short setups to confirm on.
    assert any(s.direction == Direction.SHORT for s in runner.engine.setups.values())


def test_a_hand_picked_symbol_is_never_evicted_by_the_candidate_limit() -> None:
    """Eviction sorts by score ascending and manual seeds carry a neutral 0.5,
    so without a guard the ranker would silently evict the watchlist."""
    from strategies.intraday_structure.engine import IntradayStructureEngine
    from strategies.intraday_structure.models import Candidate

    config = IntradayStructureConfig(
        enabled=True, min_average_dollar_volume=0.0, candidate_limit=3,
        manual_watchlist=("GLD",),
    )
    engine = IntradayStructureEngine(config)
    for candidate in manual_candidates(["GLD"], timestamp=NOON_ET):
        engine.register_candidate(candidate)

    # Flood with higher-scoring candidates from a ranker.
    for i in range(8):
        engine.register_candidate(Candidate(
            ticker=f"T{i}", timestamp=NOON_ET, direction=Direction.LONG,
            sources=("meta_ranker",), score=0.99,
        ))

    survivors = {t for t, _ in engine.candidates}
    assert "GLD" in survivors, "the watchlist must outlive the limit"
    assert len(engine.candidates) <= max(config.candidate_limit, 2)


def test_the_limit_still_bites_on_ordinary_candidates() -> None:
    from strategies.intraday_structure.engine import IntradayStructureEngine
    from strategies.intraday_structure.models import Candidate

    config = IntradayStructureConfig(enabled=True, min_average_dollar_volume=0.0, candidate_limit=3)
    engine = IntradayStructureEngine(config)
    for i in range(10):
        engine.register_candidate(Candidate(
            ticker=f"T{i}", timestamp=NOON_ET, direction=Direction.LONG,
            sources=("meta_ranker",), score=0.1 * i,
        ))
    assert len(engine.candidates) == 3
    assert {t for t, _ in engine.candidates} == {"T7", "T8", "T9"}, "lowest scores go first"
