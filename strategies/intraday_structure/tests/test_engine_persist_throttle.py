"""on_bar() must only persist when a real state transition happened.

2026-07-20/21 live audit: the shared bar-stream queue backed up in the last
~40min of RTH even after throttling the candidate-feed I/O. Root cause:
`persist()` serializes the WHOLE engine state (candidates/setups/histories --
17.9MB by end of day) and was called unconditionally on every `on_bar()` that
touched an active candidate, so the per-bar cost scaled with total
accumulated state instead of being O(1). `on_price_update()` already only
persists on an actual transition; `on_bar()` now matches that pattern.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from strategies.intraday_structure.config import IntradayStructureConfig
from strategies.intraday_structure.engine import IntradayStructureEngine
from strategies.intraday_structure.models import Bar, Candidate, Direction


class _SpyStore:
    def __init__(self) -> None:
        self.saves = 0

    def load(self):
        return None

    def save(self, state) -> None:
        self.saves += 1


def test_on_bar_does_not_persist_without_a_transition(tmp_path):
    config = replace(IntradayStructureConfig(), state_path=str(tmp_path / "state.json"))
    store = _SpyStore()
    engine = IntradayStructureEngine(config, state_store=store)

    candidate = Candidate(
        "XYZ", datetime(2026, 7, 21, 13, 30, tzinfo=timezone.utc), Direction.LONG, ("manual",), score=0.8,
    )
    engine.register_candidate(candidate)  # registration itself still persists
    saves_after_register = store.saves
    assert saves_after_register >= 1

    # A single bar is far short of min_history_bars, so _evaluate_candidate
    # returns before any transition can fire.
    bar = Bar("XYZ", datetime(2026, 7, 21, 13, 31, tzinfo=timezone.utc), 10.0, 10.1, 9.9, 10.0, 1_000.0)
    transitions = engine.on_bar(bar)

    assert transitions == []
    assert store.saves == saves_after_register  # no extra save for a no-op bar


def test_on_bar_persists_when_a_transition_actually_fires(tmp_path):
    config = replace(IntradayStructureConfig(), state_path=str(tmp_path / "state.json"))
    store = _SpyStore()
    engine = IntradayStructureEngine(config, state_store=store)

    candidate = Candidate(
        "XYZ", datetime(2026, 7, 21, 13, 30, tzinfo=timezone.utc), Direction.LONG, ("manual",), score=0.8,
    )
    engine.register_candidate(candidate)
    saves_after_register = store.saves

    # A bar priced below the configured minimum forces every open setup
    # straight to CLOSED -- a real, deterministic transition to assert against.
    bar = Bar(
        "XYZ", datetime(2026, 7, 21, 13, 31, tzinfo=timezone.utc),
        0.01, 0.01, 0.01, 0.01, 1_000.0,
    )
    transitions = engine.on_bar(bar)

    assert len(transitions) > 0
    assert store.saves == saves_after_register + 1


def _closing_bar(ticker, minute):
    # Priced below the configured minimum: every open setup goes straight to CLOSED.
    return Bar(ticker, datetime(2026, 11, 2, 14, minute, tzinfo=timezone.utc), 0.01, 0.01, 0.01, 0.01, 1_000.0)


def test_transitions_inside_the_window_are_saved_once_then_flushed(tmp_path, monkeypatch):
    """The open emits 700+ transitions an hour; a ~2.4 s save per transition kept
    the engine 17-21 minutes behind the tape on 2026-10-30 and 11-02."""
    import strategies.intraday_structure.engine as engine_module

    clock = {"now": 1000.0}
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: clock["now"])
    config = replace(IntradayStructureConfig(), state_path=str(tmp_path / "state.json"))
    store = _SpyStore()
    engine = IntradayStructureEngine(config, state_store=store)
    for ticker in ("AAA", "BBB"):
        engine.register_candidate(Candidate(
            ticker, datetime(2026, 11, 2, 14, 30, tzinfo=timezone.utc), Direction.LONG, ("manual",), score=0.8,
        ))
    base = store.saves

    assert engine.on_bar(_closing_bar("AAA", 31))
    assert store.saves == base + 1                     # the first transition saves at once
    clock["now"] += 5
    assert engine.on_bar(_closing_bar("BBB", 31))
    assert store.saves == base + 1                     # held inside the window
    engine.flush_if_due()
    assert store.saves == base + 1
    clock["now"] += engine_module.TRANSITION_PERSIST_SECONDS
    engine.flush_if_due()
    assert store.saves == base + 2                     # flushed once the window passed
    engine.flush_if_due()
    assert store.saves == base + 2                     # nothing left to save


def test_a_forced_flush_saves_held_transitions_immediately(tmp_path, monkeypatch):
    import strategies.intraday_structure.engine as engine_module

    monkeypatch.setattr(engine_module.time, "monotonic", lambda: 1000.0)
    config = replace(IntradayStructureConfig(), state_path=str(tmp_path / "state.json"))
    store = _SpyStore()
    engine = IntradayStructureEngine(config, state_store=store)
    for ticker in ("AAA", "BBB"):
        engine.register_candidate(Candidate(
            ticker, datetime(2026, 11, 2, 14, 30, tzinfo=timezone.utc), Direction.LONG, ("manual",), score=0.8,
        ))
    engine.on_bar(_closing_bar("AAA", 31))
    engine.on_bar(_closing_bar("BBB", 31))
    held = store.saves

    engine.flush_if_due(force=True)

    assert store.saves == held + 1
