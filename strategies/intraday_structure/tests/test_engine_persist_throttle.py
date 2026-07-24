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
