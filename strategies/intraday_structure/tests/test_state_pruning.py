"""persist() must stay cheap: settled state is pruned before it is written.

2026-10-26: state.json had reached 188 MB (4,623 CLOSED setups, bar histories
for 862 symbols against 140 candidates). Each persist() took 18-20 s and ran on
every transition (~1,700 a session), so the engine sat 50-95 minutes behind the
tape and starved the SPY daytrader's inference thread in the same process.
"""
from __future__ import annotations

import queue
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from strategies.intraday_structure.config import IntradayStructureConfig
from strategies.intraday_structure.engine import IntradayStructureEngine
from strategies.intraday_structure.models import Bar, Candidate, Direction, SetupState
from strategies.intraday_structure.runner import IntradayStructureRunner
from strategies.intraday_structure.state_store import JsonStateStore

T0 = datetime(2026, 10, 26, 14, 0, tzinfo=timezone.utc)


def _bar(symbol: str, at: datetime) -> Bar:
    return Bar(symbol, at, 10.0, 10.1, 9.9, 10.0, 1_000.0)


def _engine(tmp_path) -> IntradayStructureEngine:
    config = replace(IntradayStructureConfig(), state_path=str(tmp_path / "state.json"))
    return IntradayStructureEngine(config, state_store=JsonStateStore(config.state_path))


def _closed_setup(engine, ticker: str, *, terminal_bar_count: int):
    engine.register_candidate(Candidate(ticker, T0, Direction.LONG, ("manual",), score=0.8))
    setup = next(s for s in engine.setups.values() if s.ticker == ticker)
    setup.state = SetupState.CLOSED
    setup.metadata["terminal_bar_count"] = terminal_bar_count
    engine.candidates.pop((ticker, Direction.LONG), None)
    return setup.setup_id


def test_a_closed_setup_past_its_cooldown_is_pruned_and_one_inside_it_is_kept(tmp_path):
    engine = _engine(tmp_path)
    cooldown = engine.config.failure_cooldown_bars
    settled = _closed_setup(engine, "OLD", terminal_bar_count=0)
    engine._bar_counts["OLD"] = cooldown
    cooling = _closed_setup(engine, "NEW", terminal_bar_count=0)
    engine._bar_counts["NEW"] = cooldown - 1

    engine.persist()

    assert settled not in engine.setups
    assert cooling in engine.setups


def test_only_dormant_histories_of_symbols_nothing_watches_are_dropped(tmp_path):
    engine = _engine(tmp_path)
    engine.register_candidate(Candidate("LIVE", T0, Direction.LONG, ("manual",), score=0.8))
    engine.histories["LIVE"] = [_bar("LIVE", T0 - timedelta(days=30))]
    engine.histories["SPY"] = [_bar("SPY", T0 - timedelta(days=30))]      # context symbol
    engine.histories["RECENT"] = [_bar("RECENT", T0 - timedelta(days=2))]
    engine.histories["GONE"] = [_bar("GONE", T0 - timedelta(days=12))]
    engine.histories["NOW"] = [_bar("NOW", T0)]

    engine.persist()

    assert set(engine.histories) == {"LIVE", "SPY", "RECENT", "NOW"}


def test_the_compact_state_restores_what_was_saved(tmp_path):
    engine = _engine(tmp_path)
    engine.register_candidate(Candidate("XYZ", T0, Direction.LONG, ("manual",), score=0.8))
    engine.histories["XYZ"] = [_bar("XYZ", T0)]
    engine.persist()

    restored = _engine(tmp_path)
    assert restored.restore_from_store() is True
    assert restored.snapshot() == engine.snapshot()
    assert "\n" not in (tmp_path / "state.json").read_text()


class _CountingEngine:
    def __init__(self) -> None:
        self.saves = 0

    def persist(self) -> None:
        self.saves += 1


def test_opening_registrations_are_saved_once_per_window_not_once_each(tmp_path, monkeypatch):
    runner = IntradayStructureRunner.__new__(IntradayStructureRunner)
    runner.engine = _CountingEngine()
    runner._unsaved_registrations = False
    runner._next_registration_persist = 0.0
    clock = iter([1000.0, 1001.0, 1002.0, 1031.0])
    monkeypatch.setattr("strategies.intraday_structure.runner.time.monotonic", lambda: next(clock))

    for _ in range(4):
        runner._unsaved_registrations = True
        runner._persist_registrations()

    # Saved at t=1000, held through 1001 and 1002, saved again at 1031.
    assert runner.engine.saves == 2
    assert runner._unsaved_registrations is False


def test_a_quiet_tape_still_flushes_held_registrations(tmp_path, monkeypatch):
    config = replace(
        IntradayStructureConfig(),
        state_path=str(tmp_path / "state.json"), archive_bars=False,
        transition_log_path=str(tmp_path / "t.jsonl"), signal_path=str(tmp_path / "s.json"),
        ledger_path=str(tmp_path / "l.jsonl"), abstention_path=str(tmp_path / "a.jsonl"),
        execution=replace(IntradayStructureConfig().execution, enabled=False),
    )
    runner = IntradayStructureRunner(config, queue.Queue())

    class _NoCandidates:
        last_decisions = ()

        def poll(self, **_kwargs):
            return []

    for name in ("feed", "dealer_feed", "liquidity_feed", "catalyst_feed", "manual_feed"):
        setattr(runner, name, _NoCandidates())
    runner._unsaved_registrations = True
    saves = []
    monkeypatch.setattr(runner.engine, "persist", lambda: saves.append(1))
    monkeypatch.setattr(runner, "_flatten_expiring", lambda: runner._stop.set())

    runner._run()  # empty queue -> the Empty branch runs once, then stop

    assert saves == [1]
