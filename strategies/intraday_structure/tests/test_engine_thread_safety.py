"""The engine is read and mutated from different threads and must be locked.

2026-08-24: `IntradayStructureEngine` had no lock at all. The runner ingests
bars and registers candidates on its own thread (`intraday-structure-runner`)
while HTTP threads call `snapshot()`/`active_signals()` and POST manual
candidates, so a read could iterate `self.setups` while the runner inserted
into it:

    File "strategies/intraday_structure/engine.py", line 214, in snapshot
      "setups": [setup.to_dict() for setup in self.setups.values()],
    RuntimeError: dictionary changed size during iteration

It surfaced as a ~1-in-3 flake in UI/tests/test_intraday_structure_dashboard.py,
but it is worse than a failed page load: `snapshot()` is also what `persist()`
serializes, so the runner thread could die mid-session and lose state.

These tests shorten the interpreter's thread-switch interval so the interleaving
is hit in milliseconds rather than left to chance. Against an unlocked engine
every one of them fails in ~0.01s; the shortened interval is restored after.
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from strategies.intraday_structure.config import IntradayStructureConfig
from strategies.intraday_structure.engine import IntradayStructureEngine
from strategies.intraday_structure.models import Candidate, Direction

START = datetime(2026, 8, 24, 13, 30, tzinfo=timezone.utc)
_SEED = 150        # enough state that one read spans several thread switches
_BUDGET = 2.0      # seconds; the unlocked engine raises within ~0.01s


@pytest.fixture
def eager_switching():
    """Force the interpreter to switch threads aggressively, then restore."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


def _candidate(name: str, offset: int) -> Candidate:
    return Candidate(name, START + timedelta(seconds=offset), Direction.LONG, ("manual",), score=0.8)


@pytest.fixture
def seeded_engine():
    # No state store: persist() is not what is under test, and serializing on
    # every registration would throttle the writer enough to mask the race.
    config = replace(IntradayStructureConfig(), supported_tickers=(), candidate_limit=10_000)
    engine = IntradayStructureEngine(config, state_store=None)
    for i in range(_SEED):
        engine.register_candidate(_candidate(f"S{i:04d}", i))
    assert engine.setups, "the seed must produce setups, or the read has nothing to race"
    return engine


def _hammer(engine: IntradayStructureEngine, read) -> list[BaseException]:
    """Run `read` against a thread registering candidates; collect what raises."""
    errors: list[BaseException] = []
    stop = threading.Event()

    def writer() -> None:
        i = 0
        while not stop.is_set():
            try:
                engine.register_candidate(_candidate(f"W{i:06d}", i))
            except BaseException as exc:  # noqa: BLE001 - what it raises IS the test
                errors.append(exc)
                stop.set()
                return
            i += 1

    def reader() -> None:
        while not stop.is_set():
            try:
                read()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                stop.set()
                return

    threads = [threading.Thread(target=writer, name="writer"),
               threading.Thread(target=reader, name="reader")]
    deadline = time.monotonic() + _BUDGET
    for t in threads:
        t.start()
    while time.monotonic() < deadline and not stop.is_set():
        time.sleep(0.005)
    stop.set()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive(), f"{t.name} thread hung — a non-reentrant lock would do this"
    return errors


def test_snapshot_is_safe_against_a_concurrent_writer(seeded_engine, eager_switching):
    errors = _hammer(seeded_engine, seeded_engine.snapshot)
    assert errors == [], f"snapshot() raced the writer: {errors!r}"


def test_active_signals_is_safe_against_a_concurrent_writer(seeded_engine, eager_switching):
    errors = _hammer(seeded_engine, seeded_engine.active_signals)
    assert errors == [], f"active_signals() raced the writer: {errors!r}"


def test_persist_is_safe_against_a_concurrent_writer(seeded_engine, eager_switching):
    """persist() serializes snapshot() — this is the path that loses state."""
    saves = []
    seeded_engine.state_store = type("_Store", (), {
        "load": lambda self: None,
        "save": lambda self, state: saves.append(len(state["setups"])),
    })()
    errors = _hammer(seeded_engine, seeded_engine.persist)
    assert errors == [], f"persist() raced the writer: {errors!r}"
    assert saves, "the store should have been written at least once"


def test_the_lock_is_reentrant(seeded_engine):
    """Mutators call persist() -> snapshot() on their way out, and
    write_active_signals() calls active_signals(); a plain Lock would deadlock
    the runner thread on its first registration."""
    with seeded_engine.read_lock():
        assert seeded_engine.register_candidate(_candidate("AAA", 9_999))
        assert seeded_engine.snapshot()["candidates"]
