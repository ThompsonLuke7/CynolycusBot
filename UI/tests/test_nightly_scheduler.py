"""Tests for the combined-server nightly scheduler (pure schedule logic)."""
from __future__ import annotations

import datetime as dt
import threading

import pytest

from UI.nightly_scheduler import NightlyScheduler, next_run_at, parse_hhmm

pytestmark = pytest.mark.safe

ET = dt.timezone(dt.timedelta(hours=-4))  # fixed offset is fine for these checks


def test_parse_hhmm():
    assert parse_hhmm("16:30") == (16, 30)
    assert parse_hhmm(" 9:05 ") == (9, 5)
    for bad in ("1630", "25:00", "10:61", "x:y"):
        with pytest.raises(ValueError):
            parse_hhmm(bad)


def test_next_run_same_day_when_before_target():
    now = dt.datetime(2026, 6, 10, 9, 0, tzinfo=ET)  # Wednesday morning
    nxt = next_run_at(now, 16, 30)
    assert nxt == dt.datetime(2026, 6, 10, 16, 30, tzinfo=ET)


def test_next_run_rolls_to_next_day_after_target():
    now = dt.datetime(2026, 6, 10, 17, 0, tzinfo=ET)  # after 16:30
    nxt = next_run_at(now, 16, 30)
    assert nxt == dt.datetime(2026, 6, 11, 16, 30, tzinfo=ET)


def test_next_run_skips_weekend():
    now = dt.datetime(2026, 6, 12, 17, 0, tzinfo=ET)  # Friday after close
    nxt = next_run_at(now, 16, 30, weekdays_only=True)
    assert nxt.weekday() == 0  # Monday
    assert nxt == dt.datetime(2026, 6, 15, 16, 30, tzinfo=ET)


def test_next_run_weekend_allowed_when_flag_off():
    now = dt.datetime(2026, 6, 12, 17, 0, tzinfo=ET)  # Friday after close
    nxt = next_run_at(now, 16, 30, weekdays_only=False)
    assert nxt == dt.datetime(2026, 6, 13, 16, 30, tzinfo=ET)  # Saturday


def test_scheduler_fires_when_time_reached(monkeypatch):
    """Arm just before the target, advance the clock across it, assert one fire."""
    calls = []
    stop = threading.Event()
    sched = NightlyScheduler(
        lambda: calls.append(1),
        hour=16,
        minute=30,
        stop_event=stop,
        poll_seconds=0.01,
    )
    # Mutable clock: start at 16:29 (arms for today 16:30), then jump past it.
    clock = {"now": dt.datetime(2026, 6, 10, 16, 29, 0, tzinfo=ET)}
    monkeypatch.setattr(sched, "_now", lambda: clock["now"])

    sched.start()
    threading.Event().wait(0.05)              # let the loop arm + sleep
    clock["now"] = dt.datetime(2026, 6, 10, 16, 30, 5, tzinfo=ET)  # cross the target

    deadline = dt.datetime.now() + dt.timedelta(seconds=2)
    while not calls and dt.datetime.now() < deadline:
        threading.Event().wait(0.02)
    # Hold the clock steady a moment to prove the same-day guard prevents re-fire.
    threading.Event().wait(0.1)
    stop.set()
    sched.join(timeout=2)

    assert calls, "scheduler should have fired once the target time was reached"
    assert len(calls) == 1, "same-day guard must prevent repeated firing"
