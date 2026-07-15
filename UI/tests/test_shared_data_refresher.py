"""Tests for the background shared-data refresher's window + cycle wiring."""
from __future__ import annotations

import datetime as dt
import threading

import UI.shared_data_refresher as sdr
from UI.shared_data_refresher import SharedDataRefresher, _tz, is_within_window

ET = _tz("America/New_York")


def _at(y, m, d, hh, mm):
    return dt.datetime(y, m, d, hh, mm, tzinfo=ET)


def test_window_boundaries():
    # Monday 2026-06-29; window default 13:45-16:40 keeps startup/open trading-only
    # while still covering the 14:20 and 16:20 4H decisions.
    assert not is_within_window(_at(2026, 6, 29, 13, 44))  # before afternoon refresh
    assert is_within_window(_at(2026, 6, 29, 13, 45))      # open inclusive
    assert is_within_window(_at(2026, 6, 29, 14, 20))      # covers 2 PM decision
    assert is_within_window(_at(2026, 6, 29, 16, 20))      # covers post-close decision
    assert not is_within_window(_at(2026, 6, 29, 16, 40))  # close exclusive


def test_weekend_closed():
    assert not is_within_window(_at(2026, 6, 27, 12, 0))   # Saturday
    assert not is_within_window(_at(2026, 6, 28, 12, 0))   # Sunday


def test_run_cycle_issues_bars_then_matrix(monkeypatch):
    class _Guard:
        ok = True
        reason = "test"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(sdr, "heavy_job_guard", lambda *a, **k: _Guard())
    r = SharedDataRefresher(stop_event=threading.Event())
    calls: list[tuple[str, list[str]]] = []
    r._run_step = lambda label, argv, timeout: calls.append((label, argv)) or True  # type: ignore

    r.run_cycle(with_feeds=False)
    assert [c[0] for c in calls] == ["bars", "matrix"]
    # bars step must use the fast incremental path
    bars_argv = calls[0][1]
    assert "--eligible-only" in bars_argv and "--no-1d" in bars_argv


def test_run_cycle_with_feeds_inserts_light_feeds(monkeypatch):
    class _Guard:
        ok = True
        reason = "test"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(sdr, "heavy_job_guard", lambda *a, **k: _Guard())
    r = SharedDataRefresher(stop_event=threading.Event(), feeds_enabled=True)
    calls: list[str] = []
    r._run_step = lambda label, argv, timeout: calls.append(label) or True  # type: ignore

    r.run_cycle(with_feeds=True)
    assert calls == ["bars", "feeds", "matrix"]
