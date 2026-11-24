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
    assert [c[0] for c in calls] == ["bars", "context", "matrix"]
    # bars step must use the fast incremental path
    bars_argv = calls[0][1]
    assert "--eligible-only" in bars_argv and "--no-1d" in bars_argv
    # context step refreshes the 4H context caches only: an intraday daily fetch
    # would write today's partial daily bar into the shared 1d cache
    ctx_argv = calls[1][1]
    assert ctx_argv[-2:] == ["scripts/refresh_shared_context_bars.py", "--context-only"]


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
    assert calls == ["bars", "context", "feeds", "matrix"]


def test_sleep_wakes_for_the_next_bar_ready_anchor():
    # 14:17:00 -> the 14:20:30 anchor is 210 s away, sooner than a 300 s interval.
    assert sdr.seconds_until_next_anchor(_at(2026, 11, 20, 14, 17)) == 210
    assert sdr.seconds_until_next_anchor(_at(2026, 11, 20, 15, 0)) == 80 * 60 + 30
    assert sdr.seconds_until_next_anchor(_at(2026, 11, 20, 16, 30)) is None


def test_a_failed_step_logs_its_output_tail(tmp_path, caplog):
    """2026-11-23: three matrix failures logged only 'FAILED'; the cause went to /dev/null."""
    import logging
    import sys

    from UI import shared_data_refresher as mod

    r = mod.SharedDataRefresher.__new__(mod.SharedDataRefresher)
    r._repo = tmp_path
    with caplog.at_level(logging.INFO, logger=mod.logger.name):
        ok = r._run_step("matrix", [sys.executable, "-c",
                                    "import sys; print('step one'); sys.exit('ABORT: bars incomplete')"],
                         timeout=30)
    assert ok is False
    text = caplog.text
    assert "matrix FAILED" in text
    assert "exit=1" in text and "ABORT: bars incomplete" in text and "step one" in text


def test_a_successful_step_logs_no_output(tmp_path, caplog):
    import logging
    import sys

    from UI import shared_data_refresher as mod

    r = mod.SharedDataRefresher.__new__(mod.SharedDataRefresher)
    r._repo = tmp_path
    with caplog.at_level(logging.INFO, logger=mod.logger.name):
        assert r._run_step("bars", [sys.executable, "-c", "print('noise')"], timeout=30)
    assert "noise" not in caplog.text
