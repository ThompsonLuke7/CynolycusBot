"""The watchdog alerts when a server it saw running disappears — at any hour.

2026-08-20. The combined_server's last log line was 22:43:21 ET, with no exit
code, no OOM signature and no watchdog alert; the supervisor started a fresh
instance at 23:45. The stop fell outside 09:30-16:00 ET, and every alert path
was gated on RTH, so nothing fired. It was not free: the 22:15 data-readiness
job died with the process, its catch-up lost the heavy-job lock race to the
nightly rerun, and Meta's 14:20 ET run the next day skipped five live entries on
a stale readiness stamp.
"""
from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

_ET = ZoneInfo("America/New_York")
_SPEC = importlib.util.spec_from_file_location(
    "heartbeat_watchdog",
    Path(__file__).resolve().parents[2] / "scripts" / "heartbeat_watchdog.py",
)


@pytest.fixture
def watchdog():
    module = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(module)
    return module


class _Clock:
    """Stands in for the module's `time`, so the real one is never patched."""

    def __init__(self):
        # A realistic epoch, not 0.0: the alert cooldown compares
        # time.time() - last_alert_ts against ALERT_COOLDOWN, and a clock
        # starting at zero suppresses the very first alert.
        self.now = 1_800_000_000.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 8, 20, 22, 43, tzinfo=_ET)


class _Loop:
    """Drives main()'s body once per supplied process-alive reading."""

    def __init__(self, module, readings):
        self.readings = list(readings)
        self.alerts = []
        module.alert = self.alerts.append
        module.log = lambda *_a, **_k: None
        module.server_running = self._next
        module.freshest_write_age_secs = lambda: 0.0
        # 22:43 ET, outside RTH: this alert must not need market hours.
        module.datetime = _FrozenDatetime
        module.time = _Clock()

    def _next(self):
        if not self.readings:
            raise KeyboardInterrupt
        return self.readings.pop(0)


def _run(module, readings):
    loop = _Loop(module, readings)
    try:
        module.main()
    except (KeyboardInterrupt, IndexError):
        pass
    return loop.alerts


def test_alerts_when_a_running_server_disappears_out_of_hours(watchdog):
    alerts = _run(watchdog, [True, False])
    assert len(alerts) == 1
    assert "DISAPPEARED" in alerts[0]
    assert "22:43" in alerts[0]


def test_never_started_is_not_an_alert(watchdog):
    """The watchdog must not nag because the server simply is not up."""
    assert _run(watchdog, [False, False, False]) == []


def test_a_healthy_server_is_quiet(watchdog):
    assert _run(watchdog, [True, True, True]) == []
