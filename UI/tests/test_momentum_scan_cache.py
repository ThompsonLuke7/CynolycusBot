"""The momentum dashboard's scan cache must never block a request thread.

Regression coverage for 2026-08-24. `_scan` used to build under a lock and
stamp the cache with the time the build STARTED. A full-universe
`evaluate_now()` measured 1018.2s against `SCAN_TTL = 120.0`, so every entry
was ~15 minutes stale the moment it was written: the post-build re-check missed,
and each of the queued pollers (hub fan-out + the page's own 5s tick) went on to
run its own full scan. The live session ran 41 back-to-back scans in 14.5h and
accumulated ~1770 threads parked on that lock plus 1635 CLOSE-WAIT sockets on
port 8770. The process could not reach interpreter exit on shutdown, so it
outlived its supervisor and kept every dashboard port bound -- which is what
made the next `run_live_server.sh` refuse to start.
"""
from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from UI import momentum_dashboard as md


@pytest.fixture()
def app(monkeypatch):
    """A dashboard app with no broker client and no real scan behind it."""
    monkeypatch.setattr(md, "AlpacaOptionsClient", lambda **kw: object())
    return md.MomentumDashboardApp(env_file=".env#PAPER", top_k=3)


def _install_build(app, monkeypatch, *, duration: float, calls: list):
    """Make the real ``_build_scan`` run against a runner that just sleeps.

    Deliberately NOT a patch of ``_build_scan`` itself: the bug lived in that
    method's stamping, so the tests have to execute it.
    """
    class FakeRunner:
        def __init__(self, **kw):
            calls.append(time.time())

        def evaluate_now(self):
            time.sleep(duration)
            return [{"rank": 1, "ticker": "MRNA", "expansion_score": 1.25,
                     "trigger_rule": "4h_break", "bar_close": 42.0}]

    stub = types.ModuleType("strategies.momentum_expansion.live.runner")
    stub.MomentumLiveRunner = FakeRunner
    monkeypatch.setitem(sys.modules, "strategies.momentum_expansion.live.runner", stub)


def test_scan_returns_without_waiting_for_the_build(app, monkeypatch):
    """A poll must return promptly even while a long scan is in flight."""
    calls: list = []
    _install_build(app, monkeypatch, duration=2.0, calls=calls)

    t0 = time.time()
    first = app._scan()
    elapsed = time.time() - t0

    assert elapsed < 0.5, f"_scan blocked for {elapsed:.2f}s on the build"
    assert first["picks"] == []
    assert first["building"] is True
    assert first["ts"] is None          # nothing cached yet, so nothing to serve


def test_concurrent_polls_share_one_build(app, monkeypatch):
    """N pollers must not launch N scans -- that was the original pileup."""
    calls: list = []
    _install_build(app, monkeypatch, duration=1.0, calls=calls)

    threads = [threading.Thread(target=app._scan) for _ in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert all(not t.is_alive() for t in threads), "a poller blocked on the build"
    assert len(calls) == 1, f"{len(calls)} concurrent builds started, expected 1"


def test_cache_is_stamped_after_the_build_not_before(app, monkeypatch):
    """A build slower than SCAN_TTL must still produce a FRESH entry.

    This is the exact inversion that wedged the live session: stamped with the
    start time, an entry from a build longer than the TTL is born expired and
    the very next poll starts another one.
    """
    monkeypatch.setattr(md, "SCAN_TTL", 1.0)
    calls: list = []
    _install_build(app, monkeypatch, duration=1.5, calls=calls)   # build > TTL

    app._scan()                       # kicks off the build
    time.sleep(2.0)                   # let it finish
    assert len(calls) == 1

    served = app._scan()              # must be a HIT, not a second build
    assert served["stale"] is False, "cache entry was stale on arrival"
    assert served["building"] is False
    assert served["picks"] == [{"rank": 1, "ticker": "MRNA", "expansion": 1.25,
                                "trigger": "4h_break", "close": 42.0}]
    assert len(calls) == 1, "a fresh cache entry still triggered a rebuild"


def test_expired_cache_is_served_while_it_refreshes(app, monkeypatch):
    """Past the TTL, the poller gets the old picks -- it does not wait."""
    monkeypatch.setattr(md, "SCAN_TTL", 0.5)
    calls: list = []
    _install_build(app, monkeypatch, duration=2.0, calls=calls)

    with app._lock:                                    # seed a completed scan
        app._scan_cache = (time.time(), {"picks": [{"ticker": "MRNA"}], "ts": "old"})
    time.sleep(0.7)                                    # let it age past the TTL

    t0 = time.time()
    out = app._scan()
    assert time.time() - t0 < 0.5, "poller waited on the refresh"
    assert out["picks"] == [{"ticker": "MRNA"}]        # stale, but served
    assert out["stale"] is True
    assert out["building"] is True
    assert out["age_s"] >= 0.5
