"""The risk pass supervisor's thread body actually runs.

`RiskPassSupervisor.run` is marked `# pragma: no cover - thread body`, and
nothing exercised it, so a NameError sat inside it: `datetime.now(tz)` against a
module that never imported datetime. The loop catches Exception and logs
"tick failed; continuing", so it degraded into a thread that woke every 5
minutes, raised, logged a traceback and did no risk pass at all -- for the whole
session, while looking alive.

That shape is the thing worth testing, not the import: a supervisor whose only
failure mode is a log line needs one test that proves the body reaches its work.
"""

from __future__ import annotations

import threading

import pytest

from UI.combined_server import RiskPassSupervisor


@pytest.fixture
def one_pass(monkeypatch):
    """Run exactly one loop iteration, whatever happens inside it.

    Both the success path and the exception path set the stop event, so a
    regression fails the assertion instead of hanging the suite.
    """

    import core.calendar
    import UI.combined_server as server

    calls: list[object] = []
    stop = threading.Event()

    def _is_market_open_now(now=None):
        calls.append(now)
        stop.set()
        return False  # never actually fire a risk pass from a test

    # The supervisor imports the calendar gate inside run(), so patching the
    # module attribute is what it sees.
    monkeypatch.setattr(core.calendar, "is_market_open_now", _is_market_open_now)

    # The loop swallows Exception and continues, so a regression inside the body
    # would spin forever instead of failing. Release the loop on that path too.
    original = server.logger.exception

    def _exception(*args, **kwargs):
        stop.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(server.logger, "exception", _exception)
    return calls, stop


def test_the_loop_body_reaches_the_market_hours_check(one_pass) -> None:
    """The regression. Before the fix this raised NameError before calling it."""

    calls, stop = one_pass

    supervisor = RiskPassSupervisor(interval=60, stop_event=stop, submit=False)
    supervisor.run()

    assert calls, "the loop never reached the market-hours check"


def test_the_time_it_checks_is_timezone_aware(one_pass) -> None:
    """The market-hours check converts to ET, so the time handed to it must be
    aware; a naive one would silently read as local/UTC and put the RTH window
    four or five hours out."""

    calls, stop = one_pass

    supervisor = RiskPassSupervisor(interval=60, stop_event=stop, submit=False)
    supervisor.run()

    now = calls[0]
    assert now.tzinfo is not None and now.utcoffset() is not None


def test_a_stop_before_the_first_tick_does_no_work(one_pass) -> None:
    calls, stop = one_pass
    stop.set()

    RiskPassSupervisor(interval=60, stop_event=stop, submit=False).run()

    assert calls == []
