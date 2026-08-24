"""A readiness catch-up retries while it is only losing a race.

2026-08-21. The 08-20 22:15 ET readiness run was killed by the 23:45 server
restart. The restart's own catch-up asked for the heavy-job lock, the 23:45
nightly rerun already held it, and the catch-up logged

    Data readiness: skipped (combined-server-data-readiness: another heavy data
    job is already running)

and gave up. Nothing re-checked. Meta's 14:20 ET run the next day then hit

    readiness gate: skipped 5 entry orders (readiness stamp predates latest
    completed trading session (2026-08-20 16:00 ET))

so five live entries were lost to a lock that had cleared hours earlier.
"""
from __future__ import annotations

import UI.combined_server as cs


class _FakeClock:
    """A clock that only moves when something sleeps.

    The retry deadline is measured against the clock, so a test that stubs
    sleeping without advancing time spins forever. Pairing the two here is the
    point rather than a convenience.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


def test_retries_until_the_lock_clears():
    clock = _FakeClock()
    attempts = []

    def runner(*, catch_up):
        attempts.append(catch_up)
        return len(attempts) >= 3      # blocked twice, then the lock frees

    ran = cs._run_data_readiness_with_retry(
        catch_up=True, interval_s=600.0, deadline_s=6 * 3600.0,
        sleep_fn=clock.sleep, monotonic_fn=clock.monotonic, runner=runner,
    )

    assert ran is True
    assert attempts == [True, True, True]
    assert clock.slept == [600.0, 600.0]


def test_first_attempt_succeeding_does_not_sleep():
    clock = _FakeClock()
    ran = cs._run_data_readiness_with_retry(
        catch_up=True, sleep_fn=clock.sleep, monotonic_fn=clock.monotonic,
        runner=lambda *, catch_up: True,
    )
    assert ran is True
    assert clock.slept == []


def test_gives_up_at_the_deadline_rather_than_spinning(caplog):
    """A permanently-held lock must not leave a thread retrying forever."""
    clock = _FakeClock()
    attempts = []

    def runner(*, catch_up):
        attempts.append(catch_up)
        return False

    with caplog.at_level("ERROR"):
        ran = cs._run_data_readiness_with_retry(
            catch_up=True, interval_s=600.0, deadline_s=1800.0,
            sleep_fn=clock.sleep, monotonic_fn=clock.monotonic, runner=runner,
        )

    assert ran is False
    # Attempts at t=0, 600, 1200 and 1800: the loop sleeps while the next wake
    # would still land inside the deadline, so it uses the whole window and
    # makes a final attempt exactly at it.
    assert len(attempts) == 4
    assert clock.now == 1800.0
    assert "never ran" in caplog.text
