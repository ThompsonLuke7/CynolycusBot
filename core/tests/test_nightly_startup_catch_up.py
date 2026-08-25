"""The nightly slot missed while the server was down must be made up.

``NightlyScheduler`` computes its next fire from "now", so a 16:45 ET slot that
passed while the process was dead is skipped, never repeated. On 2026-08-19 the
server died at 14:10 ET; a restart that evening would have armed the scheduler
for 08-20 and opened that session on a full day of stale collection.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from UI.combined_server import _nightly_needs_catch_up

ET = ZoneInfo("America/New_York")


@pytest.fixture()
def stamp(tmp_path, monkeypatch):
    """Point the check at a temporary stamp and return a writer for it."""
    import UI.combined_server as cs

    path = tmp_path / "nightly_market_data_latest.json"
    monkeypatch.setattr(cs, "NIGHTLY_STAMP_PATH", path)

    def write(completed_at_et, status="success"):
        path.write_text(
            '{"completed_at_utc": "%s", "status": "%s", "version": 1}'
            % (completed_at_et.astimezone(ZoneInfo("UTC")).isoformat(), status)
        )

    return write


def test_evening_restart_after_a_missed_slot_runs_the_catch_up(stamp):
    # 2026-08-19: crash at 14:10, restart at 22:50, 16:45 slot never fired.
    stamp(datetime(2026, 8, 18, 21, 36, tzinfo=ET))
    needed, reason = _nightly_needs_catch_up(
        now_et=datetime(2026, 8, 19, 22, 50, tzinfo=ET),
        nightly_time="16:45",
        within_hours=14.0,
    )
    assert needed is True
    assert "16:45" in reason


def test_a_run_after_the_last_slot_is_not_repeated(stamp):
    stamp(datetime(2026, 8, 19, 17, 10, tzinfo=ET))
    needed, reason = _nightly_needs_catch_up(
        now_et=datetime(2026, 8, 19, 22, 50, tzinfo=ET),
        nightly_time="16:45",
        within_hours=14.0,
    )
    assert needed is False
    assert "already ran" in reason


def test_a_mid_session_restart_does_not_launch_a_heavy_collection(stamp):
    # 10:30 ET Thursday: the next open is ~23h out, well outside the window.
    stamp(datetime(2026, 8, 18, 21, 36, tzinfo=ET))
    needed, reason = _nightly_needs_catch_up(
        now_et=datetime(2026, 8, 20, 10, 30, tzinfo=ET),
        nightly_time="16:45",
        within_hours=14.0,
    )
    assert needed is False
    assert "away" in reason


def test_a_missing_stamp_counts_as_never_run(stamp, tmp_path, monkeypatch):
    import UI.combined_server as cs

    monkeypatch.setattr(cs, "NIGHTLY_STAMP_PATH", tmp_path / "absent.json")
    needed, reason = _nightly_needs_catch_up(
        now_et=datetime(2026, 8, 19, 22, 50, tzinfo=ET),
        nightly_time="16:45",
        within_hours=14.0,
    )
    assert needed is True
    assert "no completion stamp" in reason


def test_a_failed_run_does_not_satisfy_the_check(stamp):
    stamp(datetime(2026, 8, 19, 17, 10, tzinfo=ET), status="failed")
    needed, reason = _nightly_needs_catch_up(
        now_et=datetime(2026, 8, 19, 22, 50, tzinfo=ET),
        nightly_time="16:45",
        within_hours=14.0,
    )
    assert needed is True
    assert "failed" in reason


def test_an_early_morning_restart_with_runway_still_catches_up(stamp):
    """02:00 leaves 7.5h before the open — the job fits, so it runs.

    This test used to assert the same thing about 06:00. That was the old
    contract and it was wrong: 06:00 leaves 3.5h for a job that takes about
    six, so "catching up" meant running until mid-session. See the runway test
    below for what 06:00 does now.
    """

    stamp(datetime(2026, 8, 18, 21, 36, tzinfo=ET))
    needed, reason = _nightly_needs_catch_up(
        now_et=datetime(2026, 8, 20, 2, 0, tzinfo=ET),
        nightly_time="16:45",
        within_hours=14.0,
    )
    assert needed is True
    assert "fits" in reason


def test_a_restart_too_close_to_the_open_defers_to_the_scheduled_slot(stamp):
    """2026-08-24, the case this bound exists for.

    The server started at 09:00 with the open half an hour away. The old check
    saw only an upper bound — "is the open close enough that the gap will be
    traded on?" — read 0.5h as urgent, and launched a five-hour collection that
    ran until 14:24: through the entire session, holding the heavy-job lock
    readiness needed and pushing momentum's feature panel from 425s to 2,237s.

    Skipping is not giving up. The 16:45 slot still runs it after the close,
    which is the same call the scheduled slot already makes.
    """

    stamp(datetime(2026, 8, 21, 21, 42, tzinfo=ET))
    needed, reason = _nightly_needs_catch_up(
        now_et=datetime(2026, 8, 24, 9, 0, tzinfo=ET),
        nightly_time="16:45",
        within_hours=14.0,
    )
    assert needed is False
    assert "inside the session" in reason
    assert "16:45" in reason, "the log must say where the work actually goes"


def test_the_runway_bound_is_the_job_duration_not_a_fixed_hour(stamp):
    """Same instant, different estimated duration, opposite answers — so the
    bound tracks how long the job really takes rather than a magic constant.
    """

    stamp(datetime(2026, 8, 21, 21, 42, tzinfo=ET))
    at_0600 = dict(
        now_et=datetime(2026, 8, 24, 6, 0, tzinfo=ET),
        nightly_time="16:45",
        within_hours=14.0,
    )

    assert _nightly_needs_catch_up(**at_0600, min_runway_hours=6.0)[0] is False
    assert _nightly_needs_catch_up(**at_0600, min_runway_hours=2.0)[0] is True


def test_the_default_runway_matches_the_measured_nightly_duration():
    """2026-08-24 ran the nightly twice: 5h24m and 4h57m. A default below that
    would re-open the hole this bound closes.
    """

    from UI.combined_server import NIGHTLY_CATCH_UP_MIN_RUNWAY_H

    assert NIGHTLY_CATCH_UP_MIN_RUNWAY_H >= 5.5
