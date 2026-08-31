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
    """10:30 ET Thursday: the 16:45 slot fires today, long before Friday opens,
    so the staleness gets fixed before anything trades on it.

    The assertion moved off the old hours-to-open wording deliberately — the
    bound is now "will a slot beat the open?" rather than a raw hour count.
    """

    stamp(datetime(2026, 8, 18, 21, 36, tzinfo=ET))
    needed, reason = _nightly_needs_catch_up(
        now_et=datetime(2026, 8, 20, 10, 30, tzinfo=ET),
        nightly_time="16:45",
        within_hours=72.0,
    )
    assert needed is False
    assert "before the" in reason and "open" in reason


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

    stamp(datetime(2026, 8, 20, 21, 42, tzinfo=ET))
    needed, reason = _nightly_needs_catch_up(
        now_et=datetime(2026, 8, 24, 9, 0, tzinfo=ET),
        nightly_time="16:45",
        within_hours=72.0,
    )
    assert needed is False
    assert "inside the session" in reason
    assert "16:45" in reason, "the log must say where the work actually goes"


def test_the_runway_bound_is_the_job_duration_not_a_fixed_hour(stamp):
    """Same instant, different estimated duration, opposite answers — so the
    bound tracks how long the job really takes rather than a magic constant.
    """

    stamp(datetime(2026, 8, 20, 21, 42, tzinfo=ET))
    at_0600 = dict(
        now_et=datetime(2026, 8, 24, 6, 0, tzinfo=ET),
        nightly_time="16:45",
        within_hours=72.0,
    )

    assert _nightly_needs_catch_up(**at_0600, min_runway_hours=6.0)[0] is False
    assert _nightly_needs_catch_up(**at_0600, min_runway_hours=2.0)[0] is True


def test_the_default_runway_matches_the_measured_nightly_duration():
    """2026-08-24 ran the nightly twice: 5h24m and 4h57m. A default below that
    would re-open the hole this bound closes.
    """

    from UI.combined_server import NIGHTLY_CATCH_UP_MIN_RUNWAY_H

    assert NIGHTLY_CATCH_UP_MIN_RUNWAY_H >= 5.5


# --- the weekend, where the scheduler has no slot at all -------------------

def test_a_sunday_evening_restart_runs_the_weekend_refresh(stamp):
    """The scheduler is weekdays_only, so from Friday's close until Monday
    16:45 there is no slot to defer to.

    This is what the old 14h hours-to-open bound got wrong: a Sunday 19:00
    start measured 14.5h to Monday's open, failed the bound, and handed the
    work to a slot that would not fire until 16:45 Monday — seven hours after
    the session it was supposed to prepare had already opened.
    """

    stamp(datetime(2026, 8, 21, 17, 0, tzinfo=ET), status="failed")
    needed, reason = _nightly_needs_catch_up(
        now_et=datetime(2026, 8, 23, 19, 0, tzinfo=ET),
        nightly_time="16:45",
        within_hours=72.0,
    )
    assert needed is True
    assert "fits" in reason


def test_a_saturday_restart_can_still_run_the_missed_weekend_refresh(stamp):
    """45h to the open used to be refused outright by the hours bound, which
    made the entire weekend window unreachable — precisely when a missed
    refresh has the most time to be repaired safely.
    """

    stamp(datetime(2026, 8, 21, 17, 0, tzinfo=ET), status="failed")
    needed, _ = _nightly_needs_catch_up(
        now_et=datetime(2026, 8, 22, 12, 0, tzinfo=ET),
        nightly_time="16:45",
        within_hours=72.0,
    )
    assert needed is True


def test_a_weekend_restart_after_a_good_friday_run_stays_put(stamp):
    """The weekend opening up must not turn into running every boot. Friday's
    nightly completed after Friday's slot, so nothing is missed.
    """

    stamp(datetime(2026, 8, 21, 21, 42, tzinfo=ET))
    needed, reason = _nightly_needs_catch_up(
        now_et=datetime(2026, 8, 23, 19, 0, tzinfo=ET),
        nightly_time="16:45",
        within_hours=72.0,
    )
    assert needed is False
    assert "already ran" in reason


def test_a_weekday_evening_restart_after_a_missed_slot_catches_up(stamp):
    """Monday 17:00, just after the 16:45 slot was missed. The next slot is
    Tuesday 16:45 — after Tuesday's open — so nothing else will fix this.
    The old 14h bound refused it at 16.5h to the open.
    """

    stamp(datetime(2026, 8, 21, 21, 42, tzinfo=ET))
    needed, _ = _nightly_needs_catch_up(
        now_et=datetime(2026, 8, 24, 17, 0, tzinfo=ET),
        nightly_time="16:45",
        within_hours=72.0,
    )
    assert needed is True


def test_the_hours_cap_is_a_backstop_not_the_working_rule(stamp):
    """It still fires when handed an absurd value, so the guard is real — but
    the default must be loose enough to clear a Friday-evening-to-Monday-open
    weekend (64.5h), or it silently re-breaks the weekend refresh.
    """

    stamp(datetime(2026, 8, 21, 17, 0, tzinfo=ET), status="failed")
    at_sunday = dict(
        now_et=datetime(2026, 8, 23, 19, 0, tzinfo=ET),
        nightly_time="16:45",
    )

    needed, reason = _nightly_needs_catch_up(**at_sunday, within_hours=2.0)
    assert needed is False and "cap" in reason

    import inspect

    from UI.combined_server import run_combined

    default = inspect.signature(run_combined).parameters["nightly_catch_up_hours"].default
    assert default >= 64.5


def test_a_monday_morning_boot_after_a_good_friday_nightly_stays_put(stamp):
    """2026-08-24 root cause, and the one that actually fired.

    The server booted Monday 09:00 with Friday's 21:42 nightly on the stamp —
    the normal, current state, since Monday's open is meant to trade on Friday
    night's collection. ``last_slot`` compared it against a SUNDAY 16:45 slot
    that the weekdays_only scheduler never has, decided Friday's run predated
    it, and launched a five-hour collection into the session.

    Fixing the phantom slot alone would have prevented that morning; the runway
    bound is the second line of defence, not the only one.
    """

    stamp(datetime(2026, 8, 21, 21, 42, tzinfo=ET))
    needed, reason = _nightly_needs_catch_up(
        now_et=datetime(2026, 8, 24, 9, 0, tzinfo=ET),
        nightly_time="16:45",
        within_hours=72.0,
    )
    assert needed is False
    assert "already ran" in reason


def test_the_last_slot_never_lands_on_a_weekend(stamp):
    """Directly: whatever the boot time, the slot we measure staleness against
    has to be one the weekdays_only scheduler would really have fired.
    """

    # A Friday-evening run is current for every boot across the weekend.
    stamp(datetime(2026, 8, 21, 17, 30, tzinfo=ET))
    for now in (
        datetime(2026, 8, 22, 9, 0, tzinfo=ET),   # Saturday
        datetime(2026, 8, 23, 19, 0, tzinfo=ET),  # Sunday evening
        datetime(2026, 8, 24, 8, 0, tzinfo=ET),   # Monday pre-open
    ):
        needed, reason = _nightly_needs_catch_up(
            now_et=now, nightly_time="16:45", within_hours=72.0,
        )
        assert needed is False, f"{now:%a %H:%M} re-ran a current nightly"
        assert "already ran" in reason
