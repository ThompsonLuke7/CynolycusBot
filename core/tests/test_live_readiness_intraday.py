"""Intraday bar-completeness gate for same-day 4H decisions (2026-11-22)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from core import live_readiness as lr

ET = ZoneInfo("America/New_York")


def _et(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=ET)


def test_required_end_follows_the_newest_completed_4h_bar():
    # Friday 2026-11-20 (trading day); Monday 2026-11-23.
    assert lr.required_intraday_bars_end(_et(2026, 11, 20, 14, 25)) == _et(2026, 11, 20, 14).astimezone(timezone.utc)
    assert lr.required_intraday_bars_end(_et(2026, 11, 20, 16, 25)) == _et(2026, 11, 20, 16).astimezone(timezone.utc)
    # Before 14:00 and on a weekend: the previous session's close.
    assert lr.required_intraday_bars_end(_et(2026, 11, 20, 9, 35)) == _et(2026, 11, 19, 16).astimezone(timezone.utc)
    assert lr.required_intraday_bars_end(_et(2026, 11, 22, 12, 0)) == _et(2026, 11, 20, 16).astimezone(timezone.utc)
    assert lr.required_intraday_bars_end(_et(2026, 11, 23, 9, 35)) == _et(2026, 11, 20, 16).astimezone(timezone.utc)


def test_four_hour_bar_end_and_required_start():
    assert lr.four_hour_bar_end(_et(2026, 11, 20, 10)) == _et(2026, 11, 20, 14).astimezone(timezone.utc)
    assert lr.four_hour_bar_end(_et(2026, 11, 20, 14)) == _et(2026, 11, 20, 16).astimezone(timezone.utc)
    # Summer (EDT) stamps resolve the same way.
    assert lr.four_hour_bar_end(_et(2026, 7, 1, 10)) == _et(2026, 7, 1, 14).astimezone(timezone.utc)
    assert lr.required_4h_bar_start(_et(2026, 11, 20, 14, 25)) == _et(2026, 11, 20, 10).astimezone(timezone.utc)
    assert lr.required_4h_bar_start(_et(2026, 11, 20, 16, 25)) == _et(2026, 11, 20, 14).astimezone(timezone.utc)
    assert lr.required_4h_bar_start(_et(2026, 11, 23, 9, 35)) == _et(2026, 11, 20, 14).astimezone(timezone.utc)


def test_status_reads_the_catch_up_stamp(tmp_path):
    stamp = tmp_path / "bars.json"
    now = _et(2026, 11, 20, 14, 25)
    ok, why = lr.intraday_bars_status(now=now, path=stamp)
    assert not ok and "no usable" in why
    # The 14:03 cycle (end 13:43): the 13:00 hour was still forming.
    lr.write_intraday_bars_stamp(end=_et(2026, 11, 20, 13, 43), tickers=2903, errors=0, scope="eligible", path=stamp)
    ok, why = lr.intraday_bars_status(now=now, path=stamp)
    assert not ok and "13:43" in why and "14:00" in why
    # The 14:20:30 cycle (end 14:00:30) covers it.
    lr.write_intraday_bars_stamp(end=_et(2026, 11, 20, 14, 0) + timedelta(seconds=30), tickers=2903,
                                 errors=0, scope="eligible", path=stamp)
    assert lr.intraday_bars_status(now=now, path=stamp)[0]
    assert lr.intraday_bars_fetched_through(stamp) == (_et(2026, 11, 20, 14, 0) + timedelta(seconds=30))


def test_wait_polls_until_the_stamp_lands_or_times_out(tmp_path):
    stamp = tmp_path / "bars.json"
    clock = {"t": _et(2026, 11, 20, 14, 25)}
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        clock["t"] += timedelta(seconds=s)
        if len(sleeps) == 3:   # the catch-up finishes during the third poll
            lr.write_intraday_bars_stamp(end=_et(2026, 11, 20, 14, 1), tickers=1, errors=0,
                                         scope="eligible", path=stamp)

    ok, _why, waited = lr.wait_for_intraday_bars(timeout_s=900, poll_s=20, path=stamp,
                                                 now_fn=lambda: clock["t"], sleep_fn=sleep)
    assert ok and waited == 60 and sleeps == [20, 20, 20]

    stamp.unlink()
    clock["t"] = _et(2026, 11, 20, 14, 25)
    ok, _why, waited = lr.wait_for_intraday_bars(timeout_s=50, poll_s=20, path=stamp,
                                                 now_fn=lambda: clock["t"],
                                                 sleep_fn=lambda s: clock.__setitem__("t", clock["t"] + timedelta(seconds=s)))
    assert not ok and waited == 50
