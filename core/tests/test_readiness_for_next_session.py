"""The 22:15 readiness job must test the session it exists to authorize.

The consumer gate asks "is the stamp good NOW". Used as the 22:15 skip test that
question is wrong: at 22:15 Monday prev_trading_day(Mon) is Friday, so a
same-morning stamp passes and the job skips — then Tuesday's gate advances the
threshold to Monday 16:00 and that stamp is stale, so the session opens dark.
Observed on 07-28, 08-05, 08-07, 08-10, 08-12, and 08-14.
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import core.live_readiness as lr

pytestmark = pytest.mark.safe

ET = ZoneInfo("America/New_York")


def _stamp(tmp_path, when_et: datetime):
    path = tmp_path / "latest_success.json"
    path.write_text(json.dumps({
        "job": "nightly_data_readiness",
        "status": "success",
        "completed_at_utc": when_et.astimezone(ZoneInfo("UTC")).isoformat(),
        "version": 1,
    }))
    return path


def test_same_morning_stamp_does_not_authorize_tomorrow(tmp_path):
    """The exact 08-14 shape: Thursday-night stamp, Friday 22:15 skip test."""
    stamp = _stamp(tmp_path, datetime(2026, 8, 13, 22, 54, tzinfo=ET))
    now = datetime(2026, 8, 14, 22, 15, tzinfo=ET)

    ok_now, _, _ = lr.readiness_status(path=stamp, now=now)
    ok_next, reason, _ = lr.readiness_status(path=stamp, now=now, for_next_session=True)

    assert ok_now, "precondition: the plain gate is what wrongly satisfied the skip"
    assert not ok_next, "for_next_session must refuse a stamp that goes stale tomorrow"
    assert "predates" in reason


def test_monday_night_requires_a_post_monday_close_stamp(tmp_path):
    """22:15 Monday: authorizing Tuesday needs proof generated after Mon 16:00."""
    now = datetime(2026, 8, 17, 22, 15, tzinfo=ET)

    morning = _stamp(tmp_path, datetime(2026, 8, 17, 8, 14, tzinfo=ET))
    ok_morning, _, _ = lr.readiness_status(path=morning, now=now, for_next_session=True)
    assert not ok_morning, "a pre-close Monday stamp cannot authorize Tuesday"

    evening = _stamp(tmp_path, datetime(2026, 8, 17, 21, 30, tzinfo=ET))
    ok_evening, _, _ = lr.readiness_status(path=evening, now=now, for_next_session=True)
    assert ok_evening, "a post-close Monday stamp is exactly what Tuesday needs"


def test_friday_night_looks_ahead_to_monday_not_saturday(tmp_path):
    """next_trading_day must skip the weekend rather than land on Saturday."""
    now = datetime(2026, 8, 14, 22, 15, tzinfo=ET)
    stamp = _stamp(tmp_path, datetime(2026, 8, 14, 21, 0, tzinfo=ET))

    ok, _, _ = lr.readiness_status(path=stamp, now=now, for_next_session=True)

    # Monday's gate requires a stamp after Friday 16:00; this one qualifies.
    assert ok


def test_default_behaviour_is_unchanged_for_consumers(tmp_path):
    """Live order paths must keep asking the now-question — no silent widening."""
    stamp = _stamp(tmp_path, datetime(2026, 8, 13, 22, 54, tzinfo=ET))
    now = datetime(2026, 8, 14, 9, 30, tzinfo=ET)

    ok_default, _, _ = lr.readiness_status(path=stamp, now=now)
    ok_explicit, _, _ = lr.readiness_status(path=stamp, now=now, for_next_session=False)

    assert ok_default is ok_explicit is True


def test_a_stale_stamp_is_still_refused_by_both_modes(tmp_path):
    """for_next_session loosens nothing — it moves the threshold forward."""
    stamp = _stamp(tmp_path, datetime(2026, 8, 3, 22, 0, tzinfo=ET))
    now = datetime(2026, 8, 17, 22, 15, tzinfo=ET)

    assert not lr.readiness_status(path=stamp, now=now)[0]
    assert not lr.readiness_status(path=stamp, now=now, for_next_session=True)[0]
