"""A market snapshot is keyed on the session it describes, not the wall clock.

The dealer and IV-surface captures used ``now.date()``. A catch-up run after
midnight or over a weekend still reads the last session's close, so that filed
Monday's close under Tuesday, where Tuesday evening's capture then overwrote it:
the 2026-10-06 session was lost this way (captured 10-07 01:56, stored as 10-07).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from core.calendar import latest_session_date

pytestmark = pytest.mark.safe

ET = ZoneInfo("America/New_York")


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        # Regular session and the post-close nightly slot: today.
        (datetime(2026, 10, 12, 9, 30, tzinfo=ET), date(2026, 10, 12)),
        (datetime(2026, 10, 12, 15, 15, tzinfo=ET), date(2026, 10, 12)),
        (datetime(2026, 10, 12, 16, 49, tzinfo=ET), date(2026, 10, 12)),
        (datetime(2026, 10, 12, 23, 59, tzinfo=ET), date(2026, 10, 12)),
        # After midnight and pre-open: still the previous session.
        (datetime(2026, 10, 13, 1, 49, tzinfo=ET), date(2026, 10, 12)),
        (datetime(2026, 10, 13, 9, 29, tzinfo=ET), date(2026, 10, 12)),
        # Weekend, and Monday before the open: Friday.
        (datetime(2026, 10, 10, 12, 0, tzinfo=ET), date(2026, 10, 9)),
        (datetime(2026, 10, 11, 23, 0, tzinfo=ET), date(2026, 10, 9)),
        (datetime(2026, 10, 12, 1, 0, tzinfo=ET), date(2026, 10, 9)),
        # Full-day closure (Labor Day 2026-09-07) and the morning after it.
        (datetime(2026, 9, 7, 17, 0, tzinfo=ET), date(2026, 9, 4)),
        (datetime(2026, 9, 8, 8, 0, tzinfo=ET), date(2026, 9, 4)),
    ],
)
def test_latest_session_date(now, expected):
    assert latest_session_date(now) == expected


def test_reads_the_clock_in_eastern_time():
    # 01:00 UTC Tuesday is 21:00 ET Monday: Monday's session, not Tuesday's.
    assert latest_session_date(datetime(2026, 10, 13, 1, 0, tzinfo=timezone.utc)) == date(2026, 10, 12)
