"""Self-contained US equity-market trading calendar (NYSE/Nasdaq).

No external dependency: the full-day NYSE closures are computed from the standard
holiday rules (fixed-date holidays roll to the nearest weekday; the floating ones
— MLK, Presidents', Memorial, Labor, Thanksgiving — and Good Friday are derived).
This is enough for "is the market open on day X / what's the next session" logic
used by the option-expiry-before-closure guard.

Half-days (1:00pm early closes) are NOT modeled here — those days are still
trading days, which is all the expiry guard needs.

If you'd rather use the maintained library instead, install pandas_market_calendars
and swap next_trading_day for mcal.get_calendar("XNYS").valid_days(...). This module
is intentionally dependency-free so the live runner can't break on a missing import.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_RTH_OPEN = time(9, 30)
_RTH_CLOSE = time(16, 0)


def _nearest_weekday(d: date) -> date:
    """Observed date for a fixed holiday: Sat -> Fri, Sun -> Mon."""
    if d.weekday() == 5:  # Saturday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th `weekday` (Mon=0) of month, e.g. 3rd Monday."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """The last `weekday` of a month (e.g. last Monday = Memorial Day)."""
    if month == 12:
        d = date(year, 12, 31)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _good_friday(year: int) -> date:
    """Good Friday = 2 days before Easter Sunday (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    day = ((h + m - 7 * n + 114) % 31) + 1
    easter = date(year, month, day)
    return easter - timedelta(days=2)


@lru_cache(maxsize=64)
def market_holidays(year: int) -> frozenset[date]:
    """Full-day NYSE/Nasdaq closures for a calendar year."""
    h: set[date] = {
        _nearest_weekday(date(year, 1, 1)),    # New Year's Day
        _nth_weekday(year, 1, 0, 3),           # MLK Jr. Day (3rd Mon Jan)
        _nth_weekday(year, 2, 0, 3),           # Presidents' Day (3rd Mon Feb)
        _good_friday(year),                    # Good Friday
        _last_weekday(year, 5, 0),             # Memorial Day (last Mon May)
        _nearest_weekday(date(year, 6, 19)),   # Juneteenth (since 2021)
        _nearest_weekday(date(year, 7, 4)),    # Independence Day
        _nth_weekday(year, 9, 0, 1),           # Labor Day (1st Mon Sep)
        _nth_weekday(year, 11, 3, 4),          # Thanksgiving (4th Thu Nov)
        _nearest_weekday(date(year, 12, 25)),  # Christmas
    }
    return frozenset(h)


def is_market_holiday(d: date) -> bool:
    return d in market_holidays(d.year)


def is_trading_day(d: date) -> bool:
    """Weekday that isn't a full-day market closure."""
    return d.weekday() < 5 and not is_market_holiday(d)


def next_trading_day(d: date) -> date:
    """First trading session strictly after `d`."""
    nxt = d + timedelta(days=1)
    while not is_trading_day(nxt):
        nxt += timedelta(days=1)
    return nxt


def prev_trading_day(d: date) -> date:
    """Most recent trading session strictly before `d`."""
    prev = d - timedelta(days=1)
    while not is_trading_day(prev):
        prev -= timedelta(days=1)
    return prev


def is_market_open_now(now: datetime | None = None) -> bool:
    """True during US equity/options regular trading hours (09:30-16:00 ET) on a
    real trading day (weekday, minus full-day NYSE holidays).

    Single source of truth for "is RTH open right now" — previously hand-rolled
    independently in a couple of live-trading modules, both of which checked
    only weekday + time-of-day and so reported the market open on a holiday
    weekday. Half-days (1pm early closes) are still not modeled (see module
    docstring); on those days this can report open past the actual 1pm close.
    """
    et = (now or datetime.now(_ET)).astimezone(_ET)
    if not is_trading_day(et.date()):
        return False
    return _RTH_OPEN <= et.time() < _RTH_CLOSE
