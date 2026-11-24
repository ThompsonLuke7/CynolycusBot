"""In-process daily scheduler for the combined server.

Lets long-running processes (the combined dashboard server) fire the nightly
jobs themselves at a fixed wall-clock time — handy when the machine is reliably
on after the close but no system cron is configured.

The schedule logic (``next_run_at``) is pure and unit-tested; the thread just
sleeps until the next occurrence (in small, interruptible chunks so shutdown is
immediate) and invokes a callable.
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
from typing import Callable

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_TZ = "America/New_York"


def _tz(name: str):
    if ZoneInfo is None:
        return dt.timezone.utc
    try:
        return ZoneInfo(name)
    except Exception:
        return dt.timezone.utc


def next_run_at(
    now: dt.datetime,
    hour: int,
    minute: int,
    *,
    weekdays_only: bool = True,
    close_relative: bool = False,
    trading_days_only: bool = False,
) -> dt.datetime:
    """Next datetime >= now landing on ``hour:minute``.

    When ``weekdays_only`` is set, rolls forward past Saturday/Sunday so the job
    only fires on (US) trading weekdays. ``now`` must be timezone-aware; the
    result carries the same tzinfo.

    ``close_relative`` is for near-close jobs (``hour:minute`` is ET): on a
    13:00 half-day the slot moves to the same distance before that close
    (15:15 -> 12:15), so the job still runs while the market is open. A slot
    anchored to the close needs a session, so with ``weekdays_only`` it also
    skips full-day market holidays.

    ``trading_days_only`` skips full-day market holidays as well as weekends,
    for jobs that trade or score a new 4H bar. On a holiday such a job would
    re-score the previous session's bar (ageing every position one phantom bar)
    and send its orders to a closed market.
    """
    if close_relative or trading_days_only:
        from core.calendar import close_relative_time, is_trading_day

        skip_holidays = trading_days_only or weekdays_only
        day = now.date()
        for _ in range(14):
            if not skip_holidays or is_trading_day(day):
                slot = dt.time(hour, minute)
                if close_relative:
                    slot = close_relative_time(slot, day)
                target = dt.datetime.combine(day, slot, tzinfo=now.tzinfo)
                if target > now:
                    return target
            day += dt.timedelta(days=1)
        raise RuntimeError("next_run_at: no slot found within 14 days")
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += dt.timedelta(days=1)
    if weekdays_only:
        while target.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            target += dt.timedelta(days=1)
    return target


def parse_hhmm(value: str) -> tuple[int, int]:
    """Parse a ``"HH:MM"`` string into (hour, minute)."""
    parts = str(value).strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected HH:MM, got {value!r}")
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Out-of-range time {value!r}")
    return hour, minute


class NightlyScheduler(threading.Thread):
    """Daemon thread that runs ``job`` once per day at a fixed local time."""

    def __init__(
        self,
        job: Callable[[], None],
        *,
        hour: int,
        minute: int,
        tz: str = DEFAULT_TZ,
        weekdays_only: bool = True,
        close_relative: bool = False,
        trading_days_only: bool = False,
        stop_event: threading.Event | None = None,
        name: str = "nightly-scheduler",
        poll_seconds: float = 30.0,
    ) -> None:
        super().__init__(daemon=True, name=name)
        self._job = job
        self._hour = hour
        self._minute = minute
        self._tz = _tz(tz)
        self._tz_name = tz
        self._weekdays_only = weekdays_only
        self._close_relative = close_relative
        self._trading_days_only = trading_days_only
        self._stop_evt = stop_event or threading.Event()
        self._poll = poll_seconds
        self._last_run_date: dt.date | None = None

    def _now(self) -> dt.datetime:
        return dt.datetime.now(self._tz)

    def run(self) -> None:  # noqa: D401 - thread entry
        nxt = next_run_at(
            self._now(), self._hour, self._minute,
            weekdays_only=self._weekdays_only, close_relative=self._close_relative,
            trading_days_only=self._trading_days_only,
        )
        logger.info(
            "Nightly scheduler armed: next run %s (%s), weekdays_only=%s",
            nxt.isoformat(timespec="minutes"),
            self._tz_name,
            self._weekdays_only,
        )
        while not self._stop_evt.is_set():
            now = self._now()
            if now >= nxt:
                # Guard against double-firing within the same calendar day.
                if self._last_run_date != now.date():
                    self._last_run_date = now.date()
                    self._fire()
                nxt = next_run_at(
                    self._now(), self._hour, self._minute,
                    weekdays_only=self._weekdays_only, close_relative=self._close_relative,
                    trading_days_only=self._trading_days_only,
                )
                logger.info("Nightly scheduler: next run %s (%s)", nxt.isoformat(timespec="minutes"), self._tz_name)
                continue
            remaining = (nxt - now).total_seconds()
            self._stop_evt.wait(timeout=min(self._poll, max(1.0, remaining)))

    def _fire(self) -> None:
        logger.info("Nightly scheduler: firing job at %s", self._now().isoformat(timespec="seconds"))
        try:
            self._job()
        except Exception as exc:  # never let a job error kill the thread
            logger.error("Nightly job failed: %s", exc, exc_info=True)
        else:
            logger.info("Nightly scheduler: job complete")
