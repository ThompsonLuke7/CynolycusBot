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
) -> dt.datetime:
    """Next datetime >= now landing on ``hour:minute``.

    When ``weekdays_only`` is set, rolls forward past Saturday/Sunday so the job
    only fires on (US) trading weekdays. ``now`` must be timezone-aware; the
    result carries the same tzinfo.
    """
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
        self._stop_evt = stop_event or threading.Event()
        self._poll = poll_seconds
        self._last_run_date: dt.date | None = None

    def _now(self) -> dt.datetime:
        return dt.datetime.now(self._tz)

    def run(self) -> None:  # noqa: D401 - thread entry
        nxt = next_run_at(self._now(), self._hour, self._minute, weekdays_only=self._weekdays_only)
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
                nxt = next_run_at(self._now(), self._hour, self._minute, weekdays_only=self._weekdays_only)
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
