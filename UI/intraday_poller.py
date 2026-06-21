"""Market-hours supervisor for the intraday catalyst poller.

Launches ``scripts/intraday_catalyst_poll.py`` as a subprocess during regular
trading hours and terminates it after the close, keeping the live catalyst
ledger (``live_catalyst_records.parquet``, consumed by the swing scanner) fresh
only while the market is open.

The window logic (``is_market_hours``) is pure and unit-tested; the thread polls
in small interruptible chunks so shutdown is immediate, and relaunches the child
if it dies mid-session.
"""
from __future__ import annotations

import datetime as dt
import logging
import subprocess
import sys
import threading
from pathlib import Path

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


def is_market_hours(
    now: dt.datetime,
    *,
    open_hm: tuple[int, int] = (9, 30),
    close_hm: tuple[int, int] = (16, 0),
    weekdays_only: bool = True,
) -> bool:
    """True when ``now`` is within [open, close) on a trading weekday.

    ``now`` must be timezone-aware (compared in its own tz).
    """
    if weekdays_only and now.weekday() >= 5:  # 5=Sat, 6=Sun
        return False
    open_t = now.replace(hour=open_hm[0], minute=open_hm[1], second=0, microsecond=0)
    close_t = now.replace(hour=close_hm[0], minute=close_hm[1], second=0, microsecond=0)
    return open_t <= now < close_t


class IntradayPollerSupervisor(threading.Thread):
    """Daemon thread that runs the catalyst poller only during market hours."""

    def __init__(
        self,
        *,
        script: Path | str | None = None,
        python: str | None = None,
        interval: int = 300,
        tz: str = DEFAULT_TZ,
        open_hm: tuple[int, int] = (9, 30),
        close_hm: tuple[int, int] = (16, 0),
        weekdays_only: bool = True,
        stop_event: threading.Event | None = None,
        poll_seconds: float = 30.0,
        name: str = "intraday-poller",
    ) -> None:
        super().__init__(daemon=True, name=name)
        repo_root = Path(__file__).resolve().parents[1]
        self._script = Path(script) if script else repo_root / "scripts" / "intraday_catalyst_poll.py"
        self._python = python or sys.executable
        self._interval = int(interval)
        self._cwd = repo_root
        self._tz = _tz(tz)
        self._tz_name = tz
        self._open = open_hm
        self._close = close_hm
        self._weekdays_only = weekdays_only
        self._stop_evt = stop_event or threading.Event()
        self._poll = poll_seconds
        self._proc: subprocess.Popen | None = None

    def _now(self) -> dt.datetime:
        return dt.datetime.now(self._tz)

    def _running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _start_proc(self) -> None:
        if self._running():
            return
        cmd = [self._python, "-u", str(self._script), "--interval", str(self._interval)]
        logger.info("Intraday catalyst poller: starting (%s)", " ".join(cmd))
        try:
            self._proc = subprocess.Popen(cmd, cwd=str(self._cwd))
        except Exception as exc:  # noqa: BLE001 - never let a launch failure kill the thread
            logger.error("Intraday catalyst poller: failed to start: %s", exc)
            self._proc = None

    def _stop_proc(self) -> None:
        if self._running():
            assert self._proc is not None
            logger.info("Intraday catalyst poller: stopping (pid=%s)", self._proc.pid)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.warning("Intraday catalyst poller: kill after terminate timeout")
                self._proc.kill()
        self._proc = None

    def run(self) -> None:  # noqa: D401 - thread entry
        logger.info(
            "Intraday poller supervisor armed: %02d:%02d–%02d:%02d %s, weekdays_only=%s",
            self._open[0], self._open[1], self._close[0], self._close[1],
            self._tz_name, self._weekdays_only,
        )
        try:
            while not self._stop_evt.is_set():
                if is_market_hours(
                    self._now(),
                    open_hm=self._open,
                    close_hm=self._close,
                    weekdays_only=self._weekdays_only,
                ):
                    self._start_proc()  # relaunches if the child died mid-session
                else:
                    self._stop_proc()
                self._stop_evt.wait(timeout=self._poll)
        finally:
            self._stop_proc()
