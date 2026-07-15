"""Continuous background refresher for the shared-universe data stack.

Keeps ``Data/shared/bars/{1h,4h}`` and the Meta Ranker matrix fresh OUT-OF-BAND
so the 4H trading loops (Meta / HTF / Momentum) only have to read + score. This
replaces the old design where each Meta pass ran bars -> feeds -> matrix inline
(~100 min), which delayed the 2 PM decision to the close and aborted HTF.

Each cycle (during the RTH-ish window) runs, as subprocesses:
  1. catch up shared bars for the eligible set  (incremental; ``--no-1d``)
  2. light per-bar feeds, throttled to ``feeds_interval``  (news/econ/guidance)
  3. rebuild the Meta Ranker matrix  (incremental append)

The heavy daily feeds (earnings-calendar, news-catalyst rescore) are NOT run
here — they live in the nightly readiness job. Steps are isolated subprocesses so
one failure can't take the server down, and the loop sleeps in small
interruptible chunks for immediate shutdown.
"""
from __future__ import annotations

import datetime as dt
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

from core.live_job_guard import heavy_job_guard

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


def is_within_window(
    now: dt.datetime,
    *,
    open_hm: tuple[int, int] = (13, 45),
    close_hm: tuple[int, int] = (16, 40),
    weekdays_only: bool = True,
) -> bool:
    """True when ``now`` (tz-aware) is within [open, close) on a trading weekday.

    Window defaults to 13:45–16:40 ET. Nightly/pre-open readiness prepares the
    cached universe; live refresh starts only near the first 4H close so startup
    and the open stay trading-only.
    """
    if weekdays_only and now.weekday() >= 5:
        return False
    open_t = now.replace(hour=open_hm[0], minute=open_hm[1], second=0, microsecond=0)
    close_t = now.replace(hour=close_hm[0], minute=close_hm[1], second=0, microsecond=0)
    return open_t <= now < close_t


class SharedDataRefresher(threading.Thread):
    """Daemon thread that keeps shared bars + Meta matrix fresh during the session."""

    def __init__(
        self,
        *,
        python: str | None = None,
        interval: int = 300,
        feeds_interval: int = 3600,
        tz: str = DEFAULT_TZ,
        open_hm: tuple[int, int] = (13, 45),
        close_hm: tuple[int, int] = (16, 40),
        weekdays_only: bool = True,
        feeds_enabled: bool = False,
        stop_event: threading.Event | None = None,
        poll_seconds: float = 15.0,
        name: str = "shared-data-refresher",
    ) -> None:
        super().__init__(daemon=True, name=name)
        self._repo = Path(__file__).resolve().parents[1]
        self._python = python or sys.executable
        self._interval = int(interval)
        self._feeds_interval = int(feeds_interval)
        self._tz = _tz(tz)
        self._open = open_hm
        self._close = close_hm
        self._weekdays_only = weekdays_only
        self._feeds_enabled = bool(feeds_enabled)
        self._stop_evt = stop_event or threading.Event()
        self._poll = poll_seconds
        self._last_feeds_at: float = 0.0

    def _now(self) -> dt.datetime:
        return dt.datetime.now(self._tz)

    def _run_step(self, label: str, argv: list[str], timeout: int) -> bool:
        import os

        env = {**os.environ, "PYTHONPATH": str(self._repo)}
        t0 = time.time()
        try:
            rc = subprocess.run(
                argv, cwd=str(self._repo), env=env, timeout=timeout,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode
            ok = rc == 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("[refresher] %s raised %s: %s", label, type(exc).__name__, exc)
            ok = False
        logger.info("[refresher] %s %s (%.0fs)", label, "OK" if ok else "FAILED", time.time() - t0)
        return ok

    def run_cycle(self, *, with_feeds: bool) -> None:
        """One refresh cycle: bars -> (light feeds) -> matrix. Public for testing."""
        with heavy_job_guard(
            "shared-data-refresher",
            block_live_window=True,
            min_available_mb=4096,
            min_swap_free_mb=4096,
        ) as guard:
            if not guard.ok:
                logger.warning("[refresher] skipped: %s", guard.reason)
                return
            self._run_step(
                "bars",
                [self._python, "scripts/catchup_shared_bars.py",
                 "--workers", "6", "--eligible-only", "--no-1d"],
                timeout=1800,
            )
            if with_feeds and self._feeds_enabled:
                self._run_step(
                    "feeds",
                    [self._python, "signals/meta_context/meta_ranker/update_feeds.py"],
                    timeout=1500,
                )
            self._run_step(
                "matrix",
                [self._python, "signals/meta_context/meta_ranker/update_meta_matrix.py"],
                timeout=1500,
            )

    def _sleep(self, seconds: float) -> None:
        """Interruptible sleep in small chunks."""
        end = time.time() + seconds
        while time.time() < end and not self._stop_evt.is_set():
            self._stop_evt.wait(min(self._poll, max(0.0, end - time.time())))

    def run(self) -> None:  # pragma: no cover - thread loop exercised via run_cycle
        logger.info(
            "[refresher] up: refresh every %ds, light feeds every %ds, window %02d:%02d-%02d:%02d %s",
            self._interval, self._feeds_interval, *self._open, *self._close, self._tz,
        )
        while not self._stop_evt.is_set():
            if not is_within_window(
                self._now(), open_hm=self._open, close_hm=self._close,
                weekdays_only=self._weekdays_only,
            ):
                self._sleep(self._poll * 4)
                continue
            with_feeds = self._feeds_enabled and (time.time() - self._last_feeds_at) >= self._feeds_interval
            if with_feeds:
                self._last_feeds_at = time.time()
            try:
                self.run_cycle(with_feeds=with_feeds)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[refresher] cycle error: %s", exc)
            self._sleep(self._interval)
