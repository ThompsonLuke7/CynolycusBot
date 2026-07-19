"""Guards for heavyweight live-adjacent data jobs.

The live server can stream and trade while light work happens, but it should not
let multiple full-universe refresh/rebuild jobs stack up during the trading day.
This module provides a small process-wide lock plus memory/window checks that
callers can use before launching bars/feed/matrix/readiness subprocesses.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK_PATH = REPO_ROOT / "Data/runtime/live_data_jobs.lock"
DEFAULT_TZ = "America/New_York"


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    reason: str = ""


def _tz(name: str):
    if ZoneInfo is None:
        return dt.timezone.utc
    try:
        return ZoneInfo(name)
    except Exception:
        return dt.timezone.utc


def _meminfo_mb() -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            parts = raw.strip().split()
            if parts:
                out[key] = float(parts[0]) / 1024.0
    except Exception:
        return {}
    return out


def memory_budget_ok(*, min_available_mb: float = 4096, min_swap_free_mb: float = 4096) -> GuardResult:
    info = _meminfo_mb()
    if not info:
        return GuardResult(True, "meminfo_unavailable")
    avail = float(info.get("MemAvailable", 0.0))
    swap_free = float(info.get("SwapFree", 0.0))
    if avail < min_available_mb:
        return GuardResult(False, f"low MemAvailable {avail:.0f} MB < {min_available_mb:.0f} MB")
    if swap_free < min_swap_free_mb:
        return GuardResult(False, f"low SwapFree {swap_free:.0f} MB < {min_swap_free_mb:.0f} MB")
    return GuardResult(True, f"MemAvailable={avail:.0f} MB SwapFree={swap_free:.0f} MB")


def is_blocked_live_window(
    now: dt.datetime | None = None,
    *,
    tz: str = DEFAULT_TZ,
    start_hm: tuple[int, int] = (7, 45),
    end_hm: tuple[int, int] = (10, 5),
) -> bool:
    """Block heavy jobs during the fragile morning/open window on weekdays."""
    local = now.astimezone(_tz(tz)) if now is not None else dt.datetime.now(_tz(tz))
    if local.weekday() >= 5:
        return False
    start = local.replace(hour=start_hm[0], minute=start_hm[1], second=0, microsecond=0)
    end = local.replace(hour=end_hm[0], minute=end_hm[1], second=0, microsecond=0)
    return start <= local < end


@contextlib.contextmanager
def heavy_job_guard(
    label: str,
    *,
    lock_path: Path = DEFAULT_LOCK_PATH,
    block_live_window: bool = True,
    min_available_mb: float = 4096,
    min_swap_free_mb: float = 4096,
) -> Iterator[GuardResult]:
    """Non-blocking single-flight guard for heavy bars/feed/matrix jobs."""
    if block_live_window and os.getenv("CYNOLYCUS_ALLOW_LIVE_HEAVY_JOBS", "0") != "1":
        if is_blocked_live_window():
            yield GuardResult(False, f"{label}: blocked during live startup/open window")
            return

    mem = memory_budget_ok(min_available_mb=min_available_mb, min_swap_free_mb=min_swap_free_mb)
    if not mem.ok and os.getenv("CYNOLYCUS_ALLOW_LOW_MEMORY_HEAVY_JOBS", "0") != "1":
        yield GuardResult(False, f"{label}: {mem.reason}")
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("a+")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield GuardResult(False, f"{label}: another heavy data job is already running")
            return
        fh.seek(0)
        fh.truncate()
        fh.write(f"{label} pid={os.getpid()} started={dt.datetime.now(dt.timezone.utc).isoformat()}\n")
        fh.flush()
        yield GuardResult(True, mem.reason)
    finally:
        try:
            # Clear normal-completion metadata while this process still owns
            # the lock. A crash may leave stale text, but flock remains the
            # authority and the next owner will overwrite it.
            fh.seek(0)
            fh.truncate()
            fh.flush()
        except Exception:
            logger.debug("failed to clear heavy-job owner metadata", exc_info=True)
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            logger.debug("failed to unlock heavy-job guard", exc_info=True)
        fh.close()
