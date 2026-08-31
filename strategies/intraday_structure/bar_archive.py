"""Append-only 1-minute bar archive — the missing input for a counterfactual.

WHY. Step D2 asks what the same candidates would have done under a different
configuration (price-only vs +levels vs +dealer). That is a counterfactual, and
a live ledger cannot answer it: the ledger observes the one arm that actually
ran. Answering it needs the same bars replayed under each config, and
``Data/shared/bars/`` holds 1d/1h/4h only. The gap has been open since
2026-07-28.

It is forward-only. Nothing reconstructs a bar that was not stored, so every
session without this is a session D2 can never use.

Design points that matter:

* **Two clocks, kept apart.** ``timestamp`` is when the bar happened;
  ``arrival_at`` is when this process received it. A bar that shows up 90
  seconds late was not available for a decision at its own timestamp, and only
  the second clock can say so.
* **JSONL, not parquet, during the session.** Appending a line is crash-safe and
  costs nothing; rewriting a parquet on every flush is neither. Compaction to
  parquet is a separate, idempotent step.
* **Off the hot path.** The bar-consumer loop has backed the shared queue up
  before (2026-07-20, ~27% of bars dropped for a session). Writes are buffered
  and flushed on a size or time threshold, never per bar.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
ARCHIVE_SCHEMA_VERSION = "intraday_1m_bar_archive_v1"


@dataclass(frozen=True)
class ArchiveStats:
    buffered: int
    written: int
    flushes: int
    dropped: int


class BarArchive:
    """Buffered append-only writer, one JSONL file per ET session date."""

    def __init__(
        self,
        root: str | Path = "Data/archive/intraday_1m",
        *,
        flush_every: int = 500,
        flush_seconds: float = 60.0,
        max_buffer: int = 20_000,
    ) -> None:
        if max_buffer < flush_every:
            logger.warning(
                "bar archive max_buffer=%d is below flush_every=%d; bars will be "
                "dropped before a size-triggered flush can run", max_buffer, flush_every,
            )
        self.root = Path(root)
        self.flush_every = max(1, int(flush_every))
        self.flush_seconds = max(1.0, float(flush_seconds))
        # Taken literally. Clamping this up to flush_every silently overrode a
        # caller who asked for a small hard cap, and the cap is the one thing
        # here that exists to bound damage.
        self.max_buffer = max(1, int(max_buffer))
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._written = 0
        self._flushes = 0
        self._dropped = 0

    # -- writing ----------------------------------------------------------
    def record(self, payload: dict[str, Any], *, arrival_at: datetime | None = None) -> None:
        """Buffer one bar exactly as it arrived. Never raises into the caller."""
        try:
            symbol = str(payload.get("symbol") or payload.get("ticker") or "").strip().upper()
            timestamp = payload.get("timestamp")
            if not symbol or timestamp is None:
                return
            row = {
                "symbol": symbol,
                "timestamp": _iso(timestamp),
                "arrival_at": (arrival_at or datetime.now(timezone.utc)).isoformat(),
                "open": _num(payload.get("open")),
                "high": _num(payload.get("high")),
                "low": _num(payload.get("low")),
                "close": _num(payload.get("close")),
                "volume": _num(payload.get("volume")),
                "trade_count": _num(payload.get("trade_count")),
                "vwap": _num(payload.get("vwap")),
            }
        except Exception:
            logger.debug("bar archive could not normalise a payload", exc_info=True)
            return

        with self._lock:
            if len(self._buffer) >= self.max_buffer:
                # Losing the oldest buffered bar beats letting the buffer grow
                # without bound if the disk is wedged. Counted, never silent.
                self._buffer.pop(0)
                self._dropped += 1
            self._buffer.append(row)
            due = (
                len(self._buffer) >= self.flush_every
                or (time.monotonic() - self._last_flush) >= self.flush_seconds
            )
        if due:
            self.flush()

    def flush(self) -> int:
        """Append everything buffered, grouped into its session file."""
        with self._lock:
            pending, self._buffer = self._buffer, []
            self._last_flush = time.monotonic()
        if not pending:
            return 0
        by_session: dict[str, list[dict[str, Any]]] = {}
        for row in pending:
            by_session.setdefault(_session_date(row["timestamp"]), []).append(row)
        written = 0
        for session, rows in by_session.items():
            path = self.path_for(session)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
                written += len(rows)
            except OSError:
                logger.exception("bar archive write failed for %s", session)
                with self._lock:
                    self._dropped += len(rows)
        self._written += written
        self._flushes += 1
        return written

    def path_for(self, session: str) -> Path:
        return self.root / f"bars_{session.replace('-', '')}.jsonl"

    def stats(self) -> ArchiveStats:
        with self._lock:
            return ArchiveStats(len(self._buffer), self._written, self._flushes, self._dropped)

    # -- manifest ---------------------------------------------------------
    def write_manifest(self, session: str, *, extra: dict[str, Any] | None = None) -> Path:
        """Bind a session's bars to the code and config that produced them.

        Without this a replay cannot say which engine wrote the rows it is
        reading, which is the whole point of keeping them.
        """
        path = self.path_for(session)
        manifest = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "session": session,
            "bars_path": str(path),
            "written_at": datetime.now(timezone.utc).isoformat(),
            "git_revision": _git_revision(),
            **_session_quality(path),
            **(extra or {}),
        }
        out = self.root / f"manifest_{session.replace('-', '')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return out


def read_session(path: str | Path) -> list[dict[str, Any]]:
    """Read one session's bars, skipping any line a crash left half-written."""
    file = Path(path)
    if not file.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _session_quality(path: Path) -> dict[str, Any]:
    """Counts a replay needs before it trusts a session: gaps, dupes, late bars."""
    rows = read_session(path)
    if not rows:
        return {"bar_count": 0, "symbols": 0, "duplicates": 0, "late_bars": 0, "max_lateness_seconds": None}
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    lateness: list[float] = []
    for row in rows:
        key = (row["symbol"], row["timestamp"])
        if key in seen:
            duplicates += 1
        seen.add(key)
        try:
            delay = (
                datetime.fromisoformat(row["arrival_at"]) - datetime.fromisoformat(row["timestamp"])
            ).total_seconds()
            lateness.append(delay)
        except (KeyError, TypeError, ValueError):
            continue
    # A 1-minute bar cannot arrive before its own close; >120s means the feed
    # was behind, and a decision stamped at the bar time was not really possible.
    late = [x for x in lateness if x > 120.0]
    return {
        "bar_count": len(rows),
        "symbols": len({r["symbol"] for r in rows}),
        "duplicates": duplicates,
        "late_bars": len(late),
        "max_lateness_seconds": round(max(lateness), 1) if lateness else None,
    }


def _session_date(timestamp: str) -> str:
    try:
        return datetime.fromisoformat(timestamp).astimezone(ET).date().isoformat()
    except (TypeError, ValueError):
        return "unknown"


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    text = str(value)
    return text.replace("Z", "+00:00")


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _git_revision() -> str | None:
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parents[2],
        )
        return out.stdout.strip() or None
    except Exception:
        return None
