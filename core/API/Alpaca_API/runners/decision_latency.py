"""Stage-by-stage latency instrumentation for the SPY interval decision path.

WHY: on 2026-07-30 the SPY daytrader's 10-minute decisions were recorded 2.4 to
19.9 minutes *after* their bar closed, with no relationship to CPU load (mean
7.8m with no heavy job running vs 9.3m with one; the fastest decision of the day
happened during the readiness catch-up). The lag arrived in bursts — a long
stall, then three bars cleared two minutes apart — which looks like a consumer
draining a backlog rather than a fixed pipeline delay. The run artifacts only
carried the bar timestamp and the final write time, so every intermediate stage
was invisible and the cause could not be established.

This module makes each stage measurable. It records, per decision:

* ``bar_close -> trigger_arrival`` — a bucket only closes when the *first bar of
  the next bucket* arrives, so a gap in the 1-minute stream defers the close by
  exactly that gap. This is the leading hypothesis and the main thing to read.
* ``trigger_arrival -> handler_start`` — time spent queued behind other stream
  work before this decision was picked up.
* ``inference_sec`` / ``policy_sec`` — model scoring and order-policy time.
* ``bar_close -> emitted`` — the end-to-end number the daily report quotes.

Everything here is best-effort: a failure to measure must never disturb the
trading path, so every public method swallows its own exceptions.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from core.live_signal_audit import append_jsonl


def _to_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        import pandas as pd

        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        return None if parsed is None or pd.isna(parsed) else parsed.to_pydatetime()
    except Exception:
        return None


def _seconds(later: datetime | None, earlier: datetime | None) -> float | None:
    if later is None or earlier is None:
        return None
    return round((later - earlier).total_seconds(), 3)


@dataclass
class _PendingClose:
    """What the aggregator knew at the moment a bucket rolled over."""

    bar_close_utc: datetime | None
    trigger_bar_ts_utc: datetime | None
    trigger_arrival_utc: datetime
    #: 1-minute bars actually seen inside the bucket. A short count means the
    #: feed had gaps, which is what defers the close.
    bars_in_bucket: int
    #: Ingest lag of the triggering bar itself (arrival - its own timestamp).
    trigger_ingest_sec: float | None


class DecisionLatency:
    """Collects per-stage timings and writes one record per decision."""

    def __init__(
        self,
        *,
        log_path: Path | None,
        interval_minutes: int,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._log_path = log_path
        #: Injectable so tests can pin lag arithmetic without patching datetime.
        self._now = clock or (lambda: datetime.now(timezone.utc))
        self._interval_minutes = int(interval_minutes)
        self._pending: dict[str, _PendingClose] = {}
        self._bucket_counts: dict[str, int] = {}
        self._last_bar_arrival: dict[str, datetime] = {}
        self._stages: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return self._log_path is not None

    # -- ingest side -------------------------------------------------------

    def on_1m_bar(self, symbol: str, bar: dict) -> None:
        """Count bars per bucket and remember when the newest one landed."""
        if not self.enabled:
            return
        try:
            now = self._now()
            self._last_bar_arrival[symbol] = now
            self._bucket_counts[symbol] = self._bucket_counts.get(symbol, 0) + 1
        except Exception:
            pass

    def on_bucket_closed(self, symbol: str, closed_bar: dict, trigger_bar: dict) -> None:
        """A bucket rolled because ``trigger_bar`` (the next bucket's first bar) arrived."""
        if not self.enabled:
            return
        try:
            now = self._now()
            bucket_start = _to_utc(closed_bar.get("timestamp"))
            bar_close = (
                bucket_start.timestamp() + self._interval_minutes * 60 if bucket_start else None
            )
            bar_close_utc = (
                datetime.fromtimestamp(bar_close, timezone.utc) if bar_close else None
            )
            trigger_ts = _to_utc(trigger_bar.get("timestamp"))
            # on_1m_bar already counted the triggering bar, but that bar belongs
            # to the *next* bucket — it is what rolled this one over. Attribute
            # it forward, or every bucket reads one bar too full and a gappy
            # feed looks complete.
            counted = self._bucket_counts.get(symbol, 0)
            self._pending[symbol] = _PendingClose(
                bar_close_utc=bar_close_utc,
                trigger_bar_ts_utc=trigger_ts,
                trigger_arrival_utc=now,
                bars_in_bucket=max(0, counted - 1),
                trigger_ingest_sec=_seconds(now, trigger_ts),
            )
            self._bucket_counts[symbol] = 1
        except Exception:
            pass

    # -- decision side -----------------------------------------------------

    @contextmanager
    def stage(self, name: str):
        """Time one stage of the handler. Never raises on its own."""
        started = time.monotonic()
        try:
            yield
        finally:
            try:
                self._stages[name] = round(time.monotonic() - started, 3)
            except Exception:
                pass

    def emit(self, symbol: str, closed_bar: dict, *, handler_start: float | None = None) -> None:
        """Write the assembled record for one decision and reset stage timers."""
        if not self.enabled:
            return
        try:
            now = self._now()
            pending = self._pending.pop(symbol, None)
            bucket_start = _to_utc(closed_bar.get("timestamp"))
            bar_close_utc = pending.bar_close_utc if pending else None
            record: dict[str, Any] = {
                "event": "decision_latency",
                "module": "spy_day_trader",
                "symbol": symbol,
                "interval_minutes": self._interval_minutes,
                "bar_start_utc": bucket_start.isoformat() if bucket_start else None,
                "bar_close_utc": bar_close_utc.isoformat() if bar_close_utc else None,
                "emitted_utc": now.isoformat(),
                # The headline number, and the one the daily report quotes.
                "total_lag_after_close_sec": _seconds(now, bar_close_utc),
                "stages_sec": dict(self._stages),
            }
            if pending is not None:
                record.update(
                    {
                        # Leading hypothesis: the bucket cannot close until the
                        # next bucket's first bar shows up.
                        "close_detection_lag_sec": _seconds(
                            pending.trigger_arrival_utc, bar_close_utc
                        ),
                        "trigger_bar_ts_utc": (
                            pending.trigger_bar_ts_utc.isoformat()
                            if pending.trigger_bar_ts_utc
                            else None
                        ),
                        "trigger_arrival_utc": pending.trigger_arrival_utc.isoformat(),
                        "trigger_ingest_sec": pending.trigger_ingest_sec,
                        "bars_in_bucket": pending.bars_in_bucket,
                        "expected_bars_in_bucket": self._interval_minutes,
                        "handler_queue_sec": (
                            round(handler_start - pending.trigger_arrival_utc.timestamp(), 3)
                            if handler_start is not None
                            else None
                        ),
                    }
                )
            append_jsonl(self._log_path, record)
        except Exception:
            pass
        finally:
            self._stages = {}
