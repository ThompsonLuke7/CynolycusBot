"""
SharedBarStream — process-wide singleton that maintains one Alpaca WebSocket
connection and fan-outs 1m bars to all registered queues.

Used by UI/combined_server.py so the intraday SPY dashboard and the
multi-ticker swing dashboard share a single connection and stay within Alpaca
IEX's one-concurrent-stream limit.

When running standalone (python -m UI.live_dashboard or UI.swing_dashboard)
the shared stream is never started, so each dashboard falls back to creating
its own AlpacaBarStreamer as before — no behaviour change.
"""
from __future__ import annotations

import logging
import queue as queue_mod
import threading
import time
from datetime import datetime
from typing import Iterable
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_instance: "SharedBarStream | None" = None


def get_shared_bar_stream() -> "SharedBarStream":
    global _instance
    with _lock:
        if _instance is None:
            _instance = SharedBarStream()
        return _instance


class SharedBarStream:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queues: list[queue_mod.Queue] = []
        self._queue_labels: dict[int, str] = {}
        self._queue_symbols: dict[int, set[str] | None] = {}
        self._started = False
        self._streamer = None
        self._symbols: list[str] = []
        self._env_file = ".env"
        self._last_bar_monotonic: float | None = None
        self._last_bar_symbol: str | None = None
        self._last_bar_ts: object | None = None
        self._stream_start_monotonic: float | None = None
        self._delivered_count = 0
        self._dropped_count = 0
        self._last_drop_log_monotonic = 0.0
        self._watchdog_thread: threading.Thread | None = None
        self._stop_watchdog = threading.Event()
        self._reconnect_lock = threading.Lock()
        self._reconnect_count = 0
        # Reviving a dead stream thread is now allowed outside RTH, so it
        # needs a brake: the failure that killed it (Alpaca's
        # `connection limit exceeded`) is one that reconnecting in a tight
        # loop would keep re-triggering.
        self._revive_attempts = 0
        self._next_revive_monotonic = 0.0

    def is_started(self) -> bool:
        with self._lock:
            return self._started

    def register(
        self,
        q: queue_mod.Queue,
        *,
        name: str | None = None,
        symbols: Iterable[str] | None = None,
    ) -> None:
        """Subscribe a queue to receive every incoming bar dict."""
        symbol_filter = None
        if symbols is not None:
            symbol_filter = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
        with self._lock:
            if q not in self._queues:
                self._queues.append(q)
            if name:
                self._queue_labels[id(q)] = name
            self._queue_symbols[id(q)] = symbol_filter
        logger.debug("SharedBarStream: registered queue %s (%d total)", name or "?", len(self._queues))

    def unregister(self, q: queue_mod.Queue) -> None:
        """Stop delivering bars to this queue."""
        with self._lock:
            self._queues = [x for x in self._queues if x is not q]
            self._queue_labels.pop(id(q), None)
            self._queue_symbols.pop(id(q), None)
        logger.debug("SharedBarStream: unregistered queue (%d total)", len(self._queues))

    def start(self, symbols: Iterable[str], env_file: str = ".env") -> None:
        """
        Start the WebSocket stream. Call once at process startup with the
        union of all symbols needed by any runner.
        """
        with self._lock:
            if self._started:
                logger.warning("SharedBarStream already started — ignoring duplicate start().")
                return
            self._started = True

        symbols_list = sorted(set(symbols))
        self._symbols = symbols_list
        self._env_file = env_file

        streamer = self._new_streamer(symbols_list, env_file, self._make_on_bar())
        streamer.start_in_thread(daemon=True)
        with self._lock:
            self._streamer = streamer
            self._stream_start_monotonic = time.monotonic()
        self._start_watchdog()
        logger.info("SharedBarStream started: %d symbols.", len(symbols_list))

    def _new_streamer(self, symbols: list[str], env_file: str, on_bar):
        from alpaca.data.enums import DataFeed
        from core.API.Alpaca_API.market_data.live_stream import AlpacaBarStreamer

        return AlpacaBarStreamer(
            symbols=symbols,
            feed=DataFeed.IEX,
            env_file=env_file,
            on_bar=on_bar,
        )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "started": self._started,
                "registered_queues": len(self._queues),
                "queue_labels": list(self._queue_labels.values()),
                "queue_symbols": {
                    self._queue_labels.get(queue_id, str(queue_id)): (
                        sorted(symbols) if symbols is not None else None
                    )
                    for queue_id, symbols in self._queue_symbols.items()
                },
                "symbols": list(self._symbols),
                "env_file": self._env_file,
                "last_bar_symbol": self._last_bar_symbol,
                "last_bar_ts": self._last_bar_ts,
                "delivered_count": self._delivered_count,
                "dropped_count": self._dropped_count,
                "reconnect_count": self._reconnect_count,
            }

    def _make_on_bar(self):
        def _on_bar(bar: dict) -> None:
            self._fanout_bar(bar)

        return _on_bar

    def _fanout_bar(self, bar: dict) -> None:
        now = time.monotonic()
        with self._lock:
            self._last_bar_monotonic = now
            # Bars are flowing: the next outage starts from a clean slate.
            self._revive_attempts = 0
            self._next_revive_monotonic = 0.0
            self._last_bar_symbol = str(bar.get("symbol") or "")
            self._last_bar_ts = bar.get("timestamp")
            queues = [
                (q, self._queue_symbols.get(id(q)))
                for q in self._queues
            ]
        symbol = str(bar.get("symbol") or "").strip().upper()
        for q, symbol_filter in queues:
            if symbol_filter is not None and symbol not in symbol_filter:
                continue
            try:
                q.put_nowait(bar)
                with self._lock:
                    self._delivered_count += 1
            except queue_mod.Full:
                replaced = False
                try:
                    q.get_nowait()
                    q.put_nowait(bar)
                    replaced = True
                except Exception:
                    replaced = False
                if replaced:
                    with self._lock:
                        self._delivered_count += 1
                self._record_drop(q)

    @staticmethod
    def _is_rth_watch_window() -> bool:
        now = datetime.now(ZoneInfo("America/New_York"))
        if now.weekday() >= 5:
            return False
        minutes = now.hour * 60 + now.minute
        return (9 * 60 + 35) <= minutes <= (16 * 60 + 5)

    def _start_watchdog(self) -> None:
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        self._stop_watchdog.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name="shared-bar-stream-watchdog",
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        while not self._stop_watchdog.wait(30.0):
            if not self.is_started():
                continue
            # Staleness is only meaningful inside RTH — no bars at 03:00 is
            # normal. A thread that has *exited* is not: it will never
            # produce another bar at any hour, and waiting for the RTH
            # window to notice costs the open. On 2026-09-03 the streamer
            # died at 07:05 ET on `connection limit exceeded` and nothing
            # reconnected until 09:35, five minutes into the session.
            watch_window = self._is_rth_watch_window()
            with self._lock:
                streamer = self._streamer
                last_bar = self._last_bar_monotonic
                last_symbol = self._last_bar_symbol
                last_ts = self._last_bar_ts
                stream_started = self._stream_start_monotonic
            alive = bool(streamer and getattr(streamer, "is_alive", lambda: False)())
            err = getattr(streamer, "thread_error", None) if streamer is not None else None
            no_bars_yet = last_bar is None
            clock = time.monotonic()
            stale_secs = (clock - last_bar) if last_bar is not None else None
            startup_wait_secs = (clock - stream_started) if stream_started is not None else None
            if alive:
                # Only a live thread gets the benefit of the staleness rules,
                # and those apply inside the watch window alone.
                if not watch_window:
                    continue
                if not no_bars_yet and stale_secs is not None and stale_secs < 180:
                    continue
                if no_bars_yet and startup_wait_secs is not None and startup_wait_secs < 180:
                    continue
            else:
                with self._lock:
                    next_try = self._next_revive_monotonic
                    attempts = self._revive_attempts
                if clock < next_try:
                    continue
                backoff = min(30.0 * (2 ** attempts), 600.0)
                with self._lock:
                    self._revive_attempts = attempts + 1
                    self._next_revive_monotonic = clock + backoff
            reason = "no bars received since startup" if no_bars_yet else f"last bar stale for {stale_secs:.0f}s"
            if not alive:
                reason = f"stream thread not alive; {reason}"
            if err is not None:
                reason = f"{reason}; thread_error={err}"
            logger.warning(
                "SharedBarStream watchdog reconnecting: %s last_symbol=%s last_ts=%s",
                reason, last_symbol, last_ts,
            )
            self._reconnect(reason=reason)

    def _record_drop(self, q: queue_mod.Queue) -> None:
        now = time.monotonic()
        with self._lock:
            self._dropped_count += 1
            dropped = self._dropped_count
            delivered = self._delivered_count
            label = self._queue_labels.get(id(q), "?")
            should_log = now - self._last_drop_log_monotonic >= 60.0
            if should_log:
                self._last_drop_log_monotonic = now
        if should_log:
            try:
                qsize = q.qsize()
                maxsize = q.maxsize
            except Exception:
                qsize = "?"
                maxsize = "?"
            logger.warning(
                "SharedBarStream queue full; dropping bars for one subscriber "
                "(subscriber=%s queue=%s/%s delivered=%d dropped=%d)",
                label,
                qsize,
                maxsize,
                delivered,
                dropped,
            )

    def _reconnect(self, *, reason: str) -> None:
        if not self._reconnect_lock.acquire(blocking=False):
            return
        try:
            with self._lock:
                if not self._started:
                    return
                old_streamer = self._streamer
                symbols = list(self._symbols)
                env_file = self._env_file
            if not symbols:
                return
            if old_streamer is not None:
                try:
                    old_streamer.stop()
                except Exception:
                    pass
                try:
                    old_streamer.join(timeout=5.0)
                except Exception:
                    pass

            new_streamer = self._new_streamer(symbols, env_file, self._make_on_bar())
            new_streamer.start_in_thread(daemon=True)
            with self._lock:
                self._streamer = new_streamer
                self._last_bar_monotonic = None
                self._last_bar_symbol = None
                self._last_bar_ts = None
                self._stream_start_monotonic = time.monotonic()
                self._reconnect_count += 1
            logger.info(
                "SharedBarStream reconnected (%d): %s symbols=%d",
                self._reconnect_count, reason, len(symbols),
            )
        finally:
            self._reconnect_lock.release()

    def stop(self) -> None:
        with self._lock:
            self._started = False
            streamer = self._streamer
            self._streamer = None
        self._stop_watchdog.set()
        if streamer is not None:
            try:
                streamer.stop()
            except Exception:
                pass
            try:
                streamer.join(timeout=5.0)
            except Exception:
                pass
        logger.info("SharedBarStream stopped.")
