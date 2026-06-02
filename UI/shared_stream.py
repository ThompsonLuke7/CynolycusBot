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

    def is_started(self) -> bool:
        with self._lock:
            return self._started

    def register(self, q: queue_mod.Queue, *, name: str | None = None) -> None:
        """Subscribe a queue to receive every incoming bar dict."""
        with self._lock:
            if q not in self._queues:
                self._queues.append(q)
            if name:
                self._queue_labels[id(q)] = name
        logger.debug("SharedBarStream: registered queue %s (%d total)", name or "?", len(self._queues))

    def unregister(self, q: queue_mod.Queue) -> None:
        """Stop delivering bars to this queue."""
        with self._lock:
            self._queues = [x for x in self._queues if x is not q]
            self._queue_labels.pop(id(q), None)
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

        def _on_bar(bar: dict) -> None:
            now = time.monotonic()
            with self._lock:
                self._last_bar_monotonic = now
                self._last_bar_symbol = str(bar.get("symbol") or "")
                self._last_bar_ts = bar.get("timestamp")
                queues = list(self._queues)
            for q in queues:
                try:
                    q.put_nowait(bar)
                    with self._lock:
                        self._delivered_count += 1
                except queue_mod.Full:
                    self._record_drop(q)

        streamer = self._new_streamer(symbols_list, env_file, _on_bar)
        streamer.start_in_thread(daemon=True)
        with self._lock:
            self._streamer = streamer
            self._stream_start_monotonic = time.monotonic()
        self._start_watchdog()
        logger.info("SharedBarStream started: %d symbols.", len(symbols_list))

    def _new_streamer(self, symbols: list[str], env_file: str, on_bar):
        from alpaca.data.enums import DataFeed
        from API.Alpaca_API.market_data.live_stream import AlpacaBarStreamer

        return AlpacaBarStreamer(
            symbols=symbols,
            feed=DataFeed.IEX,
            env_file=env_file,
            on_bar=on_bar,
        )

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
            if not self.is_started() or not self._is_rth_watch_window():
                continue
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
            if alive and not no_bars_yet and stale_secs is not None and stale_secs < 180:
                continue
            if alive and no_bars_yet and startup_wait_secs is not None and startup_wait_secs < 180:
                continue
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

            def _on_bar(bar: dict) -> None:
                now = time.monotonic()
                with self._lock:
                    self._last_bar_monotonic = now
                    self._last_bar_symbol = str(bar.get("symbol") or "")
                    self._last_bar_ts = bar.get("timestamp")
                    queues = list(self._queues)
                for q in queues:
                    try:
                        q.put_nowait(bar)
                        with self._lock:
                            self._delivered_count += 1
                    except queue_mod.Full:
                        self._record_drop(q)

            new_streamer = self._new_streamer(symbols, env_file, _on_bar)
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
