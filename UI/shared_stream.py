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
from typing import Iterable

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
        self._started = False
        self._streamer = None

    def is_started(self) -> bool:
        with self._lock:
            return self._started

    def register(self, q: queue_mod.Queue) -> None:
        """Subscribe a queue to receive every incoming bar dict."""
        with self._lock:
            if q not in self._queues:
                self._queues.append(q)
        logger.debug("SharedBarStream: registered queue (%d total)", len(self._queues))

    def unregister(self, q: queue_mod.Queue) -> None:
        """Stop delivering bars to this queue."""
        with self._lock:
            self._queues = [x for x in self._queues if x is not q]
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

        from alpaca.data.enums import DataFeed
        from API.Alpaca_API.market_data.live_stream import AlpacaBarStreamer

        symbols_list = sorted(set(symbols))

        feed = DataFeed.IEX

        def _on_bar(bar: dict) -> None:
            with self._lock:
                queues = list(self._queues)
            for q in queues:
                try:
                    q.put_nowait(bar)
                except queue_mod.Full:
                    pass

        streamer = AlpacaBarStreamer(
            symbols=symbols_list,
            feed=feed,
            env_file=env_file,
            on_bar=_on_bar,
        )
        streamer.start_in_thread(daemon=True)
        with self._lock:
            self._streamer = streamer
        logger.info("SharedBarStream started: %d symbols.", len(symbols_list))

    def stop(self) -> None:
        with self._lock:
            self._started = False
            streamer = self._streamer
            self._streamer = None
        if streamer is not None:
            try:
                streamer.stop()
            except Exception:
                pass
        logger.info("SharedBarStream stopped.")
