from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional
import queue as queue_mod

from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream
from alpaca.data.models import Bar

from ..core.config import AlpacaConfig

# Alpaca SDK logger that emits connection errors
_ALPACA_WS_LOGGER = "alpaca.data.live.websocket"

# Error substrings that are permanent (retrying will never succeed)
_FATAL_PHRASES = ("connection limit exceeded",)

BarCallback = Callable[[dict], None]


def _to_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def bar_to_dict(bar: Bar) -> dict:
    """
    Normalize an Alpaca Bar into a dict with the expected schema.
    """
    payload = {
        "symbol": bar.symbol,
        "timestamp": _to_utc(bar.timestamp),
        "open": float(bar.open),
        "high": float(bar.high),
        "low": float(bar.low),
        "close": float(bar.close),
        "volume": float(bar.volume),
    }
    # Optional fields (only if present on the model)
    trade_count = getattr(bar, "trade_count", None)
    vwap = getattr(bar, "vwap", None)
    if trade_count is not None:
        payload["trade_count"] = int(trade_count)
    if vwap is not None:
        payload["vwap"] = float(vwap)
    return payload


class _FatalErrorWatcher(logging.Handler):
    """
    Attaches to the Alpaca WebSocket logger and calls stop() + queues an error
    sentinel the first time a permanent (non-retriable) error is detected.

    This prevents the SDK's built-in retry loop from hammering the server with
    429s when the connection limit is exceeded or credentials are invalid.
    """

    def __init__(
        self,
        stream: StockDataStream,
        queue: Optional[queue_mod.Queue],
        error_holder: list,
    ) -> None:
        super().__init__(level=logging.ERROR)
        self._stream = stream
        self._queue = queue
        self._error_holder = error_holder
        self._triggered = False

    def emit(self, record: logging.LogRecord) -> None:
        if self._triggered:
            return
        msg = record.getMessage().lower()
        for phrase in _FATAL_PHRASES:
            if phrase in msg:
                self._triggered = True
                self._error_holder.append(record.getMessage())
                logging.getLogger(_ALPACA_WS_LOGGER).info(
                    "Fatal WebSocket error detected (%s) — stopping retry loop.", phrase
                )
                try:
                    self._stream.stop()
                except Exception:
                    pass
                if self._queue is not None:
                    try:
                        self._queue.put_nowait({
                            "_sentinel": True,
                            "_error": record.getMessage(),
                        })
                    except Exception:
                        pass
                break


class AlpacaBarStreamer:
    """
    Thin wrapper around StockDataStream that pushes bars into a queue and/or callback.

    Notes:
      - This class uses a background thread when start_in_thread() is called.
      - Use a thread-safe queue.Queue for cross-thread communication.
    """

    def __init__(
        self,
        *,
        symbols: Iterable[str],
        feed: DataFeed = DataFeed.IEX,
        env_file: str | None = ".env",
        queue: Optional[queue_mod.Queue] = None,
        on_bar: Optional[BarCallback] = None,
    ) -> None:
        cfg = AlpacaConfig.from_env(env_file)
        self._symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not self._symbols:
            raise ValueError("At least one symbol is required.")
        self._queue = queue
        self._on_bar = on_bar
        self._stream = StockDataStream(cfg.key_id, cfg.secret_key, feed=feed)
        self._thread: Optional[threading.Thread] = None
        self._thread_error: BaseException | None = None

    async def _handle_bar(self, bar: Bar) -> None:
        payload = bar_to_dict(bar)
        if self._queue is not None:
            try:
                self._queue.put_nowait(payload)
            except queue_mod.Full:
                # Drop newest on overflow to avoid backpressure in the stream callback.
                pass
        if self._on_bar is not None:
            self._on_bar(payload)

    def start(self) -> None:
        self._stream.subscribe_bars(self._handle_bar, *self._symbols)

        # Install the fatal-error watcher before run() so it intercepts the very
        # first "connection limit exceeded" log and tells the SDK to stop retrying.
        error_holder: list[str] = []
        watcher = _FatalErrorWatcher(self._stream, self._queue, error_holder)
        ws_logger = logging.getLogger(_ALPACA_WS_LOGGER)
        ws_logger.addHandler(watcher)
        try:
            self._stream.run()
        finally:
            ws_logger.removeHandler(watcher)

        # If the watcher caught a fatal error, re-raise so the caller knows.
        if error_holder:
            raise ConnectionError(
                f"Alpaca WebSocket fatal error: {error_holder[0]}\n"
                "Check that no other stream is already connected on this account."
            )

    def start_in_thread(self, daemon: bool = True) -> None:
        if self._thread and self._thread.is_alive():
            return

        def _target() -> None:
            try:
                self.start()
            except BaseException as exc:
                self._thread_error = exc
                logging.getLogger(__name__).exception("AlpacaBarStreamer thread exited with error: %s", exc)

        self._thread_error = None
        self._thread = threading.Thread(target=_target, daemon=daemon, name="alpaca-bar-stream")
        self._thread.start()

    def stop(self) -> None:
        self._stream.stop()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def thread_error(self) -> BaseException | None:
        return self._thread_error
