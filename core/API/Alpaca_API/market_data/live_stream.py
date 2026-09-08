from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional
import queue as queue_mod

from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream
from alpaca.data.models import Bar, Quote, Trade, TradingStatus

from ..core.config import AlpacaConfig

# Alpaca SDK logger that emits connection errors
_ALPACA_WS_LOGGER = "alpaca.data.live.websocket"

# Error substrings that are permanent (retrying will never succeed)
_FATAL_PHRASES = ("connection limit exceeded",)

BarCallback = Callable[[dict], None]
MarketEventCallback = Callable[[dict], None]


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


def _event_value(event: Any, *names: str) -> Any:
    """Read an SDK model or raw websocket payload without changing its time."""

    if isinstance(event, dict):
        for name in names:
            if name in event:
                return event[name]
        return None
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return value
    return None


def quote_to_dict(quote: Quote | dict, *, feed: DataFeed) -> dict:
    """Normalize a two-sided quote for timestamp-preserving strategy buffers."""

    timestamp = _event_value(quote, "timestamp", "t")
    if timestamp is None:
        raise ValueError("quote event has no timestamp")
    return {
        "ticker": str(_event_value(quote, "symbol", "S") or "").upper(),
        "timestamp": _to_utc(timestamp),
        "bid": float(_event_value(quote, "bid_price", "bp")),
        "ask": float(_event_value(quote, "ask_price", "ap")),
        "bid_size": float(_event_value(quote, "bid_size", "bs") or 0.0),
        "ask_size": float(_event_value(quote, "ask_size", "as") or 0.0),
        "feed": feed.value.upper(),
    }


def trade_to_dict(trade: Trade | dict, *, feed: DataFeed) -> dict:
    """Normalize a trade print for timestamp-preserving strategy buffers."""

    timestamp = _event_value(trade, "timestamp", "t")
    if timestamp is None:
        raise ValueError("trade event has no timestamp")
    return {
        "ticker": str(_event_value(trade, "symbol", "S") or "").upper(),
        "timestamp": _to_utc(timestamp),
        "price": float(_event_value(trade, "price", "p")),
        "size": float(_event_value(trade, "size", "s")),
        "feed": feed.value.upper(),
    }


def trading_status_to_dict(status: TradingStatus | dict, *, feed: DataFeed) -> dict:
    """Normalize halt/resume notices.  The original status code is retained."""

    timestamp = _event_value(status, "timestamp", "t")
    if timestamp is None:
        raise ValueError("trading-status event has no timestamp")
    return {
        "ticker": str(_event_value(status, "symbol", "S") or "").upper(),
        "timestamp": _to_utc(timestamp),
        "status": str(_event_value(status, "status_code", "sc") or ""),
        "reason": str(_event_value(status, "reason_code", "rc") or ""),
        "feed": feed.value.upper(),
    }


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
        self._drop_count = 0

    async def _handle_bar(self, bar: Bar) -> None:
        payload = bar_to_dict(bar)
        if self._queue is not None:
            try:
                self._queue.put_nowait(payload)
            except queue_mod.Full:
                # Drop newest on overflow to avoid backpressure in the stream callback.
                self._drop_count += 1
                if self._drop_count in {1, 10} or self._drop_count % 100 == 0:
                    logging.getLogger(__name__).warning(
                        "Alpaca stream queue full; dropped %s bar(s). Latest bar=%s %s",
                        self._drop_count,
                        payload.get("symbol"),
                        payload.get("timestamp"),
                    )
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


class AlpacaQuoteTradeStatusStreamer:
    """One paper-safe market-data stream for quotes, trades, and halt statuses.

    This is deliberately data-only: it creates no trading client and exposes no
    order operation.  Callers receive the vendor event timestamp and a separate
    ``received_at`` timestamp, which prevents a live strategy from treating a
    late observation as if it had been available earlier.
    """

    def __init__(
        self,
        *,
        symbols: Iterable[str],
        feed: DataFeed,
        env_file: str | None = ".env",
        on_quote: MarketEventCallback | None = None,
        on_trade: MarketEventCallback | None = None,
        on_status: MarketEventCallback | None = None,
        stream: StockDataStream | None = None,
    ) -> None:
        if feed is not DataFeed.SIP:
            raise ValueError("quote-aware momentum execution requires the SIP feed")
        self._symbols = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
        if not self._symbols:
            raise ValueError("At least one symbol is required.")
        self._feed = feed
        self._on_quote = on_quote
        self._on_trade = on_trade
        self._on_status = on_status
        if stream is None:
            cfg = AlpacaConfig.from_env(env_file)
            stream = StockDataStream(cfg.key_id, cfg.secret_key, feed=feed)
        self._stream = stream
        self._thread: Optional[threading.Thread] = None
        self._thread_error: BaseException | None = None

    @staticmethod
    def _received_at() -> datetime:
        return datetime.now(timezone.utc)

    async def _handle_quote(self, quote: Quote | dict) -> None:
        if self._on_quote is not None:
            payload = quote_to_dict(quote, feed=self._feed)
            payload["received_at"] = self._received_at()
            self._on_quote(payload)

    async def _handle_trade(self, trade: Trade | dict) -> None:
        if self._on_trade is not None:
            payload = trade_to_dict(trade, feed=self._feed)
            payload["received_at"] = self._received_at()
            self._on_trade(payload)

    async def _handle_status(self, status: TradingStatus | dict) -> None:
        if self._on_status is not None:
            payload = trading_status_to_dict(status, feed=self._feed)
            payload["received_at"] = self._received_at()
            self._on_status(payload)

    def start(self) -> None:
        self._stream.subscribe_quotes(self._handle_quote, *self._symbols)
        self._stream.subscribe_trades(self._handle_trade, *self._symbols)
        self._stream.subscribe_trading_statuses(self._handle_status, *self._symbols)
        self._stream.run()

    def start_in_thread(self, daemon: bool = True) -> None:
        if self._thread and self._thread.is_alive():
            return

        def _target() -> None:
            try:
                self.start()
            except BaseException as exc:
                self._thread_error = exc
                logging.getLogger(__name__).exception(
                    "AlpacaQuoteTradeStatusStreamer thread exited: %s", exc
                )

        self._thread_error = None
        self._thread = threading.Thread(
            target=_target, daemon=daemon, name="alpaca-quote-trade-status-stream"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stream.stop()

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def thread_error(self) -> BaseException | None:
        return self._thread_error
