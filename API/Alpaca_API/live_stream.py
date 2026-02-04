from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional
import queue as queue_mod

from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream
from alpaca.data.models import Bar

from .config import AlpacaConfig

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
        self._stream.run()

    def start_in_thread(self, daemon: bool = True) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.start, daemon=daemon)
        self._thread.start()

    def stop(self) -> None:
        self._stream.stop()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)
