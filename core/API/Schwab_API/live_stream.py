from __future__ import annotations

import asyncio
import logging
import os
import queue as queue_mod
import threading
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional


BarCallback = Callable[[dict], None]
_LOGGER = logging.getLogger(__name__)


def _to_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def chart_bar_to_dict(bar: dict) -> dict | None:
    if not isinstance(bar, dict):
        return None
    if str(bar.get("key", "")).strip().upper() == "":
        return None
    ts_ms = bar.get("CHART_TIME_MILLIS")
    if ts_ms is None:
        return None
    try:
        timestamp = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)
        return {
            "symbol": str(bar.get("key", "")).strip().upper(),
            "timestamp": timestamp,
            "open": float(bar["OPEN_PRICE"]),
            "high": float(bar["HIGH_PRICE"]),
            "low": float(bar["LOW_PRICE"]),
            "close": float(bar["CLOSE_PRICE"]),
            "volume": float(bar["VOLUME"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


class SchwabBarStreamer:
    """
    Schwab CHART_EQUITY wrapper that mirrors the live runner interface used by
    AlpacaBarStreamer.
    """

    def __init__(
        self,
        *,
        symbols: Iterable[str],
        queue: Optional[queue_mod.Queue] = None,
        on_bar: Optional[BarCallback] = None,
        account_id: str | None = None,
        preferred_account_hash: str | None = None,
    ) -> None:
        self._symbols = [s.strip().upper() for s in symbols if s and str(s).strip()]
        if not self._symbols:
            raise ValueError("At least one symbol is required.")
        self._queue = queue
        self._on_bar = on_bar
        self._account_id = str(account_id).strip() if account_id is not None else ""
        self._preferred_account_hash = preferred_account_hash
        self._thread: Optional[threading.Thread] = None
        self._thread_error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._message_task: asyncio.Task[None] | None = None
        self._drop_count = 0

    def _resolve_account_id(self) -> str:
        if self._account_id:
            return self._account_id
        env_account = str(os.getenv("SCHWAB_STREAM_ACCOUNT_ID", "")).strip()
        if env_account:
            self._account_id = env_account
            return self._account_id

        from core.API.Schwab_API.schwab_client import SchwabClient

        client = SchwabClient()
        self._account_id = client.get_stream_account_id(preferred_hash=self._preferred_account_hash)
        return self._account_id

    def _emit_payload(self, payload: dict) -> None:
        if self._queue is not None:
            try:
                self._queue.put_nowait(payload)
            except queue_mod.Full:
                self._drop_count += 1
                if self._drop_count in {1, 10} or self._drop_count % 100 == 0:
                    _LOGGER.warning(
                        "Schwab stream queue full; dropped %s bar(s). Latest bar=%s %s",
                        self._drop_count,
                        payload.get("symbol"),
                        payload.get("timestamp"),
                    )
                return
        if self._on_bar is not None:
            self._on_bar(payload)

    def _handle_chart_message(self, message: dict) -> None:
        if not isinstance(message, dict):
            return
        if str(message.get("service", "")).strip().upper() != "CHART_EQUITY":
            return
        content = message.get("content")
        if not isinstance(content, list):
            return
        for item in content:
            payload = chart_bar_to_dict(item)
            if payload is None:
                continue
            if payload["symbol"] not in self._symbols:
                continue
            self._emit_payload(payload)

    async def _run_message_loop(self, stream_client: object) -> None:
        handle_message = getattr(stream_client, "handle_message", None)
        if handle_message is None:
            raise RuntimeError("Schwab StreamClient does not expose handle_message().")
        while self._stop_event is not None and not self._stop_event.is_set():
            await handle_message()

    async def _run_async(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        account_id = self._resolve_account_id()

        from core.API.Schwab_API.schwab_client import SchwabClient
        from schwab.streaming import StreamClient

        client = SchwabClient()
        stream_client = StreamClient(client.client, account_id=account_id)
        await stream_client.login()
        stream_client.add_chart_equity_handler(self._handle_chart_message)
        await stream_client.chart_equity_subs(self._symbols)
        self._message_task = asyncio.create_task(self._run_message_loop(stream_client))
        try:
            await self._stop_event.wait()
        finally:
            if self._message_task is not None:
                self._message_task.cancel()
                try:
                    await self._message_task
                except asyncio.CancelledError:
                    pass
            try:
                await stream_client.logout()
            except Exception:
                pass
            self._message_task = None
            self._stop_event = None
            self._loop = None

    def start(self) -> None:
        asyncio.run(self._run_async())

    def start_in_thread(self, daemon: bool = True) -> None:
        if self._thread and self._thread.is_alive():
            return

        def _target() -> None:
            try:
                self.start()
            except BaseException as exc:
                self._thread_error = exc
                _LOGGER.exception("SchwabBarStreamer thread exited with error: %s", exc)

        self._thread_error = None
        self._thread = threading.Thread(target=_target, daemon=daemon, name="schwab-bar-stream")
        self._thread.start()

    def stop(self) -> None:
        loop = self._loop
        stop_event = self._stop_event
        message_task = self._message_task
        if loop is None or stop_event is None:
            return
        loop.call_soon_threadsafe(stop_event.set)
        if message_task is not None:
            loop.call_soon_threadsafe(message_task.cancel)

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def thread_error(self) -> BaseException | None:
        return self._thread_error
