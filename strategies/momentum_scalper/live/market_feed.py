"""SIP market-data wiring for a preselected momentum candidate list.

The class is intentionally order-blind.  It only moves broker observations
into the strategy's timestamp-preserving buffer after the premarket scanner has
chosen symbols to monitor.
"""
from __future__ import annotations

from collections.abc import Iterable

from alpaca.data.enums import DataFeed

from core.API.Alpaca_API.market_data.live_stream import AlpacaQuoteTradeStatusStreamer
from strategies.momentum_scalper.data.live_buffer import LiveMarketBuffer


class MomentumSIPMarketFeed:
    """Subscribe to SIP quote/trade/halt events for a bounded candidate set."""

    def __init__(
        self,
        *,
        symbols: Iterable[str],
        buffer: LiveMarketBuffer,
        env_file: str | None = ".env",
        streamer: AlpacaQuoteTradeStatusStreamer | None = None,
    ) -> None:
        self.buffer = buffer
        self.streamer = streamer or AlpacaQuoteTradeStatusStreamer(
            symbols=symbols,
            feed=DataFeed.SIP,
            env_file=env_file,
            on_quote=self._on_quote,
            on_trade=self._on_trade,
            on_status=self._on_status,
        )

    def _on_quote(self, event: dict) -> None:
        received_at = event["received_at"]
        self.buffer.ingest_quote(event, received_at=received_at)

    def _on_trade(self, event: dict) -> None:
        received_at = event["received_at"]
        self.buffer.ingest_trade(event, received_at=received_at)

    def _on_status(self, event: dict) -> None:
        received_at = event["received_at"]
        self.buffer.ingest_status(event, received_at=received_at)

    def start_in_thread(self) -> None:
        self.streamer.start_in_thread()

    def stop(self) -> None:
        self.streamer.stop()


__all__ = ["MomentumSIPMarketFeed"]
