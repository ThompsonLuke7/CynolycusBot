from __future__ import annotations

from datetime import datetime, timezone

import pytest

from strategies.momentum_scalper.data.live_buffer import LiveMarketBuffer
from strategies.momentum_scalper.live.ledger import ShadowLedger
from strategies.momentum_scalper.live.market_feed import MomentumSIPMarketFeed


UTC = timezone.utc


def test_live_buffer_preserves_event_and_received_times() -> None:
    buffer = LiveMarketBuffer(max_events_per_kind=2)
    event_at = datetime(2026, 8, 10, 8, 0, tzinfo=UTC)
    received_at = datetime(2026, 8, 10, 8, 0, 1, tzinfo=UTC)
    buffer.ingest_quote({"ticker": "rUnR", "timestamp": event_at, "bid": 9.99, "ask": 10.0, "feed": "SIP"}, received_at=received_at)

    _, quotes, _ = buffer.frames()
    assert quotes.iloc[0]["ticker"] == "RUNR"
    assert buffer.quote_age_seconds(ticker="RUNR", now=datetime(2026, 8, 10, 8, 0, 2, tzinfo=UTC)) == 1.0
    with pytest.raises(ValueError, match="non-crossed"):
        buffer.ingest_quote({"ticker": "RUNR", "timestamp": event_at, "bid": 10.1, "ask": 10.0}, received_at=received_at)


def test_shadow_ledger_is_append_only(tmp_path) -> None:
    ledger = ShadowLedger(tmp_path)
    first = ledger.append("decisions", {"timestamp": datetime(2026, 8, 10, 8, 0, tzinfo=UTC), "reason": "ok"})
    ledger.append("decisions", {"timestamp": datetime(2026, 8, 10, 8, 1, tzinfo=UTC), "reason": "reject"})

    assert first.read_text(encoding="utf-8").count("\n") == 2


class _NoopStreamer:
    def start_in_thread(self) -> None:
        return None

    def stop(self) -> None:
        return None


def test_sip_market_feed_wires_received_time_into_strategy_buffer() -> None:
    buffer = LiveMarketBuffer()
    feed = MomentumSIPMarketFeed(symbols=["RUNR"], buffer=buffer, streamer=_NoopStreamer())  # type: ignore[arg-type]
    event_at = datetime(2026, 8, 10, 8, 0, tzinfo=UTC)
    received_at = datetime(2026, 8, 10, 8, 0, 1, tzinfo=UTC)

    feed._on_quote({"ticker": "RUNR", "timestamp": event_at, "bid": 9.99, "ask": 10.0, "feed": "SIP", "received_at": received_at})
    feed._on_trade({"ticker": "RUNR", "timestamp": event_at, "price": 10.0, "size": 100, "feed": "SIP", "received_at": received_at})
    feed._on_status({"ticker": "RUNR", "timestamp": event_at, "status": "H", "received_at": received_at})

    trades, quotes, statuses = buffer.frames()
    assert len(trades) == len(quotes) == len(statuses) == 1
    assert quotes.iloc[0]["received_at"] == received_at
