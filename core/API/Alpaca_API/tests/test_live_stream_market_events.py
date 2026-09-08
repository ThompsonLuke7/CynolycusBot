from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from alpaca.data.enums import DataFeed

from core.API.Alpaca_API.market_data.live_stream import (
    AlpacaQuoteTradeStatusStreamer,
    quote_to_dict,
    trade_to_dict,
    trading_status_to_dict,
)


UTC = timezone.utc


def test_market_event_normalizers_preserve_vendor_event_time_and_feed() -> None:
    when = datetime(2026, 8, 31, 8, 0, tzinfo=UTC)
    quote = quote_to_dict(
        {"S": "runr", "t": when, "bp": 9.99, "ap": 10.0, "bs": 100, "as": 200},
        feed=DataFeed.SIP,
    )
    trade = trade_to_dict({"S": "runr", "t": when, "p": 10.0, "s": 50}, feed=DataFeed.SIP)
    status = trading_status_to_dict(
        {"S": "runr", "t": when, "sc": "H", "rc": "T1"}, feed=DataFeed.SIP
    )

    assert quote == {
        "ticker": "RUNR", "timestamp": when, "bid": 9.99, "ask": 10.0,
        "bid_size": 100.0, "ask_size": 200.0, "feed": "SIP",
    }
    assert trade == {"ticker": "RUNR", "timestamp": when, "price": 10.0, "size": 50.0, "feed": "SIP"}
    assert status == {"ticker": "RUNR", "timestamp": when, "status": "H", "reason": "T1", "feed": "SIP"}


class _FakeStream:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.handlers = {}

    def subscribe_quotes(self, handler, *symbols) -> None:
        self.calls.append(("quotes", symbols))
        self.handlers["quotes"] = handler

    def subscribe_trades(self, handler, *symbols) -> None:
        self.calls.append(("trades", symbols))
        self.handlers["trades"] = handler

    def subscribe_trading_statuses(self, handler, *symbols) -> None:
        self.calls.append(("statuses", symbols))
        self.handlers["statuses"] = handler

    def run(self) -> None:
        return None

    def stop(self) -> None:
        return None


def test_market_event_stream_subscribes_all_required_event_kinds() -> None:
    fake = _FakeStream()
    received: list[dict] = []
    streamer = AlpacaQuoteTradeStatusStreamer(
        symbols=["runr"], feed=DataFeed.SIP, stream=fake,  # type: ignore[arg-type]
        on_quote=received.append, on_trade=received.append, on_status=received.append,
    )
    streamer.start()

    assert fake.calls == [("quotes", ("RUNR",)), ("trades", ("RUNR",)), ("statuses", ("RUNR",))]
    when = datetime(2026, 8, 31, 8, 0, tzinfo=UTC)
    asyncio.run(fake.handlers["quotes"]({"S": "RUNR", "t": when, "bp": 9.9, "ap": 10.0}))
    assert received[0]["timestamp"] == when
    assert received[0]["received_at"] >= when


def test_market_event_stream_refuses_incomplete_iex_feed() -> None:
    try:
        AlpacaQuoteTradeStatusStreamer(symbols=["RUNR"], feed=DataFeed.IEX, stream=_FakeStream())  # type: ignore[arg-type]
    except ValueError as exc:
        assert "SIP" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected SIP-only safety gate")
