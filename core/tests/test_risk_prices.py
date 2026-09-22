"""Current-quote safeguards for the between-bar underlying stop."""
from __future__ import annotations

import pandas as pd

from core.risk_prices import CurrentUnderlyingPrices


class _Quotes:
    def __init__(self, quote):
        self.quote = quote
        self.calls = []

    def get_stock_quotes(self, **kwargs):
        self.calls.append(kwargs)
        return {"quotes": {"ABC": self.quote}}


def test_current_underlying_price_requires_a_fresh_two_sided_quote():
    now = pd.Timestamp("2026-09-18T14:00:00Z")
    client = _Quotes({"t": "2026-09-18T13:59:30Z", "bp": 99, "ap": 101})
    prices = CurrentUnderlyingPrices(client, ["ABC"], now, feed="iex")

    assert prices.get("ABC") == 100
    assert client.calls == [{"symbols": "ABC", "feed": "iex"}]


def test_stale_underlying_quote_falls_back_instead_of_using_an_old_close():
    now = pd.Timestamp("2026-09-18T14:00:00Z")
    client = _Quotes({"t": "2026-09-18T13:55:00Z", "bp": 99, "ap": 101})

    assert CurrentUnderlyingPrices(client, ["ABC"], now, max_age_seconds=60).get("ABC") is None
