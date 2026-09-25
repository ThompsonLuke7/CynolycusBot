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


class _QuotesAndBars(_Quotes):
    def __init__(self, quote, bar):
        super().__init__(quote)
        self.bar = bar

    def get_latest_stock_bars(self, **kwargs):
        self.calls.append(("bars", kwargs))
        return {"bars": {"ABC": self.bar}}


NOW = pd.Timestamp("2026-09-18T14:00:00Z")


def _prices(client, *, fetched=NOW, **kw):
    return CurrentUnderlyingPrices(client, ["ABC"], NOW, clock=lambda: fetched, **kw)


def test_current_underlying_price_requires_a_fresh_two_sided_quote():
    client = _Quotes({"t": "2026-09-18T13:59:30Z", "bp": 99, "ap": 101})
    prices = _prices(client, feed="iex")

    assert prices.get("ABC") == 100
    assert prices.source["ABC"] == "quote"
    assert client.calls == [{"symbols": "ABC", "feed": "iex"}]


def test_stale_underlying_quote_falls_back_instead_of_using_an_old_close():
    client = _Quotes({"t": "2026-09-18T13:55:00Z", "bp": 99, "ap": 101})

    assert _prices(client, max_age_seconds=60).get("ABC") is None


def test_quote_newer_than_pass_start_is_fresh_not_future():
    """The 09-17 bug: now_et is taken at pass start, the quote is fetched ~2s later
    and is newer than now_et. It must be accepted, not rejected as negative age."""
    client = _Quotes({"t": "2026-09-18T14:00:02Z", "bp": 99, "ap": 101})
    assert _prices(client, fetched=NOW + pd.Timedelta(seconds=3)).get("ABC") == 100


def test_quote_beyond_clock_skew_is_rejected():
    client = _Quotes({"t": "2026-09-18T14:01:00Z", "bp": 99, "ap": 101})
    assert _prices(client).get("ABC") is None


def test_age_is_measured_at_fetch_time_not_pass_start():
    """A quote fresh at pass start but >60s old by the time it is fetched is stale."""
    client = _Quotes({"t": "2026-09-18T14:00:00Z", "bp": 99, "ap": 101})
    assert _prices(client, fetched=NOW + pd.Timedelta(seconds=90)).get("ABC") is None


def test_stale_quote_uses_a_recent_one_minute_bar_close():
    client = _QuotesAndBars({"t": "2026-09-18T13:55:00Z", "bp": 99, "ap": 101},
                            {"t": "2026-09-18T13:58:00Z", "c": 98.5})
    prices = _prices(client)
    assert prices.get("ABC") == 98.5
    assert prices.source["ABC"] == "bar"


def test_old_bar_does_not_price_the_stop():
    client = _QuotesAndBars({"t": "2026-09-18T13:55:00Z", "bp": 99, "ap": 101},
                            {"t": "2026-09-18T13:50:00Z", "c": 98.5})
    prices = _prices(client, max_bar_age_seconds=300)
    assert prices.get("ABC") is None
    assert prices.source["ABC"] is None


def test_quote_request_failure_still_tries_the_bar():
    class _Down(_QuotesAndBars):
        def get_stock_quotes(self, **kwargs):
            raise RuntimeError("down")
    client = _Down(None, {"t": "2026-09-18T13:59:00Z", "c": 97.0})
    assert _prices(client).get("ABC") == 97.0
