from __future__ import annotations

import pandas as pd

from strategies.momentum_scalper.data.validation import assess_market_feed


def test_source_validation_refuses_single_exchange_or_missing_quote_evidence() -> None:
    trades = pd.DataFrame([
        {"ticker": "RUNR", "timestamp": "2026-08-10T08:00:00Z", "price": 10.0, "size": 100, "feed": "IEX"},
    ])
    quotes = pd.DataFrame([
        {"ticker": "RUNR", "timestamp": "2026-08-10T08:00:00Z", "bid": 9.99, "ask": 10.0, "feed": "IEX"},
    ])

    report = assess_market_feed(trades=trades, quotes=quotes, expected_feed="SIP")

    assert report["trades_schema_ok"]
    assert report["quotes_schema_ok"]
    assert report["quotes_expected_feed_present"] is False
    assert report["fit_for_quote_aware_replay"] is False
