from __future__ import annotations

from datetime import datetime, timezone

from core.API.Schwab_API.live_stream import chart_bar_to_dict
from core.API.Schwab_API.schwab_client import SchwabClient


def test_chart_bar_to_dict_normalizes_schwab_chart_payload() -> None:
    payload = chart_bar_to_dict(
        {
            "key": "SPY",
            "CHART_TIME_MILLIS": 1781796600000,
            "OPEN_PRICE": 600.25,
            "HIGH_PRICE": 601.0,
            "LOW_PRICE": 599.75,
            "CLOSE_PRICE": 600.5,
            "VOLUME": 12345,
        }
    )

    assert payload == {
        "symbol": "SPY",
        "timestamp": datetime(2026, 6, 18, 15, 30, tzinfo=timezone.utc),
        "open": 600.25,
        "high": 601.0,
        "low": 599.75,
        "close": 600.5,
        "volume": 12345.0,
    }


def test_schwab_client_prefers_matching_stream_account_hash() -> None:
    client = SchwabClient.__new__(SchwabClient)
    client.get_account_numbers = lambda: [
        {"accountNumber": "111111111", "hashValue": "wrong"},
        {"accountNumber": "222222222", "hashValue": "target"},
    ]

    account_id = client.get_stream_account_id(preferred_hash="target")

    assert account_id == "222222222"


def test_schwab_client_falls_back_to_first_available_stream_account() -> None:
    client = SchwabClient.__new__(SchwabClient)
    client.get_account_numbers = lambda: [
        {"accountNumber": "333333333", "hashValue": "first"},
        {"accountNumber": "444444444", "hashValue": "second"},
    ]

    account_id = client.get_stream_account_id(preferred_hash="missing")

    assert account_id == "333333333"
