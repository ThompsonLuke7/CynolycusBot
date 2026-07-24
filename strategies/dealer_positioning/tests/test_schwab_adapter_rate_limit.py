"""Schwab chain lookups must retry a 429 (rate limited) instead of failing the
whole poll cycle for that symbol.

2026-07-21 live audit: SPY/IWM/SLV/QQQ each hit a 429 from Schwab's option-chain
endpoint at least once during the RTH poll window, and `_poll_symbol` gave up
immediately on any non-2xx response, logging "dealer poll failed" and skipping
that symbol for the whole cycle. A couple of short backoff-and-retry attempts
recover most of these transient rate limits.
"""
from __future__ import annotations

from datetime import date

import pytest

from strategies.dealer_positioning.config import DealerPositioningConfig
from strategies.dealer_positioning.schwab_adapter import SchwabDealerDataClient
from strategies.dealer_positioning.tests.test_dealer_positioning_core import _chain_fixture


class _Response:
    def __init__(self, status_code: int, headers: dict | None = None):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = "rate limited" if status_code == 429 else ""

    def json(self) -> dict:
        return _chain_fixture()


class _RawClient:
    class Options:
        class ContractType:
            ALL = "ALL"

        class Strategy:
            SINGLE = "SINGLE"

    def __init__(self, responses: list[_Response]):
        self._responses = list(responses)
        self.calls = 0

    def get_option_chain(self, **kwargs):
        resp = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return resp


def _adapter(raw_client: _RawClient) -> SchwabDealerDataClient:
    adapter = SchwabDealerDataClient.__new__(SchwabDealerDataClient)
    adapter._client = type("_Client", (), {"client": raw_client})()
    adapter._config = DealerPositioningConfig(dte_offsets=(0, 1, 2))
    return adapter


def test_recovers_after_one_429_then_a_200(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("strategies.dealer_positioning.schwab_adapter.time.sleep", lambda s: sleeps.append(s))

    raw_client = _RawClient([_Response(429), _Response(200)])
    payload = _adapter(raw_client).get_option_chain("SPY", date(2026, 7, 21))

    assert raw_client.calls == 2
    assert payload["_requested_symbol"] == "SPY"
    assert sleeps == [2.0]  # default backoff, no Retry-After header


def test_honors_retry_after_header(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("strategies.dealer_positioning.schwab_adapter.time.sleep", lambda s: sleeps.append(s))

    raw_client = _RawClient([_Response(429, headers={"Retry-After": "5"}), _Response(200)])
    _adapter(raw_client).get_option_chain("SPY", date(2026, 7, 21))

    assert sleeps == [5.0]


def test_gives_up_after_max_retries_and_raises(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("strategies.dealer_positioning.schwab_adapter.time.sleep", lambda s: sleeps.append(s))

    raw_client = _RawClient([_Response(429), _Response(429), _Response(429)])
    with pytest.raises(RuntimeError, match="Schwab chain lookup failed"):
        _adapter(raw_client).get_option_chain("SPY", date(2026, 7, 21))

    assert raw_client.calls == 3  # initial attempt + 2 retries, then give up
    assert len(sleeps) == 2  # slept before each retry, not after the final failure


def test_non_429_error_does_not_retry(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("strategies.dealer_positioning.schwab_adapter.time.sleep", lambda s: sleeps.append(s))

    raw_client = _RawClient([_Response(400)])
    with pytest.raises(RuntimeError, match="Schwab chain lookup failed"):
        _adapter(raw_client).get_option_chain("SPY", date(2026, 7, 21))

    assert raw_client.calls == 1
    assert sleeps == []
