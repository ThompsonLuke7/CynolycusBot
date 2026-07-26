"""Coverage for the Schwab-backed per-contract liquidity lookup.

Background: the 4H modules and Dealer Ranker gated option routing on open
interest and volume read from Alpaca's option-snapshot payload, which contains
neither field. Both always coerced to 0, so the floors never passed and every
candidate fell back to shares -- 294 equity routes and 0 option routes across
the whole recorded live history, with every reject reading the identical
``illiquid_option(oi=0,vol=0)``.
"""
from __future__ import annotations

from datetime import date

import pytest

from core import option_liquidity
from core.option_liquidity import ContractLiquidity, contract_liquidity, reset_cache


class _Resp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class _FakeSchwab:
    """Minimal stand-in for schwab-py's client."""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self._status = status_code
        self.calls: list[dict] = []

    def get_option_chain(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp(self._payload, self._status)


def _chain(strike=135.0, oi=1200, volume=450):
    return {
        "callExpDateMap": {
            "2026-08-21:28": {
                str(strike): [{
                    "putCall": "CALL",
                    "strikePrice": strike,
                    "openInterest": oi,
                    "totalVolume": volume,
                }]
            }
        },
        "putExpDateMap": {
            "2026-08-21:28": {
                str(strike): [{
                    "putCall": "PUT",
                    "strikePrice": strike,
                    "openInterest": 77,
                    "totalVolume": 88,
                }]
            }
        },
    }


@pytest.fixture(autouse=True)
def _clean_cache():
    reset_cache()
    yield
    reset_cache()


def _install(monkeypatch, fake):
    monkeypatch.setattr(option_liquidity, "_get_raw_client", lambda: fake)


def test_reads_open_interest_and_volume_from_the_chain(monkeypatch):
    fake = _FakeSchwab(_chain())
    _install(monkeypatch, fake)
    liq = contract_liquidity("CDW", expiry="2026-08-21", strike=135.0, option_type="C")
    assert liq == ContractLiquidity(open_interest=1200, volume=450, source="schwab_chain")


def test_put_and_call_sides_are_not_confused(monkeypatch):
    _install(monkeypatch, _FakeSchwab(_chain()))
    put = contract_liquidity("CDW", expiry="2026-08-21", strike=135.0, option_type="P")
    assert put is not None
    assert (put.open_interest, put.volume) == (77, 88)


def test_accepts_a_date_object_and_a_datetime_string(monkeypatch):
    _install(monkeypatch, _FakeSchwab(_chain()))
    assert contract_liquidity("CDW", expiry=date(2026, 8, 21), strike=135.0) is not None
    assert contract_liquidity("CDW", expiry="2026-08-21T00:00:00Z", strike=135.0) is not None


def test_unknown_strike_returns_none_not_zero(monkeypatch):
    """The whole point of the rewrite: absence must be distinguishable from 0."""
    _install(monkeypatch, _FakeSchwab(_chain(strike=135.0)))
    assert contract_liquidity("CDW", expiry="2026-08-21", strike=999.0) is None


def test_http_error_returns_none(monkeypatch):
    _install(monkeypatch, _FakeSchwab(_chain(), status_code=401))
    assert contract_liquidity("CDW", expiry="2026-08-21", strike=135.0) is None


def test_missing_schwab_client_returns_none(monkeypatch):
    monkeypatch.setattr(option_liquidity, "_get_raw_client", lambda: None)
    assert contract_liquidity("CDW", expiry="2026-08-21", strike=135.0) is None


def test_genuinely_zero_open_interest_is_reported_as_zero(monkeypatch):
    _install(monkeypatch, _FakeSchwab(_chain(oi=0, volume=0)))
    liq = contract_liquidity("CDW", expiry="2026-08-21", strike=135.0)
    assert liq is not None
    assert (liq.open_interest, liq.volume) == (0, 0)


def test_chain_is_fetched_once_per_expiry(monkeypatch):
    fake = _FakeSchwab(_chain())
    _install(monkeypatch, fake)
    for _ in range(4):
        contract_liquidity("CDW", expiry="2026-08-21", strike=135.0)
    assert len(fake.calls) == 1


def test_cache_is_scoped_per_underlying(monkeypatch):
    fake = _FakeSchwab(_chain())
    _install(monkeypatch, fake)
    contract_liquidity("CDW", expiry="2026-08-21", strike=135.0)
    contract_liquidity("RNG", expiry="2026-08-21", strike=135.0)
    assert len(fake.calls) == 2
    assert {c["symbol"] for c in fake.calls} == {"CDW", "RNG"}
