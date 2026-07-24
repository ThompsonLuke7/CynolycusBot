"""Regression coverage for the 2026-07-23 IOT260724C00031500 incident.

Dealer Ranker's ATM contract picker chose the nearest-strike non-0DTE option
purely by strike distance, with no floor on open interest or volume -- unlike
its sibling 4H modules (Meta/HTF/Momentum), whose shared
`options_exec.route_option_or_shares` has always required
open_interest >= 500 and volume >= 100 before trading a name as options.

On 2026-07-23 this picked a contract with zero open interest and a bid/ask of
0.03/2.16 (spread ~195% of mid). The fill crossed near the ask; the position
was already down ~98% marked to the bid before the underlying moved at all --
not a bad directional call, a bad contract. `_select_atm_option` now applies
the same liquidity floor and returns None instead of selecting a name this
thin.
"""
from __future__ import annotations

from datetime import date, timedelta
from zoneinfo import ZoneInfo

from strategies.dealer_positioning.live_ranked_options import _select_atm_option

_ET = ZoneInfo("America/New_York")
_TODAY = date(2026, 7, 23)


def _now_et():
    from datetime import datetime
    return datetime(_TODAY.year, _TODAY.month, _TODAY.day, 11, 51, tzinfo=_ET)


def _contract(symbol: str, root: str, expiry: date, strike: float) -> dict:
    return {
        "symbol": symbol,
        "root_symbol": root,
        "expiration_date": expiry.isoformat(),
        "strike_price": str(strike),
        "multiplier": "100",
        "size": "100",
        "tradable": True,
        "type": "call",
    }


class _FakeClient:
    def __init__(self, *, open_interest: int, volume: int, bid: float = 0.03, ask: float = 2.16):
        self.open_interest = open_interest
        self.volume = volume
        self.bid = bid
        self.ask = ask
        self.expiry = _TODAY + timedelta(days=7)
        self.symbol = "IOT260730C00031500"

    def get_option_contracts(self, **_kwargs):
        return {"option_contracts": [_contract(self.symbol, "IOT", self.expiry, 31.5)]}

    def get_option_snapshots(self, *_args, **_kwargs):
        return {
            self.symbol: {
                "latestQuote": {"bp": self.bid, "ap": self.ask},
                "openInterest": self.open_interest,
                "dailyVolume": self.volume,
                "greeks": {"delta": 0.55},
            }
        }


def test_zero_open_interest_contract_is_rejected():
    client = _FakeClient(open_interest=0, volume=0)
    order, reason = _select_atm_option(
        client, "IOT", 31.71, option_type="call", min_dte=1, max_dte=21, now_et=_now_et(),
    )
    assert order is None
    assert reason == "illiquid_option(oi=0,vol=0)"


def test_below_min_volume_contract_is_rejected():
    client = _FakeClient(open_interest=600, volume=5)
    order, reason = _select_atm_option(
        client, "IOT", 31.71, option_type="call", min_dte=1, max_dte=21, now_et=_now_et(),
    )
    assert order is None
    assert reason == "illiquid_option(oi=600,vol=5)"


def test_liquid_contract_is_still_selected():
    client = _FakeClient(open_interest=750, volume=250, bid=2.10, ask=2.20)
    order, reason = _select_atm_option(
        client, "IOT", 31.71, option_type="call", min_dte=1, max_dte=21, now_et=_now_et(),
    )
    assert reason == "ok"
    assert order is not None
    assert order["open_interest"] == 750
    assert order["volume"] == 250
