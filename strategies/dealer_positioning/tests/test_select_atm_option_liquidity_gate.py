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

Liquidity itself comes from Schwab's chain, not Alpaca's option snapshot: the
snapshot payload carries no ``openInterest``/``dailyVolume`` at all, so reading
them there returned 0 for every contract and force-routed every candidate to
shares. These tests stub `core.option_liquidity.contract_liquidity`, which is
the real source, and cover the third state it introduced -- liquidity that
cannot be determined, which must not read as zero.
"""
from __future__ import annotations

from datetime import date, timedelta
from zoneinfo import ZoneInfo

import pytest

from core.option_liquidity import ContractLiquidity
from strategies.dealer_positioning import live_ranked_options
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


@pytest.fixture
def liquidity(monkeypatch):
    """Stub the Schwab-backed liquidity lookup. `set(None)` = cannot determine."""
    holder: dict[str, ContractLiquidity | None] = {"value": None}

    def _fake(_underlying, *, expiry, strike, option_type="C"):  # noqa: ARG001
        return holder["value"]

    monkeypatch.setattr(live_ranked_options, "contract_liquidity", _fake)

    def _set(oi: int | None, vol: int | None = None):
        holder["value"] = (
            None if oi is None
            else ContractLiquidity(open_interest=oi, volume=int(vol or 0), source="test")
        )

    return _set


class _FakeClient:
    def __init__(self, *, bid: float = 0.03, ask: float = 2.16):
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
                "greeks": {"delta": 0.55},
            }
        }


def _select(client):
    return _select_atm_option(
        client, "IOT", 31.71, option_type="call", min_dte=1, max_dte=21, now_et=_now_et(),
    )


def test_zero_open_interest_contract_is_rejected(liquidity):
    liquidity(0, 0)
    order, reason = _select(_FakeClient())
    assert order is None
    assert reason == "illiquid_option(oi=0,vol=0)"


def test_below_min_volume_contract_is_rejected(liquidity):
    liquidity(600, 5)
    order, reason = _select(_FakeClient())
    assert order is None
    assert reason == "illiquid_option(oi=600,vol=5)"


def test_liquid_contract_is_still_selected(liquidity):
    liquidity(750, 250)
    order, reason = _select(_FakeClient(bid=2.10, ask=2.20))
    assert reason == "ok"
    assert order is not None
    assert order["open_interest"] == 750
    assert order["volume"] == 250


def test_undeterminable_liquidity_is_not_reported_as_zero(liquidity):
    """The whole-fleet outage: an unreachable liquidity source read as oi=0,vol=0
    and silently force-routed every candidate to shares. Unknown must be its own
    reason so the broken data path is visible in the audit."""
    liquidity(None)
    order, reason = _select(_FakeClient(bid=2.10, ask=2.20))
    assert order is None
    assert reason == "liquidity_unavailable(src=unavailable)"
    assert "illiquid_option" not in reason
