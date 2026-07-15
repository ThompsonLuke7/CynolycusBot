"""route_option_or_shares: options for good chains, shares otherwise (no spread gate)."""
from __future__ import annotations

import pytest

from signals.meta_context.meta_ranker import options_exec as ox


class _FakeClient:
    """Serves one call contract for a target expiry with configurable OI/vol/quote."""

    def __init__(self, *, oi=1000, vol=500, bid=1.0, ask=1.5, delta=0.45, have_contract=True):
        self.oi, self.vol, self.bid, self.ask, self.delta = oi, vol, bid, ask, delta
        self.have_contract = have_contract

    def get_option_contracts(self, **_):
        if not self.have_contract:
            return {"option_contracts": []}
        return {"option_contracts": [{"symbol": "ABC260717C00010000", "strike_price": "10",
                                      "underlying_symbol": "ABC", "type": "call"}]}

    def get_option_snapshots(self, *_a, **_k):
        return {"ABC260717C00010000": {
            "greeks": {"delta": self.delta},
            "latestQuote": {"bp": self.bid, "ap": self.ask},
            "openInterest": self.oi, "dailyVolume": self.vol,
        }}


def _route(client, price):
    return ox.route_option_or_shares(client, "ABC", price, roll_trading_days=5)


def test_healthy_chain_routes_option():
    route, order, reason = _route(_FakeClient(oi=1000, vol=500), price=50.0)
    assert route == "option" and reason == "ok"
    assert order["open_interest"] == 1000 and order["volume"] == 500


def test_cheap_underlying_routes_shares():
    route, _order, reason = _route(_FakeClient(), price=8.0)
    assert route == "equity" and reason == "underlying_lt_10"


def test_low_open_interest_routes_shares():
    route, _order, reason = _route(_FakeClient(oi=100, vol=500), price=50.0)
    assert route == "equity" and reason.startswith("illiquid_option(oi=100")


def test_low_volume_routes_shares():
    route, _order, reason = _route(_FakeClient(oi=1000, vol=10), price=50.0)
    assert route == "equity" and "vol=10" in reason


def test_no_contract_routes_shares():
    route, _order, reason = _route(_FakeClient(have_contract=False), price=50.0)
    assert route == "equity" and reason.startswith("no_option(")


def test_wide_spread_is_not_a_gate():
    # 0.50 / 1.50 = 67% spread but healthy OI/vol -> still an option (no spread filter).
    route, order, reason = _route(_FakeClient(oi=1000, vol=500, bid=0.5, ask=1.5), price=50.0)
    assert route == "option" and reason == "ok"
    assert order["spread"] > 0.5  # reported for audit, not gated


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
