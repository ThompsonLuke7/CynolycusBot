"""route_option_or_shares: options for good chains, shares otherwise (no spread gate).

Open interest and volume come from Schwab's chain via `core.option_liquidity`,
not from Alpaca's option snapshot -- the snapshot has neither field, so the old
lookup returned 0 for every contract and routed every candidate to shares. The
`liquidity` fixture stubs that lookup; `liquidity(None)` is the "cannot
determine" case, which must not be reported as zero.
"""
from __future__ import annotations

import pytest

from core.option_liquidity import ContractLiquidity
from signals.meta_context.meta_ranker import options_exec as ox


class _FakeClient:
    """Serves one call contract for a target expiry with a configurable quote."""

    def __init__(self, *, bid=1.0, ask=1.5, delta=0.45, have_contract=True):
        self.bid, self.ask, self.delta = bid, ask, delta
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
        }}


@pytest.fixture(autouse=True)
def liquidity(monkeypatch):
    """Stub the chain lookup. Healthy by default so tests must opt into failure;
    autouse so no test can accidentally reach the real Schwab endpoint."""
    holder: dict[str, ContractLiquidity | None] = {
        "value": ContractLiquidity(open_interest=1000, volume=500, source="test")
    }
    monkeypatch.setattr(
        ox, "contract_liquidity",
        lambda _u, *, expiry, strike, option_type="C": holder["value"],  # noqa: ARG005
    )

    def _set(oi: int | None, vol: int | None = None):
        holder["value"] = (
            None if oi is None
            else ContractLiquidity(open_interest=oi, volume=int(vol or 0), source="test")
        )

    return _set


def _route(client, price):
    return ox.route_option_or_shares(client, "ABC", price, roll_trading_days=5)


def test_healthy_chain_routes_option(liquidity):
    liquidity(1000, 500)
    route, order, reason = _route(_FakeClient(), price=50.0)
    assert route == "option" and reason == "ok"
    assert order["open_interest"] == 1000 and order["volume"] == 500
    assert order["liquidity_source"] == "test"


def test_cheap_underlying_routes_shares():
    route, _order, reason = _route(_FakeClient(), price=8.0)
    assert route == "equity" and reason == "underlying_lt_10"


def test_low_open_interest_routes_shares(liquidity):
    liquidity(100, 500)
    route, _order, reason = _route(_FakeClient(), price=50.0)
    assert route == "equity" and reason.startswith("illiquid_option(oi=100")


def test_low_volume_routes_shares(liquidity):
    liquidity(1000, 10)
    route, _order, reason = _route(_FakeClient(), price=50.0)
    assert route == "equity" and "vol=10" in reason


def test_no_contract_routes_shares():
    route, _order, reason = _route(_FakeClient(have_contract=False), price=50.0)
    assert route == "equity" and reason.startswith("no_option(")


def test_wide_spread_is_not_a_gate(liquidity):
    # 0.50 / 1.50 = 67% spread but healthy OI/vol -> still an option (no spread filter).
    liquidity(1000, 500)
    route, order, reason = _route(_FakeClient(bid=0.5, ask=1.5), price=50.0)
    assert route == "option" and reason == "ok"
    assert order["spread"] > 0.5  # reported for audit, not gated


def test_undeterminable_liquidity_routes_shares_with_its_own_reason(liquidity):
    """The 2026-07-24 fleet outage: a liquidity source that returns nothing must
    not be recorded as `illiquid_option(oi=0,vol=0)`, which made a broken data
    path look like a market with no open interest."""
    liquidity(None)
    route, _order, reason = _route(_FakeClient(), price=50.0)
    assert route == "equity"
    assert reason.startswith("liquidity_unavailable(")
    assert "illiquid_option" not in reason


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
