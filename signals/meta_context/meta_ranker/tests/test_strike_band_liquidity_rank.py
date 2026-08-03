"""select_option ranks the whole delta band by liquidity, not by delta alone.

Worked example this locks in (live, 2026-07-28): CRWV at ref $66.51 selected the
delta-closest August call, which carried oi=85/vol=8, failed the routing floors
and traded 75 shares instead. The 70 strike -- same expiry, still inside the
[0.35, 0.60] delta band -- held >3,000 contracts of open interest and would have
routed as an option. Forcing one specific strike and then gating that strike
rejects names whose chain is plainly liquid; the band is the risk control, so
within it prefer the strike that can actually be traded.
"""
from __future__ import annotations

import pytest

from core.option_liquidity import ContractLiquidity
from signals.meta_context.meta_ranker import options_exec as ox

# (strike, delta, open_interest, volume) -- a CRWV-shaped August band at ~$66.51.
# The 68 strike is delta-closest to the 0.45 target but nearly untraded; the
# round 70 strike is deeper in the band and genuinely liquid.
_BAND = [
    (66.0, 0.57, 120, 15),
    (68.0, 0.46, 85, 8),
    (70.0, 0.41, 3200, 640),
    (75.0, 0.36, 900, 210),
]


def _occ(strike: float) -> str:
    return f"CRWV260821C{int(strike * 1000):08d}"


class _BandClient:
    """Serves a multi-strike call chain for one expiry."""

    def __init__(self, band=_BAND, *, quoteless: set[float] | None = None):
        self.band = band
        self.quoteless = quoteless or set()
        self.quote_calls: list[str] = []

    def get_option_contracts(self, **_):
        return {"option_contracts": [
            {"symbol": _occ(k), "strike_price": str(k),
             "underlying_symbol": "CRWV", "type": "call"}
            for k, *_rest in self.band
        ]}

    def get_option_snapshots(self, *_a, **_k):
        out = {}
        for strike, delta, _oi, _vol in self.band:
            snap = {"greeks": {"delta": delta}}
            if strike not in self.quoteless:
                snap["latestQuote"] = {"bp": 4.0, "ap": 4.2}
            out[_occ(strike)] = snap
        return out

    def get_option_quotes(self, symbols):
        self.quote_calls.append(symbols)
        return {"quotes": {}}  # no recoverable quote


@pytest.fixture(autouse=True)
def liquidity(monkeypatch):
    """Stub the Schwab chain lookup, keyed by strike. Autouse so no test can
    reach the real endpoint; returns a setter for tests that serve a different
    band than `_BAND`."""
    holder = {"table": {k: (oi, vol) for k, _d, oi, vol in _BAND}}

    def _lookup(_underlying, *, expiry, strike, option_type="C"):  # noqa: ARG001
        hit = holder["table"].get(round(float(strike), 4))
        if hit is None:
            return None
        return ContractLiquidity(open_interest=hit[0], volume=hit[1], source="test")

    monkeypatch.setattr(ox, "contract_liquidity", _lookup)

    def _set(band):
        holder["table"] = {k: (oi, vol) for k, _d, oi, vol in band}

    return _set


def _select(client, **kw):
    return ox.select_option(client, "CRWV", 66.51, 1e12, roll_trading_days=5, **kw)


def test_picks_deepest_open_interest_in_band_not_delta_closest():
    order, reason = _select(_BandClient())
    assert reason == "ok"
    # 68 is delta-closest (0.46 vs target 0.45) but has oi=85; 70 clears the
    # floors with oi=3200 and wins.
    assert order["strike"] == 70.0
    assert order["open_interest"] == 3200
    assert order["band_size"] == 4


def test_selected_contract_now_routes_as_an_option():
    """End-to-end: the same chain that produced 75 CRWV shares routes options."""
    route, order, reason = ox.route_option_or_shares(_BandClient(), "CRWV", 66.51)
    assert (route, reason) == ("option", "ok")
    assert order["strike"] == 70.0


def test_thin_chain_rejects_on_the_bands_best_strike(liquidity):
    """A genuinely thin chain must still reject -- reporting the DEEPEST-OI
    strike in the band, so the reason is evidence about the whole chain rather
    than about one arbitrary delta-closest strike."""
    thin = [(66.0, 0.57, 12, 1), (68.0, 0.46, 9, 0), (70.0, 0.41, 30, 4)]
    liquidity(thin)
    client = _BandClient(thin)
    order, reason = _select(client)
    assert reason == "ok"
    assert order["strike"] == 70.0
    route, routed, route_reason = ox.route_option_or_shares(client, "CRWV", 66.51)
    assert route == "equity"
    assert route_reason == "illiquid_option(oi=30,vol=4)"  # the band's ceiling
    assert routed["strike"] == 70.0  # rejected contract is reported for audit


def test_unknown_liquidity_degrades_to_delta_closest(monkeypatch):
    """Schwab down: every OI is None, so ranking must not reorder the band."""
    monkeypatch.setattr(
        ox, "contract_liquidity",
        lambda *_a, **_k: None,
    )
    order, reason = _select(_BandClient())
    assert reason == "ok"
    assert order["strike"] == 68.0
    assert order["open_interest"] is None
    assert order["liquidity_source"] == "unavailable"


def test_skips_an_unquotable_strike_and_takes_the_next_best():
    """The deepest-OI strike with no two-sided quote must not sink the name."""
    client = _BandClient(quoteless={70.0})
    order, reason = _select(client)
    assert reason == "ok"
    assert order["strike"] == 75.0  # next-deepest that clears the floors
    assert client.quote_calls == [_occ(70.0)]  # one fallback fetch, then moved on


def test_quote_attempts_are_bounded():
    """A chain with no quotes anywhere must not fire one request per strike."""
    client = _BandClient(quoteless={k for k, *_ in _BAND})
    order, reason = _select(client)
    assert (order, reason) == (None, "no_quote")
    assert len(client.quote_calls) <= ox._MAX_QUOTE_ATTEMPTS
