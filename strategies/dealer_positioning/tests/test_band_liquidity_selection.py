"""Liquidity-aware strike selection across the +/-10% band.

2026-08-05: all ten Dealer Ranker targets were rejected as `illiquid_option` for
the third consecutive session — 0 orders across every reviewed day. The chains
were not the problem. The selector took the single nearest-ATM strike and gated
that one contract, so CGNX was rejected on its ATM 70 strike (oi=886, vol=35)
while the 75 strike — same expiry, still inside the band — held oi=1,246 and
vol=185 and clears both floors.

meta_ranker.options_exec._rank_candidates fixed the identical defect on its
delta band on 2026-07-28 (the CRWV case). This is the same rule applied to this
module's strike band: the band is the risk control, and within it prefer the
strike that can actually be traded.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from core.option_liquidity import ContractLiquidity
from strategies.dealer_positioning import live_ranked_options
from strategies.dealer_positioning.live_ranked_options import (
    _rank_band_by_liquidity,
    _select_atm_option,
    _select_optionable_targets,
)

_ET = ZoneInfo("America/New_York")
_TODAY = date(2026, 8, 5)
_EXPIRY = date(2026, 8, 21)
_SPOT = 70.71  # CGNX close on 2026-08-05

# The live CGNX August band: (strike, open_interest, volume).
# 70.0 is nearest the money and fails; 75.0 is deeper but tradeable.
_CGNX_BAND = {70.0: (886, 35), 75.0: (1246, 185), 65.0: (400, 12)}


def _now_et():
    return datetime(_TODAY.year, _TODAY.month, _TODAY.day, 15, 45, tzinfo=_ET)


def _occ(strike: float) -> str:
    return f"CGNX260821C{int(strike * 1000):08d}"


def _contract(strike: float) -> dict:
    return {
        "symbol": _occ(strike),
        "root_symbol": "CGNX",
        "expiration_date": _EXPIRY.isoformat(),
        "strike_price": str(strike),
        "multiplier": "100",
        "size": "100",
        "tradable": True,
        "type": "call",
    }


@pytest.fixture
def band_liquidity(monkeypatch):
    """Per-strike liquidity, the way Schwab's chain actually varies across a band."""
    table: dict[float, tuple[int, int] | None] = dict(_CGNX_BAND)

    def _fake(_underlying, *, expiry, strike, option_type="C"):  # noqa: ARG001
        hit = table.get(round(float(strike), 4))
        if hit is None:
            return None
        return ContractLiquidity(open_interest=hit[0], volume=hit[1], source="test")

    monkeypatch.setattr(live_ranked_options, "contract_liquidity", _fake)
    return table


class _BandClient:
    """Quotes every strike in the band two-sided unless told otherwise."""

    def __init__(self, *, strikes=(65.0, 70.0, 75.0), no_quote=(), deltas=None):
        self.strikes = list(strikes)
        self.no_quote = set(no_quote)
        self.deltas = dict(deltas or {})
        self.quote_calls: list[str] = []

    def get_option_contracts(self, **_kwargs):
        return {"option_contracts": [_contract(s) for s in self.strikes]}

    def get_option_snapshots(self, *_args, **_kwargs):
        out = {}
        for s in self.strikes:
            greeks = {"delta": self.deltas.get(s, 0.45)}
            if s in self.no_quote:
                out[_occ(s)] = {"greeks": greeks}
                continue
            out[_occ(s)] = {"latestQuote": {"bp": 2.10, "ap": 2.20}, "greeks": greeks}
        return out

    def get_option_quotes(self, symbols, **_kwargs):
        self.quote_calls.append(symbols)
        return {"quotes": {}}


def _select(client, spot=_SPOT):
    return _select_atm_option(
        client, "CGNX", spot, option_type="call", min_dte=1, max_dte=21, now_et=_now_et(),
    )


# --- band ranking --------------------------------------------------------------

def test_liquid_strike_outranks_the_nearest_atm_strike(band_liquidity):
    ranked = _rank_band_by_liquidity(
        "CGNX", _EXPIRY.isoformat(), [_contract(s) for s in (65.0, 70.0, 75.0)],
        _SPOT, option_type="call",
    )
    assert ranked[0]["strike"] == 75.0
    assert ranked[0]["passes"] is True
    # 70.0 is nearer the money but fails the volume floor.
    assert [c["strike"] for c in ranked if c["passes"]] == [75.0]


def test_ranking_reports_the_bands_ceiling_when_nothing_passes(band_liquidity):
    band_liquidity[75.0] = (1246, 20)  # drop the only tradeable strike below the floor
    ranked = _rank_band_by_liquidity(
        "CGNX", _EXPIRY.isoformat(), [_contract(s) for s in (65.0, 70.0, 75.0)],
        _SPOT, option_type="call",
    )
    assert not any(c["passes"] for c in ranked)
    # Deepest OI first, so the reject reason describes the best the band offers.
    assert ranked[0]["open_interest"] == 1246


def test_ranking_degrades_to_nearest_atm_when_the_chain_is_unavailable(monkeypatch):
    monkeypatch.setattr(live_ranked_options, "contract_liquidity",
                        lambda *a, **k: None)
    ranked = _rank_band_by_liquidity(
        "CGNX", _EXPIRY.isoformat(), [_contract(s) for s in (65.0, 70.0, 75.0)],
        _SPOT, option_type="call",
    )
    assert ranked[0]["strike"] == 70.0  # nearest ATM
    assert ranked[0]["liquidity_source"] == "unavailable"


# --- end-to-end selection ------------------------------------------------------

def test_cgnx_now_selects_the_tradeable_strike(band_liquidity):
    """The whole point: this name produced no order on 2026-08-05."""
    order, reason = _select(_BandClient())
    assert reason == "ok"
    assert order["strike"] == 75.0
    assert (order["open_interest"], order["volume"]) == (1246, 185)
    assert order["selection_method"] == "band_liquidity_ranked_non_0dte"


def test_selection_stays_inside_the_ten_percent_band(band_liquidity):
    """A far-OTM strike must not win on open interest alone."""
    band_liquidity[90.0] = (99999, 9999)
    order, _ = _select(_BandClient(strikes=(65.0, 70.0, 75.0, 90.0)))
    # 90 is outside +/-10% of 70.71 and never enters the candidate list.
    assert order["strike"] == 75.0
    assert abs(order["atm_offset_pct"]) <= 0.10


def test_open_interest_cannot_drag_selection_outside_the_delta_band(band_liquidity):
    """The call wall holds the most OI and is the wrong thing to buy.

    Measured on the 2026-08-05 capture, ranking a +/-10% strike band purely by
    open interest put 70% of picks outside [0.35, 0.60] with a 5th-percentile
    delta of 0.05, against 13% for the nearest-ATM rule. Liquidity ranking has
    to happen INSIDE the delta band, not instead of it.
    """
    band_liquidity[75.0] = (99999, 9999)  # a call-wall pile at delta 0.12
    band_liquidity[70.0] = (900, 150)     # tradeable, and the only in-band delta
    order, reason = _select(_BandClient(deltas={65.0: 0.72, 70.0: 0.48, 75.0: 0.12}))
    assert reason == "ok"
    assert order["strike"] == 70.0
    assert 0.35 <= abs(order["delta"]) <= 0.60


def test_the_delta_band_is_not_abandoned_when_nothing_in_it_is_liquid(band_liquidity):
    """Rejecting the name is correct; reaching for the call wall is not."""
    band_liquidity[75.0] = (99999, 9999)  # liquid but delta 0.12
    order, reason = _select(_BandClient(deltas={65.0: 0.72, 70.0: 0.48, 75.0: 0.12}))
    assert order is None
    assert reason == "illiquid_option(oi=886,vol=35)"  # the in-band strike's numbers


def test_missing_greeks_fall_back_to_the_strike_band(band_liquidity):
    """A thin snapshot must not report a tradeable name as untradeable."""
    client = _BandClient()
    client.get_option_snapshots = lambda *a, **k: {  # type: ignore[method-assign]
        _occ(s): {"latestQuote": {"bp": 2.10, "ap": 2.20}} for s in client.strikes
    }
    order, reason = _select(client)
    assert reason == "ok"
    assert order["strike"] == 75.0


def test_a_genuinely_illiquid_band_is_still_rejected(band_liquidity):
    for strike in list(band_liquidity):
        band_liquidity[strike] = (64, 15)  # the real MTZ shape
    order, reason = _select(_BandClient())
    assert order is None
    assert reason == "illiquid_option(oi=64,vol=15)"


def test_unavailable_liquidity_is_still_its_own_reason(monkeypatch):
    monkeypatch.setattr(live_ranked_options, "contract_liquidity", lambda *a, **k: None)
    order, reason = _select(_BandClient())
    assert order is None
    assert reason == "liquidity_unavailable(src=unavailable)"


def test_a_liquid_strike_without_a_quote_falls_through_to_the_next(band_liquidity):
    band_liquidity[70.0] = (5000, 900)  # now the top-ranked strike...
    client = _BandClient(no_quote={70.0})  # ...but it has no two-sided quote
    order, reason = _select(client)
    assert reason == "ok"
    assert order["strike"] == 75.0
    assert client.quote_calls == [_occ(70.0)]  # one retry, then moved on


# --- target screening ----------------------------------------------------------

def test_screening_walks_past_untradeable_top_ranked_names(band_liquidity, monkeypatch):
    """2026-08-05: the ten best tradeable names sat at ranks 6-41.

    A fixed top-K slice froze on names that all failed at order time. Screening
    on the real selector makes the scan walk down the ranking instead.
    """
    rankings = pd.DataFrame({
        "symbol": ["DEAD1", "DEAD2", "DEAD3", "CGNX", "CGNX2"],
        "dealer_direction": ["bullish"] * 5,
    })
    monkeypatch.setattr(live_ranked_options, "_select_atm_option",
                        lambda _c, ticker, _px, **k: (
                            ({"occ": _occ(75.0), "strike": 75.0}, "ok")
                            if str(ticker).startswith("CGNX")
                            else (None, "illiquid_option(oi=64,vol=15)")))
    cache: dict = {}
    top, skipped = _select_optionable_targets(
        _BandClient(), rankings, top_k=2, side_mode="call", min_dte=1, max_dte=21,
        spot_map={s: 70.0 for s in rankings["symbol"]}, selection_cache=cache,
        now_et=_now_et(),
    )
    assert list(top["symbol"]) == ["CGNX", "CGNX2"]
    assert skipped == ["DEAD1", "DEAD2", "DEAD3"]
    # Memoized so routing does not re-select every kept name.
    assert cache[("CGNX", "call")][1] == "ok"


def test_screening_falls_back_to_the_listing_check_without_a_spot(monkeypatch):
    """No snapshot spot means no strike band; it must not empty the target list."""
    rankings = pd.DataFrame({"symbol": ["AAA"], "dealer_direction": ["bullish"]})
    monkeypatch.setattr(live_ranked_options, "_has_tradable_contracts",
                        lambda *a, **k: True)
    monkeypatch.setattr(live_ranked_options, "_select_atm_option",
                        lambda *a, **k: pytest.fail("must not need a band without spot"))
    top, skipped = _select_optionable_targets(
        _BandClient(), rankings, top_k=1, side_mode="call", min_dte=1, max_dte=21,
        spot_map={}, now_et=_now_et(),
    )
    assert list(top["symbol"]) == ["AAA"]
    assert skipped == []
