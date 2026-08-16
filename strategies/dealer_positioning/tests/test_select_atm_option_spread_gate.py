"""Dealer Ranker must not buy a contract whose bid/ask spread eats the trade.

The 2026-07-23 IOT incident added open-interest and volume floors, but those do
not bound the spread. On 2026-08-13 every entry cleared the liquidity floors and
still quoted 15-76% of mid:

    SGI260821C00070000   oi=4647 vol=3084   bid 0.33 / ask 0.68   69% of mid
    IONS260821C00055000  oi=1212 vol=621    bid 1.23 / ask 2.73   76% of mid
    ZM260814C00110000    oi=968  vol=995    bid 1.05 / ask 1.91   57% of mid

`_select_atm_option` returns the ask as the entry limit, so those fills booked
an instant 8-38% loss against mid before the underlying moved. The -50% stop is
then measured from the inflated basis, which is how 4 of the 5 dealer exits on
2026-08-13 were stops. The cap is 0.45: joining 19 closed dealer trades to their entry spread shows
only the >0.45 bucket is unambiguously bad (3 trades, 0 wins, -$11,360), while
the <0.18 bucket also lost. See the constant's comment for the full split.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from core.option_liquidity import ContractLiquidity
from strategies.dealer_positioning import live_ranked_options
from strategies.dealer_positioning.live_ranked_options import _select_atm_option

_ET = ZoneInfo("America/New_York")
_TODAY = date(2026, 8, 13)


def _now_et():
    return datetime(_TODAY.year, _TODAY.month, _TODAY.day, 15, 52, tzinfo=_ET)


def _contract(symbol: str, root: str, expiry: date, strike: float) -> dict:
    return {
        "symbol": symbol, "root_symbol": root, "expiration_date": expiry.isoformat(),
        "strike_price": str(strike), "multiplier": "100", "size": "100",
        "tradable": True, "type": "call",
    }


@pytest.fixture
def liquidity(monkeypatch):
    """All candidates clear the OI/volume floors — the spread is the only gate."""
    def _fake(_underlying, *, expiry, strike, option_type="C"):  # noqa: ARG001
        return ContractLiquidity(open_interest=4647, volume=3084, source="test")

    monkeypatch.setattr(live_ranked_options, "contract_liquidity", _fake)


class _Chain:
    """One or more strikes in the band, each with its own quote."""

    def __init__(self, quotes: dict[float, tuple[float, float]]):
        self.expiry = _TODAY + timedelta(days=8)
        self.quotes = quotes
        self.symbols = {
            strike: f"SGI{self.expiry:%y%m%d}C{int(strike * 1000):08d}"
            for strike in quotes
        }

    def get_option_contracts(self, **_kwargs):
        return {"option_contracts": [
            _contract(self.symbols[s], "SGI", self.expiry, s) for s in self.quotes
        ]}

    def get_option_snapshots(self, *_args, **_kwargs):
        return {
            self.symbols[s]: {"latestQuote": {"bp": bid, "ap": ask},
                              "greeks": {"delta": 0.45}}
            for s, (bid, ask) in self.quotes.items()
        }


def _select(client):
    return _select_atm_option(
        client, "SGI", 70.0, option_type="call", min_dte=1, max_dte=21, now_et=_now_et(),
    )


def test_wide_spread_contract_is_rejected(liquidity):
    """The real SGI quote: 0.33/0.68, 69% of mid, deep OI and volume."""
    order, reason = _select(_Chain({70.0: (0.33, 0.68)}))
    assert order is None
    assert reason.startswith("entry_spread_too_wide")
    assert "best=0.69" in reason and "max=0.45" in reason


def test_tight_spread_contract_is_still_selected(liquidity):
    order, reason = _select(_Chain({70.0: (1.90, 2.10)}))
    assert reason == "ok"
    assert order is not None
    assert order["spread"] < 0.18
    assert order["limit"] == 2.10          # entry limit is still the ask


def test_mid_bucket_spread_is_deliberately_kept(liquidity):
    """0.30-0.45 was NOT a losing bucket (+$292, 4 of 6 wins), so it stays tradeable.

    Guards against quietly tightening the cap to the swing sleeve's 0.18 on
    consistency grounds — that bucket lost money here, so consistency would be
    the wrong reason to block these.
    """
    order, reason = _select(_Chain({70.0: (0.85, 1.15)}))   # 30% of mid
    assert reason == "ok"
    assert order is not None
    assert 0.18 < order["spread"] < 0.45


def test_gate_falls_through_to_a_tighter_strike_in_the_band(liquidity):
    """`ranked` is liquid-first, so a wide strike must not abort the whole name."""
    order, reason = _select(_Chain({70.0: (0.33, 0.68), 72.5: (1.95, 2.05)}))
    assert reason == "ok"
    assert order is not None
    assert order["strike"] == 72.5
    assert order["spread"] < 0.18


def test_no_two_sided_quote_is_distinct_from_too_wide(liquidity):
    """A data gap and a rejected-but-quoted contract are different failures."""
    order, reason = _select(_Chain({70.0: (0.0, 0.0)}))
    assert order is None
    assert reason == "no_two_sided_quote"


def test_cap_is_configurable(liquidity, monkeypatch):
    monkeypatch.setattr(live_ranked_options, "_MAX_ENTRY_SPREAD_PCT_MID", 0.80)
    order, reason = _select(_Chain({70.0: (0.33, 0.68)}))
    assert reason == "ok"
    assert order is not None
