"""Shadow tracker must walk UNDERLYING bars against an UNDERLYING entry price.

Regression cover for 2026-08-03: option-routed managed positions handed their
option PREMIUM to `evaluate_exit`, which walks the underlying's 4H bars. Every
option position therefore cleared the harvest sleeve's +7% target on its first
bar (WDC implied +1599%, BE +915%, MLTX +831%, CIFR +701%, FCEL +652%, BLZE
+588%), fabricating six "target" exits on positions whose real option legs were
-9% to -21% on the day.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.shadow.two_sleeve_shadow_tracker import (
    HARVEST_POLICY,
    entry_price_is_plausible,
    evaluate_exit,
    resolve_underlying_entry_price,
)


def _bars(closes, start="2026-08-03T14:00:00Z"):
    idx = pd.date_range(start, periods=len(closes), freq="4h", tz="UTC")
    return pd.DataFrame(
        {"close": closes, "high": [c * 1.01 for c in closes], "low": [c * 0.99 for c in closes]},
        index=idx,
    )


# --- resolve_underlying_entry_price -------------------------------------------

def test_option_route_uses_underlying_price_not_premium():
    """The WDC case: premium 31.75 against a 532.49 underlying."""
    pos = {
        "route": "option",
        "occ": "WDC260821C00590000",
        "entry_avg_price": 31.75,  # option premium — must NOT be used
        "order_audit": {"instrument": "option", "underlying_price": 532.49, "premium": 31.97},
    }
    price, source = resolve_underlying_entry_price(pos)
    assert price == pytest.approx(532.49)
    assert source == "order_audit.underlying_price"


def test_option_route_falls_back_to_signal_bar_close():
    pos = {
        "route": "option",
        "entry_avg_price": 2.20,
        "order_audit": {"instrument": "option"},  # no underlying_price
        "signal_audit": {"extra": {"bar_close": 14.565}},
    }
    price, source = resolve_underlying_entry_price(pos)
    assert price == pytest.approx(14.565)
    assert source == "signal_audit.extra.bar_close"


def test_option_route_unresolvable_returns_none():
    pos = {"route": "option", "entry_avg_price": 2.20, "order_audit": {"instrument": "option"}}
    price, source = resolve_underlying_entry_price(pos)
    assert price is None
    assert source == "unresolved"


def test_equity_route_still_prefers_entry_avg_price():
    pos = {
        "route": "equity",
        "entry_avg_price": 12.06,
        "order_audit": {"instrument": "equity", "reference_price": 11.57},
    }
    price, source = resolve_underlying_entry_price(pos)
    assert price == pytest.approx(12.06)
    assert source == "entry_avg_price"


def test_equity_route_falls_back_to_reference_price_when_unpriced():
    """Freshly opened equity positions have no entry_avg_price until the next pass."""
    pos = {"route": "equity", "order_audit": {"instrument": "equity", "reference_price": 11.57}}
    price, source = resolve_underlying_entry_price(pos)
    assert price == pytest.approx(11.57)
    assert source == "order_audit.reference_price"


def test_non_positive_and_non_numeric_prices_are_rejected():
    pos = {
        "route": "equity",
        "entry_avg_price": 0.0,
        "order_audit": {"instrument": "equity", "reference_price": "n/a"},
        "signal_audit": {"extra": {"bar_close": 11.57}},
    }
    price, source = resolve_underlying_entry_price(pos)
    assert price == pytest.approx(11.57)
    assert source == "signal_audit.extra.bar_close"


# --- entry_price_is_plausible --------------------------------------------------

def test_premium_against_share_price_is_implausible():
    bars = _bars([532.49, 536.0])
    entry_ts = bars.index[0]
    assert entry_price_is_plausible(532.49, bars, entry_ts) is True
    assert entry_price_is_plausible(31.75, bars, entry_ts) is False


def test_ordinary_slippage_stays_plausible():
    bars = _bars([100.0, 101.0])
    entry_ts = bars.index[0]
    for px in (100.0, 112.0, 91.0):
        assert entry_price_is_plausible(px, bars, entry_ts) is True


def test_missing_reference_bar_is_not_treated_as_implausible():
    bars = _bars([100.0, 101.0])
    before_any_bar = bars.index[0] - pd.Timedelta("8h")
    assert entry_price_is_plausible(1.0, bars, before_any_bar) is True


# --- end-to-end: the fabricated exit is gone ----------------------------------

def test_premium_entry_fabricates_an_immediate_target_exit():
    """Documents the defect: the old code path books +7% on bar one."""
    bars = _bars([532.49, 539.43 / 1.01])
    result = evaluate_exit(31.75, bars, bars.index[0], HARVEST_POLICY)
    assert result is not None
    assert result["reason"] == "target"
    assert result["bars_held"] == 1
    assert result["underlying_ret"] == pytest.approx(HARVEST_POLICY["target"])


def test_underlying_entry_does_not_exit_on_a_one_percent_bar():
    """Same bars, correct entry price: +1.3% high is nowhere near the +7% target."""
    bars = _bars([532.49, 539.43 / 1.01])
    assert evaluate_exit(532.49, bars, bars.index[0], HARVEST_POLICY) is None


def test_genuine_target_hit_still_exits():
    """REPL 2026-08-03 was a real equity target hit (+8.7% high) and must survive."""
    bars = _bars([11.57, 12.58 / 1.01])
    result = evaluate_exit(11.57, bars, bars.index[0], HARVEST_POLICY)
    assert result is not None
    assert result["reason"] == "target"
