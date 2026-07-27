"""Tests for research/options_lab/strategies.py.

No network calls: every chain used here is synthetic, built by pricing
contracts with `pricing.bsm_price`/`bsm_greeks` from a chosen "true" IV so
delta-based selection has a real, reliable, internally-consistent surface to
work against. Covers:
  - textbook payoff-at-expiry shapes (built directly from Leg/Structure, no
    chain needed) for every strategy family, verified at multiple spot
    points below/between/above strikes,
  - put-call parity (long call + short put == synthetic long stock),
  - closed-form checks for vertical debit/credit spreads (max_loss,
    max_gain, breakeven, buying-power),
  - undefined-risk structures reporting max_loss as None (not a fabricated
    finite number),
  - chain-based selection returning `None` with the correct machine-
    readable reason code when the chain lacks strikes/expiries/liquidity,
  - both sizing hooks (matched notional, matched max-loss),
  - the no-lookahead guard actually raising.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from research.options_lab import strategies as S
from research.options_lab.pricing import bsm_greeks, bsm_price

ASOF = "2026-07-25"
SPOT = 100.0
R = 0.04
Q = 0.0


# --------------------------------------------------------------------------
# Manual Leg/Structure builders (textbook payoff-shape tests don't need a
# chain -- they exercise the payoff/greeks/BP math directly).
# --------------------------------------------------------------------------


def _opt_leg(right, strike, quantity, entry_price, dte=30, iv=0.30):
    expiry = (pd.Timestamp(ASOF) + pd.Timedelta(days=dte)).strftime("%Y-%m-%d")
    return S.Leg(osi_symbol=f"TEST{right}{strike}", right=right, strike=float(strike),
                 expiry=expiry, quantity=quantity, entry_price=entry_price, multiplier=100, iv=iv)


def _share_leg(quantity, entry_price=SPOT):
    return S.Leg(osi_symbol="TEST_SHARES", right="S", strike=None, expiry=None,
                 quantity=quantity, entry_price=entry_price, multiplier=1)


def _structure(name, legs, spot=SPOT, asof=ASOF, r=R, q=Q):
    return S.Structure(name=name, legs=legs, entry_spot=spot, entry_asof=asof, r=r, q=q)


# --------------------------------------------------------------------------
# Synthetic chain builder for selection-function tests
# --------------------------------------------------------------------------


def _chain_row(right, strike, dte, *, asof=ASOF, spot=SPOT, r=R, q=Q, iv=0.30,
                reliable=True, oi=1000, vol=500, tc=50, quote_asof=None) -> dict:
    asof_ts = pd.Timestamp(asof)
    expiry_ts = asof_ts + pd.Timedelta(days=dte)
    expiry = expiry_ts.strftime("%Y-%m-%d")
    T = dte / 365.25
    if T > 0:
        price = float(bsm_price(spot, strike, T, r, q, iv, right))
        vega = float(bsm_greeks(spot, strike, T, r, q, iv, right).vega)
    else:
        price = max(spot - strike, 0.0) if right == "C" else max(strike - spot, 0.0)
        vega = 0.0
    return {
        "osi_symbol": f"TEST{expiry.replace('-', '')[2:]}{right}{int(round(strike * 1000)):08d}",
        "ticker": "TEST",
        "expiry": expiry,
        "strike": float(strike),
        "right": right,
        "price": price,
        "iv": iv if reliable else None,
        "vega": vega if reliable else 0.001,
        "quote_asof": quote_asof or asof,
        "open_interest": oi,
        "bar_volume": vol,
        "trade_count": tc,
    }


def _chain(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=S.CHAIN_COLUMNS)


def _standard_chain(dte=14, strikes=(90, 95, 100, 105, 110, 115, 120), **kwargs) -> pd.DataFrame:
    rows = []
    for k in strikes:
        rows.append(_chain_row("C", k, dte, **kwargs))
        rows.append(_chain_row("P", k, dte, **kwargs))
    return _chain(rows)


# ==========================================================================
# Textbook payoff shapes (direct Leg/Structure construction)
# ==========================================================================


def test_long_call_payoff_shape_and_bounds():
    st = _structure("long_call", (_opt_leg("C", 100, 1, 3.0),))
    pl = S.payoff_at_expiry(st, [80, 100, 120, 150]) - st.entry_cost
    assert pl[0] == pytest.approx(-300.0)   # below strike: lose full premium
    assert pl[1] == pytest.approx(-300.0)   # at strike: still lose full premium
    assert pl[2] == pytest.approx(1700.0)   # 100*(120-100) - 300
    assert pl[3] == pytest.approx(4700.0)
    assert st.max_loss == pytest.approx(300.0)
    assert st.max_gain is None              # unbounded upside
    assert st.breakevens == (103.0,)
    assert st.entry_cost == pytest.approx(300.0)
    assert st.buying_power_required == pytest.approx(300.0)  # defined-risk debit rule


def test_long_put_payoff_shape_and_bounds():
    st = _structure("long_put", (_opt_leg("P", 100, 1, 3.0),))
    pl = S.payoff_at_expiry(st, [50, 100, 130]) - st.entry_cost
    assert pl[0] == pytest.approx(4700.0)   # 100*(100-50) - 300
    assert pl[1] == pytest.approx(-300.0)
    assert pl[2] == pytest.approx(-300.0)
    # A long put's downside is bounded too: spot floors at 0, so BOTH sides
    # are finite (unlike a long call, whose upside is unbounded).
    assert st.max_loss == pytest.approx(300.0)
    assert st.max_gain == pytest.approx(9700.0)  # 100*100 - 300
    assert st.breakevens == (97.0,)


def test_long_shares_and_short_shares_payoff():
    long_st = _structure("long_shares", (_share_leg(50, entry_price=SPOT),))
    pl = S.payoff_at_expiry(long_st, [0, 100, 150]) - long_st.entry_cost
    assert pl[0] == pytest.approx(-5000.0)
    assert pl[1] == pytest.approx(0.0)
    assert pl[2] == pytest.approx(2500.0)
    assert long_st.max_loss == pytest.approx(5000.0)   # bounded: spot floors at 0
    assert long_st.max_gain is None                    # unbounded upside

    short_st = _structure("short_shares", (_share_leg(-50, entry_price=SPOT),))
    assert short_st.max_gain == pytest.approx(5000.0)  # bounded: spot floors at 0
    assert short_st.max_loss is None                   # unbounded loss to the upside


def test_vertical_debit_spread_closed_form():
    long_leg = _opt_leg("C", 100, 1, 5.0)
    short_leg = _opt_leg("C", 105, -1, 2.0)
    st = _structure("vertical_debit_spread", (long_leg, short_leg))
    assert st.entry_cost == pytest.approx(300.0)          # net debit
    assert st.max_loss == pytest.approx(300.0)             # == net debit
    assert st.max_gain == pytest.approx(200.0)              # width - debit = 500-300
    assert st.breakevens == (103.0,)
    assert st.buying_power_required == pytest.approx(300.0)  # defined-risk debit rule
    assert st.is_defined_risk is True
    assert st.has_assignment_risk is True  # short leg present


def test_vertical_credit_spread_closed_form():
    short_leg = _opt_leg("P", 100, -1, 5.0)
    long_leg = _opt_leg("P", 95, 1, 2.0)
    st = _structure("vertical_credit_spread", (short_leg, long_leg))
    assert st.entry_cost == pytest.approx(-300.0)           # net credit
    assert st.max_loss == pytest.approx(200.0)               # width - credit = 500-300
    assert st.max_gain == pytest.approx(300.0)                # == credit received
    assert st.breakevens == (97.0,)
    # Reg-T vertical credit spread BP == width*mult - credit
    assert st.buying_power_required == pytest.approx(200.0)


def test_put_call_parity_synthetic_long_stock():
    call = _opt_leg("C", 100, 1, 4.0)
    put = _opt_leg("P", 100, -1, 4.0)
    st = _structure("synthetic_long", (call, put))
    grid = np.array([50.0, 80.0, 100.0, 120.0, 150.0])
    payoff = S.payoff_at_expiry(st, grid)
    # long call + short put at the same strike == 100 shares of synthetic
    # long stock: payoff(S) = 100*(S - 100) identically.
    np.testing.assert_allclose(payoff, 100.0 * (grid - 100.0))
    assert st.entry_cost == pytest.approx(0.0)  # premiums cancel


def test_long_straddle_shape_and_bounds():
    call = _opt_leg("C", 100, 1, 4.0)
    put = _opt_leg("P", 100, 1, 4.0)
    st = _structure("long_straddle", (call, put))
    assert st.entry_cost == pytest.approx(800.0)
    pl = S.payoff_at_expiry(st, [0, 100, 200]) - st.entry_cost
    assert pl[0] == pytest.approx(9200.0)
    assert pl[1] == pytest.approx(-800.0)   # max loss, at the money at expiry
    assert st.max_loss == pytest.approx(800.0)   # == debit paid
    assert st.max_gain is None                    # unbounded via the call
    assert st.breakevens == (92.0, 108.0)
    assert st.buying_power_required == pytest.approx(800.0)


def test_short_strangle_is_undefined_risk():
    call = _opt_leg("C", 105, -1, 2.0)
    put = _opt_leg("P", 95, -1, 2.0)
    st = _structure("short_strangle", (call, put))
    assert st.entry_cost == pytest.approx(-400.0)
    assert st.max_gain == pytest.approx(400.0)   # capped at the credit received
    assert st.max_loss is None                   # genuinely unbounded, not a big number
    assert not isinstance(st.max_loss, float)     # explicitly NOT a finite float
    assert st.is_defined_risk is False
    assert st.has_assignment_risk is True


def test_broken_wing_butterfly_asymmetric_bounds():
    near = _opt_leg("C", 95, 1, 8.0)
    body = _opt_leg("C", 100, -2, 4.0)
    far = _opt_leg("C", 110, 1, 1.0)
    st = _structure("broken_wing_butterfly", (near, body, far))
    assert st.entry_cost == pytest.approx(100.0)  # (8 - 2*4 + 1) * 100
    pl = S.payoff_at_expiry(st, [0, 95, 100, 110, 200]) - st.entry_cost
    np.testing.assert_allclose(pl, [-100.0, -100.0, 400.0, -600.0, -600.0])
    assert st.max_gain == pytest.approx(400.0)   # at the body strike
    assert st.max_loss == pytest.approx(600.0)   # the wider (broken) wing side
    assert st.breakevens == (96.0, 104.0)
    assert st.is_defined_risk is True  # equal long/short contract counts -> bounded both sides


def test_covered_call_shape_and_bp():
    shares = _share_leg(100, entry_price=50.0)
    call = _opt_leg("C", 55, -1, 2.0)
    st = _structure("covered_call", (shares, call), spot=50.0)
    assert st.entry_cost == pytest.approx(4800.0)  # 5000 paid - 200 premium
    pl = S.payoff_at_expiry(st, [0, 55, 100]) - st.entry_cost
    assert pl[0] == pytest.approx(-4800.0)   # shares -> 0, max loss
    assert pl[1] == pytest.approx(700.0)      # capped gain: (55-50)*100 + 200
    assert pl[2] == pytest.approx(700.0)      # flat beyond the short strike
    assert st.max_loss == pytest.approx(4800.0)
    assert st.max_gain == pytest.approx(700.0)
    # Rule 1 catches this: defined-risk, net debit, single expiry.
    assert st.buying_power_required == pytest.approx(4800.0)


def test_cash_secured_put_shape_and_bp():
    put = _opt_leg("P", 45, -1, 2.0)
    st = _structure("cash_secured_put", (put,))
    assert st.entry_cost == pytest.approx(-200.0)
    pl = S.payoff_at_expiry(st, [0, 45, 100]) - st.entry_cost
    assert pl[0] == pytest.approx(-4300.0)  # assigned at 0: lose strike value net of premium
    assert pl[1] == pytest.approx(200.0)
    assert pl[2] == pytest.approx(200.0)
    assert st.max_loss == pytest.approx(4300.0)
    assert st.max_gain == pytest.approx(200.0)
    # Rule 2: cash-secured put BP = strike * multiplier
    assert st.buying_power_required == pytest.approx(4500.0)


def test_naked_short_call_bp_and_unbounded_loss():
    call = _opt_leg("C", 105, -1, 2.0)
    st = _structure("naked_short_call", (call,), spot=100.0)
    assert st.max_loss is None
    assert st.max_gain == pytest.approx(200.0)
    # Reg T: max(20%*S - OTM + prem, 10%*K + prem) * 100
    otm = max(105.0 - 100.0, 0.0)
    rule_a = 0.20 * 100.0 * 100 - otm * 100 + 2.0 * 100
    rule_b = 0.10 * 105.0 * 100 + 2.0 * 100
    assert st.buying_power_required == pytest.approx(max(rule_a, rule_b))


# ==========================================================================
# Structure/Leg validation (fail-fast on malformed input)
# ==========================================================================


def test_leg_rejects_zero_quantity():
    with pytest.raises(ValueError):
        _opt_leg("C", 100, 0, 3.0)


def test_leg_rejects_bad_right():
    with pytest.raises(ValueError):
        S.Leg(osi_symbol="X", right="X", strike=100.0, expiry="2026-08-21", quantity=1, entry_price=1.0)


def test_leg_rejects_wrong_multiplier():
    with pytest.raises(ValueError):
        S.Leg(osi_symbol="X", right="C", strike=100.0, expiry="2026-08-21", quantity=1,
              entry_price=1.0, multiplier=1)


def test_structure_rejects_empty_legs():
    with pytest.raises(ValueError):
        S.Structure(name="empty", legs=(), entry_spot=100.0, entry_asof=ASOF, r=R, q=Q)


def test_structure_rejects_expired_leg():
    expired_leg = _opt_leg("C", 100, 1, 3.0, dte=-5)
    with pytest.raises(ValueError):
        _structure("bad", (expired_leg,))


def test_payoff_at_expiry_rejects_mixed_expiries():
    near = _opt_leg("C", 100, -1, 3.0, dte=14)
    far = _opt_leg("C", 100, 1, 6.0, dte=60)
    st = _structure("calendar", (near, far))
    with pytest.raises(ValueError):
        S.payoff_at_expiry(st, [100.0])


def test_calendar_spread_max_loss_is_debit_approximation_max_gain_none():
    near = _opt_leg("C", 100, -1, 3.0, dte=14)
    far = _opt_leg("C", 100, 1, 6.0, dte=60)
    st = _structure("calendar_spread", (near, far))
    assert st.entry_cost == pytest.approx(300.0)
    assert st.max_loss == pytest.approx(300.0)  # documented approximation, see Structure._pl_bounds_multi_expiry
    assert st.max_gain is None
    assert st.breakevens == ()


# ==========================================================================
# Chain-based selection: success paths
# ==========================================================================


def test_long_call_selects_target_delta():
    chain = _standard_chain(dte=14)
    sel = S.long_call("TEST", ASOF, SPOT, "long", chain, dte_min=8, dte_max=21, target_delta=0.50)
    assert sel.structure is not None
    assert sel.reason is None
    assert sel.selection_method == "delta"
    assert sel.structure.legs[0].right == "C"
    assert sel.structure.legs[0].strike == pytest.approx(100.0)  # ~0.50 delta is ATM


def test_deep_itm_call_selects_high_delta_strike():
    chain = _standard_chain(dte=30, strikes=(70, 75, 80, 85, 90, 95, 100, 105, 110))
    sel = S.deep_itm_call("TEST", ASOF, SPOT, "long", chain, dte_min=22, dte_max=45, target_delta=0.80)
    assert sel.structure is not None
    leg = sel.structure.legs[0]
    assert leg.strike < SPOT  # deep ITM call: strike below spot
    delta = bsm_greeks(SPOT, leg.strike, 30 / 365.25, R, Q, leg.iv, "C").delta
    assert delta == pytest.approx(0.80, abs=0.08)


def test_vertical_debit_spread_selection_end_to_end():
    chain = _standard_chain(dte=14)
    sel = S.vertical_debit_spread(
        "TEST", ASOF, SPOT, "long", chain, dte_min=8, dte_max=21, long_delta=0.60, width=5.0,
    )
    assert sel.structure is not None
    assert sel.structure.name == "vertical_debit_spread"
    assert len(sel.structure.legs) == 2
    long_leg = next(l for l in sel.structure.legs if l.quantity > 0)
    short_leg = next(l for l in sel.structure.legs if l.quantity < 0)
    assert short_leg.strike - long_leg.strike == pytest.approx(5.0)
    assert sel.structure.entry_cost > 0  # debit


def test_straddle_selects_matching_pair_at_nearest_strike():
    chain = _standard_chain(dte=14)
    sel = S.straddle("TEST", ASOF, SPOT, "long", chain, dte_min=8, dte_max=21)
    assert sel.structure is not None
    strikes = {leg.strike for leg in sel.structure.legs}
    assert strikes == {100.0}


def test_calendar_spread_selection_matches_strike_across_expiries():
    near = [_chain_row("C", k, 14) for k in (95, 100, 105)]
    far = [_chain_row("C", k, 60) for k in (95, 100, 105)]
    chain = _chain(near + far)
    sel = S.calendar_spread(
        "TEST", ASOF, SPOT, "long", chain,
        near_dte_min=8, near_dte_max=21, far_dte_min=46, far_dte_max=90, target_delta=0.50,
    )
    assert sel.structure is not None
    strikes = {leg.strike for leg in sel.structure.legs}
    assert len(strikes) == 1  # same strike both expiries
    near_leg = next(l for l in sel.structure.legs if l.quantity < 0)
    far_leg = next(l for l in sel.structure.legs if l.quantity > 0)
    assert near_leg.expiry < far_leg.expiry


# ==========================================================================
# Chain-based selection: correct None + reason on failure
# ==========================================================================


def test_no_expiry_in_bucket():
    chain = _standard_chain(dte=14)
    sel = S.long_call("TEST", ASOF, SPOT, "long", chain, dte_min=46, dte_max=90)
    assert sel.structure is None
    assert sel.reason == S.REASON_NO_EXPIRY_IN_BUCKET


def test_no_strike_near_delta_when_only_wrong_side_present():
    rows = [_chain_row("P", k, 14) for k in (90, 95, 100, 105, 110)]
    chain = _chain(rows)
    sel = S.long_call("TEST", ASOF, SPOT, "long", chain, dte_min=8, dte_max=21)
    assert sel.structure is None
    assert sel.reason == S.REASON_NO_STRIKE_NEAR_DELTA


def test_leg_illiquid_reason():
    chain = _standard_chain(dte=14, oi=10, vol=5)  # below DEFAULT_TIER floors
    sel = S.long_call("TEST", ASOF, SPOT, "long", chain, dte_min=8, dte_max=21, target_delta=0.50)
    assert sel.structure is None
    assert sel.reason == S.REASON_LEG_ILLIQUID


def test_vertical_debit_spread_no_strike_at_width():
    # Only two strikes 5 apart from spot exist; ask for a much wider spread
    # than the chain can offer.
    rows = [_chain_row("C", k, 14) for k in (95, 100)]
    chain = _chain(rows)
    sel = S.vertical_debit_spread(
        "TEST", ASOF, SPOT, "long", chain, dte_min=8, dte_max=21, long_delta=0.60, width=500.0,
    )
    assert sel.structure is None
    assert sel.reason == S.REASON_NO_STRIKE_AT_WIDTH


def test_straddle_no_matching_pair():
    # Calls exist at 100 but there is no put at all -> can't pair.
    rows = [_chain_row("C", k, 14) for k in (95, 100, 105)]
    chain = _chain(rows)
    sel = S.straddle("TEST", ASOF, SPOT, "long", chain, dte_min=8, dte_max=21)
    assert sel.structure is None
    assert sel.reason == S.REASON_NO_MATCHING_PAIR


def test_calendar_spread_no_matching_pair_across_expiries():
    near = [_chain_row("C", k, 14) for k in (95, 100, 105)]
    far = [_chain_row("C", k, 60) for k in (70, 75, 80)]  # disjoint strikes
    chain = _chain(near + far)
    sel = S.calendar_spread(
        "TEST", ASOF, SPOT, "long", chain,
        near_dte_min=8, near_dte_max=21, far_dte_min=46, far_dte_max=90, target_delta=0.50,
    )
    assert sel.structure is None
    assert sel.reason == S.REASON_NO_MATCHING_PAIR


def test_unreliable_iv_falls_back_to_moneyness_and_marks_it():
    rows = [_chain_row("C", k, 14, reliable=False) for k in (90, 95, 100, 105, 110)]
    chain = _chain(rows)
    sel = S.long_call(
        "TEST", ASOF, SPOT, "long", chain, dte_min=8, dte_max=21,
        target_delta=0.50, moneyness_offset=0.0,
    )
    assert sel.structure is not None
    assert sel.selection_method == "moneyness_fallback"


def test_unreliable_iv_with_no_moneyness_fallback_fails():
    rows = [_chain_row("C", k, 14, reliable=False) for k in (90, 95, 100, 105, 110)]
    chain = _chain(rows)
    sel = S.long_call(
        "TEST", ASOF, SPOT, "long", chain, dte_min=8, dte_max=21,
        target_delta=0.50, moneyness_offset=None,
    )
    assert sel.structure is None
    assert sel.reason == "unreliable_iv"


# ==========================================================================
# Sizing hooks
# ==========================================================================


def test_resize_to_notional_matches_target():
    shares_sel = S.long_shares("TEST", ASOF, SPOT, "long", target_notional=5000.0)
    st = shares_sel.structure
    resized = S.resize_to_notional(st, target_notional=1234.0)
    notional = abs(resized.entry_greeks.delta) * resized.entry_spot
    assert notional == pytest.approx(1234.0)


def test_resize_to_notional_for_option_structure():
    chain = _standard_chain(dte=14)
    sel = S.long_call("TEST", ASOF, SPOT, "long", chain, dte_min=8, dte_max=21, target_delta=0.50)
    resized = S.resize_to_notional(sel.structure, target_notional=2000.0)
    notional = abs(resized.entry_greeks.delta) * resized.entry_spot
    assert notional == pytest.approx(2000.0)


def test_resize_to_max_loss_matches_target():
    chain = _standard_chain(dte=14)
    sel = S.vertical_debit_spread(
        "TEST", ASOF, SPOT, "long", chain, dte_min=8, dte_max=21, long_delta=0.60, width=5.0,
    )
    resized = S.resize_to_max_loss(sel.structure, target_max_loss=1000.0)
    assert resized.max_loss == pytest.approx(1000.0)


def test_resize_to_max_loss_raises_for_undefined_risk():
    call = _opt_leg("C", 105, -1, 2.0)
    st = _structure("naked_short_call", (call,))
    with pytest.raises(ValueError):
        S.resize_to_max_loss(st, target_max_loss=1000.0)


def test_resize_to_notional_raises_for_delta_neutral_structure():
    call = _opt_leg("C", 100, 1, 4.0)
    put = _opt_leg("P", 100, -1, 4.0)  # net delta ~ 0 pairing offsets differently, force exact 0 manually
    st = _structure("neutral", (call, put))
    # Synthetic long stock (call+short put) has a real nonzero delta, so
    # instead directly test the zero-delta guard by monkeypatching entry_greeks
    # via a structure whose net delta is exactly zero: two offsetting share legs.
    zero_delta_st = _structure("flat", (_share_leg(50), _share_leg(-50)))
    with pytest.raises(ValueError):
        S.resize_to_notional(zero_delta_st, target_notional=1000.0)


# ==========================================================================
# No-lookahead guard
# ==========================================================================


def test_lookahead_guard_raises():
    chain = _standard_chain(dte=14, quote_asof="2026-07-30")  # observed AFTER asof
    with pytest.raises(ValueError):
        S.long_call("TEST", ASOF, SPOT, "long", chain, dte_min=8, dte_max=21)


def test_selection_result_requires_exactly_one_of_structure_or_reason():
    with pytest.raises(ValueError):
        S.Selection(structure=None, reason=None)
    call = _opt_leg("C", 100, 1, 3.0)
    st = _structure("x", (call,))
    with pytest.raises(ValueError):
        S.Selection(structure=st, reason="oops")
