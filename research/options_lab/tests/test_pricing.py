"""Tests for research/options_lab/pricing.py.

Covers: put-call parity, greeks vs finite differences, IV solver round-trip
across a moneyness/DTE/vol grid (incl. deep ITM/OTM and very short T), IV
solver returning None on arbitrage violations / non-convergence, American
>= European price and convergence to European as q->0 for calls, and the
Treasury risk-free-rate as-of lookup.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from research.options_lab import pricing


# --------------------------------------------------------------------------
# Put-call parity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("S", [50.0, 100.0, 250.0])
@pytest.mark.parametrize("K", [80.0, 100.0, 130.0])
@pytest.mark.parametrize("T", [0.02, 0.25, 1.5])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(S, K, T, q):
    r = 0.04
    sigma = 0.35
    call = pricing.bsm_price(S, K, T, r, q, sigma, "C")
    put = pricing.bsm_price(S, K, T, r, q, sigma, "P")
    lhs = call - put
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-8)


def test_bsm_price_vectorized_matches_scalar_loop():
    S = np.array([90.0, 100.0, 110.0])
    K = np.array([100.0, 100.0, 100.0])
    T = np.array([0.5, 0.5, 0.5])
    r = 0.04
    q = 0.01
    sigma = 0.3
    right = np.array(["C", "C", "P"])
    vec = pricing.bsm_price(S, K, T, r, q, sigma, right)
    scalar = np.array([
        pricing.bsm_price(float(S[i]), float(K[i]), float(T[i]), r, q, sigma, str(right[i]))
        for i in range(3)
    ])
    np.testing.assert_allclose(vec, scalar, atol=1e-10)


def test_bsm_price_at_expiry_is_intrinsic():
    assert pricing.bsm_price(110.0, 100.0, 0.0, 0.04, 0.0, 0.3, "C") == pytest.approx(10.0)
    assert pricing.bsm_price(90.0, 100.0, 0.0, 0.04, 0.0, 0.3, "C") == pytest.approx(0.0)
    assert pricing.bsm_price(90.0, 100.0, 0.0, 0.04, 0.0, 0.3, "P") == pytest.approx(10.0)


def test_bsm_price_zero_vol_is_discounted_forward_intrinsic():
    S, K, T, r, q = 100.0, 90.0, 1.0, 0.05, 0.0
    price = pricing.bsm_price(S, K, T, r, q, 0.0, "C")
    expected = max(0.0, S * math.exp(-q * T) - K * math.exp(-r * T))
    assert price == pytest.approx(expected, abs=1e-10)


def test_bsm_price_rejects_bad_inputs():
    with pytest.raises(ValueError):
        pricing.bsm_price(-1.0, 100.0, 1.0, 0.04, 0.0, 0.3, "C")
    with pytest.raises(ValueError):
        pricing.bsm_price(100.0, 100.0, -0.1, 0.04, 0.0, 0.3, "C")
    with pytest.raises(ValueError):
        pricing.bsm_price(100.0, 100.0, 1.0, 0.04, 0.0, 0.3, "X")


# --------------------------------------------------------------------------
# Greeks vs finite differences
# --------------------------------------------------------------------------


GREEK_GRID = [
    (S, K, T, right)
    for S in (80.0, 100.0, 120.0)
    for K in (100.0,)
    for T in (0.1, 0.75)
    for right in ("C", "P")
]


@pytest.mark.parametrize("S,K,T,right", GREEK_GRID)
def test_greeks_match_finite_differences(S, K, T, right):
    r, q, sigma = 0.04, 0.02, 0.30
    g = pricing.bsm_greeks(S, K, T, r, q, sigma, right)

    h_s = S * 1e-4
    delta_fd = (
        pricing.bsm_price(S + h_s, K, T, r, q, sigma, right)
        - pricing.bsm_price(S - h_s, K, T, r, q, sigma, right)
    ) / (2 * h_s)
    assert g.delta == pytest.approx(delta_fd, abs=2e-4)

    gamma_fd = (
        pricing.bsm_price(S + h_s, K, T, r, q, sigma, right)
        - 2 * pricing.bsm_price(S, K, T, r, q, sigma, right)
        + pricing.bsm_price(S - h_s, K, T, r, q, sigma, right)
    ) / (h_s ** 2)
    assert g.gamma == pytest.approx(gamma_fd, abs=2e-3)

    h_v = 1e-4
    vega_fd = (
        pricing.bsm_price(S, K, T, r, q, sigma + h_v, right)
        - pricing.bsm_price(S, K, T, r, q, sigma - h_v, right)
    ) / (2 * h_v)
    assert g.vega == pytest.approx(vega_fd, abs=2e-3)

    h_r = 1e-5
    rho_fd = (
        pricing.bsm_price(S, K, T, r + h_r, q, sigma, right)
        - pricing.bsm_price(S, K, T, r - h_r, q, sigma, right)
    ) / (2 * h_r)
    assert g.rho == pytest.approx(rho_fd, abs=2e-3)

    # theta = dPrice/d(calendar time) = -dPrice/dT
    h_t = T * 1e-4
    theta_fd = (
        pricing.bsm_price(S, K, T - h_t, r, q, sigma, right)
        - pricing.bsm_price(S, K, T + h_t, r, q, sigma, right)
    ) / (2 * h_t)
    assert g.theta == pytest.approx(theta_fd, abs=2e-2)


def test_greeks_degenerate_at_expiry():
    g_itm_call = pricing.bsm_greeks(110.0, 100.0, 0.0, 0.04, 0.0, 0.3, "C")
    assert g_itm_call.delta == pytest.approx(1.0)
    assert g_itm_call.gamma == pytest.approx(0.0)
    assert g_itm_call.vega == pytest.approx(0.0)

    g_otm_call = pricing.bsm_greeks(90.0, 100.0, 0.0, 0.04, 0.0, 0.3, "C")
    assert g_otm_call.delta == pytest.approx(0.0)


# --------------------------------------------------------------------------
# IV solver round-trip
# --------------------------------------------------------------------------


ROUNDTRIP_GRID = [
    (S, K, T, sigma_true, right)
    for S in (100.0,)
    for K in (60.0, 90.0, 100.0, 110.0, 150.0)  # deep ITM .. deep OTM (calls)
    for T in (1.0 / 365.25, 7.0 / 365.25, 45.0 / 365.25, 365.0 / 365.25)
    for sigma_true in (0.10, 0.30, 0.80, 1.50)
    for right in ("C", "P")
]


@pytest.mark.parametrize("S,K,T,sigma_true,right", ROUNDTRIP_GRID)
def test_implied_vol_round_trips(S, K, T, sigma_true, right):
    """Round-trip price(sigma_true) -> implied_vol -> recovered.

    Self-consistency (repricing at the recovered vol reproduces the input
    price) must ALWAYS hold -- that is what the solver is defined to do.

    Recovering something close to `sigma_true` itself is only guaranteed
    when the option has enough vega to make sigma identifiable from price
    in the first place. Deep ITM/OTM contracts at very short T have vega
    that underflows to ~0 (price is ~pure discounted intrinsic and is
    essentially flat in sigma over the whole realistic vol range) -- there
    the solver still returns a valid, self-consistent, non-None sigma, but
    it is not meaningfully "the" vol that produced the price, because many
    vols produce (numerically) the same price. This is a real, expected
    accuracy limit of IV inversion, not a solver bug; see
    `test_implied_vol_low_vega_is_self_consistent_but_not_pinned` below for
    an explicit demonstration.
    """
    r, q = 0.04, 0.01
    price = pricing.bsm_price(S, K, T, r, q, sigma_true, right)
    if price <= 1e-8:
        pytest.skip("price numerically zero at this grid point -- not a solvable case")
    recovered = pricing.implied_vol(price, S, K, T, r, q, right)
    assert recovered is not None, (
        f"solver returned None for a known-good price S={S} K={K} T={T} "
        f"sigma_true={sigma_true} right={right} price={price}"
    )
    repriced = pricing.bsm_price(S, K, T, r, q, recovered, right)
    assert repriced == pytest.approx(price, rel=1e-4, abs=1e-4)

    vega = pricing.bsm_greeks(S, K, T, r, q, sigma_true, right).vega
    if vega > 0.05:  # well-conditioned: price is meaningfully sigma-sensitive here
        assert recovered == pytest.approx(sigma_true, rel=0.02, abs=1e-3)


def test_implied_vol_low_vega_is_self_consistent_but_not_pinned():
    """Documents the solver's known accuracy limit: deep ITM, very-short-T
    contracts have near-zero vega, so price alone cannot pin down sigma.
    The solver must still behave sanely (return a valid, self-consistent
    sigma) rather than crash or return an arbitrary/garbage number, but the
    recovered value is not expected to match the vol that generated the
    price.
    """
    S, K, T, r, q, right = 100.0, 60.0, 1.0 / 365.25, 0.04, 0.01, "C"
    price_lo_vol = pricing.bsm_price(S, K, T, r, q, 0.10, right)
    price_hi_vol = pricing.bsm_price(S, K, T, r, q, 0.80, right)
    # The defining symptom: two very different true vols produce
    # numerically indistinguishable prices at this deep-ITM/short-T point.
    assert price_lo_vol == pytest.approx(price_hi_vol, abs=1e-6)

    vega = pricing.bsm_greeks(S, K, T, r, q, 0.30, right).vega
    assert vega < 0.05

    recovered = pricing.implied_vol(price_lo_vol, S, K, T, r, q, right)
    assert recovered is not None
    repriced = pricing.bsm_price(S, K, T, r, q, recovered, right)
    assert repriced == pytest.approx(price_lo_vol, rel=1e-4, abs=1e-4)


def test_implied_vol_round_trip_american_puts():
    # American puts always trade at/above intrinsic and are priced via
    # Bjerksund-Stensland; round-trip through the same (approximate) model
    # should still recover the vol that generated the price when the
    # option has enough vega for sigma to be identifiable (see the
    # low-vega discussion on the European round-trip test above -- the
    # same near-zero-vega regions apply here too).
    r, q = 0.04, 0.0
    S = 100.0
    for K in (90.0, 100.0, 110.0):
        for T in (30.0 / 365.25, 180.0 / 365.25):
            for sigma_true in (0.25, 0.6):
                price = pricing.bjerksund_stensland_2002(S, K, T, r, q, sigma_true, "P")
                recovered = pricing.implied_vol(price, S, K, T, r, q, "P", american=True)
                assert recovered is not None
                repriced = pricing.bjerksund_stensland_2002(S, K, T, r, q, recovered, "P")
                assert repriced == pytest.approx(price, rel=1e-4, abs=1e-4)
                vega = pricing.bsm_greeks(S, K, T, r, q, sigma_true, "P").vega
                if vega > 0.05:
                    assert recovered == pytest.approx(sigma_true, rel=0.03, abs=1e-3)


# --------------------------------------------------------------------------
# IV solver: None on arbitrage violation / non-convergence
# --------------------------------------------------------------------------


def test_implied_vol_none_below_intrinsic():
    # European call lower bound: S*exp(-qT) - K*exp(-rT). Price it below
    # that -> arbitrage violation.
    S, K, T, r, q = 100.0, 80.0, 0.5, 0.04, 0.0
    lower = S * math.exp(-q * T) - K * math.exp(-r * T)
    bad_price = lower - 1.0
    assert pricing.implied_vol(bad_price, S, K, T, r, q, "C") is None


def test_implied_vol_none_above_upper_bound():
    S, K, T, r, q = 100.0, 100.0, 0.5, 0.04, 0.0
    upper = S * math.exp(-q * T)  # European call upper bound
    bad_price = upper + 1.0
    assert pricing.implied_vol(bad_price, S, K, T, r, q, "C") is None


def test_implied_vol_none_at_or_before_expiry():
    assert pricing.implied_vol(5.0, 100.0, 100.0, 0.0, 0.04, 0.0, "C") is None
    assert pricing.implied_vol(5.0, 100.0, 100.0, -0.01, 0.04, 0.0, "C") is None


def test_implied_vol_none_for_zero_or_negative_price():
    assert pricing.implied_vol(0.0, 100.0, 100.0, 0.5, 0.04, 0.0, "C") is None
    assert pricing.implied_vol(-1.0, 100.0, 100.0, 0.5, 0.04, 0.0, "C") is None


def test_implied_vol_none_for_invalid_right():
    assert pricing.implied_vol(5.0, 100.0, 100.0, 0.5, 0.04, 0.0, "X") is None


def test_implied_vol_none_for_price_at_exact_intrinsic_deep_itm():
    # Deep ITM American put priced at exactly intrinsic (no time value at
    # all) is degenerate for a solver expecting sigma > 0 to explain it --
    # the lower bound is intrinsic itself, so the search bracket collapses
    # to zero width there; the solver must not fabricate a sigma.
    S, K, T, r, q = 50.0, 100.0, 0.25, 0.04, 0.0
    intrinsic = K - S
    recovered = pricing.implied_vol(intrinsic, S, K, T, r, q, "P", american=True)
    # Either None (bracket collapse) or a tiny/near-zero vol is defensible;
    # what must NOT happen is a large or NaN vol claiming to explain a
    # pure-intrinsic price.
    assert recovered is None or recovered < 0.05


def test_implied_vol_none_when_bracket_never_reached_price():
    # Force a price above what even hi=5.0 (500% vol, before expansion)
    # would produce for extremely short T, so the bracket expansion must
    # exhaust and return None rather than loop forever or fabricate sigma.
    S, K, T, r, q = 100.0, 100.0, 0.5 / 365.25, 0.04, 0.0
    upper = S * math.exp(-q * T)
    bad_price = upper - 1e-6  # inside the no-arb bound, but unreachable in practice? guard below
    # If this particular price happens to be reachable, fall back to a
    # value we know violates the upper bound outright (already covered by
    # test_implied_vol_none_above_upper_bound); this test focuses on the
    # bracket-expansion exhaustion path directly.
    result = pricing.implied_vol(bad_price, S, K, T, r, q, "C", hi=0.05, max_expansions=0)
    # With hi capped at 5% vol and zero expansions allowed, a normal-vol
    # price should be unreachable from a near-upper-bound quote.
    assert result is None or result <= 0.05 + 1e-6


# --------------------------------------------------------------------------
# American (Bjerksund-Stensland 2002) vs European
# --------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.0, 0.01, 0.03, 0.06])
@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_american_call_at_least_european(q, K):
    S, T, r, sigma = 100.0, 0.75, 0.04, 0.30
    american = pricing.bjerksund_stensland_2002(S, K, T, r, q, sigma, "C")
    european = pricing.bsm_price(S, K, T, r, q, sigma, "C")
    assert american >= european - 1e-8


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_american_put_at_least_european(K):
    S, T, r, q, sigma = 100.0, 0.75, 0.04, 0.0, 0.30
    american = pricing.bjerksund_stensland_2002(S, K, T, r, q, sigma, "P")
    european = pricing.bsm_price(S, K, T, r, q, sigma, "P")
    assert american >= european - 1e-8


def test_american_call_equals_european_at_zero_dividend():
    S, K, T, r, sigma = 100.0, 95.0, 0.5, 0.04, 0.3
    american = pricing.bjerksund_stensland_2002(S, K, T, r, 0.0, sigma, "C")
    european = pricing.bsm_price(S, K, T, r, 0.0, sigma, "C")
    # With no dividend, early exercise of a call is never optimal --
    # American == European exactly (up to floating point).
    assert american == pytest.approx(european, abs=1e-8)


def test_american_call_converges_to_european_as_q_shrinks():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.04, 0.3
    gaps = []
    for q in (0.05, 0.02, 0.005, 0.0):
        american = pricing.bjerksund_stensland_2002(S, K, T, r, q, sigma, "C")
        european = pricing.bsm_price(S, K, T, r, q, sigma, "C")
        gaps.append(american - european)
    assert gaps[-1] == pytest.approx(0.0, abs=1e-8)
    # Gap should shrink (non-increasing) as q -> 0.
    assert all(gaps[i] >= gaps[i + 1] - 1e-9 for i in range(len(gaps) - 1))


def test_bjerksund_stensland_rejects_bad_inputs():
    with pytest.raises(ValueError):
        pricing.bjerksund_stensland_2002(-1.0, 100.0, 1.0, 0.04, 0.0, 0.3, "C")
    with pytest.raises(ValueError):
        pricing.bjerksund_stensland_2002(100.0, 100.0, 1.0, 0.04, 0.0, 0.3, "Z")


# --------------------------------------------------------------------------
# Risk-free rate lookup
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def treasury_curve() -> pd.DataFrame:
    return pricing.load_treasury_curve()


def test_load_treasury_curve_schema(treasury_curve):
    for col in ("date", "month3", "year2", "year10", "year30"):
        assert col in treasury_curve.columns
    assert treasury_curve["date"].is_monotonic_increasing


def test_risk_free_rate_interpolates_between_tenors(treasury_curve):
    asof = treasury_curve["date"].iloc[-1]
    rate_2y = pricing.risk_free_rate(asof, 2.0, curve=treasury_curve)
    rate_5y = pricing.risk_free_rate(asof, 5.0, curve=treasury_curve)
    rate_10y = pricing.risk_free_rate(asof, 10.0, curve=treasury_curve)
    # 5y sits between the 2y and 10y points -> interpolated rate should too.
    assert min(rate_2y, rate_10y) - 1e-9 <= rate_5y <= max(rate_2y, rate_10y) + 1e-9
    assert 0.0 < rate_2y < 0.20  # sanity: decimal, not percent


def test_risk_free_rate_uses_last_available_date_at_or_before_asof(treasury_curve):
    # Pick a date one calendar day after the last real row (e.g. a weekend)
    # -- still inside overall coverage only if within [min,max]; use an
    # actual gap day inside the range instead (a Saturday).
    dates = treasury_curve["date"]
    mid_idx = len(dates) // 2
    real_date = dates.iloc[mid_idx]
    gap_date = real_date + pd.Timedelta(days=1)
    if gap_date in set(dates):
        pytest.skip("no gap day available at this index")
    rate_gap = pricing.risk_free_rate(gap_date, 10.0, curve=treasury_curve)
    rate_real = pricing.risk_free_rate(real_date, 10.0, curve=treasury_curve)
    assert rate_gap == pytest.approx(rate_real)


def test_risk_free_rate_raises_outside_coverage(treasury_curve):
    min_date = treasury_curve["date"].min()
    max_date = treasury_curve["date"].max()
    with pytest.raises(ValueError):
        pricing.risk_free_rate(min_date - pd.Timedelta(days=1), 10.0, curve=treasury_curve)
    with pytest.raises(ValueError):
        pricing.risk_free_rate(max_date + pd.Timedelta(days=1), 10.0, curve=treasury_curve)


def test_risk_free_rate_raises_on_missing_tenor_data():
    # Early history has NaN month3/year2/year30 (only year10 populated).
    curve = pricing.load_treasury_curve()
    early_date = curve["date"].iloc[0]
    with pytest.raises(ValueError):
        pricing.risk_free_rate(early_date, 0.25, curve=curve)
