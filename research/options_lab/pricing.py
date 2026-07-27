"""Black-Scholes-Merton pricing, American-exercise approximation, and an
implied-volatility solver for `research/options_lab`.

Scope (Phase 1 of docs/superpowers/plans/2026-07-25-options-instrument-routing-experiment.md):
- European call/put price + analytic greeks (delta, gamma, theta, vega, rho)
  under Black-Scholes-Merton with a continuous dividend yield `q`.
- Bjerksund & Stensland (2002) American-exercise approximation, for the
  early-exercise-sensitive cases (American puts; dividend-paying calls).
- A robust implied-vol solver (`implied_vol`) that inverts price -> sigma via
  bracketed Brent search with a bisection fallback, and returns `None`
  (never a garbage number) on arbitrage violations or non-convergence.
- `risk_free_rate`: an as-of, no-lookahead lookup into the repo's cached FMP
  Treasury curve.

Dependency-injection contract (per AGENTS.md -- no hidden global state):
this module never fetches option chain data itself and does NOT import
`research/options_lab/chain_cache.py` (owned by a concurrent task). Every
pricing function takes plain numbers/arrays (spot, strike, T, r, q, sigma)
as arguments. The one file this module reads directly is the already
existing, versioned Treasury-rate parquet -- an explicit, narrow read, not a
hidden dependency on another in-flight module.

Units convention (all functions in this module):
- S, K: price in dollars, must be > 0.
- T: time to expiry in YEARS (calendar days / 365.25), must be >= 0.
- r, q: continuously-compounded ANNUAL rates in DECIMAL (0.04, not 4).
- sigma: annualized volatility in DECIMAL (0.30, not 30).
- right: 'C'/'CALL' or 'P'/'PUT' (case-insensitive).

Greek units: raw partial derivatives of price with respect to the stated
variable for one full unit of that variable (e.g. vega = dPrice/dsigma for
sigma increasing by 1.00, i.e. 100 vol points, not 1). Divide by 100 for the
conventional "per 1 vol point" / "per 1% rate" greek, or divide theta by 365
for a "per calendar day" decay figure. `theta` is already signed as
dPrice/d(calendar time) = -dPrice/dT (negative for long premium, as usually
quoted), not the raw dPrice/dT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

ArrayLike = Union[float, np.ndarray]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TREASURY_PATH_DEFAULT = (
    _REPO_ROOT / "signals" / "meta_context" / "data" / "processed" / "fmp_treasury_rates.parquet"
)
# tenor (years) -> column name in the treasury parquet. Rates stored in
# PERCENT units (e.g. 3.87 == 3.87%), converted to decimal on return.
_TENOR_COLUMNS = {0.25: "month3", 2.0: "year2", 10.0: "year10", 30.0: "year30"}


# --------------------------------------------------------------------------
# Shared input handling
# --------------------------------------------------------------------------


def _right_sign_one(value: object) -> float:
    v = str(value).strip().upper()
    if v in ("C", "CALL"):
        return 1.0
    if v in ("P", "PUT"):
        return -1.0
    raise ValueError(f"right must be 'C'/'CALL' or 'P'/'PUT', got {value!r}")


def _right_sign(right: object) -> np.ndarray:
    arr = np.asarray(right, dtype=object)
    if arr.ndim == 0:
        return np.array(_right_sign_one(arr.item()))
    return np.vectorize(_right_sign_one)(arr)


def _broadcast_inputs(S, K, T, r, q, sigma, right):
    S_a = np.asarray(S, dtype=float)
    K_a = np.asarray(K, dtype=float)
    T_a = np.asarray(T, dtype=float)
    r_a = np.asarray(r, dtype=float)
    q_a = np.asarray(q, dtype=float)
    sig_a = np.asarray(sigma, dtype=float)
    phi = _right_sign(right)
    S_a, K_a, T_a, r_a, q_a, sig_a, phi = np.broadcast_arrays(S_a, K_a, T_a, r_a, q_a, sig_a, phi)
    if np.any(~np.isfinite(S_a)) or np.any(S_a <= 0):
        raise ValueError("spot (S) must be finite and > 0")
    if np.any(~np.isfinite(K_a)) or np.any(K_a <= 0):
        raise ValueError("strike (K) must be finite and > 0")
    if np.any(~np.isfinite(T_a)) or np.any(T_a < 0):
        raise ValueError("T (time to expiry, years) must be finite and >= 0")
    if np.any(~np.isfinite(sig_a)) or np.any(sig_a < 0):
        raise ValueError("sigma (volatility) must be finite and >= 0")
    if np.any(~np.isfinite(r_a)) or np.any(~np.isfinite(q_a)):
        raise ValueError("r and q must be finite")
    return (
        S_a.copy(), K_a.copy(), T_a.copy(), r_a.copy(), q_a.copy(), sig_a.copy(), phi.copy(),
    )


def _scalar_if_possible(arr: np.ndarray):
    return float(arr) if arr.ndim == 0 else arr


def _d1_d2(S, K, T, r, q, sigma):
    """d1/d2 assuming T>0 and sigma>0 element-wise (caller must guarantee)."""
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


# --------------------------------------------------------------------------
# European Black-Scholes-Merton price + greeks
# --------------------------------------------------------------------------


def bsm_price(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike,
              sigma: ArrayLike, right: object) -> ArrayLike:
    """European option price under Black-Scholes-Merton with continuous
    dividend yield `q`. Vectorized: any of S/K/T/r/q/sigma/right may be
    numpy-broadcastable arrays.

    T<=0 or sigma<=0 are handled explicitly (no division by zero): price
    collapses to the discounted-forward intrinsic value
    max(0, phi*(S*exp(-qT) - K*exp(-rT))), which equals max(0, phi*(S-K))
    exactly at T=0.
    """
    S_a, K_a, T_a, r_a, q_a, sig_a, phi = _broadcast_inputs(S, K, T, r, q, sigma, right)
    degenerate = (T_a <= 0) | (sig_a <= 0)
    safe_T = np.where(degenerate, 1.0, T_a)
    safe_sig = np.where(degenerate, 1.0, sig_a)
    d1, d2 = _d1_d2(S_a, K_a, safe_T, r_a, q_a, safe_sig)
    disc_q = np.exp(-q_a * T_a)
    disc_r = np.exp(-r_a * T_a)

    bsm_val = phi * (S_a * disc_q * norm.cdf(phi * d1) - K_a * disc_r * norm.cdf(phi * d2))
    degenerate_val = np.maximum(phi * (S_a * disc_q - K_a * disc_r), 0.0)
    price = np.where(degenerate, degenerate_val, bsm_val)
    return _scalar_if_possible(price)


@dataclass(frozen=True)
class Greeks:
    delta: ArrayLike
    gamma: ArrayLike
    theta: ArrayLike
    vega: ArrayLike
    rho: ArrayLike


def bsm_greeks(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike,
               sigma: ArrayLike, right: object) -> Greeks:
    """Analytic BSM greeks (see module docstring for units/sign conventions).

    At T<=0 or sigma<=0 (no time value / no vol), gamma/vega/theta/rho are
    defined as 0 (no convexity or sensitivity left) and delta collapses to
    the expiry step function: +/-1 if in the money, else 0 (the exact-ATM
    boundary is a measure-zero edge case and is reported as 0).
    """
    S_a, K_a, T_a, r_a, q_a, sig_a, phi = _broadcast_inputs(S, K, T, r, q, sigma, right)
    degenerate = (T_a <= 0) | (sig_a <= 0)
    safe_T = np.where(degenerate, 1.0, T_a)
    safe_sig = np.where(degenerate, 1.0, sig_a)
    d1, d2 = _d1_d2(S_a, K_a, safe_T, r_a, q_a, safe_sig)
    disc_q = np.exp(-q_a * T_a)
    disc_r = np.exp(-r_a * T_a)
    pdf_d1 = norm.pdf(d1)

    delta = phi * disc_q * norm.cdf(phi * d1)
    gamma = disc_q * pdf_d1 / (S_a * safe_sig * np.sqrt(safe_T))
    vega = S_a * disc_q * pdf_d1 * np.sqrt(safe_T)
    theta = (
        -S_a * disc_q * pdf_d1 * safe_sig / (2.0 * np.sqrt(safe_T))
        - phi * r_a * K_a * disc_r * norm.cdf(phi * d2)
        + phi * q_a * S_a * disc_q * norm.cdf(phi * d1)
    )
    rho = phi * K_a * T_a * disc_r * norm.cdf(phi * d2)

    itm = (phi * (S_a - K_a)) > 0
    delta = np.where(degenerate, np.where(itm, phi, 0.0), delta)
    gamma = np.where(degenerate, 0.0, gamma)
    vega = np.where(degenerate, 0.0, vega)
    theta = np.where(degenerate, 0.0, theta)
    rho = np.where(degenerate, 0.0, rho)

    return Greeks(
        delta=_scalar_if_possible(delta),
        gamma=_scalar_if_possible(gamma),
        theta=_scalar_if_possible(theta),
        vega=_scalar_if_possible(vega),
        rho=_scalar_if_possible(rho),
    )


# --------------------------------------------------------------------------
# Bjerksund & Stensland (2002) American approximation
# --------------------------------------------------------------------------
#
# Reference: Bjerksund, P., & Stensland, G. (2002). "Closed Form Valuation
# of American Options." Discussion paper 2002/09, Institute of Finance and
# Management Science, Norwegian School of Economics and Business
# Administration (NHH), Bergen.
#
# Formula transcribed per the widely-used reference implementation in Haug,
# E. G., "The Complete Guide to Option Pricing Formulas" (2nd ed.), using
# cost-of-carry b = r - q throughout. The American put is obtained from the
# American call via the model's own S<->K, r<->b symmetry transform,
# avoiding a second independent derivation.


def _cbnd(a: float, b: float, rho: float) -> float:
    """Bivariate standard normal CDF P(X<=a, Y<=b) with correlation rho."""
    rho = min(max(rho, -1.0 + 1e-12), 1.0 - 1e-12)
    cov = [[1.0, rho], [rho, 1.0]]
    return float(multivariate_normal(mean=[0.0, 0.0], cov=cov).cdf([a, b]))


def _bs2002_phi(S, T, gamma, H, I, r, b, v):
    v2 = v * v
    lam = (-r + gamma * b + 0.5 * gamma * (gamma - 1.0) * v2) * T
    d = -(math.log(S / H) + (b + (gamma - 0.5) * v2) * T) / (v * math.sqrt(T))
    kappa = 2.0 * b / v2 + (2.0 * gamma - 1.0)
    return math.exp(lam) * (S ** gamma) * (
        norm.cdf(d) - ((I / S) ** kappa) * norm.cdf(d - 2.0 * math.log(I / S) / (v * math.sqrt(T)))
    )


def _bs2002_psi(S, T2, gamma, H, I2, I1, T1, r, b, v):
    v2 = v * v
    vst1 = v * math.sqrt(T1)
    vst2 = v * math.sqrt(T2)
    drift = b + (gamma - 0.5) * v2

    e1 = (math.log(S / I1) + drift * T1) / vst1
    e2 = (math.log((I2 ** 2) / (S * I1)) + drift * T1) / vst1
    e3 = (math.log(S / I1) - drift * T1) / vst1
    e4 = (math.log((I2 ** 2) / (S * I1)) - drift * T1) / vst1

    f1 = (math.log(S / H) + drift * T2) / vst2
    f2 = (math.log((I2 ** 2) / (S * H)) + drift * T2) / vst2
    f3 = (math.log((I1 ** 2) / (S * H)) + drift * T2) / vst2
    f4 = (math.log((S * I1 ** 2) / (H * I2 ** 2)) + drift * T2) / vst2

    rho = math.sqrt(T1 / T2)
    lam = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * v2
    kappa = 2.0 * b / v2 + (2.0 * gamma - 1.0)

    return math.exp(lam * T2) * (S ** gamma) * (
        _cbnd(-e1, -f1, rho)
        - ((I2 / S) ** kappa) * _cbnd(-e2, -f2, rho)
        - ((I1 / S) ** kappa) * _cbnd(-e3, -f3, -rho)
        + ((I1 / I2) ** kappa) * _cbnd(-e4, -f4, -rho)
    )


def _bs2002_call(S: float, K: float, T: float, r: float, b: float, sigma: float) -> float:
    """American call under cost-of-carry `b` (b = r - q).

    Numerical note: the closed-form flex-boundary terms (beta, I1, I2) blow
    up as sigma -> 0 whenever b < 0 (equivalently q > r), because beta is
    driven by b/sigma^2. That regime is also exactly where the price is
    least sensitive to sigma (near-zero vega -- see `implied_vol`'s
    docstring), so a fallback to the (always-valid, if slightly
    conservative) European-with-carry-b price on numerical failure never
    produces a materially wrong answer for anything downstream that
    actually cares about that regime.
    """
    if T <= 0.0:
        return max(S - K, 0.0)
    if sigma <= 0.0:
        return max(S * math.exp((b - r) * T) - K * math.exp(-r * T), 0.0)
    if b >= r:
        # Never optimal to exercise early -> American call == European call
        # with cost of carry b, i.e. bsm_price with q = r - b.
        return float(bsm_price(S, K, T, r, r - b, sigma, "C"))

    european_floor = max(float(bsm_price(S, K, T, r, r - b, sigma, "C")), S - K, 0.0)
    try:
        v2 = sigma * sigma
        beta = (0.5 - b / v2) + math.sqrt((b / v2 - 0.5) ** 2 + 2.0 * r / v2)
        b_infinity = beta / (beta - 1.0) * K
        b0 = max(K, (r / (r - b)) * K)

        t1 = 0.5 * (math.sqrt(5.0) - 1.0) * T
        t2 = T

        h_t1 = -(b * t1 + 2.0 * sigma * math.sqrt(t1)) * b0 / (b_infinity - b0)
        h_t2 = -(b * t2 + 2.0 * sigma * math.sqrt(t2)) * b0 / (b_infinity - b0)
        I1 = b0 + (b_infinity - b0) * (1.0 - math.exp(h_t1))
        I2 = b0 + (b_infinity - b0) * (1.0 - math.exp(h_t2))

        if S >= I2:
            return S - K

        alpha1 = (I1 - K) * I1 ** (-beta)
        alpha2 = (I2 - K) * I2 ** (-beta)

        value = (
            alpha2 * S ** beta
            - alpha2 * _bs2002_phi(S, t1, beta, I2, I2, r, b, sigma)
            + _bs2002_phi(S, t1, 1.0, I2, I2, r, b, sigma)
            - _bs2002_phi(S, t1, 1.0, I1, I2, r, b, sigma)
            - K * _bs2002_phi(S, t1, 0.0, I2, I2, r, b, sigma)
            + K * _bs2002_phi(S, t1, 0.0, I1, I2, r, b, sigma)
            + alpha1 * _bs2002_phi(S, t1, beta, I1, I2, r, b, sigma)
            - alpha1 * _bs2002_psi(S, t2, beta, I1, I2, I1, t1, r, b, sigma)
            + _bs2002_psi(S, t2, 1.0, I1, I2, I1, t1, r, b, sigma)
            - _bs2002_psi(S, t2, 1.0, K, I2, I1, t1, r, b, sigma)
            - K * _bs2002_psi(S, t2, 0.0, I1, I2, I1, t1, r, b, sigma)
            + K * _bs2002_psi(S, t2, 0.0, K, I2, I1, t1, r, b, sigma)
        )
    except (OverflowError, ZeroDivisionError, ValueError):
        return european_floor
    if not math.isfinite(value):
        return european_floor
    # The closed-form approximation can dip fractionally below the
    # intrinsic/European floor near the flex boundary from numerical noise;
    # clamp rather than report a sub-floor American price.
    return max(value, european_floor)


def bjerksund_stensland_2002(S: float, K: float, T: float, r: float, q: float,
                              sigma: float, right: str) -> float:
    """American option price via the Bjerksund & Stensland (2002) closed-form
    approximation. Scalar only (see note below); loop over arrays at the
    call site if a batch is needed.

    Puts are obtained from the call formula via the model's own symmetry
    transform: P(S,K,T,r,b,sigma) = C(K,S,T,r-b,-b,sigma), b = r-q.

    Not vectorized: the flex-boundary terms (`_bs2002_psi`) evaluate a
    bivariate normal CDF per call via `scipy.stats.multivariate_normal`,
    which has no cheap batched form here; correctness was prioritized over
    vectorization for this piece, per the task's own priority order.
    """
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    q = float(q)
    sigma = float(sigma)
    if not (math.isfinite(S) and S > 0):
        raise ValueError("spot (S) must be finite and > 0")
    if not (math.isfinite(K) and K > 0):
        raise ValueError("strike (K) must be finite and > 0")
    if not (math.isfinite(T) and T >= 0):
        raise ValueError("T must be finite and >= 0")
    if not (math.isfinite(sigma) and sigma >= 0):
        raise ValueError("sigma must be finite and >= 0")
    if not (math.isfinite(r) and math.isfinite(q)):
        raise ValueError("r and q must be finite")

    right_key = str(right).strip().upper()
    is_call = right_key in ("C", "CALL")
    if not is_call and right_key not in ("P", "PUT"):
        raise ValueError(f"right must be 'C'/'CALL' or 'P'/'PUT', got {right!r}")

    b = r - q
    if is_call:
        return _bs2002_call(S, K, T, r, b, sigma)
    return _bs2002_call(K, S, T, r - b, -b, sigma)


# --------------------------------------------------------------------------
# Implied volatility solver
# --------------------------------------------------------------------------


def _no_arbitrage_bounds(S: float, K: float, T: float, r: float, q: float,
                          is_call: bool, american: bool) -> tuple[float, float]:
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if is_call:
        euro_lower = max(0.0, S * disc_q - K * disc_r)
        euro_upper = S * disc_q
        if american:
            return max(euro_lower, S - K, 0.0), S
        return euro_lower, euro_upper
    euro_lower = max(0.0, K * disc_r - S * disc_q)
    euro_upper = K * disc_r
    if american:
        return max(euro_lower, K - S, 0.0), K
    return euro_lower, euro_upper


def _bisect_root(f, lo: float, hi: float, f_lo: float, f_hi: float,
                  xtol: float, max_iter: int = 200) -> Optional[float]:
    a, b = lo, hi
    fa, fb = f_lo, f_hi
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if fm == 0.0 or (b - a) / 2.0 < xtol:
            return m
        if fa * fm < 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return None


def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float,
                 right: str, american: bool = False, lo: float = 1e-4, hi: float = 5.0,
                 max_expansions: int = 8, xtol: float = 1e-8) -> Optional[float]:
    """Invert observed option `price` for Black-Scholes-Merton (or
    Bjerksund-Stensland American, if `american=True`) implied volatility.

    Solver design: uses bracketed Brent search (`scipy.optimize.brentq`)
    over the *price* difference (never a Newton step on vega), with a
    bisection fallback if Brent raises. This is deliberate: Newton-Raphson
    on vega is the classic IV-solver failure mode for deep ITM/OTM
    contracts, where vega collapses toward zero and a vega-based step
    blows up or oscillates. Brent/bisection only ever evaluate the pricing
    function itself, so they stay well-behaved in that regime -- the
    remaining risk there is a numerically FLAT price-vs-sigma curve (tiny
    time value), which shows up as an unreachable bracket, handled below.

    Returns `None` (never 0.0, never an unchecked/garbage number) when:
      * `T <= 0` -- no time value left, IV undefined at/after expiry.
      * `price <= 0`, non-finite, or any of S/K/T/r/q is non-finite.
      * `price` violates the model-free no-arbitrage bounds for this
        right/exercise-style (below intrinsic or above the max-payoff
        bound), within a small floating-point tolerance.
      * the bracket [lo, hi] (expanded up to `max_expansions` times by
        doubling `hi`) never contains a sign change -- i.e. no sigma in
        the search range reproduces `price`.
      * Brent raises AND the bisection fallback also fails to converge.

    Known accuracy limits (see also the tests):
      * Deep ITM/near-expiry contracts trading very close to their
        no-arbitrage floor have near-zero vega; the *price* is still
        solvable (Brent needs no vega), but the recovered sigma has wide
        confidence bounds for a tiny quoted-price error -- a $0.01 bid/ask
        difference can imply several vol points there. This is a property
        of the option, not a solver bug.
      * `american=True` calls `bjerksund_stensland_2002` (an
        approximation, not an exact American price) at every Brent
        iteration; the recovered "IV" therefore inherits that
        approximation's error, which is largest for deep ITM American
        puts with large T. See its own docstring.
    """
    try:
        S_f = float(S); K_f = float(K); T_f = float(T)
        r_f = float(r); q_f = float(q); price_f = float(price)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (S_f, K_f, T_f, r_f, q_f, price_f)):
        return None
    if S_f <= 0 or K_f <= 0 or T_f <= 0 or price_f <= 0:
        return None

    right_key = str(right).strip().upper()
    if right_key not in ("C", "CALL", "P", "PUT"):
        return None
    is_call = right_key in ("C", "CALL")

    lower, upper = _no_arbitrage_bounds(S_f, K_f, T_f, r_f, q_f, is_call, american)
    tol = 1e-9 * max(1.0, S_f)
    if price_f < lower - tol or price_f > upper + tol:
        return None

    def model_price(sigma: float) -> float:
        if american:
            return bjerksund_stensland_2002(S_f, K_f, T_f, r_f, q_f, sigma, right_key)
        return float(bsm_price(S_f, K_f, T_f, r_f, q_f, sigma, right_key))

    def f(sigma: float) -> float:
        return model_price(sigma) - price_f

    lo_b, hi_b = lo, hi
    f_lo, f_hi = f(lo_b), f(hi_b)
    expansions = 0
    while f_lo * f_hi > 0.0 and expansions < max_expansions:
        hi_b *= 2.0
        f_hi = f(hi_b)
        expansions += 1
    if f_lo * f_hi > 0.0:
        return None

    sigma: Optional[float]
    try:
        sigma = brentq(f, lo_b, hi_b, xtol=xtol, maxiter=200)
    except Exception:
        sigma = _bisect_root(f, lo_b, hi_b, f_lo, f_hi, xtol=xtol)

    if sigma is None or not math.isfinite(sigma) or sigma <= 0.0:
        return None
    return float(sigma)


def implied_vol_batch(prices, S, K, T, r, q, rights, american: bool = False,
                       **kwargs) -> np.ndarray:
    """Elementwise `implied_vol` over broadcastable arrays.

    Looped, not solver-vectorized: `implied_vol`'s Brent/bisection search is
    inherently scalar-iterative, and there is no closed-form batch
    inversion for BSM (unlike `bsm_price`/`bsm_greeks`, which ARE
    vectorized). Returns an object array so a per-element solver failure
    is `None`, never silently coerced into NaN or 0.0 by a numeric dtype.
    """
    prices_a, S_a, K_a, T_a, r_a, q_a, rights_a = np.broadcast_arrays(
        np.asarray(prices, dtype=float), np.asarray(S, dtype=float),
        np.asarray(K, dtype=float), np.asarray(T, dtype=float),
        np.asarray(r, dtype=float), np.asarray(q, dtype=float),
        np.asarray(rights, dtype=object),
    )
    out = np.empty(prices_a.shape, dtype=object)
    it = np.nditer(prices_a, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        out[idx] = implied_vol(
            price=prices_a[idx], S=S_a[idx], K=K_a[idx], T=T_a[idx],
            r=r_a[idx], q=q_a[idx], right=rights_a[idx], american=american, **kwargs,
        )
    return out


# --------------------------------------------------------------------------
# Risk-free rate lookup
# --------------------------------------------------------------------------


def load_treasury_curve(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load and validate the cached FMP Treasury curve.

    Schema (inspected from the real file, not assumed): columns
    `date` (datetime64[ns], naive, one row per available business day),
    `month3`, `year2`, `year10`, `year30` (percent units, e.g. 3.87 means
    3.87%; NaN before each tenor's own history begins -- `year10` is the
    only column populated back to the file's earliest date), plus
    `spread_2s10s`, `spread_3m10y`, `inverted` (unused here).
    """
    p = Path(path) if path is not None else _TREASURY_PATH_DEFAULT
    if not p.exists():
        raise FileNotFoundError(f"treasury rate file not found: {p}")
    df = pd.read_parquet(p)
    required = {"date", "month3", "year2", "year10", "year30"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"treasury rate file {p} missing required columns: {sorted(missing)}")
    return df.sort_values("date").reset_index(drop=True)


def risk_free_rate(asof, tenor_years: float, curve: Optional[pd.DataFrame] = None) -> float:
    """Interpolate an annualized, continuously-compounded-equivalent
    risk-free rate (decimal, e.g. 0.0405) for `tenor_years` from the cached
    FMP Treasury curve.

    Time correctness: uses the LAST available curve row AT OR BEFORE
    `asof` (a backward-only as-of lookup that tolerates weekends/holidays
    in the curve's own publication calendar) -- never interpolates across
    calendar time and never fills forward into a date after `asof`. Fails
    loudly (raises `ValueError`) if `asof` is outside
    [curve['date'].min(), curve['date'].max()], or if the selected row's
    needed tenor columns are NaN, rather than silently extrapolating past
    the data's real coverage.

    Tenor interpolation: linear across the four available tenor points
    (0.25y / 2y / 10y / 30y); flat (nearest-endpoint) extrapolation for
    `tenor_years` outside [0.25, 30].
    """
    if curve is None:
        curve = load_treasury_curve()
    if not (math.isfinite(tenor_years) and tenor_years > 0):
        raise ValueError(f"tenor_years must be finite and > 0, got {tenor_years}")

    asof_ts = pd.Timestamp(asof)
    if asof_ts.tzinfo is not None:
        asof_ts = asof_ts.tz_convert("UTC").tz_localize(None)
    asof_ts = asof_ts.normalize()

    min_date, max_date = curve["date"].min(), curve["date"].max()
    if asof_ts < min_date or asof_ts > max_date:
        raise ValueError(
            f"asof={asof_ts.date()} is outside treasury curve coverage "
            f"[{min_date.date()}, {max_date.date()}]"
        )
    eligible = curve[curve["date"] <= asof_ts]
    if eligible.empty:
        raise ValueError(f"no treasury curve row at or before asof={asof_ts.date()}")
    row = eligible.iloc[-1]

    tenors = sorted(_TENOR_COLUMNS)
    rates_pct = [row[_TENOR_COLUMNS[t]] for t in tenors]
    if any(pd.isna(v) for v in rates_pct):
        raise ValueError(
            f"treasury curve row for {row['date'].date()} has missing tenor rate(s): "
            f"{dict(zip(tenors, rates_pct))}"
        )
    rate_pct = float(np.interp(tenor_years, tenors, rates_pct))
    return rate_pct / 100.0
