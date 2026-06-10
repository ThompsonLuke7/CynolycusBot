"""
Synthetic option pricing for backtest (NOT for live).

Black-Scholes-ish call/put pricer. Used by the backtester to estimate
P&L on options when historical option chains aren't available. Assumptions
documented and intentionally simple — this is for relative ranking, not
for production execution.
"""
from __future__ import annotations

import math


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(
    *,
    spot: float,
    strike: float,
    t_years: float,
    iv: float,
    r: float = 0.04,
    is_call: bool = True,
) -> tuple[float, float]:
    """
    Returns (price, delta) for a European call/put at current state.
    """
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        intrinsic = max(0.0, spot - strike) if is_call else max(0.0, strike - spot)
        return intrinsic, (1.0 if is_call and spot > strike else 0.0)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
    d2 = d1 - iv * math.sqrt(t_years)
    if is_call:
        price = spot * _norm_cdf(d1) - strike * math.exp(-r * t_years) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
    else:
        price = strike * math.exp(-r * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1.0
    return price, delta


def iv_from_vix(vix: float, *, ticker_beta: float = 1.0) -> float:
    """Crude IV proxy: scale VIX by beta. VIX in pct (e.g. 20 -> 0.20)."""
    if vix <= 0:
        return 0.30
    base = vix / 100.0 if vix > 1.0 else vix
    return float(min(2.0, max(0.10, base * max(0.5, ticker_beta))))
