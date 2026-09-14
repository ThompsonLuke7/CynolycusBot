"""Implied-volatility surface features from one Schwab option chain.

Forward collection only: Schwab serves live chains, not history, so these
features exist from the first nightly capture onward and a missed night can
never be backfilled.

The feature set follows the 2015-2026 option-implied predictability study
(arXiv 2608.26115) and keeps the signals that survived its 2023-2026 regime:

  cw_iv_spread_30      Cremers-Weinbaum: call IV minus put IV at matched
                       strikes within 5% of spot, open-interest weighted
                       (|t| > 4 in every regime). See the note below.
  rn_skew_30           Bakshi-Kapadia-Madan risk-neutral skewness of the log
                       return (the most robust signal in the study)
  iv_term_slope_30_60  ATM IV at 60d minus ATM IV at 30d

plus the Xing-Zhang-Zhao smirk, which DECAYED to insignificance in 2023-2026
and is kept only as a comparison column, and the ATM quoted spread, which is a
cost history (historical option quotes are otherwise unavailable on our plans).

Timing: the nightly run captures after the 16:00 ET close, so these are closing
quotes. ``quote_time_utc`` is the latest contract quote time in the chain. A
consumer must join a row AS-OF the next session -- the decision timestamp must
be after ``captured_at`` -- never onto the same day's bars.

Cleaning rules, documented because they change what the features mean:
  * IV is Schwab's ``volatility`` in percent. Only (1%, 500%] is kept: Schwab
    sends -999 when it has no IV and >500% on near-worthless wings.
  * bid >= $0.05 and ask >= bid, i.e. a real two-sided quote.
  * 0.05 <= |delta| <= 0.95 and DTE >= 7, the study's own filters.
  * nonStandard / mini contracts are dropped (adjusted deliverables).
The delta filter truncates the far wings, so ``rn_skew`` is a within-filter
estimate: comparable across names and days, not an absolute moment.
A chain without ``interestRate`` uses r = 0; r only enters BKM through e^{rT}
(~0.3% at 30 days) and the raw rate is kept in ``interest_rate`` either way.

CW spread note: Schwab's ``volatility`` is ONE value per strike -- identical for
the call and the put in 99.94% of 16,943 matched pairs checked on 2026-09-10 --
so it cannot carry a call-put spread (it read exactly 0.0 on all 20 names).
Both IVs are instead backed out of each side's mid with European Black-Scholes
using the chain's ``interestRate`` and ``dividendYield``. American puts carry an
early-exercise premium a European model reads as extra put IV, which biases the
spread slightly negative and most for high-dividend names; the window is kept
near the money, where that premium is smallest, and ``dividend_yield`` is
stored so research can control for it.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from strategies.dealer_positioning.chain import _iter_contracts_from_side_map, _normalize_option_type

FEATURE_VERSION = "iv_surface_v1"

IV_MIN = 0.01
IV_MAX = 5.0
MIN_BID = 0.05
DELTA_MIN = 0.05
DELTA_MAX = 0.95
MIN_DTE = 7
NEAR_TARGET_DTE = 30
FAR_TARGET_DTE = 60
# Shape features (CW spread, smirk, skew, spread) come from the single expiry
# closest to 30 days, and only if one falls inside this window.
NEAR_EXPIRY_WINDOW = (14, 60)
# ATM IV at a target tenor is interpolated in total variance between the two
# bracketing expiries; without a bracket, the nearest expiry is used flat only
# if it is within this many days of the target.
MAX_FLAT_EXTRAPOLATION_DAYS = 7
MIN_OTM_PER_SIDE = 3
CW_MONEYNESS_WINDOW = (0.95, 1.05)

CONTRACT_COLUMNS = [
    "expiration",
    "dte",
    "strike",
    "option_type",
    "bid",
    "ask",
    "mark",
    "volatility_pct",
    "delta",
    "open_interest",
    "volume",
    "quote_time_utc",
    "non_standard",
]


def _num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def flatten_chain(chain: dict[str, Any]) -> pd.DataFrame:
    """One row per contract, values exactly as Schwab sent them (IV in percent)."""
    rows: list[dict[str, Any]] = []
    for side_key, fallback in (("callExpDateMap", "C"), ("putExpDateMap", "P")):
        for expiration, dte, strike, contract in _iter_contracts_from_side_map(chain.get(side_key) or {}, fallback):
            quote_ms = _num(contract.get("quoteTimeInLong"))
            strike_raw = contract.get("strikePrice")
            rows.append(
                {
                    "expiration": expiration,
                    "dte": dte if dte is not None else _num(contract.get("daysToExpiration")),
                    "strike": _num(strike_raw if strike_raw is not None else strike),
                    "option_type": _normalize_option_type(contract.get("putCall"), fallback),
                    "bid": _num(contract.get("bid")),
                    "ask": _num(contract.get("ask")),
                    "mark": _num(contract.get("mark")),
                    "volatility_pct": _num(contract.get("volatility")),
                    "delta": _num(contract.get("delta")),
                    "open_interest": _num(contract.get("openInterest")),
                    "volume": _num(contract.get("totalVolume")),
                    "quote_time_utc": pd.to_datetime(quote_ms, unit="ms", utc=True) if math.isfinite(quote_ms) else pd.NaT,
                    "non_standard": bool(contract.get("nonStandard")) or bool(contract.get("mini")),
                }
            )
    frame = pd.DataFrame(rows, columns=CONTRACT_COLUMNS)
    frame["dte"] = pd.to_numeric(frame["dte"], errors="coerce")
    frame["quote_time_utc"] = pd.to_datetime(frame["quote_time_utc"], utc=True)
    return frame


def clean_contracts(contracts: pd.DataFrame) -> pd.DataFrame:
    """Apply the module-docstring cleaning rules; adds ``iv`` (fraction) and ``mid``."""
    iv = contracts["volatility_pct"] / 100.0
    bid = contracts["bid"]
    ask = contracts["ask"]
    mask = (
        (iv > IV_MIN)
        & (iv <= IV_MAX)
        & (bid >= MIN_BID)
        & (ask >= bid)
        & contracts["delta"].abs().between(DELTA_MIN, DELTA_MAX)
        & (contracts["dte"] >= MIN_DTE)
        & ~contracts["non_standard"].astype(bool)
    )
    out = contracts.loc[mask].copy()
    out["iv"] = iv[mask]
    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    return out


def _side(expiry: pd.DataFrame, option_type: str) -> pd.DataFrame:
    side = expiry[expiry["option_type"] == option_type]
    return side.drop_duplicates("strike").set_index("strike").sort_index()


def _atm_strike(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> float | None:
    strikes = calls.index.union(puts.index).to_numpy(dtype=float)
    if strikes.size == 0:
        return None
    return float(strikes[np.argmin(np.abs(strikes - spot))])


def _atm_iv(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> float:
    strike = _atm_strike(calls, puts, spot)
    if strike is None:
        return math.nan
    values = [float(side.at[strike, "iv"]) for side in (calls, puts) if strike in side.index]
    return float(np.mean(values))


def _atm_spread_pct(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> float:
    strike = _atm_strike(calls, puts, spot)
    if strike is None:
        return math.nan
    values = [
        float((side.at[strike, "ask"] - side.at[strike, "bid"]) / side.at[strike, "mid"])
        for side in (calls, puts)
        if strike in side.index
    ]
    return float(np.mean(values))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(*, spot: float, strike: float, years: float, rate: float, div_yield: float, sigma: float, kind: str) -> float:
    vol_t = sigma * math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate - div_yield + 0.5 * sigma**2) * years) / vol_t
    d2 = d1 - vol_t
    if kind == "C":
        return spot * math.exp(-div_yield * years) * _norm_cdf(d1) - strike * math.exp(-rate * years) * _norm_cdf(d2)
    return strike * math.exp(-rate * years) * _norm_cdf(-d2) - spot * math.exp(-div_yield * years) * _norm_cdf(-d1)


def _implied_vol(price: float, **market: Any) -> float:
    """European Black-Scholes IV by bisection on [IV_MIN, IV_MAX]; NaN outside the no-arbitrage band."""
    lo, hi = IV_MIN, IV_MAX
    if not _bs_price(sigma=lo, **market) < price < _bs_price(sigma=hi, **market):
        return math.nan
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _bs_price(sigma=mid, **market) < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _cw_iv_spread(
    calls: pd.DataFrame, puts: pd.DataFrame, *, spot: float, rate: float, div_yield: float, years: float
) -> float:
    pairs = calls[["mid", "open_interest"]].join(
        puts[["mid", "open_interest"]], how="inner", lsuffix="_call", rsuffix="_put"
    )
    moneyness = pairs.index.to_numpy(dtype=float) / spot
    pairs = pairs[(moneyness >= CW_MONEYNESS_WINDOW[0]) & (moneyness <= CW_MONEYNESS_WINDOW[1])]
    if pairs.empty or years <= 0:
        return math.nan
    diffs: list[float] = []
    weights: list[float] = []
    for strike, row in pairs.iterrows():
        market = {"spot": spot, "strike": float(strike), "years": years, "rate": rate, "div_yield": div_yield}
        call_iv = _implied_vol(float(row["mid_call"]), kind="C", **market)
        put_iv = _implied_vol(float(row["mid_put"]), kind="P", **market)
        if math.isfinite(call_iv) and math.isfinite(put_iv):
            diffs.append(call_iv - put_iv)
            weights.append((_num(row["open_interest_call"]) or 0.0) / 2.0 + (_num(row["open_interest_put"]) or 0.0) / 2.0)
    if not diffs:
        return math.nan
    diff = np.asarray(diffs)
    weight = np.nan_to_num(np.asarray(weights))
    if float(weight.sum()) <= 0.0:
        return float(diff.mean())
    return float((diff * weight).sum() / weight.sum())


def _smirk(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> float:
    """Xing-Zhang-Zhao: OTM put IV (K/S nearest 0.95 in [0.80, 0.95]) minus ATM call IV (K/S nearest 1)."""
    put_m = puts.index.to_numpy(dtype=float) / spot
    call_m = calls.index.to_numpy(dtype=float) / spot
    otm_puts = puts[(put_m >= 0.80) & (put_m <= 0.95)]
    atm_calls = calls[(call_m >= 0.95) & (call_m <= 1.05)]
    if otm_puts.empty or atm_calls.empty:
        return math.nan
    put_iv = otm_puts["iv"].iloc[np.argmin(np.abs(otm_puts.index.to_numpy(dtype=float) / spot - 0.95))]
    call_iv = atm_calls["iv"].iloc[np.argmin(np.abs(atm_calls.index.to_numpy(dtype=float) / spot - 1.0))]
    return float(put_iv - call_iv)


def _bkm_skewness(calls: pd.DataFrame, puts: pd.DataFrame, *, spot: float, rate: float, years: float) -> float:
    """Bakshi-Kapadia-Madan (2003) risk-neutral skewness from OTM option mids.

    Strike integrals use VIX-style widths (np.gradient: central differences,
    one-sided at the ends).
    """
    otm_calls = calls[calls.index > spot]
    otm_puts = puts[puts.index < spot]
    if len(otm_calls) < MIN_OTM_PER_SIDE or len(otm_puts) < MIN_OTM_PER_SIDE or years <= 0:
        return math.nan
    kc = otm_calls.index.to_numpy(dtype=float)
    kp = otm_puts.index.to_numpy(dtype=float)
    wc = otm_calls["mid"].to_numpy(dtype=float) * np.gradient(kc) / kc**2
    wp = otm_puts["mid"].to_numpy(dtype=float) * np.gradient(kp) / kp**2
    xc = np.log(kc / spot)
    xp = np.log(spot / kp)
    v = np.sum(2.0 * (1.0 - xc) * wc) + np.sum(2.0 * (1.0 + xp) * wp)
    w = np.sum((6.0 * xc - 3.0 * xc**2) * wc) - np.sum((6.0 * xp + 3.0 * xp**2) * wp)
    x = np.sum((12.0 * xc**2 - 4.0 * xc**3) * wc) + np.sum((12.0 * xp**2 + 4.0 * xp**3) * wp)
    ert = math.exp(rate * years)
    mu = ert - 1.0 - ert * v / 2.0 - ert * w / 6.0 - ert * x / 24.0
    variance = ert * v - mu**2
    if not variance > 0.0:
        return math.nan
    return float((ert * w - 3.0 * mu * ert * v + 2.0 * mu**3) / variance**1.5)


def expiry_shape(expiry: pd.DataFrame, *, spot: float, rate: float, div_yield: float = 0.0) -> dict[str, float]:
    """Shape features for one expiration's cleaned contracts."""
    calls = _side(expiry, "C")
    puts = _side(expiry, "P")
    dte = float(expiry["dte"].iloc[0])
    return {
        "dte": dte,
        "atm_iv": _atm_iv(calls, puts, spot),
        "cw_iv_spread": _cw_iv_spread(calls, puts, spot=spot, rate=rate, div_yield=div_yield, years=dte / 365.0),
        "smirk": _smirk(calls, puts, spot),
        "rn_skew": _bkm_skewness(calls, puts, spot=spot, rate=rate, years=dte / 365.0),
        "atm_spread_pct": _atm_spread_pct(calls, puts, spot),
        "median_spread_pct": float(((expiry["ask"] - expiry["bid"]) / expiry["mid"]).median()),
    }


def interpolate_atm_iv(shapes: pd.DataFrame, target_dte: float) -> float:
    """ATM IV at ``target_dte``, linear in total variance (iv^2 * T) between bracketing expiries."""
    points = shapes.loc[shapes["atm_iv"].notna() & (shapes["dte"] > 0), ["dte", "atm_iv"]].sort_values("dte")
    if points.empty:
        return math.nan
    below = points[points["dte"] <= target_dte]
    above = points[points["dte"] >= target_dte]
    if not below.empty and not above.empty:
        d1, v1 = float(below["dte"].iloc[-1]), float(below["atm_iv"].iloc[-1])
        d2, v2 = float(above["dte"].iloc[0]), float(above["atm_iv"].iloc[0])
        if d1 == d2:
            return v1
        w1, w2 = v1**2 * d1, v2**2 * d2
        total = w1 + (w2 - w1) * (target_dte - d1) / (d2 - d1)
        return math.sqrt(total / target_dte) if total > 0 else math.nan
    nearest = points.iloc[int(np.argmin(np.abs(points["dte"].to_numpy() - target_dte)))]
    if abs(float(nearest["dte"]) - target_dte) <= MAX_FLAT_EXTRAPOLATION_DAYS:
        return float(nearest["atm_iv"])
    return math.nan


def surface_features(
    chain: dict[str, Any],
    *,
    symbol: str,
    snapshot_date: str,
    captured_at: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return (one feature row, the raw contract rows it was computed from)."""
    spot = _num(chain.get("underlyingPrice"))
    if not spot > 0:
        raise ValueError(f"{symbol}: option chain has no usable underlyingPrice")
    rate_pct = _num(chain.get("interestRate"))
    rate = rate_pct / 100.0 if math.isfinite(rate_pct) else 0.0
    div_pct = _num(chain.get("dividendYield"))
    div_yield = div_pct / 100.0 if math.isfinite(div_pct) else 0.0

    contracts = flatten_chain(chain)
    if contracts.empty:
        raise ValueError(f"{symbol}: option chain has no contracts")
    clean = clean_contracts(contracts)
    shapes = pd.DataFrame(
        [
            {"expiration": exp, **expiry_shape(group, spot=spot, rate=rate, div_yield=div_yield)}
            for exp, group in clean.groupby("expiration")
        ],
        columns=["expiration", "dte", "atm_iv", "cw_iv_spread", "smirk", "rn_skew", "atm_spread_pct", "median_spread_pct"],
    )
    iv_30 = interpolate_atm_iv(shapes, NEAR_TARGET_DTE)
    iv_60 = interpolate_atm_iv(shapes, FAR_TARGET_DTE)
    near = shapes[shapes["dte"].between(*NEAR_EXPIRY_WINDOW)]
    near_row = near.iloc[int(np.argmin(np.abs(near["dte"].to_numpy() - NEAR_TARGET_DTE)))] if not near.empty else None

    def _near(col: str) -> float:
        return float(near_row[col]) if near_row is not None else math.nan

    features = {
        "symbol": symbol.upper(),
        "snapshot_date": snapshot_date,
        "captured_at": captured_at,
        "quote_time_utc": contracts["quote_time_utc"].max(),
        "feature_version": FEATURE_VERSION,
        "spot": spot,
        "interest_rate": rate_pct / 100.0 if math.isfinite(rate_pct) else math.nan,
        "dividend_yield": div_pct / 100.0 if math.isfinite(div_pct) else math.nan,
        "is_delayed": bool(chain.get("isDelayed")),
        "n_contracts_raw": int(len(contracts)),
        "n_contracts_clean": int(len(clean)),
        "n_expirations_clean": int(len(shapes)),
        "total_open_interest": float(contracts["open_interest"].sum(skipna=True)),
        "total_volume": float(contracts["volume"].sum(skipna=True)),
        "iv_atm_30": iv_30,
        "iv_atm_60": iv_60,
        "iv_term_slope_30_60": iv_60 - iv_30,
        "surface_expiration_30": near_row["expiration"] if near_row is not None else None,
        "surface_dte_30": _near("dte"),
        "cw_iv_spread_30": _near("cw_iv_spread"),
        "rn_skew_30": _near("rn_skew"),
        "smirk_30": _near("smirk"),
        "atm_spread_pct_30": _near("atm_spread_pct"),
        "median_spread_pct_30": _near("median_spread_pct"),
    }
    archive = contracts.assign(
        symbol=symbol.upper(),
        snapshot_date=snapshot_date,
        captured_at=captured_at,
        spot=spot,
    )
    return features, archive
