"""Point-in-time implied-vol surface, IV rank/percentile, and realized-vol
estimators for `research/options_lab`.

Scope (Phase 1 of docs/superpowers/plans/2026-07-25-options-instrument-routing-experiment.md):
- `build_iv_surface`: per-contract implied vol for one ticker on one `asof`
  day, from real contract bars + contract meta.
- `atm_iv_by_expiry` / `term_structure_slope` / `fit_skew`: surface summary
  statistics (ATM IV per expiry, term-structure slope, quadratic
  log-moneyness skew fit).
- `iv_rank` / `iv_percentile`: rank/percentile of today's ATM IV against a
  caller-supplied trailing history.
- `close_to_close_vol` / `yang_zhang_vol`: realized-vol estimators from
  daily OHLC bars.
- `iv_rv_premium`: ATM IV / realized vol.

Dependency-injection contract (per AGENTS.md -- no hidden global state):
this module never fetches or caches option-chain or bar data itself, and
does NOT import `research/options_lab/chain_cache.py` (owned by a
concurrent task). Every function takes plain DataFrames/values as
arguments. Expected input schemas:

  contract bars (option-contract daily bars, one ticker):
      columns [osi_symbol, ts (UTC tz-aware datetime64), open, high, low,
               close, volume, trade_count, vwap]

  contract meta (one ticker):
      columns [osi_symbol, ticker, expiry (date), strike (float),
               right ('C'|'P')]

  underlying daily bars (one ticker):
      columns [symbol, timestamp (UTC tz-aware datetime64), open, high,
               low, close, volume, trade_count, vwap]
      -- matches the real, inspected schema of
      Data/shared/bars/1d/{TICKER}.parquet.

CRITICAL time correctness (AGENTS.md): every function here takes an `asof`
and uses only data at or before it, at DAY granularity (these are daily/EOD
surfaces, not intraday). Each function that consumes a timestamped frame
calls `_assert_no_lookahead`, which RAISES if any row's date is after
`asof`, rather than silently filtering rows away -- a lookahead violation
in the input is treated as a caller bug to fail on, not paper over.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from research.options_lab.pricing import implied_vol

# --------------------------------------------------------------------------
# Shared time-correctness guard
# --------------------------------------------------------------------------


def _normalize_asof(asof) -> pd.Timestamp:
    ts = pd.Timestamp(asof)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _normalize_dates(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    return dt.dt.normalize()


def _assert_no_lookahead(df: pd.DataFrame, ts_col: str, asof) -> pd.Timestamp:
    """Raise if `df[ts_col]` contains any date strictly after `asof`
    (day-granularity comparison). Returns the normalized asof date.
    """
    if ts_col not in df.columns:
        raise ValueError(f"expected column {ts_col!r} in input frame, got {list(df.columns)}")
    asof_date = _normalize_asof(asof)
    if df.empty:
        return asof_date
    dates = _normalize_dates(df[ts_col])
    violations = dates > asof_date
    if violations.any():
        bad = dates[violations].max()
        raise ValueError(
            f"lookahead violation: column {ts_col!r} contains a date "
            f"({bad.date()}) after asof={asof_date.date()}"
        )
    return asof_date


# --------------------------------------------------------------------------
# IV surface
# --------------------------------------------------------------------------


def build_iv_surface(
    contract_bars: pd.DataFrame,
    contract_meta: pd.DataFrame,
    *,
    asof,
    spot: float,
    r: float,
    q: float = 0.0,
    price_field: str = "close",
    american: bool = False,
) -> pd.DataFrame:
    """Build a point-in-time, per-contract IV surface for one ticker on `asof`.

    For each `osi_symbol`, takes the LAST bar at or before `asof` (an as-of
    join over whatever history `contract_bars` contains -- passing just one
    day's bars, or a longer history, both work). Guarded against lookahead:
    raises if `contract_bars` contains any `ts` after `asof`.

    Args:
        contract_bars: see module docstring for schema.
        contract_meta: see module docstring for schema.
        asof: the snapshot date.
        spot: underlying close/mark on `asof`. Supplied by the caller --
            this module never fetches prices.
        r, q: annualized continuously-compounded risk-free rate / dividend
            yield in decimal, applied uniformly across the surface. Use
            `pricing.risk_free_rate(asof, tenor)` for `r` if desired.
        price_field: which `contract_bars` column is the observed mark for
            IV inversion. Default 'close'; 'vwap' is a defensible
            alternative given there is no historical bid/ask (see the
            Phase 1 plan's known-risks section on fill realism) -- close
            is used by default as the simpler, more standard convention.
        american: if True, invert IV against the Bjerksund-Stensland
            American approximation instead of European BSM (relevant for
            puts and dividend-paying-name calls). Default False.

    Returns a DataFrame, one row per contract, columns:
        osi_symbol, ticker, expiry, strike, right, T (years, ACT/365.25;
        can be negative if the contract had already expired as of `asof`),
        log_moneyness (ln(strike/spot)), price, iv (float or None).
    Rows with T <= 0 get iv=None (rather than being dropped), so callers
    can see coverage/exclusion explicitly; `implied_vol` itself also
    returns None for out-of-arbitrage-bounds or non-convergent prices.
    """
    _assert_no_lookahead(contract_bars, "ts", asof)

    required_bar_cols = {"osi_symbol", "ts", price_field}
    missing_bars = required_bar_cols - set(contract_bars.columns)
    if missing_bars:
        raise ValueError(f"contract_bars missing required columns: {sorted(missing_bars)}")
    required_meta_cols = {"osi_symbol", "ticker", "expiry", "strike", "right"}
    missing_meta = required_meta_cols - set(contract_meta.columns)
    if missing_meta:
        raise ValueError(f"contract_meta missing required columns: {sorted(missing_meta)}")
    if not (math.isfinite(spot) and spot > 0):
        raise ValueError(f"spot must be finite and > 0, got {spot}")

    if contract_bars.empty:
        return pd.DataFrame(
            columns=["osi_symbol", "ticker", "expiry", "strike", "right", "T",
                     "log_moneyness", "price", "iv"]
        )

    bars_sorted = contract_bars.sort_values("ts")
    last_bar = bars_sorted.groupby("osi_symbol", as_index=False).last()
    merged = last_bar.merge(contract_meta, on="osi_symbol", how="inner")
    if merged.empty:
        raise ValueError("no contract_bars rows matched contract_meta on osi_symbol")

    asof_date = _normalize_asof(asof)
    expiry_dates = _normalize_dates(merged["expiry"])
    merged["T"] = (expiry_dates - asof_date).dt.days / 365.25
    merged["log_moneyness"] = np.log(merged["strike"].astype(float) / float(spot))
    merged["price"] = merged[price_field].astype(float)

    ivs: list[Optional[float]] = []
    for row in merged.itertuples(index=False):
        T_i = float(getattr(row, "T"))
        if T_i <= 0:
            ivs.append(None)
            continue
        ivs.append(
            implied_vol(
                price=float(getattr(row, "price")),
                S=float(spot),
                K=float(getattr(row, "strike")),
                T=T_i,
                r=float(r),
                q=float(q),
                right=str(getattr(row, "right")),
                american=american,
            )
        )
    merged["iv"] = ivs

    out_cols = ["osi_symbol", "ticker", "expiry", "strike", "right", "T",
                "log_moneyness", "price", "iv"]
    return merged[out_cols].reset_index(drop=True)


def atm_iv_by_expiry(surface: pd.DataFrame) -> pd.DataFrame:
    """Per-expiry ATM IV from a `build_iv_surface` output.

    ATM IV := mean of the nearest-to-spot (min |log_moneyness|) priced call
    IV and nearest-to-spot priced put IV for that expiry, when both sides
    are present; otherwise whichever single side is available. Rows with
    iv is None are excluded before the nearest-strike search.

    Returns columns [expiry, T, atm_iv], sorted by T ascending.
    """
    if surface.empty:
        return pd.DataFrame(columns=["expiry", "T", "atm_iv"])
    priced = surface[surface["iv"].notna()].copy()
    if priced.empty:
        return pd.DataFrame(columns=["expiry", "T", "atm_iv"])
    priced["abs_logm"] = priced["log_moneyness"].abs()

    out_rows = []
    for expiry, grp in priced.groupby("expiry"):
        side_ivs = []
        for right in ("C", "P"):
            side = grp[grp["right"] == right]
            if not side.empty:
                nearest = side.loc[side["abs_logm"].idxmin()]
                side_ivs.append(float(nearest["iv"]))
        if not side_ivs:
            continue
        out_rows.append({
            "expiry": expiry,
            "T": float(grp["T"].iloc[0]),
            "atm_iv": float(np.mean(side_ivs)),
        })
    return pd.DataFrame(out_rows).sort_values("T").reset_index(drop=True)


def term_structure_slope(atm_by_expiry: pd.DataFrame) -> float:
    """Least-squares slope of ATM IV vs time-to-expiry T (years) across the
    expiries in `atm_by_expiry` (output of `atm_iv_by_expiry`).

    Units: decimal vol per year of T. Positive = upward-sloping/contango,
    negative = inverted/backwardation (common around near-term events).
    Raises if fewer than 2 expiries are present (a slope needs >= 2 points).
    """
    if len(atm_by_expiry) < 2:
        raise ValueError("need >= 2 expiries to fit a term-structure slope")
    T = atm_by_expiry["T"].to_numpy(dtype=float)
    iv = atm_by_expiry["atm_iv"].to_numpy(dtype=float)
    slope, _intercept = np.polyfit(T, iv, 1)
    return float(slope)


def fit_skew(surface: pd.DataFrame, expiry) -> dict:
    """Quadratic skew fit for one expiry: iv ~ a + b*k + c*k^2, where
    k = log_moneyness = ln(strike/spot).

    Choice of quadratic-in-log-moneyness: the simplest parametric form that
    captures both skew (b: the linear term; equity single-names are
    typically put-skewed, b<0 expected) and smile curvature (c: the
    quadratic term) without overfitting the handful of strikes per expiry
    that a thin single-name chain realistically offers (richer
    parametrizations like SVI/SABR need more points per expiry than that).

    Only out-of-the-money contracts are used (puts for k<=0, calls for
    k>=0), to avoid the stale/thin quotes typical of deep-ITM single-name
    equity option bars given there is no historical bid/ask (Phase 1
    plan's known-risks section).

    Returns {'a', 'b', 'c', 'n_points'}. Raises if fewer than 3 usable
    (OTM, priced) points exist for `expiry`.
    """
    sub = surface[(surface["expiry"] == expiry) & surface["iv"].notna()]
    otm = sub[
        ((sub["right"] == "P") & (sub["log_moneyness"] <= 0))
        | ((sub["right"] == "C") & (sub["log_moneyness"] >= 0))
    ]
    if len(otm) < 3:
        raise ValueError(
            f"need >= 3 OTM priced contracts to fit a skew for expiry={expiry}, "
            f"got {len(otm)}"
        )
    k = otm["log_moneyness"].to_numpy(dtype=float)
    iv = otm["iv"].to_numpy(dtype=float)
    c, b, a = np.polyfit(k, iv, 2)
    return {"a": float(a), "b": float(b), "c": float(c), "n_points": int(len(otm))}


# --------------------------------------------------------------------------
# IV rank / percentile
# --------------------------------------------------------------------------


def _trailing_window(history: pd.DataFrame, asof_date: pd.Timestamp,
                      lookback_days: int) -> pd.DataFrame:
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    dates = _normalize_dates(history["date"])
    start = asof_date - pd.Timedelta(days=lookback_days)
    window = history.loc[(dates > start) & (dates <= asof_date)]
    if window.empty:
        raise ValueError(
            f"no history rows in ({start.date()}, {asof_date.date()}]"
        )
    return window


def _value_at(window: pd.DataFrame, asof_date: pd.Timestamp) -> float:
    dates = _normalize_dates(window["date"])
    at_asof = window.loc[dates == asof_date, "atm_iv"]
    if at_asof.empty:
        raise ValueError(
            f"history has no row at asof={asof_date.date()} (needed as 'today')"
        )
    return float(at_asof.iloc[-1])


def iv_rank(history: pd.DataFrame, asof, lookback_days: int) -> float:
    """IV Rank: where today's ATM IV sits between the trailing window's min
    and max, on a 0-100 scale (100 = today is the window high).

    history: columns ['date', 'atm_iv'] -- a caller-supplied time series of
        one ticker's daily ATM IV (e.g. `atm_iv_by_expiry` run per
        historical day for a fixed tenor convention, then concatenated).
        This module never fetches or caches history itself.
    asof: must have a row in `history` with date == asof ("today"). Guarded
        against lookahead: raises if `history` contains any date after
        `asof`.
    lookback_days: trailing CALENDAR-day window ending at `asof` (inclusive).

    Degenerate case: if every value in the window is identical (max==min),
    returns 50.0 rather than dividing by zero.
    """
    asof_date = _assert_no_lookahead(history, "date", asof)
    window = _trailing_window(history, asof_date, lookback_days)
    current = _value_at(window, asof_date)
    lo, hi = float(window["atm_iv"].min()), float(window["atm_iv"].max())
    if hi == lo:
        return 50.0
    return float((current - lo) / (hi - lo) * 100.0)


def iv_percentile(history: pd.DataFrame, asof, lookback_days: int) -> float:
    """IV Percentile: fraction of trailing-window days whose ATM IV was
    <= today's, on a 0-100 scale. Less sensitive to a single outlier day
    than `iv_rank`. Same inputs/guards as `iv_rank`.
    """
    asof_date = _assert_no_lookahead(history, "date", asof)
    window = _trailing_window(history, asof_date, lookback_days)
    current = _value_at(window, asof_date)
    return float((window["atm_iv"] <= current).mean() * 100.0)


# --------------------------------------------------------------------------
# Realized volatility (from underlying daily OHLC bars)
# --------------------------------------------------------------------------


def _trailing_bars(bars: pd.DataFrame, asof, window: int) -> tuple[pd.DataFrame, pd.Timestamp]:
    asof_date = _assert_no_lookahead(bars, "timestamp", asof)
    b = bars.sort_values("timestamp")
    dates = _normalize_dates(b["timestamp"])
    b = b.loc[dates <= asof_date]
    if len(b) < window + 1:
        raise ValueError(
            f"need >= {window + 1} daily bars at/before {asof_date.date()} "
            f"for a {window}-bar window, got {len(b)}"
        )
    return b.tail(window + 1).reset_index(drop=True), asof_date


def close_to_close_vol(bars: pd.DataFrame, asof, window: int) -> float:
    """Close-to-close realized volatility (annualized, 252 trading days/yr).

    bars: daily OHLCV bars for ONE ticker -- see module docstring for the
        exact schema (matches Data/shared/bars/1d/{TICKER}.parquet).
    asof: last date to include (inclusive). Guarded against lookahead:
        raises if `bars` contains any timestamp after `asof`.
    window: number of trailing daily log returns to use (needs window+1
        closes).
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    b, _asof_date = _trailing_bars(bars, asof, window)
    closes = b["close"].to_numpy(dtype=float)
    log_ret = np.diff(np.log(closes))
    sigma_daily = log_ret.std(ddof=1)
    return float(sigma_daily * math.sqrt(252.0))


def yang_zhang_vol(bars: pd.DataFrame, asof, window: int) -> float:
    """Yang & Zhang (2000) drift-independent realized-vol estimator,
    annualized (252 trading days/yr).

    Reference: Yang, D., & Zhang, Q. (2000). "Drift-Independent Volatility
    Estimation Based on High, Low, Open, and Close Prices." Journal of
    Business, 73(3), 477-491.

    Combines overnight (close-to-open) variance, open-to-close variance,
    and the Rogers-Satchell intraday-range estimator, weighted to minimize
    estimator variance regardless of drift or opening-jump size:
        sigma_YZ^2 = V_overnight + k * V_open_to_close + (1-k) * V_rogers_satchell
        k = 0.34 / (1.34 + (n+1)/(n-1))
    where V_overnight/V_open_to_close are sample variances (ddof=1) of the
    per-period log returns and V_rogers_satchell is their mean.

    bars/asof/window: see `close_to_close_vol`. Needs window+1 daily bars
    (period i's overnight return needs period i-1's close).
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    b, _asof_date = _trailing_bars(bars, asof, window)

    close = b["close"].to_numpy(dtype=float)
    open_ = b["open"].to_numpy(dtype=float)
    high = b["high"].to_numpy(dtype=float)
    low = b["low"].to_numpy(dtype=float)

    overnight = np.log(open_[1:] / close[:-1])
    op = open_[1:]
    u = np.log(high[1:] / op)
    d = np.log(low[1:] / op)
    c = np.log(close[1:] / op)

    n = window
    v_o = overnight.var(ddof=1)
    v_c = c.var(ddof=1)
    v_rs = float(np.mean(u * (u - c) + d * (d - c)))
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    var_yz = v_o + k * v_c + (1.0 - k) * v_rs
    if var_yz < 0:
        raise ValueError(
            "Yang-Zhang variance estimate is negative -- check input bars for "
            "bad/inconsistent OHLC data (e.g. high < low, or close outside "
            "[low, high])"
        )
    return float(math.sqrt(var_yz * 252.0))


def iv_rv_premium(atm_iv: float, realized_vol: float) -> float:
    """ATM IV / realized vol (both annualized, decimal units).

    >1 means options are pricing in more movement than has realized
    recently (a headwind for naked long premium); <1 means the reverse.
    Raises if either input is non-finite, or if `realized_vol` <= 0 (a
    zero/negative realized vol makes the ratio either undefined or a sign
    artifact, not a usable premium).
    """
    if not (math.isfinite(realized_vol) and realized_vol > 0):
        raise ValueError(f"realized_vol must be finite and > 0, got {realized_vol}")
    if not (math.isfinite(atm_iv) and atm_iv >= 0):
        raise ValueError(f"atm_iv must be finite and >= 0, got {atm_iv}")
    return float(atm_iv / realized_vol)
