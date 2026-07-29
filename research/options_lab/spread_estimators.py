"""Per-contract empirical bid/ask spread estimators for `research/options_lab`.

Follow-up to Gate G1c (`research/options_experiment/01_gate_g1_verdict.md`).
`fills.py`'s `estimate_spread` is a 5-variable log-linear regression fit
against the G1b fill-residual proxy with cross-sectional **R^2 ~ 0.10** (see
that module's docstring for the fit derivation). It predicts approximately
the *median* spread for every contract, which is fatal to the specific
question this experiment exists to answer: naked single-leg vs 2-4 leg
structures, where leg count multiplies exactly the per-contract spread cost
the regression cannot distinguish (a wide-market thin small-cap contract
pays that cost 2-4x; a tight-market liquid one pays it once).

No real historical bid/ask exists anywhere in this repo's data (see
`fills.py`'s module docstring: Alpaca's `/v1beta1/options/quotes` verified
404s; the Schwab dealer snapshots have no bid/ask column on any of the 16
cached days). Every estimator below is therefore inferred *indirectly* from
things Alpaca does serve -- real trade prints (`chain_cache.fetch_trades`)
and real OHLC bars (`chain_cache.fetch_bars`) -- using established
market-microstructure theory instead of a single global regression:

1. **Roll (1984)** effective spread: the bid/ask bounce induces NEGATIVE
   serial covariance in trade-to-trade price changes; the spread is
   recovered as `2 * sqrt(-cov(dp_t, dp_{t-1}))`. Undefined (returns
   `None`) whenever the serial covariance is non-negative -- Roll's own
   paper treats that case as "the model does not apply here," not as
   "spread is zero." We never clamp to zero or take `abs()`: a fabricated
   zero-spread is worse than an honest "unmeasurable."

2. **Corwin & Schultz (2012)** high-low spread: derived from the ratio of
   the 2-period combined high/low range to the sum of the two single-period
   ranges (the more of the combined range that is "new" vs. overlapping,
   the more of it is diffusion rather than spread). See
   `corwin_schultz_spread_pct` for the exact alpha/beta/gamma formulas and
   the negative-value convention chosen (set-to-zero, not discard --
   documented there).

3. **Price clustering** (heuristic, weakest of the three): trade prints
   execute AT the bid or AT the ask far more often than in between, for the
   size/liquidity profile of contracts in this book; the gap between the
   two most-frequently-traded price levels in a window approximates the
   spread. No theoretical grounding, placed last in the fallback ladder for
   exactly that reason.

`estimate_contract_spread` combines these with an explicit fallback ladder
(Roll -> Corwin-Schultz -> clustering -> `fills.py`'s regression as last
resort) and reports which method actually produced the number, so
downstream code and the validation report can state estimator provenance
per contract instead of silently mixing methods.

See `research/options_experiment/02_spread_model.md` for the validation of
each estimator against the 470 real-fill residuals in
`research/options_experiment/data/g1b_reprice.csv`, including the explicit
caveat that those residuals also contain the model's own pricing error and
so are an UPPER BOUND on true spread, not a clean bid/ask ground truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from research.options_lab import chain_cache, fills

# --------------------------------------------------------------------------
# Minimum-sample thresholds. Below these, an estimator returns `None` rather
# than a number computed from too few observations to trust -- mirrors
# liquidity.py / fills.py's "unknown must stay unknown" discipline.
# --------------------------------------------------------------------------

ROLL_MIN_PRINTS = 8          # >=7 diffs -> >=6 lag-1 pairs for the covariance
CS_MIN_WINDOWS = 3           # >=3 two-bar windows before trusting the mean
CLUSTER_MIN_TRADES = 8
CLUSTER_MIN_DISTINCT_PRICES = 2

_CS_K = 3.0 - 2.0 * math.sqrt(2.0)


# --------------------------------------------------------------------------
# 1. Roll (1984) effective spread
# --------------------------------------------------------------------------

def roll_effective_spread_pct(
    prices: Sequence[float], *, min_prints: int = ROLL_MIN_PRINTS
) -> Optional[float]:
    """Roll (1984) effective spread from consecutive real trade prints.

    `prices` must be in chronological order (as returned by
    `chain_cache.fetch_trades`'s `p` column). Computes
    `spread = 2 * sqrt(-cov(dp_t, dp_{t-1}))` where `dp_t` are trade-to-trade
    price changes.

    Returns `None` (never a fabricated number) when:
    - fewer than `min_prints` usable (finite, positive) prints are given,
    - or the serial covariance of price changes is >= 0. A non-negative
      serial covariance means the bid/ask-bounce signature Roll's model
      requires is not present in this window (e.g. a strongly trending
      contract, or too few prints for the covariance to be reliably
      negative) -- Roll's own derivation is explicit that the estimator is
      inapplicable in that case. We do NOT clamp to zero or take `abs()`:
      both would silently convert "cannot measure this" into a specific,
      wrong number.

    The result is expressed as a fraction of the mean trade price in the
    window (same units as `fills.estimate_spread`'s round-trip spread_pct).
    """
    px = [float(p) for p in prices if p is not None and math.isfinite(p) and p > 0]
    if len(px) < min_prints:
        return None
    diffs = np.diff(np.asarray(px, dtype=float))
    if len(diffs) < 3:
        return None
    cov = float(np.cov(diffs[1:], diffs[:-1], ddof=1)[0, 1])
    if cov >= 0:
        return None
    dollar_spread = 2.0 * math.sqrt(-cov)
    ref_price = float(np.mean(px))
    if not (math.isfinite(ref_price) and ref_price > 0):
        return None
    return dollar_spread / ref_price


# --------------------------------------------------------------------------
# 2. Corwin & Schultz (2012) high-low spread
# --------------------------------------------------------------------------

def _cs_window_spread(h1: float, l1: float, h2: float, l2: float) -> Optional[float]:
    """One two-period Corwin-Schultz spread estimate. `None` on an invalid
    (non-finite, non-positive, or high < low) input bar -- a bad bar must
    not silently poison the average."""
    vals = (h1, l1, h2, l2)
    if not all(math.isfinite(v) and v > 0 for v in vals):
        return None
    if h1 < l1 or h2 < l2:
        return None
    beta = math.log(h1 / l1) ** 2 + math.log(h2 / l2) ** 2
    h2p, l2p = max(h1, h2), min(l1, l2)
    if h2p <= 0 or l2p <= 0:
        return None
    gamma = math.log(h2p / l2p) ** 2
    alpha = (math.sqrt(2.0 * beta) - math.sqrt(beta)) / _CS_K - math.sqrt(gamma / _CS_K)
    return 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))


def corwin_schultz_spread_pct(
    highs: Sequence[float], lows: Sequence[float], *, min_windows: int = CS_MIN_WINDOWS
) -> Optional[float]:
    """Corwin & Schultz (2012) high-low spread estimator, adapted to
    intraday bars.

    The original paper uses consecutive TRADING-DAY bars. Most contracts in
    this book are 0-7 DTE and never accumulate more than a handful of
    trading days, so a daily-bar version would starve on data almost
    everywhere it matters. We instead feed this function consecutive
    **30-minute intraday bars from a single session** (see
    `estimate_contract_spread`'s caller for how those are fetched) -- the
    derivation only requires two consecutive, equal-length, non-overlapping
    periods with a high/low, which an intraday bar satisfies exactly as
    well as a daily one. The tradeoff, stated plainly: intraday bars are
    more likely to carry within-session trend/autocorrelation than
    close-to-close daily bars, which the CS derivation assumes away: this
    can bias individual two-bar estimates in either direction. We do not
    have the multi-day history to test daily-bar CS against this
    intraday-bar CS on the same contracts, so this bias is not quantified
    here -- treat CS's cross-sectional performance (reported in
    `02_spread_model.md`) as reflecting the intraday adaptation, not the
    textbook daily estimator.

    For each pair of consecutive bars: `beta` = sum of squared single-bar
    log(high/low); `gamma` = squared log(2-bar-high / 2-bar-low); `alpha`
    solves the CS closed form; the 2-bar spread is
    `2*(e^alpha - 1)/(1 + e^alpha)`.

    **Negative-value convention (documented, not buried):** `alpha` (and
    therefore the raw 2-bar spread) can come out negative when the 2-bar
    combined range barely exceeds the two single-bar ranges (i.e. very
    little "new" range beyond overlap -- consistent with an unusually tight
    market in that window). We SET these windows' spread to **zero** rather
    than discarding them, following Corwin & Schultz's own recommendation
    for averaging over many periods. Discarding would selectively remove
    only the tight-market windows and bias the surviving average upward --
    exactly the wrong direction for an estimator we are trying to keep
    honest. The average reported is therefore a mean over ALL usable
    windows, with negative windows counted as a true zero-spread reading,
    not omitted.

    Returns `None` when fewer than `min_windows` two-bar windows produce a
    valid estimate (bad/missing high-low data, or too short a bar series).
    """
    highs = list(highs)
    lows = list(lows)
    if len(highs) != len(lows) or len(highs) < 2:
        return None
    window_spreads: list[float] = []
    for i in range(len(highs) - 1):
        s = _cs_window_spread(highs[i], lows[i], highs[i + 1], lows[i + 1])
        if s is None:
            continue
        window_spreads.append(max(s, 0.0))  # negative-alpha convention -- see docstring
    if len(window_spreads) < min_windows:
        return None
    return float(np.mean(window_spreads))


# --------------------------------------------------------------------------
# 3. Price clustering (heuristic)
# --------------------------------------------------------------------------

def price_clustering_spread_pct(
    prices: Sequence[float],
    *,
    tick_size: float = 0.01,
    min_trades: int = CLUSTER_MIN_TRADES,
    min_distinct_prices: int = CLUSTER_MIN_DISTINCT_PRICES,
) -> Optional[float]:
    """Heuristic spread proxy from trade-price clustering.

    A market maker crossing retail flow fills AT the quoted bid or AT the
    quoted ask far more often than at a price strictly in between (price
    improvement inside the spread is the exception for the contract sizes
    in this book, not the rule). Real trade prints therefore tend to
    cluster around (at most) two dominant price levels within a short
    window. We round prints to `tick_size`, take the two MOST-FREQUENTLY
    traded distinct price levels, and use their gap as a spread proxy.

    This has no theoretical grounding (unlike Roll or Corwin-Schultz) --
    it is a pure heuristic, placed last in the fallback ladder before the
    regression for exactly that reason. It also degenerates when a
    contract trades entirely at one clustered price in-window (a single
    quoted level dominating, or simply too few distinct prints): returns
    `None` rather than reporting a fabricated zero spread.

    Returns `None` when fewer than `min_trades` usable prints are given, or
    fewer than `min_distinct_prices` distinct (tick-rounded) price levels
    are present.
    """
    px = [round(float(p) / tick_size) * tick_size for p in prices if p is not None and math.isfinite(p) and p > 0]
    if len(px) < min_trades:
        return None
    counts = pd.Series(px).value_counts()
    if len(counts) < min_distinct_prices:
        return None
    top2 = counts.nlargest(2)
    levels = sorted(top2.index.tolist())
    dollar_spread = levels[-1] - levels[0]
    if not (math.isfinite(dollar_spread) and dollar_spread > 0):
        return None
    ref_price = float(np.mean(px))
    if not (math.isfinite(ref_price) and ref_price > 0):
        return None
    return dollar_spread / ref_price


# --------------------------------------------------------------------------
# Fallback ladder
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SpreadEstimate:
    """A per-contract spread estimate plus which method actually produced
    it, so downstream reporting can state estimator provenance instead of
    silently mixing methods with very different reliability."""

    spread_pct: Optional[float]  # full round-trip spread as a fraction of price; None if unmeasurable
    method: str  # "roll" | "corwin_schultz" | "clustering" | "regression" | "none"


# Corwin-Schultz is DELIBERATELY EXCLUDED from this ladder.
#
# Measured on 401 real contracts against the realized half-spread implied by
# actual fills (`scripts/validate_spread_estimators.py`), Corwin-Schultz is
# significantly WRONG-SIGNED: Spearman = -0.326, p = 2.8e-11. It does not merely
# fail to help, it ranks contracts backwards.
#
# The mechanism is a broken assumption, not a coding error. Corwin-Schultz infers
# the spread from consecutive high/low ranges, assuming the intraday range is
# driven mainly by bid/ask bounce. For options that assumption collapses: the
# high/low range is dominated by real movement in the underlying, and the
# contracts with the widest ranges tend to be the actively-traded ones with the
# TIGHTEST spreads -- hence the inversion.
#
# Including it in the ladder was actively harmful: it served 155 of 401 contracts
# and dragged the combined estimator to Spearman 0.044 / R^2 0.008, far worse than
# any single component including the regression it was meant to improve on.
# The estimator function is retained above for reference and testing, but must not
# feed the combined estimate on options data.
_LADDER: tuple[str, ...] = ("roll", "clustering", "regression")


def combine_spread_estimates(
    *,
    roll_pct: Optional[float] = None,
    corwin_schultz_pct: Optional[float] = None,
    clustering_pct: Optional[float] = None,
    regression_pct: Optional[float] = None,
) -> SpreadEstimate:
    """Apply the fallback ladder: Roll -> clustering -> regression -> (none).
    The first candidate that is a finite, strictly positive number wins;
    `None`/non-finite/non-positive candidates are skipped, never coerced.

    Roll is the only contract-specific estimator that measures correctly on this
    data (Spearman 0.523 vs the regression's 0.340, with clean monotonic tercile
    separation: 5.3% / 10.8% / 16.6% realized half-spread). Clustering is a weak
    heuristic; the regression is the R^2~0.12 last resort that predicts
    approximately the median for every contract.

    ``corwin_schultz_pct`` is accepted for call-compatibility and reporting but is
    IGNORED for combination -- see the block comment above `_LADDER` for the
    measured evidence.
    """
    candidates = {
        "roll": roll_pct,
        "clustering": clustering_pct,
        "regression": regression_pct,
    }
    for method in _LADDER:
        val = candidates[method]
        if val is not None and math.isfinite(val) and val > 0:
            return SpreadEstimate(float(val), method)
    return SpreadEstimate(None, "none")


# --------------------------------------------------------------------------
# Network-touching orchestration: fetches real trades/bars via chain_cache
# and applies the ladder. Not exercised by the unit tests (no network in
# tests, per test_spread_estimators.py's docstring) -- covered by
# `scripts/validate_spread_estimators.py` against the real G1b/G1c sample.
# --------------------------------------------------------------------------

def estimate_contract_spread(
    osi_symbol: str,
    trades_start: str,
    trades_end: str,
    *,
    bars_start: Optional[str] = None,
    bars_end: Optional[str] = None,
    moneyness: Optional[float] = None,
    dte: Optional[float] = None,
    oi: Optional[float] = None,
    volume: Optional[float] = None,
    underlying_adv: Optional[float] = None,
    calibration: Optional["fills.SpreadCalibration"] = None,
    env_file: Optional[str] = ".env",
) -> SpreadEstimate:
    """Estimate one contract's round-trip spread from real trade prints and
    real 30-minute bars, falling back to `fills.py`'s regression only when
    every empirical method is unmeasurable for this contract.

    `trades_start`/`trades_end`: window for `chain_cache.fetch_trades` --
    feeds Roll and clustering. `bars_start`/`bars_end` (defaults to
    `trades_start`/`trades_end` if not given): window for
    `chain_cache.fetch_bars` at `"30Min"` -- feeds Corwin-Schultz.
    `moneyness`/`dte`/`oi`/`volume`/`underlying_adv`: passed straight to
    `fills.estimate_spread` as the last-resort input (any `None` there
    means the regression itself returns `None`, per its own contract).
    """
    trades = chain_cache.fetch_trades([osi_symbol], trades_start, trades_end, env_file=env_file)
    prices: list[float] = []
    if not trades.empty:
        sub = trades.loc[trades["osi_symbol"] == osi_symbol]
        prices = sub["p"].dropna().tolist()
    roll_pct = roll_effective_spread_pct(prices)
    clustering_pct = price_clustering_spread_pct(prices)

    bars = chain_cache.fetch_bars(
        [osi_symbol], "30Min", bars_start or trades_start, bars_end or trades_end, env_file=env_file
    )
    cs_pct = None
    if not bars.empty:
        sub_bars = bars.loc[bars["osi_symbol"] == osi_symbol].copy()
        if not sub_bars.empty:
            sub_bars["t"] = pd.to_datetime(sub_bars["t"], utc=True)
            sub_bars = sub_bars.sort_values("t")
            cs_pct = corwin_schultz_spread_pct(sub_bars["h"].tolist(), sub_bars["l"].tolist())

    calib = calibration if calibration is not None else fills.DEFAULT_CALIBRATION
    regression_pct = fills.estimate_spread(moneyness, dte, oi, volume, underlying_adv, calibration=calib)

    return combine_spread_estimates(
        roll_pct=roll_pct,
        corwin_schultz_pct=cs_pct,
        clustering_pct=clustering_pct,
        regression_pct=regression_pct,
    )
