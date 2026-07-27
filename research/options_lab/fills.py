"""Fill/execution-cost model for `research/options_lab`.

Part of Gate G1 (`docs/superpowers/plans/2026-07-25-options-instrument-routing-experiment.md`,
Phase 1). Real historical bid/ask is not available from Alpaca
(`chain_cache.py`'s own docstring: `/v1beta1/options/quotes` returns 404 --
verified, not assumed). The plan's primary idea (Phase 1's Gate-G1 checklist)
was to calibrate a spread model against the 16 days of Schwab
dealer-positioning chain snapshots, which the plan believed carried true
bid/ask.

**That assumption did not hold, verified here, not assumed:**
`Data/dealer_positioning/historical_snapshots/*/dealer_strike_ladder.parquet`
has NO bid/ask column of any kind, on any of the 16 available days
(checked by `scripts/validate_options_gate_g1.py::check_bid_ask_columns`,
which lists every column and finds nothing matching "bid"/"ask"). The plan's
own documented fallback therefore applies: calibrate against the G1b
real-fill reprice residuals instead --
`Data/analysis/multi_ticker_swing_live/paired_option_trades.csv`'s 575 real
closed option trades, repriced from real Alpaca option bars at the real
entry/exit timestamps
(`research/options_experiment/data/g1b_reprice.csv` /
`g1c_calibration_inputs.csv`).

Calibration derivation
-----------------------
For each repriceable real trade, opening a long pays *above* the bar-based
"modeled" price and closing it realizes *below* that price -- exactly what a
real bid/ask spread does (pay near the ask, receive near the bid). So:

    round_trip_cost_dollars = (actual_entry - modeled_entry) - (actual_exit - modeled_exit)
    round_trip_cost_pct     = round_trip_cost_dollars / modeled_entry_price

is an empirical, per-trade proxy for the round-trip bid/ask spread cost.
`DEFAULT_CALIBRATION`'s coefficients are an OLS fit of
`log(round_trip_cost_pct)` on moneyness, `log1p(DTE)`, `log1p(open interest)`,
`log1p(contract volume)`, `log1p(underlying ADV)` over the winsorized (2nd -
98th percentile), positive-cost subset of repriceable trades. See
`scripts/validate_options_gate_g1.py` (`run_g1c` / `summarize_g1c`) for the
fit itself and `research/options_experiment/01_gate_g1.md` for the reported
diagnostics (n, R^2, residual std) this specific run produced -- reproduce
by re-running that script; the numbers below are a point-in-time snapshot,
not hand-tuned.

**Important limitation, stated loudly, not buried:** this calibration is fit
against the SAME residual data it is meant to explain -- there is no
independent held-out bid/ask to validate against, because none exists in
this repo's data. It is the best available estimate given that hard data
constraint, not an independently verified spread model, and it is fit on
575 trades from one 4-week window (2026-05) of one module's live book, so it
may not generalize to different tickers, moneyness ranges, or DTE outside
what that sample covered. Every Phase 3 result MUST therefore be reported
under all three fill assumptions below (per the plan's own §6 requirement),
and any conclusion that does not survive `"pessimistic"` is not a conclusion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

DEFAULT_COMMISSION_PER_CONTRACT = 0.65

_VALID_SIDES = ("buy", "sell")
_VALID_ASSUMPTIONS = ("optimistic", "calibrated", "pessimistic")


@dataclass(frozen=True)
class SpreadCalibration:
    """OLS fit of log(round-trip spread pct) on the features below. See
    module docstring for the derivation, data source, and its circularity
    caveat -- this is a fit against fill residuals, not independent bid/ask.
    """

    intercept: float
    moneyness_coef: float
    log1p_dte_coef: float
    log1p_oi_coef: float
    log1p_volume_coef: float
    log1p_adv_coef: float
    min_spread_pct: float = 0.01   # floor: no real market quotes a literal 0 spread
    max_spread_pct: float = 3.0    # cap: guards against runaway extrapolation outside the fit range


# Calibrated by scripts/validate_options_gate_g1.py --task g1c against the
# 575-trade real-fill ledger (see that script + research/options_experiment/
# 01_gate_g1.md for the exact n / R^2 / residual-std this run produced).
# This is a genuinely weak fit (proxy target, small/narrow sample) -- treat
# the "calibrated" assumption as a directional central estimate, and always
# cross-check against "pessimistic" before trusting a Phase 3 conclusion.
DEFAULT_CALIBRATION = SpreadCalibration(
    intercept=-1.1626,
    moneyness_coef=1.7132,
    log1p_dte_coef=-0.0523,
    log1p_oi_coef=-0.0405,
    log1p_volume_coef=0.0032,
    log1p_adv_coef=-0.0335,
)


def estimate_spread(
    moneyness: Optional[float],
    dte: Optional[float],
    oi: Optional[float],
    volume: Optional[float],
    underlying_adv: Optional[float],
    *,
    calibration: SpreadCalibration = DEFAULT_CALIBRATION,
) -> Optional[float]:
    """Estimate the full round-trip bid/ask spread as a fraction of price.

    Inputs:
        moneyness: abs(log(strike / spot)), >= 0.
        dte: calendar days to expiry, >= 0.
        oi: open interest (contracts), >= 0.
        volume: contract volume over the reference lookback (contracts), >= 0.
        underlying_adv: trailing average daily share volume of the
            underlying, >= 0.

    Returns `None` (never a fabricated number) when any input is `None`,
    i.e. the caller does not actually know that liquidity fact -- mirrors
    `liquidity.py`'s "unknown must stay unknown, never coerced to a
    guessed/zero value" discipline. Raises `ValueError` for inputs that ARE
    known but out of domain (negative, non-finite) -- that is a caller bug,
    not missing data.

    The resulting spread_pct is clipped to
    [`calibration.min_spread_pct`, `calibration.max_spread_pct`] -- the fit
    is a log-linear extrapolation and can blow up outside the range the 575
    calibration trades actually covered (see module docstring).
    """
    values = {"moneyness": moneyness, "dte": dte, "oi": oi, "volume": volume,
              "underlying_adv": underlying_adv}
    if any(v is None for v in values.values()):
        return None
    for name, v in values.items():
        if not (math.isfinite(v) and v >= 0):
            raise ValueError(f"{name} must be finite and >= 0, got {v!r}")

    log_spread = (
        calibration.intercept
        + calibration.moneyness_coef * moneyness
        + calibration.log1p_dte_coef * math.log1p(dte)
        + calibration.log1p_oi_coef * math.log1p(oi)
        + calibration.log1p_volume_coef * math.log1p(volume)
        + calibration.log1p_adv_coef * math.log1p(underlying_adv)
    )
    spread_pct = math.exp(log_spread)
    return float(min(max(spread_pct, calibration.min_spread_pct), calibration.max_spread_pct))


def apply_fill(mid: float, side: str, spread_pct: float, assumption: str) -> float:
    """Apply a spread-cost assumption to a mid price and return the modeled
    fill price for one leg.

    `side`: "buy" (opening a long or closing a short -- you pay, price moves
        against you upward) or "sell" (closing a long or opening a short --
        you receive, price moves against you downward).
    `spread_pct`: the FULL round-trip bid/ask spread as a fraction of `mid`
        (i.e. (ask - bid) / mid), e.g. from `estimate_spread`.
    `assumption`:
        "optimistic"  -- fill at `mid` exactly (no cost). Never a real fill;
            an upper bound on achievable execution.
        "calibrated"  -- cross HALF the spread (the standard "mid +/- half
            spread" convention), scaled by the empirically fit spread.
        "pessimistic" -- cross the FULL spread (worst-realistic single-leg
            execution). Per the plan's §6: any conclusion that does not
            survive this assumption is not a conclusion.

    Multi-leg structures: call this once per leg and sum -- the plan is
    explicit that "multi-leg structures pay this cost N times."
    """
    if not (math.isfinite(mid) and mid > 0):
        raise ValueError(f"mid must be finite and > 0, got {mid}")
    if side not in _VALID_SIDES:
        raise ValueError(f"side must be one of {_VALID_SIDES}, got {side!r}")
    if not (math.isfinite(spread_pct) and spread_pct >= 0):
        raise ValueError(f"spread_pct must be finite and >= 0, got {spread_pct}")
    if assumption not in _VALID_ASSUMPTIONS:
        raise ValueError(f"assumption must be one of {_VALID_ASSUMPTIONS}, got {assumption!r}")

    sign = 1.0 if side == "buy" else -1.0
    cross_fraction = {"optimistic": 0.0, "calibrated": 0.5, "pessimistic": 1.0}[assumption]
    return float(mid * (1.0 + sign * cross_fraction * spread_pct))


def estimate_spread_empirical(
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
    calibration: SpreadCalibration = DEFAULT_CALIBRATION,
    env_file: Optional[str] = ".env",
):
    """Thin hook to `research.options_lab.spread_estimators.estimate_contract_spread`.

    Gate G1c follow-up (see `research/options_experiment/02_spread_model.md`):
    this module's own `estimate_spread` regression has cross-sectional
    R^2 ~ 0.10 and predicts approximately the median spread for every
    contract. `spread_estimators.py` replaces it as the PRIMARY per-contract
    source (Roll -> Corwin-Schultz -> clustering -> `estimate_spread` here
    as the last-resort fallback), returning which method produced the
    number. This function only forwards to that module -- it is deliberately
    NOT a change to `estimate_spread`'s signature or behavior, so every
    existing caller and test of `estimate_spread`/`apply_fill` is untouched.

    Imports `spread_estimators` locally (not at module level) because that
    module imports `fills` for its own regression fallback -- a top-level
    import here would be circular.
    """
    from research.options_lab import spread_estimators  # local: avoid import cycle

    return spread_estimators.estimate_contract_spread(
        osi_symbol,
        trades_start,
        trades_end,
        bars_start=bars_start,
        bars_end=bars_end,
        moneyness=moneyness,
        dte=dte,
        oi=oi,
        volume=volume,
        underlying_adv=underlying_adv,
        calibration=calibration,
        env_file=env_file,
    )


def commission_cost(n_contracts: int, *, rate: float = DEFAULT_COMMISSION_PER_CONTRACT) -> float:
    """Commission for one leg: `rate` dollars per contract, `n_contracts`
    contracts. Configurable per the task spec's default ($0.65/contract)."""
    if not (isinstance(n_contracts, (int, float)) and math.isfinite(n_contracts) and n_contracts >= 0):
        raise ValueError(f"n_contracts must be finite and >= 0, got {n_contracts}")
    if not (math.isfinite(rate) and rate >= 0):
        raise ValueError(f"rate must be finite and >= 0, got {rate}")
    return float(rate) * float(n_contracts)


def structure_commission(contracts_per_leg: Sequence[int], *, rate: float = DEFAULT_COMMISSION_PER_CONTRACT) -> float:
    """Multi-leg commission: cost applies per leg (plan §Phase 1 `fills.py`
    responsibility: "Multi-leg: cost applies per leg"). `contracts_per_leg`
    is the contract count traded on each leg (e.g. `[1, 1]` for a 1-lot
    vertical spread's two legs); total commission sums each leg
    independently -- a 4-leg iron condor pays 4x a single-leg long call at
    the same contract count.
    """
    if not contracts_per_leg:
        raise ValueError("contracts_per_leg must be non-empty -- a structure needs at least one leg")
    return float(sum(commission_cost(c, rate=rate) for c in contracts_per_leg))
