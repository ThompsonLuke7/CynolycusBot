"""Composable options-structure library for `research/options_lab`.

Scope (Phase 2 of docs/superpowers/plans/2026-07-25-options-instrument-routing-experiment.md):
every strategy below is a *selection rule* that, given (ticker, asof, spot,
direction, an available options chain), picks concrete contracts and returns
a `Selection` wrapping either a built `Structure` (legs + entry context) or
`None` plus a machine-readable failure reason. Returning `None` is a common,
correct outcome (missing expiry bucket, no strike near the target delta, an
illiquid leg) -- never silently substitute a far-off strike to force a
structure into existence.

Dependency-injection contract (per AGENTS.md -- no hidden global state):
this module never fetches or caches chain/bar data itself and does NOT
import `research/options_lab/chain_cache.py`. It consumes a plain
`chain` DataFrame the caller has already built (typically via
`chain_cache.discover_contracts` + `chain_cache.fetch_bars` +
`surface.build_iv_surface` for IV, and `pricing.bsm_greeks` for vega) and
reads `research/options_lab/{pricing,liquidity}.py` for pure functions
(pricing math, tradability gates). It does not import
`research/options_lab/fills.py` (owned by a concurrent task, not needed
here: entry/exit slippage and commission are a Phase 3 concern layered on
top of the structures this module builds).

Chain schema (`CHAIN_COLUMNS`) -- one row per candidate contract:
    osi_symbol, ticker, expiry ('YYYY-MM-DD'), strike (float),
    right ('C'|'P'), price (float -- the entry mark, e.g. close or vwap;
    caller's choice, see `surface.build_iv_surface`'s own `price_field`
    note), iv (float|None -- implied vol at that mark, e.g. from
    `surface.build_iv_surface`), vega (float|None -- raw dPrice/dsigma
    from `pricing.bsm_greeks`, used ONLY as the IV-reliability gate, per
    Phase 1b: vega < ~$0.05/vol-pt means the recovered IV is not
    identifiable and must not drive delta-based selection), quote_asof
    (the timestamp the price/iv/vega were actually observed at -- the
    no-lookahead guard checks this against the requested `asof`),
    open_interest, bar_volume, trade_count (int|None, fed straight into
    `liquidity.LiquidityStats` for the leg-level tradability gate).

Core types:
    `Leg` -- one contract or a share lot in a position.
    `Structure` -- an immutable, ENTERED position: legs plus the market
        context (entry_spot, entry_asof, r, q) it was entered under. Unlike
        `entry_cost`/`max_loss`/etc in a naive design, baking entry context
        into the Structure itself (rather than threading spot/asof/r/q
        through every property call) is what lets every listed "computed
        property" in the Phase-2 spec actually be a bare zero-arg property:
        the selection functions already have spot/asof/r/q in hand at
        construction time.
    `payoff_at_expiry(structure, spot_grid)` / `value_at(structure, spot,
        asof, iv, r, q)` are free functions (not properties) because they
        evaluate the structure at ARBITRARY spot/time/vol -- e.g. mark-to-
        market on some later date before expiry -- not just its own entry
        state.

Sizing hooks (`resize_to_notional`, `resize_to_max_loss`) scale every leg's
`quantity` by one scalar factor. This deliberately produces fractional
contract counts: the point is Phase-3 apples-to-apples PnL-per-dollar
comparability across instrument types (matched notional / matched max-loss),
not a literal order ticket. A caller that needs live-tradeable integer
contracts must round separately -- that rounding is a live-execution
concern, not a research-normalization one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from research.options_lab import liquidity
from research.options_lab.pricing import Greeks, bsm_greeks, bsm_price

ArrayLike = Union[float, np.ndarray]

# --------------------------------------------------------------------------
# Chain schema + shared time-correctness / candidate-filtering helpers
# --------------------------------------------------------------------------

CHAIN_COLUMNS = [
    "osi_symbol", "ticker", "expiry", "strike", "right", "price", "iv", "vega",
    "quote_asof", "open_interest", "bar_volume", "trade_count",
]

# Phase 1b finding: below this raw vega ($ / 1.00 vol-pt, i.e. divide by 100
# for the "per vol point" convention), the recovered IV self-consistently
# reprices its own input but is not identifiable as *the* generating vol.
# A delta computed from such an IV must not drive selection.
DEFAULT_VEGA_FLOOR = 0.05

# Canonical, machine-readable selection-failure reason codes (Phase 3 reports
# *why* a structure was unavailable, so these must stay stable strings).
REASON_NO_EXPIRY_IN_BUCKET = "no_expiry_in_bucket"
REASON_NO_STRIKE_NEAR_DELTA = "no_strike_near_delta"
REASON_NO_STRIKE_AT_WIDTH = "no_strike_at_width"
REASON_NO_MATCHING_PAIR = "no_matching_pair"
REASON_LEG_ILLIQUID = "leg_illiquid"


def _normalize_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _years_between(asof, expiry) -> float:
    return (_normalize_ts(expiry) - _normalize_ts(asof)).days / 365.25


def _validate_chain_columns(chain: pd.DataFrame) -> None:
    missing = set(CHAIN_COLUMNS) - set(chain.columns)
    if missing:
        raise ValueError(f"chain is missing required columns: {sorted(missing)}")


def _assert_chain_no_lookahead(chain: pd.DataFrame, asof) -> pd.Timestamp:
    """Raise if any row's `quote_asof` (when the price/iv/vega were actually
    observed) is strictly after the requested `asof` -- selecting a contract
    using a mark that had not yet been observed at decision time would be a
    lookahead violation, and this fails loudly rather than silently
    excluding the offending rows."""
    asof_ts = _normalize_ts(asof)
    if chain.empty:
        return asof_ts
    quote_ts = pd.to_datetime(chain["quote_asof"], utc=True, errors="coerce").dt.tz_localize(None)
    if quote_ts.isna().any():
        raise ValueError("chain contains unparseable quote_asof value(s)")
    violations = quote_ts > asof_ts
    if violations.any():
        bad = quote_ts[violations].max()
        raise ValueError(
            f"lookahead violation: chain contains quote_asof={bad} after requested asof={asof_ts}"
        )
    return asof_ts


def _dte_days(chain: pd.DataFrame, asof_ts: pd.Timestamp) -> pd.Series:
    expiry_ts = pd.to_datetime(chain["expiry"])
    return (expiry_ts - asof_ts).dt.days


def _filter_dte_bucket(chain: pd.DataFrame, asof_ts: pd.Timestamp, dte_min: int, dte_max: int) -> pd.DataFrame:
    if dte_min < 0 or dte_max < dte_min:
        raise ValueError(f"invalid dte bucket [{dte_min}, {dte_max}]")
    dte = _dte_days(chain, asof_ts)
    return chain.loc[(dte >= dte_min) & (dte <= dte_max)]


def _reliable_iv_mask(chain: pd.DataFrame, vega_floor: float) -> pd.Series:
    iv_ok = chain["iv"].notna()
    vega_ok = chain["vega"].notna() & (chain["vega"].abs() >= vega_floor)
    return iv_ok & vega_ok


def _row_delta(row, spot: float, asof_ts: pd.Timestamp, r: float, q: float) -> Optional[float]:
    T = (pd.Timestamp(row["expiry"]) - asof_ts).days / 365.25
    if T <= 0 or row["iv"] is None or not math.isfinite(float(row["iv"])):
        return None
    g = bsm_greeks(spot, float(row["strike"]), T, r, q, float(row["iv"]), str(row["right"]))
    return float(g.delta)


def _row_liquidity_ok(row) -> bool:
    stats = liquidity.LiquidityStats(
        osi_symbol=str(row["osi_symbol"]),
        asof=str(row.get("quote_asof")),
        lookback_days=1,
        open_interest=None if pd.isna(row.get("open_interest")) else int(row["open_interest"]),
        bar_volume=None if pd.isna(row.get("bar_volume")) else int(row["bar_volume"]),
        trade_count=None if pd.isna(row.get("trade_count")) else int(row["trade_count"]),
    )
    return liquidity.is_tradable(stats)


def _select_by_delta_or_moneyness(
    bucket: pd.DataFrame, right: str, spot: float, asof_ts: pd.Timestamp, r: float, q: float, *,
    target_delta: Optional[float] = None, moneyness_offset: Optional[float] = None,
    vega_floor: float = DEFAULT_VEGA_FLOOR,
) -> tuple[Optional[pd.Series], Optional[str]]:
    """Pick one contract from `bucket` (already DTE-filtered) matching
    `right`. Prefers a real delta computed from a reliable IV; falls back to
    nearest-strike-to-moneyness-offset when IV is missing/unreliable for
    every candidate, and is explicitly marked as such via the returned
    method string ('delta' vs 'moneyness_fallback'). Returns (None, None)
    if `bucket` has no rows of the requested `right` at all (caller
    translates that to REASON_NO_EXPIRY_IN_BUCKET).
    """
    side = bucket[bucket["right"] == right]
    if side.empty:
        return None, None

    if target_delta is not None:
        reliable = side[_reliable_iv_mask(side, vega_floor)]
        if not reliable.empty:
            deltas = reliable.apply(lambda row: _row_delta(row, spot, asof_ts, r, q), axis=1)
            reliable = reliable.assign(_delta=deltas)
            reliable = reliable[reliable["_delta"].notna()]
            if not reliable.empty:
                dist = (reliable["_delta"].abs() - abs(target_delta)).abs()
                best = reliable.loc[dist.idxmin()]
                return best, "delta"
        if moneyness_offset is not None:
            return _nearest_strike(side, spot, moneyness_offset), "moneyness_fallback"
        return None, "unreliable_iv"

    if moneyness_offset is not None:
        return _nearest_strike(side, spot, moneyness_offset), "moneyness_fallback"

    raise ValueError("must supply target_delta and/or moneyness_offset")


def _nearest_strike(side: pd.DataFrame, spot: float, moneyness_offset: float) -> pd.Series:
    target_strike = spot * (1.0 + moneyness_offset)
    dist = (side["strike"].astype(float) - target_strike).abs()
    return side.loc[dist.idxmin()]


def _nearest_strike_to(side: pd.DataFrame, target_strike: float) -> Optional[pd.Series]:
    if side.empty:
        return None
    dist = (side["strike"].astype(float) - target_strike).abs()
    return side.loc[dist.idxmin()]


# --------------------------------------------------------------------------
# Core types: Leg, NetGreeks, Structure
# --------------------------------------------------------------------------

_VALID_RIGHTS = ("C", "P", "S")


@dataclass(frozen=True)
class Leg:
    """One contract (right='C'/'P') or a share lot (right='S') in a position.

    quantity: signed -- positive is long, negative is short. For shares this
        is a share count; for options it is a contract count.
    entry_price: premium per share of underlying for options (i.e. the
        standard quoted per-contract price before multiplying by
        `multiplier`), or price per share for `right='S'`.
    multiplier: 100 for options, 1 for shares -- enforced in `__post_init__`
        rather than trusted, since a wrong multiplier silently misprices
        every downstream dollar figure.
    iv: the contract's implied vol at entry (required for `entry_greeks`;
        `None` only ever means "could not be recovered", not "unknown but
        fine to ignore").
    """

    osi_symbol: str
    right: str
    strike: Optional[float]
    expiry: Optional[str]
    quantity: float
    entry_price: float
    multiplier: int = 100
    iv: Optional[float] = None

    def __post_init__(self) -> None:
        if self.right not in _VALID_RIGHTS:
            raise ValueError(f"right must be one of {_VALID_RIGHTS}, got {self.right!r}")
        if self.quantity == 0:
            raise ValueError(f"leg {self.osi_symbol!r} has zero quantity (not a position)")
        if not math.isfinite(self.quantity):
            raise ValueError(f"leg {self.osi_symbol!r} quantity must be finite")
        if not (math.isfinite(self.entry_price) and self.entry_price >= 0):
            raise ValueError(f"leg {self.osi_symbol!r} entry_price must be finite and >= 0")
        if self.right == "S":
            if self.multiplier != 1:
                raise ValueError(f"share leg {self.osi_symbol!r} must have multiplier=1, got {self.multiplier}")
            if self.strike is not None or self.expiry is not None:
                raise ValueError(f"share leg {self.osi_symbol!r} must have strike=None and expiry=None")
        else:
            if self.multiplier != 100:
                raise ValueError(f"option leg {self.osi_symbol!r} must have multiplier=100, got {self.multiplier}")
            if self.strike is None or not (math.isfinite(self.strike) and self.strike > 0):
                raise ValueError(f"option leg {self.osi_symbol!r} strike must be finite and > 0")
            if self.expiry is None:
                raise ValueError(f"option leg {self.osi_symbol!r} must have an expiry")
            if self.iv is not None and not (math.isfinite(self.iv) and self.iv > 0):
                raise ValueError(f"option leg {self.osi_symbol!r} iv must be None or finite and > 0")


@dataclass(frozen=True)
class NetGreeks:
    delta: float
    gamma: float
    theta: float
    vega: float


@dataclass(frozen=True)
class Structure:
    """An entered position: legs plus the market context they were entered
    under. See module docstring for why entry_spot/entry_asof/r/q live here
    rather than being threaded through every property call."""

    name: str
    legs: tuple = field(default_factory=tuple)
    entry_spot: float = 0.0
    entry_asof: Optional[str] = None
    r: float = 0.0
    q: float = 0.0

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError(f"structure {self.name!r} must have at least one leg")
        object.__setattr__(self, "legs", tuple(self.legs))
        if not (math.isfinite(self.entry_spot) and self.entry_spot > 0):
            raise ValueError(f"structure {self.name!r} entry_spot must be finite and > 0")
        if self.entry_asof is None:
            raise ValueError(f"structure {self.name!r} requires entry_asof")
        asof_ts = _normalize_ts(self.entry_asof)
        if not (math.isfinite(self.r) and math.isfinite(self.q)):
            raise ValueError(f"structure {self.name!r} r and q must be finite")
        for leg in self.legs:
            if leg.expiry is not None and _normalize_ts(leg.expiry) < asof_ts:
                raise ValueError(
                    f"structure {self.name!r} leg {leg.osi_symbol!r} expiry {leg.expiry} "
                    f"is before entry_asof {self.entry_asof} -- cannot enter an expired contract"
                )

    # -- pure cash-flow properties (no market state needed) ----------------

    @property
    def entry_cost(self) -> float:
        """Net debit (positive) / credit (negative) to establish the
        position, in dollars."""
        return float(sum(leg.quantity * leg.entry_price * leg.multiplier for leg in self.legs))

    @property
    def has_assignment_risk(self) -> bool:
        return any(leg.quantity < 0 and leg.right in ("C", "P") for leg in self.legs)

    def _is_multi_expiry(self) -> bool:
        expiries = {leg.expiry for leg in self.legs if leg.right in ("C", "P")}
        return len(expiries) > 1

    def _slope_at_infinity(self) -> float:
        """d(payoff)/dS as S -> +infinity: positive means unbounded upside
        (long calls / long shares net long), negative means unbounded
        downside-to-the-upside (short calls / short shares net short, i.e.
        the classic naked-short unbounded-loss case). Puts contribute 0
        here -- their payoff flattens to 0 as spot rises."""
        return float(sum(
            leg.quantity * leg.multiplier for leg in self.legs if leg.right in ("C", "S")
        ))

    def _candidate_spots(self) -> np.ndarray:
        strikes = [leg.strike for leg in self.legs if leg.strike is not None]
        anchor = max([self.entry_spot] + strikes)
        far = anchor * 4.0 + 1000.0
        points = {0.0, far}
        points.update(strikes)
        return np.array(sorted(points), dtype=float)

    def _pl_bounds(self) -> tuple[Optional[float], Optional[float]]:
        if self._is_multi_expiry():
            return self._pl_bounds_multi_expiry()
        candidates = self._candidate_spots()
        pl = payoff_at_expiry(self, candidates) - self.entry_cost
        slope_up = self._slope_at_infinity()
        max_gain = None if slope_up > 0 else float(np.max(pl))
        max_loss = None if slope_up < 0 else float(-np.min(pl))
        return max_loss, max_gain

    def _pl_bounds_multi_expiry(self) -> tuple[Optional[float], Optional[float]]:
        # Calendar/diagonal spreads span two expiries: "payoff at expiry" is
        # not a single well-defined function of spot without a vol model for
        # the far leg's remaining time value at the near leg's expiry. We
        # deliberately do NOT fabricate a number here. The one convention
        # that is defensible without a vol model: a net-DEBIT calendar's
        # worst realistic case (spot moves far from the strike so both the
        # expiring near leg and the far leg's extrinsic value collapse
        # toward the debit-neutral floor) is bounded by the debit paid --
        # this is the standard retail-platform convention, reported here as
        # an approximation, not an exact bound. Max gain and, for a net-
        # credit calendar, max loss are NOT computable without assuming a
        # vol path, so both are `None` -- that `None` means "not computable
        # without a volatility model" here, not necessarily "mathematically
        # unbounded"; see this comment, not the has_assignment_risk-style
        # unbounded-tail case.
        debit = self.entry_cost
        if debit > 0:
            return float(debit), None
        return None, None

    @property
    def max_loss(self) -> Optional[float]:
        return self._pl_bounds()[0]

    @property
    def max_gain(self) -> Optional[float]:
        return self._pl_bounds()[1]

    @property
    def is_defined_risk(self) -> bool:
        return self.max_loss is not None

    @property
    def breakevens(self) -> tuple:
        if self._is_multi_expiry():
            # See _pl_bounds_multi_expiry: not computable without a vol model.
            return ()
        candidates = self._candidate_spots()
        pl = payoff_at_expiry(self, candidates) - self.entry_cost
        roots: list[float] = []
        for i in range(len(candidates) - 1):
            s0, s1 = candidates[i], candidates[i + 1]
            p0, p1 = pl[i], pl[i + 1]
            if p0 == 0.0:
                roots.append(float(s0))
            if p0 * p1 < 0.0:
                roots.append(float(s0 + (0.0 - p0) * (s1 - s0) / (p1 - p0)))
        if pl[-1] == 0.0:
            roots.append(float(candidates[-1]))
        return tuple(sorted({round(r, 6) for r in roots}))

    @property
    def entry_greeks(self) -> NetGreeks:
        asof_ts = _normalize_ts(self.entry_asof)
        delta = gamma = theta = vega = 0.0
        for leg in self.legs:
            if leg.right == "S":
                delta += leg.quantity * leg.multiplier * 1.0
                continue
            if leg.iv is None:
                raise ValueError(
                    f"leg {leg.osi_symbol!r} has no iv on record; cannot compute entry_greeks"
                )
            T = (_normalize_ts(leg.expiry) - asof_ts).days / 365.25
            if T <= 0:
                raise ValueError(
                    f"leg {leg.osi_symbol!r} expiry {leg.expiry} is at/before entry_asof "
                    f"{self.entry_asof}"
                )
            g: Greeks = bsm_greeks(self.entry_spot, leg.strike, T, self.r, self.q, leg.iv, leg.right)
            delta += leg.quantity * leg.multiplier * g.delta
            gamma += leg.quantity * leg.multiplier * g.gamma
            theta += leg.quantity * leg.multiplier * g.theta
            vega += leg.quantity * leg.multiplier * g.vega
        return NetGreeks(delta=delta, gamma=gamma, theta=theta, vega=vega)

    # -- buying power (Reg-T-style; each rule commented individually) ------

    def _short_option_legs(self) -> list:
        return [leg for leg in self.legs if leg.right in ("C", "P") and leg.quantity < 0]

    def _is_vertical_credit_spread(self) -> bool:
        opts = [leg for leg in self.legs if leg.right in ("C", "P")]
        if len(opts) != 2 or self._is_multi_expiry():
            return False
        rights = {leg.right for leg in opts}
        if len(rights) != 1:
            return False
        shorts = [leg for leg in opts if leg.quantity < 0]
        longs = [leg for leg in opts if leg.quantity > 0]
        if len(shorts) != 1 or len(longs) != 1:
            return False
        return self.entry_cost < 0  # net credit

    def _vertical_width(self) -> float:
        opts = [leg for leg in self.legs if leg.right in ("C", "P")]
        strikes = [leg.strike for leg in opts]
        return float(abs(strikes[0] - strikes[1]))

    def _is_covered_by_shares(self) -> bool:
        share_legs = [leg for leg in self.legs if leg.right == "S" and leg.quantity > 0]
        if not share_legs:
            return False
        shares = sum(leg.quantity for leg in share_legs)
        short_calls = sum(-leg.quantity for leg in self.legs if leg.right == "C" and leg.quantity < 0)
        return short_calls * 100 <= shares

    @property
    def buying_power_required(self) -> float:
        short_opts = self._short_option_legs()

        # Rule 1: defined-risk, net-debit structure (single-expiry) -- Reg T
        # requires only the debit paid; the long leg(s) cap the loss, so no
        # additional margin is needed. Covers long calls/puts, debit
        # verticals, long straddles/strangles, long butterflies, deep-ITM
        # replacement longs.
        if self.is_defined_risk and self.entry_cost >= 0 and not self._is_multi_expiry():
            return float(self.entry_cost)

        # Rule 2: single-leg cash-secured put -- BP = strike * multiplier *
        # |contracts|, the cash required to buy shares if assigned.
        if (
            len(self.legs) == 1
            and short_opts
            and short_opts[0].right == "P"
        ):
            leg = short_opts[0]
            return float(abs(leg.quantity) * leg.strike * leg.multiplier)

        # Rule 3: two-leg vertical CREDIT spread (same expiry, same right,
        # one long/one short, net credit) -- BP = width*multiplier*contracts
        # minus the credit received.
        if self._is_vertical_credit_spread():
            width = self._vertical_width()
            leg = short_opts[0]
            credit = -self.entry_cost
            return float(width * leg.multiplier * abs(leg.quantity) - credit)

        # Rule 4: short call(s) fully covered by a long share lot (covered
        # call / collar in a net-credit or zero-cost configuration not
        # already caught by Rule 1) -- approximated here as the covering
        # shares' own cash-account notional (conservative; does not model a
        # margin account's reduced share requirement).
        if self._is_covered_by_shares():
            share_leg = next(leg for leg in self.legs if leg.right == "S")
            return float(max(share_leg.quantity, 0.0) * self.entry_spot * share_leg.multiplier)

        # Rule 5: naked short option(s) not covered by any of the above --
        # standard Reg T formula per contract, summed across short legs:
        #   max(20% * underlying - out-of-money amount + premium,
        #       10% * strike + premium)
        if short_opts:
            total = 0.0
            for leg in short_opts:
                premium = abs(leg.entry_price) * leg.multiplier
                if leg.right == "C":
                    otm = max(leg.strike - self.entry_spot, 0.0)
                else:
                    otm = max(self.entry_spot - leg.strike, 0.0)
                rule_a = 0.20 * self.entry_spot * leg.multiplier - otm * leg.multiplier + premium
                rule_b = 0.10 * leg.strike * leg.multiplier + premium
                total += max(rule_a, rule_b) * abs(leg.quantity)
            return float(total)

        # Rule 6: fallback -- no short legs at all (e.g. long shares) -- BP
        # is simply the cash paid.
        return float(max(self.entry_cost, 0.0))


# --------------------------------------------------------------------------
# Free functions: payoff_at_expiry, value_at
# --------------------------------------------------------------------------


def payoff_at_expiry(structure: Structure, spot_grid: ArrayLike) -> np.ndarray:
    """Raw payoff (NOT net of entry_cost) at expiry across `spot_grid`.

    Requires every option leg to share one common expiry -- payoff "at
    expiry" is not a single well-defined function of spot for multi-expiry
    structures (calendar/diagonal spreads); use `value_at` for those.
    """
    expiries = {leg.expiry for leg in structure.legs if leg.right in ("C", "P")}
    if len(expiries) > 1:
        raise ValueError(
            f"payoff_at_expiry requires all option legs to share one expiry, got {sorted(expiries)}; "
            "use value_at for multi-expiry (calendar/diagonal) structures"
        )
    S = np.asarray(spot_grid, dtype=float)
    if np.any(~np.isfinite(S)) or np.any(S < 0):
        raise ValueError("spot_grid must be finite and >= 0")
    total = np.zeros_like(S)
    for leg in structure.legs:
        if leg.right == "S":
            total = total + leg.quantity * leg.multiplier * S
        elif leg.right == "C":
            total = total + leg.quantity * leg.multiplier * np.maximum(S - leg.strike, 0.0)
        else:  # 'P'
            total = total + leg.quantity * leg.multiplier * np.maximum(leg.strike - S, 0.0)
    return total


def value_at(
    structure: Structure, spot: float, asof, iv: Optional[float] = None,
    r: Optional[float] = None, q: Optional[float] = None, *, american: bool = False,
) -> float:
    """Mark-to-market dollar VALUE (not P&L) of `structure` at an arbitrary
    `spot`/`asof` before expiry -- e.g. re-marking the position at a later
    bar than entry, or a scenario spot/vol.

    iv: if given, a single flat vol applied to every option leg (a parallel-
        shift scenario -- e.g. "what is this worth if IV moves to X
        uniformly"). If `None` (default), each leg's OWN recorded `iv` is
        used, which is the realistic per-leg-smile behavior.
    r, q: override the structure's own entry-time rate/yield if given (e.g.
        re-marking on a later date with a different curve read); default to
        the structure's own `r`/`q` when omitted.
    american: use the Bjerksund-Stensland American approximation instead of
        European BSM for option legs with early-exercise risk. Default
        False, matching `surface.build_iv_surface`'s own default.
    """
    if not (math.isfinite(spot) and spot > 0):
        raise ValueError(f"spot must be finite and > 0, got {spot}")
    asof_ts = _normalize_ts(asof)
    r_eff = structure.r if r is None else r
    q_eff = structure.q if q is None else q
    total = 0.0
    for leg in structure.legs:
        if leg.right == "S":
            total += leg.quantity * leg.multiplier * spot
            continue
        leg_iv = leg.iv if iv is None else iv
        if leg_iv is None:
            raise ValueError(f"leg {leg.osi_symbol!r} has no iv and none was supplied to value_at")
        T = (_normalize_ts(leg.expiry) - asof_ts).days / 365.25
        if american:
            from research.options_lab.pricing import bjerksund_stensland_2002
            price = bjerksund_stensland_2002(spot, leg.strike, max(T, 0.0), r_eff, q_eff, leg_iv, leg.right)
        else:
            price = float(bsm_price(spot, leg.strike, max(T, 0.0), r_eff, q_eff, leg_iv, leg.right))
        total += leg.quantity * leg.multiplier * price
    return float(total)


# --------------------------------------------------------------------------
# Sizing / normalization hooks (mandatory per the Phase-2 plan)
# --------------------------------------------------------------------------


def _scale_structure(structure: Structure, factor: float) -> Structure:
    if not (math.isfinite(factor) and factor > 0):
        raise ValueError(f"resize factor must be finite and > 0, got {factor}")
    new_legs = tuple(replace(leg, quantity=leg.quantity * factor) for leg in structure.legs)
    return replace(structure, legs=new_legs)


def resize_to_notional(structure: Structure, target_notional: float = 5000.0) -> Structure:
    """Scale `structure` so its net delta-equivalent underlying exposure
    (`entry_greeks.delta * entry_spot`) matches `target_notional` -- the
    "matched notional" comparability framing (Phase 3 requires every
    structure comparable to $5,000 of shares on this basis)."""
    if not (math.isfinite(target_notional) and target_notional > 0):
        raise ValueError(f"target_notional must be finite and > 0, got {target_notional}")
    net_delta_shares = structure.entry_greeks.delta
    if net_delta_shares == 0:
        raise ValueError(
            f"structure {structure.name!r} has zero net delta -- cannot resize to a target notional"
        )
    current_notional = abs(net_delta_shares) * structure.entry_spot
    return _scale_structure(structure, target_notional / current_notional)


def resize_to_max_loss(structure: Structure, target_max_loss: float) -> Structure:
    """Scale `structure` so its `max_loss` matches `target_max_loss` -- the
    "matched max-loss" comparability framing. Raises for undefined-risk
    structures (max_loss is None) since there is no dollar figure to scale
    against -- silently substituting a fabricated cap would defeat the whole
    point of leaving max_loss unbounded."""
    if not (math.isfinite(target_max_loss) and target_max_loss > 0):
        raise ValueError(f"target_max_loss must be finite and > 0, got {target_max_loss}")
    current = structure.max_loss
    if current is None:
        raise ValueError(
            f"structure {structure.name!r} has unbounded max_loss -- cannot resize to a target max-loss"
        )
    if current == 0:
        raise ValueError(f"structure {structure.name!r} has zero max_loss -- cannot scale to a nonzero target")
    return _scale_structure(structure, target_max_loss / current)


# --------------------------------------------------------------------------
# Selection result
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    """Result of a strategy-selection call. `structure` is `None` exactly
    when `reason` is set (a common, correct outcome -- see module
    docstring); `selection_method` ('delta' | 'moneyness_fallback') is set
    exactly when `structure` is not `None`, recording which contract-
    selection path was actually used for the (or each) delta-targeted leg."""

    structure: Optional[Structure]
    reason: Optional[str] = None
    selection_method: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.structure is None) == (self.reason is None):
            raise ValueError("exactly one of structure/reason must be set")


def _fail(reason: str) -> Selection:
    return Selection(structure=None, reason=reason)


def _ok(structure: Structure, method: Optional[str]) -> Selection:
    return Selection(structure=structure, selection_method=method)


# --------------------------------------------------------------------------
# Leg-building helpers
# --------------------------------------------------------------------------


def _leg_from_row(row, quantity: float) -> Leg:
    return Leg(
        osi_symbol=str(row["osi_symbol"]),
        right=str(row["right"]),
        strike=float(row["strike"]),
        expiry=str(row["expiry"]),
        quantity=quantity,
        entry_price=float(row["price"]),
        multiplier=100,
        iv=None if pd.isna(row.get("iv")) else float(row["iv"]),
    )


def _direction_right(direction: str) -> str:
    if direction == "long":
        return "C"
    if direction == "short":
        return "P"
    raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")


def _require_long_only(name: str, direction: str) -> None:
    if direction != "long":
        raise ValueError(
            f"{name} is a long-bias overlay strategy (long shares as the base position); "
            f"direction must be 'long', got {direction!r}"
        )


# --------------------------------------------------------------------------
# B0 -- long/short shares baseline
# --------------------------------------------------------------------------


def long_shares(
    ticker: str, asof, spot: float, direction: str, chain: Optional[pd.DataFrame] = None, *,
    target_notional: float = 5000.0, r: float = 0.0, q: float = 0.0,
) -> Selection:
    """B0 baseline: shares only, sized to `target_notional` dollars of
    underlying exposure. `chain` is accepted (and ignored) purely for
    call-signature uniformity with the option strategies below -- shares
    have no chain-availability failure mode, so this never returns a
    `reason`; malformed input raises instead (per AGENTS.md fail-fast)."""
    del chain
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
    if not (math.isfinite(spot) and spot > 0):
        raise ValueError(f"spot must be finite and > 0, got {spot}")
    sign = 1.0 if direction == "long" else -1.0
    qty = sign * target_notional / spot
    leg = Leg(osi_symbol=f"{ticker.upper()}_SHARES", right="S", strike=None, expiry=None,
              quantity=qty, entry_price=spot, multiplier=1)
    name = "long_shares" if direction == "long" else "short_shares"
    structure = Structure(name=name, legs=(leg,), entry_spot=spot, entry_asof=asof, r=r, q=q)
    return _ok(structure, method=None)


# --------------------------------------------------------------------------
# B1 -- naked long call/put (the incumbent)
# --------------------------------------------------------------------------


def _naked_long_option(
    name: str, right: str, ticker: str, asof, spot: float, chain: pd.DataFrame, *,
    dte_min: int, dte_max: int, target_delta: Optional[float] = 0.50,
    moneyness_offset: Optional[float] = None, contracts: float = 1.0,
    r: float = 0.0, q: float = 0.0, vega_floor: float = DEFAULT_VEGA_FLOOR,
) -> Selection:
    _validate_chain_columns(chain)
    asof_ts = _assert_chain_no_lookahead(chain, asof)
    bucket = _filter_dte_bucket(chain, asof_ts, dte_min, dte_max)
    if bucket.empty:
        return _fail(REASON_NO_EXPIRY_IN_BUCKET)
    row, method = _select_by_delta_or_moneyness(
        bucket, right, spot, asof_ts, r, q,
        target_delta=target_delta, moneyness_offset=moneyness_offset, vega_floor=vega_floor,
    )
    if row is None:
        return _fail(REASON_NO_STRIKE_NEAR_DELTA if method is None else "unreliable_iv")
    if not _row_liquidity_ok(row):
        return _fail(REASON_LEG_ILLIQUID)
    leg = _leg_from_row(row, quantity=contracts)
    structure = Structure(name=name, legs=(leg,), entry_spot=spot, entry_asof=asof, r=r, q=q)
    return _ok(structure, method)


def long_call(
    ticker: str, asof, spot: float, direction: str, chain: pd.DataFrame, *,
    dte_min: int = 8, dte_max: int = 21, target_delta: Optional[float] = 0.50,
    moneyness_offset: Optional[float] = None, contracts: float = 1.0,
    r: float = 0.0, q: float = 0.0,
) -> Selection:
    """B1 incumbent, bullish leg. `direction='long'` is the only sensible
    call here (a naked long call IS the bullish instrument); accepted for
    signature uniformity but must be 'long'."""
    if direction != "long":
        raise ValueError(f"long_call direction must be 'long', got {direction!r}")
    return _naked_long_option(
        "long_call", "C", ticker, asof, spot, chain, dte_min=dte_min, dte_max=dte_max,
        target_delta=target_delta, moneyness_offset=moneyness_offset, contracts=contracts, r=r, q=q,
    )


def long_put(
    ticker: str, asof, spot: float, direction: str, chain: pd.DataFrame, *,
    dte_min: int = 8, dte_max: int = 21, target_delta: Optional[float] = 0.50,
    moneyness_offset: Optional[float] = None, contracts: float = 1.0,
    r: float = 0.0, q: float = 0.0,
) -> Selection:
    """B1 incumbent, bearish leg. `direction` must be 'short' (bearish)."""
    if direction != "short":
        raise ValueError(f"long_put direction must be 'short', got {direction!r}")
    return _naked_long_option(
        "long_put", "P", ticker, asof, spot, chain, dte_min=dte_min, dte_max=dte_max,
        target_delta=target_delta, moneyness_offset=moneyness_offset, contracts=contracts, r=r, q=q,
    )


# --------------------------------------------------------------------------
# Stock replacement
# --------------------------------------------------------------------------


def deep_itm_call(
    ticker: str, asof, spot: float, direction: str, chain: pd.DataFrame, *,
    dte_min: int = 22, dte_max: int = 45, target_delta: float = 0.75,
    moneyness_offset: Optional[float] = -0.15, contracts: float = 1.0,
    r: float = 0.0, q: float = 0.0,
) -> Selection:
    if direction != "long":
        raise ValueError(f"deep_itm_call direction must be 'long', got {direction!r}")
    if not (0.70 <= target_delta <= 0.85):
        raise ValueError(f"deep_itm_call target_delta must be in [0.70, 0.85], got {target_delta}")
    return _naked_long_option(
        "deep_itm_call", "C", ticker, asof, spot, chain, dte_min=dte_min, dte_max=dte_max,
        target_delta=target_delta, moneyness_offset=moneyness_offset, contracts=contracts, r=r, q=q,
    )


def deep_itm_put(
    ticker: str, asof, spot: float, direction: str, chain: pd.DataFrame, *,
    dte_min: int = 22, dte_max: int = 45, target_delta: float = 0.75,
    moneyness_offset: Optional[float] = 0.15, contracts: float = 1.0,
    r: float = 0.0, q: float = 0.0,
) -> Selection:
    if direction != "short":
        raise ValueError(f"deep_itm_put direction must be 'short', got {direction!r}")
    if not (0.70 <= target_delta <= 0.85):
        raise ValueError(f"deep_itm_put target_delta must be in [0.70, 0.85], got {target_delta}")
    return _naked_long_option(
        "deep_itm_put", "P", ticker, asof, spot, chain, dte_min=dte_min, dte_max=dte_max,
        target_delta=target_delta, moneyness_offset=moneyness_offset, contracts=contracts, r=r, q=q,
    )


def extended_dte_long(
    ticker: str, asof, spot: float, direction: str, chain: pd.DataFrame, *,
    dte_min: int = 46, dte_max: int = 90, target_delta: Optional[float] = 0.50,
    moneyness_offset: Optional[float] = None, contracts: float = 1.0,
    r: float = 0.0, q: float = 0.0,
) -> Selection:
    right = _direction_right(direction)
    name = "extended_dte_long"
    return _naked_long_option(
        name, right, ticker, asof, spot, chain, dte_min=dte_min, dte_max=dte_max,
        target_delta=target_delta, moneyness_offset=moneyness_offset, contracts=contracts, r=r, q=q,
    )


# --------------------------------------------------------------------------
# Directional, defined-risk spreads
# --------------------------------------------------------------------------


def vertical_debit_spread(
    ticker: str, asof, spot: float, direction: str, chain: pd.DataFrame, *,
    dte_min: int = 8, dte_max: int = 21, long_delta: float = 0.60, width: float = 5.0,
    contracts: float = 1.0, r: float = 0.0, q: float = 0.0, vega_floor: float = DEFAULT_VEGA_FLOOR,
) -> Selection:
    """Buy the closer-to-the-money leg (target `long_delta`), sell the leg
    `width` dollars further out of the money, same expiry. Bull call spread
    for direction='long', bear put spread for direction='short'."""
    right = _direction_right(direction)
    _validate_chain_columns(chain)
    asof_ts = _assert_chain_no_lookahead(chain, asof)
    bucket = _filter_dte_bucket(chain, asof_ts, dte_min, dte_max)
    if bucket.empty:
        return _fail(REASON_NO_EXPIRY_IN_BUCKET)

    long_row, method = _select_by_delta_or_moneyness(
        bucket, right, spot, asof_ts, r, q, target_delta=long_delta, vega_floor=vega_floor,
    )
    if long_row is None:
        return _fail(REASON_NO_STRIKE_NEAR_DELTA)

    same_expiry = bucket[(bucket["right"] == right) & (bucket["expiry"] == long_row["expiry"])]
    offset = width if right == "C" else -width
    short_row = _nearest_strike_to(same_expiry, float(long_row["strike"]) + offset)
    if short_row is None or short_row["osi_symbol"] == long_row["osi_symbol"]:
        return _fail(REASON_NO_STRIKE_AT_WIDTH)

    for row in (long_row, short_row):
        if not _row_liquidity_ok(row):
            return _fail(REASON_LEG_ILLIQUID)

    long_leg = _leg_from_row(long_row, quantity=contracts)
    short_leg = _leg_from_row(short_row, quantity=-contracts)
    structure = Structure(
        name="vertical_debit_spread", legs=(long_leg, short_leg),
        entry_spot=spot, entry_asof=asof, r=r, q=q,
    )
    if structure.entry_cost < 0:
        return _fail(REASON_NO_STRIKE_AT_WIDTH)  # would be a credit, not a debit -- width/delta combo invalid
    return _ok(structure, method)


def vertical_credit_spread(
    ticker: str, asof, spot: float, direction: str, chain: pd.DataFrame, *,
    dte_min: int = 8, dte_max: int = 21, short_delta: float = 0.30, width: float = 5.0,
    contracts: float = 1.0, r: float = 0.0, q: float = 0.0, vega_floor: float = DEFAULT_VEGA_FLOOR,
) -> Selection:
    """Sell the near-the-money leg (target `short_delta`), buy protection
    `width` dollars further out of the money, same expiry. Bear call spread
    for direction='short' (theta-positive, bearish), bull put spread for
    direction='long' (theta-positive, bullish) -- the "I want theta on my
    side" tool, opposite convention from `vertical_debit_spread`."""
    right = "P" if direction == "long" else "C" if direction == "short" else None
    if right is None:
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
    _validate_chain_columns(chain)
    asof_ts = _assert_chain_no_lookahead(chain, asof)
    bucket = _filter_dte_bucket(chain, asof_ts, dte_min, dte_max)
    if bucket.empty:
        return _fail(REASON_NO_EXPIRY_IN_BUCKET)

    short_row, method = _select_by_delta_or_moneyness(
        bucket, right, spot, asof_ts, r, q, target_delta=short_delta, vega_floor=vega_floor,
    )
    if short_row is None:
        return _fail(REASON_NO_STRIKE_NEAR_DELTA)

    same_expiry = bucket[(bucket["right"] == right) & (bucket["expiry"] == short_row["expiry"])]
    # Protection leg is further OTM than the short leg: for a bear call
    # spread (short_delta call), further OTM is a HIGHER strike; for a bull
    # put spread (short_delta put), further OTM is a LOWER strike.
    offset = width if right == "C" else -width
    long_row = _nearest_strike_to(same_expiry, float(short_row["strike"]) + offset)
    if long_row is None or long_row["osi_symbol"] == short_row["osi_symbol"]:
        return _fail(REASON_NO_STRIKE_AT_WIDTH)

    for row in (short_row, long_row):
        if not _row_liquidity_ok(row):
            return _fail(REASON_LEG_ILLIQUID)

    short_leg = _leg_from_row(short_row, quantity=-contracts)
    long_leg = _leg_from_row(long_row, quantity=contracts)
    structure = Structure(
        name="vertical_credit_spread", legs=(short_leg, long_leg),
        entry_spot=spot, entry_asof=asof, r=r, q=q,
    )
    if structure.entry_cost >= 0:
        return _fail(REASON_NO_STRIKE_AT_WIDTH)  # would be a debit -- width/delta combo invalid for a credit spread
    return _ok(structure, method)


def broken_wing_butterfly(
    ticker: str, asof, spot: float, direction: str, chain: pd.DataFrame, *,
    dte_min: int = 8, dte_max: int = 21, body_delta: float = 0.50,
    near_width: float = 5.0, far_width: float = 10.0,
    contracts: float = 1.0, r: float = 0.0, q: float = 0.0, vega_floor: float = DEFAULT_VEGA_FLOOR,
) -> Selection:
    """Buy 1x near wing, sell 2x body, buy 1x far wing, all same expiry.
    `near_width != far_width` is what makes it "broken" (skewed payoff/BP
    vs a symmetric butterfly); calls for direction='long', puts for
    direction='short'."""
    if near_width == far_width:
        raise ValueError("broken_wing_butterfly requires near_width != far_width (else it's a symmetric butterfly)")
    right = _direction_right(direction)
    _validate_chain_columns(chain)
    asof_ts = _assert_chain_no_lookahead(chain, asof)
    bucket = _filter_dte_bucket(chain, asof_ts, dte_min, dte_max)
    if bucket.empty:
        return _fail(REASON_NO_EXPIRY_IN_BUCKET)

    body_row, method = _select_by_delta_or_moneyness(
        bucket, right, spot, asof_ts, r, q, target_delta=body_delta, vega_floor=vega_floor,
    )
    if body_row is None:
        return _fail(REASON_NO_STRIKE_NEAR_DELTA)

    same_expiry = bucket[(bucket["right"] == right) & (bucket["expiry"] == body_row["expiry"])]
    body_strike = float(body_row["strike"])
    near_row = _nearest_strike_to(same_expiry, body_strike - near_width)
    far_row = _nearest_strike_to(same_expiry, body_strike + far_width)
    rows = [body_row, near_row, far_row]
    symbols = {r_["osi_symbol"] for r_ in rows if r_ is not None}
    if near_row is None or far_row is None or len(symbols) < 3:
        return _fail(REASON_NO_STRIKE_AT_WIDTH)

    for row in rows:
        if not _row_liquidity_ok(row):
            return _fail(REASON_LEG_ILLIQUID)

    legs = (
        _leg_from_row(near_row, quantity=contracts),
        _leg_from_row(body_row, quantity=-2.0 * contracts),
        _leg_from_row(far_row, quantity=contracts),
    )
    structure = Structure(name="broken_wing_butterfly", legs=legs, entry_spot=spot, entry_asof=asof, r=r, q=q)
    return _ok(structure, method)


# --------------------------------------------------------------------------
# Hedged / income overlays on shares
# --------------------------------------------------------------------------


def covered_call(
    ticker: str, asof, spot: float, direction: str, chain: pd.DataFrame, *,
    dte_min: int = 8, dte_max: int = 21, short_delta: float = 0.30,
    target_notional: float = 5000.0, r: float = 0.0, q: float = 0.0,
    vega_floor: float = DEFAULT_VEGA_FLOOR,
) -> Selection:
    _require_long_only("covered_call", direction)
    _validate_chain_columns(chain)
    asof_ts = _assert_chain_no_lookahead(chain, asof)
    if not (math.isfinite(spot) and spot > 0):
        raise ValueError(f"spot must be finite and > 0, got {spot}")

    contracts = max(1, round(target_notional / spot / 100.0))
    share_qty = float(contracts * 100)

    bucket = _filter_dte_bucket(chain, asof_ts, dte_min, dte_max)
    if bucket.empty:
        return _fail(REASON_NO_EXPIRY_IN_BUCKET)
    call_row, method = _select_by_delta_or_moneyness(
        bucket, "C", spot, asof_ts, r, q, target_delta=short_delta, vega_floor=vega_floor,
    )
    if call_row is None:
        return _fail(REASON_NO_STRIKE_NEAR_DELTA)
    if not _row_liquidity_ok(call_row):
        return _fail(REASON_LEG_ILLIQUID)

    share_leg = Leg(osi_symbol=f"{ticker.upper()}_SHARES", right="S", strike=None, expiry=None,
                     quantity=share_qty, entry_price=spot, multiplier=1)
    call_leg = _leg_from_row(call_row, quantity=-float(contracts))
    structure = Structure(name="covered_call", legs=(share_leg, call_leg), entry_spot=spot, entry_asof=asof, r=r, q=q)
    return _ok(structure, method)


def collar(
    ticker: str, asof, spot: float, direction: str, chain: pd.DataFrame, *,
    dte_min: int = 8, dte_max: int = 21, short_call_delta: float = 0.30, long_put_delta: float = 0.30,
    target_notional: float = 5000.0, r: float = 0.0, q: float = 0.0,
    vega_floor: float = DEFAULT_VEGA_FLOOR,
) -> Selection:
    """Covered call plus a protective long put -- caps both tails."""
    _require_long_only("collar", direction)
    _validate_chain_columns(chain)
    asof_ts = _assert_chain_no_lookahead(chain, asof)
    if not (math.isfinite(spot) and spot > 0):
        raise ValueError(f"spot must be finite and > 0, got {spot}")

    contracts = max(1, round(target_notional / spot / 100.0))
    share_qty = float(contracts * 100)

    bucket = _filter_dte_bucket(chain, asof_ts, dte_min, dte_max)
    if bucket.empty:
        return _fail(REASON_NO_EXPIRY_IN_BUCKET)
    call_row, call_method = _select_by_delta_or_moneyness(
        bucket, "C", spot, asof_ts, r, q, target_delta=short_call_delta, vega_floor=vega_floor,
    )
    put_row, put_method = _select_by_delta_or_moneyness(
        bucket, "P", spot, asof_ts, r, q, target_delta=long_put_delta, vega_floor=vega_floor,
    )
    if call_row is None or put_row is None:
        return _fail(REASON_NO_STRIKE_NEAR_DELTA)
    for row in (call_row, put_row):
        if not _row_liquidity_ok(row):
            return _fail(REASON_LEG_ILLIQUID)

    share_leg = Leg(osi_symbol=f"{ticker.upper()}_SHARES", right="S", strike=None, expiry=None,
                     quantity=share_qty, entry_price=spot, multiplier=1)
    call_leg = _leg_from_row(call_row, quantity=-float(contracts))
    put_leg = _leg_from_row(put_row, quantity=float(contracts))
    structure = Structure(
        name="collar", legs=(share_leg, call_leg, put_leg), entry_spot=spot, entry_asof=asof, r=r, q=q,
    )
    method = call_method if call_method == put_method else "mixed"
    return _ok(structure, method)


def cash_secured_put(
    ticker: str, asof, spot: float, direction: str, chain: pd.DataFrame, *,
    dte_min: int = 8, dte_max: int = 21, short_delta: float = 0.30,
    contracts: float = 1.0, r: float = 0.0, q: float = 0.0, vega_floor: float = DEFAULT_VEGA_FLOOR,
) -> Selection:
    _require_long_only("cash_secured_put", direction)
    _validate_chain_columns(chain)
    asof_ts = _assert_chain_no_lookahead(chain, asof)
    bucket = _filter_dte_bucket(chain, asof_ts, dte_min, dte_max)
    if bucket.empty:
        return _fail(REASON_NO_EXPIRY_IN_BUCKET)
    row, method = _select_by_delta_or_moneyness(
        bucket, "P", spot, asof_ts, r, q, target_delta=short_delta, vega_floor=vega_floor,
    )
    if row is None:
        return _fail(REASON_NO_STRIKE_NEAR_DELTA)
    if not _row_liquidity_ok(row):
        return _fail(REASON_LEG_ILLIQUID)
    leg = _leg_from_row(row, quantity=-contracts)
    structure = Structure(name="cash_secured_put", legs=(leg,), entry_spot=spot, entry_asof=asof, r=r, q=q)
    return _ok(structure, method)


# --------------------------------------------------------------------------
# Volatility-structure strategies
# --------------------------------------------------------------------------


def straddle(
    ticker: str, asof, spot: float, direction: str, chain: pd.DataFrame, *,
    dte_min: int = 8, dte_max: int = 21, contracts: float = 1.0,
    r: float = 0.0, q: float = 0.0,
) -> Selection:
    """Same-strike (nearest to spot), same-expiry call + put.
    direction='long' buys both (long vol), 'short' sells both (short vol)."""
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
    _validate_chain_columns(chain)
    asof_ts = _assert_chain_no_lookahead(chain, asof)
    bucket = _filter_dte_bucket(chain, asof_ts, dte_min, dte_max)
    if bucket.empty:
        return _fail(REASON_NO_EXPIRY_IN_BUCKET)

    calls = bucket[bucket["right"] == "C"]
    if calls.empty:
        return _fail(REASON_NO_EXPIRY_IN_BUCKET)
    call_row = _nearest_strike_to(calls, spot)
    puts_same_expiry = bucket[(bucket["right"] == "P") & (bucket["expiry"] == call_row["expiry"])]
    put_row = _nearest_strike_to(puts_same_expiry, float(call_row["strike"]))
    if put_row is None or float(put_row["strike"]) != float(call_row["strike"]):
        return _fail(REASON_NO_MATCHING_PAIR)

    for row in (call_row, put_row):
        if not _row_liquidity_ok(row):
            return _fail(REASON_LEG_ILLIQUID)

    sign = 1.0 if direction == "long" else -1.0
    legs = (
        _leg_from_row(call_row, quantity=sign * contracts),
        _leg_from_row(put_row, quantity=sign * contracts),
    )
    name = "long_straddle" if direction == "long" else "short_straddle"
    structure = Structure(name=name, legs=legs, entry_spot=spot, entry_asof=asof, r=r, q=q)
    return _ok(structure, method=None)


def strangle(
    ticker: str, asof, spot: float, direction: str, chain: pd.DataFrame, *,
    dte_min: int = 8, dte_max: int = 21, call_delta: float = 0.30, put_delta: float = 0.30,
    contracts: float = 1.0, r: float = 0.0, q: float = 0.0, vega_floor: float = DEFAULT_VEGA_FLOOR,
) -> Selection:
    """OTM call + OTM put, same expiry. direction='long' buys both (long
    vol), 'short' sells both (short vol, undefined risk)."""
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
    _validate_chain_columns(chain)
    asof_ts = _assert_chain_no_lookahead(chain, asof)
    bucket = _filter_dte_bucket(chain, asof_ts, dte_min, dte_max)
    if bucket.empty:
        return _fail(REASON_NO_EXPIRY_IN_BUCKET)

    call_row, call_method = _select_by_delta_or_moneyness(
        bucket, "C", spot, asof_ts, r, q, target_delta=call_delta, vega_floor=vega_floor,
    )
    if call_row is None:
        return _fail(REASON_NO_STRIKE_NEAR_DELTA)
    puts_same_expiry = bucket[(bucket["right"] == "P") & (bucket["expiry"] == call_row["expiry"])]
    if puts_same_expiry.empty:
        return _fail(REASON_NO_MATCHING_PAIR)
    put_row, put_method = _select_by_delta_or_moneyness(
        puts_same_expiry, "P", spot, asof_ts, r, q, target_delta=put_delta, vega_floor=vega_floor,
    )
    if put_row is None:
        return _fail(REASON_NO_MATCHING_PAIR)

    for row in (call_row, put_row):
        if not _row_liquidity_ok(row):
            return _fail(REASON_LEG_ILLIQUID)

    sign = 1.0 if direction == "long" else -1.0
    legs = (
        _leg_from_row(call_row, quantity=sign * contracts),
        _leg_from_row(put_row, quantity=sign * contracts),
    )
    name = "long_strangle" if direction == "long" else "short_strangle"
    structure = Structure(name=name, legs=legs, entry_spot=spot, entry_asof=asof, r=r, q=q)
    method = call_method if call_method == put_method else "mixed"
    return _ok(structure, method)


def calendar_spread(
    ticker: str, asof, spot: float, direction: str, chain: pd.DataFrame, *,
    near_dte_min: int = 8, near_dte_max: int = 21, far_dte_min: int = 46, far_dte_max: int = 90,
    target_delta: float = 0.50, contracts: float = 1.0, r: float = 0.0, q: float = 0.0,
    vega_floor: float = DEFAULT_VEGA_FLOOR,
) -> Selection:
    """Long calendar: sell the near-expiry leg, buy the far-expiry leg at
    the SAME strike, same right. Calls for direction='long', puts for
    direction='short'."""
    right = _direction_right(direction)
    _validate_chain_columns(chain)
    asof_ts = _assert_chain_no_lookahead(chain, asof)
    near_bucket = _filter_dte_bucket(chain, asof_ts, near_dte_min, near_dte_max)
    far_bucket = _filter_dte_bucket(chain, asof_ts, far_dte_min, far_dte_max)
    if near_bucket.empty or far_bucket.empty:
        return _fail(REASON_NO_EXPIRY_IN_BUCKET)

    near_row, near_method = _select_by_delta_or_moneyness(
        near_bucket, right, spot, asof_ts, r, q, target_delta=target_delta, vega_floor=vega_floor,
    )
    if near_row is None:
        return _fail(REASON_NO_STRIKE_NEAR_DELTA)

    far_same_strike = far_bucket[(far_bucket["right"] == right) & (far_bucket["strike"] == near_row["strike"])]
    if far_same_strike.empty:
        return _fail(REASON_NO_MATCHING_PAIR)
    far_row = far_same_strike.iloc[0]

    for row in (near_row, far_row):
        if not _row_liquidity_ok(row):
            return _fail(REASON_LEG_ILLIQUID)

    legs = (
        _leg_from_row(near_row, quantity=-contracts),
        _leg_from_row(far_row, quantity=contracts),
    )
    structure = Structure(name="calendar_spread", legs=legs, entry_spot=spot, entry_asof=asof, r=r, q=q)
    return _ok(structure, near_method)


def diagonal_spread(
    ticker: str, asof, spot: float, direction: str, chain: pd.DataFrame, *,
    near_dte_min: int = 8, near_dte_max: int = 21, far_dte_min: int = 46, far_dte_max: int = 90,
    near_delta: float = 0.30, far_delta: float = 0.50, contracts: float = 1.0,
    r: float = 0.0, q: float = 0.0, vega_floor: float = DEFAULT_VEGA_FLOOR,
) -> Selection:
    """Diagonal: sell the near-expiry leg at `near_delta`, buy the far-
    expiry leg at `far_delta` (independent strikes, unlike `calendar_spread`).
    Calls for direction='long', puts for direction='short'."""
    right = _direction_right(direction)
    _validate_chain_columns(chain)
    asof_ts = _assert_chain_no_lookahead(chain, asof)
    near_bucket = _filter_dte_bucket(chain, asof_ts, near_dte_min, near_dte_max)
    far_bucket = _filter_dte_bucket(chain, asof_ts, far_dte_min, far_dte_max)
    if near_bucket.empty or far_bucket.empty:
        return _fail(REASON_NO_EXPIRY_IN_BUCKET)

    near_row, near_method = _select_by_delta_or_moneyness(
        near_bucket, right, spot, asof_ts, r, q, target_delta=near_delta, vega_floor=vega_floor,
    )
    far_row, far_method = _select_by_delta_or_moneyness(
        far_bucket, right, spot, asof_ts, r, q, target_delta=far_delta, vega_floor=vega_floor,
    )
    if near_row is None or far_row is None:
        return _fail(REASON_NO_STRIKE_NEAR_DELTA)

    for row in (near_row, far_row):
        if not _row_liquidity_ok(row):
            return _fail(REASON_LEG_ILLIQUID)

    legs = (
        _leg_from_row(near_row, quantity=-contracts),
        _leg_from_row(far_row, quantity=contracts),
    )
    structure = Structure(name="diagonal_spread", legs=legs, entry_spot=spot, entry_asof=asof, r=r, q=q)
    method = near_method if near_method == far_method else "mixed"
    return _ok(structure, method)


def ratio_backspread(
    ticker: str, asof, spot: float, direction: str, chain: pd.DataFrame, *,
    dte_min: int = 8, dte_max: int = 21, short_delta: float = 0.50, long_delta: float = 0.25,
    ratio: int = 2, contracts: float = 1.0, r: float = 0.0, q: float = 0.0,
    vega_floor: float = DEFAULT_VEGA_FLOOR,
) -> Selection:
    """Sell `contracts` near-the-money (target `short_delta`), buy
    `ratio * contracts` further OTM (target `long_delta`), same expiry, same
    right. Calls for direction='long' (bullish backspread), puts for
    direction='short' (bearish backspread). Undefined-risk on the side with
    net short exposure if `ratio` doesn't fully offset the short leg."""
    if ratio < 2:
        raise ValueError(f"ratio_backspread requires ratio >= 2 (else it's just a vertical), got {ratio}")
    right = _direction_right(direction)
    _validate_chain_columns(chain)
    asof_ts = _assert_chain_no_lookahead(chain, asof)
    bucket = _filter_dte_bucket(chain, asof_ts, dte_min, dte_max)
    if bucket.empty:
        return _fail(REASON_NO_EXPIRY_IN_BUCKET)

    short_row, short_method = _select_by_delta_or_moneyness(
        bucket, right, spot, asof_ts, r, q, target_delta=short_delta, vega_floor=vega_floor,
    )
    if short_row is None:
        return _fail(REASON_NO_STRIKE_NEAR_DELTA)
    same_expiry = bucket[(bucket["right"] == right) & (bucket["expiry"] == short_row["expiry"])]
    long_row, long_method = _select_by_delta_or_moneyness(
        same_expiry, right, spot, asof_ts, r, q, target_delta=long_delta, vega_floor=vega_floor,
    )
    if long_row is None or long_row["osi_symbol"] == short_row["osi_symbol"]:
        return _fail(REASON_NO_STRIKE_NEAR_DELTA)

    for row in (short_row, long_row):
        if not _row_liquidity_ok(row):
            return _fail(REASON_LEG_ILLIQUID)

    legs = (
        _leg_from_row(short_row, quantity=-contracts),
        _leg_from_row(long_row, quantity=float(ratio) * contracts),
    )
    structure = Structure(name="ratio_backspread", legs=legs, entry_spot=spot, entry_asof=asof, r=r, q=q)
    method = short_method if short_method == long_method else "mixed"
    return _ok(structure, method)
