"""Pure construction, validation, and P&L helpers for the multi-leg study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LegSpec:
    symbol: str
    right: str
    strike: float | None
    quantity: float
    expiry: str | None


def nearest_contract(contracts: list[dict], right: str, target: float) -> dict | None:
    rows = [
        row for row in contracts
        if str(row.get("type", "")).lower().startswith(right.lower())
        and _positive(row.get("strike_price"))
    ]
    if not rows:
        return None
    return min(rows, key=lambda row: (abs(float(row["strike_price"]) - target), float(row["strike_price"])))


def exact_contract(contracts: list[dict], right: str, strike: float) -> dict | None:
    rows = [
        row for row in contracts
        if str(row.get("type", "")).lower().startswith(right.lower())
        and _positive(row.get("strike_price"))
        and abs(float(row["strike_price"]) - strike) < 1e-9
    ]
    return rows[0] if rows else None


def _positive(value: object) -> bool:
    try:
        return np.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def leg(row: dict, quantity: float) -> LegSpec:
    return LegSpec(
        symbol=str(row["symbol"]),
        right="C" if str(row["type"]).lower().startswith("call") else "P",
        strike=float(row["strike_price"]),
        quantity=float(quantity),
        expiry=str(row["expiration_date"]),
    )


def share_leg(ticker: str, quantity: float = 100.0) -> LegSpec:
    return LegSpec(f"{ticker}_SHARES", "S", None, quantity, None)


def build_structures(
    ticker: str,
    spot: float,
    base_symbol: str,
    far_contracts: list[dict],
    near_contracts: list[dict] | None = None,
) -> dict[str, tuple[LegSpec, ...]]:
    """Build the fixed, pre-registered menu; omit structures with unavailable strikes."""
    near_contracts = near_contracts or []
    by_symbol = {str(row.get("symbol")): row for row in far_contracts}
    base = by_symbol.get(base_symbol)
    targets = {
        "atm_c": ("call", spot), "atm_p": ("put", spot),
        "c5": ("call", spot * 1.05), "c10": ("call", spot * 1.10),
        "c15": ("call", spot * 1.15), "p5": ("put", spot * 0.95),
        "p10": ("put", spot * 0.90),
    }
    picked = {name: nearest_contract(far_contracts, right, strike) for name, (right, strike) in targets.items()}
    out: dict[str, tuple[LegSpec, ...]] = {}

    def add(name: str, items: Iterable[tuple[dict | None, float] | LegSpec]) -> None:
        legs: list[LegSpec] = []
        for item in items:
            if isinstance(item, LegSpec):
                legs.append(item)
                continue
            row, qty = item
            if row is None:
                return
            legs.append(leg(row, qty))
        option_symbols = [x.symbol for x in legs if x.right != "S"]
        if len(option_symbols) != len(set(option_symbols)):
            return
        out[name] = tuple(legs)

    add("long_shares", [share_leg(ticker)])
    add("long_call", [(base, 1)])
    add("bull_call_debit", [(base, 1), (picked["c10"], -1)])
    add("bull_put_credit", [(picked["p5"], -1), (picked["p10"], 1)])
    add("call_backspread", [(picked["atm_c"], -1), (picked["c10"], 2)])
    add("call_butterfly", [(picked["atm_c"], 1), (picked["c5"], -2), (picked["c10"], 1)])
    add("broken_wing_call_butterfly", [(picked["atm_c"], 1), (picked["c5"], -2), (picked["c15"], 1)])
    add("covered_call", [share_leg(ticker), (picked["c10"], -1)])
    add("protective_collar", [share_leg(ticker), (picked["p10"], 1), (picked["c10"], -1)])
    add("long_straddle", [(picked["atm_c"], 1), (picked["atm_p"], 1)])
    add("long_strangle", [(picked["c5"], 1), (picked["p5"], 1)])
    add("iron_condor", [(picked["p10"], 1), (picked["p5"], -1), (picked["c5"], -1), (picked["c10"], 1)])
    add("iron_butterfly", [(picked["p10"], 1), (picked["atm_p"], -1), (picked["atm_c"], -1), (picked["c10"], 1)])

    if near_contracts:
        far_atm = picked["atm_c"]
        if far_atm is not None:
            k = float(far_atm["strike_price"])
            near_same = exact_contract(near_contracts, "call", k)
            add("call_calendar", [(near_same, -1), (far_atm, 1)])
            near_c5 = nearest_contract(near_contracts, "call", spot * 1.05)
            add("call_diagonal", [(near_c5, -1), (far_atm, 1)])
    return out


def validate_leg_bars(
    bars: list[dict], underlying: pd.DataFrame, right: str,
    *, min_bars: int = 6, min_coverage: float = 0.60,
    max_stale: float = 0.40, min_abs_corr: float = 0.50,
) -> tuple[bool, dict]:
    if len(bars) < min_bars:
        return False, {"reason": "too_few_bars", "n_bars": len(bars)}
    frame = pd.DataFrame(bars).copy()
    frame["date"] = pd.to_datetime(frame["t"], utc=True).dt.tz_convert("America/New_York").dt.date.astype(str)
    frame = frame.drop_duplicates("date", keep="last").set_index("date").sort_index()
    common = frame.index.intersection(underlying.index)
    if len(common) < min_bars:
        return False, {"reason": "too_few_overlap", "n_bars": len(common)}
    option_close = frame.loc[common, "c"].astype(float)
    stock_close = underlying.loc[common, "close"].astype(float)
    option_ret = option_close.pct_change().dropna()
    stock_ret = stock_close.pct_change().dropna()
    corr = float(option_ret.corr(stock_ret))
    span = underlying.loc[(underlying.index >= common[0]) & (underlying.index <= common[-1])]
    coverage = len(common) / max(1, len(span))
    stale = float(option_close.diff().fillna(1).eq(0).mean())
    expected_ok = corr >= min_abs_corr if right == "C" else corr <= -min_abs_corr
    stats = {"n_bars": len(common), "coverage": coverage, "stale_share": stale, "corr": corr}
    if coverage < min_coverage:
        return False, {**stats, "reason": "low_coverage"}
    if not np.isfinite(corr) or not expected_ok:
        return False, {**stats, "reason": "corr_wrong_or_low"}
    if stale > max_stale:
        return False, {**stats, "reason": "too_stale"}
    return True, stats


def terminal_payoff(legs: tuple[LegSpec, ...], spot: np.ndarray) -> np.ndarray:
    payoff = np.zeros_like(spot, dtype=float)
    for item in legs:
        if item.right == "S":
            payoff += item.quantity * spot
        elif item.right == "C":
            payoff += item.quantity * 100.0 * np.maximum(spot - float(item.strike), 0.0)
        else:
            payoff += item.quantity * 100.0 * np.maximum(float(item.strike) - spot, 0.0)
    return payoff


def capital_required(legs: tuple[LegSpec, ...], entry_prices: dict[str, float], entry_spot: float) -> tuple[float | None, str]:
    cash = 0.0
    expiries = set()
    for item in legs:
        if item.right == "S":
            cash += item.quantity * entry_spot
        else:
            cash += item.quantity * 100.0 * entry_prices[item.symbol]
            expiries.add(item.expiry)
    if len(expiries) > 1:
        return (cash if cash > 0 else None), "calendar_net_debit_approximation"
    strikes = [float(x.strike) for x in legs if x.strike is not None]
    hi = max([entry_spot * 5.0 + 100.0] + [x * 5.0 + 100.0 for x in strikes])
    grid = np.unique(np.concatenate([np.linspace(0.0, hi, 5001), np.asarray(strikes)]))
    initial_value = sum(
        x.quantity * (entry_spot if x.right == "S" else 100.0 * entry_prices[x.symbol]) for x in legs
    )
    pnl = terminal_payoff(legs, grid) - initial_value
    tail_slope = sum(x.quantity * (1.0 if x.right == "S" else 100.0) for x in legs if x.right in ("S", "C"))
    if tail_slope < -1e-9:
        return None, "unbounded"
    bound = float(max(0.0, -np.min(pnl)))
    # Sparse daily trade prints from different moments can violate vertical
    # no-arbitrage bounds and fabricate a free/near-free structure. Such a
    # denominator creates absurd returns and is evidence the bars cannot price
    # the combination, not an opportunity.
    if bound < 1.0:
        return None, "near_zero_or_arbitrage_bound"
    return bound, "exact_expiry_bound"


def structure_return(
    legs: tuple[LegSpec, ...], entry_prices: dict[str, float], exit_prices: dict[str, float],
    entry_spot: float, exit_spot: float, half_spread: float, commission: float = 0.65,
) -> tuple[float | None, dict]:
    capital, basis = capital_required(legs, entry_prices, entry_spot)
    if capital is None or capital <= 0:
        return None, {"reason": basis}
    gross = 0.0
    cost = 0.0
    for item in legs:
        if item.right == "S":
            gross += item.quantity * (exit_spot - entry_spot)
            continue
        gross += item.quantity * 100.0 * (exit_prices[item.symbol] - entry_prices[item.symbol])
        contracts = abs(item.quantity)
        cost += contracts * (2.0 * half_spread * 100.0 + 2.0 * commission)
    return (gross - cost) / capital, {"gross_pnl": gross, "cost": cost, "capital": capital, "capital_basis": basis}


def week_block_ci(differences: pd.DataFrame, *, draws: int = 10_000, seed: int = 20260904) -> tuple[float, float]:
    grouped = differences.groupby("week_key", sort=True)["difference"].mean().dropna().to_numpy()
    if len(grouped) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(grouped), size=(draws, len(grouped)))
    means = grouped[idx].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))
