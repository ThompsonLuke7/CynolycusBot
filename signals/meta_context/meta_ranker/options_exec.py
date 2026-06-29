"""
Options execution helper for the Meta Ranker runner.

Contract policy (long-only -> calls):
  * EXPIRY: the next MONTHLY expiration (third Friday). Roll to the following month
    once the nearest monthly is within --roll-trading-days (default 5) trading days,
    so positions get 1-2 weeks to play out instead of decaying into expiry.
  * STRIKE: |delta| in [0.35, 0.60] (target 0.45) from live snapshot Greeks.
  * QUOTE: a real two-sided quote with spread <= --max-spread-pct.
A name failing any gate is SKIPPED (no untradeable junk).

Reuses the swing runner's contract parsing helpers for consistency.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
from strategies.multi_ticker_swing.live.runner import (
    _DELTA_HI,
    _DELTA_LO,
    _DELTA_TGT,
    _contract_strike,
    _contract_symbol,
    _is_standard_100_contract,
)

_ET = ZoneInfo("America/New_York")


def _third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    days_to_fri = (4 - first.weekday()) % 7  # weekday 4 = Friday
    return first + timedelta(days=days_to_fri) + timedelta(weeks=2)


def target_monthly_expiry(ref_date: date, roll_trading_days: int = 5) -> date:
    """Next monthly (3rd Friday); roll to the following month if the nearest is
    within `roll_trading_days` trading days of ref_date."""
    year, month = ref_date.year, ref_date.month
    for _ in range(13):
        exp = _third_friday(year, month)
        if exp >= ref_date:
            td = int(np.busday_count(ref_date, exp))  # trading days to expiry
            if td >= roll_trading_days:
                return exp
        month += 1
        if month > 12:
            month, year = 1, year + 1
    raise ValueError(f"Could not determine monthly expiry after {ref_date}")


def _latest_quote(client: AlpacaOptionsClient, occ: str) -> tuple[float, float] | None:
    try:
        resp = client.get_option_quotes(symbols=occ)
    except Exception:
        return None
    quotes = resp.get("quotes", resp) if isinstance(resp, dict) else None
    if not isinstance(quotes, dict):
        return None
    q = quotes.get(occ) or (next(iter(quotes.values())) if quotes else None)
    if not isinstance(q, dict):
        return None
    bid, ask = q.get("bp", q.get("bid_price")), q.get("ap", q.get("ask_price"))
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return None
    return (bid, ask) if bid > 0 and ask > 0 else None


def select_option(
    client: AlpacaOptionsClient,
    ticker: str,
    current_price: float,
    per_name_usd: float,
    *,
    max_spread_pct: float = 0.15,
    roll_trading_days: int = 5,
    now_et: datetime | None = None,
) -> tuple[dict | None, str]:
    """Pick a delta-filtered monthly call for `ticker`, sized to `per_name_usd`."""
    now_et = now_et or datetime.now(_ET)
    want_exp = target_monthly_expiry(now_et.date(), roll_trading_days)
    exp_str = want_exp.isoformat()

    # Contracts for the target monthly expiry only.
    try:
        resp = client.get_option_contracts(
            underlying_symbol=ticker.upper(), type="call",
            expiration_date=exp_str, limit=500,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"contracts_error({exc})"
    contracts = resp.get("option_contracts", resp) if isinstance(resp, dict) else resp
    if not contracts:
        return None, f"no_contracts_for_{exp_str}"
    contracts = [c for c in contracts if _is_standard_100_contract(c, ticker)]
    if not contracts:
        return None, "no_standard_contracts"
    by_symbol = {_contract_symbol(c): c for c in contracts}

    # Delta filter from snapshots for that expiry.
    try:
        snaps = client.get_option_snapshots(ticker, expiration_date=exp_str, type="call")
    except Exception as exc:  # noqa: BLE001
        return None, f"snapshots_error({exc})"
    cands = []
    for occ, snap in (snaps or {}).items():
        occ = str(occ).strip().upper()
        if occ not in by_symbol:
            continue
        d = (snap.get("greeks") or {}).get("delta")
        if d is None:
            continue
        cands.append((occ, abs(float(d)), snap))
    if not cands:
        return None, "no_greeks"
    in_range = [c for c in cands if _DELTA_LO <= c[1] <= _DELTA_HI]
    if not in_range:
        return None, "delta_filter_failed(out_of_range)"
    occ, dlt, _ = min(in_range, key=lambda x: abs(x[1] - _DELTA_TGT))

    quote = _latest_quote(client, occ)
    if quote is None:
        return None, "no_quote"
    bid, ask = quote
    mid = (bid + ask) / 2.0
    spread = (ask - bid) / mid if mid > 0 else 1.0
    if spread > max_spread_pct:
        return None, f"wide_spread({spread:.0%})"
    contracts_n = int(math.floor(per_name_usd / (mid * 100.0)))
    if contracts_n < 1:
        return None, "budget_lt_1_contract"
    return (
        {
            "ticker": ticker, "occ": occ, "contracts": contracts_n, "mid": mid,
            "limit": round(ask, 2), "delta": dlt, "expiry": exp_str,
            "strike": _contract_strike(by_symbol[occ]), "notional": contracts_n * mid * 100.0,
        },
        "ok",
    )
