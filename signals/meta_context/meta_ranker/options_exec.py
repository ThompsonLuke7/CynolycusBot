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
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np
from pydantic import ValidationError

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
from core.nervous_system.execution.options.quotes import (
    OptionQuote,
    QuoteError,
    parse_occ_symbol,
)
from core.calendar import is_market_open_now
from core.option_liquidity import contract_liquidity
from strategies.dealer_positioning.gate import (
    SCOPE_MONTHLY,
    evaluate_dealer_gate,
    gate_enabled,
)
from strategies.multi_ticker_swing.live.runner import (
    _DELTA_HI,
    _DELTA_LO,
    _DELTA_TGT,
    _contract_strike,
    _contract_symbol,
    _is_standard_100_contract,
)

_ET = ZoneInfo("America/New_York")


def equity_order_tif(now: datetime | None = None) -> str:
    """Time-in-force for EQUITY market orders.

    Returns 'opg' (market-on-open) when the US equity market is CLOSED at
    submission time, else 'day' (fill now). This makes the post-close 2nd-4H-bar
    entries fill deterministically at the official opening auction — the next-open
    fill that backtest_exits.py --entry next_open validated (edge intact, ~0.17%
    mean/trade erosion). During RTH the 1st-bar pass stays 'day' and fills at once.

    NOTE: options orders do NOT support 'opg' on Alpaca (day only); a day options
    order placed after hours already queues to the next open, so the options path
    keeps 'day' and does not call this.
    """
    return "day" if is_market_open_now(now) else "opg"


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


def _quote_timestamp(payload: dict) -> datetime | None:
    """Read the moment the market was observed, or nothing.

    Deliberately never falls back to the current clock. An undated quote is
    indistinguishable from a stale one, and stamping it with `now` would make a
    stale market look fresh — the exact failure that invalidated the 2026-07
    options study.
    """

    for key in ("t", "timestamp", "quote_at"):
        raw = payload.get(key)
        if raw is None:
            continue
        if isinstance(raw, datetime):
            parsed = raw
        else:
            text = str(raw).strip()
            if not text:
                continue
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                continue
        if parsed.tzinfo is None:
            # A naive broker timestamp has no defined instant; refuse it rather
            # than assume a zone.
            continue
        return parsed.astimezone(timezone.utc)
    return None


def _parse_two_sided(payload: dict) -> tuple[float, float, datetime] | None:
    """Return a dated two-sided market, or nothing at all."""

    if not isinstance(payload, dict):
        return None
    bid = _as_float(payload.get("bp", payload.get("bid_price")))
    ask = _as_float(payload.get("ap", payload.get("ask_price")))
    if not (bid > 0 and ask > 0):
        return None
    quote_at = _quote_timestamp(payload)
    if quote_at is None:
        return None
    return bid, ask, quote_at


def _latest_quote(
    client: AlpacaOptionsClient, occ: str
) -> tuple[tuple[float, float, datetime] | None, str]:
    """Refetch one quote. Returns (quote, reason); reason is 'ok' on success.

    "No two-sided market" and "a market we cannot date" are different failures
    and are reported separately: the first is an untradeable contract, the
    second is a data-integrity problem worth seeing in the audit trail.
    """

    try:
        resp = client.get_option_quotes(symbols=occ)
    except Exception:
        return None, "no_quote"
    quotes = resp.get("quotes", resp) if isinstance(resp, dict) else None
    if not isinstance(quotes, dict):
        return None, "no_quote"
    q = quotes.get(occ) or (next(iter(quotes.values())) if quotes else None)
    if not isinstance(q, dict):
        return None, "no_quote"
    bid = _as_float(q.get("bp", q.get("bid_price")))
    ask = _as_float(q.get("ap", q.get("ask_price")))
    if not (bid > 0 and ask > 0):
        return None, "no_quote"
    quote_at = _quote_timestamp(q)
    if quote_at is None:
        return None, "no_quote_timestamp"
    return (bid, ask, quote_at), "ok"


def select_option(
    client: AlpacaOptionsClient,
    ticker: str,
    current_price: float,
    per_name_usd: float,
    *,
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
    occ, dlt, sel_snap = min(in_range, key=lambda x: abs(x[1] - _DELTA_TGT))

    # Prefer the snapshot's own quote (saves a second API call / 429s); fall back
    # to a fresh quote fetch only if the snapshot lacks a dated two-sided quote.
    lq = sel_snap.get("latestQuote") or {}
    observed = _parse_two_sided(lq)
    if observed is None:
        snapshot_bid = _as_float(lq.get("bp", lq.get("bid_price")))
        snapshot_ask = _as_float(lq.get("ap", lq.get("ask_price")))
        if snapshot_bid > 0 and snapshot_ask > 0:
            # The market itself was fine; only its timestamp was missing. That
            # is a data problem, not an untradeable contract, so say so instead
            # of burning a refetch that will not add a timestamp.
            return None, "no_quote_timestamp"
        observed, reason = _latest_quote(client, occ)
        if observed is None:
            return None, reason
    bid, ask, quote_at = observed
    if ask < bid:
        return None, "crossed_quote"
    mid = (bid + ask) / 2.0
    # Spread is reported for the audit only — it is NOT a gate. Liquidity is
    # judged by open interest + volume (see route_option_or_shares); a wide %%
    # spread on a cheap contract (e.g. 0.50/1.00) is normal and not disqualifying.
    spread = (ask - bid) / mid if mid > 0 else 1.0
    # Alpaca's option snapshot carries no open interest or daily volume, so
    # reading them off `sel_snap` always yielded 0 and force-routed every name to
    # shares. Schwab's chain has both. `None` here means UNKNOWN, not zero — the
    # routing gate must be able to tell those apart.
    strike_value = _contract_strike(by_symbol[occ])
    liq = contract_liquidity(ticker, expiry=exp_str, strike=strike_value, option_type="C")
    oi = liq.open_interest if liq is not None else None
    vol = liq.volume if liq is not None else None
    contracts_n = int(math.floor(per_name_usd / (mid * 100.0)))
    if contracts_n < 1:
        return None, "budget_lt_1_contract"
    try:
        quote = _option_quote(
            occ,
            underlying=ticker.upper(),
            bid=bid,
            ask=ask,
            quote_at=quote_at,
            delta=dlt,
            open_interest=oi,
            volume=vol,
        )
    except (QuoteError, ValidationError) as exc:
        # The governed path will not accept an instrument we cannot describe
        # exactly, so neither does this one.
        return None, f"invalid_quote({exc.__class__.__name__})"
    return (
        {
            "ticker": ticker, "occ": occ, "contracts": contracts_n, "mid": mid,
            "limit": round(ask, 2), "delta": dlt, "expiry": exp_str,
            "strike": strike_value, "notional": contracts_n * mid * 100.0,
            "open_interest": oi, "volume": vol, "spread": spread,
            "liquidity_source": liq.source if liq is not None else "unavailable",
            # The validated mark. Everything above is the legacy audit view of
            # the same numbers; this is what the governed path builds legs from.
            "quote": quote,
        },
        "ok",
    )


def _option_quote(
    occ: str,
    *,
    underlying: str,
    bid: float,
    ask: float,
    quote_at: datetime,
    delta: float | None,
    open_interest: int | None,
    volume: int | None,
) -> OptionQuote:
    """Build the contract-level mark, deriving identity from the OCC symbol.

    `last_trade_price` is deliberately never populated: a trade print is not a
    mark, and OptionQuote keeps the two strictly apart.
    """

    identity = parse_occ_symbol(occ)
    return OptionQuote(
        symbol=occ,
        underlying=underlying,
        option_type=identity.option_type,
        strike=identity.strike,
        expiration=identity.expiration,
        quote_at=quote_at,
        bid=Decimal(str(bid)),
        ask=Decimal(str(ask)),
        delta=None if delta is None else Decimal(str(delta)),
        open_interest=open_interest,
        volume=volume,
    )


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# Optionability gate: names that aren't good option candidates trade shares
# instead. Liquidity is judged by open interest + volume only (no spread gate).
ROUTE_PRICE_FLOOR = 10.0      # underlying < $10 -> shares
ROUTE_MIN_OPEN_INTEREST = 500  # selected contract OI < 500 -> shares
ROUTE_MIN_VOLUME = 100         # selected contract daily volume < 100 -> shares


def route_option_or_shares(
    client: AlpacaOptionsClient,
    ticker: str,
    current_price: float,
    *,
    roll_trading_days: int = 5,
    price_floor: float = ROUTE_PRICE_FLOOR,
    min_open_interest: int = ROUTE_MIN_OPEN_INTEREST,
    min_volume: int = ROUTE_MIN_VOLUME,
    now_et: datetime | None = None,
    dealer_scope: str = SCOPE_MONTHLY,
) -> tuple[str, dict | None, str]:
    """Decide whether to trade a name as OPTIONS or SHARES.

    Returns (route, order, reason) where route is 'option' or 'equity'.
      * 'option' -> `order` is the select_option dict (10 contracts by caller).
      * 'equity' -> trade shares (100 by caller); `order` may be None or the
        rejected contract (for audit). Reasons: underlying_lt_price_floor,
        no_option(<select reason>), illiquid_option(oi=..,vol=..),
        dealer_veto(<reason>).

    The 4H modules are long/calls-only, so the dealer gate is evaluated on the
    call side against the monthly (`through_month`) scope. A veto falls back to
    SHARES (keeps directional exposure, drops leveraged premium into a wall).
    The verdict is always attached to `order['dealer_gate']` for audit; it only
    changes routing when DEALER_GATE_ENABLED is set (observe-first).
    """
    if current_price is not None and current_price < price_floor:
        return "equity", None, f"underlying_lt_{price_floor:.0f}"
    order, reason = select_option(
        client, ticker, current_price, 1e12,
        roll_trading_days=roll_trading_days, now_et=now_et,
    )
    if order is None:
        return "equity", None, f"no_option({reason})"
    oi, vol = order.get("open_interest"), order.get("volume")
    if oi is None or vol is None:
        # Unverifiable liquidity is not the same as thin liquidity. Still route
        # shares (the gate exists precisely to avoid trading a contract we cannot
        # vet) but say so distinctly, so a broken data path is never mistaken for
        # a market with no open interest.
        return "equity", order, (
            f"liquidity_unavailable(src={order.get('liquidity_source', 'unknown')})"
        )
    if oi < min_open_interest or vol < min_volume:
        return "equity", order, f"illiquid_option(oi={oi},vol={vol})"
    verdict = evaluate_dealer_gate(ticker, "call", current_price, dealer_scope)
    order["dealer_gate"] = verdict.to_dict()
    if verdict.vetoed and gate_enabled():
        return "equity", order, f"dealer_veto({verdict.reason})"
    return "option", order, "ok"
