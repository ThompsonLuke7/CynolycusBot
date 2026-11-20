"""Read-only Schwab holdings for the personal portfolio page.

Calls exactly two GET endpoints through schwab-py: ``get_accounts`` (balances +
positions for every linked account) and ``get_quotes`` (marks, greeks, and
descriptions). There is no order code here, and a test pins that.

Account numbers are masked to their last 3 digits before anything leaves this
module. Schwab account hashes are never returned.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any

OCC_RE = re.compile(r"^(?P<root>[A-Z0-9./]{1,6})\s*(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<cp>[CP])(?P<strike>\d{8})$")
_KIND = {"OPTION": "option", "EQUITY": "equity", "COLLECTIVE_INVESTMENT": "etf",
         "CASH_EQUIVALENT": "cash", "MUTUAL_FUND": "fund", "FIXED_INCOME": "bond"}
_BALANCES = {"liquidation_value": "liquidationValue", "equity": "equity", "cash_balance": "cashBalance",
             "available_funds": "availableFunds", "buying_power": "buyingPower",
             "maintenance_requirement": "maintenanceRequirement", "margin_balance": "marginBalance",
             "long_market_value": "longMarketValue", "long_option_market_value": "longOptionMarketValue",
             "short_market_value": "shortMarketValue", "money_market_fund": "moneyMarketFund"}
QUOTE_CHUNK = 100


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def mask_account_number(number: Any) -> str:
    text = str(number or "").strip()
    return f"…{text[-3:]}" if text else "…"


def parse_occ_symbol(symbol: str) -> dict | None:
    """``"SOXL  261218C00070000"`` -> root SOXL, expiry 2026-12-18, CALL, strike 70.0."""
    m = OCC_RE.match(str(symbol or "").strip())
    if not m:
        return None
    expiry = date(2000 + int(m["yy"]), int(m["mm"]), int(m["dd"]))
    return {"root": m["root"], "expiry": expiry.isoformat(),
            "put_call": "CALL" if m["cp"] == "C" else "PUT", "strike": int(m["strike"]) / 1000.0}


def normalize_position(raw: dict) -> dict:
    inst = raw.get("instrument") or {}
    asset = str(inst.get("assetType") or "").upper()
    long_q, short_q = _num(raw.get("longQuantity")) or 0.0, _num(raw.get("shortQuantity")) or 0.0
    qty = long_q - short_q
    avg = _num(raw.get("averageLongPrice") if qty >= 0 else raw.get("averageShortPrice"))
    if avg is None:
        avg = _num(raw.get("averagePrice"))
    open_pl = _num(raw.get("longOpenProfitLoss") if qty >= 0 else raw.get("shortOpenProfitLoss"))
    pos = {
        "symbol": str(inst.get("symbol") or "").strip(),
        "raw_symbol": str(inst.get("symbol") or ""),
        "asset_type": asset,
        "kind": _KIND.get(asset, "other"),
        "description": inst.get("description"),
        "quantity": qty,
        "average_price": avg,
        "market_value": _num(raw.get("marketValue")),
        "day_pl": _num(raw.get("currentDayProfitLoss")),
        "day_pl_pct": _num(raw.get("currentDayProfitLossPercentage")),
        "open_pl": open_pl,
        "multiplier": 1.0,
    }
    if pos["kind"] == "option":
        occ = parse_occ_symbol(pos["raw_symbol"]) or {}
        pos.update({"underlying": str(inst.get("underlyingSymbol") or occ.get("root") or "").strip() or None,
                    "put_call": inst.get("putCall") or occ.get("put_call"),
                    "strike": occ.get("strike"), "expiry": occ.get("expiry"), "multiplier": 100.0})
    cost = avg * qty * pos["multiplier"] if avg is not None else None
    pos["cost_basis"] = cost
    pos["open_pl_pct"] = (open_pl / abs(cost)) if open_pl is not None and cost else None
    return pos


def normalize_accounts(raw_accounts: list[dict]) -> list[dict]:
    if not isinstance(raw_accounts, list):
        raise ValueError(f"Schwab accounts response is not a list: {type(raw_accounts).__name__}")
    out = []
    for item in raw_accounts:
        sa = (item or {}).get("securitiesAccount")
        if not isinstance(sa, dict):
            raise ValueError("Schwab account entry has no securitiesAccount")
        cur = sa.get("currentBalances") or {}
        out.append({
            "account": mask_account_number(sa.get("accountNumber")),
            "type": sa.get("type"),
            "is_day_trader": bool(sa.get("isDayTrader")),
            "closing_only": bool(sa.get("isClosingOnlyRestricted")),
            "balances": {k: _num(cur.get(v)) for k, v in _BALANCES.items()},
            "positions": [normalize_position(p) for p in sa.get("positions") or []],
        })
    return out


def _quote_fields(q: dict) -> dict:
    quote, ref = q.get("quote") or {}, q.get("reference") or {}
    out = {"mark": _num(quote.get("mark")), "last": _num(quote.get("lastPrice")),
           "bid": _num(quote.get("bidPrice")), "ask": _num(quote.get("askPrice")),
           "close": _num(quote.get("closePrice")), "net_pct": _num(quote.get("netPercentChange")),
           "description": ref.get("description"), "asset_main_type": q.get("assetMainType"),
           "asset_sub_type": q.get("assetSubType"),
           "quote_time": quote.get("quoteTime")}
    if q.get("assetMainType") == "OPTION":
        out.update({k: _num(quote.get(src)) for k, src in (
            ("delta", "delta"), ("gamma", "gamma"), ("theta", "theta"), ("vega", "vega"),
            ("iv", "volatility"), ("underlying_price", "underlyingPrice"), ("time_value", "timeValue"),
            ("intrinsic", "moneyIntrinsicValue"), ("open_interest", "openInterest"))})
        out.update({"dte": _num(ref.get("daysToExpiration")), "multiplier": _num(ref.get("multiplier")),
                    "strike": _num(ref.get("strikePrice")), "exercise_type": ref.get("exerciseType")})
    return out


def enrich_positions(accounts: list[dict], quotes: dict[str, dict]) -> None:
    """Attach live quote fields in place. A missing quote leaves the fields absent (shown as unknown)."""
    for acct in accounts:
        for pos in acct["positions"]:
            q = quotes.get(pos["raw_symbol"]) or quotes.get(pos["symbol"])
            if not q:
                continue
            pos["quote"] = q
            if pos["kind"] == "option":
                pos["multiplier"] = q.get("multiplier") or pos["multiplier"]
                if q.get("strike") is not None:
                    pos["strike"] = q["strike"]


def _get_json(resp, what: str) -> Any:
    if getattr(resp, "status_code", None) != 200:
        body = getattr(resp, "text", "")[:300]
        raise RuntimeError(f"Schwab {what} failed: HTTP {getattr(resp, 'status_code', '?')} {body}")
    return resp.json()


def fetch_holdings(client=None, extra_symbols: list[str] | tuple[str, ...] = ()) -> dict:
    """Balances, positions, and quotes for every linked account (GET only).

    ``extra_symbols`` (watchlist names) ride along in the same quotes call so
    the page shows a live mark for them without another request per name.
    """
    if client is None:
        from core.API.Schwab_API.schwab_client import SchwabClient
        client = SchwabClient().client
    fields = [client.Account.Fields.POSITIONS]
    accounts = normalize_accounts(_get_json(client.get_accounts(fields=fields), "get_accounts"))
    held = {p["raw_symbol"] for a in accounts for p in a["positions"] if p["raw_symbol"]}
    unders = {p["underlying"] for a in accounts for p in a["positions"] if p.get("underlying")}
    symbols = sorted(held | unders | {str(s).upper() for s in extra_symbols if s})
    quotes: dict[str, dict] = {}
    for i in range(0, len(symbols), QUOTE_CHUNK):
        raw = _get_json(client.get_quotes(symbols[i:i + QUOTE_CHUNK]), "get_quotes")
        quotes.update({sym: _quote_fields(q) for sym, q in (raw or {}).items() if isinstance(q, dict)})
    enrich_positions(accounts, quotes)
    return {"fetched_at": datetime.now(timezone.utc).isoformat(), "accounts": accounts, "quotes": quotes}
