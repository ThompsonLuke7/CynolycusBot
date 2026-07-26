"""Per-contract option liquidity (open interest + daily volume) from Schwab.

Why this exists
---------------
The 4H modules and the Dealer Ranker gate option routing on the selected
contract's open interest and volume. They read those two numbers off Alpaca's
``/v1beta1/options/snapshots`` payload, which only carries ``greeks``,
``impliedVolatility``, ``latestQuote`` and ``latestTrade`` — there is no open
interest and no daily volume in it. Both lookups therefore coerced to 0 on every
contract, the floors never passed, and every candidate silently fell back to
shares: across the full recorded live history the routing tally was 294 equity
and 0 option, with all 181 liquidity rejects reading the identical
``illiquid_option(oi=0,vol=0)``.

Schwab's chain endpoint does carry ``openInterest``/``totalVolume``, and the
repo already holds a Schwab client for dealer positioning. This module fetches
just the one expiry a caller needs, caches it briefly (a 4H pass asks for ~10
names, and the same expiry repeats across names within a pass), and returns
``None`` when the number genuinely could not be determined so callers never
mistake "unknown" for "zero".
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)

# A chain snapshot is reused inside a single decision pass but must not go stale
# across passes; OI updates once daily and volume accumulates intraday.
CACHE_TTL_SECONDS = 900.0
_STRIKE_TOLERANCE = 1e-4


@dataclass(frozen=True)
class ContractLiquidity:
    open_interest: int
    volume: int
    source: str


_cache: dict[tuple[str, str], tuple[float, dict[tuple[str, float], ContractLiquidity]]] = {}
_cache_lock = threading.Lock()
_client_lock = threading.Lock()
_client: Any = None
_client_failed = False


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # reject NaN


def _normalize_expiry(expiry: Any) -> str:
    if isinstance(expiry, (date, datetime)):
        return expiry.strftime("%Y-%m-%d")
    text = str(expiry).strip()
    return text[:10]


def _normalize_type(option_type: Any) -> str:
    text = str(option_type or "").strip().upper()
    if text.startswith("P"):
        return "P"
    return "C"


def _get_raw_client() -> Any:
    """Lazily build the Schwab client once. A failure is remembered so a broken
    or unauthorized token does not re-prompt or re-raise on every contract."""
    global _client, _client_failed
    with _client_lock:
        if _client is not None or _client_failed:
            return _client
        try:
            from core.API.Schwab_API.schwab_client import SchwabClient

            raw = getattr(SchwabClient(), "client", None)
            if raw is None or not hasattr(raw, "get_option_chain"):
                raise RuntimeError("Schwab client does not expose get_option_chain")
            _client = raw
        except Exception as exc:  # noqa: BLE001
            _client_failed = True
            logger.warning("option liquidity: Schwab chain unavailable (%s)", exc)
            return None
        return _client


def _parse_chain(payload: dict[str, Any]) -> dict[tuple[str, float], ContractLiquidity]:
    """Flatten Schwab's callExpDateMap/putExpDateMap into {(type, strike): rows}."""
    out: dict[tuple[str, float], ContractLiquidity] = {}
    for map_key, side in (("callExpDateMap", "C"), ("putExpDateMap", "P")):
        side_map = payload.get(map_key)
        if not isinstance(side_map, dict):
            continue
        for strikes in side_map.values():
            if not isinstance(strikes, dict):
                continue
            for strike_key, contracts in strikes.items():
                if not isinstance(contracts, list):
                    continue
                for contract in contracts:
                    if not isinstance(contract, dict):
                        continue
                    strike = _as_float(contract.get("strikePrice"))
                    if strike is None:
                        strike = _as_float(strike_key)
                    if strike is None:
                        continue
                    oi = _as_float(contract.get("openInterest")) or 0.0
                    vol = _as_float(contract.get("totalVolume")) or 0.0
                    out[(side, round(strike, 4))] = ContractLiquidity(
                        open_interest=int(oi), volume=int(vol), source="schwab_chain")
    return out


def _load_expiry(underlying: str, expiry: str) -> dict[tuple[str, float], ContractLiquidity] | None:
    key = (underlying, expiry)
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and (now - hit[0]) < CACHE_TTL_SECONDS:
            return hit[1]

    raw = _get_raw_client()
    if raw is None:
        return None
    try:
        exp_date = date.fromisoformat(expiry)
    except ValueError:
        logger.warning("option liquidity: bad expiry %r for %s", expiry, underlying)
        return None

    try:
        options = getattr(raw, "Options", None)
        resp = raw.get_option_chain(
            symbol=underlying,
            contract_type=getattr(getattr(options, "ContractType", None), "ALL", "ALL"),
            strategy=getattr(getattr(options, "Strategy", None), "SINGLE", "SINGLE"),
            from_date=exp_date,
            to_date=exp_date,
            include_underlying_quote=False,
        )
        if int(getattr(resp, "status_code", 0)) < 200 or int(resp.status_code) >= 300:
            logger.warning("option liquidity: %s %s chain HTTP %s",
                           underlying, expiry, getattr(resp, "status_code", "?"))
            return None
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("option liquidity: %s %s chain error (%s)", underlying, expiry, exc)
        return None

    if not isinstance(payload, dict):
        return None
    parsed = _parse_chain(payload)
    with _cache_lock:
        _cache[key] = (time.monotonic(), parsed)
    return parsed


def contract_liquidity(
    underlying: str,
    *,
    expiry: Any,
    strike: Any,
    option_type: Any = "C",
) -> ContractLiquidity | None:
    """Open interest + volume for one contract, or ``None`` when unknown.

    ``None`` means the chain could not be fetched or the strike is not in it —
    it must NOT be treated as zero liquidity.
    """
    sym = str(underlying or "").strip().upper()
    strike_value = _as_float(strike)
    if not sym or strike_value is None:
        return None
    rows = _load_expiry(sym, _normalize_expiry(expiry))
    if not rows:
        return None
    side = _normalize_type(option_type)
    exact = rows.get((side, round(strike_value, 4)))
    if exact is not None:
        return exact
    # Schwab occasionally reports 135.0 where the contract record says 135.00001;
    # fall back to the nearest strike within a tick of the request.
    best: ContractLiquidity | None = None
    best_gap = _STRIKE_TOLERANCE
    for (row_side, row_strike), row in rows.items():
        if row_side != side:
            continue
        gap = abs(row_strike - strike_value)
        if gap <= best_gap:
            best, best_gap = row, gap
    return best


def reset_cache() -> None:
    """Drop cached chains (tests, and long-lived processes crossing sessions)."""
    global _client, _client_failed
    with _cache_lock:
        _cache.clear()
    with _client_lock:
        _client = None
        _client_failed = False
