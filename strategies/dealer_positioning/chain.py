from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

from strategies.dealer_positioning.models import OptionContractRow


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _normalize_option_type(value: Any, fallback: str) -> str:
    raw = str(value or fallback).strip().upper()
    if raw.startswith("C"):
        return "C"
    if raw.startswith("P"):
        return "P"
    return fallback


def _parse_expiration_key(key: str) -> tuple[str, int | None]:
    # Schwab maps look like "2026-06-12:0".
    parts = str(key).split(":")
    exp = parts[0]
    dte = _as_int(parts[1]) if len(parts) > 1 else None
    return exp, dte


def _iter_contracts_from_side_map(side_map: dict[str, Any], fallback_type: str) -> Iterable[tuple[str, int | None, float, dict]]:
    for exp_key, strikes in (side_map or {}).items():
        expiration, dte = _parse_expiration_key(exp_key)
        if not isinstance(strikes, dict):
            continue
        for strike_key, contracts in strikes.items():
            strike = _as_float(strike_key, default=0.0)
            if isinstance(contracts, dict):
                contracts = [contracts]
            if not isinstance(contracts, list):
                continue
            for contract in contracts:
                if isinstance(contract, dict):
                    yield expiration, dte, strike, contract


def parse_schwab_option_chain(
    chain: dict[str, Any],
    *,
    symbol: str,
    timestamp: datetime | None = None,
    dte_offsets: set[int] | None = None,
    expiration_buckets: dict[str, int] | None = None,
) -> tuple[float | None, list[OptionContractRow]]:
    """Flatten Schwab option-chain JSON into contract rows.

    Accepts Schwab's callExpDateMap/putExpDateMap shape and a lightweight
    fixture shape with an "options" list for tests or archived snapshots.
    """
    ts = timestamp or datetime.now(timezone.utc)
    spot = _as_float(
        chain.get("underlyingPrice")
        or (chain.get("underlying") or {}).get("last")
        or (chain.get("quote") or {}).get("lastPrice"),
        default=0.0,
    )
    rows: list[OptionContractRow] = []

    if isinstance(chain.get("options"), list):
        for item in chain["options"]:
            if not isinstance(item, dict):
                continue
            dte = _as_int(item.get("dte") or item.get("daysToExpiration"))
            expiration = str(item.get("expirationDate") or item.get("expiration") or "")
            expiry_bucket = expiration_buckets.get(expiration) if expiration_buckets else None
            if expiration_buckets is not None and expiration not in expiration_buckets:
                continue
            if expiration_buckets is None and dte_offsets is not None and dte not in dte_offsets:
                continue
            row = _row_from_contract(ts, symbol, item, expiration, dte, expiry_bucket=expiry_bucket)
            if row.strike > 0.0:
                rows.append(row)
        return (spot or None), rows

    for side_key, fallback_type in (("callExpDateMap", "C"), ("putExpDateMap", "P")):
        for expiration, dte, strike, contract in _iter_contracts_from_side_map(chain.get(side_key) or {}, fallback_type):
            expiry_bucket = expiration_buckets.get(expiration) if expiration_buckets else None
            if expiration_buckets is not None and expiration not in expiration_buckets:
                continue
            if expiration_buckets is None and dte_offsets is not None and dte not in dte_offsets:
                continue
            row = _row_from_contract(
                ts,
                symbol,
                contract,
                expiration,
                dte,
                fallback_type=fallback_type,
                fallback_strike=strike,
                expiry_bucket=expiry_bucket,
            )
            if row.strike > 0.0:
                rows.append(row)

    return (spot or None), rows


def _row_from_contract(
    timestamp: datetime,
    symbol: str,
    contract: dict[str, Any],
    expiration: str | None,
    dte: int | None,
    *,
    fallback_type: str = "C",
    fallback_strike: float | None = None,
    expiry_bucket: int | None = None,
) -> OptionContractRow:
    gamma = _as_float(contract.get("gamma") or (contract.get("greeks") or {}).get("gamma"))
    delta_raw = contract.get("delta") or (contract.get("greeks") or {}).get("delta")
    vega_raw = contract.get("vega") or (contract.get("greeks") or {}).get("vega")
    iv_raw = (
        contract.get("volatility")
        or contract.get("impliedVolatility")
        or contract.get("iv")
        or (contract.get("greeks") or {}).get("iv")
    )
    return OptionContractRow(
        timestamp=timestamp,
        symbol=symbol.strip().upper(),
        expiration=str(expiration or contract.get("expirationDate") or ""),
        dte=dte,
        strike=_as_float(contract.get("strikePrice") or contract.get("strike"), default=_as_float(fallback_strike)),
        option_type=_normalize_option_type(contract.get("putCall") or contract.get("optionType"), fallback_type),
        open_interest=_as_float(contract.get("openInterest") or contract.get("open_interest")),
        volume=_as_float(contract.get("totalVolume") or contract.get("volume")),
        gamma=gamma,
        delta=_as_float(delta_raw, default=math.nan) if delta_raw is not None else None,
        vega=_as_float(vega_raw, default=math.nan) if vega_raw is not None else None,
        iv=_as_float(iv_raw, default=math.nan) if iv_raw is not None else None,
        expiry_bucket=expiry_bucket,
    )


def target_expiration_dates(ref_date: date, dte_offsets: Iterable[int]) -> tuple[date, date]:
    buckets = trading_expiration_buckets(ref_date, dte_offsets)
    dates = sorted(date.fromisoformat(expiration) for expiration in buckets)
    return dates[0], dates[-1]


def trading_expiration_buckets(ref_date: date, dte_offsets: Iterable[int]) -> dict[str, int]:
    offsets = sorted(int(x) for x in dte_offsets)
    if not offsets:
        offsets = [0, 1, 2]
    out: dict[str, int] = {}
    current = ref_date
    bucket = 0
    wanted = set(offsets)
    max_offset = max(offsets)
    while bucket <= max_offset:
        if current.weekday() < 5:
            if bucket in wanted:
                out[current.isoformat()] = bucket
            bucket += 1
        current += timedelta(days=1)
    return out
