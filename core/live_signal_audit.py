from __future__ import annotations

import json
import math
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "signal_audit_v1"


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    f = finite_float(value, default=None)
    if f is not None and not isinstance(value, bool):
        return f
    return value


def score_bucket(score: Any, *, width: float = 0.05, prefix: str = "score") -> str | None:
    val = finite_float(score)
    if val is None:
        return None
    width = max(float(width), 1e-6)
    lo = math.floor(val / width) * width
    hi = lo + width
    return f"{prefix}_{lo:.2f}_{hi:.2f}"


def rank_bucket(rank_pct: Any, *, width: float = 0.05) -> str | None:
    val = finite_float(rank_pct)
    if val is None:
        return None
    val = max(0.0, min(1.0, val))
    lo = math.floor(val / width) * width
    hi = min(1.0, lo + width)
    return f"rank_{lo:.2f}_{hi:.2f}"


def build_signal_audit(
    *,
    module: str,
    ticker: str,
    score: Any = None,
    side: str | None = "long",
    rank: Any = None,
    rank_pct: Any = None,
    regime: str | None = None,
    signal_ts: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score_val = finite_float(score)
    rank_val = finite_float(rank)
    rank_pct_val = finite_float(rank_pct)
    out = {
        "schema_version": SCHEMA_VERSION,
        "module": str(module),
        "ticker": str(ticker).upper(),
        "side": side,
        "score": score_val,
        "score_bucket": score_bucket(score_val),
        "rank": int(rank_val) if rank_val is not None and float(rank_val).is_integer() else rank_val,
        "rank_pct": rank_pct_val,
        "rank_bucket": rank_bucket(rank_pct_val),
        "regime": regime,
        "signal_ts": json_safe(signal_ts),
        "extra": json_safe(extra or {}),
    }
    return {k: v for k, v in out.items() if v is not None}


def build_equity_order_audit(
    *,
    signal_audit: dict[str, Any] | None,
    symbol: str,
    side: str,
    qty: Any,
    reason: str | None = None,
    reference_price: Any = None,
) -> dict[str, Any]:
    return {
        "schema_version": "order_audit_v1",
        "instrument": "equity",
        "symbol": str(symbol).upper(),
        "side": str(side).lower(),
        "qty": int(finite_float(qty, 0) or 0),
        "reason": reason,
        "reference_price": finite_float(reference_price),
        "signal_audit": signal_audit,
    }


def build_option_order_audit(
    *,
    signal_audit: dict[str, Any] | None,
    option_symbol: str,
    route: str,
    side: str,
    qty: Any,
    underlying_price: Any = None,
    strike: Any = None,
    premium: Any = None,
    limit_price: Any = None,
    mid_price: Any = None,
    delta: Any = None,
    dte: Any = None,
    spread_pct: Any = None,
    expiration: Any = None,
) -> dict[str, Any]:
    underlying = finite_float(underlying_price)
    strike_val = finite_float(strike)
    premium_val = finite_float(premium)
    if premium_val is None:
        premium_val = finite_float(limit_price)
    if premium_val is None:
        premium_val = finite_float(mid_price)
    side_l = str(side or "").lower()
    signal_side = str((signal_audit or {}).get("side") or "").lower()
    option_side = "put" if ("put" in signal_side or signal_side == "short") else "call"
    breakeven_move_pct = None
    if underlying and underlying > 0 and strike_val is not None and premium_val is not None:
        if option_side == "put":
            breakeven = strike_val - premium_val
            breakeven_move_pct = (underlying - breakeven) / underlying
        else:
            breakeven = strike_val + premium_val
            breakeven_move_pct = (breakeven - underlying) / underlying
    return {
        "schema_version": "order_audit_v1",
        "instrument": "option",
        "option_symbol": str(option_symbol),
        "route": route,
        "side": side_l,
        "qty": int(finite_float(qty, 0) or 0),
        "underlying_price": underlying,
        "strike": strike_val,
        "premium": premium_val,
        "limit_price": finite_float(limit_price),
        "mid_price": finite_float(mid_price),
        "delta": finite_float(delta),
        "dte": int(finite_float(dte, 0) or 0) if dte is not None else None,
        "spread_pct": finite_float(spread_pct),
        "expiration": json_safe(expiration),
        "breakeven_move_pct": breakeven_move_pct,
        "signal_audit": signal_audit,
    }


def append_jsonl(path: str | Path | None, payload: dict[str, Any]) -> None:
    """Append one JSON record, durably.

    The fsync is not optional bookkeeping. Without it, ext4 delayed allocation
    journals the new file SIZE but not the data blocks, so an unclean host kill
    leaves the record as a run of NUL bytes — a hole in the file that no reader
    can parse. That is exactly what happened to
    Data/inference/momentum_expansion/live_signal_audit.jsonl on 2026-07-24
    (found 2026-08-03): 223 NUL bytes where one order_plan record had been, on a
    box with a known WSL2 crash history. These are live trading audit logs, and
    a few hundred appends a day makes the sync cost irrelevant next to losing a
    record of what the system decided.
    """
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(json_safe(payload), default=str, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
