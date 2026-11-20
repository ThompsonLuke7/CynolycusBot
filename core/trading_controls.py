"""Account-wide trading controls: the switch that stops new exposure.

Every live module (4H family, Meta's governed path, swing 30m, SPY daytrader,
Intraday Structure, Dealer/Amethyst, the startup queue) submits through
``AlpacaOptionsClient``, which asks :func:`order_refusal` before it posts an
order. The file is re-read on every order, so flipping it takes effect on the
next submission with no restart.

``Data/runtime/trading_controls.json`` (``CYNOLYCUS_TRADING_CONTROLS`` overrides
the path)::

    {"version": 1,
     "entries": "frozen",            # or "open"
     "reason": "paper account at the margin limit, 2026-11-03",
     "set_at_utc": "...",
     "limits": {"max_gross_leverage": 1.5, "min_margin_cushion_pct": 0.15}}

- No file: entries open, no limits (the behaviour before this existed).
- ``"entries": "frozen"``: every order that opens exposure is refused. Sells,
  cancels and replaces never reach this check, and a buy that covers an
  existing short (an option assignment the SPY daytrader flattens) passes.
- ``limits`` apply to opening orders while entries are open. Each is optional;
  null means off. ``max_gross_leverage`` refuses an opening order once
  (long + |short| market value) / equity has reached it.
  ``min_margin_cushion_pct`` refuses it once (equity - maintenance margin) /
  equity is below it. Both are read from the broker at the time of the order.
- A file that exists but cannot be read or parsed refuses opening orders: an
  unreadable kill switch must never read as "off".

The 2026-07-15 paper account went from 0.23x to 2.38x gross while equity fell
from $1.0M to $373k, and ended 11-02 with $0 buying power and a $14.6k margin
cushion: no limit existed anywhere, so the broker became the only one.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROLS_PATH = REPO_ROOT / "Data" / "runtime" / "trading_controls.json"
CONTROLS_ENV = "CYNOLYCUS_TRADING_CONTROLS"


@dataclass(frozen=True)
class TradingControls:
    entries_frozen: bool = False
    reason: str = ""
    max_gross_leverage: float | None = None
    min_margin_cushion_pct: float | None = None
    source: str = ""


def controls_path() -> Path:
    override = os.environ.get(CONTROLS_ENV, "").strip()
    return Path(override) if override else DEFAULT_CONTROLS_PATH


def _optional_positive(limits: Mapping[str, Any], key: str) -> float | None:
    value = limits.get(key)
    if value is None:
        return None
    number = float(value)
    if not number > 0:
        raise ValueError(f"{key} must be positive or null, got {value!r}")
    return number


def load_controls(path: Path | None = None) -> TradingControls:
    """Read the control file. Never raises: a bad file freezes entries."""

    target = path or controls_path()
    if not target.exists():
        return TradingControls(source=str(target))
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        entries = str(data.get("entries", "open")).strip().lower()
        if entries not in {"open", "frozen"}:
            raise ValueError(f"entries must be 'open' or 'frozen', got {entries!r}")
        limits = data.get("limits") or {}
        if not isinstance(limits, Mapping):
            raise ValueError("limits must be an object")
        return TradingControls(
            entries_frozen=entries == "frozen",
            reason=str(data.get("reason") or ""),
            max_gross_leverage=_optional_positive(limits, "max_gross_leverage"),
            min_margin_cushion_pct=_optional_positive(limits, "min_margin_cushion_pct"),
            source=str(target),
        )
    except Exception as exc:  # noqa: BLE001 - any unreadable file fails closed
        return TradingControls(
            entries_frozen=True,
            reason=f"unreadable trading controls file ({type(exc).__name__}: {exc})",
            source=str(target),
        )


def write_controls(
    *,
    entries: str,
    reason: str,
    set_at_utc: str,
    max_gross_leverage: float | None = None,
    min_margin_cushion_pct: float | None = None,
    path: Path | None = None,
) -> Path:
    if entries not in {"open", "frozen"}:
        raise ValueError(f"entries must be 'open' or 'frozen', got {entries!r}")
    target = path or controls_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "entries": entries,
        "reason": reason,
        "set_at_utc": set_at_utc,
        "limits": {
            "max_gross_leverage": max_gross_leverage,
            "min_margin_cushion_pct": min_margin_cushion_pct,
        },
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(target)
    load = load_controls(target)
    if load.reason.startswith("unreadable"):
        raise RuntimeError(f"wrote an unreadable controls file: {load.reason}")
    return target


def _opens_exposure(
    payload: Mapping[str, Any],
    get_position: Callable[[str], Mapping[str, Any] | None],
) -> bool:
    """Whether an order payload adds exposure.

    Explicit ``position_intent`` decides when present. Without it, a sell is an
    exit (the system holds no shorts) and a buy opens exposure unless it covers
    an existing short of at least its size.
    """

    if payload.get("order_class") == "mleg":
        return any(
            str(leg.get("position_intent", "")).endswith("_to_open")
            for leg in payload.get("legs") or []
        )
    intent = str(payload.get("position_intent") or "").lower()
    if intent:
        return intent.endswith("_to_open")
    if str(payload.get("side", "")).lower() != "buy":
        return False
    position = get_position(str(payload.get("symbol", "")))
    held = float((position or {}).get("qty") or 0.0)
    return not (held < 0 and float(payload.get("qty") or 0) <= abs(held))


def order_refusal(
    payload: Mapping[str, Any],
    *,
    get_position: Callable[[str], Mapping[str, Any] | None],
    get_account: Callable[[], Mapping[str, Any]],
    controls: TradingControls | None = None,
) -> dict[str, str] | None:
    """Return ``{"code", "message"}`` when the order must not be sent, else None."""

    ctl = controls if controls is not None else load_controls()
    if not (ctl.entries_frozen or ctl.max_gross_leverage or ctl.min_margin_cushion_pct):
        return None
    try:
        opening = _opens_exposure(payload, get_position)
    except Exception as exc:  # noqa: BLE001 - cannot prove it closes, so it opens
        opening = True
        lookup_error = f"; position lookup failed ({type(exc).__name__}: {exc})"
    else:
        lookup_error = ""
    if not opening:
        return None
    if ctl.entries_frozen:
        return {
            "code": "entries_frozen",
            "message": f"new entries are frozen by {ctl.source}: {ctl.reason or 'no reason given'}{lookup_error}",
        }

    try:
        account = get_account()
        equity = float(account["equity"])
        gross = float(account.get("long_market_value") or 0.0) + abs(
            float(account.get("short_market_value") or 0.0)
        )
        maintenance = float(account.get("maintenance_margin") or 0.0)
    except Exception as exc:  # noqa: BLE001 - an unread limit is not a passed limit
        return {
            "code": "risk_limit_unverified",
            "message": f"account read failed, risk limits unverified ({type(exc).__name__}: {exc})",
        }
    if equity <= 0:
        return {"code": "risk_limit_breached", "message": f"equity is {equity:.2f}"}
    if ctl.max_gross_leverage and gross / equity >= ctl.max_gross_leverage:
        return {
            "code": "gross_leverage_limit",
            "message": (
                f"gross {gross / equity:.2f}x of equity ${equity:,.0f} is at or above "
                f"the {ctl.max_gross_leverage:.2f}x limit"
            ),
        }
    if ctl.min_margin_cushion_pct:
        cushion = (equity - maintenance) / equity
        if cushion < ctl.min_margin_cushion_pct:
            return {
                "code": "margin_cushion_limit",
                "message": (
                    f"margin cushion {cushion:.1%} (equity ${equity:,.0f}, maintenance "
                    f"${maintenance:,.0f}) is below the {ctl.min_margin_cushion_pct:.1%} limit"
                ),
            }
    return None
