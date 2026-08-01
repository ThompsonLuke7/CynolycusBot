"""Durable broker equity/position snapshots for daily P&L attribution.

The broker's account-level day P&L is authoritative, while individual position
``unrealized_intraday_pl`` fields can use a different mark basis.  Capturing
both at the regular and extended-session close makes future long/call/put/share
attribution reproducible from the exact broker marks observed at those times.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5
from zoneinfo import ZoneInfo

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
from core.nervous_system.contracts.enums import AssetClass, DataQualitySeverity
from core.nervous_system.contracts.quality import DataQualityIssue, DataQualitySummary
from core.nervous_system.contracts.states import PortfolioPosition, PortfolioState


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "Data/inference/account_snapshots"
_ET = ZoneInfo("America/New_York")
_ACCOUNT_FIELDS = ("equity", "last_equity", "portfolio_value", "cash", "buying_power")
_POSITION_FIELDS = (
    "symbol", "asset_class", "qty", "side", "avg_entry_price", "current_price",
    "market_value", "cost_basis", "unrealized_pl", "unrealized_plpc",
    "unrealized_intraday_pl", "unrealized_intraday_plpc",
)
_PORTFOLIO_PRODUCER = "core.broker_equity_snapshot"
_PORTFOLIO_MODEL_VERSION = "broker-portfolio-adapter@1"
_PORTFOLIO_FEATURE_VERSION = "broker-portfolio@1"
_PORTFOLIO_CONFIG_VERSION = "broker-portfolio@1:validity-24h"
_PORTFOLIO_VALIDITY = timedelta(hours=24)
_OCC_TAIL = re.compile(r"^\d{6}[CP]\d{8}$")
_OPEN_ORDER_ERROR_CODES = {
    "FAILED": "OPEN_ORDERS_READ_FAILED",
    "NOT_OBSERVED": "OPEN_ORDERS_NOT_OBSERVED",
    "UNAVAILABLE": "OPEN_ORDERS_READ_UNAVAILABLE",
    "UNSUPPORTED": "OPEN_ORDERS_READ_UNSUPPORTED",
}
_PORTFOLIO_PUBLICATION_ERROR = "PORTFOLIO_STATE_PUBLICATION_FAILED"


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _required_finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    result = _as_float(value)
    if result is None:
        raise ValueError(f"{field} must be finite")
    return result


def _optional_finite(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    return _required_finite(value, field=field)


def _aware_utc(value: Any, *, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field} must be a valid ISO-8601 timestamp") from exc
    else:
        raise ValueError(f"{field} must be a timezone-aware timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _canonical_json(value: Any) -> str:
    def default(item: Any) -> str:
        if isinstance(item, datetime):
            return item.isoformat()
        return str(item)

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=default,
        allow_nan=False,
    )


def _position_asset_class(value: Any, *, field: str) -> AssetClass:
    key = str(value or "").strip().upper().replace("-", "_")
    if key in {"US_EQUITY", "EQUITY", "STOCK"}:
        return AssetClass.EQUITY
    if key in {"US_OPTION", "OPTION"}:
        return AssetClass.OPTION
    raise ValueError(f"{field} has unsupported asset class {value!r}")


def _position_underlying(symbol: str, asset_class: AssetClass) -> str:
    if asset_class is AssetClass.EQUITY:
        return symbol
    if len(symbol) <= 15 or _OCC_TAIL.fullmatch(symbol[-15:]) is None:
        raise ValueError(f"option symbol {symbol!r} is not a valid OCC identity")
    underlying = symbol[:-15].rstrip()
    if not underlying:
        raise ValueError(f"option symbol {symbol!r} has no underlying")
    return underlying


def _open_order_ids(raw: Mapping[str, Any]) -> tuple[str, ...]:
    values = raw.get("open_order_ids")
    if values is None:
        values = raw.get("open_orders", ())
    if not isinstance(values, (list, tuple)):
        raise ValueError("open_order_ids/open_orders must be a list")
    identifiers: list[str] = []
    for index, value in enumerate(values, start=1):
        candidate = value.get("id") if isinstance(value, Mapping) else value
        if candidate is None or not str(candidate).strip():
            raise ValueError(f"open order {index} has no id")
        identifiers.append(str(candidate))
    return tuple(sorted(identifiers))


def _open_orders_observation(
    raw: Mapping[str, Any],
) -> tuple[tuple[str, ...], str, str | None]:
    has_orders = "open_order_ids" in raw or "open_orders" in raw
    evidence = raw.get("open_orders_observation")
    if evidence is None:
        if has_orders:
            return _open_order_ids(raw), "OBSERVED", None
        return (), "NOT_OBSERVED", _OPEN_ORDER_ERROR_CODES["NOT_OBSERVED"]
    if not isinstance(evidence, Mapping):
        raise ValueError("open_orders_observation must be a mapping")
    status = str(evidence.get("status") or "").strip().upper()
    if status not in {"OBSERVED", "UNAVAILABLE", "UNSUPPORTED", "FAILED", "NOT_OBSERVED"}:
        raise ValueError(f"open_orders_observation has unsupported status {status!r}")
    if status == "OBSERVED":
        if not has_orders:
            raise ValueError("observed open-order evidence must include open_orders")
        return _open_order_ids(raw), status, None
    if has_orders:
        raise ValueError(f"open_orders must be absent when observation status is {status}")
    return (), status, _OPEN_ORDER_ERROR_CODES[status]


def _safe_exception_type(exc: Exception) -> str:
    for exception_type in (
        TimeoutError,
        PermissionError,
        ConnectionError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ):
        if isinstance(exc, exception_type):
            return exception_type.__name__
    return "Exception"


def _observation_failure(
    status: str,
    *,
    exception: Exception | None = None,
) -> dict[str, Any]:
    evidence = {
        "status": status,
        "error_code": _OPEN_ORDER_ERROR_CODES[status],
    }
    if exception is not None:
        evidence["exception_type"] = _safe_exception_type(exception)
    return {"open_orders_observation": evidence}


def _capture_open_orders(client: Any) -> dict[str, Any]:
    try:
        get_orders = getattr(client, "get_orders")
    except AttributeError:
        return _observation_failure("UNSUPPORTED")
    except Exception as exc:
        return _observation_failure("FAILED", exception=exc)
    if not callable(get_orders):
        return _observation_failure("UNSUPPORTED")
    try:
        values = get_orders(status="open")
        if not isinstance(values, (list, tuple)):
            raise ValueError("get_orders(status='open') must return a list")
        orders = []
        for index, value in enumerate(values, start=1):
            if not isinstance(value, Mapping):
                raise ValueError(f"open order {index} must be a mapping")
            identifier = str(value.get("id") or "").strip()
            if not identifier:
                raise ValueError(f"open order {index} has no id")
            orders.append({"id": identifier})
    except Exception as exc:
        return _observation_failure("FAILED", exception=exc)
    return {
        "open_orders": sorted(orders, key=lambda order: order["id"]),
        "open_orders_observation": {"status": "OBSERVED"},
    }


def _lineage_id(*, source_hash: str, account_alias: str, captured_at: datetime) -> str:
    return json.dumps(
        {
            "content_hash": source_hash,
            "record_locator": f"account_snapshot:{account_alias}:{captured_at.isoformat()}",
            "source_id": _PORTFOLIO_PRODUCER,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def adapt_broker_portfolio_snapshot(
    raw: Mapping[str, object],
    *,
    strategy_ownership: Mapping[str, str],
) -> PortfolioState:
    """Adapt one durable broker snapshot without reading broker/API state."""
    if not isinstance(raw, Mapping):
        raise ValueError("raw broker snapshot must be a mapping")
    if not isinstance(strategy_ownership, Mapping):
        raise ValueError("strategy_ownership must be a mapping")

    captured_at = _aware_utc(raw.get("captured_at_utc"), field="captured_at_utc")
    account_alias_value = raw.get("account_label")
    if account_alias_value is None or not str(account_alias_value).strip():
        raise ValueError("account_label must be non-empty")
    account_alias = str(account_alias_value).strip()
    account = raw.get("account")
    if not isinstance(account, Mapping):
        raise ValueError("account must be a mapping")
    equity = _required_finite(account.get("equity"), field="equity")
    cash = _required_finite(account.get("cash"), field="cash")
    buying_power = _required_finite(account.get("buying_power"), field="buying_power")
    day_pl = _optional_finite(raw.get("day_pl"), field="day_pl")

    positions_value = raw.get("positions", ())
    if not isinstance(positions_value, (list, tuple)):
        raise ValueError("positions must be a list")
    positions: list[PortfolioPosition] = []
    unmatched_symbols: list[str] = []
    for index, value in enumerate(positions_value, start=1):
        if not isinstance(value, Mapping):
            raise ValueError(f"position {index} must be a mapping")
        symbol_value = value.get("symbol")
        if symbol_value is None or not str(symbol_value).strip():
            raise ValueError(f"position {index} symbol must be non-empty")
        symbol = str(symbol_value)
        asset_class = _position_asset_class(
            value.get("asset_class"), field=f"position {index} asset_class"
        )
        quantity = _required_finite(value.get("qty"), field=f"position {index} qty")
        strategy_id: str | None = None
        if symbol in strategy_ownership:
            supplied_strategy = strategy_ownership[symbol]
            if supplied_strategy is None or not str(supplied_strategy).strip():
                raise ValueError(f"strategy_ownership[{symbol!r}] must be non-empty")
            strategy_id = str(supplied_strategy)
        else:
            unmatched_symbols.append(symbol)
        positions.append(
            PortfolioPosition(
                broker_position_id=str(value.get("broker_position_id") or value.get("id") or symbol),
                symbol=symbol,
                underlying=_position_underlying(symbol, asset_class),
                asset_class=asset_class,
                quantity=quantity,
                average_entry_price=_optional_finite(
                    value.get("avg_entry_price"), field=f"position {index} avg_entry_price"
                ),
                current_price=_optional_finite(
                    value.get("current_price"), field=f"position {index} current_price"
                ),
                market_value=_optional_finite(
                    value.get("market_value"), field=f"position {index} market_value"
                ),
                strategy_id=strategy_id,
                ownership_status="ASSIGNED" if strategy_id is not None else "UNASSIGNED",
            )
        )

    positions_tuple = tuple(sorted(positions, key=lambda item: (item.symbol, item.broker_position_id)))
    open_order_ids, open_orders_status, open_orders_error_code = _open_orders_observation(raw)
    source_hash = hashlib.sha256(_canonical_json(raw).encode("utf-8")).hexdigest()
    lineage_id = _lineage_id(
        source_hash=source_hash,
        account_alias=account_alias,
        captured_at=captured_at,
    )
    quality_issues: list[DataQualityIssue] = []
    if unmatched_symbols:
        quality_issues.append(
            DataQualityIssue(
                code="UNMATCHED_POSITION_OWNERSHIP",
                severity=DataQualitySeverity.WARNING,
                component=_PORTFOLIO_PRODUCER,
                message=(
                    "fill-derived ownership was unavailable for: "
                    + ", ".join(sorted(unmatched_symbols))
                ),
            )
        )
    if open_orders_status != "OBSERVED":
        detail = f"; {open_orders_error_code}" if open_orders_error_code else ""
        quality_issues.append(
            DataQualityIssue(
                code="OPEN_ORDERS_NOT_OBSERVED",
                severity=DataQualitySeverity.WARNING,
                component=_PORTFOLIO_PRODUCER,
                message=f"open orders were not observed ({open_orders_status}{detail})",
                fallback_used="open_order_ids left empty",
            )
        )
    data_quality = DataQualitySummary(issues=tuple(quality_issues))
    identity = {
        "account_alias": account_alias,
        "captured_at": captured_at.isoformat(),
        "ownership": sorted(
            (position.symbol, position.strategy_id)
            for position in positions_tuple
            if position.strategy_id is not None
        ),
        "producer": _PORTFOLIO_PRODUCER,
        "source_hash": source_hash,
    }
    state_id = uuid5(
        NAMESPACE_URL,
        _canonical_json(identity),
    )
    return PortfolioState(
        state_id=state_id,
        entity_id=account_alias,
        as_of=captured_at,
        available_at=captured_at,
        generated_at=captured_at,
        valid_until=captured_at + _PORTFOLIO_VALIDITY,
        source_window_start=captured_at,
        source_window_end=captured_at,
        schema_version=1,
        producer=_PORTFOLIO_PRODUCER,
        model_version=_PORTFOLIO_MODEL_VERSION,
        feature_version=_PORTFOLIO_FEATURE_VERSION,
        config_version=_PORTFOLIO_CONFIG_VERSION,
        lineage_ids=(lineage_id,),
        data_quality=data_quality,
        account_alias=account_alias,
        equity=equity,
        cash=cash,
        buying_power=buying_power,
        day_pl=day_pl,
        positions=positions_tuple,
        open_order_ids=open_order_ids,
        broker_observed_at=captured_at,
    )


def session_phase(now: datetime) -> str:
    """Which trading phase the marks in this snapshot came from.

    Without it a consumer cannot tell a live regular-session mark from a stale
    post-close one, and the 20:05 ET capture looks identical to the 16:05 ET
    capture even when nothing has re-priced.
    """
    et = now.astimezone(_ET)
    if et.weekday() >= 5:
        return "closed"
    minutes = et.hour * 60 + et.minute
    if 4 * 60 <= minutes < 9 * 60 + 30:
        return "premarket"
    if 9 * 60 + 30 <= minutes < 16 * 60:
        return "regular"
    if 16 * 60 <= minutes < 20 * 60:
        return "extended"
    return "closed"


def snapshot_path(*, account_label: str, root: Path, now: datetime) -> Path:
    session = now.astimezone(_ET).strftime("%Y%m%d")
    safe_label = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in account_label)
    return root / f"broker_equity_{session}_{safe_label}.jsonl"


def _position_record(position: dict[str, Any]) -> dict[str, Any]:
    out = {field: _json_scalar(position.get(field)) for field in _POSITION_FIELDS}
    market_value = _as_float(position.get("market_value"))
    cost_basis = _as_float(position.get("cost_basis"))
    if market_value is not None and cost_basis is not None:
        # Independent of the broker's own field, so a disagreement is visible
        # instead of silently propagating into the daily attribution. No side
        # adjustment: shorts already carry a negative qty, market_value and
        # cost_basis, so the subtraction gets the sign right for both.
        out["unrealized_pl_derived"] = round(market_value - cost_basis, 2)
    else:
        out["unrealized_pl_derived"] = None
    return out


def capture_snapshot(
    *,
    client: Any,
    account_label: str = "paper",
    root: Path = DEFAULT_ROOT,
    now: datetime | None = None,
    unit_of_work: Any | None = None,
    strategy_ownership: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Read broker account/positions and append one JSONL close-mark snapshot.

    If supplied, ``unit_of_work`` remains caller-owned.  The local JSONL record
    is durable before the optional state staging begins.  Successful insertion
    returns ``publication_status="STAGED"`` because the caller must still commit;
    failures are returned alongside the durable local backup.
    """
    captured = _aware_utc(
        now if now is not None else datetime.now(timezone.utc),
        field="now",
    )
    account = client.get_account() or {}
    positions = client.get_positions() or []
    open_orders_evidence = _capture_open_orders(client)
    account_fields = {field: _json_scalar(account.get(field)) for field in _ACCOUNT_FIELDS}
    equity = _as_float(account.get("equity"))
    last_equity = _as_float(account.get("last_equity"))
    record = {
        "schema_version": 2,
        "captured_at_utc": captured.isoformat(),
        "captured_at_et": captured.astimezone(_ET).isoformat(),
        "session_date_et": captured.astimezone(_ET).date().isoformat(),
        "session_phase": session_phase(captured),
        "account_label": account_label,
        "account": account_fields,
        # The account-level delta is the one number that is always right; per
        # position `unrealized_intraday_pl` is computed against a previous close
        # that does not exist for anything opened today, so it is kept as the
        # broker reported it and cross-checked by `unrealized_pl_derived`.
        "day_pl": (
            round(equity - last_equity, 2)
            if equity is not None and last_equity is not None and last_equity != 0.0
            else None
        ),
        "positions": [_position_record(position) for position in positions],
        **open_orders_evidence,
    }
    out = snapshot_path(account_label=account_label, root=root, now=captured)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    result: dict[str, Any] = {
        "path": str(out),
        "position_count": len(record["positions"]),
        **record,
    }
    if unit_of_work is None:
        return result
    open_orders_status = record["open_orders_observation"]["status"]
    if open_orders_status != "OBSERVED":
        observation = record["open_orders_observation"]
        result.update(
            {
                "publication_status": "FAILED",
                "publication_error": observation["error_code"],
            }
        )
        if "exception_type" in observation:
            result["publication_exception_type"] = observation["exception_type"]
        return result
    try:
        state = adapt_broker_portfolio_snapshot(
            record,
            strategy_ownership=strategy_ownership or {},
        )
        unit_of_work.states.insert_states_idempotently((state,))
    except Exception as exc:  # publication must be explicit after local durability
        result.update(
            {
                "publication_status": "FAILED",
                "publication_error": _PORTFOLIO_PUBLICATION_ERROR,
                "publication_exception_type": _safe_exception_type(exc),
            }
        )
    else:
        result["publication_status"] = "STAGED"
    return result


def capture_from_env(*, env_file: str, account_label: str = "paper", root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    return capture_snapshot(
        client=AlpacaOptionsClient(env_file=env_file),
        account_label=account_label,
        root=root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a read-only broker equity/position snapshot.")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--account-label", default="paper")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = capture_from_env(env_file=args.env, account_label=args.account_label, root=args.root)
    print(json.dumps({"path": result["path"], "position_count": result["position_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
