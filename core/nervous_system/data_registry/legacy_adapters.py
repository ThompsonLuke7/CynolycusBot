"""Explicit, causal adapters for legacy operational evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections.abc import Mapping
import math
from typing import Any, Literal
from uuid import uuid4

from core.nervous_system.contracts.base import ContractModel, UtcDatetime
from core.nervous_system.contracts.enums import AssetClass
from core.nervous_system.contracts.quality import DataQualitySummary
from core.nervous_system.contracts.states import (
    PortfolioPosition,
    PortfolioState,
)


class AdapterIssue(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class LegacyOperationalEvidence(ContractModel):
    """Typed evidence retained when no current state/decision contract fits."""

    event_type: str
    entity_id: str
    as_of: UtcDatetime | None
    available_at: UtcDatetime
    observed_at: UtcDatetime | None = None
    adapter: str
    payload: dict[str, Any]


class OwnershipCandidateEvidence(LegacyOperationalEvidence):
    ownership_status: Literal["UNASSIGNED"] = "UNASSIGNED"


@dataclass(frozen=True)
class LegacyAdapterResult:
    target_type: str
    contract: ContractModel | None
    warnings: tuple[str, ...]
    quarantine_code: str | None
    quarantine_message: str | None


def _value(payload: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    for nested_name in ("account", "account_snapshot", "state", "metadata"):
        nested = payload.get(nested_name)
        if isinstance(nested, Mapping):
            for name in names:
                if name in nested and nested[name] is not None:
                    return nested[name]
    return None


def _timestamp(value: Any, *, field_name: str, code_prefix: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise AdapterIssue(
                f"INVALID_{code_prefix}",
                f"{field_name} is not an ISO-8601 timestamp: {value!r}",
            ) from exc
    else:
        raise AdapterIssue(
            f"INVALID_{code_prefix}",
            f"{field_name} must be an explicit ISO-8601 timestamp",
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AdapterIssue(
            f"NAIVE_{code_prefix}",
            f"{field_name} must be timezone-aware",
        )
    return parsed.astimezone(timezone.utc)


def _optional_timestamp(
    payload: Mapping[str, Any],
    names: tuple[str, ...],
    *,
    field_name: str,
    code_prefix: str,
) -> datetime | None:
    value = _value(payload, names)
    if value is None:
        return None
    return _timestamp(value, field_name=field_name, code_prefix=code_prefix)


def _required_availability(payload: Mapping[str, Any]) -> datetime:
    names = (
        "available_at",
        "available_at_utc",
        "observed_at",
        "captured_at_utc",
        "captured_at",
        "filled_at",
        "closed_at",
        "event_time",
        "timestamp",
        "created_at",
        "updated_at",
    )
    for name in names:
        value = _value(payload, (name,))
        if value is not None:
            return _timestamp(
                value,
                field_name=name,
                code_prefix="AVAILABLE_AT",
            )
    raise AdapterIssue(
        "MISSING_AVAILABLE_AT",
        "legacy record has no reliable explicit availability timestamp",
    )


def _as_of(
    payload: Mapping[str, Any],
    available_at: datetime,
    *,
    signal: bool = False,
) -> datetime:
    names = (
        ("bar", "as_of", "selected_bar", "decision_time", "event_time")
        if signal
        else ("as_of", "selected_bar", "event_time", "timestamp", "decision_time")
    )
    value = _value(payload, names)
    if value is None:
        return available_at
    return _timestamp(value, field_name=names[0], code_prefix="AS_OF")


def _common_warnings(payload: Mapping[str, Any]) -> tuple[str, ...]:
    if any(key in payload for key in ("score", "raw_score", "confidence")):
        return ("legacy score retained as raw evidence; not mapped to probability",)
    return ()


def _generic_contract(
    payload: Mapping[str, Any],
    *,
    adapter: str,
    target_type: str,
    signal: bool = False,
    ownership_candidate: bool = False,
) -> tuple[ContractModel, tuple[str, ...]]:
    available_at = _required_availability(payload)
    as_of = _as_of(payload, available_at, signal=signal)
    observed_at = _optional_timestamp(
        payload,
        ("observed_at", "captured_at_utc", "captured_at", "timestamp"),
        field_name="observed_at",
        code_prefix="OBSERVED_AT",
    )
    normalized_payload = deepcopy(dict(payload))
    warnings = list(_common_warnings(payload))
    if ownership_candidate:
        normalized_payload = _unassigned_payload(normalized_payload)
        warnings.append("legacy managed state retained as UNASSIGNED ownership candidate")
    contract_type = OwnershipCandidateEvidence if ownership_candidate else LegacyOperationalEvidence
    contract = contract_type(
        event_type=str(payload.get("event") or payload.get("type") or target_type),
        entity_id=str(
            payload.get("ticker")
            or payload.get("symbol")
            or payload.get("module")
            or payload.get("strategy_id")
            or "LEGACY"
        ),
        as_of=as_of,
        available_at=available_at,
        observed_at=observed_at,
        adapter=adapter,
        payload=normalized_payload,
    )
    return contract, tuple(warnings)


def _unassigned_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove legacy ownership claims while retaining every other raw field."""

    payload["ownership_status"] = "UNASSIGNED"
    payload["strategy_id"] = None
    positions = payload.get("positions")
    if isinstance(positions, list):
        normalized_positions = []
        for position in positions:
            if isinstance(position, Mapping):
                candidate = dict(position)
                candidate["ownership_status"] = "UNASSIGNED"
                candidate["strategy_id"] = None
                candidate["confirmed_ownership"] = False
                normalized_positions.append(candidate)
            else:
                normalized_positions.append(position)
        payload["positions"] = normalized_positions
    return payload


def _finite_number(payload: Mapping[str, Any], names: tuple[str, ...]) -> float | None:
    value = _value(payload, names)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterIssue("INVALID_PORTFOLIO_VALUE", f"{names[0]} is not numeric") from exc
    if not math.isfinite(number):
        raise AdapterIssue("INVALID_PORTFOLIO_VALUE", f"{names[0]} must be finite")
    return number


def _portfolio_position(value: Mapping[str, Any], index: int) -> PortfolioPosition:
    symbol = str(value.get("symbol") or value.get("asset") or value.get("underlying") or "UNKNOWN")
    underlying = str(value.get("underlying") or symbol)
    asset_value = str(value.get("asset_class") or "EQUITY").upper()
    try:
        asset_class = AssetClass(asset_value)
    except ValueError:
        asset_class = AssetClass.EQUITY
    quantity = value.get("quantity", value.get("qty", 0.0))
    try:
        quantity_float = float(quantity)
    except (TypeError, ValueError) as exc:
        raise AdapterIssue(
            "INVALID_PORTFOLIO_POSITION",
            f"position {index} quantity is not numeric",
        ) from exc
    if not math.isfinite(quantity_float):
        raise AdapterIssue(
            "INVALID_PORTFOLIO_POSITION",
            f"position {index} quantity must be finite",
        )
    return PortfolioPosition(
        broker_position_id=str(value.get("broker_position_id") or value.get("id") or symbol),
        symbol=symbol,
        underlying=underlying,
        asset_class=asset_class,
        quantity=quantity_float,
        average_entry_price=value.get("average_entry_price", value.get("avg_entry_price")),
        current_price=value.get("current_price", value.get("market_price")),
        market_value=value.get("market_value"),
        strategy_id=None,
        ownership_status="UNASSIGNED",
    )


def _portfolio_state(payload: Mapping[str, Any], adapter: str) -> tuple[PortfolioState, tuple[str, ...]]:
    available_at = _required_availability(payload)
    as_of = _as_of(payload, available_at)
    account_alias = str(_value(payload, ("account_alias", "account_id", "account")) or "legacy")
    equity = _finite_number(payload, ("equity", "account_equity", "portfolio_value"))
    cash = _finite_number(payload, ("cash", "cash_balance"))
    buying_power = _finite_number(payload, ("buying_power", "available_buying_power"))
    if equity is None or cash is None or buying_power is None:
        raise AdapterIssue(
            "INCOMPLETE_PORTFOLIO_STATE",
            "account snapshot lacks equity, cash, or buying_power",
        )
    positions_value = _value(payload, ("positions",)) or ()
    if isinstance(positions_value, Mapping):
        positions_value = tuple(positions_value.values())
    if not isinstance(positions_value, (tuple, list)):
        raise AdapterIssue("INVALID_PORTFOLIO_POSITION", "positions must be a list")
    positions = tuple(
        _portfolio_position(position, index)
        for index, position in enumerate(positions_value, start=1)
        if isinstance(position, Mapping)
    )
    generated_at = _optional_timestamp(
        payload,
        ("generated_at", "observed_at", "captured_at_utc", "captured_at"),
        field_name="generated_at",
        code_prefix="GENERATED_AT",
    ) or available_at
    valid_until = _optional_timestamp(
        payload,
        ("valid_until",),
        field_name="valid_until",
        code_prefix="VALID_UNTIL",
    ) or (available_at + timedelta(days=1))
    source_start = _optional_timestamp(
        payload,
        ("source_window_start",),
        field_name="source_window_start",
        code_prefix="SOURCE_WINDOW_START",
    ) or as_of
    source_end = _optional_timestamp(
        payload,
        ("source_window_end",),
        field_name="source_window_end",
        code_prefix="SOURCE_WINDOW_END",
    ) or as_of
    try:
        state = PortfolioState(
            state_id=uuid4(),
            entity_id=account_alias,
            as_of=as_of,
            available_at=available_at,
            generated_at=generated_at,
            valid_until=valid_until,
            source_window_start=source_start,
            source_window_end=source_end,
            schema_version=1,
            producer="legacy_import",
            model_version="legacy@1",
            feature_version="legacy@1",
            config_version="legacy@1",
            lineage_ids=(),
            data_quality=DataQualitySummary(),
            account_alias=account_alias,
            equity=equity,
            cash=cash,
            buying_power=buying_power,
            day_pl=_finite_number(payload, ("day_pl", "day_pnl")),
            positions=positions,
            open_order_ids=tuple(str(item) for item in (_value(payload, ("open_order_ids",)) or ())),
            broker_observed_at=available_at,
        )
    except ValueError as exc:
        raise AdapterIssue("INVALID_PORTFOLIO_STATE", str(exc)) from exc
    return state, _common_warnings(payload)


def adapt_legacy_record(
    source_kind: str,
    adapter: str,
    raw_payload: Mapping[str, Any],
) -> LegacyAdapterResult:
    """Adapt one source-specific payload without guessing causal timestamps."""

    payload = dict(raw_payload)
    try:
        if adapter == "broker_equity_snapshot":
            try:
                state, warnings = _portfolio_state(payload, adapter)
                return LegacyAdapterResult("PORTFOLIO_STATE", state, warnings, None, None)
            except AdapterIssue as issue:
                # A snapshot with an explicit timestamp is still useful raw
                # evidence when its typed portfolio fields are incomplete.
                if issue.code == "INCOMPLETE_PORTFOLIO_STATE":
                    contract, warnings = _generic_contract(
                        payload,
                        adapter=adapter,
                        target_type="BROKER_EQUITY_SNAPSHOT",
                    )
                    return LegacyAdapterResult(
                        "BROKER_EQUITY_SNAPSHOT", contract, warnings, None, None
                    )
                raise
        if adapter == "managed_state":
            contract, warnings = _generic_contract(
                payload,
                adapter=adapter,
                target_type="OWNERSHIP_CANDIDATE",
                ownership_candidate=True,
            )
            return LegacyAdapterResult("OWNERSHIP_CANDIDATE", contract, warnings, None, None)
        if adapter == "live_signal_audit":
            contract, warnings = _generic_contract(
                payload,
                adapter=adapter,
                target_type="SIGNAL_AUDIT",
                signal=True,
            )
            return LegacyAdapterResult("SIGNAL_AUDIT", contract, warnings, None, None)
        if adapter in {
            "closed_trade",
            "raw_operational_event",
            "broker_state",
            "broker_trade_event",
            "legacy_decision",
            "legacy_policy",
            "swing_session",
            "runtime_evidence",
        }:
            contract, warnings = _generic_contract(
                payload,
                adapter=adapter,
                target_type=source_kind.upper(),
            )
            return LegacyAdapterResult(source_kind.upper(), contract, warnings, None, None)
        return LegacyAdapterResult(
            source_kind.upper(),
            None,
            (),
            "UNKNOWN_ADAPTER",
            f"no legacy adapter is registered for {adapter!r}",
        )
    except AdapterIssue as exc:
        return LegacyAdapterResult(
            source_kind.upper(),
            None,
            _common_warnings(payload),
            exc.code,
            str(exc),
        )
    except (TypeError, ValueError) as exc:
        return LegacyAdapterResult(
            source_kind.upper(),
            None,
            _common_warnings(payload),
            "INVALID_LEGACY_RECORD",
            str(exc),
        )


__all__ = [
    "AdapterIssue",
    "LegacyAdapterResult",
    "LegacyOperationalEvidence",
    "OwnershipCandidateEvidence",
    "adapt_legacy_record",
]
