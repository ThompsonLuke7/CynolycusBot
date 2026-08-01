"""Causal adapters for publishing market-regime producer rows as states.

The producer tables remain the canonical research outputs.  This module only
normalizes an already-persisted row into the reviewed nervous-system state
contracts and optionally inserts those immutable states in one UOW transaction.
It deliberately does not classify rule scores as probabilities.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime, time, timezone
from decimal import Decimal
import json
import math
from numbers import Real
import re
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core.nervous_system.contracts.enums import Direction, MarketRegime, StateType
from core.nervous_system.contracts.quality import DataQualitySummary, LineageRef
from core.nervous_system.contracts.states import MarketState, SectorState
from core.nervous_system.persistence.uow import UnitOfWork as NervousSystemUnitOfWork

from .sector_map import sector_etf_for


UTC = timezone.utc
_EASTERN = ZoneInfo("America/New_York")
_MARKET_CLOSE = time(16, 0)

_PRODUCER = "signals.market_regime"
_MODEL_VERSION = "rules@1"
_FEATURE_VERSION = "market-regime@1"
_CONFIG_VERSION = "market-regime@1"
_UNKNOWN_RULE_VECTOR = "MARKET_REGIME_UNCLASSIFIED_RULE_VECTOR"
FINITE_ROW_PUBLICATION_POLICY = "leading-warmup-finite-required-metrics@1"

# These tuples mirror the producer's ordered output columns.  A row is
# publishable only when every required producer metric is finite.
MARKET_REQUIRED_METRICS = (
    "risk_appetite_xly_xlp_z",
    "risk_appetite_iwm_spy_z",
    "risk_appetite_hyg_iei_z",
    "risk_appetite_rsp_spy_z",
    "risk_appetite_z",
    "risk_appetite_z_n_components",
    "risk_appetite_z_stale_days",
    "liquidity_stress_amihud_z",
    "liquidity_stress_dollar_vol_z",
    "liquidity_stress_credit_z",
    "liquidity_stress_rv20_z",
    "liquidity_stress_z",
    "liquidity_stress_z_n_components",
    "liquidity_stress_z_stale_days",
    "credit_risk_ratio",
    "credit_risk_z",
    "credit_risk_z_n_components",
    "credit_risk_z_stale_days",
    "credit_risk_hyg_lqd_z",
    "breadth_raw",
    "breadth_z",
    "breadth_z_n_components",
    "breadth_z_stale_days",
    "sector_dispersion_raw",
    "sector_dispersion_z",
    "sector_dispersion_z_n_components",
    "sector_dispersion_z_stale_days",
    "spy_rv20_raw",
    "spy_rv20_z",
    "spy_rv20_z_n_components",
    "spy_rv20_z_stale_days",
    "spy_trend_state",
    "spy_trend_state_n_components",
    "spy_trend_state_stale_days",
)
SECTOR_REQUIRED_METRICS = (
    "excess_21d",
    "excess_63d",
    "rank_21d",
    "rank_63d",
    "rs_accel",
    "above_20d",
    "above_50d",
    "stale_days",
)
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")

_TIMESTAMP_FIELDS = frozenset(
    {
        "date",
        "session_date",
        "available_at",
        "generated_at",
        "valid_until",
        "source_window_start",
        "source_window_end",
    }
)
_IDENTITY_FIELDS = frozenset(
    {
        "entity_id",
        "ticker",
        "sector_id",
        "sector_etf",
        "source_id",
        "source_artifact_id",
        "content_hash",
        "artifact_hash",
        "source_hash",
        "record_locator",
        "source_row_locator",
        "lineage",
        "label",
        "regime",
        "sector_regime",
        "capital_flow_direction",
        "_row_index",
    }
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NaT or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, bool) and result


def _timestamp(value: object, *, field_name: str) -> datetime:
    if _is_missing(value):
        raise ValueError(f"{field_name} must be an explicit timezone-aware timestamp")
    if isinstance(value, pd.Timestamp):
        parsed = value.to_pydatetime()
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} is not an ISO-8601 timestamp") from exc
    else:
        raise ValueError(f"{field_name} must be an explicit timezone-aware timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(UTC)


def _session_date(row: Mapping[str, object]) -> date:
    value = row.get("date", row.get("session_date"))
    if _is_missing(value):
        raise ValueError("row must contain a market session date")
    if isinstance(value, datetime):
        if value.tzinfo is not None and value.utcoffset() is not None:
            return value.astimezone(_EASTERN).date()
        return value.date()
    if isinstance(value, date):
        return value
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("row date is not a valid market session") from exc
    if pd.isna(parsed):
        raise ValueError("row date is not a valid market session")
    if parsed.tzinfo is not None:
        return parsed.tz_convert(_EASTERN).date()
    return parsed.date()


def _session_close(session: date) -> datetime:
    return datetime.combine(session, _MARKET_CLOSE, tzinfo=_EASTERN).astimezone(UTC)


def _is_boolean_scalar(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _finite(value: object, *, field_name: str) -> float:
    if _is_boolean_scalar(value):
        raise ValueError(f"{field_name} must be a finite numeric value")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite numeric value") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _optional_finite(
    row: Mapping[str, object], names: tuple[str, ...], *, field_name: str
) -> float | None:
    value = next((row[name] for name in names if name in row), None)
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    return _finite(value, field_name=field_name)


def _metric_values(row: Mapping[str, object]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, value in row.items():
        if name in _TIMESTAMP_FIELDS or name in _IDENTITY_FIELDS or value is None:
            continue
        if _is_boolean_scalar(value):
            raise ValueError(f"{name} must be a finite numeric value, not boolean")
        if isinstance(value, (str, bytes, date, datetime, Mapping)):
            continue
        if not isinstance(value, (Real, Decimal)):
            try:
                is_null = pd.isna(value)
            except (TypeError, ValueError):
                continue
            if isinstance(is_null, bool) and is_null:
                raise ValueError(f"{name} must be finite")
            try:
                float(value)
            except (TypeError, ValueError):
                continue
        metrics[name] = _finite(value, field_name=name)
    return metrics


def _lineage_ids(lineage: tuple[LineageRef, ...]) -> tuple[str, ...]:
    """Encode every supplied lineage component in the state's ID-only field."""

    return tuple(
        json.dumps(
            {
                "content_hash": ref.content_hash,
                "record_locator": ref.record_locator,
                "source_id": ref.source_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for ref in lineage
    )


def _validated_lineage(lineage: tuple[LineageRef, ...]) -> tuple[LineageRef, ...]:
    if not lineage:
        raise ValueError("exact source artifact hash and record locator are required")
    validated: list[LineageRef] = []
    for ref in lineage:
        if not isinstance(ref, LineageRef):
            raise ValueError("lineage must contain LineageRef values")
        if not ref.source_id.strip():
            raise ValueError("lineage source_id must be non-empty")
        if _SHA256_RE.fullmatch(ref.content_hash) is None:
            raise ValueError("lineage content_hash must be an exact SHA-256 hex digest")
        if ref.record_locator is None or not ref.record_locator.strip():
            raise ValueError("exact original record locator is required for lineage")
        validated.append(
            ref.model_copy(
                update={
                    "source_id": ref.source_id.strip(),
                    "content_hash": ref.content_hash.lower(),
                    "record_locator": ref.record_locator.strip(),
                }
            )
        )
    return tuple(validated)


def _stable_state_id(
    *,
    state_type: StateType,
    entity_id: str,
    as_of: datetime,
    available_at: datetime,
    lineage: tuple[LineageRef, ...],
) -> UUID:
    material = {
        "state_type": state_type.value,
        "entity_id": entity_id,
        "as_of": as_of.isoformat(),
        "available_at": available_at.isoformat(),
        "schema_version": 1,
        "producer": _PRODUCER,
        "model_version": _MODEL_VERSION,
        "feature_version": _FEATURE_VERSION,
        "config_version": _CONFIG_VERSION,
        "lineage": [
            {
                "source_id": ref.source_id,
                "content_hash": ref.content_hash,
                "record_locator": ref.record_locator,
            }
            for ref in lineage
        ],
    }
    return uuid5(
        NAMESPACE_URL,
        json.dumps(material, sort_keys=True, separators=(",", ":")),
    )


def _common_state_values(
    row: Mapping[str, object],
    *,
    valid_until: datetime,
    lineage: tuple[LineageRef, ...],
    entity_id: str,
    state_type: StateType,
) -> dict[str, Any]:
    lineage = _validated_lineage(lineage)
    session = _session_date(row)
    as_of = _session_close(session)
    available_at = _timestamp(row.get("available_at"), field_name="available_at")
    generated_at = (
        _timestamp(row["generated_at"], field_name="generated_at")
        if "generated_at" in row and not _is_missing(row["generated_at"])
        else _utc_now()
    )
    valid_until_utc = _timestamp(valid_until, field_name="valid_until")
    source_window_start = (
        _timestamp(row["source_window_start"], field_name="source_window_start")
        if "source_window_start" in row and not _is_missing(row["source_window_start"])
        else as_of
    )
    source_window_end = (
        _timestamp(row["source_window_end"], field_name="source_window_end")
        if "source_window_end" in row and not _is_missing(row["source_window_end"])
        else as_of
    )
    if valid_until_utc <= available_at:
        raise ValueError("valid_until must be exclusive and after available_at")
    return {
        "state_id": _stable_state_id(
            state_type=state_type,
            entity_id=entity_id,
            as_of=as_of,
            available_at=available_at,
            lineage=lineage,
        ),
        "entity_id": entity_id,
        "as_of": as_of,
        "available_at": available_at,
        "generated_at": generated_at,
        "valid_until": valid_until_utc,
        "source_window_start": source_window_start,
        "source_window_end": source_window_end,
        "schema_version": 1,
        "producer": _PRODUCER,
        "model_version": _MODEL_VERSION,
        "feature_version": _FEATURE_VERSION,
        "config_version": _CONFIG_VERSION,
        "lineage_ids": _lineage_ids(lineage),
        "data_quality": DataQualitySummary(),
    }


def adapt_market_row(
    row: Mapping[str, object],
    *,
    valid_until: datetime,
    lineage: tuple[LineageRef, ...],
) -> MarketState:
    """Adapt one persisted daily-regime row without changing its semantics."""

    values = _common_state_values(
        row,
        valid_until=valid_until,
        lineage=lineage,
        entity_id=str(row.get("entity_id") or "US"),
        state_type=StateType.MARKET,
    )
    return MarketState(
        **values,
        regime=MarketRegime.UNKNOWN,
        risk_on_probability=None,
        risk_off_probability=None,
        metrics=_metric_values(row),
        reason_codes=(_UNKNOWN_RULE_VECTOR,),
    )


def _sector_id(row: Mapping[str, object]) -> str:
    ticker = row.get("ticker")
    if not _is_missing(ticker) and str(ticker).strip():
        mapped = sector_etf_for(str(ticker))
        if mapped is not None:
            return mapped.upper()
        # Frozen models use the former broad XLK fallback for unmapped names.
        provided = row.get("sector_id", row.get("sector_etf"))
        if _is_missing(provided):
            return "XLK"
    provided = row.get("sector_id", row.get("sector_etf"))
    if _is_missing(provided) or not str(provided).strip():
        raise ValueError("sector row must contain sector_etf or sector_id")
    return str(provided).upper()


def _sector_direction(row: Mapping[str, object]) -> Direction:
    value = row.get("capital_flow_direction")
    if _is_missing(value):
        return Direction.UNKNOWN
    try:
        return Direction(str(value).upper())
    except ValueError:
        return Direction.UNKNOWN


def adapt_sector_row(
    row: Mapping[str, object],
    *,
    valid_until: datetime,
    lineage: tuple[LineageRef, ...],
) -> SectorState:
    """Adapt one persisted sector-state row without probability inference."""

    sector_id = _sector_id(row)
    values = _common_state_values(
        row,
        valid_until=valid_until,
        lineage=lineage,
        entity_id=sector_id,
        state_type=StateType.SECTOR,
    )
    relative_strength = _optional_finite(
        row, ("relative_strength", "excess_21d"), field_name="relative_strength"
    )
    above_20d = _optional_finite(row, ("above_20d",), field_name="above_20d")
    above_50d = _optional_finite(row, ("above_50d",), field_name="above_50d")
    breadth_value = _optional_finite(row, ("breadth",), field_name="breadth")
    if breadth_value is None:
        breadth_values = [value for value in (above_20d, above_50d) if value is not None]
        breadth_value = (
            sum(breadth_values) / len(breadth_values) if breadth_values else None
        )
    rotation_rank = _optional_finite(
        row, ("rotation_rank", "rank_21d"), field_name="rotation_rank"
    )
    long_rank = _optional_finite(row, ("rank_63d",), field_name="rank_63d")
    rank_change = _optional_finite(row, ("rank_change",), field_name="rank_change")
    if rank_change is None and rotation_rank is not None and long_rank is not None:
        rank_change = rotation_rank - long_rank
    return SectorState(
        **values,
        sector_id=sector_id,
        sector_regime=str(row.get("sector_regime") or "UNKNOWN"),
        relative_strength=relative_strength,
        breadth=breadth_value,
        momentum=_optional_finite(row, ("momentum", "rs_accel"), field_name="momentum"),
        volatility=_optional_finite(row, ("volatility",), field_name="volatility"),
        rotation_rank=rotation_rank,
        rank_change=rank_change,
        capital_flow_direction=_sector_direction(row),
        transition_probabilities={},
    )


def _lineage_for_row(
    row: Mapping[str, object], *, table_name: str, row_index: object, attrs: Mapping[str, object]
) -> tuple[LineageRef, ...]:
    supplied = row.get("lineage")
    if isinstance(supplied, LineageRef):
        return _validated_lineage((supplied,))
    if isinstance(supplied, (tuple, list)):
        return _validated_lineage(tuple(supplied))
    if supplied is not None:
        raise ValueError("lineage must contain LineageRef values")

    source_id = next(
        (
            row.get(name)
            for name in ("source_id", "source_artifact_id")
            if not _is_missing(row.get(name))
        ),
        attrs.get("source_id"),
    )
    content_hash = next(
        (
            row.get(name)
            for name in ("content_hash", "artifact_hash", "source_hash")
            if not _is_missing(row.get(name))
        ),
        attrs.get("content_hash"),
    )
    locator = next(
        (
            row.get(name)
            for name in ("record_locator", "source_row_locator")
            if not _is_missing(row.get(name))
        ),
        None,
    )
    if _is_missing(locator):
        locators = attrs.get("record_locators")
        if isinstance(locators, Mapping):
            locator = locators.get(row_index)
            if _is_missing(locator):
                locator = locators.get(str(row_index))
    if _is_missing(source_id) or _is_missing(content_hash) or _is_missing(locator):
        raise ValueError(
            f"exact source lineage is required for {table_name} row {row_index}: "
            "artifact hash and original record locator are mandatory"
        )
    return _validated_lineage(
        (
            LineageRef(
                source_id=str(source_id),
                content_hash=str(content_hash),
                record_locator=str(locator),
            ),
        )
    )


def _rows(frame: pd.DataFrame) -> list[tuple[dict[str, object], object]]:
    return [(series.to_dict(), index) for index, series in frame.iterrows()]


def _required_metric_gaps(
    row: Mapping[str, object], required_metrics: tuple[str, ...]
) -> tuple[str, ...]:
    gaps: list[str] = []
    for name in required_metrics:
        if name not in row or _is_missing(row[name]) or _is_boolean_scalar(row[name]):
            gaps.append(name)
            continue
        try:
            value = float(row[name])
        except (TypeError, ValueError):
            gaps.append(name)
            continue
        if not math.isfinite(value):
            gaps.append(name)
    return tuple(gaps)


def _publication_rows(
    rows: list[tuple[dict[str, object], object]],
    *,
    table_name: str,
    required_metrics: tuple[str, ...],
    entity_for: Callable[[Mapping[str, object]], str],
) -> list[tuple[dict[str, object], object]]:
    grouped: dict[str, list[tuple[dict[str, object], object, int]]] = {}
    for ordinal, (row, row_index) in enumerate(rows):
        entity = entity_for(row)
        grouped.setdefault(entity, []).append((row, row_index, ordinal))

    selected: list[tuple[dict[str, object], object]] = []
    for entity in sorted(grouped):
        group = sorted(
            grouped[entity],
            key=lambda item: (_session_date(item[0]), item[2]),
        )
        publication_started = False
        for row, row_index, _ in group:
            gaps = _required_metric_gaps(row, required_metrics)
            if not gaps:
                publication_started = True
                selected.append((row, row_index))
                continue
            if publication_started:
                gap_text = ",".join(gaps)
                raise ValueError(
                    f"{FINITE_ROW_PUBLICATION_POLICY}: non-finite required metric "
                    f"after warm-up for {table_name} entity={entity} "
                    f"session={_session_date(row).isoformat()} metrics={gap_text}"
                )
    return selected


def _validate_unique_sector_rows(
    rows: list[tuple[dict[str, object], object]],
) -> None:
    seen: set[tuple[date, str]] = set()
    for row, _ in rows:
        key = (_session_date(row), _sector_id(row))
        if key in seen:
            raise ValueError(f"duplicate sector state for {key[0]} / {key[1]}")
        seen.add(key)


def persist_market_regime_outputs(
    daily_regime: pd.DataFrame,
    sector_state: pd.DataFrame,
    *,
    unit_of_work: NervousSystemUnitOfWork,
    valid_until_for: Callable[[datetime], datetime],
) -> tuple[int, int]:
    """Publish both producer tables through the caller-owned UOW."""

    market_rows = _publication_rows(
        _rows(daily_regime),
        table_name="daily_regime",
        required_metrics=MARKET_REQUIRED_METRICS,
        entity_for=lambda row: str(row.get("entity_id") or "US"),
    )
    sector_rows = _publication_rows(
        _rows(sector_state),
        table_name="sector_state",
        required_metrics=SECTOR_REQUIRED_METRICS,
        entity_for=_sector_id,
    )
    _validate_unique_sector_rows(sector_rows)
    states: list[MarketState | SectorState] = []
    market_states: list[MarketState] = []
    sector_states: list[SectorState] = []
    for row, row_index in market_rows:
        lineage = _lineage_for_row(
            row,
            table_name="daily_regime",
            row_index=row_index,
            attrs=daily_regime.attrs,
        )
        available_at = _timestamp(row.get("available_at"), field_name="available_at")
        market_states.append(
            adapt_market_row(
                row,
                valid_until=valid_until_for(available_at),
                lineage=lineage,
            )
        )
    for row, row_index in sector_rows:
        lineage = _lineage_for_row(
            row,
            table_name="sector_state",
            row_index=row_index,
            attrs=sector_state.attrs,
        )
        available_at = _timestamp(row.get("available_at"), field_name="available_at")
        sector_states.append(
            adapt_sector_row(
                row,
                valid_until=valid_until_for(available_at),
                lineage=lineage,
            )
        )
    states.extend(market_states)
    states.extend(sector_states)
    if not states:
        return 0, 0
    unit_of_work.states.insert_states_idempotently(states)
    return len(market_states), len(sector_states)


__all__ = [
    "FINITE_ROW_PUBLICATION_POLICY",
    "MARKET_REQUIRED_METRICS",
    "NervousSystemUnitOfWork",
    "SECTOR_REQUIRED_METRICS",
    "adapt_market_row",
    "adapt_sector_row",
    "persist_market_regime_outputs",
]
