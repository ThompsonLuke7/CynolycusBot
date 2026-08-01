"""Causal catalyst/news/event adapters for the nervous system.

The Parquet producer tables remain the analytical authority.  This module
only normalizes rows that already carry source evidence into immutable state
contracts and optionally inserts those contracts through a caller-owned UOW.
It deliberately keeps raw directional scores separate from probability
fields: an uncalibrated score is never promoted to a probability.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import re
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import pandas as pd

from core.nervous_system.contracts.enums import DataQualitySeverity, StateType
from core.nervous_system.contracts.quality import (
    DataQualityIssue,
    DataQualitySummary,
    LineageRef,
)
from core.nervous_system.contracts.states import CatalystEvent, CatalystPressure
from core.nervous_system.persistence.uow import UnitOfWork as NervousSystemUnitOfWork


UTC = timezone.utc
TIMESTAMP_SEMANTICS_VERSION = "catalyst-time@1"
CATALYST_VALIDITY = timedelta(days=1)
_PRODUCER = "signals.catalysts"
_MODEL_VERSION = "catalyst-adapter@1"
_FEATURE_VERSION = "catalyst-time@1"
_DEFAULT_CONFIG_VERSION = "catalyst@1"
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")

_RUNTIME_FIELDS = frozenset(
    {
        "observed_at",
        "available_at",
        "generated_at",
        "valid_until",
        "source_artifact_hash",
        "timestamp_semantics_version",
    }
)
_NON_IDENTITY_FIELDS = _RUNTIME_FIELDS | frozenset(
    {
        "path",
        "file_path",
        "source_path",
        "artifact_path",
        "source_uri",
        "uri",
        "file_mtime",
        "mtime",
        "scored_at",
        "created_at",
        "updated_at",
        "collection_time",
        "captured_at",
    }
)
_HINDSIGHT_EARNINGS_FIELDS = frozenset(
    {
        "fwd_ret_1d",
        "fwd_ret_5d",
        "fwd_ret_10d",
        "fwd_ret_20d",
        "max_drawdown",
        "label",
        "target",
        "metric_revenue_actual",
        "metric_eps_actual",
    }
)


class AdapterIssue(ValueError):
    """A deterministic quarantine reason for one source row."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CatalystAdapterResult:
    event: CatalystEvent | None
    warnings: tuple[str, ...]
    quarantine_code: str | None
    quarantine_message: str | None


def _missing(value: object) -> bool:
    if value is None or value is pd.NaT or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, bool) and result


def _first(row: Mapping[str, object], names: tuple[str, ...]) -> object | None:
    for name in names:
        value = row.get(name)
        if _missing(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _timestamp(value: object, *, field_name: str) -> datetime:
    if _missing(value):
        raise AdapterIssue(
            f"MISSING_{field_name.upper()}",
            f"{field_name} must be an explicit timezone-aware timestamp",
        )
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
            raise AdapterIssue(
                f"INVALID_{field_name.upper()}",
                f"{field_name} is not an ISO-8601 timestamp",
            ) from exc
    else:
        raise AdapterIssue(
            f"INVALID_{field_name.upper()}",
            f"{field_name} must be an explicit timezone-aware timestamp",
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AdapterIssue(
            f"NAIVE_{field_name.upper()}",
            f"{field_name} must be timezone-aware",
        )
    return parsed.astimezone(UTC)


def _optional_timestamp(row: Mapping[str, object], names: tuple[str, ...], *, field_name: str) -> datetime | None:
    value = _first(row, names)
    return None if value is None else _timestamp(value, field_name=field_name)


def _canonical(value: object) -> object:
    if _missing(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().astimezone(UTC).isoformat() if value.tzinfo else value.isoformat()
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat() if value.tzinfo else value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _canonical(nested) for key, nested in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), default=str)


def _validated_lineage(source_artifact: LineageRef) -> LineageRef:
    if not isinstance(source_artifact, LineageRef):
        raise AdapterIssue("INVALID_SOURCE_LINEAGE", "source_artifact must be a LineageRef")
    if not source_artifact.source_id.strip():
        raise AdapterIssue("INVALID_SOURCE_LINEAGE", "source artifact source_id must be non-empty")
    if _SHA256_RE.fullmatch(source_artifact.content_hash) is None:
        raise AdapterIssue("INVALID_SOURCE_LINEAGE", "source artifact content_hash must be SHA-256")
    if source_artifact.record_locator is None or not source_artifact.record_locator.strip():
        raise AdapterIssue("MISSING_SOURCE_LINEAGE", "source artifact record_locator is required")
    return source_artifact.model_copy(
        update={
            "source_id": source_artifact.source_id.strip(),
            "content_hash": source_artifact.content_hash.lower(),
            "record_locator": source_artifact.record_locator.strip(),
        }
    )


def _lineage_id(lineage: LineageRef) -> str:
    return json.dumps(
        {
            "content_hash": lineage.content_hash,
            "record_locator": lineage.record_locator,
            "source_id": lineage.source_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _source_logical_key(row: Mapping[str, object], *, source_artifact: LineageRef) -> str:
    source = str(_first(row, ("source", "channel")) or source_artifact.source_id).strip()
    source_record_id = _first(row, ("source_record_id", "source_id", "id", "record_id"))
    if source_record_id is None:
        source_material = {
            name: row.get(name)
            for name in (
                "ticker",
                "event_type",
                "event_time",
                "published_at",
                "timestamp",
                "headline",
                "title",
                "summary",
                "body",
                "url",
                "score",
                "raw_score",
                "directional_score",
                "catalyst_score",
            )
        }
        source_record_id = hashlib.sha256(_canonical_json(source_material).encode("utf-8")).hexdigest()[:32]
    return f"{source}:{str(source_record_id).strip()}"


def _revision_hash(row: Mapping[str, object]) -> str:
    material = {
        str(key): value
        for key, value in row.items()
        if key not in _NON_IDENTITY_FIELDS and key not in {"record_locator", "source_row_locator"}
    }
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def _event_type(row: Mapping[str, object]) -> str:
    value = _first(row, ("event_type", "type", "catalyst_kind"))
    if value is None:
        raise AdapterIssue("MISSING_EVENT_TYPE", "catalyst row must contain event_type")
    normalized = str(value).strip()
    if not normalized:
        raise AdapterIssue("MISSING_EVENT_TYPE", "catalyst row must contain event_type")
    return normalized


def _ticker(row: Mapping[str, object]) -> str | None:
    value = _first(row, ("ticker", "symbol"))
    if value is None:
        return None
    cleaned = str(value).upper().replace("$", "").strip()
    return cleaned or None


def _channel(row: Mapping[str, object], *, ticker: str | None) -> str:
    value = _first(row, ("channel", "catalyst_kind", "source"))
    if value is None:
        return "NEWS" if ticker else "SCHEDULED"
    return str(value).strip().upper() or ("NEWS" if ticker else "SCHEDULED")


def _optional_probability(row: Mapping[str, object]) -> float | None:
    value = _first(row, ("relation_confidence", "relationship_confidence"))
    if value is None:
        return None
    if isinstance(value, bool):
        raise AdapterIssue("INVALID_RELATION_CONFIDENCE", "relation_confidence must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterIssue("INVALID_RELATION_CONFIDENCE", "relation_confidence must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise AdapterIssue("INVALID_RELATION_CONFIDENCE", "relation_confidence must be in [0, 1]")
    return result


def _optional_bool(row: Mapping[str, object]) -> bool | None:
    value = _first(row, ("is_direct", "is_direct_catalyst"))
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0"}:
        return False
    return None


def _is_hindsight_earnings_row(row: Mapping[str, object]) -> bool:
    event_type = str(_first(row, ("event_type", "type", "catalyst_kind")) or "").lower()
    kind = str(row.get("catalyst_kind") or "").lower()
    if "earnings_result" in event_type or "earnings_result" in kind:
        return True
    if "earnings" not in event_type and "earnings" not in kind:
        return False
    return any(
        key in row and not _missing(row.get(key))
        for key in _HINDSIGHT_EARNINGS_FIELDS
    )


def _quality_issue(code: str, message: str) -> DataQualityIssue:
    return DataQualityIssue(
        code=code,
        severity=DataQualitySeverity.WARNING,
        component=_PRODUCER,
        message=message,
    )


def _with_quality_issue(event: CatalystEvent, code: str, message: str) -> CatalystEvent:
    return event.model_copy(
        update={
            "data_quality": DataQualitySummary(
                issues=tuple(event.data_quality.issues) + (_quality_issue(code, message),)
            )
        }
    )


def _quarantine(code: str, message: str, warnings: Iterable[str] = ()) -> CatalystAdapterResult:
    return CatalystAdapterResult(
        event=None,
        warnings=tuple(dict.fromkeys(warnings)),
        quarantine_code=code,
        quarantine_message=message,
    )


def normalize_catalyst_record(
    row: Mapping[str, object],
    *,
    source_artifact: LineageRef,
) -> CatalystAdapterResult:
    """Normalize one source row while keeping event and availability time distinct."""

    warnings: list[str] = []
    try:
        lineage = _validated_lineage(source_artifact)
        if _is_hindsight_earnings_row(row):
            return _quarantine(
                "HINDSIGHT_EARNINGS_EVIDENCE",
                "earnings result labels/features are not eligible decision-time catalyst evidence",
            )

        event_type = _event_type(row)
        ticker = _ticker(row)
        source = str(_first(row, ("source",)) or lineage.source_id).strip()
        headline_value = _first(row, ("headline", "title"))
        headline = None if headline_value is None else str(headline_value).strip()
        channel = _channel(row, ticker=ticker)

        event_time = _optional_timestamp(row, ("event_time",), field_name="event_time")
        timestamp_value = _first(row, ("timestamp",))
        published_at = _optional_timestamp(row, ("published_at",), field_name="published_at")
        observed_at = _optional_timestamp(
            row,
            ("observed_at", "collection_time", "captured_at"),
            field_name="observed_at",
        )
        available_value = _first(row, ("available_at",))
        if available_value is None:
            if observed_at is None:
                raise AdapterIssue(
                    "MISSING_AVAILABLE_AT",
                    "catalyst row has no reliable explicit availability timestamp",
                )
            available_at = observed_at
        else:
            available_at = _timestamp(available_value, field_name="available_at")
        if observed_at is None:
            raise AdapterIssue(
                "MISSING_OBSERVED_AT",
                "catalyst row has availability but no observation timestamp",
            )
        if event_time is None and timestamp_value is not None:
            # Compatibility timestamp can describe the occurrence, but never
            # supplies availability.  Legacy rows without explicit metadata
            # have already quarantined above with MISSING_AVAILABLE_AT.
            event_time = _timestamp(timestamp_value, field_name="event_time")
            warnings.append("AMBIGUOUS_TIMESTAMP_SEMANTICS")
        if observed_at > available_at:
            raise AdapterIssue(
                "OBSERVED_AFTER_AVAILABLE_AT",
                "observed_at cannot follow available_at",
            )
        if published_at is not None and published_at > observed_at:
            raise AdapterIssue(
                "PUBLISHED_AFTER_OBSERVED_AT",
                "published_at cannot follow observed_at",
            )

        if published_at is None:
            warnings.append("MISSING_PUBLISHED_AT")

        explicit_valid_until = _first(row, ("valid_until",))
        valid_until = (
            _timestamp(explicit_valid_until, field_name="valid_until")
            if explicit_valid_until is not None
            else available_at + CATALYST_VALIDITY
        )
        explicit_generated_at = _first(row, ("generated_at",))
        generated_at = (
            _timestamp(explicit_generated_at, field_name="generated_at")
            if explicit_generated_at is not None
            else available_at
        )
        as_of = min(event_time, available_at) if event_time is not None else available_at
        source_window_start_value = _first(row, ("source_window_start",))
        source_window_end_value = _first(row, ("source_window_end",))
        source_window_start = (
            _timestamp(source_window_start_value, field_name="source_window_start")
            if source_window_start_value is not None
            else as_of
        )
        source_window_end = (
            _timestamp(source_window_end_value, field_name="source_window_end")
            if source_window_end_value is not None
            else as_of
        )

        source_hash = _first(row, ("source_artifact_hash",))
        if source_hash is not None and str(source_hash).lower() != lineage.content_hash:
            warnings.append("CONFLICTING_SOURCE_ARTIFACT_HASH")

        logical_key = _source_logical_key(row, source_artifact=lineage)
        revision_hash = _revision_hash(row)
        event_id = uuid5(
            NAMESPACE_URL,
            _canonical_json(
                {
                    "state_type": StateType.CATALYST_EVENT.value,
                    "source_logical_key": logical_key,
                    "revision_hash": revision_hash,
                    "lineage": {
                        "source_id": lineage.source_id,
                        "content_hash": lineage.content_hash,
                        "record_locator": lineage.record_locator,
                    },
                }
            ),
        )

        raw_score_present = any(
            key in row and not _missing(row.get(key))
            for key in ("score", "raw_score", "directional_score", "catalyst_score")
        )
        if raw_score_present:
            warnings.append("RAW_DIRECTIONAL_SCORE_NOT_PROBABILITY")

        issues = [
            _quality_issue(code, f"catalyst adapter warning: {code}")
            for code in warnings
            if code not in {"AMBIGUOUS_TIMESTAMP_SEMANTICS"}
        ]
        event = CatalystEvent(
            state_id=event_id,
            state_type=StateType.CATALYST_EVENT,
            entity_id=str(event_id),
            as_of=as_of,
            available_at=available_at,
            generated_at=generated_at,
            valid_until=valid_until,
            source_window_start=source_window_start,
            source_window_end=source_window_end,
            schema_version=1,
            producer=_PRODUCER,
            model_version=_MODEL_VERSION,
            feature_version=_FEATURE_VERSION,
            config_version=str(row.get("config_version") or _DEFAULT_CONFIG_VERSION),
            lineage_ids=(_lineage_id(lineage),),
            data_quality=DataQualitySummary(issues=tuple(issues)),
            event_id=event_id,
            ticker=ticker,
            event_type=event_type,
            event_time=event_time,
            published_at=published_at,
            observed_at=observed_at,
            source=source,
            headline=headline,
            channel=channel,
            relation_confidence=_optional_probability(row),
            is_direct=_optional_bool(row),
        )
        return CatalystAdapterResult(
            event=event,
            warnings=tuple(dict.fromkeys(warnings)),
            quarantine_code=None,
            quarantine_message=None,
        )
    except AdapterIssue as exc:
        return _quarantine(exc.code, str(exc), warnings)
    except (TypeError, ValueError) as exc:
        return _quarantine("INVALID_CATALYST_RECORD", str(exc), warnings)


def _logical_key_for_result(row: Mapping[str, object], source_artifact: LineageRef) -> str:
    return _source_logical_key(row, source_artifact=source_artifact)


def _batch_lineage(
    source_artifact: LineageRef,
    row: Mapping[str, object],
    row_index: int,
) -> LineageRef:
    """Expand the pipeline's batch locator into stable row locators.

    Direct callers pass the exact source-artifact/record lineage and it is
    preserved byte-for-byte.  Only the adapter's own synthetic batch marker
    receives a deterministic row suffix; no filesystem path or runtime value
    participates.
    """

    if source_artifact.record_locator != "catalyst_records:batch":
        return source_artifact
    explicit = _first(row, ("record_locator", "source_record_locator"))
    source_record_id = _first(row, ("source_record_id", "source_id", "id", "record_id"))
    locator = str(explicit or f"catalyst_records:row:{source_record_id or row_index}").strip()
    return source_artifact.model_copy(update={"record_locator": locator})


def normalize_catalyst_records(
    rows: Iterable[Mapping[str, object]],
    *,
    source_artifact: LineageRef,
) -> tuple[CatalystAdapterResult, ...]:
    """Normalize a batch, converging identical revisions and retaining conflicts."""

    lineage = _validated_lineage(source_artifact)
    results: list[CatalystAdapterResult] = []
    by_event_id: dict[UUID, int] = {}
    by_logical_key: dict[str, set[UUID]] = {}
    for row_index, row in enumerate(rows):
        row_lineage = _batch_lineage(lineage, row, row_index)
        result = normalize_catalyst_record(row, source_artifact=row_lineage)
        event = result.event
        if event is None:
            results.append(result)
            continue
        if event.event_id in by_event_id:
            existing_index = by_event_id[event.event_id]
            existing = results[existing_index]
            merged_warnings = tuple(dict.fromkeys(existing.warnings + result.warnings))
            results[existing_index] = replace(existing, warnings=merged_warnings)
            continue
        key = _logical_key_for_result(row, row_lineage)
        prior_ids = by_logical_key.setdefault(key, set())
        if prior_ids:
            message = "source logical key has conflicting observed source values/revisions"
            warning = "CONFLICTING_SOURCE_VALUES"
            result = replace(
                result,
                warnings=tuple(dict.fromkeys(result.warnings + (warning,))),
                event=_with_quality_issue(event, warning, message),
            )
            for prior_id in prior_ids:
                prior_index = by_event_id[prior_id]
                prior_result = results[prior_index]
                if prior_result.event is not None:
                    results[prior_index] = replace(
                        prior_result,
                        warnings=tuple(dict.fromkeys(prior_result.warnings + (warning,))),
                        event=_with_quality_issue(prior_result.event, warning, message),
                    )
        prior_ids.add(event.event_id)
        by_event_id[event.event_id] = len(results)
        results.append(result)
    return tuple(results)


def _decision_time(value: datetime, *, field_name: str) -> datetime:
    return _timestamp(value, field_name=field_name)


def _scope_type(entity_id: str) -> str:
    return "MARKET" if entity_id.upper() in {"US", "MARKET"} else "TICKER"


def _pressure_id(
    *,
    entity_id: str,
    decision_time: datetime,
    event_ids: tuple[UUID, ...],
    channel_scores: Mapping[str, float],
    config_version: str,
) -> UUID:
    return uuid5(
        NAMESPACE_URL,
        _canonical_json(
            {
                "state_type": StateType.CATALYST_PRESSURE.value,
                "scope_id": entity_id,
                "decision_time": decision_time.isoformat(),
                "event_ids": [str(event_id) for event_id in event_ids],
                "channel_scores": dict(channel_scores),
                "config_version": config_version,
            }
        ),
    )


def _aggregate_pressure(
    events: Sequence[CatalystEvent],
    *,
    entity_id: str,
    decision_time: datetime,
    valid_until: datetime,
    config_version: str,
    raw_scores: Mapping[UUID, float] | None = None,
) -> CatalystPressure:
    decision = _decision_time(decision_time, field_name="decision_time")
    valid_until_utc = _decision_time(valid_until, field_name="valid_until")
    if valid_until_utc <= decision:
        raise ValueError("valid_until must be after decision_time")
    scores = raw_scores or {}
    eligible: list[CatalystEvent] = []
    late_count = 0
    for event in events:
        if event.available_at > decision:
            late_count += 1
            continue
        if event.ticker is not None and event.ticker != entity_id:
            continue
        eligible.append(event)
    eligible.sort(key=lambda event: str(event.event_id))
    event_ids = tuple(event.event_id for event in eligible)

    per_channel: dict[str, list[float]] = {}
    for event in eligible:
        raw_score = scores.get(event.event_id)
        if raw_score is None:
            continue
        value = float(raw_score)
        if not math.isfinite(value):
            raise ValueError("raw directional score must be finite")
        per_channel.setdefault(event.channel, []).append(value)
    channel_scores = {
        channel: sum(values) / len(values)
        for channel, values in sorted(per_channel.items())
    }
    all_scores = [value for values in per_channel.values() for value in values]
    aggregate_score = sum(all_scores) / len(all_scores) if all_scores else None

    quality_issues: list[DataQualityIssue] = []
    if late_count:
        quality_issues.append(
            _quality_issue(
                "LATE_OBSERVATION_EXCLUDED",
                f"{late_count} catalyst event(s) were unavailable at decision_time",
            )
        )
    if all_scores:
        quality_issues.append(
            _quality_issue(
                "RAW_DIRECTIONAL_SCORE_NOT_PROBABILITY",
                "aggregate_score preserves the producer score scale; no probability was synthesized",
            )
        )

    source_window_start = min((event.available_at for event in eligible), default=decision)
    pressure_id = _pressure_id(
        entity_id=entity_id,
        decision_time=decision,
        event_ids=event_ids,
        channel_scores=channel_scores,
        config_version=config_version,
    )
    lineage_ids = tuple(
        sorted({lineage_id for event in eligible for lineage_id in event.lineage_ids})
    )
    return CatalystPressure(
        state_id=pressure_id,
        state_type=StateType.CATALYST_PRESSURE,
        entity_id=entity_id,
        as_of=decision,
        available_at=decision,
        generated_at=decision,
        valid_until=valid_until_utc,
        source_window_start=source_window_start,
        source_window_end=decision,
        schema_version=1,
        producer=_PRODUCER,
        model_version=_MODEL_VERSION,
        feature_version=_FEATURE_VERSION,
        config_version=config_version,
        lineage_ids=lineage_ids,
        data_quality=DataQualitySummary(issues=tuple(quality_issues)),
        scope_type=_scope_type(entity_id),
        scope_id=entity_id,
        channel_scores=channel_scores,
        aggregate_score=aggregate_score,
        event_ids=event_ids,
        # Raw directional scores are intentionally not copied into a
        # probability map without a calibrated model.
        transition_probabilities={},
    )


def aggregate_catalyst_pressure(
    events: Sequence[CatalystEvent],
    *,
    entity_id: str,
    decision_time: datetime,
    valid_until: datetime,
    config_version: str,
) -> CatalystPressure:
    """Aggregate only events available at the decision timestamp."""

    return _aggregate_pressure(
        events,
        entity_id=entity_id,
        decision_time=decision_time,
        valid_until=valid_until,
        config_version=config_version,
    )


def _raw_score(row: Mapping[str, object]) -> float | None:
    value = _first(row, ("directional_score", "raw_score", "catalyst_score", "score"))
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("raw directional score must be numeric") from exc
    if not math.isfinite(score):
        raise ValueError("raw directional score must be finite")
    return score


def publish_catalyst_states(
    rows: Iterable[Mapping[str, object]],
    *,
    unit_of_work: NervousSystemUnitOfWork,
    source_artifact: LineageRef,
    decision_time: datetime | None = None,
    valid_until: datetime | None = None,
    config_version: str = _DEFAULT_CONFIG_VERSION,
) -> tuple[CatalystAdapterResult, ...]:
    """Validate and publish through the caller-owned UOW without committing."""

    materialized_rows = tuple(rows)
    results = normalize_catalyst_records(materialized_rows, source_artifact=source_artifact)
    events = tuple(result.event for result in results if result.event is not None)
    states: list[CatalystEvent | CatalystPressure] = list(events)
    if decision_time is not None:
        decision = _decision_time(decision_time, field_name="decision_time")
        pressure_valid_until = valid_until or decision + CATALYST_VALIDITY
        raw_scores: dict[UUID, float] = {}
        for row, result in zip(materialized_rows, results):
            if result.event is None:
                continue
            score = _raw_score(row)
            if score is not None:
                raw_scores[result.event.event_id] = score
        states.append(
            _aggregate_pressure(
                events,
                entity_id=str(_first(materialized_rows[0], ("entity_id", "ticker")) or "US")
                if materialized_rows
                else "US",
                decision_time=decision,
                valid_until=pressure_valid_until,
                config_version=config_version,
                raw_scores=raw_scores,
            )
        )
    if states:
        unit_of_work.states.insert_states_idempotently(tuple(states))
    return results


persist_catalyst_outputs = publish_catalyst_states


__all__ = [
    "CATALYST_VALIDITY",
    "CatalystAdapterResult",
    "NervousSystemUnitOfWork",
    "aggregate_catalyst_pressure",
    "normalize_catalyst_record",
    "normalize_catalyst_records",
    "persist_catalyst_outputs",
    "publish_catalyst_states",
]
