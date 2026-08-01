"""Adapt the legacy readiness result into the causal readiness contract."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from enum import Enum
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from core.nervous_system.contracts.enums import DataQualitySeverity, StateType
from core.nervous_system.contracts.quality import DataQualityIssue, DataQualitySummary
from core.nervous_system.contracts.states import ReadinessState


UTC = timezone.utc
_PRODUCER = "core.live_readiness"
_MODEL_VERSION = "readiness-adapter@1"
_FEATURE_VERSION = "readiness@1"
_CONFIG_VERSION = "readiness-policy@1"
_DEFAULT_JOB = "UNKNOWN"
_DEFAULT_SESSION = "UNKNOWN"
_PREDATES_LATEST_SESSION_REASON = (
    "readiness stamp predates latest completed trading session"
)


def _timestamp(value: object, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
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
        raise ValueError(f"{field_name} must be a timezone-aware timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(UTC)


def _canonical_value(value: object) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("payload contains a non-finite number")
    return value


def _source_hash(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        _canonical_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _text(value: object, *, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _quality_issue(
    code: str,
    message: str,
    *,
    severity: DataQualitySeverity,
) -> DataQualityIssue:
    return DataQualityIssue(
        code=code,
        severity=severity,
        component=_PRODUCER,
        message=message,
        fallback_used="ready=False" if severity is not DataQualitySeverity.WARNING else None,
    )


def _stable_state_id(
    *,
    job: str,
    as_of: datetime,
    available_at: datetime,
    generated_at: datetime,
    status: str,
    ready: bool,
    reason_codes: tuple[str, ...],
    source_hash: str,
    max_age_hours: float,
) -> str:
    material = {
        "state_type": StateType.READINESS.value,
        "job": job,
        "as_of": as_of.isoformat(),
        "available_at": available_at.isoformat(),
        "generated_at": generated_at.isoformat(),
        "status": status,
        "ready": ready,
        "reason_codes": reason_codes,
        "source_hash": source_hash,
        "max_age_hours": max_age_hours,
        "schema_version": 1,
        "producer": _PRODUCER,
        "model_version": _MODEL_VERSION,
        "feature_version": _FEATURE_VERSION,
        "config_version": _CONFIG_VERSION,
    }
    return str(
        uuid5(
            NAMESPACE_URL,
            json.dumps(material, sort_keys=True, separators=(",", ":")),
        )
    )


def adapt_readiness_status(
    *,
    ready: bool,
    reason: str,
    payload: Mapping[str, object],
    checked_at: datetime,
    max_age_hours: float,
) -> ReadinessState:
    """Publish one readiness observation without changing execution behavior.

    ``ready`` is supplied by :func:`core.live_readiness.readiness_status`, so
    its latest-completed-session rule remains authoritative.  The adapter only
    validates the supplied evidence and records the observation; it never
    reads the readiness file, calls a broker, or invents a completion time.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    checked_at_utc = _timestamp(checked_at, field_name="checked_at")
    try:
        max_age = float(max_age_hours)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_age_hours must be a positive finite number") from exc
    if not math.isfinite(max_age) or max_age <= 0:
        raise ValueError("max_age_hours must be a positive finite number")

    source_hash = _source_hash(payload)
    job = _text(payload.get("job"), default=_DEFAULT_JOB)
    latest_required_session = _text(
        payload.get("latest_required_session"),
        default=_DEFAULT_SESSION,
    )
    raw_completed_at = payload.get("completed_at_utc")
    reason_text = _text(reason, default="readiness result did not provide a reason")
    reason_codes: list[str] = []
    issues: list[DataQualityIssue] = []
    completed_at: datetime | None = None
    status: str
    available_at: datetime
    effective_ready = bool(ready) if isinstance(ready, bool) else False
    environment = os.getenv("CYNOLYCUS_ENVIRONMENT", "UNKNOWN")
    environment = environment.strip().replace("-", "_").upper()
    gate_disabled = os.getenv("CYNOLYCUS_READINESS_REQUIRED", "1").strip() == "0"

    if gate_disabled:
        status = "DISABLED"
        available_at = checked_at_utc
        if environment == "QA_PAPER":
            effective_ready = False
            reason_codes.append("READINESS_DISABLED_NOT_ACCEPTED_IN_QA")
            issues.append(
                _quality_issue(
                    "READINESS_DISABLED_NOT_ACCEPTED_IN_QA",
                    "CYNOLYCUS_READINESS_REQUIRED=0 is not accepted in QA-paper",
                    severity=DataQualitySeverity.ERROR,
                )
            )
        elif environment == "DEVELOPMENT":
            reason_codes.append("READINESS_DISABLED_WARNING")
            issues.append(
                _quality_issue(
                    "READINESS_DISABLED_WARNING",
                    "CYNOLYCUS_READINESS_REQUIRED=0 is active in development",
                    severity=DataQualitySeverity.WARNING,
                )
            )
        elif environment == "PRODUCTION_LIVE":
            effective_ready = False
            reason_codes.append("READINESS_DISABLED_NOT_ACCEPTED_IN_LIVE")
            issues.append(
                _quality_issue(
                    "READINESS_DISABLED_NOT_ACCEPTED_IN_LIVE",
                    "CYNOLYCUS_READINESS_REQUIRED=0 is not accepted in production-live",
                    severity=DataQualitySeverity.ERROR,
                )
            )
        else:
            effective_ready = False
            reason_codes.append("READINESS_DISABLED_NOT_ACCEPTED_IN_RUNTIME")
            issues.append(
                _quality_issue(
                    "READINESS_DISABLED_NOT_ACCEPTED_IN_RUNTIME",
                    "CYNOLYCUS_READINESS_REQUIRED=0 requires an explicit supported environment",
                    severity=DataQualitySeverity.ERROR,
                )
            )
    elif raw_completed_at is None or (
        isinstance(raw_completed_at, str) and not raw_completed_at.strip()
    ):
        status = "MISSING"
        available_at = checked_at_utc
        effective_ready = False
        reason_codes.append("READINESS_MISSING")
        issues.append(
            _quality_issue(
                "READINESS_MISSING",
                reason_text,
                severity=DataQualitySeverity.ERROR,
            )
        )
    else:
        try:
            completed_at = _timestamp(raw_completed_at, field_name="completed_at_utc")
        except ValueError as exc:
            status = "INVALID"
            available_at = checked_at_utc
            effective_ready = False
            reason_codes.append("READINESS_INVALID_TIMESTAMP")
            issues.append(
                _quality_issue(
                    "READINESS_INVALID_TIMESTAMP",
                    str(exc),
                    severity=DataQualitySeverity.ERROR,
                )
            )
        else:
            if completed_at > checked_at_utc:
                status = "INVALID"
                available_at = checked_at_utc
                completed_at = None
                effective_ready = False
                reason_codes.append("READINESS_INVALID_TIMESTAMP")
                issues.append(
                    _quality_issue(
                        "READINESS_INVALID_TIMESTAMP",
                        "completed_at_utc follows checked_at",
                        severity=DataQualitySeverity.ERROR,
                    )
                )
            elif payload.get("status") != "success":
                status = "INVALID"
                available_at = completed_at
                effective_ready = False
                reason_codes.append("READINESS_STATUS_NOT_SUCCESS")
                issues.append(
                    _quality_issue(
                        "READINESS_STATUS_NOT_SUCCESS",
                        reason_text,
                        severity=DataQualitySeverity.ERROR,
                    )
                )
            else:
                available_at = completed_at
                age_hours = (checked_at_utc - completed_at).total_seconds() / 3600.0
                if age_hours > max_age:
                    status = "STALE"
                    effective_ready = False
                    reason_codes.append("READINESS_STALE")
                    issues.append(
                        _quality_issue(
                            "READINESS_STALE",
                            reason_text,
                            severity=DataQualitySeverity.ERROR,
                        )
                    )
                elif not effective_ready:
                    status = "STALE"
                    code = (
                        "READINESS_PREDATES_LATEST_COMPLETED_SESSION"
                        if reason_text.startswith(_PREDATES_LATEST_SESSION_REASON)
                        else "READINESS_STALE"
                    )
                    reason_codes.append(code)
                    issues.append(
                        _quality_issue(
                            code,
                            reason_text,
                            severity=DataQualitySeverity.ERROR,
                        )
                    )
                else:
                    status = "CURRENT"

    reason_codes_tuple = tuple(dict.fromkeys(reason_codes))
    data_quality = DataQualitySummary(issues=tuple(issues))
    as_of = available_at
    valid_until = available_at + timedelta(hours=max_age)
    generated_at = checked_at_utc
    state_id = _stable_state_id(
        job=job,
        as_of=as_of,
        available_at=available_at,
        generated_at=generated_at,
        status=status,
        ready=effective_ready,
        reason_codes=reason_codes_tuple,
        source_hash=source_hash,
        max_age_hours=max_age,
    )
    return ReadinessState(
        state_id=state_id,
        state_type=StateType.READINESS,
        entity_id=job,
        as_of=as_of,
        available_at=available_at,
        generated_at=generated_at,
        valid_until=valid_until,
        source_window_start=as_of,
        source_window_end=as_of,
        schema_version=1,
        producer=_PRODUCER,
        model_version=_MODEL_VERSION,
        feature_version=_FEATURE_VERSION,
        config_version=_CONFIG_VERSION,
        lineage_ids=(source_hash,),
        data_quality=data_quality,
        job=job,
        status=status,
        ready=effective_ready,
        completed_at=completed_at,
        checked_at=checked_at_utc,
        max_age_hours=max_age,
        latest_required_session=latest_required_session,
        reason_codes=reason_codes_tuple,
    )


__all__ = ["adapt_readiness_status"]
