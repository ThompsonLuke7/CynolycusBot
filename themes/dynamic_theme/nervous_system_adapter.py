"""Adapters from dynamic-theme artifacts to immutable nervous-system states."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, time, timezone
import json
import math
import re
from numbers import Real
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, UUID, uuid5
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core.nervous_system.contracts.enums import DataQualitySeverity, StateType, ThemeRegime
from core.nervous_system.contracts.quality import DataQualityIssue, DataQualitySummary, LineageRef
from core.nervous_system.contracts.states import ThemeState
from themes.dynamic_theme.stages.step08_memberships import canonical_theme_id

if TYPE_CHECKING:
    from core.nervous_system.persistence.uow import UnitOfWork


UTC = timezone.utc
_EASTERN = ZoneInfo("America/New_York")
_MARKET_CLOSE = time(16, 0)
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_PRODUCER = "themes.dynamic_theme"
_MODEL_VERSION = "theme-taxonomy@1"
_FEATURE_VERSION = "theme-features@1"
_CONFIG_VERSION = "theme-state@1"
_IDENTITY_COLUMNS = {
    "ticker",
    "theme",
    "primary_theme",
    "date",
    "as_of",
    "available_at",
    "generated_at",
    "taxonomy_version",
    "producer_version",
    "membership_score",
}


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NaT or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _timestamp(value: object, *, field_name: str) -> datetime:
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
        raise ValueError(f"{field_name} must be a timezone-aware timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(UTC)


def _represented_date(value: object, *, field_name: str) -> date:
    if _is_missing(value):
        raise ValueError(f"{field_name} is required")
    if isinstance(value, datetime):
        if value.tzinfo is not None and value.utcoffset() is not None:
            return value.astimezone(_EASTERN).date()
        return value.date()
    if isinstance(value, date):
        return value
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"{field_name} is not a valid date")
    if parsed.tzinfo is not None:
        return parsed.tz_convert(_EASTERN).date()
    return parsed.date()


def _as_of_timestamp(value: date) -> datetime:
    return datetime.combine(value, _MARKET_CLOSE, tzinfo=_EASTERN).astimezone(UTC)


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be a finite numeric value")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite numeric value") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _validated_lineage(lineage: tuple[LineageRef, ...]) -> tuple[LineageRef, ...]:
    if not lineage:
        raise ValueError(
            "lineage is required: exact source artifact hash and record locator are required"
        )
    normalized: list[LineageRef] = []
    for ref in lineage:
        if not isinstance(ref, LineageRef):
            raise ValueError("lineage must contain LineageRef values")
        if not ref.source_id.strip():
            raise ValueError("lineage source_id must be non-empty")
        if _SHA256_RE.fullmatch(ref.content_hash) is None:
            raise ValueError("lineage content_hash must be an exact SHA-256 hex digest")
        if ref.record_locator is None or not ref.record_locator.strip():
            raise ValueError("exact original record locator is required for lineage")
        normalized.append(
            ref.model_copy(
                update={
                    "source_id": ref.source_id.strip(),
                    "content_hash": ref.content_hash.lower(),
                    "record_locator": ref.record_locator.strip(),
                }
            )
        )
    return tuple(
        sorted(
            normalized,
            key=lambda ref: (ref.content_hash, ref.source_id, ref.record_locator or ""),
        )
    )


def _lineage_ids(lineage: tuple[LineageRef, ...]) -> tuple[str, ...]:
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


def _stable_state_id(
    *,
    theme_id: str,
    as_of: datetime,
    available_at: datetime,
    taxonomy_version: str,
    lineage: tuple[LineageRef, ...],
) -> UUID:
    material = {
        "state_type": StateType.THEME.value,
        "theme_id": theme_id,
        "as_of": as_of.isoformat(),
        "available_at": available_at.isoformat(),
        "schema_version": 1,
        "producer": _PRODUCER,
        "model_version": _MODEL_VERSION,
        "feature_version": _FEATURE_VERSION,
        "config_version": f"{_CONFIG_VERSION}:{taxonomy_version}",
        "lineage": sorted(
            [
                {
                    "source_id": ref.source_id,
                    "content_hash": ref.content_hash,
                    "record_locator": ref.record_locator,
                }
                for ref in lineage
            ],
            key=lambda item: (
                item["source_id"],
                item["record_locator"],
                item["content_hash"],
            ),
        ),
    }
    return uuid5(NAMESPACE_URL, json.dumps(material, sort_keys=True, separators=(",", ":")))


def _normalize_memberships(
    memberships: pd.DataFrame,
    *,
    taxonomy_version: str,
) -> pd.DataFrame:
    required = {"ticker", "theme", "membership_score"}
    missing = sorted(required - set(memberships.columns))
    if missing:
        raise ValueError(f"memberships missing columns: {missing}")
    frame = memberships.copy()
    if "taxonomy_version" in frame.columns:
        frame = frame[frame["taxonomy_version"].astype(str) == taxonomy_version].copy()
    if frame.empty:
        frame["as_of"] = pd.Series(dtype=object)
        return frame
    date_column = "as_of" if "as_of" in frame.columns else "date"
    if date_column not in frame.columns:
        raise ValueError("memberships must contain as_of or date")
    frame["as_of"] = frame[date_column].map(
        lambda value: _represented_date(value, field_name=date_column)
    )
    frame["ticker"] = frame["ticker"].map(lambda value: str(value).strip().upper())
    frame["theme"] = frame["theme"].map(canonical_theme_id)
    frame["membership_score"] = frame["membership_score"].map(
        lambda value: _finite(value, field_name="membership_score")
    )
    return frame


def _normalize_features(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=["theme", "as_of"])
    frame = features.copy()
    theme_column = "theme" if "theme" in frame.columns else "primary_theme"
    date_column = "as_of" if "as_of" in frame.columns else "date"
    if theme_column not in frame.columns or date_column not in frame.columns:
        raise ValueError("features must contain theme/primary_theme and as_of/date")
    frame["theme"] = frame[theme_column].map(canonical_theme_id)
    frame["as_of"] = frame[date_column].map(
        lambda value: _represented_date(value, field_name=date_column)
    )
    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].map(lambda value: str(value).strip().upper())
    return frame


def _numeric_feature_metrics(frame: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for column in frame.columns:
        if column in _IDENTITY_COLUMNS:
            continue
        values: list[float] = []
        for value in frame[column].tolist():
            if _is_missing(value):
                continue
            if isinstance(value, (str, bytes, date, datetime, Mapping)):
                continue
            if not isinstance(value, (Real, np.number)):
                continue
            values.append(_finite(value, field_name=str(column)))
        if values:
            metrics[str(column)] = float(np.mean(values))
    return metrics


def _feature_metric(
    frame: pd.DataFrame,
    names: tuple[str, ...],
) -> float | None:
    for name in names:
        if name not in frame.columns:
            continue
        values = [
            _finite(value, field_name=name)
            for value in frame[name].tolist()
            if not _is_missing(value)
        ]
        if values:
            return float(np.mean(values))
    return None


def _generated_at(
    membership_group: pd.DataFrame,
    feature_group: pd.DataFrame,
    *,
    available_at: datetime,
) -> datetime:
    timestamps: list[datetime] = []
    for frame in (membership_group, feature_group):
        if "generated_at" not in frame.columns:
            continue
        for value in frame["generated_at"].tolist():
            if not _is_missing(value):
                timestamps.append(_timestamp(value, field_name="generated_at"))
    return max([available_at, *timestamps]) if timestamps else available_at


def _quality_issue(code: str, message: str) -> DataQualityIssue:
    return DataQualityIssue(
        code=code,
        severity=DataQualitySeverity.WARNING,
        component=_PRODUCER,
        message=message,
    )


def adapt_theme_states(
    memberships: pd.DataFrame,
    features: pd.DataFrame,
    *,
    available_at: datetime,
    valid_until: datetime,
    taxonomy_version: str,
    lineage: tuple[LineageRef, ...],
) -> tuple[ThemeState, ...]:
    """Adapt one taxonomy version into one immutable state per theme and date."""

    available_at_utc = _timestamp(available_at, field_name="available_at")
    valid_until_utc = _timestamp(valid_until, field_name="valid_until")
    if valid_until_utc <= available_at_utc:
        raise ValueError("valid_until must be exclusive and after available_at")
    taxonomy_version = str(taxonomy_version).strip()
    if not taxonomy_version:
        raise ValueError("taxonomy_version must be non-empty")
    validated_lineage = _validated_lineage(lineage)

    normalized_memberships = _normalize_memberships(
        memberships, taxonomy_version=taxonomy_version
    )
    normalized_features = _normalize_features(features)
    membership_keys = set(
        zip(normalized_memberships.get("theme", ()), normalized_memberships.get("as_of", ()))
    )
    feature_keys = set(
        zip(normalized_features.get("theme", ()), normalized_features.get("as_of", ()))
    )
    keys = sorted(membership_keys | feature_keys, key=lambda item: (item[0], item[1]))
    states: list[ThemeState] = []

    for theme_id, represented_date in keys:
        membership_group = normalized_memberships[
            (normalized_memberships["theme"] == theme_id)
            & (normalized_memberships["as_of"] == represented_date)
        ]
        feature_group = normalized_features[
            (normalized_features["theme"] == theme_id)
            & (normalized_features["as_of"] == represented_date)
        ]
        membership_scores: dict[str, float] = {}
        for ticker, group in membership_group.groupby("ticker", sort=True):
            scores = group["membership_score"].tolist()
            if len(set(scores)) != 1:
                raise ValueError(
                    f"conflicting membership scores for {represented_date} / {ticker} / {theme_id}"
                )
            membership_scores[str(ticker)] = scores[0]

        issues: list[DataQualityIssue] = []
        if membership_group.empty:
            issues.append(
                _quality_issue(
                    "THEME_MEMBERSHIPS_MISSING",
                    f"no membership rows for {theme_id} on {represented_date.isoformat()}",
                )
            )
        if feature_group.empty:
            issues.append(
                _quality_issue(
                    "THEME_FEATURES_MISSING",
                    f"no feature rows for {theme_id} on {represented_date.isoformat()}",
                )
            )

        as_of = _as_of_timestamp(represented_date)
        generated_at = _generated_at(
            membership_group,
            feature_group,
            available_at=available_at_utc,
        )
        metrics = _numeric_feature_metrics(feature_group)
        config_version = f"{_CONFIG_VERSION}:{taxonomy_version}"
        state = ThemeState(
            state_id=_stable_state_id(
                theme_id=theme_id,
                as_of=as_of,
                available_at=available_at_utc,
                taxonomy_version=taxonomy_version,
                lineage=validated_lineage,
            ),
            state_type=StateType.THEME,
            entity_id=theme_id,
            as_of=as_of,
            available_at=available_at_utc,
            generated_at=generated_at,
            valid_until=valid_until_utc,
            source_window_start=as_of,
            source_window_end=as_of,
            schema_version=1,
            producer=_PRODUCER,
            model_version=_MODEL_VERSION,
            feature_version=_FEATURE_VERSION,
            config_version=config_version,
            lineage_ids=_lineage_ids(validated_lineage),
            data_quality=DataQualitySummary(issues=tuple(issues)),
            theme_id=theme_id,
            theme_regime=ThemeRegime.UNKNOWN,
            relative_strength=_feature_metric(
                feature_group, ("relative_strength", "theme_strength")
            ),
            breadth=_feature_metric(feature_group, ("breadth", "theme_breadth")),
            momentum=_feature_metric(
                feature_group, ("momentum", "theme_momentum", "theme_heat_score")
            ),
            distribution_score=_feature_metric(
                feature_group, ("distribution_score", "theme_distribution")
            ),
            correlation_score=_feature_metric(
                feature_group, ("correlation_score", "theme_correlation")
            ),
            volatility_score=_feature_metric(
                feature_group, ("volatility_score", "theme_volatility")
            ),
            catalyst_pressure=_feature_metric(
                feature_group, ("catalyst_pressure", "theme_catalyst_pressure")
            ),
            dealer_fragility=_feature_metric(
                feature_group, ("dealer_fragility", "theme_dealer_fragility")
            ),
            leadership_score=_feature_metric(
                feature_group, ("leadership_score", "theme_strength")
            ),
            rotation_rank=_feature_metric(
                feature_group, ("rotation_rank", "primary_theme_rank")
            ),
            membership_scores=membership_scores,
            crowding=_feature_metric(feature_group, ("crowding", "theme_crowding")),
            persistence=_feature_metric(
                feature_group, ("persistence", "theme_persistence")
            ),
            metrics=metrics,
            transition_probabilities={},
        )
        states.append(state)
    return tuple(states)


def persist_theme_states(
    memberships: pd.DataFrame,
    features: pd.DataFrame,
    *,
    unit_of_work: "UnitOfWork",
    available_at: datetime,
    valid_until: datetime,
    taxonomy_version: str,
    lineage: tuple[LineageRef, ...],
) -> int:
    """Insert states through the caller-owned UOW; never commit or roll back."""

    states = adapt_theme_states(
        memberships,
        features,
        available_at=available_at,
        valid_until=valid_until,
        taxonomy_version=taxonomy_version,
        lineage=lineage,
    )
    if states:
        unit_of_work.states.insert_states_idempotently(states)
    return len(states)


publish_theme_states = persist_theme_states


__all__ = [
    "adapt_theme_states",
    "persist_theme_states",
    "publish_theme_states",
]
