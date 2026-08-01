"""Adapt captured dealer-positioning rows into causal nervous-system state."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timezone
from decimal import Decimal
import json
import math
from numbers import Number, Real
import re
from uuid import NAMESPACE_URL, UUID, uuid5

import pandas as pd

from core.nervous_system.contracts.enums import DataQualitySeverity, DealerRegime, StateType
from core.nervous_system.contracts.quality import (
    DataQualityIssue,
    DataQualitySummary,
    LineageRef,
)
from core.nervous_system.contracts.states import DealerState


UTC = timezone.utc
_PRODUCER = "strategies.dealer_positioning"
_MODEL_VERSION = "rules@1"
_FEATURE_VERSION = "dealer-positioning@1"
_CONFIG_VERSION = "dealer-positioning@1"
_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_SCOPE_WEIGHTS = {
    "daily_week": 0.45,
    "through_month": 0.35,
    "two_months": 0.20,
}
_DIRECTION_REGIMES = {
    "bullish": DealerRegime.UPSIDE_ACCELERATION,
    "neutral": DealerRegime.NEUTRAL_GAMMA,
    "bearish": DealerRegime.DOWNSIDE_ACCELERATION,
}

_KNOWN_NUMERIC_FIELDS = frozenset(
    """
    spot spot_price strike_window_pct row_count total_gex put_wall call_wall
    nearest_magnet magnet next_magnet_above next_magnet_below vega_wall
    vega_above vega_below next_vega_wall_above next_vega_wall_below gamma_flip
    ceiling floor air_gap_above_score air_gap_below_score
    magnet_threshold_abs_net_gex vega_threshold_total_vex
    avg_dollar_volume_20d market_cap total_option_oi total_option_volume
    pct_to_gamma_flip pct_to_call_wall pct_to_put_wall pct_to_magnet
    pct_to_vega_wall pct_to_vega_above pct_to_vega_below
    distance_callwall_putwall distance_gammaflip_callwall
    distance_gammaflip_magnet distance_magnet_putwall distance_magnet_callwall
    distance_floor_ceiling largest_positive_gex largest_negative_gex
    largest_call_gex largest_put_gex total_positive_gex total_negative_gex
    net_gex total_abs_gex weighted_average_strike weighted_average_call
    weighted_average_put median_gex_strike gex_entropy
    gex_concentration_index strike_skew strike_kurtosis positive_gex_above
    positive_gex_below negative_gex_above negative_gex_below call_gex_above
    put_gex_below ratio_above_vs_below vacuum_above vacuum_below
    nearest_level_distance vacuum_score compression_score pinning_score
    acceleration_score dealer_bias magnet_strength magnet_distance
    magnet_relative_strength nearest_competing_magnet gamma_density_5pct
    gamma_density_10pct gamma_density wall_dominance dealer_imbalance
    call_gex_total put_gex_total avg_strike_spacing gamma_flip_change_1d
    call_wall_change put_wall_change magnet_change vega_wall_change
    ceiling_change floor_change gamma_flip_velocity_3d magnet_velocity_3d
    callwall_velocity_3d dealer_imbalance_rank vacuum_score_rank
    distance_to_gamma_flip_percentile gamma_density_percentile
    gex_concentration_percentile distance_to_magnet_percentile
    wall_change_1d wall_change_1d_gap_days wall_change_3d
    wall_change_3d_gap_days gex_concentration_change gamma_flip_velocity
    distance_to_call_wall_atr distance_to_put_wall_atr level_stability_days
    oi_change_by_dte volume_to_prior_oi iv_skew_change
    near_level_option_volume_share prior_gap_days third_gap_days atr_14d
    scope_count scope_weight_sum dealer_swing_potential_score
    dealer_direction_bias dealer_change_intensity_score
    dealer_change_direction_bias max_scope_score max_change_scope_score
    avg_vacuum_component avg_sparse_gamma_component
    avg_pinning_room_component avg_imbalance_component
    avg_wall_room_component avg_magnet_room_component
    avg_wall_change_component avg_magnet_change_component
    avg_gamma_flip_change_component avg_velocity_change_component
    avg_floor_ceiling_change_component avg_vega_change_component
    avg_gamma_density avg_dealer_imbalance min_pct_to_magnet
    avg_gamma_flip_change_1d avg_call_wall_change avg_put_wall_change
    avg_magnet_change avg_vega_wall_change avg_ceiling_change avg_floor_change
    avg_gamma_flip_velocity_3d avg_magnet_velocity_3d
    avg_callwall_velocity_3d dealer_swing_rank dealer_change_intensity_rank
    dealer_change_bullish_rank dealer_change_bearish_rank stale_days
    option_data_age_days data_staleness_days print_count option_print_count
    trade_print_count expected_prints expected_print_count print_coverage
    print_coverage_ratio option_print_coverage
    """.split()
)
_KNOWN_NUMERIC_PATTERNS = (
    re.compile(r".+_(?:score|rank|percentile|component|bias)\Z"),
    re.compile(r".+_(?:change(?:_[13]d)?|velocity(?:_3d)?|gap_days)\Z"),
)

_NON_METRIC_FIELDS = frozenset(
    {
        "available_at",
        "captured_at",
        "chain_query_date_et",
        "cap_tier",
        "expirations",
        "fetched_at",
        "liquidity_pass_reason",
        "market_cap_bucket",
        "nearest_level_type",
        "dealer_direction",
        "dealer_change_direction",
        "dealer_regime",
        "regime",
        "prior_snapshot_date",
        "third_snapshot_date",
        "ranking_source_path",
        "ref_date_et",
        "scope",
        "scope_label",
        "snapshot_date",
        "source_id",
        "source_artifact_id",
        "source_hash",
        "scope_scores_json",
        "record_locator",
        "symbol",
        "ticker",
        "timestamp",
        "available_expirations",
        "above_call_wall",
        "above_gamma_flip",
        "above_magnet",
        "below_vega_wall",
        "has_prior_snapshot_1d",
        "has_prior_snapshot_3d",
        "inside_call_wall_zone",
        "prior_gap_stale",
        "third_gap_stale",
    }
)


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NaT or value is pd.NA:
        return True
    if not pd.api.types.is_scalar(value):
        return False
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    try:
        return bool(result)
    except (TypeError, ValueError):
        return False


def _timestamp(value: object, *, field_name: str) -> datetime:
    """Parse only timestamps carrying an explicit timezone."""

    if _is_missing(value):
        raise ValueError(f"{field_name} must be an explicit timezone-aware timestamp")
    if hasattr(value, "to_pydatetime") and not isinstance(value, datetime):
        try:
            value = value.to_pydatetime()
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} must be an explicit timezone-aware timestamp"
            ) from exc
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} is not an ISO-8601 timestamp") from exc
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be an explicit timezone-aware timestamp")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _finite(value: object, *, field_name: str) -> float:
    if (
        isinstance(value, (bool, bytes, date, datetime))
        or not pd.api.types.is_scalar(value)
    ):
        raise ValueError(f"{field_name} must be a finite numeric value")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a finite numeric value") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _reject_nonfinite(value: object, *, path: str) -> None:
    if _is_missing(value):
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_nonfinite(nested, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for index, nested in enumerate(value):
            _reject_nonfinite(nested, path=f"{path}[{index}]")
        return
    if isinstance(value, (bool, str, bytes, date, datetime)):
        return
    if isinstance(value, (Real, Decimal, Number)):
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{path} must be finite") from exc
        if not finite:
            raise ValueError(f"{path} must be finite")


def _mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings")
    _reject_nonfinite(value, path=field_name)
    _validate_known_numeric_evidence(value, path=field_name)
    return value


def _is_known_numeric_field(name: str) -> bool:
    return name in _KNOWN_NUMERIC_FIELDS or any(
        pattern.fullmatch(name) is not None for pattern in _KNOWN_NUMERIC_PATTERNS
    )


def _validate_known_numeric_evidence(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for name, nested in value.items():
            nested_path = f"{path}.{name}"
            if isinstance(name, str) and _is_known_numeric_field(name):
                if not _is_missing(nested):
                    _finite(nested, field_name=nested_path)
                continue
            _validate_known_numeric_evidence(nested, path=nested_path)
        return
    if pd.api.types.is_list_like(value) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _validate_known_numeric_evidence(nested, path=f"{path}[{index}]")


def _validated_lineage(lineage: object) -> tuple[LineageRef, ...]:
    if not isinstance(lineage, (tuple, list)) or not lineage:
        raise ValueError("lineage must contain at least one LineageRef")
    normalized: list[LineageRef] = []
    for item in lineage:
        if not isinstance(item, LineageRef):
            raise ValueError("lineage must contain LineageRef values")
        source_id = item.source_id.strip()
        content_hash = item.content_hash.strip().lower()
        record_locator = item.record_locator.strip() if item.record_locator is not None else ""
        if not source_id:
            raise ValueError("lineage source_id must be non-empty")
        if _SHA256_RE.fullmatch(content_hash) is None:
            raise ValueError("lineage content_hash must be an exact SHA-256 hex digest")
        if not record_locator:
            raise ValueError("lineage record_locator must be non-empty")
        normalized.append(
            item.model_copy(
                update={
                    "source_id": source_id,
                    "content_hash": content_hash,
                    "record_locator": record_locator,
                }
            )
        )
    return tuple(normalized)


def _lineage_ids(lineage: tuple[LineageRef, ...]) -> tuple[str, ...]:
    return tuple(
        json.dumps(
            {
                "content_hash": item.content_hash,
                "record_locator": item.record_locator,
                "source_id": item.source_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for item in lineage
    )


def _alias_value(
    row: Mapping[str, object], names: tuple[str, ...], *, field_name: str
) -> float | None:
    values: list[float] = []
    for name in names:
        if name not in row or _is_missing(row[name]):
            continue
        values.append(_finite(row[name], field_name=field_name))
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"conflicting {field_name} values")
    return values[0]


def _required_alias_value(
    row: Mapping[str, object], names: tuple[str, ...], *, field_name: str
) -> float:
    value = _alias_value(row, names, field_name=field_name)
    if value is None:
        raise ValueError(f"snapshot must contain finite {field_name}")
    return value


def _ticker(row: Mapping[str, object]) -> str:
    values: list[str] = []
    for name in ("symbol", "ticker"):
        if name not in row or _is_missing(row[name]):
            continue
        if not isinstance(row[name], str):
            raise ValueError(f"{name} must be a non-empty ticker string")
        value = row[name].strip().upper()
        if not value:
            raise ValueError(f"{name} must be a non-empty ticker string")
        values.append(value)
    if not values:
        raise ValueError("snapshot must contain symbol")
    if any(value != values[0] for value in values[1:]):
        raise ValueError("conflicting symbol and ticker values")
    return values[0]


def _selected_scope(
    snapshot: Mapping[str, object], dynamics: Mapping[str, object] | None
) -> str:
    raw_scope = snapshot.get("scope")
    if not isinstance(raw_scope, str) or raw_scope not in _SCOPE_WEIGHTS:
        names = ", ".join(_SCOPE_WEIGHTS)
        raise ValueError(f"snapshot scope must be exactly one of: {names}")
    if dynamics is not None:
        dynamics_scope = dynamics.get("scope")
        if not isinstance(dynamics_scope, str) or dynamics_scope not in _SCOPE_WEIGHTS:
            names = ", ".join(_SCOPE_WEIGHTS)
            raise ValueError(f"dynamics scope must be exactly one of: {names}")
        if dynamics_scope != raw_scope:
            raise ValueError("dynamics scope conflicts with snapshot scope")
    return raw_scope


def _snapshot_date(row: Mapping[str, object], *, field_name: str) -> date:
    value = row.get("snapshot_date")
    if _is_missing(value):
        raise ValueError(f"{field_name} must contain snapshot_date")
    if hasattr(value, "to_pydatetime") and not isinstance(value, datetime):
        try:
            value = value.to_pydatetime()
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} snapshot_date must be an ISO date") from exc
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError as exc:
            raise ValueError(f"{field_name} snapshot_date must be an ISO date") from exc
    raise ValueError(f"{field_name} must contain snapshot_date")


def _dealer_regime(snapshot: Mapping[str, object]) -> DealerRegime:
    evidence: list[DealerRegime] = []
    if "dealer_direction" in snapshot and not _is_missing(
        snapshot["dealer_direction"]
    ):
        direction = snapshot["dealer_direction"]
        if not isinstance(direction, str) or direction not in _DIRECTION_REGIMES:
            names = ", ".join(_DIRECTION_REGIMES)
            raise ValueError(f"dealer_direction must be exactly one of: {names}")
        evidence.append(_DIRECTION_REGIMES[direction])

    for field_name in ("dealer_regime", "regime"):
        if field_name not in snapshot or _is_missing(snapshot[field_name]):
            continue
        value = snapshot[field_name]
        try:
            regime = DealerRegime(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a canonical DealerRegime value") from exc
        if regime is DealerRegime.UNKNOWN:
            raise ValueError(f"{field_name} cannot be UNKNOWN")
        evidence.append(regime)

    if not evidence:
        return DealerRegime.UNKNOWN
    if any(regime is not evidence[0] for regime in evidence[1:]):
        raise ValueError("conflicting dealer direction/regime values")
    return evidence[0]


def _check_capture_metadata(
    row: Mapping[str, object],
    *,
    captured_at: datetime,
    field_name: str,
    required: bool = False,
) -> None:
    if "captured_at" not in row or _is_missing(row["captured_at"]):
        if required:
            raise ValueError(f"{field_name} must contain captured_at")
        return
    row_captured_at = _timestamp(
        row["captured_at"], field_name=f"{field_name} captured_at"
    )
    if row_captured_at != captured_at:
        raise ValueError(f"{field_name} captured_at conflicts with explicit captured_at")


def _numeric_metrics(row: Mapping[str, object]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, value in row.items():
        if name in _NON_METRIC_FIELDS or _is_missing(value):
            continue
        if _is_known_numeric_field(name):
            metrics[name] = _finite(value, field_name=name)
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (str, bytes, date, datetime, Mapping, list, tuple, set, frozenset)):
            continue
        if isinstance(value, (Real, Decimal, Number)):
            metrics[name] = _finite(value, field_name=name)
    return metrics


def _merge_metrics(
    snapshot: Mapping[str, object], dynamics: Mapping[str, object] | None
) -> dict[str, float]:
    merged: dict[str, float] = {}
    for row in (snapshot, dynamics or {}):
        for name, value in _numeric_metrics(row).items():
            if name in merged and merged[name] != value:
                raise ValueError(f"conflicting {name} values across snapshot and dynamics")
            merged[name] = value
    return merged


def _source_number(
    rows: tuple[Mapping[str, object], ...], names: tuple[str, ...], *, field_name: str
) -> float | None:
    values: list[float] = []
    for row in rows:
        for name in names:
            if name in row and not _is_missing(row[name]):
                values.append(_finite(row[name], field_name=field_name))
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"conflicting {field_name} values")
    return values[0]


def _source_bool(
    rows: tuple[Mapping[str, object], ...], names: tuple[str, ...], *, field_name: str
) -> bool | None:
    values: list[bool] = []
    for row in rows:
        for name in names:
            if name not in row or _is_missing(row[name]):
                continue
            if not isinstance(row[name], bool):
                raise ValueError(f"{field_name} must be an explicit boolean")
            values.append(row[name])
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"conflicting {field_name} values")
    return values[0]


def _quality_issues(
    snapshot: Mapping[str, object], dynamics: Mapping[str, object] | None
) -> tuple[DataQualityIssue, ...]:
    rows = (snapshot, dynamics) if dynamics is not None else (snapshot,)
    issues: list[DataQualityIssue] = []

    def add_once(code: str, message: str) -> None:
        if not any(issue.code == code for issue in issues):
            issues.append(
                DataQualityIssue(
                    code=code,
                    severity=DataQualitySeverity.WARNING,
                    component=_PRODUCER,
                    message=message,
                )
            )

    stale_days = _source_number(
        rows,
        ("stale_days", "option_data_age_days", "data_staleness_days"),
        field_name="stale_days",
    )
    if stale_days is not None:
        if stale_days < 0:
            raise ValueError("stale_days must be non-negative")
        if stale_days > 0:
            add_once("STALE_OPTION_DATA", "option evidence is older than the capture")

    stale_flag = _source_bool(
        rows,
        ("stale", "is_stale", "option_data_stale", "data_stale", "prior_gap_stale", "third_gap_stale"),
        field_name="stale flag",
    )
    if stale_flag:
        add_once("STALE_OPTION_DATA", "option evidence is marked stale")

    print_sparse = _source_bool(
        rows,
        ("print_sparse", "option_data_print_sparse"),
        field_name="print_sparse",
    )
    if print_sparse:
        add_once("PRINT_SPARSE_OPTION_DATA", "option evidence is marked print-sparse")

    print_count = _source_number(
        rows,
        ("print_count", "option_print_count", "trade_print_count"),
        field_name="print_count",
    )
    expected_prints = _source_number(
        rows,
        ("expected_prints", "expected_print_count"),
        field_name="expected_prints",
    )
    if print_count is not None and print_count < 0:
        raise ValueError("print_count must be non-negative")
    if expected_prints is not None and expected_prints < 0:
        raise ValueError("expected_prints must be non-negative")
    if (
        print_count is not None
        and expected_prints is not None
        and expected_prints > 0
        and print_count < expected_prints
    ):
        add_once("PRINT_SPARSE_OPTION_DATA", "observed option prints are below expectation")

    coverage = _source_number(
        rows,
        ("print_coverage", "print_coverage_ratio", "option_print_coverage"),
        field_name="print_coverage",
    )
    if coverage is not None:
        if not 0.0 <= coverage <= 1.0:
            raise ValueError("print_coverage must be between 0 and 1")
        if coverage < 1.0:
            add_once("PRINT_SPARSE_OPTION_DATA", "option print coverage is incomplete")

    return tuple(issues)


def _scope_quality_issue(scope: str) -> DataQualityIssue:
    return DataQualityIssue(
        code="DEALER_SCOPE_SELECTED",
        severity=DataQualitySeverity.INFO,
        component=_PRODUCER,
        message=f"selected canonical producer scope: {scope}",
    )


def _uncalibrated_direction_quality_issue() -> DataQualityIssue:
    return DataQualityIssue(
        code="DEALER_DIRECTION_UNCALIBRATED",
        severity=DataQualitySeverity.WARNING,
        component=_PRODUCER,
        message=(
            "no calibrated dealer direction or regime evidence; "
            "dealer regime remains UNKNOWN"
        ),
    )


def _stable_state_id(
    *,
    ticker: str,
    as_of: datetime,
    available_at: datetime,
    valid_until: datetime,
    metrics: Mapping[str, float],
    fields: Mapping[str, float | str | None],
    quality: tuple[DataQualityIssue, ...],
    lineage_ids: tuple[str, ...],
) -> UUID:
    material = {
        "state_type": StateType.DEALER.value,
        "ticker": ticker,
        "as_of": as_of.isoformat(),
        "available_at": available_at.isoformat(),
        "valid_until": valid_until.isoformat(),
        "schema_version": _SCHEMA_VERSION,
        "producer": _PRODUCER,
        "model_version": _MODEL_VERSION,
        "feature_version": _FEATURE_VERSION,
        "config_version": _CONFIG_VERSION,
        "lineage_ids": lineage_ids,
        "fields": dict(fields),
        "metrics": dict(metrics),
        "quality": [
            {
                "code": issue.code,
                "severity": issue.severity.value,
                "component": issue.component,
                "message": issue.message,
                "fallback_used": issue.fallback_used,
            }
            for issue in quality
        ],
    }
    return uuid5(
        NAMESPACE_URL,
        json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False),
    )


def adapt_dealer_state(
    snapshot: Mapping[str, object],
    dynamics: Mapping[str, object] | None,
    *,
    captured_at: datetime,
    valid_until: datetime,
    lineage: tuple[LineageRef, ...],
) -> DealerState:
    """Adapt one captured dealer summary and optional level-dynamics row.

    The caller must first select exactly one row for the ticker from the
    canonical producer scopes ``daily_week``, ``through_month``, or
    ``two_months``. This adapter never chooses among same-ticker scope rows;
    any cross-scope aggregation must happen upstream and still declare the
    canonical scope supplying the singleton level fields.

    Raw rows must carry producer ``snapshot_date`` and ``captured_at``
    evidence, and row ``captured_at`` must exactly match the explicit
    parameter. ``snapshot_date`` may precede capture for a backfill but may
    never describe a later calendar date. Raw availability is captured_at.
    Dynamics must carry their own explicit, timezone-aware ``available_at``;
    ``GammaLevels.timestamp`` is deliberately ignored as capture evidence.
    """

    snapshot_row = _mapping(snapshot, field_name="snapshot")
    dynamics_row = (
        _mapping(dynamics, field_name="dynamics") if dynamics is not None else None
    )
    captured_at_utc = _timestamp(captured_at, field_name="captured_at")
    valid_until_utc = _timestamp(valid_until, field_name="valid_until")
    if valid_until_utc <= captured_at_utc:
        raise ValueError("valid_until must be exclusive and after captured_at")
    _check_capture_metadata(
        snapshot_row,
        captured_at=captured_at_utc,
        field_name="snapshot",
        required=True,
    )
    snapshot_day = _snapshot_date(snapshot_row, field_name="snapshot")
    if snapshot_day > captured_at_utc.date():
        raise ValueError("snapshot snapshot_date cannot be after captured_at")

    available_at = captured_at_utc
    if dynamics_row is not None:
        if "available_at" not in dynamics_row or _is_missing(
            dynamics_row["available_at"]
        ):
            raise ValueError("dynamics must contain explicit available_at")
        available_at = _timestamp(dynamics_row["available_at"], field_name="available_at")
        if available_at < captured_at_utc:
            raise ValueError("dynamics available_at cannot precede captured_at")
        _check_capture_metadata(dynamics_row, captured_at=captured_at_utc, field_name="dynamics")
    if valid_until_utc <= available_at:
        raise ValueError("valid_until must be exclusive and after available_at")

    validated_lineage = _validated_lineage(lineage)
    lineage_ids = _lineage_ids(validated_lineage)
    ticker = _ticker(snapshot_row)
    scope = _selected_scope(snapshot_row, dynamics_row)
    dealer_regime = _dealer_regime(snapshot_row)
    spot = _required_alias_value(snapshot_row, ("spot", "spot_price"), field_name="spot")
    if spot <= 0:
        raise ValueError("spot must be greater than zero")

    state_fields: dict[str, float | None] = {
        "total_gex": _alias_value(snapshot_row, ("total_gex",), field_name="total_gex"),
        "call_wall": _alias_value(snapshot_row, ("call_wall",), field_name="call_wall"),
        "put_wall": _alias_value(snapshot_row, ("put_wall",), field_name="put_wall"),
        "nearest_magnet": _alias_value(
            snapshot_row, ("nearest_magnet", "magnet"), field_name="nearest_magnet"
        ),
        "gamma_flip": _alias_value(snapshot_row, ("gamma_flip",), field_name="gamma_flip"),
        "air_gap_above_score": _alias_value(
            snapshot_row, ("air_gap_above_score",), field_name="air_gap_above_score"
        ),
        "air_gap_below_score": _alias_value(
            snapshot_row, ("air_gap_below_score",), field_name="air_gap_below_score"
        ),
        "pinning_score": _alias_value(
            snapshot_row, ("pinning_score",), field_name="pinning_score"
        ),
        "acceleration_score": _alias_value(
            snapshot_row, ("acceleration_score",), field_name="acceleration_score"
        ),
    }
    metrics = _merge_metrics(snapshot_row, dynamics_row)
    metrics[f"selected_scope_{scope}"] = 1.0
    metrics["selected_scope_weight"] = _SCOPE_WEIGHTS[scope]
    quality_issues = (
        *_quality_issues(snapshot_row, dynamics_row),
        *(
            (_uncalibrated_direction_quality_issue(),)
            if dealer_regime is DealerRegime.UNKNOWN
            else ()
        ),
        _scope_quality_issue(scope),
    )
    quality = DataQualitySummary(issues=quality_issues)
    state_id = _stable_state_id(
        ticker=ticker,
        as_of=captured_at_utc,
        available_at=available_at,
        valid_until=valid_until_utc,
        metrics=metrics,
        fields={"dealer_regime": dealer_regime.value, "spot": spot, **state_fields},
        quality=quality_issues,
        lineage_ids=lineage_ids,
    )

    return DealerState(
        state_id=state_id,
        state_type=StateType.DEALER,
        entity_id=ticker,
        as_of=captured_at_utc,
        available_at=available_at,
        generated_at=available_at,
        valid_until=valid_until_utc,
        source_window_start=captured_at_utc,
        source_window_end=captured_at_utc,
        schema_version=_SCHEMA_VERSION,
        producer=_PRODUCER,
        model_version=_MODEL_VERSION,
        feature_version=_FEATURE_VERSION,
        config_version=_CONFIG_VERSION,
        lineage_ids=lineage_ids,
        data_quality=quality,
        ticker=ticker,
        dealer_regime=dealer_regime,
        spot=spot,
        total_gex=state_fields["total_gex"],
        call_wall=state_fields["call_wall"],
        put_wall=state_fields["put_wall"],
        nearest_magnet=state_fields["nearest_magnet"],
        gamma_flip=state_fields["gamma_flip"],
        air_gap_above_score=state_fields["air_gap_above_score"],
        air_gap_below_score=state_fields["air_gap_below_score"],
        pinning_score=state_fields["pinning_score"],
        acceleration_score=state_fields["acceleration_score"],
        metrics=metrics,
    )


def adapt_dealer_state_with_ranking(
    snapshot: Mapping[str, object],
    dynamics: Mapping[str, object] | None,
    ranking: Mapping[str, object],
    *,
    captured_at: datetime,
    ranking_available_at: datetime,
    valid_until: datetime,
    lineage: tuple[LineageRef, ...],
    ranking_lineage: tuple[LineageRef, ...],
) -> DealerState:
    """Join one selected raw scope row with separately available ranking evidence.

    ``ranking`` must be the ticker-aggregate output of ``build_rankings`` and
    therefore must not declare a scope. The selected canonical scope and all
    singleton dealer levels continue to come only from ``snapshot``.
    """

    snapshot_row = _mapping(snapshot, field_name="snapshot")
    dynamics_row = (
        _mapping(dynamics, field_name="dynamics") if dynamics is not None else None
    )
    ranking_row = _mapping(ranking, field_name="ranking")
    if "scope" in ranking_row:
        raise ValueError("aggregate ranking must not declare or fabricate scope")

    captured_at_utc = _timestamp(captured_at, field_name="captured_at")
    ranking_available_at_utc = _timestamp(
        ranking_available_at, field_name="ranking_available_at"
    )
    if ranking_available_at_utc < captured_at_utc:
        raise ValueError("ranking_available_at cannot precede captured_at")
    if "available_at" in ranking_row and not _is_missing(ranking_row["available_at"]):
        row_available_at = _timestamp(
            ranking_row["available_at"], field_name="ranking available_at"
        )
        if row_available_at != ranking_available_at_utc:
            raise ValueError("ranking available_at conflicts with ranking_available_at")

    if "captured_at" not in ranking_row or _is_missing(ranking_row["captured_at"]):
        raise ValueError("ranking must contain captured_at")
    ranking_captured_at = _timestamp(
        ranking_row["captured_at"], field_name="ranking captured_at"
    )
    if ranking_captured_at != captured_at_utc:
        raise ValueError("ranking captured_at conflicts with captured_at")

    snapshot_ticker = _ticker(snapshot_row)
    ranking_ticker = _ticker(ranking_row)
    if ranking_ticker != snapshot_ticker:
        raise ValueError("ranking ticker conflicts with snapshot ticker")
    if _snapshot_date(ranking_row, field_name="ranking") != _snapshot_date(
        snapshot_row, field_name="snapshot"
    ):
        raise ValueError("ranking snapshot_date conflicts with snapshot snapshot_date")

    if "scope_count" not in ranking_row or "scope_weight_sum" not in ranking_row:
        raise ValueError("ranking must contain aggregate scope_count and scope_weight_sum")
    scope_count = _finite(ranking_row["scope_count"], field_name="scope_count")
    scope_weight_sum = _finite(
        ranking_row["scope_weight_sum"], field_name="scope_weight_sum"
    )
    if scope_count < 1 or not scope_count.is_integer() or scope_weight_sum <= 0:
        raise ValueError("ranking aggregate scope evidence is invalid")

    raw_regime = _dealer_regime(snapshot_row)
    ranking_regime = _dealer_regime(ranking_row)
    if ranking_regime is DealerRegime.UNKNOWN:
        raise ValueError("ranking must contain explicit dealer direction or regime evidence")
    if raw_regime is not DealerRegime.UNKNOWN and raw_regime is not ranking_regime:
        raise ValueError("ranking dealer regime conflicts with snapshot dealer regime")

    selected_scope = _selected_scope(snapshot_row, None)
    dynamics_available_at: datetime | None = None
    if dynamics_row is not None:
        try:
            dynamics_ticker = _ticker(dynamics_row)
        except ValueError as exc:
            raise ValueError(f"dynamics ticker is invalid: {exc}") from exc
        if dynamics_ticker != snapshot_ticker:
            raise ValueError("dynamics ticker conflicts with snapshot ticker")
        if _snapshot_date(dynamics_row, field_name="dynamics") != _snapshot_date(
            snapshot_row, field_name="snapshot"
        ):
            raise ValueError(
                "dynamics snapshot_date conflicts with snapshot snapshot_date"
            )
        if "captured_at" not in dynamics_row or _is_missing(
            dynamics_row["captured_at"]
        ):
            raise ValueError("dynamics must contain captured_at")
        dynamics_captured_at = _timestamp(
            dynamics_row["captured_at"], field_name="dynamics captured_at"
        )
        if dynamics_captured_at != captured_at_utc:
            raise ValueError("dynamics captured_at conflicts with captured_at")
        if "scope" in dynamics_row and not _is_missing(dynamics_row["scope"]):
            dynamics_scope = dynamics_row["scope"]
            if dynamics_scope != selected_scope:
                raise ValueError("dynamics scope conflicts with snapshot scope")
        if "available_at" not in dynamics_row or _is_missing(
            dynamics_row["available_at"]
        ):
            raise ValueError("dynamics must contain explicit available_at")
        dynamics_available_at = _timestamp(
            dynamics_row["available_at"], field_name="dynamics available_at"
        )
        if dynamics_available_at < captured_at_utc:
            raise ValueError("dynamics available_at cannot precede captured_at")

    normalized_snapshot = dict(snapshot_row)
    for field_name in ("dealer_direction", "dealer_regime", "regime"):
        if field_name in ranking_row and not _is_missing(ranking_row[field_name]):
            normalized_snapshot[field_name] = ranking_row[field_name]

    normalized_dynamics = dict(dynamics_row or {})
    if dynamics_row is None:
        normalized_dynamics["scope"] = selected_scope
        normalized_dynamics["captured_at"] = captured_at_utc
        evidence_available_at = captured_at_utc
    else:
        normalized_dynamics["scope"] = selected_scope
        evidence_available_at = dynamics_available_at
    normalized_dynamics["available_at"] = max(
        evidence_available_at, ranking_available_at_utc
    )
    for name, value in _numeric_metrics(ranking_row).items():
        if name in normalized_dynamics and normalized_dynamics[name] != value:
            raise ValueError(f"conflicting {name} values across dynamics and ranking")
        normalized_dynamics[name] = value

    validated_lineage = _validated_lineage(lineage)
    validated_ranking_lineage = _validated_lineage(ranking_lineage)
    lineage_ids = set(_lineage_ids(validated_lineage))
    if lineage_ids.intersection(_lineage_ids(validated_ranking_lineage)):
        raise ValueError("ranking_lineage must identify separate ranking evidence")

    return adapt_dealer_state(
        normalized_snapshot,
        normalized_dynamics,
        captured_at=captured_at_utc,
        valid_until=valid_until,
        lineage=(*validated_lineage, *validated_ranking_lineage),
    )


__all__ = ["adapt_dealer_state", "adapt_dealer_state_with_ranking"]
