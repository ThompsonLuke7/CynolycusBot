"""Adapt one causal Meta Ranker matrix row into a :class:`TickerState`."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timezone
from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import json
import math
from numbers import Real
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import pandas as pd

from core.nervous_system.contracts.enums import (
    DataQualitySeverity,
    DecisionKind,
    Direction,
    InstrumentFamily,
    StateType,
    TickerSetup,
)
from core.nervous_system.contracts.intent import TradeIntent
from core.nervous_system.contracts.quality import (
    DataQualityIssue,
    DataQualitySummary,
    LineageRef,
)
from core.nervous_system.contracts.states import TickerState


UTC = timezone.utc
_PRODUCER = "signals.meta_context.meta_ranker"
_MODEL_VERSION = "meta-ranker-adapter@1"
_FEATURE_VERSION = "meta-ranker-matrix@1"
_CONFIG_VERSION = "meta-ranker-ticker@1"
_SHA256_LENGTH = 64

_DEFAULT_INSTRUMENT_PREFERENCES = tuple(InstrumentFamily)


@dataclass(frozen=True)
class MetaIntentConfig:
    """Pure, versioned metadata for Meta Ranker entry intents."""

    strategy_id: str = "meta_ranker"
    quality_floor: float = 0.4
    held_tickers: frozenset[str] = field(default_factory=frozenset)
    requested_notional: Decimal | float | int = Decimal("5000")
    model_version: str = "meta-ranker-model@1"
    feature_version: str = _FEATURE_VERSION
    config_version: str = "meta-ranker-intent@1"
    instrument_preferences: tuple[InstrumentFamily, ...] = _DEFAULT_INSTRUMENT_PREFERENCES
    expected_holding_period: str = "53x4h"
    entry_window: str = "current-or-next-open"
    reason_codes: tuple[str, ...] = ("META_TOP_K", "META_LONG_ENTRY")

    def __post_init__(self) -> None:
        for field_name in ("strategy_id", "model_version", "feature_version", "config_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"{field_name} must be non-empty text")
        if self.config_version.strip().upper() == "UNKNOWN":
            raise ValueError("config_version must not be UNKNOWN")
        if isinstance(self.quality_floor, (bool, str, bytes)) or not isinstance(self.quality_floor, (Real, Decimal)):
            raise TypeError("quality_floor must be a finite numeric value")
        quality_floor = float(self.quality_floor)
        if not math.isfinite(quality_floor) or not 0.0 <= quality_floor <= 1.0:
            raise ValueError("quality_floor must be finite and within [0, 1]")
        if isinstance(self.requested_notional, (bool, str, bytes)) or not isinstance(
            self.requested_notional, (Real, Decimal)
        ):
            raise TypeError("requested_notional must be a finite numeric value")
        try:
            notional = Decimal(str(self.requested_notional))
        except (ArithmeticError, ValueError) as exc:
            raise ValueError("requested_notional must be a finite non-negative amount") from exc
        if notional.is_nan() or not notional.is_finite() or notional < 0:
            raise ValueError("requested_notional must be a finite non-negative amount")
        if not self.instrument_preferences:
            raise ValueError("instrument_preferences must be non-empty")
        preferences = tuple(self.instrument_preferences)
        if any(not isinstance(item, InstrumentFamily) for item in preferences):
            raise TypeError("instrument_preferences must contain InstrumentFamily values")
        if len(set(preferences)) != len(preferences):
            raise ValueError("instrument_preferences must contain unique values")
        object.__setattr__(self, "quality_floor", quality_floor)
        for field_name in ("model_version", "feature_version", "config_version"):
            object.__setattr__(self, field_name, getattr(self, field_name).strip())
        object.__setattr__(
            self,
            "held_tickers",
            _canonical_ticker_set(self.held_tickers, field_name="held_tickers"),
        )
        object.__setattr__(self, "requested_notional", notional)


def _canonical_ticker(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a ticker string")
    ticker = value.strip().upper()
    if not ticker:
        raise ValueError(f"{field_name} must be non-empty")
    return ticker


def _canonical_ticker_set(value: object, *, field_name: str) -> frozenset[str]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of ticker strings")
    try:
        return frozenset(_canonical_ticker(item, field_name=field_name) for item in value)  # type: ignore[union-attr]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of ticker strings") from exc

_NON_METRIC_FIELDS = frozenset(
    {
        "ticker",
        "timestamp",
        "selected_bar",
        "date",
        "theme",
        "primary_theme",
        "ticker_setup",
        "trend_state",
        "relative_strength_state",
        "support_state",
        "volume_state",
        "reversal_state",
        "breakdown_state",
        "source_id",
        "source_hash",
        "source_artifact_hash",
        "artifact_hash",
        "content_hash",
        "record_locator",
        "source_row_locator",
        "available_at",
        "generated_at",
        "valid_until",
        "source_window_start",
        "source_window_end",
        "lineage",
        # Forward outcomes and training labels are not current ticker state.
        "meta_label",
        "meta_good",
        "meta_upside",
        "trade_quality",
    }
)
_HINDSIGHT_PREFIXES = (
    "fwd_",
    "forward_",
    "target",
    "post_",
    "realized_",
    "realised_",
)


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NaT or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    try:
        return bool(result)
    except (TypeError, ValueError):
        return False


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


def _timestamp_series(values: pd.Series, *, field_name: str) -> pd.Series:
    normalized: list[datetime] = []
    for value in values.tolist():
        normalized.append(_timestamp(value, field_name=field_name))
    return pd.Series(normalized, index=values.index, name=values.name)


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, (bool,)):
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
        raise ValueError("lineage is required: source hash and record locator are mandatory")
    validated: list[LineageRef] = []
    for ref in lineage:
        if not isinstance(ref, LineageRef):
            raise ValueError("lineage must contain LineageRef values")
        source_id = ref.source_id.strip()
        content_hash = ref.content_hash.strip().lower()
        locator = (ref.record_locator or "").strip()
        if not source_id:
            raise ValueError("lineage source_id must be non-empty")
        if len(content_hash) != _SHA256_LENGTH:
            raise ValueError("lineage content_hash must be an exact SHA-256 hex digest")
        try:
            int(content_hash, 16)
        except ValueError as exc:
            raise ValueError("lineage content_hash must be an exact SHA-256 hex digest") from exc
        if not locator:
            raise ValueError("lineage record_locator must be non-empty")
        validated.append(
            ref.model_copy(
                update={
                    "source_id": source_id,
                    "content_hash": content_hash,
                    "record_locator": locator,
                }
            )
        )
    return tuple(validated)


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
    ticker: str,
    selected_bar: datetime,
    available_at: datetime,
    lineage: tuple[LineageRef, ...],
    content_revision: str,
) -> UUID:
    material = {
        "state_type": StateType.TICKER.value,
        "ticker": ticker,
        "selected_bar": selected_bar.isoformat(),
        "available_at": available_at.isoformat(),
        "schema_version": 1,
        "producer": _PRODUCER,
        "model_version": _MODEL_VERSION,
        "feature_version": _FEATURE_VERSION,
        "config_version": _CONFIG_VERSION,
        "content_revision": content_revision,
        "lineage": [
            {
                "source_id": ref.source_id,
                "content_hash": ref.content_hash,
                "record_locator": ref.record_locator,
            }
            for ref in lineage
        ],
    }
    return uuid5(NAMESPACE_URL, json.dumps(material, sort_keys=True, separators=(",", ":")))


def _content_revision_digest(
    *,
    valid_until: datetime,
    reference_price: float,
    ticker_setup: TickerSetup,
    trend_state: str,
    relative_strength_state: str,
    support_state: str,
    volume_state: str,
    reversal_state: str,
    breakdown_state: str,
    theme_alignment: float | None,
    market_alignment: float | None,
    dealer_alignment: float | None,
    metrics: Mapping[str, float],
    data_quality: DataQualitySummary,
) -> str:
    projected_content = {
        "valid_until": valid_until.isoformat(),
        "reference_price": reference_price,
        "ticker_setup": ticker_setup.value,
        "trend_state": trend_state,
        "relative_strength_state": relative_strength_state,
        "support_state": support_state,
        "volume_state": volume_state,
        "reversal_state": reversal_state,
        "breakdown_state": breakdown_state,
        "theme_alignment": theme_alignment,
        "market_alignment": market_alignment,
        "dealer_alignment": dealer_alignment,
        "metrics": dict(metrics),
        "data_quality": data_quality.model_dump(mode="json"),
        "transition_probabilities": {},
    }
    canonical = json.dumps(
        projected_content,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _text(row: Mapping[str, object], name: str, default: str = "UNKNOWN") -> str:
    value = row.get(name)
    if _is_missing(value):
        return default
    text = str(value).strip()
    return text or default


def _ticker_setup(row: Mapping[str, object]) -> TickerSetup:
    value = row.get("ticker_setup")
    if _is_missing(value):
        return TickerSetup.UNKNOWN
    try:
        return TickerSetup(str(value).strip().upper())
    except ValueError:
        return TickerSetup.UNKNOWN


def _optional_finite(row: Mapping[str, object], name: str) -> float | None:
    value = row.get(name)
    if _is_missing(value):
        return None
    return _finite(value, field_name=name)


def _metric_values(row: Mapping[str, object]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, value in row.items():
        name = str(name)
        normalized = name.lower()
        if name in _NON_METRIC_FIELDS or normalized.startswith(_HINDSIGHT_PREFIXES):
            continue
        if _is_missing(value) or isinstance(value, (str, bytes, date, datetime, Mapping)):
            continue
        if not isinstance(value, (Real,)):
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
        metrics[name] = _finite(value, field_name=name)
    return metrics


def _quality(row: Mapping[str, object]) -> DataQualitySummary:
    issues: list[DataQualityIssue] = []
    for name in ("open", "high", "low"):
        if name not in row or _is_missing(row[name]):
            issues.append(
                DataQualityIssue(
                    code=f"MISSING_{name.upper()}",
                    severity=DataQualitySeverity.WARNING,
                    component=_PRODUCER,
                    message=f"selected Meta row has no finite {name} value",
                )
            )
    return DataQualitySummary(issues=tuple(issues))


def _ticker(value: object, *, field_name: str) -> str:
    if _is_missing(value) or not str(value).strip():
        raise ValueError(f"{field_name} must contain a ticker")
    ticker = str(value).strip().upper().replace("$", "")
    if not ticker:
        raise ValueError(f"{field_name} must contain a ticker")
    return ticker


def _intent_idempotency_key(
    *,
    strategy_id: str,
    decision_bar: datetime,
    ticker: str,
    side: Direction,
    config_version: str,
    intent_ordinal: int,
) -> str:
    material = {
        "strategy_id": strategy_id,
        "decision_bar": decision_bar.isoformat(),
        "ticker": ticker,
        "side": side.value,
        "config_version": config_version,
        "intent_ordinal": intent_ordinal,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_snapshot_mapping(snapshot_id_by_ticker: Mapping[str, UUID]) -> dict[str, UUID]:
    if not isinstance(snapshot_id_by_ticker, Mapping):
        raise TypeError("snapshot_id_by_ticker must be a mapping")
    canonical: dict[str, UUID] = {}
    for key, snapshot_id in snapshot_id_by_ticker.items():
        ticker = _canonical_ticker(key, field_name="snapshot ticker")
        if ticker in canonical:
            raise ValueError(f"case-colliding snapshot keys for {ticker}")
        if not isinstance(snapshot_id, UUID):
            raise TypeError(f"snapshot ID for {ticker} must be a UUID")
        canonical[ticker] = snapshot_id
    return canonical


def build_trade_intents(
    ranked: pd.DataFrame,
    *,
    decision_time: datetime,
    decision_bar: datetime,
    snapshot_id_by_ticker: Mapping[str, UUID],
    config: MetaIntentConfig,
) -> tuple[TradeIntent, ...]:
    """Build deterministic Meta LONG ENTRY intents from selected ranked rows."""

    if not isinstance(ranked, pd.DataFrame):
        raise TypeError("ranked must be a pandas DataFrame")
    if not isinstance(config, MetaIntentConfig):
        raise TypeError("config must be a MetaIntentConfig")
    required = {"ticker", "timestamp", "close", "s_combo", "s_upside", "s_quality"}
    missing = sorted(required.difference(ranked.columns))
    if missing:
        raise KeyError(f"ranked is missing required intent columns: {missing}")

    decision_time_utc = _timestamp(decision_time, field_name="decision_time")
    decision_bar_utc = _timestamp(decision_bar, field_name="decision_bar")
    if decision_bar_utc > decision_time_utc:
        raise ValueError("decision_bar must not be after decision_time")
    row_timestamps = _timestamp_series(ranked["timestamp"], field_name="ranked row timestamps")
    if not bool(row_timestamps.eq(pd.Timestamp(decision_bar_utc)).all()):
        raise ValueError("ranked rows must all match the exact decision bar")
    snapshot_ids = _canonical_snapshot_mapping(snapshot_id_by_ticker)

    canonical_tickers = [
        _canonical_ticker(value, field_name="ranked ticker")
        for value in ranked["ticker"].tolist()
    ]
    if len(set(canonical_tickers)) != len(canonical_tickers):
        raise ValueError("duplicate canonical ticker in ranked intent rows")

    intents: list[TradeIntent] = []
    for row, ticker in zip(ranked.to_dict(orient="records"), canonical_tickers):
        quality = _finite(row.get("s_quality"), field_name=f"{ticker} s_quality")
        if ticker in config.held_tickers or quality < config.quality_floor:
            continue

        score_components = {
            name: _finite(row.get(name), field_name=f"{ticker} {name}")
            for name in ("s_combo", "s_upside", "s_quality")
        }
        reference_price = _finite(row.get("close"), field_name=f"{ticker} close")
        if reference_price <= 0:
            raise ValueError(f"{ticker} selected-bar close must be positive")
        try:
            snapshot_id = snapshot_ids[ticker]
        except KeyError as exc:
            raise KeyError(f"missing context snapshot ID for {ticker}") from exc

        intent_ordinal = len(intents) + 1
        side = Direction.LONG
        idempotency_key = _intent_idempotency_key(
            strategy_id=config.strategy_id,
            decision_bar=decision_bar_utc,
            ticker=ticker,
            side=side,
            config_version=config.config_version,
            intent_ordinal=intent_ordinal,
        )
        intents.append(
            TradeIntent(
                intent_id=uuid5(NAMESPACE_URL, idempotency_key),
                strategy_id=config.strategy_id,
                ticker=ticker,
                direction=side,
                decision_kind=DecisionKind.ENTRY,
                raw_score=score_components["s_combo"],
                raw_probability=None,
                expected_return=None,
                expected_holding_period=config.expected_holding_period,
                snapshot_id=snapshot_id,
                selected_bar=decision_bar_utc,
                entry_window=config.entry_window,
                preferred_entry=Decimal(str(reference_price)),
                invalidation=None,
                target=None,
                stop=None,
                position_size_requested=config.requested_notional,
                instrument_preferences=config.instrument_preferences,
                feature_timestamp=decision_bar_utc,
                created_at=decision_time_utc,
                model_version=config.model_version,
                feature_version=config.feature_version,
                reason_codes=config.reason_codes,
                score_components=score_components,
                config_version=config.config_version,
                idempotency_key=idempotency_key,
            )
        )
    return tuple(intents)


def adapt_scored_ticker_state(
    scored_row: Mapping[str, object],
    selected_bar: Mapping[str, object],
    *,
    bar_ticker: str,
    decision_bar: datetime,
    available_at: datetime,
    valid_until: datetime,
    matrix_lineage: tuple[LineageRef, ...],
    bar_lineage: tuple[LineageRef, ...],
) -> TickerState:
    """Join one scored Meta row to one explicitly selected exact 4H bar."""

    if not isinstance(scored_row, Mapping):
        raise TypeError("scored_row must be a single mapping")
    if not isinstance(selected_bar, Mapping):
        raise TypeError("selected_bar must be a single selected-bar mapping")
    if not matrix_lineage:
        raise ValueError("matrix lineage is required")
    if not bar_lineage:
        raise ValueError("bar lineage is required")

    decision_bar_utc = _timestamp(decision_bar, field_name="decision_bar")
    scored_ticker = _ticker(scored_row.get("ticker"), field_name="scored row")
    selected_ticker = _ticker(bar_ticker, field_name="bar_ticker")
    if selected_ticker != scored_ticker:
        raise ValueError("selected bar ticker does not match scored-row ticker")
    if "ticker" in selected_bar:
        embedded_ticker = _ticker(selected_bar.get("ticker"), field_name="selected bar")
        if embedded_ticker != selected_ticker:
            raise ValueError("selected bar ticker evidence conflicts with bar_ticker")

    selected_timestamp = _timestamp(
        selected_bar.get("timestamp"), field_name="selected bar timestamp"
    )
    if selected_timestamp != decision_bar_utc:
        raise ValueError("selected 4H bar timestamp does not match the decision bar")

    enriched = dict(scored_row)
    for name in ("open", "high", "low", "close", "volume"):
        if name in selected_bar:
            enriched[name] = selected_bar[name]

    return adapt_ticker_state(
        enriched,
        decision_bar=decision_bar_utc,
        available_at=available_at,
        valid_until=valid_until,
        lineage=_validated_lineage(matrix_lineage) + _validated_lineage(bar_lineage),
    )


def adapt_ticker_state(
    row: Mapping[str, object],
    *,
    decision_bar: datetime,
    available_at: datetime,
    valid_until: datetime,
    lineage: tuple[LineageRef, ...],
) -> TickerState:
    """Adapt exactly one already-selected, completed Meta matrix row.

    The caller owns selection from a matrix or Parquet file.  This function
    never searches for a latest row, so appending a later bar cannot revise an
    earlier state.
    """

    selected_bar = _timestamp(decision_bar, field_name="decision_bar")
    observed_at = _timestamp(available_at, field_name="available_at")
    valid_until_utc = _timestamp(valid_until, field_name="valid_until")
    if observed_at < selected_bar:
        raise ValueError("available_at precedes the selected decision bar; causal state is invalid")
    if valid_until_utc <= observed_at:
        raise ValueError("valid_until must be exclusive and after available_at")

    row_bar_value = row.get("timestamp")
    if _is_missing(row_bar_value):
        raise ValueError("selected Meta row must contain its timestamp/selected bar")
    row_bar = _timestamp(row_bar_value, field_name="row timestamp")
    if row_bar != selected_bar:
        raise ValueError("row timestamp does not match the selected decision bar")

    ticker = _ticker(row.get("ticker"), field_name="selected Meta row")

    close = _finite(row.get("close"), field_name="close")
    validated_lineage = _validated_lineage(lineage)
    metrics = _metric_values(row)
    ticker_setup = _ticker_setup(row)
    trend_state = _text(row, "trend_state")
    relative_strength_state = _text(row, "relative_strength_state")
    support_state = _text(row, "support_state")
    volume_state = _text(row, "volume_state")
    reversal_state = _text(row, "reversal_state")
    breakdown_state = _text(row, "breakdown_state")
    theme_alignment = _optional_finite(row, "theme_alignment")
    market_alignment = _optional_finite(row, "market_alignment")
    dealer_alignment = _optional_finite(row, "dealer_alignment")
    data_quality = _quality(row)
    content_revision = _content_revision_digest(
        valid_until=valid_until_utc,
        reference_price=close,
        ticker_setup=ticker_setup,
        trend_state=trend_state,
        relative_strength_state=relative_strength_state,
        support_state=support_state,
        volume_state=volume_state,
        reversal_state=reversal_state,
        breakdown_state=breakdown_state,
        theme_alignment=theme_alignment,
        market_alignment=market_alignment,
        dealer_alignment=dealer_alignment,
        metrics=metrics,
        data_quality=data_quality,
    )

    return TickerState(
        state_id=_stable_state_id(
            ticker=ticker,
            selected_bar=selected_bar,
            available_at=observed_at,
            lineage=validated_lineage,
            content_revision=content_revision,
        ),
        state_type=StateType.TICKER,
        entity_id=ticker,
        as_of=selected_bar,
        available_at=observed_at,
        generated_at=observed_at,
        valid_until=valid_until_utc,
        source_window_start=selected_bar,
        source_window_end=selected_bar,
        schema_version=1,
        producer=_PRODUCER,
        model_version=_MODEL_VERSION,
        feature_version=_FEATURE_VERSION,
        config_version=_CONFIG_VERSION,
        lineage_ids=_lineage_ids(validated_lineage),
        data_quality=data_quality,
        ticker=ticker,
        selected_bar=selected_bar,
        reference_price=close,
        ticker_setup=ticker_setup,
        trend_state=trend_state,
        relative_strength_state=relative_strength_state,
        support_state=support_state,
        volume_state=volume_state,
        reversal_state=reversal_state,
        breakdown_state=breakdown_state,
        theme_alignment=theme_alignment,
        market_alignment=market_alignment,
        dealer_alignment=dealer_alignment,
        metrics=metrics,
        transition_probabilities={},
    )


__all__ = [
    "MetaIntentConfig",
    "adapt_scored_ticker_state",
    "adapt_ticker_state",
    "build_trade_intents",
]
