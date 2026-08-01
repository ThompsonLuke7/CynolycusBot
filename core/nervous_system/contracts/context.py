from __future__ import annotations

import re
from typing import Literal
from uuid import UUID

from pydantic import Field, TypeAdapter, model_validator

from .base import ContractModel, FiniteFloat, PositiveSchemaVersion, UtcDatetime, content_hash
from .enums import StateType
from .quality import DataQualitySummary
from .states import (
    CatalystEvent,
    CatalystPressure,
    DealerState,
    MarketState,
    PortfolioState,
    ReadinessState,
    SectorState,
    StateContract,
    StateEnvelope,
    ThemeMembership,
    ThemeState,
    TickerState,
)


FreshnessStatus = Literal["FRESH", "STALE", "MISSING", "INVALID"]


class StateRequest(ContractModel):
    state_type: StateType
    entity_id: str
    required: bool
    bar_bound: UtcDatetime | None = None


class FreshnessResult(ContractModel):
    state_type: StateType
    entity_id: str
    required: bool
    status: FreshnessStatus
    selected_state_id: UUID | None = None
    age_seconds: FiniteFloat | None = None
    max_age_seconds: FiniteFloat
    reason_code: str


class RejectedCandidate(ContractModel):
    """Evidence for a causal state candidate that was not selected."""

    state_id: UUID
    state_type: StateType
    entity_id: str
    reason_code: str


class _SnapshotHashMaterial(ContractModel):
    decision_time: UtcDatetime
    strategy_id: str
    ticker: str
    freshness_profile: str
    state_hashes: tuple[str, ...]
    stale_inputs: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    data_quality: DataQualitySummary
    config_version: str
    model_versions: tuple[str, ...]
    feature_versions: tuple[str, ...]
    schema_version: PositiveSchemaVersion


class _ExtendedSnapshotHashMaterial(ContractModel):
    decision_time: UtcDatetime
    decision_bar: UtcDatetime | None
    decision_session: str | None
    strategy_id: str
    ticker: str
    freshness_profile: str
    freshness_profile_hash: str
    state_hashes: tuple[str, ...]
    stale_inputs: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    rejected_candidates: tuple[RejectedCandidate, ...]
    requirement_results: tuple[FreshnessResult, ...]
    valid: bool
    data_quality: DataQualitySummary
    config_version: str
    model_versions: tuple[str, ...]
    feature_versions: tuple[str, ...]
    schema_version: PositiveSchemaVersion


_UTC_DATETIME_ADAPTER = TypeAdapter(UtcDatetime)


def _snapshot_hash_material(
    *,
    decision_time: UtcDatetime,
    strategy_id: str,
    ticker: str,
    freshness_profile: str,
    state_hashes: tuple[str, ...],
    stale_inputs: tuple[str, ...],
    missing_inputs: tuple[str, ...],
    data_quality: DataQualitySummary,
    config_version: str,
    model_versions: tuple[str, ...],
    feature_versions: tuple[str, ...],
    schema_version: PositiveSchemaVersion,
) -> _SnapshotHashMaterial:
    return _SnapshotHashMaterial(
        decision_time=decision_time,
        strategy_id=strategy_id,
        ticker=ticker,
        freshness_profile=freshness_profile,
        state_hashes=state_hashes,
        stale_inputs=stale_inputs,
        missing_inputs=missing_inputs,
        data_quality=data_quality,
        config_version=config_version,
        model_versions=model_versions,
        feature_versions=feature_versions,
        schema_version=schema_version,
    )


def _extended_snapshot_hash_material(
    *,
    decision_time: UtcDatetime,
    decision_bar: UtcDatetime | None,
    decision_session: str | None,
    strategy_id: str,
    ticker: str,
    freshness_profile: str,
    freshness_profile_hash: str,
    state_hashes: tuple[str, ...],
    stale_inputs: tuple[str, ...],
    missing_inputs: tuple[str, ...],
    rejected_candidates: tuple[RejectedCandidate, ...],
    requirement_results: tuple[FreshnessResult, ...],
    valid: bool,
    data_quality: DataQualitySummary,
    config_version: str,
    model_versions: tuple[str, ...],
    feature_versions: tuple[str, ...],
    schema_version: PositiveSchemaVersion,
) -> _ExtendedSnapshotHashMaterial:
    return _ExtendedSnapshotHashMaterial(
        decision_time=decision_time,
        decision_bar=decision_bar,
        decision_session=decision_session,
        strategy_id=strategy_id,
        ticker=ticker,
        freshness_profile=freshness_profile,
        freshness_profile_hash=freshness_profile_hash,
        state_hashes=state_hashes,
        stale_inputs=stale_inputs,
        missing_inputs=missing_inputs,
        rejected_candidates=rejected_candidates,
        requirement_results=requirement_results,
        valid=valid,
        data_quality=data_quality,
        config_version=config_version,
        model_versions=model_versions,
        feature_versions=feature_versions,
        schema_version=schema_version,
    )


_DISPATCH = {
    MarketState: "market_state",
    SectorState: "sector_states",
    ThemeMembership: "theme_memberships",
    ThemeState: "theme_states",
    TickerState: "ticker_state",
    CatalystEvent: "catalyst_events",
    CatalystPressure: "catalyst_pressures",
    DealerState: "dealer_state",
    PortfolioState: "portfolio_state",
    ReadinessState: "readiness_state",
}

_SINGLETON_TYPES = frozenset({MarketState, TickerState, DealerState, PortfolioState, ReadinessState})
_BAR_BOUND_STATE_TYPES = (MarketState, SectorState, ThemeMembership, ThemeState, TickerState)


class ContextSnapshot(ContractModel):
    snapshot_id: UUID
    decision_time: UtcDatetime
    strategy_id: str
    ticker: str
    freshness_profile: str
    freshness_profile_hash: str = ""
    decision_bar: UtcDatetime | None = None
    decision_session: str | None = None
    market_state: MarketState | None = None
    sector_states: tuple[SectorState, ...] = ()
    theme_memberships: tuple[ThemeMembership, ...] = ()
    theme_states: tuple[ThemeState, ...] = ()
    ticker_state: TickerState | None = None
    catalyst_events: tuple[CatalystEvent, ...] = ()
    catalyst_pressures: tuple[CatalystPressure, ...] = ()
    dealer_state: DealerState | None = None
    portfolio_state: PortfolioState | None = None
    readiness_state: ReadinessState | None = None
    state_ids: tuple[UUID, ...] = ()
    state_hashes: tuple[str, ...] = ()
    stale_inputs: tuple[str, ...] = ()
    missing_inputs: tuple[str, ...] = ()
    rejected_candidates: tuple[RejectedCandidate, ...] = ()
    requirement_results: tuple[FreshnessResult, ...] = ()
    valid: bool = True
    data_quality: DataQualitySummary = Field(default_factory=DataQualitySummary)
    config_version: str = "UNKNOWN"
    model_versions: tuple[str, ...] = ()
    feature_versions: tuple[str, ...] = ()
    schema_version: PositiveSchemaVersion = 1
    content_hash: str

    def computed_content_hash(self) -> str:
        if self.uses_extended_evidence:
            return content_hash(
                _extended_snapshot_hash_material(
                    decision_time=self.decision_time,
                    decision_bar=self.decision_bar,
                    decision_session=self.decision_session,
                    strategy_id=self.strategy_id,
                    ticker=self.ticker,
                    freshness_profile=self.freshness_profile,
                    freshness_profile_hash=self.freshness_profile_hash,
                    state_hashes=self.state_hashes,
                    stale_inputs=self.stale_inputs,
                    missing_inputs=self.missing_inputs,
                    rejected_candidates=self.rejected_candidates,
                    requirement_results=self.requirement_results,
                    valid=self.valid,
                    data_quality=self.data_quality,
                    config_version=self.config_version,
                    model_versions=self.model_versions,
                    feature_versions=self.feature_versions,
                    schema_version=self.schema_version,
                )
            )
        return content_hash(
            _snapshot_hash_material(
                decision_time=self.decision_time,
                strategy_id=self.strategy_id,
                ticker=self.ticker,
                freshness_profile=self.freshness_profile,
                state_hashes=self.state_hashes,
                stale_inputs=self.stale_inputs,
                missing_inputs=self.missing_inputs,
                data_quality=self.data_quality,
                config_version=self.config_version,
                model_versions=self.model_versions,
                feature_versions=self.feature_versions,
                schema_version=self.schema_version,
            )
        )

    @property
    def uses_extended_evidence(self) -> bool:
        return bool(
            self.freshness_profile_hash
            or self.decision_bar is not None
            or self.decision_session is not None
            or self.rejected_candidates
            or self.requirement_results
            or not self.valid
        )

    @property
    def is_valid(self) -> bool:
        return self.valid

    @property
    def profile_hash(self) -> str:
        return self.freshness_profile_hash

    @property
    def requirements(self) -> tuple[FreshnessResult, ...]:
        return self.requirement_results

    @property
    def freshness_results(self) -> tuple[FreshnessResult, ...]:
        return self.requirement_results

    @property
    def rejected_state_ids(self) -> tuple[UUID, ...]:
        return tuple(item.state_id for item in self.rejected_candidates)

    @property
    def rejected_candidate_ids(self) -> tuple[UUID, ...]:
        return self.rejected_state_ids

    @property
    def rejected_reasons(self) -> tuple[str, ...]:
        return tuple(item.reason_code for item in self.rejected_candidates)

    @property
    def warning_reasons(self) -> tuple[str, ...]:
        return tuple(
            result.reason_code
            for result in self.requirement_results
            if not result.required and result.status != "FRESH"
        )

    @model_validator(mode="after")
    def validate_hash_references(self) -> ContextSnapshot:
        if self.uses_extended_evidence:
            for field_name in ("strategy_id", "ticker", "freshness_profile"):
                value = getattr(self, field_name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{field_name} must be a non-empty string")
        if self.decision_bar is not None:
            if self.decision_bar > self.decision_time:
                raise ValueError("decision_bar must not be after decision_time")
            for state in _sorted_embedded_states(self):
                if isinstance(state, _BAR_BOUND_STATE_TYPES) and state.as_of > self.decision_bar:
                    raise ValueError(
                        f"bar-bound state {state.state_id} as_of is after decision_bar"
                    )
        if self.freshness_profile_hash and (
            len(self.freshness_profile_hash) != 64
            or re.fullmatch(r"[0-9a-fA-F]{64}", self.freshness_profile_hash) is None
        ):
            raise ValueError("freshness_profile_hash must be a SHA-256 hex string")
        if len(self.state_ids) != len(self.state_hashes):
            raise ValueError("state_ids and state_hashes must be one-to-one")
        if len(set(self.state_ids)) != len(self.state_ids):
            raise ValueError("snapshot contains duplicate state_id")
        embedded_states = _sorted_embedded_states(self)
        expected_state_ids = tuple(state.state_id for state in embedded_states)
        expected_state_hashes = tuple(
            content_hash(state, exclude={"state_id"}) for state in embedded_states
        )
        if self.state_ids != expected_state_ids:
            raise ValueError("state_ids do not match embedded states")
        if self.state_hashes != expected_state_hashes:
            raise ValueError("state_hashes do not match embedded states")
        if self.content_hash != self.computed_content_hash():
            raise ValueError("content_hash does not match snapshot content")
        return self

    @classmethod
    def from_states(
        cls,
        *,
        snapshot_id: UUID,
        decision_time: UtcDatetime,
        strategy_id: str,
        ticker: str,
        states: tuple[StateContract, ...],
        freshness_profile: str,
        stale_inputs: tuple[str, ...] = (),
        missing_inputs: tuple[str, ...] = (),
        freshness_profile_hash: str = "",
        decision_bar: UtcDatetime | None = None,
        decision_session: str | None = None,
        rejected_candidates: tuple[RejectedCandidate, ...] = (),
        requirement_results: tuple[FreshnessResult, ...] = (),
        valid: bool = True,
    ) -> ContextSnapshot:
        normalized_decision_time = _UTC_DATETIME_ADAPTER.validate_python(decision_time)
        buckets: dict[str, list[StateContract]] = {
            field_name: [] for field_name in _DISPATCH.values()
        }
        state_ids: list[UUID] = []
        state_hashes: list[str] = []
        seen_singletons: set[type[StateContract]] = set()
        seen_state_ids: set[UUID] = set()
        seen_collection_keys: set[tuple[str, ...]] = set()

        sorted_states = sorted(states, key=_state_sort_key)

        for state in sorted_states:
            field_name = _DISPATCH.get(type(state))
            if field_name is None:
                raise TypeError(f"unsupported concrete state type: {type(state).__name__}")
            if type(state) in _SINGLETON_TYPES:
                if type(state) in seen_singletons:
                    raise ValueError(f"duplicate singleton state: {type(state).__name__}")
                seen_singletons.add(type(state))

            collection_key = _collection_key(state)
            if collection_key is not None:
                if collection_key in seen_collection_keys:
                    raise ValueError(f"duplicate effective collection selection: {collection_key}")
                seen_collection_keys.add(collection_key)

            if isinstance(state, StateEnvelope):
                if state.state_id in seen_state_ids:
                    raise ValueError(f"duplicate state_id: {state.state_id}")
                seen_state_ids.add(state.state_id)
                if state.available_at > normalized_decision_time:
                    raise ValueError(
                        f"state {state.state_id} is unavailable at decision time"
                    )
                if normalized_decision_time >= state.valid_until:
                    raise ValueError(f"state {state.state_id} is expired at decision time")
                if isinstance(state, ThemeMembership) and (
                    normalized_decision_time < state.effective_from
                    or (
                        state.effective_until is not None
                        and normalized_decision_time >= state.effective_until
                    )
                ):
                    raise ValueError(
                        f"theme membership {state.ticker}/{state.theme_id} is invalid at decision time"
                    )
                state_ids.append(state.state_id)
                _validate_ticker_scope(state, ticker)

            buckets[field_name].append(state)
            state_hashes.append(content_hash(state, exclude={"state_id"}))

        envelope_states = [state for state in sorted_states if isinstance(state, StateEnvelope)]
        versions = {
            "config_version": tuple(dict.fromkeys(state.config_version for state in envelope_states)),
            "model_versions": tuple(dict.fromkeys(state.model_version for state in envelope_states)),
            "feature_versions": tuple(dict.fromkeys(state.feature_version for state in envelope_states)),
        }
        config_values = versions["config_version"]
        config_version = config_values[0] if len(config_values) == 1 else (
            "MIXED:" + ",".join(sorted(config_values)) if config_values else "UNKNOWN"
        )
        quality = DataQualitySummary(
            issues=tuple(issue for state in envelope_states for issue in state.data_quality.issues)
        )
        if (
            freshness_profile_hash
            or decision_bar is not None
            or decision_session is not None
            or rejected_candidates
            or requirement_results
            or not valid
        ):
            snapshot_content_hash = content_hash(
                _extended_snapshot_hash_material(
                    decision_time=normalized_decision_time,
                    decision_bar=decision_bar,
                    decision_session=decision_session,
                    strategy_id=strategy_id,
                    ticker=ticker,
                    freshness_profile=freshness_profile,
                    freshness_profile_hash=freshness_profile_hash,
                    state_hashes=tuple(state_hashes),
                    stale_inputs=stale_inputs,
                    missing_inputs=missing_inputs,
                    rejected_candidates=rejected_candidates,
                    requirement_results=requirement_results,
                    valid=valid,
                    data_quality=quality,
                    config_version=config_version,
                    model_versions=versions["model_versions"],
                    feature_versions=versions["feature_versions"],
                    schema_version=1,
                )
            )
        else:
            snapshot_content_hash = content_hash(
                _snapshot_hash_material(
                    decision_time=normalized_decision_time,
                    strategy_id=strategy_id,
                    ticker=ticker,
                    freshness_profile=freshness_profile,
                    state_hashes=tuple(state_hashes),
                    stale_inputs=stale_inputs,
                    missing_inputs=missing_inputs,
                    data_quality=quality,
                    config_version=config_version,
                    model_versions=versions["model_versions"],
                    feature_versions=versions["feature_versions"],
                    schema_version=1,
                )
            )
        payload = {
            "snapshot_id": snapshot_id,
            "decision_time": normalized_decision_time,
            "strategy_id": strategy_id,
            "ticker": ticker,
            "freshness_profile": freshness_profile,
            "freshness_profile_hash": freshness_profile_hash,
            "decision_bar": decision_bar,
            "decision_session": decision_session,
            "market_state": buckets["market_state"][0] if buckets["market_state"] else None,
            "sector_states": tuple(buckets["sector_states"]),
            "theme_memberships": tuple(buckets["theme_memberships"]),
            "theme_states": tuple(buckets["theme_states"]),
            "ticker_state": buckets["ticker_state"][0] if buckets["ticker_state"] else None,
            "catalyst_events": tuple(buckets["catalyst_events"]),
            "catalyst_pressures": tuple(buckets["catalyst_pressures"]),
            "dealer_state": buckets["dealer_state"][0] if buckets["dealer_state"] else None,
            "portfolio_state": buckets["portfolio_state"][0] if buckets["portfolio_state"] else None,
            "readiness_state": buckets["readiness_state"][0] if buckets["readiness_state"] else None,
            "state_ids": tuple(state_ids),
            "state_hashes": tuple(state_hashes),
            "stale_inputs": stale_inputs,
            "missing_inputs": missing_inputs,
            "rejected_candidates": rejected_candidates,
            "requirement_results": requirement_results,
            "valid": valid,
            "data_quality": quality,
            "config_version": config_version,
            "model_versions": versions["model_versions"],
            "feature_versions": versions["feature_versions"],
            "schema_version": 1,
            "content_hash": snapshot_content_hash,
        }
        return cls(**payload)


def _time_key(value: UtcDatetime | None) -> str:
    return value.isoformat() if value is not None else ""


def _state_sort_key(state: StateContract) -> tuple[str, ...]:
    if isinstance(state, ThemeMembership):
        business = (
            state.ticker,
            state.theme_id,
            _time_key(state.effective_from),
            _time_key(state.effective_until),
        )
    elif isinstance(state, CatalystEvent):
        business = (state.ticker or "", _time_key(state.event_time), str(state.event_id))
    elif isinstance(state, CatalystPressure):
        business = (state.scope_type, state.scope_id)
    elif isinstance(state, (TickerState, DealerState)):
        business = (state.ticker,)
    elif isinstance(state, SectorState):
        business = (state.sector_id,)
    elif isinstance(state, ThemeState):
        business = (state.theme_id,)
    elif isinstance(state, PortfolioState):
        business = (state.account_alias,)
    elif isinstance(state, ReadinessState):
        business = (state.job,)
    else:
        business = (state.entity_id,)

    if isinstance(state, StateEnvelope):
        return (
            type(state).__name__,
            *business,
            _time_key(state.as_of),
            _time_key(state.available_at),
            str(state.state_id),
        )
    return (type(state).__name__, *business)


def _sorted_embedded_states(snapshot: ContextSnapshot) -> tuple[StateContract, ...]:
    embedded_states: list[StateContract] = []
    for field_name in _DISPATCH.values():
        value = getattr(snapshot, field_name)
        if value is None:
            continue
        if isinstance(value, tuple):
            embedded_states.extend(value)
        else:
            embedded_states.append(value)
    return tuple(sorted(embedded_states, key=_state_sort_key))


def _collection_key(state: StateContract) -> tuple[str, ...] | None:
    if isinstance(state, SectorState):
        return ("sector", state.sector_id)
    if isinstance(state, ThemeMembership):
        return ("theme_membership", state.ticker, state.theme_id)
    if isinstance(state, ThemeState):
        return ("theme", state.theme_id)
    if isinstance(state, CatalystPressure):
        return ("catalyst_pressure", state.scope_type, state.scope_id)
    if isinstance(state, CatalystEvent):
        return ("catalyst_event", str(state.event_id))
    return None


def _validate_ticker_scope(state: StateContract, ticker: str) -> None:
    if isinstance(state, (TickerState, DealerState, ThemeMembership)):
        state_ticker = state.ticker
    elif isinstance(state, CatalystEvent) and state.ticker is not None:
        state_ticker = state.ticker
    elif isinstance(state, CatalystPressure) and state.scope_type.upper() == "TICKER":
        state_ticker = state.scope_id
    else:
        return
    if state_ticker != ticker:
        raise ValueError(
            f"ticker-scoped state {type(state).__name__} does not match snapshot ticker {ticker}"
        )


__all__ = [
    "ContextSnapshot",
    "FreshnessResult",
    "FreshnessStatus",
    "RejectedCandidate",
    "StateRequest",
]
