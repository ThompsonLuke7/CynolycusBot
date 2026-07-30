from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

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


class ContextSnapshot(ContractModel):
    snapshot_id: UUID
    decision_time: UtcDatetime
    strategy_id: str
    ticker: str
    freshness_profile: str
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
    data_quality: DataQualitySummary = Field(default_factory=DataQualitySummary)
    config_version: str = "UNKNOWN"
    model_versions: tuple[str, ...] = ()
    feature_versions: tuple[str, ...] = ()
    schema_version: PositiveSchemaVersion = 1
    content_hash: str

    def computed_content_hash(self) -> str:
        return content_hash(self, exclude={"snapshot_id", "content_hash"})

    @model_validator(mode="after")
    def validate_hash_references(self) -> ContextSnapshot:
        if len(self.state_ids) != len(self.state_hashes):
            raise ValueError("state_ids and state_hashes must be one-to-one")
        if len(set(self.state_ids)) != len(self.state_ids):
            raise ValueError("snapshot contains duplicate state_id")
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
    ) -> ContextSnapshot:
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
                if state.available_at > decision_time:
                    raise ValueError(
                        f"state {state.state_id} is unavailable at decision time"
                    )
                if decision_time >= state.valid_until:
                    raise ValueError(f"state {state.state_id} is expired at decision time")
                if isinstance(state, ThemeMembership) and (
                    decision_time < state.effective_from
                    or (
                        state.effective_until is not None
                        and decision_time >= state.effective_until
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
        payload = {
            "snapshot_id": snapshot_id,
            "decision_time": decision_time,
            "strategy_id": strategy_id,
            "ticker": ticker,
            "freshness_profile": freshness_profile,
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
            "data_quality": quality,
            "config_version": config_version,
            "model_versions": versions["model_versions"],
            "feature_versions": versions["feature_versions"],
            "schema_version": 1,
            "content_hash": "",
        }
        snapshot = cls(**payload)
        return snapshot.model_copy(update={"content_hash": snapshot.computed_content_hash()})


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


__all__ = ["ContextSnapshot", "FreshnessResult", "FreshnessStatus", "StateRequest"]
