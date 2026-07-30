from __future__ import annotations

from typing import ClassVar
from uuid import UUID

from pydantic import Field, model_validator

from .base import (
    ContractModel,
    FiniteFloat,
    ImmutableFloatMap,
    ImmutableProbabilityMap,
    PositiveSchemaVersion,
    Probability,
    UtcDatetime,
)
from .enums import (
    AssetClass,
    DealerRegime,
    Direction,
    MarketRegime,
    StateType,
    ThemeRegime,
    TickerSetup,
)
from .quality import DataQualitySummary


class StateEnvelope(ContractModel):
    state_id: UUID
    state_type: StateType
    entity_id: str
    as_of: UtcDatetime
    available_at: UtcDatetime
    generated_at: UtcDatetime
    valid_until: UtcDatetime
    source_window_start: UtcDatetime
    source_window_end: UtcDatetime
    schema_version: PositiveSchemaVersion
    producer: str
    model_version: str
    feature_version: str
    config_version: str
    lineage_ids: tuple[str, ...]
    data_quality: DataQualitySummary

    expected_state_type: ClassVar[StateType | None] = None
    identity_field: ClassVar[str | None] = None

    @model_validator(mode="after")
    def validate_time_window(self) -> StateEnvelope:
        if self.source_window_start > self.source_window_end:
            raise ValueError("source window start follows source window end")
        if self.source_window_end > self.as_of:
            raise ValueError("source window ends after as_of")
        if self.available_at > self.generated_at:
            raise ValueError("generated_at precedes available_at")
        if self.valid_until <= self.available_at:
            raise ValueError("valid_until must be exclusive and after available_at")
        if self.expected_state_type is not None and self.state_type is not self.expected_state_type:
            raise ValueError(
                f"{type(self).__name__} requires state_type={self.expected_state_type.value}"
            )
        identity_field = getattr(type(self), "identity_field", None)
        if identity_field is not None and self.entity_id != str(getattr(self, identity_field)):
            raise ValueError(
                f"entity_id does not match {identity_field} for {type(self).__name__}"
            )
        return self


class MarketState(StateEnvelope):
    state_type: StateType = StateType.MARKET
    expected_state_type: ClassVar[StateType] = StateType.MARKET
    regime: MarketRegime
    risk_on_probability: Probability | None = None
    risk_off_probability: Probability | None = None
    identity_field: ClassVar[str | None] = None
    metrics: ImmutableFloatMap = Field(default_factory=dict)
    transition_probabilities: ImmutableProbabilityMap = Field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()


class SectorState(StateEnvelope):
    state_type: StateType = StateType.SECTOR
    expected_state_type: ClassVar[StateType] = StateType.SECTOR
    sector_id: str
    sector_regime: str = "UNKNOWN"
    relative_strength: FiniteFloat | None = None
    breadth: FiniteFloat | None = None
    momentum: FiniteFloat | None = None
    volatility: FiniteFloat | None = None
    rotation_rank: FiniteFloat | None = None
    rank_change: FiniteFloat | None = None
    capital_flow_direction: Direction = Direction.UNKNOWN
    identity_field: ClassVar[str] = "sector_id"
    transition_probabilities: ImmutableProbabilityMap = Field(default_factory=dict)


class ThemeMembership(StateEnvelope):
    state_type: StateType = StateType.THEME
    expected_state_type: ClassVar[StateType] = StateType.THEME
    identity_field: ClassVar[str] = "theme_id"
    ticker: str
    theme_id: str
    weight: FiniteFloat
    membership_version: str
    effective_from: UtcDatetime
    effective_until: UtcDatetime | None = None

    @model_validator(mode="after")
    def validate_effective_window(self) -> ThemeMembership:
        if self.effective_until is not None and self.effective_until <= self.effective_from:
            raise ValueError("effective_until must be after effective_from")
        return self


class ThemeState(StateEnvelope):
    state_type: StateType = StateType.THEME
    expected_state_type: ClassVar[StateType] = StateType.THEME
    theme_id: str
    theme_regime: ThemeRegime
    relative_strength: FiniteFloat | None = None
    breadth: FiniteFloat | None = None
    momentum: FiniteFloat | None = None
    distribution_score: FiniteFloat | None = None
    correlation_score: FiniteFloat | None = None
    volatility_score: FiniteFloat | None = None
    catalyst_pressure: FiniteFloat | None = None
    dealer_fragility: FiniteFloat | None = None
    leadership_score: FiniteFloat | None = None
    rotation_rank: FiniteFloat | None = None
    identity_field: ClassVar[str] = "theme_id"
    transition_probabilities: ImmutableProbabilityMap = Field(default_factory=dict)


class TickerState(StateEnvelope):
    state_type: StateType = StateType.TICKER
    expected_state_type: ClassVar[StateType] = StateType.TICKER
    ticker: str
    selected_bar: UtcDatetime
    reference_price: FiniteFloat
    ticker_setup: TickerSetup
    trend_state: str = "UNKNOWN"
    relative_strength_state: str = "UNKNOWN"
    support_state: str = "UNKNOWN"
    volume_state: str = "UNKNOWN"
    reversal_state: str = "UNKNOWN"
    breakdown_state: str = "UNKNOWN"
    theme_alignment: FiniteFloat | None = None
    market_alignment: FiniteFloat | None = None
    dealer_alignment: FiniteFloat | None = None
    identity_field: ClassVar[str] = "ticker"
    metrics: ImmutableFloatMap = Field(default_factory=dict)
    transition_probabilities: ImmutableProbabilityMap = Field(default_factory=dict)


class CatalystEvent(StateEnvelope):
    state_type: StateType = StateType.CATALYST_EVENT
    expected_state_type: ClassVar[StateType] = StateType.CATALYST_EVENT
    event_id: UUID
    ticker: str | None = None
    event_type: str
    event_time: UtcDatetime | None = None
    published_at: UtcDatetime | None = None
    observed_at: UtcDatetime
    source: str
    headline: str | None = None
    channel: str
    relation_confidence: Probability | None = None
    is_direct: bool | None = None

    @model_validator(mode="after")
    def validate_event_timing(self) -> CatalystEvent:
        if self.observed_at > self.available_at:
            raise ValueError("observed_at cannot follow available_at")
        return self


class CatalystPressure(StateEnvelope):
    state_type: StateType = StateType.CATALYST_PRESSURE
    expected_state_type: ClassVar[StateType] = StateType.CATALYST_PRESSURE
    scope_type: str
    scope_id: str
    identity_field: ClassVar[str] = "scope_id"
    channel_scores: ImmutableFloatMap = Field(default_factory=dict)
    aggregate_score: FiniteFloat | None = None
    event_ids: tuple[UUID, ...] = ()
    transition_probabilities: ImmutableProbabilityMap = Field(default_factory=dict)


class DealerState(StateEnvelope):
    state_type: StateType = StateType.DEALER
    expected_state_type: ClassVar[StateType] = StateType.DEALER
    ticker: str
    dealer_regime: DealerRegime
    spot: FiniteFloat
    total_gex: FiniteFloat | None = None
    call_wall: FiniteFloat | None = None
    put_wall: FiniteFloat | None = None
    nearest_magnet: FiniteFloat | None = None
    gamma_flip: FiniteFloat | None = None
    air_gap_above_score: FiniteFloat | None = None
    air_gap_below_score: FiniteFloat | None = None
    pinning_score: FiniteFloat | None = None
    acceleration_score: FiniteFloat | None = None
    identity_field: ClassVar[str] = "ticker"
    metrics: ImmutableFloatMap = Field(default_factory=dict)


class PortfolioPosition(ContractModel):
    broker_position_id: str
    symbol: str
    underlying: str
    asset_class: AssetClass
    quantity: FiniteFloat
    average_entry_price: FiniteFloat | None = None
    current_price: FiniteFloat | None = None
    market_value: FiniteFloat | None = None
    strategy_id: str | None = None
    ownership_status: str = "UNKNOWN"


class PortfolioState(StateEnvelope):
    state_type: StateType = StateType.PORTFOLIO
    expected_state_type: ClassVar[StateType] = StateType.PORTFOLIO
    account_alias: str
    equity: FiniteFloat
    cash: FiniteFloat
    buying_power: FiniteFloat
    day_pl: FiniteFloat | None = None
    positions: tuple[PortfolioPosition, ...] = ()
    open_order_ids: tuple[str, ...] = ()
    identity_field: ClassVar[str] = "account_alias"
    broker_observed_at: UtcDatetime


class ReadinessState(StateEnvelope):
    state_type: StateType = StateType.READINESS
    expected_state_type: ClassVar[StateType] = StateType.READINESS
    job: str
    status: str
    ready: bool
    completed_at: UtcDatetime | None = None
    checked_at: UtcDatetime
    max_age_hours: FiniteFloat
    latest_required_session: str
    identity_field: ClassVar[str] = "job"
    reason_codes: tuple[str, ...] = ()


StateContract = (
    MarketState
    | SectorState
    | ThemeMembership
    | ThemeState
    | TickerState
    | CatalystEvent
    | CatalystPressure
    | DealerState
    | PortfolioState
    | ReadinessState
)

__all__ = [
    "CatalystEvent",
    "CatalystPressure",
    "DealerState",
    "MarketState",
    "PortfolioPosition",
    "PortfolioState",
    "ReadinessState",
    "SectorState",
    "StateContract",
    "StateEnvelope",
    "ThemeMembership",
    "ThemeState",
    "TickerState",
]
