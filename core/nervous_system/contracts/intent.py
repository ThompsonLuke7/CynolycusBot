from __future__ import annotations

from uuid import UUID

from pydantic import Field, model_validator

from .base import (
    ContractModel,
    FiniteFloat,
    ImmutableFloatMap,
    NonNegativeDecimal,
    PositiveDecimal,
    Probability,
    UtcDatetime,
)
from .enums import DecisionKind, Direction, InstrumentFamily


class TradeIntent(ContractModel):
    intent_id: UUID
    strategy_id: str
    ticker: str
    direction: Direction
    decision_kind: DecisionKind
    raw_score: FiniteFloat
    raw_probability: Probability | None
    expected_return: FiniteFloat | None
    expected_holding_period: str
    snapshot_id: UUID | None = None
    selected_bar: UtcDatetime | None = None
    entry_window: str
    preferred_entry: PositiveDecimal | None
    invalidation: PositiveDecimal | None
    target: PositiveDecimal | None
    stop: PositiveDecimal | None
    position_size_requested: NonNegativeDecimal
    instrument_preferences: tuple[InstrumentFamily, ...]
    feature_timestamp: UtcDatetime
    created_at: UtcDatetime
    model_version: str
    feature_version: str
    reason_codes: tuple[str, ...]
    # Task 14 fields are optional so previously persisted/constructed intents
    # remain valid; new producers populate all three deterministically.
    score_components: ImmutableFloatMap = Field(default_factory=dict)
    config_version: str = "UNKNOWN"
    idempotency_key: str = ""

    @model_validator(mode="after")
    def validate_timing(self) -> TradeIntent:
        if self.feature_timestamp > self.created_at:
            raise ValueError("feature_timestamp must not be after created_at")
        return self
