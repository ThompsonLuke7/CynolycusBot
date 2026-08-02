from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
import math
from numbers import Real
import re
from typing import Annotated
from uuid import UUID

from pydantic import BeforeValidator, Field, model_validator

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


def _validate_score_components_input(value: object) -> object:
    if not isinstance(value, Mapping):
        raise ValueError("score_components must be a mapping")
    for key, component in value.items():
        if not isinstance(key, str):
            raise ValueError("score component names must be strings")
        if isinstance(component, (bool, str, bytes)) or not isinstance(component, (Real, Decimal)):
            raise ValueError("score components must be numeric, not strings or booleans")
        try:
            numeric = float(component)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("score components must be finite numeric values") from exc
        if not math.isfinite(numeric):
            raise ValueError("score components must be finite numeric values")
    return value


StrictScoreComponents = Annotated[
    ImmutableFloatMap,
    BeforeValidator(_validate_score_components_input),
]


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
    score_components: StrictScoreComponents = Field(default_factory=dict, validate_default=True)
    config_version: str = "UNKNOWN"
    idempotency_key: str = ""

    @model_validator(mode="after")
    def validate_timing(self) -> TradeIntent:
        if self.feature_timestamp > self.created_at:
            raise ValueError("feature_timestamp must not be after created_at")
        legacy = self.score_components == {} and self.config_version == "UNKNOWN" and self.idempotency_key == ""
        if legacy:
            return self
        if not self.score_components:
            raise ValueError("new-format TradeIntent requires score_components")
        if not self.config_version.strip() or self.config_version.upper() == "UNKNOWN":
            raise ValueError("new-format TradeIntent requires a non-UNKNOWN config_version")
        if re.fullmatch(r"[0-9a-f]{64}", self.idempotency_key) is None:
            raise ValueError("new-format TradeIntent requires a lowercase 64-hex idempotency_key")
        if self.snapshot_id is None:
            raise ValueError("new-format TradeIntent requires snapshot_id lineage")
        return self
