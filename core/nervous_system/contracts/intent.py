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
from .enums import DecisionKind, Direction, InstrumentFamily, SizeUnit


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
    # Optional only for reductions: a held name can leave the scored universe
    # entirely, and a risk-reducing action must not depend on a fresh opinion.
    raw_score: FiniteFloat | None
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
    # An unlabelled size means dollars to the policy engine and shares to the
    # exit ladder. New producers must say which.
    position_size_unit: SizeUnit = SizeUnit.UNKNOWN

    @model_validator(mode="after")
    def validate_timing(self) -> TradeIntent:
        if self.feature_timestamp > self.created_at:
            raise ValueError("feature_timestamp must not be after created_at")
        legacy = self.score_components == {} and self.config_version == "UNKNOWN" and self.idempotency_key == ""
        if legacy:
            return self
        # A reduction may be unscored (see raw_score); an entry may not.
        if not self.score_components and self.decision_kind is DecisionKind.ENTRY:
            raise ValueError("a new-format ENTRY requires score_components")
        if not self.config_version.strip() or self.config_version.upper() == "UNKNOWN":
            raise ValueError("new-format TradeIntent requires a non-UNKNOWN config_version")
        if re.fullmatch(r"[0-9a-f]{64}", self.idempotency_key) is None:
            raise ValueError("new-format TradeIntent requires a lowercase 64-hex idempotency_key")
        if self.position_size_unit is SizeUnit.UNKNOWN:
            raise ValueError("new-format TradeIntent requires an explicit position_size_unit")
        if self.snapshot_id is None:
            raise ValueError("new-format TradeIntent requires snapshot_id lineage")
        return self

    @model_validator(mode="after")
    def validate_entry_is_explained(self) -> TradeIntent:
        # Opening risk always requires a score. Only a reduction may proceed
        # without one.
        if self.decision_kind is DecisionKind.ENTRY and self.raw_score is None:
            raise ValueError("an ENTRY requires a raw_score")
        return self

    @model_validator(mode="after")
    def validate_exit_size_unit(self) -> TradeIntent:
        # A dollar figure cannot close a position exactly, so an exit that is
        # denominated in money would always round to a residual holding.
        if (
            self.decision_kind is DecisionKind.EXIT
            and self.position_size_unit is SizeUnit.NOTIONAL_USD
        ):
            raise ValueError("an EXIT requires a typed quantity unit, not NOTIONAL_USD")
        return self
