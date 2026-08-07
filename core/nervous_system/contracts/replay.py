"""Replay evidence contracts and the option source-fitness verdict.

The types here exist to answer one question before any P&L is computed: can
this price source actually answer the question being asked of it?

That question has a documented wrong answer in this repo. A 2026-07
options-routing study was fully retracted because the option "prices" were
stale trade prints; the correlation between option returns and underlying
direction was +0.09 where a long call should be near +0.9. The gate's default
answer is therefore no — failing to disprove fitness is not the same as
establishing it.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, model_validator

from .base import (
    ContractModel,
    FiniteDecimal,
    NonNegativeDecimal,
    PositiveDecimal,
    PositiveSchemaVersion,
    Sha256Hex,
    UtcDatetime,
    content_hash,
)
from .enums import OrderSide


PositiveInt = Annotated[int, Field(gt=0)]


class MarkType(str, Enum):
    """How a price was arrived at. Only an observed two-sided market is a mark."""

    QUOTE_BID_ASK = "QUOTE_BID_ASK"
    MID = "MID"
    # Everything below is evidence of a trade, or of arithmetic — not a mark.
    TRADE_PRINT = "TRADE_PRINT"
    LAST_PRICE = "LAST_PRICE"
    SYNTHETIC = "SYNTHETIC"
    FORWARD_FILLED = "FORWARD_FILLED"
    INTERPOLATED = "INTERPOLATED"


# A trade print is not a mark, and neither is a number produced by filling a
# gap. These can never be fit for option P&L regardless of any other metric.
TRADE_DERIVED_MARKS = frozenset({MarkType.TRADE_PRINT, MarkType.LAST_PRICE})
FABRICATED_MARKS = frozenset(
    {MarkType.SYNTHETIC, MarkType.FORWARD_FILLED, MarkType.INTERPOLATED}
)
FIT_MARKS = frozenset({MarkType.QUOTE_BID_ASK})


class ObservationKind(str, Enum):
    """What kind of evidence an observation carries."""

    STATE = "STATE"
    BAR = "BAR"
    OPTION_QUOTE = "OPTION_QUOTE"
    BROKER_FILL = "BROKER_FILL"
    SOURCE_MANIFEST = "SOURCE_MANIFEST"


class Observation(ContractModel):
    """One piece of replay evidence, with the time it became knowable.

    ``as_of`` is business/event time; ``available_at`` is when we could first
    have known it. They are separate fields on purpose: a bar stamped 16:00
    that landed at 16:07 was not knowable at 16:03, and a replay that selects
    on event time will quietly outperform reality.

    ``available_at`` is always supplied by the producer and is never inferred
    from a file mtime, which records when bytes were written rather than when
    the information became available.
    """

    observation_id: UUID
    kind: ObservationKind
    instrument: str
    as_of: UtcDatetime
    available_at: UtcDatetime
    valid_until: UtcDatetime
    generated_at: UtcDatetime
    artifact_hash: Sha256Hex
    record_locator: str
    provider: str
    feed: str
    tier: str
    schema_version: PositiveSchemaVersion
    producer: str
    mark_type: MarkType | None = None
    # Bar-bound evidence is additionally clamped to the decision bar. Intraday
    # evidence such as an option quote is not: a quote observed after the bar
    # closed is legitimate for a decision taken after that bar.
    bar_bound: bool = False

    @model_validator(mode="after")
    def validate_window(self) -> Observation:
        if self.valid_until <= self.available_at:
            raise ValueError("valid_until must be after available_at")
        for name in ("instrument", "record_locator", "provider", "feed", "tier", "producer"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be a non-empty string")
        return self

    def producer_version(self) -> str:
        """Explicit version discriminator, mirroring the state selector."""

        return "|".join(
            (self.producer, f"{self.schema_version:020d}", self.provider, self.feed, self.tier)
        )


class AttributionStatus(str, Enum):
    """Whether an outcome is settled.

    PENDING is not zero. A maturing or still-open position has no result yet,
    and a zero would read as a real flat result that drags every aggregate
    built over it toward the middle.
    """

    PENDING = "PENDING"
    FINAL = "FINAL"


class FillFact(ContractModel):
    """One confirmed broker fill. Requested quantity is not a fill."""

    leg_symbol: str
    side: OrderSide
    quantity: PositiveDecimal
    price: NonNegativeDecimal
    filled_at: UtcDatetime
    fees: NonNegativeDecimal = Decimal("0")
    contract_multiplier: PositiveInt = 1

    @property
    def signed_cash(self) -> Decimal:
        """Cash effect of the fill, before fees. Buying spends, selling receives."""

        gross = self.quantity * self.price * self.contract_multiplier
        return -gross if self.side is OrderSide.BUY else gross


class OutcomeAttribution(ContractModel):
    """A realized result split into parts that sum back to it exactly."""

    status: AttributionStatus
    realized_pnl: Decimal | None = None
    underlying_movement: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")
    instrument_transformation: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    filled_entry_quantity: Decimal = Decimal("0")
    filled_exit_quantity: Decimal = Decimal("0")
    excluded_fill_count: int = 0


class SourceFitnessStatus(str, Enum):
    FIT_FOR_OPTION_PNL = "FIT_FOR_OPTION_PNL"
    SOURCE_FITNESS_INSUFFICIENT_SAMPLE = "SOURCE_FITNESS_INSUFFICIENT_SAMPLE"
    SOURCE_UNFIT_FOR_OPTION_PNL = "SOURCE_UNFIT_FOR_OPTION_PNL"


class FitnessReason(str, Enum):
    TRADE_ONLY = "TRADE_ONLY"
    ENTITLEMENT_UNVERIFIED = "ENTITLEMENT_UNVERIFIED"
    STALE_MARKS = "STALE_MARKS"
    CROSSED_QUOTES = "CROSSED_QUOTES"
    SYNTHETIC_MARKS = "SYNTHETIC_MARKS"
    LOW_DERIVATIVE_CORRELATION = "LOW_DERIVATIVE_CORRELATION"
    INSUFFICIENT_POSITIONS = "INSUFFICIENT_POSITIONS"
    INSUFFICIENT_SESSIONS = "INSUFFICIENT_SESSIONS"
    SIDE_NOT_EVALUATED = "SIDE_NOT_EVALUATED"
    CORRELATION_BELOW_WARNING_BAND = "CORRELATION_BELOW_WARNING_BAND"


OptionSide = Literal["CALL", "PUT"]


class SourceFitnessThresholds(ContractModel):
    """The bar a source must clear. A verdict is meaningless without it."""

    min_matched_positions: int = Field(default=30, ge=1)
    min_sessions: int = Field(default=10, ge=1)
    min_valid_quote_fraction: Decimal = Field(default=Decimal("0.95"), ge=0, le=1)
    max_identical_mark_fraction: Decimal = Field(default=Decimal("0.05"), ge=0, le=1)
    # A long call should track its underlying closely. +0.70 is the floor;
    # anything under +0.85 is usable but worth saying out loud.
    min_pearson: Decimal = Field(default=Decimal("0.70"), ge=-1, le=1)
    warn_pearson: Decimal = Field(default=Decimal("0.85"), ge=-1, le=1)
    max_quote_age_seconds: Decimal = Field(default=Decimal("120"), gt=0)

    @model_validator(mode="after")
    def validate_bands(self) -> SourceFitnessThresholds:
        if self.warn_pearson < self.min_pearson:
            raise ValueError("warn_pearson must not be below min_pearson")
        return self

    def content_hash(self) -> str:
        return content_hash(self)


class SideFitnessMetrics(ContractModel):
    """Measured behaviour of one option side against its underlying.

    For a put the correlation is measured against the *negated* underlying, so
    a healthy series reads positive on both sides and one threshold governs
    both.
    """

    option_type: OptionSide
    mark_type: MarkType
    matched_positions: int = Field(ge=0)
    sessions: int = Field(ge=0)
    valid_quote_fraction: Decimal = Field(ge=0, le=1)
    identical_mark_fraction: Decimal = Field(ge=0, le=1)
    pearson: FiniteDecimal = Field(ge=-1, le=1)
    spearman: FiniteDecimal = Field(ge=-1, le=1)
    max_quote_age_seconds: Decimal = Field(ge=0)
    entitlement_verified: bool


class SourceFitnessReport(ContractModel):
    status: SourceFitnessStatus
    reasons: tuple[FitnessReason, ...] = ()
    warnings: tuple[FitnessReason, ...] = ()
    sides: tuple[SideFitnessMetrics, ...] = ()
    thresholds_hash: str
    source: str
    feed: str
    tier: str

    @property
    def option_pnl_eligible(self) -> bool:
        """Only an affirmatively fit source may produce option P&L."""

        return self.status is SourceFitnessStatus.FIT_FOR_OPTION_PNL

    def side(self, option_type: OptionSide) -> SideFitnessMetrics:
        for metrics in self.sides:
            if metrics.option_type == option_type:
                return metrics
        raise KeyError(f"no fitness metrics for {option_type}")


__all__ = [
    "FABRICATED_MARKS",
    "AttributionStatus",
    "FillFact",
    "OutcomeAttribution",
    "Observation",
    "ObservationKind",
    "FIT_MARKS",
    "TRADE_DERIVED_MARKS",
    "FitnessReason",
    "MarkType",
    "OptionSide",
    "SideFitnessMetrics",
    "SourceFitnessReport",
    "SourceFitnessStatus",
    "SourceFitnessThresholds",
]
