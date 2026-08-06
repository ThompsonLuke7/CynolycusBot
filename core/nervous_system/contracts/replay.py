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
from typing import Literal

from pydantic import Field, model_validator

from .base import ContractModel, FiniteDecimal, content_hash


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
