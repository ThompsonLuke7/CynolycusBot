"""Portfolio exposure, ownership, and reconciliation contracts.

Exposure values are money or share quantities, so they are ``Decimal``
throughout.  The shared ``_freeze_mapping`` helper renders ``Decimal`` as a
string, so these contracts use their own freezing validator that preserves the
numeric type.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Annotated
from uuid import UUID

from pydantic import AfterValidator, Field, model_validator

from .base import (
    ContractModel,
    FiniteDecimal,
    FrozenDict,
    NonNegativeDecimal,
    Sha256Hex,
    UtcDatetime,
    content_hash,
)
from .enums import OwnershipStatus, ReconciliationStatus
from .quality import DataQualitySummary


def _freeze_decimal_mapping(value: Mapping[str, Decimal]) -> FrozenDict[str, Decimal]:
    frozen: dict[str, Decimal] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("exposure keys must be non-empty strings")
        if isinstance(item, bool) or not isinstance(item, Decimal):
            raise ValueError(f"exposure value for {key!r} must be a Decimal")
        if not item.is_finite():
            raise ValueError(f"exposure value for {key!r} must be finite")
        frozen[key] = item
    return FrozenDict(frozen)


ImmutableDecimalMap = Annotated[
    dict[str, Decimal], AfterValidator(_freeze_decimal_mapping)
]

_EXPOSURE_HASH_EXCLUDE = frozenset({"report_id", "content_hash"})
_RECONCILIATION_HASH_EXCLUDE = frozenset({"reconciliation_id", "content_hash"})


class ExposureLimitResult(ContractModel):
    """One post-trade limit evaluation."""

    limit_id: str
    scope: str
    scope_id: str
    observed: FiniteDecimal
    limit_value: FiniteDecimal
    breached: bool
    reason_code: str


class ExposureReport(ContractModel):
    report_id: UUID
    portfolio_state_id: UUID
    snapshot_id: UUID
    calculated_at: UtcDatetime
    gross_notional: NonNegativeDecimal
    net_notional: FiniteDecimal
    long_notional: NonNegativeDecimal
    short_notional: NonNegativeDecimal
    symbol_notional: ImmutableDecimalMap = Field(default_factory=dict)
    underlying_equivalent: ImmutableDecimalMap = Field(default_factory=dict)
    sector_notional: ImmutableDecimalMap = Field(default_factory=dict)
    theme_notional: ImmutableDecimalMap = Field(default_factory=dict)
    factor_notional: ImmutableDecimalMap = Field(default_factory=dict)
    option_greeks: ImmutableDecimalMap = Field(default_factory=dict)
    proposed_incremental_exposure: ImmutableDecimalMap = Field(default_factory=dict)
    limit_results: tuple[ExposureLimitResult, ...] = ()
    quality: DataQualitySummary = Field(default_factory=DataQualitySummary)
    config_version: str
    content_hash: Sha256Hex

    def computed_content_hash(self) -> str:
        return content_hash(self, exclude=set(_EXPOSURE_HASH_EXCLUDE))

    @classmethod
    def create(cls, *, report_id: UUID, **fields: object) -> ExposureReport:
        """Build a report, deriving ``content_hash`` from its own content."""

        probe = cls.model_construct(
            report_id=report_id, content_hash="0" * 64, **fields
        )
        return cls(
            report_id=report_id,
            content_hash=content_hash(probe, exclude=set(_EXPOSURE_HASH_EXCLUDE)),
            **fields,
        )

    @model_validator(mode="after")
    def validate_exposure(self) -> ExposureReport:
        if self.long_notional + self.short_notional != self.gross_notional:
            raise ValueError("gross_notional must equal long plus short notional")
        if self.long_notional - self.short_notional != self.net_notional:
            raise ValueError("net_notional must equal long minus short notional")
        if self.content_hash != self.computed_content_hash():
            raise ValueError("content_hash does not match exposure content")
        return self

    @property
    def has_unknown_exposure(self) -> bool:
        return not self.quality.is_usable


class OwnershipRecord(ContractModel):
    """Attribution of one broker position component to one confirmed fill."""

    ownership_id: UUID
    account_alias: str
    broker_position_key: str
    strategy_id: str | None
    decision_record_id: UUID | None
    order_request_id: UUID | None
    source_fill_id: UUID | None
    quantity: FiniteDecimal
    effective_at: UtcDatetime
    ended_at: UtcDatetime | None = None
    ownership_status: OwnershipStatus

    @model_validator(mode="after")
    def validate_ownership(self) -> OwnershipRecord:
        if self.ended_at is not None and self.ended_at < self.effective_at:
            raise ValueError("ownership ended_at must not precede effective_at")
        if self.ownership_status is OwnershipStatus.UNASSIGNED:
            if self.strategy_id is not None:
                raise ValueError("UNASSIGNED ownership cannot name a strategy")
            if self.source_fill_id is not None:
                raise ValueError("UNASSIGNED ownership cannot cite a fill")
            return self
        # Attribution is only ever created from a broker-confirmed fill.
        if self.source_fill_id is None:
            raise ValueError("assigned ownership requires a source fill")
        if self.strategy_id is None:
            raise ValueError("assigned ownership requires a strategy")
        if self.decision_record_id is None or self.order_request_id is None:
            raise ValueError("assigned ownership requires decision and order lineage")
        return self


class ReconciliationLine(ContractModel):
    broker_position_key: str
    status: ReconciliationStatus
    broker_quantity: FiniteDecimal
    owned_quantity: FiniteDecimal
    strategy_ids: tuple[str, ...] = ()
    ownership_ids: tuple[UUID, ...] = ()


class PortfolioReconciliation(ContractModel):
    reconciliation_id: UUID
    portfolio_state_id: UUID
    observed_at: UtcDatetime
    matched: tuple[ReconciliationLine, ...] = ()
    partial: tuple[ReconciliationLine, ...] = ()
    unassigned: tuple[ReconciliationLine, ...] = ()
    orphaned: tuple[ReconciliationLine, ...] = ()
    quantity_mismatches: tuple[ReconciliationLine, ...] = ()
    ownership_adjustment_ids: tuple[UUID, ...] = ()
    content_hash: Sha256Hex

    def computed_content_hash(self) -> str:
        return content_hash(self, exclude=set(_RECONCILIATION_HASH_EXCLUDE))

    @classmethod
    def create(
        cls, *, reconciliation_id: UUID, **fields: object
    ) -> PortfolioReconciliation:
        """Build a reconciliation, deriving ``content_hash`` from its content."""

        probe = cls.model_construct(
            reconciliation_id=reconciliation_id, content_hash="0" * 64, **fields
        )
        return cls(
            reconciliation_id=reconciliation_id,
            content_hash=content_hash(probe, exclude=set(_RECONCILIATION_HASH_EXCLUDE)),
            **fields,
        )

    @model_validator(mode="after")
    def validate_reconciliation(self) -> PortfolioReconciliation:
        buckets = (
            (self.matched, ReconciliationStatus.MATCHED),
            (self.partial, ReconciliationStatus.PARTIAL),
            (self.unassigned, ReconciliationStatus.UNASSIGNED),
            (self.orphaned, ReconciliationStatus.ORPHANED_OWNERSHIP),
            (self.quantity_mismatches, ReconciliationStatus.QUANTITY_MISMATCH),
        )
        for lines, expected in buckets:
            for line in lines:
                if line.status is not expected:
                    raise ValueError(
                        f"{expected.value} bucket contains a {line.status.value} line"
                    )
        keys = [
            line.broker_position_key for lines, _ in buckets for line in lines
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("a broker position cannot appear in two buckets")
        if self.content_hash != self.computed_content_hash():
            raise ValueError("content_hash does not match reconciliation content")
        return self


__all__ = [
    "ExposureLimitResult",
    "ExposureReport",
    "ImmutableDecimalMap",
    "OwnershipRecord",
    "PortfolioReconciliation",
    "ReconciliationLine",
]
