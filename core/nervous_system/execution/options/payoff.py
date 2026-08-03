"""Exact expiry payoff, finite loss bounds, and collateral.

All money is ``Decimal``.  The expiry payoff of a same-expiry option structure
is piecewise linear with kinks only at strikes, so every finite extremum lies
at zero, at a strike, or in the unbounded tail.  The upper tail slope is
derived analytically rather than sampled.

A structure whose loss tail is unbounded is not "very risky", it is invalid:
it returns an invalid profile with ``UNBOUNDED_MAX_LOSS`` and nothing
downstream may size it.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from enum import Enum

from pydantic import model_validator

from core.nervous_system.contracts.base import (
    ContractModel,
    FiniteDecimal,
    NonNegativeDecimal,
)
from core.nervous_system.contracts.enums import AssetClass, OptionType, OrderSide
from core.nervous_system.contracts.orders import OptionLeg
from core.nervous_system.contracts.states import PortfolioPosition


_ZERO = Decimal("0")
_ONE = Decimal("1")
DEFAULT_CONTRACT_MULTIPLIER = 100


class RiskReason(str, Enum):
    """Stable reason codes for a rejected or degraded structure."""

    UNBOUNDED_MAX_LOSS = "UNBOUNDED_MAX_LOSS"
    NAKED_SHORT_CALL = "NAKED_SHORT_CALL"
    NAKED_SHORT_PUT = "NAKED_SHORT_PUT"
    UNCOVERED_RATIO = "UNCOVERED_RATIO"
    INSUFFICIENT_SHARE_COVERAGE = "INSUFFICIENT_SHARE_COVERAGE"
    INSUFFICIENT_CASH_COLLATERAL = "INSUFFICIENT_CASH_COLLATERAL"
    SHORT_CALENDAR_REJECTED = "SHORT_CALENDAR_REJECTED"
    CALENDAR_QUANTITY_MISMATCH = "CALENDAR_QUANTITY_MISMATCH"
    CALENDAR_RIGHT_MISMATCH = "CALENDAR_RIGHT_MISMATCH"
    MIXED_UNDERLYING = "MIXED_UNDERLYING"
    EMPTY_STRUCTURE = "EMPTY_STRUCTURE"
    TOO_MANY_LEGS = "TOO_MANY_LEGS"
    SHARE_COST_BASIS_UNKNOWN = "SHARE_COST_BASIS_UNKNOWN"


class OptionRiskProfile(ContractModel):
    """Deterministic, finite risk description of one option structure."""

    valid: bool
    # ``None`` means the profit tail is unbounded, which is never a rejection.
    max_profit: FiniteDecimal | None
    max_loss: NonNegativeDecimal
    breakevens: tuple[FiniteDecimal, ...]
    net_debit: FiniteDecimal
    collateral: NonNegativeDecimal
    assignment_exposure: NonNegativeDecimal
    contract_multiplier: int
    quantity: int
    reason_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_profile(self) -> OptionRiskProfile:
        if self.valid and self.reason_codes:
            raise ValueError("a valid risk profile cannot carry rejection reasons")
        if not self.valid and not self.reason_codes:
            raise ValueError("an invalid risk profile requires a reason code")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be positive")
        return self

    @property
    def has_unbounded_profit(self) -> bool:
        return self.max_profit is None


def _signed_ratio(leg: OptionLeg) -> Decimal:
    sign = _ONE if leg.side is OrderSide.BUY else -_ONE
    return sign * Decimal(leg.ratio)


def _intrinsic(leg: OptionLeg, underlying_price: Decimal) -> Decimal:
    if leg.option_type is OptionType.CALL:
        return max(underlying_price - leg.strike, _ZERO)
    return max(leg.strike - underlying_price, _ZERO)


def payoff_per_share(
    legs: Sequence[OptionLeg],
    underlying_price: Decimal,
    net_debit: Decimal,
) -> Decimal:
    """Per-share expiry payoff, before the contract multiplier."""

    total = sum(
        (_signed_ratio(leg) * _intrinsic(leg, underlying_price) for leg in legs),
        _ZERO,
    )
    return total - net_debit


def expiry_payoff(
    legs: Sequence[OptionLeg],
    *,
    underlying_price: Decimal,
    net_debit: Decimal,
    contract_multiplier: int = DEFAULT_CONTRACT_MULTIPLIER,
) -> Decimal:
    """Exact expiry payoff in dollars for one structure.

    ``net_debit`` is the per-share net price: positive when premium is paid,
    negative when premium is received.
    """

    if not legs:
        raise ValueError("payoff requires at least one leg")
    if contract_multiplier <= 0:
        raise ValueError("contract_multiplier must be positive")
    if len({leg.expiration for leg in legs}) != 1:
        raise ValueError(
            "expiry payoff is only defined for a single expiration; "
            "calendars and diagonals need a separate bound"
        )
    return payoff_per_share(legs, underlying_price, net_debit) * Decimal(
        contract_multiplier
    )


def upper_tail_slope(legs: Sequence[OptionLeg]) -> Decimal:
    """Payoff slope per $1 of underlying above every strike.

    Above all strikes every call is in the money with slope 1 and every put is
    worthless with slope 0, so the tail behaviour is exact, not sampled.
    """

    return sum(
        (
            _signed_ratio(leg)
            for leg in legs
            if leg.option_type is OptionType.CALL
        ),
        _ZERO,
    )


def _evaluation_points(legs: Sequence[OptionLeg]) -> tuple[Decimal, ...]:
    strikes = sorted({leg.strike for leg in legs})
    return tuple([_ZERO, *strikes])


def _breakevens(
    legs: Sequence[OptionLeg],
    net_debit: Decimal,
    points: Sequence[Decimal],
    slope: Decimal,
    stock_contracts: Decimal = _ZERO,
    stock_cost_basis: Decimal = _ZERO,
) -> tuple[Decimal, ...]:
    found: list[Decimal] = []
    values = [
        payoff_per_share(legs, point, net_debit)
        + stock_contracts * (point - stock_cost_basis)
        for point in points
    ]

    for index, (point, value) in enumerate(zip(points, values)):
        if value == _ZERO:
            found.append(point)
            continue
        if index + 1 >= len(points):
            continue
        next_point, next_value = points[index + 1], values[index + 1]
        if (value < _ZERO) != (next_value < _ZERO) and next_value != _ZERO:
            # Linear between kinks, so the crossing is exact.
            crossing = point + (next_point - point) * (-value) / (next_value - value)
            found.append(crossing)

    if slope != _ZERO and values:
        last_point, last_value = points[-1], values[-1]
        if (last_value < _ZERO) != (slope < _ZERO) and last_value != _ZERO:
            found.append(last_point + (-last_value) / slope)

    return tuple(sorted(set(found)))


def same_expiry_risk_profile(
    legs: Sequence[OptionLeg],
    *,
    net_debit: Decimal,
    contract_multiplier: int = DEFAULT_CONTRACT_MULTIPLIER,
    quantity: int = 1,
    stock_contracts: Decimal = _ZERO,
    stock_cost_basis: Decimal = _ZERO,
) -> tuple[Decimal | None, Decimal, tuple[Decimal, ...], bool]:
    """Return ``(max_profit, max_loss, breakevens, bounded)`` in dollars.

    ``max_profit`` is ``None`` when the upside tail is unbounded.  ``bounded``
    is False when the loss tail is unbounded, in which case ``max_loss`` is
    meaningless and the caller must reject.

    ``stock_contracts`` folds verified long shares (in contract equivalents)
    into the payoff.  A covered call's option leg alone has an unbounded upper
    tail; it is the shares that bound it, so a share-backed structure must be
    evaluated as the combined position or it would be wrongly rejected.
    """

    points = _evaluation_points(legs)
    slope = upper_tail_slope(legs) + stock_contracts
    scale = Decimal(contract_multiplier) * Decimal(quantity)

    def total_per_share(price: Decimal) -> Decimal:
        return payoff_per_share(legs, price, net_debit) + stock_contracts * (
            price - stock_cost_basis
        )

    values = [total_per_share(point) * scale for point in points]

    if slope < _ZERO:
        return None, _ZERO, (), False

    worst = min(values)
    max_loss = -worst if worst < _ZERO else _ZERO
    max_profit = None if slope > _ZERO else max(values)
    breakevens = _breakevens(
        legs, net_debit, points, slope, stock_contracts, stock_cost_basis
    )
    return max_profit, max_loss, breakevens, True


def _long_equity(
    existing_holdings: Sequence[PortfolioPosition],
    underlying: str,
) -> tuple[PortfolioPosition, ...]:
    return tuple(
        position
        for position in existing_holdings
        if position.asset_class is AssetClass.EQUITY
        and position.symbol == underlying
        and position.quantity > 0
    )


def _share_coverage(
    existing_holdings: Sequence[PortfolioPosition],
    underlying: str,
    contract_multiplier: int,
) -> Decimal:
    """Long shares available to cover short calls, in contract equivalents."""

    shares = sum(
        (
            Decimal(str(position.quantity))
            for position in _long_equity(existing_holdings, underlying)
        ),
        _ZERO,
    )
    return shares / Decimal(contract_multiplier)


def share_cost_basis(
    existing_holdings: Sequence[PortfolioPosition],
    underlying: str,
) -> Decimal | None:
    """Quantity-weighted entry price of the covering shares.

    Returns ``None`` when any covering lot has no entry price: an unknown cost
    basis means the combined loss cannot be bounded, and guessing one would
    invent the very number the bound exists to establish.
    """

    positions = _long_equity(existing_holdings, underlying)
    if not positions:
        return None
    total_shares = _ZERO
    total_cost = _ZERO
    for position in positions:
        if position.average_entry_price is None:
            return None
        shares = Decimal(str(position.quantity))
        total_shares += shares
        total_cost += shares * Decimal(str(position.average_entry_price))
    if total_shares <= _ZERO:
        return None
    return total_cost / total_shares


__all__ = [
    "DEFAULT_CONTRACT_MULTIPLIER",
    "OptionRiskProfile",
    "RiskReason",
    "expiry_payoff",
    "payoff_per_share",
    "same_expiry_risk_profile",
    "upper_tail_slope",
]
