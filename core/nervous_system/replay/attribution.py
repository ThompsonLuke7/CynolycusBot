"""Split a realized result into parts that sum back to it exactly.

Pure: no clock, no IO. Every boundary — the reference prices, the underlying
levels, the target time — is passed in, because an attribution that reads a
clock cannot be reproduced from its own record.

Two rules shape the whole module:

* Broker fills are authoritative. Realized P&L uses confirmed filled quantity
  only; treating requested quantity as filled would invent performance.
* The target window is fixed. No observation after it may enter the outcome,
  and a position still open when the window closes is PENDING rather than
  zero — a zero reads as a real flat result.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from decimal import Decimal

from core.nervous_system.contracts.enums import OrderSide
from core.nervous_system.contracts.replay import (
    AttributionStatus,
    FillFact,
    OutcomeAttribution,
)


_ZERO = Decimal("0")


def _aware(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _within_window(
    fills: Sequence[FillFact], target: datetime
) -> tuple[tuple[FillFact, ...], int]:
    """Keep only fills at or before the target; count what was left out."""

    kept = tuple(fill for fill in fills if fill.filled_at <= target)
    return kept, len(fills) - len(kept)


def _quantity(fills: Sequence[FillFact], side: OrderSide) -> Decimal:
    return sum(
        (fill.quantity for fill in fills if fill.side is side), start=_ZERO
    )


def _exposure_quantity(fills: Sequence[FillFact]) -> Decimal:
    """Net opened quantity, so a spread's legs do not double-count."""

    opened = _quantity(fills, OrderSide.BUY)
    written = _quantity(fills, OrderSide.SELL)
    return opened if opened >= written else written


def attribute_outcome(
    *,
    entry_fills: Sequence[FillFact],
    exit_fills: Sequence[FillFact],
    entry_reference_price: Decimal,
    exit_reference_price: Decimal,
    underlying_entry: Decimal,
    underlying_exit: Decimal,
    underlying_exposure: Decimal,
    target_time: datetime,
) -> OutcomeAttribution:
    """Attribute one round trip, or report that it has not settled yet."""

    target = _aware(target_time, "target_time")
    entries, excluded_entries = _within_window(entry_fills, target)
    exits, excluded_exits = _within_window(exit_fills, target)
    excluded = excluded_entries + excluded_exits

    if entries and exits:
        latest_entry = max(fill.filled_at for fill in entries)
        earliest_exit = min(fill.filled_at for fill in exits)
        if earliest_exit < latest_entry:
            raise ValueError("an exit fill cannot settle before its entry fill")

    entry_quantity = _exposure_quantity(entries)
    exit_quantity = _exposure_quantity(exits)

    # Not opened, not closed, or only half closed: there is no result to report.
    # Reporting zero here would book a flat outcome for a position we still hold.
    if not entries or not exits or exit_quantity != entry_quantity:
        return OutcomeAttribution(
            status=AttributionStatus.PENDING,
            filled_entry_quantity=entry_quantity,
            filled_exit_quantity=exit_quantity,
            excluded_fill_count=excluded,
        )

    fees = sum((fill.fees for fill in entries + exits), start=_ZERO)
    cash = sum((fill.signed_cash for fill in entries + exits), start=_ZERO)
    realized = cash - fees

    # What the underlying did, at the exposure the position actually carried.
    underlying_movement = underlying_exposure * (underlying_exit - underlying_entry)

    # What we gave up between the price the decision assumed and the price we
    # got. Buying above the reference and selling below it are both costs.
    slippage = _ZERO
    for fill in entries:
        difference = entry_reference_price - fill.price
        signed = difference if fill.side is OrderSide.BUY else -difference
        slippage += signed * fill.quantity * fill.contract_multiplier
    for fill in exits:
        difference = fill.price - exit_reference_price
        signed = difference if fill.side is OrderSide.SELL else -difference
        slippage += signed * fill.quantity * fill.contract_multiplier

    # Everything the underlying-equivalent exposure and execution do not
    # explain is the cost of having traded a derivative rather than the
    # underlying. It is a residual by construction, and labelled as one, so the
    # components sum to the realized result exactly.
    transformation = realized - underlying_movement - slippage + fees

    return OutcomeAttribution(
        status=AttributionStatus.FINAL,
        realized_pnl=realized,
        underlying_movement=underlying_movement,
        slippage=slippage,
        instrument_transformation=transformation,
        # Signed contributions: a cost reads as negative so the reader does not
        # have to hold a sign convention in their head.
        fees=-fees,
        filled_entry_quantity=entry_quantity,
        filled_exit_quantity=exit_quantity,
        excluded_fill_count=excluded,
    )


__all__ = ["attribute_outcome"]
