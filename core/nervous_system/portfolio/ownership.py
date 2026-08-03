"""Fill-backed position ownership.

Ownership exists only because a broker confirmed a fill.  A submitted or
accepted order attributes nothing: until the broker says shares moved, the
system owns no position.
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal
from uuid import UUID, uuid5

from core.nervous_system.contracts.enums import (
    ExecutionStatus,
    OrderSide,
    OwnershipStatus,
)
from core.nervous_system.contracts.execution import ExecutionEvent
from core.nervous_system.contracts.orders import OrderRequest
from core.nervous_system.contracts.portfolio import OwnershipRecord


# uuid5(NAMESPACE_URL, "https://cynolycus.local/nervous-system/ownership@1")
OWNERSHIP_NAMESPACE = UUID("64bd8a4a-5b1b-5a0e-9f31-6e0b0f8b4a2d")

_FILL_STATUSES = frozenset(
    {ExecutionStatus.PARTIALLY_FILLED, ExecutionStatus.FILLED}
)
_ZERO = Decimal("0")


class OwnershipError(ValueError):
    """Raised when a fill cannot legitimately produce ownership."""


def broker_position_key(account_alias: str, symbol: str) -> str:
    """Identity of one broker position: account plus equity or OCC symbol."""

    if not account_alias or not symbol:
        raise OwnershipError("broker position key requires account and symbol")
    return f"{account_alias}:{symbol}"


def _order_symbol(order: OrderRequest) -> str:
    if order.equity_symbol is not None:
        return order.equity_symbol
    if order.legs:
        return order.legs[0].symbol
    if order.broker_position_key:
        return order.broker_position_key.split(":", 1)[-1]
    raise OwnershipError("order request carries no instrument symbol")


def _signed_side(order: OrderRequest) -> Decimal:
    if order.equity_side is not None:
        return Decimal("1") if order.equity_side is OrderSide.BUY else Decimal("-1")
    if order.legs:
        return Decimal("1") if order.legs[0].side is OrderSide.BUY else Decimal("-1")
    raise OwnershipError("order request carries no side")


def assign_fill_ownership(
    fill: ExecutionEvent,
    order_request: OrderRequest,
    *,
    strategy_id: str,
) -> OwnershipRecord:
    """Attribute one broker-confirmed fill to its originating decision.

    ``strategy_id`` is explicit because neither ``ExecutionEvent`` nor
    ``OrderRequest`` carries it: attribution lives on the originating
    ``TradeIntent``, reachable only through the decision record.  Requiring it
    keeps the caller honest instead of guessing a strategy here.
    """

    if not strategy_id or not strategy_id.strip():
        raise OwnershipError("ownership requires an explicit strategy_id")
    if fill.order_request_id != order_request.order_request_id:
        raise OwnershipError("execution event belongs to another order")
    if len(order_request.legs) > 1:
        # Each leg of a spread is its own broker position, so one record cannot
        # represent the fill.  Attributing the parent quantity to the first leg
        # would over-attribute that leg and leave the others unassigned.
        raise OwnershipError(
            "multi-leg orders require per-leg ownership records"
        )
    if fill.status not in _FILL_STATUSES or fill.filled_quantity <= _ZERO:
        raise OwnershipError(
            "ownership requires a broker-confirmed fill with positive quantity"
        )

    symbol = _order_symbol(order_request)
    key = order_request.broker_position_key or broker_position_key(
        order_request.account_alias, symbol
    )
    quantity = _signed_side(order_request) * fill.filled_quantity

    return OwnershipRecord(
        ownership_id=uuid5(
            OWNERSHIP_NAMESPACE,
            "|".join(
                (
                    str(order_request.order_request_id),
                    str(fill.execution_event_id),
                    key,
                )
            ),
        ),
        account_alias=order_request.account_alias,
        broker_position_key=key,
        strategy_id=strategy_id,
        decision_record_id=order_request.decision_id,
        order_request_id=order_request.order_request_id,
        source_fill_id=fill.execution_event_id,
        quantity=quantity,
        effective_at=fill.observed_at,
        ended_at=None,
        ownership_status=OwnershipStatus.ASSIGNED,
    )


def net_owned_quantity(records: Iterable[OwnershipRecord]) -> Decimal:
    """Net attributed quantity, clamped so it can never flip direction.

    The earliest fill establishes the direction of the position.  Later fills
    reduce it toward zero, but over-closing is a reconciliation signal rather
    than a reversal: a long can never become negative, and a short position's
    genuine negative ownership is preserved rather than floored away.
    """

    assigned = [
        record
        for record in records
        if record.ownership_status is OwnershipStatus.ASSIGNED
    ]
    if not assigned:
        return _ZERO
    total = sum((record.quantity for record in assigned), _ZERO)
    opening = min(
        assigned, key=lambda record: (record.effective_at, str(record.ownership_id))
    )
    if opening.quantity >= _ZERO:
        return total if total > _ZERO else _ZERO
    return total if total < _ZERO else _ZERO


__all__ = [
    "OWNERSHIP_NAMESPACE",
    "OwnershipError",
    "assign_fill_ownership",
    "broker_position_key",
    "net_owned_quantity",
]
