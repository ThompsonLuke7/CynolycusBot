"""Broker-facing contracts, typed errors, and the adapter protocol.

The broker is authoritative. Nothing here invents a terminal state, and a
submission response is never treated as a fill: only a broker-confirmed fill
creates ownership.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Annotated, Protocol, runtime_checkable

from pydantic import AfterValidator, Field, model_validator

from core.nervous_system.contracts.base import (
    ContractModel,
    FiniteDecimal,
    NonNegativeDecimal,
    UtcDatetime,
    _freeze_mapping,
)
from core.nervous_system.contracts.enums import AssetClass, ExecutionStatus, OrderSide
from core.nervous_system.contracts.orders import OrderRequest


ImmutableObjectMap = Annotated[dict[str, object], AfterValidator(_freeze_mapping)]


class BrokerError(Exception):
    """Base class for every broker failure."""


class BrokerRejected(BrokerError):
    """The broker understood the request and refused it."""


class BrokerUnavailable(BrokerError):
    """The broker could not be reached; the request certainly did not land."""


class BrokerAmbiguousSubmission(BrokerError):
    """The outcome is unknown and the order may or may not exist.

    Never retry blindly on this. Reconcile by client order ID first.
    """


class BrokerAuthenticationError(BrokerError):
    """Credentials or account permissions were refused."""


class BrokerContractError(BrokerError):
    """The broker payload did not match the shape this adapter requires."""


# Alpaca order status -> internal execution status.  Anything unmapped becomes
# UNKNOWN with the raw status preserved, never a guessed terminal state.
ALPACA_STATUS_MAP: Mapping[str, ExecutionStatus] = {
    "new": ExecutionStatus.ACCEPTED,
    "accepted": ExecutionStatus.ACCEPTED,
    "pending_new": ExecutionStatus.ACCEPTED,
    "accepted_for_bidding": ExecutionStatus.ACCEPTED,
    "held": ExecutionStatus.ACCEPTED,
    "calculated": ExecutionStatus.ACCEPTED,
    "partially_filled": ExecutionStatus.PARTIALLY_FILLED,
    "filled": ExecutionStatus.FILLED,
    "done_for_day": ExecutionStatus.EXPIRED,
    "expired": ExecutionStatus.EXPIRED,
    "rejected": ExecutionStatus.REJECTED,
    "canceled": ExecutionStatus.CANCELED,
    "cancelled": ExecutionStatus.CANCELED,
    # A replaced order is terminal in its own right: the replacement carries a
    # new identity, so the original stops working.
    "replaced": ExecutionStatus.CANCELED,
    "pending_cancel": ExecutionStatus.ACCEPTED,
    "pending_replace": ExecutionStatus.ACCEPTED,
    "stopped": ExecutionStatus.ACCEPTED,
    "suspended": ExecutionStatus.ACCEPTED,
}


class BrokerAccount(ContractModel):
    account_id: str
    account_alias: str
    status: str
    equity: FiniteDecimal
    cash: FiniteDecimal
    buying_power: FiniteDecimal
    observed_at: UtcDatetime
    raw: ImmutableObjectMap = Field(default_factory=dict)


class BrokerPosition(ContractModel):
    symbol: str
    asset_class: AssetClass
    quantity: FiniteDecimal
    average_entry_price: FiniteDecimal | None = None
    market_value: FiniteDecimal | None = None
    observed_at: UtcDatetime
    raw: ImmutableObjectMap = Field(default_factory=dict)


class BrokerOrderLeg(ContractModel):
    symbol: str
    ratio_quantity: int
    side: OrderSide
    position_intent: str | None = None
    raw_status: str
    filled_quantity: NonNegativeDecimal = Decimal("0")
    average_fill_price: NonNegativeDecimal | None = None
    broker_order_id: str | None = None


class BrokerOrder(ContractModel):
    broker_order_id: str
    client_order_id: str
    status: ExecutionStatus
    raw_status: str
    submitted_at: UtcDatetime | None = None
    updated_at: UtcDatetime | None = None
    filled_at: UtcDatetime | None = None
    filled_quantity: NonNegativeDecimal = Decimal("0")
    average_fill_price: NonNegativeDecimal | None = None
    legs: tuple[BrokerOrderLeg, ...] = ()
    observed_at: UtcDatetime
    raw: ImmutableObjectMap = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_order(self) -> BrokerOrder:
        if self.filled_quantity > 0 and self.average_fill_price is None:
            raise ValueError("a filled order requires an average fill price")
        if self.status is ExecutionStatus.FILLED and self.filled_quantity <= 0:
            raise ValueError("a FILLED order requires a positive filled quantity")
        return self

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            ExecutionStatus.FILLED,
            ExecutionStatus.REJECTED,
            ExecutionStatus.CANCELED,
            ExecutionStatus.EXPIRED,
        }


class OrderReplacement(ContractModel):
    """The fields the broker permits on a PATCH.

    A structural change is a new linked order request, never a replacement.
    """

    quantity: Decimal | None = None
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: str | None = None
    client_order_id: str | None = None

    @model_validator(mode="after")
    def validate_replacement(self) -> OrderReplacement:
        if not any(
            value is not None
            for value in (
                self.quantity,
                self.limit_price,
                self.stop_price,
                self.time_in_force,
                self.client_order_id,
            )
        ):
            raise ValueError("a replacement must change at least one field")
        if self.quantity is not None and self.quantity <= 0:
            raise ValueError("replacement quantity must be positive")
        return self

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.quantity is not None:
            payload["qty"] = str(self.quantity)
        if self.limit_price is not None:
            payload["limit_price"] = str(self.limit_price)
        if self.stop_price is not None:
            payload["stop_price"] = str(self.stop_price)
        if self.time_in_force is not None:
            payload["time_in_force"] = self.time_in_force
        if self.client_order_id is not None:
            payload["client_order_id"] = self.client_order_id
        return payload


@runtime_checkable
class BrokerAdapter(Protocol):
    """The inward-facing broker surface the gateway depends on."""

    def account(self) -> BrokerAccount: ...

    def positions(self) -> tuple[BrokerPosition, ...]: ...

    def orders(self, *, status: str = "all") -> tuple[BrokerOrder, ...]: ...

    def find_by_client_order_id(self, client_order_id: str) -> BrokerOrder | None: ...

    def submit(self, request: OrderRequest) -> BrokerOrder: ...

    def cancel(self, broker_order_id: str) -> BrokerOrder: ...

    def replace(
        self,
        broker_order_id: str,
        replacement: OrderReplacement,
    ) -> BrokerOrder: ...


__all__ = [
    "ALPACA_STATUS_MAP",
    "BrokerAccount",
    "BrokerAdapter",
    "BrokerAmbiguousSubmission",
    "BrokerAuthenticationError",
    "BrokerContractError",
    "BrokerError",
    "BrokerOrder",
    "BrokerOrderLeg",
    "BrokerPosition",
    "BrokerRejected",
    "BrokerUnavailable",
    "OrderReplacement",
]
