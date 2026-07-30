from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import AfterValidator, Field, model_validator

from .base import ContractModel, NonNegativeDecimal, Sha256Hex, UtcDatetime, _freeze_mapping
from .enums import ExecutionStatus


ImmutableStringMap = Annotated[dict[str, str], AfterValidator(_freeze_mapping)]
ImmutableObjectMap = Annotated[dict[str, object], AfterValidator(_freeze_mapping)]


class ExecutionEvent(ContractModel):
    execution_event_id: UUID
    order_request_id: UUID
    status: ExecutionStatus
    observed_at: UtcDatetime
    broker_event_at: UtcDatetime | None
    client_order_id: str
    broker_order_id: str | None
    broker_parent_order_id: str | None
    filled_quantity: NonNegativeDecimal
    average_fill_price: NonNegativeDecimal | None
    leg_reports: tuple[ImmutableStringMap, ...]
    sanitized_response: ImmutableObjectMap
    previous_event_hash: Sha256Hex | None
    event_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_fill_fields(self) -> ExecutionEvent:
        if self.filled_quantity > 0 and self.average_fill_price is None:
            raise ValueError("filled execution events require average_fill_price")
        return self


class ExecutionReport(ContractModel):
    order_request_id: UUID
    events: tuple[ExecutionEvent, ...] = Field(min_length=1)
    current_status: ExecutionStatus

    @model_validator(mode="after")
    def validate_event_chain(self) -> ExecutionReport:
        if self.events[-1].status is not self.current_status:
            raise ValueError("current_status must equal the final execution event status")
        prior_hash: str | None = None
        prior_time = None
        for event in self.events:
            if event.order_request_id != self.order_request_id:
                raise ValueError("execution event belongs to another order")
            if event.previous_event_hash != prior_hash:
                raise ValueError("execution event hashes must chain in tuple order")
            if prior_time is not None and event.observed_at < prior_time:
                raise ValueError("execution events must be ordered by observed_at")
            prior_hash = event.event_hash
            prior_time = event.observed_at
        return self
