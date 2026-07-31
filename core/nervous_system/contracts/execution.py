from __future__ import annotations

from typing import Annotated
from uuid import UUID, uuid4

from pydantic import AfterValidator, Field, model_validator

from .base import (
    ContractModel,
    NonNegativeDecimal,
    Sha256Hex,
    UtcDatetime,
    _freeze_mapping,
    content_hash,
)
from .enums import ExecutionStatus


ImmutableStringMap = Annotated[dict[str, str], AfterValidator(_freeze_mapping)]
ImmutableObjectMap = Annotated[dict[str, object], AfterValidator(_freeze_mapping)]
PositiveSequence = Annotated[int, Field(gt=0)]


class _ExecutionEventHashMaterial(ContractModel):
    order_request_id: UUID
    event_type: str
    status: ExecutionStatus
    observed_at: UtcDatetime
    broker_event_at: UtcDatetime | None
    client_order_id: str
    broker_order_id: str | None
    broker_parent_order_id: str | None
    filled_quantity: NonNegativeDecimal
    average_fill_price: NonNegativeDecimal | None
    sequence_no: PositiveSequence
    previous_event_id: UUID | None
    leg_reports: tuple[ImmutableStringMap, ...]
    sanitized_response: ImmutableObjectMap
    previous_event_hash: Sha256Hex | None


class ExecutionEvent(ContractModel):
    execution_event_id: UUID
    order_request_id: UUID
    event_type: str
    status: ExecutionStatus
    observed_at: UtcDatetime
    broker_event_at: UtcDatetime | None
    client_order_id: str
    broker_order_id: str | None
    broker_parent_order_id: str | None
    filled_quantity: NonNegativeDecimal
    average_fill_price: NonNegativeDecimal | None
    sequence_no: PositiveSequence
    previous_event_id: UUID | None
    leg_reports: tuple[ImmutableStringMap, ...]
    sanitized_response: ImmutableObjectMap
    previous_event_hash: Sha256Hex | None
    event_hash: Sha256Hex

    def event_hash_material(self) -> _ExecutionEventHashMaterial:
        return _ExecutionEventHashMaterial(
            order_request_id=self.order_request_id,
            event_type=self.event_type,
            status=self.status,
            observed_at=self.observed_at,
            broker_event_at=self.broker_event_at,
            client_order_id=self.client_order_id,
            broker_order_id=self.broker_order_id,
            broker_parent_order_id=self.broker_parent_order_id,
            filled_quantity=self.filled_quantity,
            average_fill_price=self.average_fill_price,
            sequence_no=self.sequence_no,
            previous_event_id=self.previous_event_id,
            leg_reports=self.leg_reports,
            sanitized_response=self.sanitized_response,
            previous_event_hash=self.previous_event_hash,
        )

    def computed_event_hash(self) -> str:
        return content_hash(self.event_hash_material())

    @classmethod
    def create(
        cls,
        *,
        order_request_id: UUID,
        event_type: str,
        sequence_no: PositiveSequence,
        status: ExecutionStatus,
        observed_at: UtcDatetime,
        broker_event_at: UtcDatetime | None,
        client_order_id: str,
        broker_order_id: str | None,
        broker_parent_order_id: str | None,
        filled_quantity: NonNegativeDecimal,
        average_fill_price: NonNegativeDecimal | None,
        leg_reports: tuple[dict[str, str], ...],
        sanitized_response: dict[str, object],
        previous_event_id: UUID | None,
        previous_event_hash: Sha256Hex | None,
        execution_event_id: UUID | None = None,
    ) -> ExecutionEvent:
        material = _ExecutionEventHashMaterial(
            order_request_id=order_request_id,
            event_type=event_type,
            status=status,
            observed_at=observed_at,
            broker_event_at=broker_event_at,
            client_order_id=client_order_id,
            broker_order_id=broker_order_id,
            broker_parent_order_id=broker_parent_order_id,
            filled_quantity=filled_quantity,
            average_fill_price=average_fill_price,
            sequence_no=sequence_no,
            previous_event_id=previous_event_id,
            leg_reports=leg_reports,
            sanitized_response=sanitized_response,
            previous_event_hash=previous_event_hash,
        )
        return cls(
            execution_event_id=execution_event_id or uuid4(),
            order_request_id=order_request_id,
            event_type=event_type,
            status=status,
            observed_at=observed_at,
            broker_event_at=broker_event_at,
            client_order_id=client_order_id,
            broker_order_id=broker_order_id,
            broker_parent_order_id=broker_parent_order_id,
            filled_quantity=filled_quantity,
            average_fill_price=average_fill_price,
            sequence_no=sequence_no,
            previous_event_id=previous_event_id,
            leg_reports=leg_reports,
            sanitized_response=sanitized_response,
            previous_event_hash=previous_event_hash,
            event_hash=content_hash(material),
        )

    from_fields = create

    @model_validator(mode="after")
    def validate_fill_fields(self) -> ExecutionEvent:
        if self.sequence_no == 1 and (
            self.previous_event_id is not None or self.previous_event_hash is not None
        ):
            raise ValueError("sequence 1 execution events cannot have a predecessor")
        if self.sequence_no > 1 and (
            self.previous_event_id is None or self.previous_event_hash is None
        ):
            raise ValueError("execution events after sequence 1 require a predecessor")
        if self.broker_event_at is not None and self.broker_event_at > self.observed_at:
            raise ValueError("broker_event_at must not be after observed_at")
        if self.filled_quantity > 0 and self.average_fill_price is None:
            raise ValueError("filled execution events require average_fill_price")
        if self.event_hash != self.computed_event_hash():
            raise ValueError("event_hash does not match execution event content")
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
        prior_id: UUID | None = None
        prior_sequence: int | None = None
        prior_time = None
        client_order_id = self.events[0].client_order_id
        broker_order_id: str | None = None
        broker_parent_order_id: str | None = None
        for event in self.events:
            if event.order_request_id != self.order_request_id:
                raise ValueError("execution event belongs to another order")
            if event.previous_event_hash != prior_hash:
                raise ValueError("execution event hashes must chain in tuple order")
            if event.previous_event_id != prior_id:
                raise ValueError("execution event predecessor IDs must chain in tuple order")
            expected_sequence = 1 if prior_sequence is None else prior_sequence + 1
            if event.sequence_no != expected_sequence:
                raise ValueError("execution event sequence numbers must be contiguous")
            if prior_time is not None and event.observed_at < prior_time:
                raise ValueError("execution events must be ordered by observed_at")
            if event.client_order_id != client_order_id:
                raise ValueError("execution client_order_id must remain consistent")
            if broker_order_id is None:
                if event.broker_order_id is not None:
                    broker_order_id = event.broker_order_id
            elif event.broker_order_id != broker_order_id:
                raise ValueError("execution broker_order_id must remain consistent after acknowledgement")
            if broker_parent_order_id is None:
                if event.broker_parent_order_id is not None:
                    broker_parent_order_id = event.broker_parent_order_id
            elif event.broker_parent_order_id != broker_parent_order_id:
                raise ValueError("execution broker_parent_order_id must remain consistent after acknowledgement")
            prior_hash = event.event_hash
            prior_id = event.execution_event_id
            prior_sequence = event.sequence_no
            prior_time = event.observed_at
        return self
