from __future__ import annotations

from datetime import date
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from .base import (
    ContractModel,
    NonNegativeDecimal,
    PositiveDecimal,
    Sha256Hex,
    UtcDatetime,
    content_hash,
)
from .enums import DebitCredit, InstrumentFamily, OptionType, OrderSide, PositionIntent, RuntimeEnvironment


class OptionLeg(ContractModel):
    symbol: str
    underlying: str
    option_type: OptionType
    strike: PositiveDecimal
    expiration: date
    side: OrderSide
    ratio: Annotated[int, Field(gt=0)]
    position_intent: PositionIntent
    quote_at: UtcDatetime
    bid: NonNegativeDecimal
    ask: NonNegativeDecimal

    @model_validator(mode="after")
    def validate_quote_and_intent(self) -> OptionLeg:
        if self.ask < self.bid:
            raise ValueError("option ask must not be below bid")
        if self.expiration < self.quote_at.date():
            raise ValueError("option expiration must not precede quote_at")
        expected_side = {
            PositionIntent.BUY_TO_OPEN: OrderSide.BUY,
            PositionIntent.BUY_TO_CLOSE: OrderSide.BUY,
            PositionIntent.SELL_TO_OPEN: OrderSide.SELL,
            PositionIntent.SELL_TO_CLOSE: OrderSide.SELL,
        }[self.position_intent]
        if self.side is not expected_side:
            raise ValueError("option side must agree with position_intent")
        return self


class _OrderRequestHashMaterial(ContractModel):
    decision_id: UUID
    policy_decision_id: UUID
    environment: RuntimeEnvironment
    account_alias: str
    instrument_family: InstrumentFamily
    equity_symbol: str | None
    equity_side: OrderSide | None
    legs: tuple[OptionLeg, ...]
    parent_quantity: PositiveDecimal
    debit_credit: DebitCredit
    net_limit_price: PositiveDecimal | None
    maximum_loss: NonNegativeDecimal
    buying_power_required: NonNegativeDecimal
    time_in_force: str
    order_type: Literal["limit", "market"]
    idempotency_key: str
    quote_snapshot_id: UUID | None
    supersedes_order_request_id: UUID | None
    created_at: UtcDatetime
    expires_at: UtcDatetime


class OrderRequest(ContractModel):
    order_request_id: UUID
    decision_id: UUID
    policy_decision_id: UUID
    environment: RuntimeEnvironment
    account_alias: str
    instrument_family: InstrumentFamily
    equity_symbol: str | None = None
    equity_side: OrderSide | None = None
    legs: tuple[OptionLeg, ...] = ()
    parent_quantity: PositiveDecimal
    debit_credit: DebitCredit
    net_limit_price: PositiveDecimal | None
    maximum_loss: NonNegativeDecimal
    buying_power_required: NonNegativeDecimal
    time_in_force: str
    order_type: Literal["limit", "market"]
    idempotency_key: str
    request_hash: Sha256Hex
    quote_snapshot_id: UUID | None = None
    supersedes_order_request_id: UUID | None = None
    created_at: UtcDatetime
    expires_at: UtcDatetime

    def request_hash_material(self) -> _OrderRequestHashMaterial:
        return _OrderRequestHashMaterial(
            decision_id=self.decision_id,
            policy_decision_id=self.policy_decision_id,
            environment=self.environment,
            account_alias=self.account_alias,
            instrument_family=self.instrument_family,
            equity_symbol=self.equity_symbol,
            equity_side=self.equity_side,
            legs=self.legs,
            parent_quantity=self.parent_quantity,
            debit_credit=self.debit_credit,
            net_limit_price=self.net_limit_price,
            maximum_loss=self.maximum_loss,
            buying_power_required=self.buying_power_required,
            time_in_force=self.time_in_force,
            order_type=self.order_type,
            idempotency_key=self.idempotency_key,
            quote_snapshot_id=self.quote_snapshot_id,
            supersedes_order_request_id=self.supersedes_order_request_id,
            created_at=self.created_at,
            expires_at=self.expires_at,
        )

    def computed_request_hash(self) -> str:
        return content_hash(self.request_hash_material())

    @classmethod
    def create(
        cls,
        *,
        decision_id: UUID,
        policy_decision_id: UUID,
        environment: RuntimeEnvironment,
        account_alias: str,
        instrument_family: InstrumentFamily,
        parent_quantity: PositiveDecimal,
        debit_credit: DebitCredit,
        net_limit_price: PositiveDecimal | None,
        maximum_loss: NonNegativeDecimal,
        buying_power_required: NonNegativeDecimal,
        time_in_force: str,
        order_type: Literal["limit", "market"],
        idempotency_key: str,
        created_at: UtcDatetime,
        expires_at: UtcDatetime,
        order_request_id: UUID | None = None,
        equity_symbol: str | None = None,
        equity_side: OrderSide | None = None,
        legs: tuple[OptionLeg, ...] = (),
        quote_snapshot_id: UUID | None = None,
        supersedes_order_request_id: UUID | None = None,
    ) -> OrderRequest:
        material = _OrderRequestHashMaterial(
            decision_id=decision_id,
            policy_decision_id=policy_decision_id,
            environment=environment,
            account_alias=account_alias,
            instrument_family=instrument_family,
            equity_symbol=equity_symbol,
            equity_side=equity_side,
            legs=legs,
            parent_quantity=parent_quantity,
            debit_credit=debit_credit,
            net_limit_price=net_limit_price,
            maximum_loss=maximum_loss,
            buying_power_required=buying_power_required,
            time_in_force=time_in_force,
            order_type=order_type,
            idempotency_key=idempotency_key,
            quote_snapshot_id=quote_snapshot_id,
            supersedes_order_request_id=supersedes_order_request_id,
            created_at=created_at,
            expires_at=expires_at,
        )
        return cls(
            order_request_id=order_request_id or uuid4(),
            decision_id=decision_id,
            policy_decision_id=policy_decision_id,
            environment=environment,
            account_alias=account_alias,
            instrument_family=instrument_family,
            equity_symbol=equity_symbol,
            equity_side=equity_side,
            legs=legs,
            parent_quantity=parent_quantity,
            debit_credit=debit_credit,
            net_limit_price=net_limit_price,
            maximum_loss=maximum_loss,
            buying_power_required=buying_power_required,
            time_in_force=time_in_force,
            order_type=order_type,
            idempotency_key=idempotency_key,
            request_hash=content_hash(material),
            quote_snapshot_id=quote_snapshot_id,
            supersedes_order_request_id=supersedes_order_request_id,
            created_at=created_at,
            expires_at=expires_at,
        )

    from_fields = create

    @model_validator(mode="after")
    def validate_order(self) -> OrderRequest:
        if self.expires_at <= self.created_at:
            raise ValueError("order expires_at must be after created_at")
        is_equity = self.equity_symbol is not None or self.equity_side is not None
        if is_equity:
            if self.equity_symbol is None or self.equity_side is None or self.legs:
                raise ValueError("equity requests require symbol and side and no option legs")
            if self.instrument_family is not InstrumentFamily.EQUITY:
                raise ValueError("equity requests must use the EQUITY instrument family")
        else:
            if not 1 <= len(self.legs) <= 4:
                raise ValueError("option requests require one to four legs")
            if self.instrument_family is InstrumentFamily.EQUITY:
                raise ValueError("option requests cannot use the EQUITY instrument family")
            if self.parent_quantity != self.parent_quantity.to_integral_value():
                raise ValueError("option parent_quantity must be integral")
            if len(self.legs) == 1:
                expected_debit_credit = (
                    DebitCredit.DEBIT if self.legs[0].side is OrderSide.BUY else DebitCredit.CREDIT
                )
                if self.debit_credit is not expected_debit_credit:
                    raise ValueError("single-option side must agree with debit_credit")
        if self.order_type == "limit" and self.net_limit_price is None:
            raise ValueError("limit orders require a positive non-null net_limit_price")
        if self.order_type == "market" and self.net_limit_price is not None:
            raise ValueError("market orders require net_limit_price to be null")
        if self.debit_credit is DebitCredit.CREDIT and (
            self.net_limit_price is None or self.net_limit_price <= 0
        ):
            raise ValueError("credit requests require a positive credit limit magnitude")
        if self.request_hash != self.computed_request_hash():
            raise ValueError("request_hash does not match order request content")
        return self
