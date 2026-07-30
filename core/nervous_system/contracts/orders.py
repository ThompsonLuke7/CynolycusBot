from __future__ import annotations

from datetime import date
from typing import Annotated
from uuid import UUID

from pydantic import Field, model_validator

from .base import ContractModel, NonNegativeDecimal, PositiveDecimal, Sha256Hex, UtcDatetime
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
    net_limit_price: PositiveDecimal
    maximum_loss: NonNegativeDecimal
    buying_power_required: NonNegativeDecimal
    time_in_force: str
    order_type: str
    idempotency_key: str
    request_hash: Sha256Hex
    quote_snapshot_id: UUID | None = None
    supersedes_order_request_id: UUID | None = None
    created_at: UtcDatetime
    expires_at: UtcDatetime

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
        if self.debit_credit is DebitCredit.CREDIT and self.net_limit_price <= 0:
            raise ValueError("credit requests require a positive credit limit magnitude")
        return self
