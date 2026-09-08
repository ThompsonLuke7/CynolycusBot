"""Explicit bridge from a paper-only scalper intent to the shared gateway contract."""
from __future__ import annotations

import hashlib
from datetime import timezone
from decimal import Decimal
from uuid import UUID

from core.nervous_system.contracts.enums import (
    DebitCredit, DecisionKind, InstrumentFamily, OrderSide, RuntimeEnvironment,
)
from core.nervous_system.contracts.orders import OrderRequest
from strategies.momentum_scalper.contracts import OrderIntent


def to_paper_order_request(
    *,
    intent: OrderIntent,
    decision_id: UUID,
    policy_decision_id: UUID,
    account_alias: str,
) -> OrderRequest:
    """Convert an intent without submitting it; production-live is impossible here."""

    if not intent.paper_only:
        raise ValueError("only paper-only momentum intents may cross the execution boundary")
    created = intent.created_at.astimezone(timezone.utc)
    expires = intent.expires_at.astimezone(timezone.utc)
    limit = Decimal(str(intent.limit_price))
    quantity = Decimal(intent.quantity)
    risk = Decimal(str(max(intent.limit_price - intent.stop_price, 0.0))) * quantity
    key_material = f"momentum_scalper_v1|{decision_id}|{intent.ticker}|{created.isoformat()}|{intent.quantity}|{intent.limit_price}|{intent.extended_hours}"
    return OrderRequest.create(
        decision_id=decision_id,
        policy_decision_id=policy_decision_id,
        environment=RuntimeEnvironment.QA_PAPER,
        account_alias=account_alias,
        decision_kind=DecisionKind.ENTRY,
        risk_reducing=False,
        instrument_family=InstrumentFamily.EQUITY,
        equity_symbol=intent.ticker,
        equity_side=OrderSide.BUY,
        parent_quantity=quantity,
        debit_credit=DebitCredit.DEBIT,
        net_limit_price=limit,
        maximum_loss=risk,
        buying_power_required=limit * quantity,
        time_in_force="day",
        order_type="limit",
        extended_hours=intent.extended_hours,
        idempotency_key=hashlib.sha256(key_material.encode("utf-8")).hexdigest(),
        created_at=created,
        expires_at=expires,
    )


__all__ = ["to_paper_order_request"]
