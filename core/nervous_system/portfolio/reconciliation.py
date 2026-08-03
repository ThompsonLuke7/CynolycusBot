"""Broker versus ownership reconciliation.

Broker facts are authoritative and are never rewritten to agree with internal
attribution.  Disagreement is reported as evidence; corrections are new
observations, not edits to what the broker said.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from uuid import UUID

from core.nervous_system.contracts.enums import OwnershipStatus, ReconciliationStatus
from core.nervous_system.contracts.portfolio import (
    OwnershipRecord,
    PortfolioReconciliation,
    ReconciliationLine,
)
from core.nervous_system.contracts.states import PortfolioState

from .ownership import broker_position_key


_ZERO = Decimal("0")


def _decimal(value: float | int | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def reconcile_portfolio(
    broker: PortfolioState,
    ownership: Sequence[OwnershipRecord],
    *,
    reconciliation_id: UUID,
) -> PortfolioReconciliation:
    """Compare broker positions with fill-backed ownership totals."""

    broker_quantities: dict[str, Decimal] = {}
    for position in broker.positions:
        key = broker_position_key(broker.account_alias, position.symbol)
        broker_quantities[key] = broker_quantities.get(key, _ZERO) + _decimal(
            position.quantity
        )

    owned_quantities: dict[str, Decimal] = {}
    owners: dict[str, set[str]] = {}
    ownership_ids: dict[str, list[UUID]] = {}
    for record in ownership:
        if record.ownership_status is not OwnershipStatus.ASSIGNED:
            continue
        key = record.broker_position_key
        owned_quantities[key] = owned_quantities.get(key, _ZERO) + record.quantity
        if record.strategy_id is not None:
            owners.setdefault(key, set()).add(record.strategy_id)
        ownership_ids.setdefault(key, []).append(record.ownership_id)

    buckets: dict[ReconciliationStatus, list[ReconciliationLine]] = {
        status: [] for status in ReconciliationStatus
    }

    for key in sorted(set(broker_quantities) | set(owned_quantities)):
        broker_quantity = broker_quantities.get(key, _ZERO)
        owned_quantity = owned_quantities.get(key, _ZERO)
        status = _classify(broker_quantity, owned_quantity)
        buckets[status].append(
            ReconciliationLine(
                broker_position_key=key,
                status=status,
                broker_quantity=broker_quantity,
                owned_quantity=owned_quantity,
                strategy_ids=tuple(sorted(owners.get(key, ()))),
                ownership_ids=tuple(sorted(ownership_ids.get(key, []), key=str)),
            )
        )

    return PortfolioReconciliation.create(
        reconciliation_id=reconciliation_id,
        portfolio_state_id=broker.state_id,
        observed_at=broker.broker_observed_at,
        matched=tuple(buckets[ReconciliationStatus.MATCHED]),
        partial=tuple(buckets[ReconciliationStatus.PARTIAL]),
        unassigned=tuple(buckets[ReconciliationStatus.UNASSIGNED]),
        orphaned=tuple(buckets[ReconciliationStatus.ORPHANED_OWNERSHIP]),
        quantity_mismatches=tuple(buckets[ReconciliationStatus.QUANTITY_MISMATCH]),
        ownership_adjustment_ids=(),
    )


def _classify(broker_quantity: Decimal, owned_quantity: Decimal) -> ReconciliationStatus:
    if owned_quantity == _ZERO:
        # The broker holds something nobody claimed: a manual or imported trade.
        return ReconciliationStatus.UNASSIGNED
    if broker_quantity == _ZERO:
        return ReconciliationStatus.ORPHANED_OWNERSHIP
    if broker_quantity == owned_quantity:
        return ReconciliationStatus.MATCHED
    # Opposite signs are never a partial attribution, they are a real conflict.
    same_direction = (broker_quantity > _ZERO) == (owned_quantity > _ZERO)
    if same_direction and abs(owned_quantity) < abs(broker_quantity):
        return ReconciliationStatus.PARTIAL
    return ReconciliationStatus.QUANTITY_MISMATCH


__all__ = ["reconcile_portfolio"]
