"""Deferred entry intents.

A deferred entry is stored as the *intent*, never as a finished order. It
deliberately holds no OCC symbol, quote, quantity, or limit price: those were
computed from a chain that will be stale by the time the intent is retried.
Retry rebuilds the snapshot, policy, chain, selection, and order from scratch
and links the new decision back to this intent.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import model_validator

from core.nervous_system.contracts.base import ContractModel, UtcDatetime


class PendingStatus(str, Enum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    RETRIED = "RETRIED"
    EXPIRED = "EXPIRED"
    SUPERSEDED = "SUPERSEDED"
    AMBIGUOUS = "AMBIGUOUS"


# Ambiguous work is never cleaned up on a timer; only authoritative
# reconciliation may retire it.
_RETIRABLE = frozenset({PendingStatus.PENDING, PendingStatus.CLAIMED})


class PendingIntent(ContractModel):
    """One deferred entry, holding identity and lineage but no instrument."""

    pending_intent_id: UUID
    intent_id: UUID
    snapshot_id: UUID | None
    strategy_id: str
    ticker: str
    account_alias: str
    original_decision_time: UtcDatetime
    expires_at: UtcDatetime
    deferral_reason: str
    status: PendingStatus = PendingStatus.PENDING
    claim_owner: str | None = None
    claim_until: UtcDatetime | None = None
    claim_token: str | None = None
    superseded_by_intent_id: UUID | None = None
    resulting_decision_id: UUID | None = None

    @model_validator(mode="after")
    def validate_pending(self) -> PendingIntent:
        if self.expires_at <= self.original_decision_time:
            raise ValueError("a pending intent must expire after its decision time")
        if not self.deferral_reason.strip():
            raise ValueError("a deferral requires a reason")
        if self.status is PendingStatus.CLAIMED and not self.claim_token:
            raise ValueError("a claimed intent requires a claim token")
        if self.status is PendingStatus.SUPERSEDED and self.superseded_by_intent_id is None:
            raise ValueError("a superseded intent must name its successor")
        if self.status is PendingStatus.RETRIED and self.resulting_decision_id is None:
            raise ValueError("a retried intent must name the decision it produced")
        return self

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at

    def is_claimable(self, now: datetime, *, owner: str | None = None) -> bool:
        """Whether this worker may take the intent right now."""

        if self.status not in _RETIRABLE:
            return False
        if self.is_expired(now):
            return False
        if self.status is PendingStatus.PENDING:
            return True
        if self.claim_until is None:
            return True
        if now >= self.claim_until:
            return True
        return owner is not None and owner == self.claim_owner


class PendingIntentStore:
    """In-memory pending-intent lifecycle.

    Task 22 binds this to PostgreSQL; the transitions live here so the rules
    are testable without a database.
    """

    def __init__(self, clock: Any) -> None:
        self._clock = clock
        self._items: dict[UUID, PendingIntent] = {}

    def add(self, intent: PendingIntent) -> PendingIntent:
        if intent.pending_intent_id in self._items:
            raise ValueError("pending intent already exists")
        self._items[intent.pending_intent_id] = intent
        return intent

    def get(self, pending_intent_id: UUID) -> PendingIntent | None:
        return self._items.get(pending_intent_id)

    def claim(
        self,
        pending_intent_id: UUID,
        *,
        owner: str,
        token: str,
        lease: timedelta,
    ) -> PendingIntent | None:
        now = self._clock()
        current = self._items.get(pending_intent_id)
        if current is None or not current.is_claimable(now, owner=owner):
            return None
        claimed = current.model_copy(
            update={
                "status": PendingStatus.CLAIMED,
                "claim_owner": owner,
                "claim_token": token,
                "claim_until": now + lease,
            }
        )
        self._items[pending_intent_id] = claimed
        return claimed

    def renew(self, pending_intent_id: UUID, *, token: str, lease: timedelta) -> bool:
        current = self._items.get(pending_intent_id)
        if current is None or current.claim_token != token:
            return False
        if current.status is not PendingStatus.CLAIMED:
            return False
        self._items[pending_intent_id] = current.model_copy(
            update={"claim_until": self._clock() + lease}
        )
        return True

    def complete(
        self,
        pending_intent_id: UUID,
        *,
        token: str,
        decision_id: UUID,
    ) -> bool:
        current = self._items.get(pending_intent_id)
        if current is None or current.claim_token != token:
            return False
        self._items[pending_intent_id] = current.model_copy(
            update={
                "status": PendingStatus.RETRIED,
                "resulting_decision_id": decision_id,
            }
        )
        return True

    def mark_ambiguous(self, pending_intent_id: UUID, *, reason: str) -> bool:
        current = self._items.get(pending_intent_id)
        if current is None:
            return False
        self._items[pending_intent_id] = current.model_copy(
            update={
                "status": PendingStatus.AMBIGUOUS,
                "deferral_reason": reason,
            }
        )
        return True

    def expire_due(self) -> tuple[PendingIntent, ...]:
        """Expire only work that is safe to abandon.

        Ambiguous items are deliberately retained: abandoning one would drop
        an order that may exist at the broker.
        """

        now = self._clock()
        expired: list[PendingIntent] = []
        for key, item in list(self._items.items()):
            if item.status not in _RETIRABLE or not item.is_expired(now):
                continue
            retired = item.model_copy(update={"status": PendingStatus.EXPIRED})
            self._items[key] = retired
            expired.append(retired)
        return tuple(expired)

    def claimable(self) -> tuple[PendingIntent, ...]:
        now = self._clock()
        return tuple(
            item
            for item in sorted(
                self._items.values(), key=lambda entry: str(entry.pending_intent_id)
            )
            if item.is_claimable(now)
        )


__all__ = ["PendingIntent", "PendingIntentStore", "PendingStatus"]
