"""Transactional outbox dispatch.

Delivery is at least once, so handlers must be idempotent. The dispatcher
guarantees the parts a handler cannot: rows are claimed with
``FOR UPDATE SKIP LOCKED`` and committed *before* any handler runs, handlers
run outside the transaction, and finalisation only succeeds while the claim
token still matches.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from core.nervous_system.persistence.repositories.operations import OutboxEventRecord


Handler = Callable[[OutboxEventRecord], None]


@dataclass
class DispatchResult:
    claimed: int = 0
    delivered: int = 0
    failed: int = 0
    skipped_no_handler: int = 0
    lost_lease: int = 0
    errors: list[str] = field(default_factory=list)


class OutboxDispatcher:
    def __init__(
        self,
        *,
        unit_of_work_factory: Callable[[], Any],
        handlers: Mapping[str, Handler],
        worker_id: str,
        clock: Callable[[], datetime],
        lease_seconds: int = 60,
        batch_size: int = 10,
    ) -> None:
        self._uow_factory = unit_of_work_factory
        self._handlers = dict(handlers)
        self._worker_id = worker_id
        self._clock = clock
        self._lease_seconds = lease_seconds
        self._batch_size = batch_size

    def dispatch_once(self) -> DispatchResult:
        result = DispatchResult()
        now = self._clock()

        # 1. Claim and commit. A handler must never run inside the claiming
        #    transaction, or a slow handler would hold locks and a rollback
        #    would erase the record that it ran.
        with self._uow_factory() as uow:
            claimed = uow.operations.claim(
                worker_id=self._worker_id,
                now=now,
                lease_seconds=self._lease_seconds,
                limit=self._batch_size,
            )
            uow.commit()
        result.claimed = len(claimed)

        for event in claimed:
            handler = self._handlers.get(event.event_type)
            if handler is None:
                result.skipped_no_handler += 1
                self._release(event, "no handler registered")
                continue
            # 2. Handler runs outside any transaction.
            try:
                handler(event)
            except Exception as exc:
                result.failed += 1
                result.errors.append(f"{event.event_type}: {type(exc).__name__}: {exc}")
                self._release(event, f"{type(exc).__name__}: {exc}")
                continue
            # 3. Finalise in a new transaction, fenced by the claim token.
            with self._uow_factory() as uow:
                finalized = uow.operations.mark_delivered(
                    event.outbox_event_id,
                    worker_id=self._worker_id,
                    claim_token=event.claim_token,
                    delivered_at=self._clock(),
                )
                uow.commit()
            if finalized:
                result.delivered += 1
            else:
                # The lease expired and someone else owns it now. The handler
                # already ran, which is why delivery is at least once.
                result.lost_lease += 1
        return result

    def _release(self, event: OutboxEventRecord, error: str) -> None:
        with self._uow_factory() as uow:
            uow.operations.mark_failed(
                event.outbox_event_id,
                worker_id=self._worker_id,
                claim_token=event.claim_token,
                error=error,
                retry_at=self._clock() + timedelta(seconds=self._lease_seconds),
            )
            uow.commit()


__all__ = ["DispatchResult", "Handler", "OutboxDispatcher"]
