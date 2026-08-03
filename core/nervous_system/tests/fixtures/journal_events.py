"""Shared journal event builders (Task 20)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from core.nervous_system.contracts.enums import RuntimeEnvironment
from core.nervous_system.execution.journal import ExecutionJournalEvent


UTC = timezone.utc
EVENT_TIME = datetime(2026, 8, 2, 18, 30, 15, 123456, tzinfo=UTC)
ORDER_REQUEST_ID = uuid5(NAMESPACE_URL, "journal-test/order")
DECISION_ID = uuid5(NAMESPACE_URL, "journal-test/decision")
CLIENT_ORDER_ID = "ab" * 24


def event(
    *,
    suffix: str = "1",
    event_time: datetime | None = None,
    account_id: str = "paper",
    order_request_id: UUID | None = None,
    payload: dict[str, Any] | None = None,
    **overrides: Any,
) -> ExecutionJournalEvent:
    when = event_time or EVENT_TIME
    fields: dict[str, Any] = {
        "event_id": uuid5(NAMESPACE_URL, f"journal-test/event/{suffix}"),
        "event_time": when,
        "observed_at": when + timedelta(milliseconds=5),
        "account_id": account_id,
        "environment": RuntimeEnvironment.QA_PAPER,
        "event_type": "SUBMISSION_INTENT",
        "decision_id": DECISION_ID,
        "order_request_id": order_request_id or ORDER_REQUEST_ID,
        "sequence_no": 1,
        "client_order_id": CLIENT_ORDER_ID,
        "broker_order_id": None,
        "payload": payload if payload is not None else {"symbol": "AMD", "qty": 25},
    }
    fields.update(overrides)
    return ExecutionJournalEvent.create(**fields)


__all__ = [
    "CLIENT_ORDER_ID",
    "DECISION_ID",
    "EVENT_TIME",
    "ORDER_REQUEST_ID",
    "event",
]
