"""Typed outbox event definitions.

Event identity is derived from content, so enqueuing the same fact twice
produces one row and a handler can deduplicate on the event ID alone.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping
from uuid import UUID


class EventType(str, Enum):
    SNAPSHOT_BUILT = "SnapshotBuilt"
    INTENT_EMITTED = "IntentEmitted"
    POLICY_DECISION_CREATED = "PolicyDecisionCreated"
    DECISION_RECORDED = "DecisionRecorded"
    ORDER_PLANNED = "OrderPlanned"
    ORDER_SUBMITTED = "OrderSubmitted"
    ORDER_RESOLVED = "OrderResolved"
    JOB_COMPLETED = "JobCompleted"
    RECONCILIATION_REQUIRED = "ReconciliationRequired"


class AggregateType(str, Enum):
    SNAPSHOT = "ContextSnapshot"
    DECISION = "DecisionRecord"
    ORDER_REQUEST = "OrderRequest"
    JOB_RUN = "JobRun"


def decision_recorded(decision_record_id: UUID, **payload: Any) -> dict[str, Any]:
    return {
        "event_type": EventType.DECISION_RECORDED.value,
        "aggregate_type": AggregateType.DECISION.value,
        "aggregate_id": str(decision_record_id),
        "payload": dict(payload),
    }


def order_submitted(order_request_id: UUID, **payload: Any) -> dict[str, Any]:
    return {
        "event_type": EventType.ORDER_SUBMITTED.value,
        "aggregate_type": AggregateType.ORDER_REQUEST.value,
        "aggregate_id": str(order_request_id),
        "payload": dict(payload),
    }


def job_completed(job_run_id: UUID, **payload: Any) -> dict[str, Any]:
    return {
        "event_type": EventType.JOB_COMPLETED.value,
        "aggregate_type": AggregateType.JOB_RUN.value,
        "aggregate_id": str(job_run_id),
        "payload": dict(payload),
    }


__all__ = [
    "AggregateType",
    "EventType",
    "decision_recorded",
    "job_completed",
    "order_submitted",
]
