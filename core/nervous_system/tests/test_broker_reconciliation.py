"""Broker reconciliation and pending intents (Task 21)."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest

from core.nervous_system.contracts.enums import (
    ExecutionStatus,
    RuntimeEnvironment,
    SubmissionAttemptStatus,
)
from core.nervous_system.execution.broker import BrokerUnavailable
from core.nervous_system.execution.pending import (
    PendingIntent,
    PendingIntentStore,
    PendingStatus,
)
from core.nervous_system.execution.reconciliation import (
    DiscrepancyKind,
    reconcile_broker_account,
)
from core.nervous_system.tests.fixtures.gateway_harness import (
    ACCOUNT,
    NOW,
    FakeBroker,
    FakeJournal,
    broker_order,
    broker_position,
    clock,
    expected_client_order_id,
    fake_uow_factory,
    order_request,
)
from core.nervous_system.tests.fixtures.journal_events import event as journal_event


D = Decimal
SINCE = NOW - timedelta(hours=1)


class ReconBroker(FakeBroker):
    def __init__(self, *, orders=(), positions=()) -> None:
        super().__init__(positions_result=positions)
        self._orders = orders

    def orders(self, *, status: str = "all"):
        if isinstance(self._orders, Exception):
            raise self._orders
        return self._orders


def uow_with(store=None):
    factory, shared = fake_uow_factory(store)
    return factory().__enter__(), shared


def reconcile(*, broker, journal=None, store=None):
    unit, _ = uow_with(store)
    return reconcile_broker_account(
        broker=broker,
        unit_of_work=unit,
        journal=journal or FakeJournal(),
        account_id=ACCOUNT,
        since=SINCE,
        observed_at=NOW,
    )


# --------------------------------------------------------------------------
# Broker facts are authoritative
# --------------------------------------------------------------------------


def test_a_broker_only_order_is_recovered_and_reported() -> None:
    request = order_request()
    broker = ReconBroker(orders=(broker_order(request),))

    report = reconcile(broker=broker)

    assert report.recovered_orders == ("brk-1",)
    assert any(
        item.kind is DiscrepancyKind.BROKER_ONLY_ORDER for item in report.discrepancies
    )
    assert report.is_clean is False


def test_ownership_comes_only_from_a_confirmed_fill() -> None:
    request = order_request()
    accepted = broker_order(request)
    filled = broker_order(
        request,
        broker_order_id="brk-2",
        status=ExecutionStatus.FILLED,
        raw_status="filled",
        filled_quantity=D("25"),
        average_fill_price=D("200.10"),
    )
    broker = ReconBroker(orders=(accepted, filled))

    report = reconcile(broker=broker)

    assert report.ownership_created == ("brk-2",)
    assert "brk-1" not in report.ownership_created, "acceptance is not a fill"


def test_a_manual_position_is_unassigned() -> None:
    broker = ReconBroker(positions=(broker_position("JPM", 40.0),))

    report = reconcile(broker=broker)

    assert report.unassigned_positions == ("JPM",)
    assert any(
        item.kind is DiscrepancyKind.UNASSIGNED_POSITION
        for item in report.discrepancies
    )


def test_a_flat_position_is_not_reported() -> None:
    broker = ReconBroker(positions=(broker_position("JPM", 0.0),))

    assert reconcile(broker=broker).unassigned_positions == ()


def test_an_unreadable_order_list_is_reported_not_assumed_empty() -> None:
    broker = ReconBroker(orders=BrokerUnavailable("broker down"))

    report = reconcile(broker=broker)

    assert any(
        item.kind is DiscrepancyKind.LOOKUP_UNAVAILABLE for item in report.discrepancies
    )
    assert report.is_clean is False


def test_orders_before_the_window_are_skipped() -> None:
    request = order_request()
    old = broker_order(request, submitted_at=SINCE - timedelta(days=2))
    broker = ReconBroker(orders=(old,))

    assert reconcile(broker=broker).recovered_orders == ()


def test_a_recorded_attempt_is_not_reported_as_broker_only() -> None:
    request = order_request()
    factory, store = fake_uow_factory()
    with factory() as unit:
        unit.executions.reserve_or_load_attempt(
            request=request,
            client_order_id=expected_client_order_id(request),
            reserved_at=NOW,
        )
    broker = ReconBroker(orders=(broker_order(request),))

    report = reconcile(broker=broker, store=store)

    assert report.recovered_orders == ()


def test_a_fill_the_record_does_not_reflect_is_a_status_mismatch() -> None:
    request = order_request()
    factory, store = fake_uow_factory()
    with factory() as unit:
        unit.executions.reserve_or_load_attempt(
            request=request,
            client_order_id=expected_client_order_id(request),
            reserved_at=NOW,
        )
    filled = broker_order(
        request,
        status=ExecutionStatus.FILLED,
        raw_status="filled",
        filled_quantity=D("25"),
        average_fill_price=D("200.10"),
    )

    report = reconcile(broker=ReconBroker(orders=(filled,)), store=store)

    assert any(
        item.kind is DiscrepancyKind.STATUS_MISMATCH for item in report.discrepancies
    )


# --------------------------------------------------------------------------
# Journal evidence
# --------------------------------------------------------------------------


class ReplayJournal(FakeJournal):
    def __init__(self, events) -> None:
        super().__init__()
        self._events = events

    def iter_events(self, *, account_id: str, after=None):
        if isinstance(self._events, Exception):
            raise self._events
        return iter(self._events)


def test_journal_only_events_are_recovered() -> None:
    record = journal_event()
    broker = ReconBroker()

    report = reconcile(broker=broker, journal=ReplayJournal([record]))

    assert report.recovered_journal_events == (record.event_id,)
    assert any(
        item.kind is DiscrepancyKind.JOURNAL_ONLY_EVENT for item in report.discrepancies
    )


def test_a_corrupt_journal_event_is_reported_and_preserved() -> None:
    from core.nervous_system.execution.journal import ExecutionJournalEvent

    record = journal_event()
    corrupt = ExecutionJournalEvent.model_construct(
        **{**record.model_dump(), "payload": {"tampered": True}}
    )

    report = reconcile(
        broker=ReconBroker(), journal=ReplayJournal([corrupt])
    )

    assert any(
        item.kind is DiscrepancyKind.CORRUPT_JOURNAL_EVENT
        for item in report.discrepancies
    )
    assert report.recovered_journal_events == (), "corrupt evidence is never replayed"


def test_an_unreadable_journal_is_reported() -> None:
    report = reconcile(
        broker=ReconBroker(), journal=ReplayJournal(RuntimeError("disk gone"))
    )

    assert any(
        item.kind is DiscrepancyKind.CORRUPT_JOURNAL_EVENT
        for item in report.discrepancies
    )


def test_a_clean_account_reports_nothing() -> None:
    assert reconcile(broker=ReconBroker()).is_clean is True


# --------------------------------------------------------------------------
# Pending intents
# --------------------------------------------------------------------------


def pending(**overrides):
    fields = {
        "pending_intent_id": uuid5(NAMESPACE_URL, "pending/1"),
        "intent_id": uuid5(NAMESPACE_URL, "pending/intent"),
        "snapshot_id": uuid5(NAMESPACE_URL, "pending/snapshot"),
        "strategy_id": "meta_ranker",
        "ticker": "AMD",
        "account_alias": ACCOUNT,
        "original_decision_time": NOW,
        "expires_at": NOW + timedelta(hours=6),
        "deferral_reason": "AFTER_CLOSE",
    }
    fields.update(overrides)
    return PendingIntent(**fields)


def test_a_pending_intent_stores_no_instrument_detail() -> None:
    item = pending()
    fields = set(item.model_dump())

    for forbidden in (
        "occ_symbol",
        "symbol",
        "quantity",
        "limit_price",
        "quote",
        "net_price",
    ):
        assert forbidden not in fields, f"{forbidden} would be stale on retry"
    assert item.intent_id is not None
    assert item.original_decision_time == NOW


def test_claiming_and_completing_a_pending_intent() -> None:
    store = PendingIntentStore(clock(NOW))
    store.add(pending())

    claimed = store.claim(
        pending().pending_intent_id,
        owner="worker-a",
        token="tok",
        lease=timedelta(minutes=5),
    )
    assert claimed.status is PendingStatus.CLAIMED

    decision_id = uuid4()
    assert store.complete(
        pending().pending_intent_id, token="tok", decision_id=decision_id
    )
    assert store.get(pending().pending_intent_id).status is PendingStatus.RETRIED
    assert store.get(pending().pending_intent_id).resulting_decision_id == decision_id


def test_a_second_worker_cannot_claim_a_live_lease() -> None:
    store = PendingIntentStore(clock(NOW))
    store.add(pending())
    store.claim(
        pending().pending_intent_id,
        owner="worker-a",
        token="tok",
        lease=timedelta(minutes=5),
    )

    assert (
        store.claim(
            pending().pending_intent_id,
            owner="worker-b",
            token="tok-b",
            lease=timedelta(minutes=5),
        )
        is None
    )


def test_an_expired_claim_may_be_taken_over() -> None:
    store = PendingIntentStore(clock(NOW))
    store.add(pending())
    store.claim(
        pending().pending_intent_id,
        owner="worker-a",
        token="tok",
        lease=timedelta(seconds=30),
    )
    store._clock = clock(NOW + timedelta(minutes=10))

    taken = store.claim(
        pending().pending_intent_id,
        owner="worker-b",
        token="tok-b",
        lease=timedelta(minutes=5),
    )
    assert taken is not None
    assert taken.claim_owner == "worker-b"


def test_renewing_requires_the_current_token() -> None:
    store = PendingIntentStore(clock(NOW))
    store.add(pending())
    store.claim(
        pending().pending_intent_id,
        owner="worker-a",
        token="tok",
        lease=timedelta(minutes=5),
    )

    assert store.renew(pending().pending_intent_id, token="tok", lease=timedelta(minutes=5))
    assert not store.renew(
        pending().pending_intent_id, token="stale", lease=timedelta(minutes=5)
    )


def test_expiry_retires_only_safe_work() -> None:
    store = PendingIntentStore(clock(NOW + timedelta(days=1)))
    store.add(pending())
    store.add(
        pending(
            pending_intent_id=uuid5(NAMESPACE_URL, "pending/2"),
            deferral_reason="AMBIGUOUS_SUBMISSION",
        )
    )
    store.mark_ambiguous(
        uuid5(NAMESPACE_URL, "pending/2"), reason="AMBIGUOUS_SUBMISSION"
    )

    expired = store.expire_due()

    assert [item.pending_intent_id for item in expired] == [
        uuid5(NAMESPACE_URL, "pending/1")
    ]
    assert (
        store.get(uuid5(NAMESPACE_URL, "pending/2")).status is PendingStatus.AMBIGUOUS
    ), "ambiguous work is retained until authoritative reconciliation"


def test_an_expired_intent_is_not_claimable() -> None:
    store = PendingIntentStore(clock(NOW + timedelta(days=1)))
    store.add(pending())

    assert store.claimable() == ()
    assert (
        store.claim(
            pending().pending_intent_id,
            owner="worker-a",
            token="tok",
            lease=timedelta(minutes=5),
        )
        is None
    )


def test_a_pending_intent_must_expire_after_its_decision() -> None:
    with pytest.raises(ValueError, match="expire after"):
        pending(expires_at=NOW - timedelta(minutes=1))


def test_a_superseded_intent_must_name_its_successor() -> None:
    with pytest.raises(ValueError, match="name its successor"):
        pending(status=PendingStatus.SUPERSEDED)
