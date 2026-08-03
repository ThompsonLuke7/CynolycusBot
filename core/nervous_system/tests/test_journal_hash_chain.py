"""Journal hash-chain tamper detection (Task 20)."""

from __future__ import annotations

from datetime import timedelta
from uuid import NAMESPACE_URL, uuid5

import pytest

from core.nervous_system.contracts.enums import RuntimeEnvironment
from core.nervous_system.execution.journal import (
    ExecutionJournalEvent,
    JournalConflict,
    PostgresPersistenceStatus,
    REDACTED,
    link_event,
    redact,
    verify_chain,
)
from core.nervous_system.tests.fixtures.journal_events import (
    CLIENT_ORDER_ID,
    DECISION_ID,
    EVENT_TIME,
    ORDER_REQUEST_ID,
    event,
)


def chain(length: int = 3) -> list[ExecutionJournalEvent]:
    events: list[ExecutionJournalEvent] = []
    previous = None
    for index in range(length):
        previous = link_event(
            previous,
            event_id=uuid5(NAMESPACE_URL, f"chain/{index}"),
            event_time=EVENT_TIME + timedelta(seconds=index),
            observed_at=EVENT_TIME + timedelta(seconds=index, milliseconds=5),
            account_id="paper",
            environment=RuntimeEnvironment.QA_PAPER,
            event_type=("SUBMISSION_INTENT" if index == 0 else "BROKER_EVENT"),
            decision_id=DECISION_ID,
            order_request_id=ORDER_REQUEST_ID,
            client_order_id=CLIENT_ORDER_ID,
            broker_order_id=None if index == 0 else "brk-1",
            payload={"step": index},
        )
        events.append(previous)
    return events


def test_a_well_formed_chain_verifies() -> None:
    verify_chain(chain())


def test_submission_intent_is_sequence_one_without_a_predecessor() -> None:
    first = chain()[0]

    assert first.sequence_no == 1
    assert first.event_type == "SUBMISSION_INTENT"
    assert first.previous_event_id is None
    assert first.previous_event_hash is None


def test_later_events_carry_both_predecessor_id_and_hash() -> None:
    events = chain()

    for earlier, later in zip(events, events[1:]):
        assert later.previous_event_id == earlier.event_id
        assert later.previous_event_hash == earlier.event_hash


def test_a_mutated_record_cannot_even_be_loaded() -> None:
    """The contract refuses a record whose hash does not match its content."""

    events = chain()
    corrupted = events[1].model_dump(mode="json")
    corrupted["payload"] = {"step": 99}

    with pytest.raises(ValueError, match="event_hash does not match"):
        ExecutionJournalEvent.model_validate(corrupted)


def test_mutating_a_payload_breaks_the_event_hash() -> None:
    events = chain()
    # model_construct bypasses validation the way a hand-edited file on disk
    # would, so verify_chain is the last line of defence rather than the first.
    tampered = ExecutionJournalEvent.model_construct(
        **{**events[1].model_dump(), "payload": {"step": 99}}
    )

    assert tampered.event_hash != tampered.computed_event_hash()
    with pytest.raises(JournalConflict, match="has been mutated"):
        verify_chain([events[0], tampered, events[2]])


def test_deleting_an_event_between_checkpoints_is_detected() -> None:
    events = chain()

    with pytest.raises(JournalConflict, match="expected sequence 2"):
        verify_chain([events[0], events[2]])


def test_reordering_is_detected() -> None:
    events = chain()

    with pytest.raises(JournalConflict):
        verify_chain([events[1], events[0], events[2]])


def test_duplicate_sequence_numbers_are_rejected() -> None:
    events = chain()

    with pytest.raises(JournalConflict):
        verify_chain([events[0], events[1], events[1]])


def test_events_from_another_order_cannot_be_mixed_in() -> None:
    events = chain()
    foreign = event(
        suffix="foreign",
        order_request_id=uuid5(NAMESPACE_URL, "journal-test/other-order"),
    )

    with pytest.raises(JournalConflict, match="exactly one order_request_id"):
        verify_chain([events[0], foreign])


def test_link_event_refuses_to_cross_order_boundaries() -> None:
    first = chain()[0]

    with pytest.raises(JournalConflict, match="cross order_request_id"):
        link_event(
            first,
            event_id=uuid5(NAMESPACE_URL, "chain/cross"),
            event_time=EVENT_TIME + timedelta(seconds=1),
            observed_at=EVENT_TIME + timedelta(seconds=1),
            account_id="paper",
            environment=RuntimeEnvironment.QA_PAPER,
            event_type="BROKER_EVENT",
            decision_id=DECISION_ID,
            order_request_id=uuid5(NAMESPACE_URL, "journal-test/other-order"),
            client_order_id=CLIENT_ORDER_ID,
            broker_order_id="brk-9",
        )


def test_a_forged_predecessor_hash_is_detected() -> None:
    events = chain(2)
    forged = ExecutionJournalEvent.model_construct(
        **{**events[1].model_dump(), "previous_event_hash": "f" * 64}
    )

    with pytest.raises(JournalConflict):
        verify_chain([events[0], forged])


def test_a_substituted_predecessor_is_caught_by_the_hash_link() -> None:
    """Rewriting an earlier event under its own ID must not go unnoticed.

    Both records here are individually valid: the substitute hashes correctly
    for its own content, and the successor still cites the right predecessor
    ID. Only the hash linkage reveals that history was rewritten.
    """

    events = chain(2)
    substitute = ExecutionJournalEvent.create(
        **{
            **events[0].model_dump(exclude={"event_hash"}),
            "payload": {"step": "rewritten"},
        }
    )

    assert substitute.event_id == events[0].event_id
    assert substitute.event_hash == substitute.computed_event_hash()
    assert events[1].previous_event_id == substitute.event_id
    assert events[1].previous_event_hash != substitute.event_hash

    with pytest.raises(JournalConflict, match="breaks the hash chain"):
        verify_chain([substitute, events[1]])


def test_sequence_one_cannot_declare_a_predecessor() -> None:
    with pytest.raises(ValueError, match="sequence 1 has no predecessor"):
        event(previous_event_id=DECISION_ID, previous_event_hash="a" * 64)


def test_later_sequences_require_a_predecessor() -> None:
    with pytest.raises(ValueError, match="require both predecessor"):
        event(sequence_no=2)


# --------------------------------------------------------------------------
# Redaction happens before hashing and before any sink sees bytes
# --------------------------------------------------------------------------


def test_credentials_are_redacted_before_hashing() -> None:
    record = event(
        payload={
            "api_key": "PKLIVEKEY",
            "secret": "shhh",
            "access_token": "tok",
            "authorization": "Bearer abc",
            "private_key": "-----BEGIN",
            "password": "hunter2",
            "account_number": "PA123456",
            "account_id": "9f0c-private",
            "client_order_id": "keep-me",
            "symbol": "AMD",
        }
    )

    for key in (
        "api_key",
        "secret",
        "access_token",
        "authorization",
        "private_key",
        "password",
        "account_number",
        "account_id",
    ):
        assert record.payload[key] == REDACTED, key
    assert record.payload["client_order_id"] == "keep-me"
    assert record.payload["symbol"] == "AMD"

    raw = record.canonical_bytes().decode("utf-8")
    for leaked in ("PKLIVEKEY", "shhh", "hunter2", "PA123456", "9f0c-private"):
        assert leaked not in raw
    # The hash covers the redacted form, so it is stable across replays.
    assert record.event_hash == record.computed_event_hash()


def test_redaction_is_recursive_through_lists_and_maps() -> None:
    cleaned = redact(
        {"outer": [{"api_key": "x"}, {"nested": {"password": "y", "keep": 1}}]}
    )

    assert cleaned["outer"][0]["api_key"] == REDACTED
    assert cleaned["outer"][1]["nested"]["password"] == REDACTED
    assert cleaned["outer"][1]["nested"]["keep"] == 1


def test_a_non_utc_timestamp_is_normalised_before_hashing() -> None:
    """This project works in ET, so an ET-aware event must just work.

    Hashing the pre-coercion value would make the record fail its own hash
    check with a misleading error.
    """

    from datetime import datetime, timezone

    eastern = timezone(timedelta(hours=-4))
    record = event(
        event_time=datetime(2026, 8, 2, 14, 30, 15, 123456, tzinfo=eastern),
        observed_at=datetime(2026, 8, 2, 14, 30, 15, 128456, tzinfo=eastern),
    )

    assert record.event_time.tzinfo == timezone.utc
    assert record.event_time.hour == 18
    assert record.event_hash == record.computed_event_hash()
    # The object path uses the normalised UTC instant, not the local wall clock.
    assert "/2026/08/02/paper/20260802T183015123456Z_" in record.object_name


def test_the_same_instant_in_two_zones_produces_one_identity() -> None:
    from datetime import datetime, timezone

    eastern = timezone(timedelta(hours=-4))
    utc_form = event(
        event_time=datetime(2026, 8, 2, 18, 30, 15, 123456, tzinfo=timezone.utc),
        observed_at=datetime(2026, 8, 2, 18, 30, 15, 128456, tzinfo=timezone.utc),
    )
    et_form = event(
        event_time=datetime(2026, 8, 2, 14, 30, 15, 123456, tzinfo=eastern),
        observed_at=datetime(2026, 8, 2, 14, 30, 15, 128456, tzinfo=eastern),
    )

    assert utc_form.object_name == et_form.object_name
    assert utc_form.event_hash == et_form.event_hash


def test_production_live_events_are_refused() -> None:
    with pytest.raises(ValueError, match="PRODUCTION_LIVE"):
        event(environment=RuntimeEnvironment.PRODUCTION_LIVE)


def test_observed_at_cannot_precede_event_time() -> None:
    with pytest.raises(ValueError, match="observed_at must not precede"):
        event(observed_at=EVENT_TIME - timedelta(seconds=1))


def test_postgres_persistence_status_is_carried_on_the_event() -> None:
    record = event(
        postgres_persistence_status=PostgresPersistenceStatus.RECONCILIATION_REQUIRED
    )

    assert (
        record.postgres_persistence_status
        is PostgresPersistenceStatus.RECONCILIATION_REQUIRED
    )
    # The status is part of the hashed evidence, not a mutable annotation.
    assert "RECONCILIATION_REQUIRED" in record.canonical_bytes().decode("utf-8")


def test_hash_covers_every_field_except_the_hash_itself() -> None:
    import json

    record = event()
    keys = set(json.loads(record.canonical_bytes(with_hash=False)))

    assert "event_hash" not in keys
    # Everything else is covered, including the predecessor linkage.
    assert keys == set(record.model_dump()) - {"event_hash"}
    for field in ("event_id", "sequence_no", "client_order_id", "observed_at"):
        assert field in keys
