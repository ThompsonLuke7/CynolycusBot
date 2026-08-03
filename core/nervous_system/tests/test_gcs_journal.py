"""GCS journal immutability with a fake storage client (Task 20).

No test touches ADC, a network, or a real bucket.
"""

from __future__ import annotations

import hashlib

import pytest

from core.nervous_system.execution.journal import (
    JournalBackend,
    JournalConflict,
    JournalUnavailable,
    JournalWriteStatus,
    GCSImmutableJournal,
)
from core.nervous_system.tests.fixtures.journal_events import event


BUCKET = "cynolycus-journal-test"


class FakePreconditionFailed(Exception):
    """Stands in for google.api_core.exceptions.PreconditionFailed."""

    code = 412


class FakeBlob:
    def __init__(self, store: dict[str, bytes], name: str, log: list) -> None:
        self._store = store
        self.name = name
        self._log = log

    def upload_from_string(self, data, content_type=None, if_generation_match=None):
        self._log.append(
            ("upload", self.name, content_type, if_generation_match)
        )
        payload = data if isinstance(data, bytes) else str(data).encode("utf-8")
        if if_generation_match == 0 and self.name in self._store:
            raise FakePreconditionFailed(f"{self.name} already exists")
        self._store[self.name] = payload

    def download_as_bytes(self) -> bytes:
        self._log.append(("download", self.name, None, None))
        if self.name not in self._store:
            raise FileNotFoundError(self.name)
        return self._store[self.name]


class FakeBucket:
    def __init__(self, store: dict[str, bytes], log: list) -> None:
        self._store = store
        self._log = log

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self._store, name, self._log)


class FakeClient:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.log: list = []

    def bucket(self, name: str) -> FakeBucket:
        self.log.append(("bucket", name, None, None))
        return FakeBucket(self.store, self.log)

    def list_blobs(self, bucket_name: str, prefix: str = ""):
        self.log.append(("list", bucket_name, prefix, None))
        return [
            FakeBlob(self.store, name, self.log)
            for name in sorted(self.store)
            if name.startswith(prefix)
        ]


def journal(client: FakeClient) -> GCSImmutableJournal:
    return GCSImmutableJournal(
        client, BUCKET, precondition_errors=(FakePreconditionFailed,)
    )


def test_object_name_matches_the_shared_identity() -> None:
    client = FakeClient()
    record = event()
    receipt = journal(client).write(record)

    assert receipt.locator.object_name == record.object_name
    assert receipt.locator.uri == f"gs://{BUCKET}/{record.object_name}"
    assert receipt.locator.backend is JournalBackend.GCS
    assert record.object_name.startswith("execution_journal/v1/2026/08/02/paper/")


def test_upload_uses_the_generation_precondition() -> None:
    client = FakeClient()
    journal(client).write(event())

    uploads = [entry for entry in client.log if entry[0] == "upload"]
    assert uploads[0][2] == "application/json"
    assert uploads[0][3] == 0, "if_generation_match=0 makes the object immutable"


def test_the_write_path_never_lists_the_bucket() -> None:
    client = FakeClient()
    sink = journal(client)
    sink.write(event())
    sink.write(event())  # idempotent retry

    assert not any(entry[0] == "list" for entry in client.log)


def test_identical_content_is_idempotent_after_hash_verification() -> None:
    client = FakeClient()
    sink = journal(client)
    record = event()

    first = sink.write(record)
    second = sink.write(record)

    assert first.status is JournalWriteStatus.WRITTEN
    assert second.status is JournalWriteStatus.IDEMPOTENT
    assert first.content_hash == second.content_hash
    # The reconcile path must actually read the stored bytes back.
    assert any(entry[0] == "download" for entry in client.log)


def test_conflicting_content_under_one_identity_fails() -> None:
    client = FakeClient()
    sink = journal(client)
    sink.write(event())

    with pytest.raises(JournalConflict, match="different bytes"):
        sink.write(event(payload={"symbol": "NVDA"}))


def test_stored_bytes_are_never_replaced_on_conflict() -> None:
    client = FakeClient()
    sink = journal(client)
    record = event()
    sink.write(record)
    original = dict(client.store)

    with pytest.raises(JournalConflict):
        sink.write(event(payload={"tampered": True}))

    assert client.store == original


def test_content_hash_matches_the_uploaded_bytes() -> None:
    client = FakeClient()
    record = event()
    receipt = journal(client).write(record)

    assert receipt.content_hash == hashlib.sha256(
        client.store[record.object_name]
    ).hexdigest()


def test_a_transport_failure_is_unavailable_not_silent() -> None:
    class BrokenClient(FakeClient):
        def bucket(self, name: str):
            class Broken:
                def blob(self, _name: str):
                    class B:
                        name = _name

                        def upload_from_string(self, *a, **kw):
                            raise ConnectionError("bucket unreachable")

                    return B()

            return Broken()

    with pytest.raises(JournalUnavailable, match="GCS upload failed"):
        journal(BrokenClient()).write(event())


def test_precondition_detected_without_a_declared_error_type() -> None:
    """A 412 is recognised even when the caller injects no error class."""

    client = FakeClient()
    sink = GCSImmutableJournal(client, BUCKET)
    record = event()
    sink.write(record)

    assert sink.write(record).status is JournalWriteStatus.IDEMPOTENT


def test_read_and_iteration_round_trip() -> None:
    client = FakeClient()
    sink = journal(client)
    record = event()
    receipt = sink.write(record)

    assert sink.read(receipt.locator) == record
    assert [item.event_id for item in sink.iter_events(account_id="paper")] == [
        record.event_id
    ]
    assert list(sink.iter_events(account_id="nobody")) == []
