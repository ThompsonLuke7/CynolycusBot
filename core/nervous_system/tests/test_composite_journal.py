"""Composite journal fan-out, degradation, and retry convergence (Task 20)."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.nervous_system.execution.journal import (
    CompositeExecutionJournal,
    CompositeStatus,
    JournalBackend,
    JournalConflict,
    JournalUnavailable,
    JournalWriteStatus,
    LocalAtomicJournal,
)
from core.nervous_system.tests.fixtures.journal_events import event
from core.nervous_system.tests.test_gcs_journal import (
    BUCKET,
    FakeClient,
    FakePreconditionFailed,
)
from core.nervous_system.execution.journal import GCSImmutableJournal


class BrokenSink:
    """A sink that always fails, without ever storing anything."""

    backend = JournalBackend.GCS

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or JournalUnavailable("sink offline")
        self.attempts = 0

    def write(self, event_):
        self.attempts += 1
        raise self.error


def local(tmp_path: Path) -> LocalAtomicJournal:
    return LocalAtomicJournal(tmp_path)


def gcs(client: FakeClient) -> GCSImmutableJournal:
    return GCSImmutableJournal(
        client, BUCKET, precondition_errors=(FakePreconditionFailed,)
    )


def test_all_required_sinks_succeeding_is_durable(tmp_path: Path) -> None:
    client = FakeClient()
    composite = CompositeExecutionJournal(required=(local(tmp_path), gcs(client)))

    result = composite.write(event())

    assert result.status is CompositeStatus.DURABLE
    assert result.is_durable
    assert len(result.receipts) == 2
    assert {receipt.backend for receipt in result.receipts} == {
        JournalBackend.LOCAL,
        JournalBackend.GCS,
    }
    assert result.failures == ()


def test_a_failed_required_sink_is_failed_not_degraded(tmp_path: Path) -> None:
    broken = BrokenSink()
    composite = CompositeExecutionJournal(required=(local(tmp_path), broken))

    result = composite.write(event())

    assert result.status is CompositeStatus.FAILED
    assert result.is_durable is False
    assert any("sink offline" in failure for failure in result.failures)
    # The sink that succeeded is never rolled back.
    assert len(result.receipts) == 1
    assert result.receipts[0].backend is JournalBackend.LOCAL


def test_a_required_sink_failure_is_never_reported_as_merely_degraded(
    tmp_path: Path,
) -> None:
    """DEGRADED means "durable but incomplete"; a missing required sink is not."""

    composite = CompositeExecutionJournal(
        required=(local(tmp_path), BrokenSink()), optional=()
    )
    result = composite.write(event())

    assert result.status is CompositeStatus.FAILED
    assert result.status is not CompositeStatus.DEGRADED
    assert result.is_durable is False


def test_a_failed_optional_sink_is_degraded_but_durable(tmp_path: Path) -> None:
    composite = CompositeExecutionJournal(
        required=(local(tmp_path),), optional=(BrokenSink(),)
    )

    result = composite.write(event())

    assert result.status is CompositeStatus.DEGRADED
    # Every required sink succeeded, so the evidence is durable where it must
    # be; the optional failure is reported, not treated as a loss.
    assert result.is_durable is True
    assert any("optional BrokenSink" in failure for failure in result.failures)
    assert len(result.receipts) == 1


def test_local_required_in_development(tmp_path: Path) -> None:
    """Development requires the local sink; GCS is not configured at all."""

    composite = CompositeExecutionJournal(required=(local(tmp_path),))
    result = composite.write(event())

    assert result.status is CompositeStatus.DURABLE
    assert result.receipts[0].backend is JournalBackend.LOCAL


def test_gcs_required_with_optional_local_in_qa_cloud_run(tmp_path: Path) -> None:
    """On Cloud Run the container filesystem is ephemeral, so GCS is required."""

    client = FakeClient()
    composite = CompositeExecutionJournal(
        required=(gcs(client),), optional=(local(tmp_path),)
    )

    result = composite.write(event())

    assert result.status is CompositeStatus.DURABLE
    assert len(result.receipts) == 2


def test_an_ephemeral_local_failure_does_not_block_a_cloud_run_write() -> None:
    client = FakeClient()
    composite = CompositeExecutionJournal(
        required=(gcs(client),),
        optional=(LocalAtomicJournal("/nonexistent-root-for-test"),),
    )

    result = composite.write(event())

    assert result.status is CompositeStatus.DEGRADED
    assert result.is_durable is True, (
        "an ephemeral local sink failing must not block a Cloud Run entry"
    )
    assert any(
        receipt.backend is JournalBackend.GCS for receipt in result.receipts
    )


def test_retrying_the_same_event_converges_without_duplicates(tmp_path: Path) -> None:
    client = FakeClient()
    composite = CompositeExecutionJournal(required=(local(tmp_path), gcs(client)))
    record = event()

    first = composite.write(record)
    second = composite.write(record)

    assert first.status is CompositeStatus.DURABLE
    assert second.status is CompositeStatus.IDEMPOTENT
    assert all(
        receipt.status is JournalWriteStatus.IDEMPOTENT for receipt in second.receipts
    )
    # Exactly one record in each sink.
    assert len(list(local(tmp_path).iter_events(account_id="paper"))) == 1
    assert len(client.store) == 1


def test_partial_success_then_retry_completes_the_missing_sink(
    tmp_path: Path,
) -> None:
    """A sink that succeeded first time stays put; the retry fills the gap."""

    client = FakeClient()
    record = event()
    failing = CompositeExecutionJournal(
        required=(local(tmp_path), BrokenSink())
    )
    assert failing.write(record).status is CompositeStatus.FAILED

    recovered = CompositeExecutionJournal(required=(local(tmp_path), gcs(client)))
    result = recovered.write(record)

    assert result.status is CompositeStatus.DEGRADED or result.is_durable
    statuses = {receipt.backend: receipt.status for receipt in result.receipts}
    assert statuses[JournalBackend.LOCAL] is JournalWriteStatus.IDEMPOTENT
    assert statuses[JournalBackend.GCS] is JournalWriteStatus.WRITTEN
    assert len(list(local(tmp_path).iter_events(account_id="paper"))) == 1


def test_conflicting_content_reports_conflict_not_failure(tmp_path: Path) -> None:
    composite = CompositeExecutionJournal(required=(local(tmp_path),))
    composite.write(event())

    result = composite.write(event(payload={"symbol": "NVDA"}))

    assert result.status is CompositeStatus.CONFLICT
    assert result.is_durable is False
    assert any("different bytes" in failure for failure in result.failures)


def test_a_conflict_in_an_optional_sink_still_surfaces(tmp_path: Path) -> None:
    client = FakeClient()
    gcs_sink = gcs(client)
    gcs_sink.write(event())

    composite = CompositeExecutionJournal(
        required=(local(tmp_path),), optional=(gcs_sink,)
    )
    result = composite.write(event(payload={"symbol": "NVDA"}))

    assert result.status is CompositeStatus.CONFLICT


def test_at_least_one_required_sink_is_mandatory() -> None:
    with pytest.raises(ValueError, match="at least one required"):
        CompositeExecutionJournal(required=(), optional=(BrokenSink(),))


def test_a_successful_sink_is_never_rolled_back(tmp_path: Path) -> None:
    sink = local(tmp_path)
    record = event()
    composite = CompositeExecutionJournal(required=(sink, BrokenSink()))

    composite.write(record)

    assert sink.path_for(record).exists(), (
        "an immutable sink that succeeded must keep its record even when a "
        "sibling sink failed"
    )
