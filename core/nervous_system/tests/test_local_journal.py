"""Local journal durability and immutability (Task 20)."""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path

import pytest

from core.nervous_system.execution.journal import (
    JOURNAL_PREFIX,
    JOURNAL_VERSION,
    JournalBackend,
    JournalConflict,
    JournalWriteStatus,
    LocalAtomicJournal,
)
from core.nervous_system.tests.fixtures.journal_events import EVENT_TIME, event


def journal(tmp_path: Path) -> LocalAtomicJournal:
    return LocalAtomicJournal(tmp_path)


def test_event_is_one_canonical_file_at_the_dated_account_path(tmp_path: Path) -> None:
    record = event()
    receipt = journal(tmp_path).write(record)

    expected = (
        tmp_path
        / JOURNAL_PREFIX
        / JOURNAL_VERSION
        / "2026"
        / "08"
        / "02"
        / "paper"
        / f"20260802T183015123456Z_{record.event_id}.json"
    )
    assert expected.exists()
    assert receipt.locator.uri == str(expected)
    assert receipt.locator.backend is JournalBackend.LOCAL
    assert receipt.status is JournalWriteStatus.WRITTEN
    assert json.loads(expected.read_text())["event_id"] == str(record.event_id)


def test_content_hash_matches_the_written_bytes(tmp_path: Path) -> None:
    import hashlib

    record = event()
    receipt = journal(tmp_path).write(record)
    written = Path(receipt.locator.uri).read_bytes()

    assert receipt.content_hash == hashlib.sha256(written).hexdigest()
    assert written == record.canonical_bytes()


def test_writer_fsyncs_the_file_and_the_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[str] = []
    real_fsync = os.fsync

    def tracking_fsync(fd: int) -> None:
        synced.append("fd")
        real_fsync(fd)

    monkeypatch.setattr(
        "core.nervous_system.execution.journal.os.fsync", tracking_fsync
    )
    journal(tmp_path).write(event())

    # One for the record, one for the directory entry.
    assert len(synced) >= 2


def test_failure_before_install_leaves_no_final_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = event()
    sink = journal(tmp_path)

    def exploding_link(src: str, dst: str) -> None:
        raise OSError("disk died before install")

    monkeypatch.setattr(
        "core.nervous_system.execution.journal.os.link", exploding_link
    )
    with pytest.raises(OSError):
        sink.write(record)

    final = sink.path_for(record)
    assert not final.exists(), "a torn write must never appear under the final name"
    leftovers = list(final.parent.glob("*"))
    assert leftovers == [], "the temp file must be cleaned up"


def test_repeating_the_same_event_is_idempotent(tmp_path: Path) -> None:
    sink = journal(tmp_path)
    record = event()

    first = sink.write(record)
    second = sink.write(record)

    assert first.status is JournalWriteStatus.WRITTEN
    assert second.status is JournalWriteStatus.IDEMPOTENT
    assert first.content_hash == second.content_hash
    assert first.locator == second.locator


def test_conflicting_content_for_the_same_identity_raises(tmp_path: Path) -> None:
    sink = journal(tmp_path)
    record = event()
    sink.write(record)

    # Same identity and timestamp, different payload: this is corruption, not
    # a retry, and must never overwrite the installed record.
    impostor = event(payload={"symbol": "NVDA", "qty": 9_999})
    assert impostor.object_name == record.object_name

    with pytest.raises(JournalConflict, match="different bytes"):
        sink.write(impostor)

    stored = json.loads(sink.path_for(record).read_text())
    assert stored["payload"]["symbol"] == "AMD"


def test_installed_records_are_never_overwritten_by_replace(tmp_path: Path) -> None:
    """os.replace would clobber; the writer must use a create-exclusive link."""

    sink = journal(tmp_path)
    record = event()
    sink.write(record)
    original = sink.path_for(record).read_bytes()

    with pytest.raises(JournalConflict):
        sink.write(event(payload={"tampered": True}))

    assert sink.path_for(record).read_bytes() == original


def test_read_uses_the_exact_locator(tmp_path: Path) -> None:
    sink = journal(tmp_path)
    record = event()
    receipt = sink.write(record)

    assert sink.read(receipt.locator) == record


def test_iter_events_is_ordered_and_filtered_by_account(tmp_path: Path) -> None:
    sink = journal(tmp_path)
    first = event(suffix="a", event_time=EVENT_TIME)
    second = event(suffix="b", event_time=EVENT_TIME + timedelta(minutes=1))
    other = event(suffix="c", account_id="other_paper")
    for record in (second, first, other):
        sink.write(record)

    found = list(sink.iter_events(account_id="paper"))
    assert [item.event_id for item in found] == [first.event_id, second.event_id]

    after = list(sink.iter_events(account_id="paper", after=EVENT_TIME))
    assert [item.event_id for item in after] == [second.event_id]


def test_records_from_different_days_land_in_different_directories(
    tmp_path: Path,
) -> None:
    sink = journal(tmp_path)
    sink.write(event(suffix="d1"))
    sink.write(event(suffix="d2", event_time=EVENT_TIME + timedelta(days=1)))

    base = tmp_path / JOURNAL_PREFIX / JOURNAL_VERSION / "2026" / "08"
    assert (base / "02" / "paper").is_dir()
    assert (base / "03" / "paper").is_dir()
