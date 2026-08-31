from __future__ import annotations

import json
import queue
from dataclasses import replace as dc_replace
from datetime import datetime, timedelta, timezone

from strategies.intraday_structure.bar_archive import BarArchive, read_session
from strategies.intraday_structure.config import IntradayStructureConfig


NOW = datetime(2026, 8, 27, 14, 30, tzinfo=timezone.utc)  # 10:30 ET


def _payload(symbol="XYZ", minute=0, **over):
    row = {
        "symbol": symbol, "timestamp": (NOW + timedelta(minutes=minute)).isoformat(),
        "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 1000,
    }
    row.update(over)
    return row


def test_bars_land_in_the_session_file_they_belong_to(tmp_path) -> None:
    archive = BarArchive(tmp_path, flush_every=2)
    archive.record(_payload(minute=0))
    archive.record(_payload(minute=1))
    rows = read_session(archive.path_for("2026-08-27"))
    assert len(rows) == 2
    assert {r["symbol"] for r in rows} == {"XYZ"}


def test_the_two_clocks_are_kept_apart(tmp_path) -> None:
    """A bar that arrives late was not available at its own timestamp."""
    archive = BarArchive(tmp_path, flush_every=1)
    late = NOW + timedelta(minutes=5)
    archive.record(_payload(minute=0), arrival_at=late)
    row = read_session(archive.path_for("2026-08-27"))[0]
    assert row["timestamp"] != row["arrival_at"]
    assert datetime.fromisoformat(row["arrival_at"]) == late


def test_a_session_boundary_splits_the_files(tmp_path) -> None:
    archive = BarArchive(tmp_path, flush_every=100)
    archive.record(_payload(minute=0))
    # 2026-08-28 01:00 UTC is still 2026-08-27 in ET; 2026-08-28 14:30 UTC is not.
    archive.record({**_payload(), "timestamp": datetime(2026, 8, 28, 1, 0, tzinfo=timezone.utc).isoformat()})
    archive.record({**_payload(), "timestamp": datetime(2026, 8, 28, 14, 30, tzinfo=timezone.utc).isoformat()})
    archive.flush()
    assert len(read_session(archive.path_for("2026-08-27"))) == 2, "ET, not UTC, defines the session"
    assert len(read_session(archive.path_for("2026-08-28"))) == 1


def test_appending_never_rewrites_what_is_already_there(tmp_path) -> None:
    archive = BarArchive(tmp_path, flush_every=1)
    archive.record(_payload(minute=0))
    first = archive.path_for("2026-08-27").read_text(encoding="utf-8")
    archive.record(_payload(minute=1))
    assert archive.path_for("2026-08-27").read_text(encoding="utf-8").startswith(first)


def test_a_malformed_payload_is_skipped_not_raised(tmp_path) -> None:
    archive = BarArchive(tmp_path, flush_every=1)
    archive.record({"nonsense": True})
    archive.record({"symbol": "", "timestamp": NOW.isoformat()})
    archive.record(_payload(minute=0))
    assert len(read_session(archive.path_for("2026-08-27"))) == 1


def test_a_wedged_disk_bounds_the_buffer_and_counts_the_loss(tmp_path) -> None:
    archive = BarArchive(tmp_path, flush_every=10_000, max_buffer=5)
    for i in range(9):
        archive.record(_payload(minute=i))
    stats = archive.stats()
    assert stats.buffered == 5
    assert stats.dropped == 4, "loss must be counted, never silent"


def test_a_truncated_line_does_not_poison_the_read(tmp_path) -> None:
    archive = BarArchive(tmp_path, flush_every=1)
    archive.record(_payload(minute=0))
    with archive.path_for("2026-08-27").open("a", encoding="utf-8") as handle:
        handle.write('{"symbol": "XY')  # a crash mid-write
    assert len(read_session(archive.path_for("2026-08-27"))) == 1


def test_the_manifest_binds_bars_to_the_code_that_wrote_them(tmp_path) -> None:
    archive = BarArchive(tmp_path, flush_every=1)
    # Explicit arrival times: the default is wall-clock now(), which would make
    # every bar in a fixture dated 2026-08-27 look late.
    archive.record(_payload(minute=0), arrival_at=NOW + timedelta(seconds=1))
    archive.record(_payload(minute=0), arrival_at=NOW + timedelta(seconds=1))  # a duplicate
    archive.record(_payload(minute=1), arrival_at=NOW + timedelta(minutes=9))  # late
    path = archive.write_manifest("2026-08-27", extra={"engine_version": "v1"})
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["bar_count"] == 3
    assert manifest["duplicates"] == 1
    assert manifest["late_bars"] == 1
    assert manifest["engine_version"] == "v1"
    assert "git_revision" in manifest


def test_the_runner_archives_what_arrived_even_if_the_engine_rejects_it(tmp_path) -> None:
    from strategies.intraday_structure.runner import IntradayStructureRunner

    config = dc_replace(
        IntradayStructureConfig(enabled=True),
        state_path=str(tmp_path / "s.json"),
        transition_log_path=str(tmp_path / "t.jsonl"),
        signal_path=str(tmp_path / "sig.json"),
        ledger_path=str(tmp_path / "l.jsonl"),
        abstention_path=str(tmp_path / "a.jsonl"),
        bar_archive_root=str(tmp_path / "archive"),
        archive_bars=True,
    )
    runner = IntradayStructureRunner(config, queue.Queue())
    assert runner.bar_archive is not None
    # A bar the engine would refuse (negative price) is still feed evidence.
    runner.bar_archive.record(_payload(minute=0, close=-1.0))
    runner.bar_archive.flush()
    assert len(read_session(runner.bar_archive.path_for("2026-08-27"))) == 1


def test_archiving_can_be_turned_off(tmp_path) -> None:
    from strategies.intraday_structure.runner import IntradayStructureRunner

    config = dc_replace(
        IntradayStructureConfig(enabled=True), archive_bars=False,
        state_path=str(tmp_path / "s.json"),
        transition_log_path=str(tmp_path / "t.jsonl"),
        signal_path=str(tmp_path / "sig.json"),
    )
    assert IntradayStructureRunner(config, queue.Queue()).bar_archive is None
