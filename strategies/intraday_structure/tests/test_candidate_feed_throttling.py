"""I/O throttling for the candidate feeds polled inside the bar-consumer loop.

Regression coverage for the 2026-07-20 incident: `IntradayStructureRunner._run`
calls `AuditCandidateFeed.poll()` and `DealerRankingCandidateFeed.poll()` once
per bar pulled off the shared stream queue (many times per second during RTH).
Neither feed rate-limited its disk I/O, so the consumer fell behind the
producer and the shared bar queue backed up all session, dropping ~27% of
bars by end-of-day. Both feeds now skip their expensive re-scan when nothing
useful could have changed since the last one.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from strategies.intraday_structure.candidate_sources import (
    AuditCandidateFeed,
    DealerRankingCandidateFeed,
)


def _write_signal_decision(path, *, bar: str, ticker: str = "AAA") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"event": "signal_decision", "bar": "%s", "targets": ["%s"], '
        '"signal_audits": {"%s": {"side": "long", "score": 0.5}}}\n' % (bar, ticker, ticker)
    )


def test_audit_feed_throttles_repeated_polls_within_interval(tmp_path):
    source = tmp_path / "meta_ranker" / "live_signal_audit.jsonl"
    _write_signal_decision(source, bar="2026-07-20T14:00:00+00:00")
    feed = AuditCandidateFeed(sources={"meta_ranker": source}, swing_audit_root=tmp_path / "swing_audit",
                              min_poll_interval_seconds=60.0)

    first = feed.poll()
    assert {c.ticker for c in first} == {"AAA"}

    # A second call immediately after must not re-scan (and, since the first
    # call already registered AAA into _seen, would return [] either way --
    # the point of this test is that it returns fast without touching disk).
    second = feed.poll()
    assert second == []


def test_audit_feed_scans_again_once_the_interval_elapses(tmp_path, monkeypatch):
    source = tmp_path / "meta_ranker" / "live_signal_audit.jsonl"
    _write_signal_decision(source, bar="2026-07-20T14:00:00+00:00", ticker="AAA")
    feed = AuditCandidateFeed(sources={"meta_ranker": source}, swing_audit_root=tmp_path / "swing_audit",
                              min_poll_interval_seconds=5.0)

    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    first = feed.poll()
    assert {c.ticker for c in first} == {"AAA"}

    # New bar written for a different ticker; within the throttle window the
    # feed must not pick it up yet.
    _write_signal_decision(source, bar="2026-07-20T18:00:00+00:00", ticker="BBB")
    clock[0] += 1.0
    assert feed.poll() == []

    # Past the throttle window, the fresh event is picked up.
    clock[0] += 10.0
    third = feed.poll()
    assert {c.ticker for c in third} == {"BBB"}


def test_dealer_ranking_feed_skips_unchanged_parquet_without_reparsing(tmp_path, monkeypatch):
    captured = datetime(2026, 7, 20, 19, 45, tzinfo=timezone.utc)
    path = tmp_path / "rankings.parquet"
    pd.DataFrame([
        {"symbol": "TOP", "captured_at": captured, "dealer_swing_rank": 1, "dealer_change_intensity_rank": 90},
    ]).to_parquet(path, index=False)
    feed = DealerRankingCandidateFeed(path, top_structural=5, top_change=5, max_age_hours=30)

    first = feed.poll(now=captured + timedelta(hours=1))
    assert {c.ticker for c in first} == {"TOP"}

    read_calls = []
    real_read_parquet = pd.read_parquet
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: (read_calls.append(1), real_read_parquet(*a, **k))[1])

    # File unchanged (same mtime/size) -> must short-circuit before read_parquet.
    second = feed.poll(now=captured + timedelta(hours=2))
    assert second == []
    assert read_calls == []


def test_dealer_ranking_feed_reparses_after_file_changes(tmp_path):
    captured = datetime(2026, 7, 20, 19, 45, tzinfo=timezone.utc)
    path = tmp_path / "rankings.parquet"
    pd.DataFrame([
        {"symbol": "TOP", "captured_at": captured, "dealer_swing_rank": 1, "dealer_change_intensity_rank": 90},
    ]).to_parquet(path, index=False)
    feed = DealerRankingCandidateFeed(path, top_structural=5, top_change=5, max_age_hours=30)
    assert {c.ticker for c in feed.poll(now=captured + timedelta(hours=1))} == {"TOP"}

    new_captured = captured + timedelta(hours=4)
    time.sleep(0.01)  # ensure a distinct mtime on fast filesystems
    pd.DataFrame([
        {"symbol": "NEXT", "captured_at": new_captured, "dealer_swing_rank": 1, "dealer_change_intensity_rank": 90},
    ]).to_parquet(path, index=False)

    refreshed = feed.poll(now=new_captured + timedelta(hours=1))
    assert {c.ticker for c in refreshed} == {"NEXT"}
