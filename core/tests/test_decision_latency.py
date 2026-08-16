"""Tests for SPY decision-latency instrumentation.

This exists to investigate the 2026-07-30 lag (2.4-19.9 min between a 10-minute
bar closing and its decision being recorded, uncorrelated with CPU load). The
instrumentation is diagnostic, so the hard requirement is that it can never
disturb the trading path it measures.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from core.API.Alpaca_API.runners.decision_latency import DecisionLatency

BUCKET_START = datetime(2026, 7, 30, 14, 50, tzinfo=timezone.utc)  # 10:50 ET
BUCKET_CLOSE = BUCKET_START + timedelta(minutes=10)


@pytest.fixture()
def tracker(tmp_path):
    return DecisionLatency(log_path=tmp_path / "decision-latency.jsonl", interval_minutes=10)


def _records(tracker):
    path = tracker._log_path
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_disabled_tracker_writes_nothing_and_never_raises(tmp_path):
    tracker = DecisionLatency(log_path=None, interval_minutes=10)

    tracker.on_1m_bar("SPY", {"timestamp": BUCKET_START})
    tracker.on_bucket_closed("SPY", {"timestamp": BUCKET_START}, {"timestamp": BUCKET_CLOSE})
    with tracker.stage("inference"):
        pass
    tracker.emit("SPY", {"timestamp": BUCKET_START})

    assert not tracker.enabled
    assert list(tmp_path.iterdir()) == []


def test_records_the_close_detection_lag(tmp_path):
    """The bucket cannot close until the next bucket's first bar arrives.

    Replays the worst real case from 2026-07-30: the 10:50 bar closed at 11:00
    and its decision was not recorded until 11:19:56.
    """
    late_arrival = BUCKET_CLOSE + timedelta(minutes=18)
    tracker = DecisionLatency(
        log_path=tmp_path / "decision-latency.jsonl",
        interval_minutes=10,
        clock=lambda: late_arrival,
    )
    tracker.on_bucket_closed(
        "SPY", {"timestamp": BUCKET_START}, {"timestamp": BUCKET_CLOSE}
    )
    tracker.emit("SPY", {"timestamp": BUCKET_START})

    record = _records(tracker)[0]
    assert record["close_detection_lag_sec"] == pytest.approx(18 * 60)
    assert record["total_lag_after_close_sec"] == pytest.approx(18 * 60)
    assert record["bar_close_utc"].startswith("2026-07-30T15:00")


def test_a_gappy_feed_is_visible_in_the_bar_count(tracker):
    """Fewer 1m bars than the interval is the signature of a sparse feed."""
    for _ in range(4):  # only 4 of an expected 10 minutes printed
        tracker.on_1m_bar("SPY", {"timestamp": BUCKET_START})
    tracker.on_1m_bar("SPY", {"timestamp": BUCKET_CLOSE})  # the trigger bar
    tracker.on_bucket_closed("SPY", {"timestamp": BUCKET_START}, {"timestamp": BUCKET_CLOSE})
    tracker.emit("SPY", {"timestamp": BUCKET_START})

    record = _records(tracker)[0]
    assert record["bars_in_bucket"] == 4
    assert record["expected_bars_in_bucket"] == 10


def test_the_trigger_bar_counts_toward_the_next_bucket_not_the_one_it_closes(tracker):
    """A full 10-bar bucket must read 10, not 11 — and the trigger carries over."""
    for _ in range(10):  # the bucket's own ten 1-minute bars
        tracker.on_1m_bar("SPY", {"timestamp": BUCKET_START})
    tracker.on_1m_bar("SPY", {"timestamp": BUCKET_CLOSE})  # first bar of the NEXT bucket
    tracker.on_bucket_closed("SPY", {"timestamp": BUCKET_START}, {"timestamp": BUCKET_CLOSE})
    tracker.emit("SPY", {"timestamp": BUCKET_START})

    for _ in range(2):  # two more, so the next bucket holds 3 including the trigger
        tracker.on_1m_bar("SPY", {"timestamp": BUCKET_CLOSE})
    tracker.on_1m_bar("SPY", {"timestamp": BUCKET_CLOSE + timedelta(minutes=10)})
    tracker.on_bucket_closed(
        "SPY", {"timestamp": BUCKET_CLOSE}, {"timestamp": BUCKET_CLOSE + timedelta(minutes=10)}
    )
    tracker.emit("SPY", {"timestamp": BUCKET_CLOSE})

    counts = [r["bars_in_bucket"] for r in _records(tracker)]
    assert counts == [10, 3], "each bucket must count only its own bars"


def test_stage_timings_are_recorded_and_then_reset(tracker):
    with tracker.stage("inference"):
        pass
    with tracker.stage("order_policy"):
        pass
    tracker.emit("SPY", {"timestamp": BUCKET_START})
    tracker.emit("SPY", {"timestamp": BUCKET_START})

    first, second = _records(tracker)
    assert set(first["stages_sec"]) == {"inference", "order_policy"}
    assert second["stages_sec"] == {}, "timings must not leak into the next decision"


def test_a_raising_stage_still_records_its_time_and_propagates(tracker):
    """Instrumentation must not swallow a real error from the trading path."""
    with pytest.raises(ValueError):
        with tracker.stage("inference"):
            raise ValueError("model blew up")

    tracker.emit("SPY", {"timestamp": BUCKET_START})
    assert "inference" in _records(tracker)[0]["stages_sec"]


def test_emit_without_a_preceding_close_still_writes(tracker):
    """A decision from a flushed session-end bucket has no trigger bar."""
    tracker.emit("SPY", {"timestamp": BUCKET_START})

    record = _records(tracker)[0]
    assert record["symbol"] == "SPY"
    assert "close_detection_lag_sec" not in record


def test_unwritable_log_path_does_not_raise(tmp_path):
    """A diagnostics failure must never take the trading loop down."""
    blocked = tmp_path / "file.txt"
    blocked.write_text("not a directory")
    tracker = DecisionLatency(log_path=blocked / "nested" / "x.jsonl", interval_minutes=10)

    tracker.on_bucket_closed("SPY", {"timestamp": BUCKET_START}, {"timestamp": BUCKET_CLOSE})
    tracker.emit("SPY", {"timestamp": BUCKET_START})  # must not raise


def test_garbage_timestamps_do_not_raise(tracker):
    tracker.on_bucket_closed("SPY", {"timestamp": "not-a-date"}, {"timestamp": None})
    tracker.emit("SPY", {"timestamp": object()})

    assert _records(tracker), "a record should still be written"

