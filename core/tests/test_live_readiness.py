from __future__ import annotations

import json
from datetime import datetime, timezone

from core.live_readiness import (
    filter_entry_orders_for_readiness,
    readiness_status,
    write_readiness_success,
)


def test_filter_entry_orders_skips_buys_without_stamp(monkeypatch, tmp_path):
    monkeypatch.delenv("CYNOLYCUS_READINESS_REQUIRED", raising=False)
    missing = tmp_path / "missing.json"
    monkeypatch.setattr("core.live_readiness.DEFAULT_READINESS_PATH", missing)
    managed = {
        "ABC": {"route": "equity", "symbol": "ABC", "runs_held": 0},
        "OLD": {"route": "equity", "symbol": "OLD", "runs_held": 2},
    }
    plan = [("ABC", "buy", 100, "entry", "equity"), ("OLD", "sell", 100, "exit", "equity")]

    kept, skipped, reason = filter_entry_orders_for_readiness(plan, new_managed=managed, max_age_hours=1)

    assert kept == [("OLD", "sell", 100, "exit", "equity")]
    assert skipped == {"ABC"}
    assert "missing readiness stamp" in reason
    assert "ABC" not in managed
    assert "OLD" in managed


def test_filter_entry_orders_allows_buys_with_fresh_stamp(monkeypatch, tmp_path):
    monkeypatch.delenv("CYNOLYCUS_READINESS_REQUIRED", raising=False)
    stamp = tmp_path / "latest_success.json"
    write_readiness_success(job="test", path=stamp)
    monkeypatch.setattr("core.live_readiness.DEFAULT_READINESS_PATH", stamp)
    plan = [("ABC", "buy", 100, "entry", "equity")]

    kept, skipped, reason = filter_entry_orders_for_readiness(plan, max_age_hours=1)

    assert kept == plan
    assert skipped == set()
    assert "OK" in reason


def test_filter_entry_orders_blocks_stale_stamp(monkeypatch, tmp_path):
    monkeypatch.delenv("CYNOLYCUS_READINESS_REQUIRED", raising=False)
    stamp = tmp_path / "latest_success.json"
    stamp.write_text(json.dumps({
        "job": "test",
        "status": "success",
        "completed_at_utc": datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat(),
    }))
    monkeypatch.setattr("core.live_readiness.DEFAULT_READINESS_PATH", stamp)

    kept, skipped, reason = filter_entry_orders_for_readiness(
        [("ABC", "buy", 100, "entry", "equity")],
        max_age_hours=1,
    )

    assert kept == []
    assert skipped == {"ABC"}
    assert "old" in reason


def test_readiness_requires_refresh_after_prior_trading_session(tmp_path):
    stamp = tmp_path / "latest_success.json"
    stamp.write_text(json.dumps({
        "job": "test",
        "status": "success",
        # Sunday is valid for Monday, but not for Tuesday after Monday traded.
        "completed_at_utc": "2026-07-12T17:26:44+00:00",
    }))

    monday_ok, monday_reason, _ = readiness_status(
        path=stamp,
        now=datetime(2026, 7, 13, 13, 0, tzinfo=timezone.utc),
    )
    tuesday_ok, tuesday_reason, _ = readiness_status(
        path=stamp,
        now=datetime(2026, 7, 14, 13, 0, tzinfo=timezone.utc),
    )

    assert monday_ok is True
    assert "OK" in monday_reason
    assert tuesday_ok is False
    assert "2026-07-13 16:00 ET" in tuesday_reason
