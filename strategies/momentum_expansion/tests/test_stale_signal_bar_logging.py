"""The stale-signal-bar abort is routine (the dashboard's read-only preview
scan hits it every time it runs before the day's first fresh 4H bar lands) and
must not show up as a WARNING in the live server log.

2026-07-21 audit: ~80 "Momentum: ABORT stale signal bar" WARNING lines in one
session, virtually all from auto_trade=False preview calls, not real trading
decisions. The signal-decision audit record is still written either way --
only the log-visibility level changed.
"""
from __future__ import annotations

import json
import logging

import pandas as pd
import pytest

import strategies.momentum_expansion.live.runner as mr


class _FakeResult:
    def __init__(self, panel):
        self.panel = panel
        self.tickers_built = len(panel)
        self.tickers_requested = len(panel)
        self.build_seconds = 0.1


def _stale_panel() -> pd.DataFrame:
    old_ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=5)
    index = pd.MultiIndex.from_tuples([(old_ts, "AAA")], names=["timestamp", "ticker"])
    return pd.DataFrame({"expansion_score": [0.5]}, index=index)


def test_stale_bar_abort_logs_at_debug_not_warning(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(mr, "load_snapshot_for", lambda bar_ts: pd.DataFrame({"ticker": ["AAA"]}))
    monkeypatch.setattr(mr, "build_live_feature_panel_4h", lambda **k: _FakeResult(_stale_panel()))
    monkeypatch.setattr(mr, "assert_manifest_coverage", lambda **k: None)
    monkeypatch.setattr(mr, "DEFAULT_SIGNAL_AUDIT_LOG", tmp_path / "audit.jsonl")

    runner = mr.MomentumLiveRunner(auto_trade=False)

    with caplog.at_level(logging.DEBUG, logger="strategies.momentum_expansion.live.runner"):
        out = runner.evaluate_now()

    assert out == []
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and "stale signal bar" in r.message]
    assert warnings == []
    debugs = [r for r in caplog.records if r.levelno == logging.DEBUG and "stale signal bar" in r.message]
    assert len(debugs) == 1


def test_stale_bar_abort_still_writes_the_audit_record(monkeypatch, tmp_path):
    monkeypatch.setattr(mr, "load_snapshot_for", lambda bar_ts: pd.DataFrame({"ticker": ["AAA"]}))
    monkeypatch.setattr(mr, "build_live_feature_panel_4h", lambda **k: _FakeResult(_stale_panel()))
    monkeypatch.setattr(mr, "assert_manifest_coverage", lambda **k: None)
    audit_log = tmp_path / "audit.jsonl"
    monkeypatch.setattr(mr, "DEFAULT_SIGNAL_AUDIT_LOG", audit_log)

    runner = mr.MomentumLiveRunner(auto_trade=False)
    runner.evaluate_now()

    rows = [json.loads(line) for line in audit_log.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["skip_reason"] == "stale_signal_bar"
