"""SPY archives the exact feature row its models scored on each 15m decision."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from core.API.Alpaca_API.inference.live_inference import LiveInferenceEngine
from core.API.Alpaca_API.runners.live_runner import archive_spy_decision
from core.decision_state_archive import load_decision_state


class _Agent:
    def last_feature_row(self):
        return pd.Series({"rsi_14": 55.0, "vwap_dist": -0.2, "regime": "trend", "flag": True})


def test_engine_exposes_the_agents_scored_row():
    eng = LiveInferenceEngine(agent=_Agent())
    assert eng.last_feature_row()["rsi_14"] == 55.0
    assert LiveInferenceEngine().last_feature_row() is None


def test_archive_spy_decision_writes_one_row():
    bar_ts = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=1)
    eng = LiveInferenceEngine(agent=_Agent())
    archive_spy_decision(
        inference=eng, symbol="SPY", bar15={"timestamp": bar_ts, "close": 600.0}, run_id="s" * 32,
        score=0.71, raw_action=1.0, raw_pos=1, exec_pos=0, gate_status="latched", probs={"p_enter_long": 0.71},
        thresholds={"enter_long": 0.6}, interval_minutes=15, inference_start=datetime.now(timezone.utc),
    )
    df = load_decision_state(module="spy_day_trader")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["ticker"] == "SPY" and row["score"] == 0.71 and row["side"] == "long"
    assert not row["selected"] and row["gate_reason"] == "execution_latch"
    assert row["f__rsi_14"] == 55.0 and row["f__flag"] == 1.0 and "f__regime" not in df.columns
    assert row["x__prob__p_enter_long"] == 0.71 and row["x__thr__enter_long"] == 0.6
    assert row["action"] == "raw=+1 exec=+0 gate=latched"


def test_missing_feature_row_still_archives_the_decision():
    bar_ts = datetime.now(timezone.utc) - timedelta(minutes=1)
    archive_spy_decision(
        inference=LiveInferenceEngine(), symbol="SPY", bar15={"timestamp": bar_ts}, run_id="t" * 32,
        score=None, raw_action=0.0, raw_pos=0, exec_pos=0, gate_status="flat", probs={}, thresholds={},
        interval_minutes=15, inference_start=datetime.now(timezone.utc),
    )
    df = load_decision_state(module="spy_day_trader")
    assert len(df) == 1 and df.iloc[0]["side"] == "flat"
