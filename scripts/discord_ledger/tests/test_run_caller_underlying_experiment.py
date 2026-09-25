from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.discord_ledger.run_caller_underlying_experiment import _forward_return, _feature_row, _outcome_summary


def _bars() -> pd.DataFrame:
    times = pd.date_range("2026-01-05T14:30:00Z", periods=91, freq="min")
    return pd.DataFrame({"timestamp": times, "high": range(101, 192), "low": range(99, 190),
                         "close": range(100, 191), "volume": [100] * 91, "vwap": range(100, 191)})


def test_forward_return_requires_contiguous_future_bars() -> None:
    frame = _bars()
    base = {"status": "ok", "close": 130.0, "latest_completed_bar_start_utc": "2026-01-05T15:00:00+00:00"}
    assert _forward_return(frame, base, 1, 20) == pytest.approx(20.0 / 130.0)
    assert _forward_return(frame.drop(index=40), base, 1, 20) is None


def test_feature_row_is_directional_and_predecision() -> None:
    row = _feature_row(kind="matched_no_nearby_alert_control", alert=None, caller="ACE", symbol="SPY",
                       decision=pd.Timestamp("2026-01-05T15:01:00Z"), sign=-1,
                       source_file=Path("fixture.parquet"), frame=_bars())
    assert row is not None
    assert row["is_alert"] == 0
    assert row["completed_bar_at_utc"] == "2026-01-05T15:00:00+00:00"
    assert row["directional_return_5m"] < 0
    assert row["directional_forward_return_20m"] < 0


def test_outcome_controls_follow_source_cited_direction_mix() -> None:
    rows = [
        {"is_alert": 1, "directional_forward_return_20m": 0.01, "directional_forward_return_60m": 0.01},
        {"is_alert": 0, "outcome_control_weight": 2, "directional_forward_return_20m": 0.02,
         "directional_forward_return_60m": 0.02},
        {"is_alert": 0, "outcome_control_weight": 0, "directional_forward_return_20m": -0.02,
         "directional_forward_return_60m": -0.02},
    ]
    result = _outcome_summary(rows)
    assert result["20m"]["matched_controls"]["n"] == 1
    assert result["20m"]["matched_controls"]["weighted_n"] == 2
    assert result["20m"]["matched_controls"]["mean_pct"] == pytest.approx(2.0)
