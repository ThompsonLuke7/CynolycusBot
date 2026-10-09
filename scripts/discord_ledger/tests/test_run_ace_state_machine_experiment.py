from __future__ import annotations

import pandas as pd
import pytest

from scripts.discord_ledger.run_ace_state_machine_experiment import daily_structure, prepare_daily_bars, state_machine


def _daily_frame() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=70, freq="B", tz="America/New_York").tz_convert("UTC")
    close = [100.0 + index for index in range(len(dates))]
    return pd.DataFrame({"timestamp": dates, "open": close, "high": [x + 2 for x in close],
                         "low": [x - 2 for x in close], "close": close})


def test_daily_structure_excludes_decision_date_bar() -> None:
    daily = prepare_daily_bars(_daily_frame())
    decision_day = daily.iloc[-1]["session_date_et"]
    decision = pd.Timestamp(decision_day, tz="America/New_York").replace(hour=15).tz_convert("UTC")
    result = daily_structure(daily, decision_at_utc=decision, entry_price=168.5, direction_sign=1)
    assert result["daily_feature_status"] == "ok"
    assert result["completed_daily_sessions_used"] == 69
    assert result["latest_completed_daily_session_et"] != decision_day.isoformat()


def test_state_machine_fails_closed_for_missing_intraday_features() -> None:
    daily = {"daily_feature_status": "ok", "trend_supportive": True, "pullback_due": True,
             "nearest_level_abs_atr": 0.1}
    result = state_machine(daily, {"directional_return_5m": None, "directional_vwap_distance_pct": 0.1,
                                   "volume_last5_vs_prior20": 2.0})
    assert result["at_level"] is True
    assert result["confirmed"] is False
    assert result["trade_hypothesis"] is False
    assert result["intraday_confirmation"] is None


def test_prepare_daily_bars_rejects_invalid_ohlc() -> None:
    frame = _daily_frame()
    frame.loc[0, "low"] = frame.loc[0, "high"] + 1
    with pytest.raises(ValueError, match="high is below low"):
        prepare_daily_bars(frame)
