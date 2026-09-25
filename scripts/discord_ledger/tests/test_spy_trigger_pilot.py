from __future__ import annotations

import pandas as pd

from scripts.discord_ledger.spy_trigger_pilot import snapshot


def _bars() -> pd.DataFrame:
    times = pd.date_range("2026-01-05T14:30:00Z", periods=31, freq="min")
    return pd.DataFrame({"timestamp": times, "high": range(101, 132),
                         "low": range(99, 130), "close": range(100, 131),
                         "volume": [100] * 31, "vwap": range(100, 131)})


def test_snapshot_uses_only_completed_bars() -> None:
    before = snapshot(_bars(), pd.Timestamp("2026-01-05T15:00:30Z"))
    after = snapshot(_bars(), pd.Timestamp("2026-01-05T15:01:00Z"))
    assert before["latest_completed_bar_start_utc"] == "2026-01-05T14:59:00+00:00"
    assert before["close"] == 129.0
    assert after["close"] == 130.0
    assert before["distance_above_opening_range_high_pct"] is not None


def test_snapshot_fails_on_gap_in_lookback() -> None:
    bars = _bars().drop(index=20)
    result = snapshot(bars, pd.Timestamp("2026-01-05T15:01:00Z"))
    assert result["status"] == "gap_in_lookback"
    assert result["return_5m"] is None
