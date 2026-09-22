from __future__ import annotations

import pandas as pd

from scripts.discord_ledger.market_context import follower_scenarios, lookup_context


def _write_bars(tmp_path, *, minute_timestamps: list[str]) -> tuple[str, str]:
    daily_path = tmp_path / "daily.parquet"
    minute_path = tmp_path / "minute.parquet"
    daily_dates = pd.date_range("2025-12-01", periods=25, freq="B", tz="UTC")
    pd.DataFrame(
        {
            "timestamp": daily_dates,
            "close": [100.0 + index for index in range(len(daily_dates))],
            "volume": [1_000.0 + index for index in range(len(daily_dates))],
        }
    ).to_parquet(daily_path, index=False)
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(minute_timestamps, utc=True),
            "close": [200.0 + index for index in range(len(minute_timestamps))],
            "volume": [100.0] * len(minute_timestamps),
        }
    ).to_parquet(minute_path, index=False)
    return str(daily_path), str(minute_path)


def test_lookup_context_excludes_incomplete_minute_and_same_session_daily(tmp_path) -> None:
    daily_path, minute_path = _write_bars(
        tmp_path,
        minute_timestamps=["2026-01-05T14:29:00Z", "2026-01-05T14:30:00Z"],
    )

    result = lookup_context(
        "abc",
        "2026-01-05T14:30:30Z",
        daily_path=daily_path,
        minute_path=minute_path,
    )

    assert result["status"] == "ok"
    assert result["minute"]["latest_bar_start_utc"] == "2026-01-05T14:29:00+00:00"
    # The Jan-05 daily bar is not complete at 09:30 ET, so Jan-02 is latest.
    assert result["daily"]["latest_bar_start_utc"].startswith("2026-01-02")
    assert result["certified_point_in_time"] is False
    assert "retrospectively fetched" in result["sources"]["minute"]["availability"]


def test_lookup_context_fails_closed_on_minute_gap(tmp_path) -> None:
    daily_path, minute_path = _write_bars(tmp_path, minute_timestamps=["2026-01-05T14:28:00Z"])

    result = lookup_context(
        "abc",
        "2026-01-05T14:30:30Z",
        daily_path=daily_path,
        minute_path=minute_path,
    )

    assert result["status"] == "partial"
    assert result["daily"] is not None
    assert result["minute"] is None
    assert any(reason.startswith("minute_gap_at_alert") for reason in result["reasons"])


def test_option_follower_scenarios_never_invent_historical_prices() -> None:
    result = follower_scenarios("SPY", "2026-01-05T14:30:30Z", "option")

    assert result["status"] == "blocked"
    assert [scenario["price"] for scenario in result["scenarios"]] == [None, None, None]
    assert all(scenario["status"] == "blocked" for scenario in result["scenarios"])
