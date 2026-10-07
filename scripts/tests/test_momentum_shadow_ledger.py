"""The schedule must give the same decision whichever day the weekly refresh runs, and never skip one."""
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from scripts.shadow import momentum_shadow_ledger as msl

ET = ZoneInfo("America/New_York")
ANCHOR = date(2026, 10, 2)
SECOND = date(2026, 10, 30)     # anchor + 4 weeks


def at(y, m, d, hh=12, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def test_week_end_session_handles_weekends_and_a_holiday_friday():
    assert msl.week_end_session(date(2026, 10, 7)) == date(2026, 10, 9)      # Wednesday -> that Friday
    assert msl.week_end_session(date(2026, 10, 10)) == date(2026, 10, 9)     # Saturday -> the Friday just ended
    assert msl.week_end_session(date(2026, 10, 11)) == date(2026, 10, 9)     # Sunday
    assert msl.week_end_session(date(2027, 3, 24)) == date(2027, 3, 25)      # Good Friday 2027-03-26 -> Thursday


@pytest.mark.parametrize("now", [
    at(2026, 10, 30, 16, 45),   # Friday evening, after the close
    at(2026, 10, 31, 9),        # Saturday
    at(2026, 11, 1, 22),        # Sunday night
    at(2026, 11, 2, 7),         # Monday before the open
    at(2026, 11, 2, 11),        # Monday during the session
    at(2026, 11, 2, 20),        # Monday after the close
    at(2026, 11, 4, 12),        # ran late: Wednesday
])
def test_any_run_day_from_friday_evening_on_records_the_same_decision(now):
    assert msl.due_decisions(now, {ANCHOR}) == [SECOND]


def test_friday_before_the_close_records_nothing_and_does_not_lose_the_decision():
    assert msl.due_decisions(at(2026, 10, 30, 15, 0), {ANCHOR}) == []
    assert msl.due_decisions(at(2026, 10, 30, 16, 29), {ANCHOR}) == []
    assert msl.due_decisions(at(2026, 10, 31, 10), {ANCHOR}) == [SECOND]      # the next run picks it up


@pytest.mark.parametrize("now", [at(2026, 10, 10), at(2026, 10, 17), at(2026, 10, 24), at(2026, 10, 26, 8)])
def test_off_weeks_record_nothing(now):
    assert msl.due_decisions(now, {ANCHOR}) == []


def test_recorded_decision_is_not_due_again():
    assert msl.due_decisions(at(2026, 11, 2, 7), {ANCHOR, SECOND}) == []


def test_a_missed_weekend_is_caught_up_the_following_week():
    assert msl.due_decisions(at(2026, 11, 7), {ANCHOR}) == [SECOND]


def test_two_missed_periods_are_both_recorded_oldest_first():
    assert msl.due_decisions(at(2026, 12, 5), {ANCHOR}) == [SECOND, date(2026, 11, 27)]


def test_holiday_friday_moves_the_decision_to_thursday():
    # 2027-03-26 is Good Friday and falls on the 4-week grid (anchor + 25 weeks is not; use a local anchor)
    anchor = date(2027, 2, 26)
    assert msl.scheduled_decisions(date(2027, 3, 31), anchor) == [date(2027, 2, 26), date(2027, 3, 25)]
    assert msl.due_decisions(at(2027, 3, 25, 17, 0), {anchor}, anchor) == [date(2027, 3, 25)]   # Thursday evening


def test_last_completed_session():
    assert msl.last_completed_session(at(2026, 10, 5, 11)) == date(2026, 10, 2)    # Monday mid-session -> Friday
    assert msl.last_completed_session(at(2026, 10, 5, 17)) == date(2026, 10, 5)
    assert msl.last_completed_session(at(2026, 10, 4, 12)) == date(2026, 10, 2)    # Sunday


def _write_bars(path, dates, close, volume=1_000_000):
    pd.DataFrame({"timestamp": pd.to_datetime(dates).tz_localize("America/New_York").tz_convert("UTC"),
                  "open": close, "close": close, "volume": volume, "trade_count": 5_000}).to_parquet(path, index=False)


def test_targets_use_only_bars_up_to_the_decision_date(tmp_path):
    """Recording late (bars after the decision already exist) must give the row a Friday-night run would."""
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2025-06-02", "2026-11-13")
    asof = date(2026, 10, 30)
    cut = int((days <= pd.Timestamp(asof)).sum())
    tickers = [f"T{i:02d}" for i in range(40)]
    early, late = tmp_path / "early", tmp_path / "late"
    early.mkdir(), late.mkdir()
    for t in tickers + ["SPY"]:
        px = 50 * np.exp(np.cumsum(rng.normal(0.001, 0.02, len(days))))
        px[cut:] *= rng.uniform(0.2, 5.0)                      # the future differs wildly
        _write_bars(late / f"{t}.parquet", days, px)
        _write_bars(early / f"{t}.parquet", days[:cut], px[:cut])
    a = msl.compute_targets(early, asof, tickers + ["SPY"], min_tradable=30)
    b = msl.compute_targets(late, asof, tickers + ["SPY"], min_tradable=30)
    pd.testing.assert_frame_equal(a, b)
    assert a.groupby("arm")["weight"].sum().round(9).eq(1.0).all()
    with pytest.raises(RuntimeError, match="no completed SPY bar"):
        msl.compute_targets(early, date(2026, 11, 6), tickers + ["SPY"], min_tradable=30)


def test_mark_prices_entry_open_to_last_close_and_needs_enough_history(tmp_path):
    """Regression: a marks fetch of ~10 sessions made split_lives drop every series (lives need 30 bars)."""
    days = pd.bdate_range("2026-06-01", "2026-10-06")
    px = pd.Series(np.linspace(100, 110, len(days)), index=days)
    for t, mult in (("SPY", 1.0), ("AAA", 2.0)):
        df = pd.DataFrame({"timestamp": days.tz_localize("America/New_York").tz_convert("UTC"),
                           "open": px.values * mult, "close": px.values * mult * 1.01,
                           "volume": 1_000_000, "trade_count": 5_000})
        df.to_parquet(tmp_path / f"{t}.parquet", index=False)
    ledger = tmp_path / "shadow" / "ledger.csv"
    ledger.parent.mkdir()
    pd.DataFrame([{"arm": "spy", "ticker": "SPY", "weight": 1.0, "decision_date": "2026-10-02"},
                  {"arm": "one", "ticker": "AAA", "weight": 1.0, "decision_date": "2026-10-02"}]).to_csv(ledger, index=False)
    txt = msl.mark(ledger, at(2026, 10, 6, 23, 0), tmp_path)
    want = px["2026-10-06"] * 1.01 / px["2026-10-05"] - 1          # Monday open -> Tuesday close
    assert f"{want:+.2%}" in txt and "marked through 2026-10-06" in txt
    assert "has not completed" in msl.mark(ledger, at(2026, 10, 3, 9, 0), tmp_path)   # Saturday: nothing to price yet
