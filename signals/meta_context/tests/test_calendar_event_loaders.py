"""Earnings and economic-calendar event loaders shared by the Meta matrix and the 30m export.

Regressions from 2026-11-24: a calendar file holding both row kinds read the wrong kind.
"""
from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from signals.meta_context.build_meta_ranker_matrix import (
    earnings_events_from_calendar,
    economic_events_from_calendar,
)

pytestmark = pytest.mark.safe


def _ts(s):
    return pd.Timestamp(s)


def test_earnings_history_and_snapshots_both_count():
    # History (past reports + scheduled) and nightly snapshots in one file, as the shared
    # calendar is. The old loader matched the snapshot schema first and dropped history.
    cal = pd.DataFrame({
        "ticker": ["AAA", "AAA", "AAA", "AAA", "BBB"],
        "date": [_ts("2024-02-01"), _ts("2024-05-01"), pd.NaT, pd.NaT, pd.NaT],
        "next_earnings_date": [pd.NaT, pd.NaT, _ts("2024-05-01"), _ts("2024-08-01"), _ts("2024-07-15")],
        "snapshot_date": [pd.NaT, pd.NaT, _ts("2024-04-20"), _ts("2024-06-01"), _ts("2024-06-01")],
    })
    ev = earnings_events_from_calendar(cal)
    got = {(t, str(d.date())) for t, d in ev.itertuples(index=False)}
    assert got == {("AAA", "2024-02-01"), ("AAA", "2024-05-01"),   # history kept
                   ("AAA", "2024-08-01"),                           # forward snapshot added
                   ("BBB", "2024-07-15")}                           # no history: snapshots


def test_earnings_snapshot_inside_history_is_not_a_second_event():
    # A snapshot estimate older than the ticker's last history date is superseded.
    cal = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "date": [_ts("2024-05-03"), pd.NaT],
        "next_earnings_date": [pd.NaT, _ts("2024-05-01")],
    })
    assert earnings_events_from_calendar(cal)["event_date"].tolist() == [_ts("2024-05-03")]


def test_earnings_unknown_schema_raises():
    with pytest.raises(ValueError):
        earnings_events_from_calendar(pd.DataFrame({"foo": [1]}))


def test_economic_mixed_file_keeps_deterministic_rows():
    # The TradingView stage concatenated its feed into the deterministic file and
    # deduplicated on (event, event_date); deterministic rows (no event_date) collapsed to
    # one per event name and the feed's rows (impact None) counted ~33 events/day.
    cal = pd.DataFrame({
        "date": [_ts("2026-11-06"), _ts("2026-12-09"), pd.NaT, pd.NaT],
        "event": ["Nonfarm payrolls (NFP)", "FOMC rate decision", "API Crude Imports", "MBA Mortgage"],
        "impact": ["high", "high", None, None],
        "country": ["US"] * 4,
        "event_date": [pd.NaT, pd.NaT, pd.Timestamp("2026-11-06 21:30", tz="UTC"),
                       pd.Timestamp("2026-11-06 12:00", tz="UTC")],
        "importance": [None, None, "-1", "1"],
    })
    ev = economic_events_from_calendar(cal)
    assert ev["event_date"].tolist() == [_ts("2026-11-06"), _ts("2026-12-09")]
    assert ev["is_high_impact"].tolist() == [1.0, 1.0]


def test_economic_deterministic_schema():
    cal = pd.DataFrame({"date": [_ts("2026-11-13")], "event": ["CPI inflation"],
                        "impact": ["high"], "country": ["US"]})
    ev = economic_events_from_calendar(cal)
    assert ev.to_dict("records") == [{"event_date": _ts("2026-11-13"), "is_high_impact": 1.0}]


def test_tradingview_stage_does_not_target_the_deterministic_file():
    from signals.news import config

    assert config.ECONOMIC_CALENDAR_TV_PATH != config.ECONOMIC_CALENDAR_PATH


def test_earnings_snapshot_falls_back_to_earnings_dates_after_a_report(monkeypatch):
    from signals.news.sources import _fetch_one_earnings_calendar

    today = pd.Timestamp.now().normalize()
    past, nxt = today - pd.Timedelta(days=20), today + pd.Timedelta(days=70)

    class FakeTicker:
        def __init__(self, _t):
            self.calendar = {"Earnings Date": [past.date()]}

        def get_earnings_dates(self, limit=4):
            idx = pd.DatetimeIndex([nxt, past]).tz_localize("America/New_York")
            return pd.DataFrame({"EPS Estimate": [1.0, 1.0]}, index=idx)

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))
    assert pd.Timestamp(_fetch_one_earnings_calendar("NVDA")).normalize() == nxt
