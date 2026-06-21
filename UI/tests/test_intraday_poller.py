"""Tests for the intraday catalyst poller's market-hours window."""
from __future__ import annotations

import datetime as dt

from UI.intraday_poller import _tz, is_market_hours

ET = _tz("America/New_York")


def _at(y, m, d, hh, mm):
    return dt.datetime(y, m, d, hh, mm, tzinfo=ET)


def test_open_close_boundaries():
    # Monday 2026-06-15
    assert not is_market_hours(_at(2026, 6, 15, 9, 29))   # before open
    assert is_market_hours(_at(2026, 6, 15, 9, 30))       # open is inclusive
    assert is_market_hours(_at(2026, 6, 15, 12, 0))       # midday
    assert not is_market_hours(_at(2026, 6, 15, 16, 0))   # close is exclusive
    assert not is_market_hours(_at(2026, 6, 15, 16, 1))   # after close


def test_weekend_closed():
    assert not is_market_hours(_at(2026, 6, 13, 12, 0))   # Saturday
    assert not is_market_hours(_at(2026, 6, 14, 12, 0))   # Sunday


def test_custom_window():
    # extended hours window honored when overridden
    assert is_market_hours(_at(2026, 6, 15, 8, 0), open_hm=(7, 0), close_hm=(20, 0))
    assert not is_market_hours(_at(2026, 6, 15, 8, 0))    # default RTH excludes pre-market
