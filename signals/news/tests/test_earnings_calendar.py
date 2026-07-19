from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from signals.news.sources import fetch_earnings_calendar


pytestmark = pytest.mark.safe


def _fake_fetch_one(ticker: str):
    if ticker == "HANG":
        time.sleep(5)
    if ticker == "EMPTY":
        return None
    return "2026-08-03"


@pytest.mark.skipif("spawn" not in mp.get_all_start_methods(), reason="multiprocessing spawn unavailable")
def test_earnings_calendar_recovers_after_one_ticker_timeout():
    started = time.monotonic()
    result = fetch_earnings_calendar(
        ["HANG", "AAPL", "EMPTY"],
        min_interval_s=0,
        request_timeout_s=1,
        progress_every=0,
        _fetch_one=_fake_fetch_one,
    )

    assert time.monotonic() - started < 4
    assert result["ticker"].tolist() == ["AAPL"]
    assert str(result.iloc[0]["next_earnings_date"].date()) == "2026-08-03"
