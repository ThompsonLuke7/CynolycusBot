"""The Meta matrix must not append a 4H bar before all its hours are fetched (2026-11-22)."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from signals.meta_context.meta_ranker.update_meta_matrix import complete_bars_only

ET = ZoneInfo("America/New_York")


def _panel(stamps):
    idx = pd.MultiIndex.from_product([pd.DatetimeIndex(stamps).tz_convert("UTC"), ["AAPL", "NVDA"]],
                                     names=["timestamp", "ticker"])
    return pd.DataFrame({"x": range(len(idx))}, index=idx)


def test_holds_back_the_bar_whose_last_hour_was_not_fetched():
    bar_b_prev = pd.Timestamp("2026-11-19 14:00", tz=ET)
    bar_a = pd.Timestamp("2026-11-20 10:00", tz=ET)
    panel = _panel([bar_b_prev, bar_a])
    # 13:45 cycle: fetched through 13:25 ET -> today's 10:00 bar is incomplete.
    out = complete_bars_only(panel, fetched_through=datetime(2026, 11, 20, 13, 25, tzinfo=ET))
    assert set(out.index.get_level_values(0)) == {bar_b_prev.tz_convert("UTC")}
    # 14:20:30 cycle: fetched through 14:00:30 ET -> both bars complete.
    out = complete_bars_only(panel, fetched_through=datetime(2026, 11, 20, 14, 0, 30, tzinfo=ET))
    assert len(out) == 4


def test_without_a_stamp_falls_back_to_the_wall_clock():
    bar_a = pd.Timestamp("2026-11-20 10:00", tz=ET)
    now = datetime(2026, 11, 20, 14, 10, tzinfo=ET).astimezone(timezone.utc)   # 14:10 - 20 min < 14:00
    assert complete_bars_only(_panel([bar_a]), fetched_through=None, now=now).empty
