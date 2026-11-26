"""The shared earnings-calendar file has two writers; each must replace only its own rows."""
from __future__ import annotations

import pandas as pd

from signals.events.earnings_calendar import merge_calendar_history, merge_calendar_snapshot

_HIST = pd.DataFrame({
    "ticker": ["AAA"] * 3 + ["BBB"] * 2,
    "date": pd.to_datetime(["2024-01-10", "2024-04-10", "2024-07-10", "2024-02-01", "2024-05-01"]),
    "snapshot_date": pd.NaT,
})


def _snap(day, next_date="2026-10-10"):
    return pd.DataFrame({"ticker": ["AAA", "BBB"], "next_earnings_date": pd.to_datetime([next_date] * 2),
                         "snapshot_date": pd.to_datetime([day] * 2)})


def test_nightly_snapshot_keeps_every_history_row():
    """The 2026-09 bug: dedupe on (ticker, snapshot_date) collapsed NaT history rows."""
    out = merge_calendar_snapshot(_HIST, _snap("2026-09-23"))
    out = merge_calendar_snapshot(out, _snap("2026-09-24"))
    assert out["snapshot_date"].isna().sum() == len(_HIST)
    assert out["snapshot_date"].notna().sum() == 4


def test_same_day_snapshot_rerun_replaces_itself():
    out = merge_calendar_snapshot(_snap("2026-09-24", "2026-10-01"), _snap("2026-09-24", "2026-10-02"))
    assert len(out) == 2 and (out["next_earnings_date"] == "2026-10-02").all()


def test_history_refresh_keeps_snapshot_rows_and_replaces_history():
    prior = merge_calendar_snapshot(_HIST, _snap("2026-09-24"))
    new_hist = _HIST.iloc[:2]
    out = merge_calendar_history(prior, new_hist)
    assert out["snapshot_date"].isna().sum() == 2
    assert out["snapshot_date"].notna().sum() == 2


def test_union_history_keeps_unfetched_rows_and_fresh_rows_win():
    """--union-history: a refresh that misses a ticker (rate limit) or old quarters keeps them."""
    from signals.events.earnings_calendar import merge_history_union

    prior = merge_calendar_snapshot(_HIST.assign(surprise_pct=[1.0, 2.0, 3.0, 4.0, 5.0]), _snap("2026-09-24"))
    fetched = pd.DataFrame({"ticker": ["AAA", "AAA"], "date": pd.to_datetime(["2024-07-10", "2024-10-10"]),
                            "surprise_pct": [30.0, 40.0], "snapshot_date": pd.NaT})
    out = merge_history_union(prior, fetched)
    hist = out[out["snapshot_date"].isna()]
    assert len(hist) == 6                                    # 5 prior + 1 new quarter
    assert set(hist.loc[hist.ticker == "BBB", "date"]) == set(_HIST.loc[_HIST.ticker == "BBB", "date"])
    assert hist.loc[(hist.ticker == "AAA") & (hist.date == "2024-07-10"), "surprise_pct"].item() == 30.0
    assert hist.loc[(hist.ticker == "AAA") & (hist.date == "2024-01-10"), "surprise_pct"].item() == 1.0
    assert out["snapshot_date"].notna().sum() == 2           # snapshot rows untouched
