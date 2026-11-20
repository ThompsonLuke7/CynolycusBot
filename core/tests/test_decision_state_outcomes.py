"""Outcomes come from an executable entry and mature honestly."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core.decision_state_outcomes import is_final, last_complete_session, ticker_outcomes

SESSIONS = pd.bdate_range("2026-10-01", periods=30)


def _bars(price_path: dict[int, tuple[float, float, float, float]] | None = None, base=100.0) -> pd.DataFrame:
    """Hourly bars 10:00..16:00 ET (14:00..20:00 UTC in October) for each session."""
    rows = []
    for i, d in enumerate(SESSIONS):
        for h in range(14, 21):
            o = hi = lo = c = base + i
            if price_path and (i, h) in price_path:
                o, hi, lo, c = price_path[(i, h)]
            rows.append({"timestamp": pd.Timestamp(d) + pd.Timedelta(hours=h), "open": o, "high": hi,
                         "low": lo, "close": c, "volume": 1000})
    df = pd.DataFrame(rows)
    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df


def _daily(base=100.0) -> pd.DataFrame:
    days = pd.bdate_range("2026-09-01", "2026-11-30")
    return pd.DataFrame({"timestamp": (days + pd.Timedelta(hours=4)).tz_localize("UTC"),
                         "open": base, "high": base * 1.02, "low": base * 0.98, "close": base, "volume": 1})


def _dec(ts: str) -> pd.DataFrame:
    return pd.DataFrame({"run_id": ["r1"], "ticker": ["AAA"], "decision_ts": [pd.Timestamp(ts)]})


CAL = pd.DatetimeIndex(SESSIONS)


def test_entry_is_the_next_bar_open_not_the_decision_close():
    bars = _bars()
    out = ticker_outcomes(_dec("2026-10-01T16:30:00Z"), bars, calendar=CAL, last_complete=CAL[-1],
                          spy=bars, daily=_daily())
    r = out.iloc[0]
    assert r["entry_ts"] == pd.Timestamp("2026-10-01T17:00:00Z") and r["entry_px"] == 100.0
    # h=1: exit at the close of the next session (price 101) -> +1%.
    assert r["status_1"] == "ok" and np.isclose(r["fret_1"], 0.01)
    assert np.isclose(r["fret_5"], 0.05) and np.isclose(r["spy_fret_5"], 0.05)


def test_after_close_decision_enters_next_session():
    bars = _bars()
    out = ticker_outcomes(_dec("2026-10-01T21:30:00Z"), bars, calendar=CAL, last_complete=CAL[-1],
                          spy=bars, daily=_daily())
    assert out.iloc[0]["entry_ts"] == pd.Timestamp("2026-10-02T14:00:00Z")


def test_horizons_past_the_last_complete_session_are_immature():
    bars = _bars()
    out = ticker_outcomes(_dec("2026-10-01T16:30:00Z"), bars, calendar=CAL, last_complete=CAL[3],
                          spy=bars, daily=_daily())
    r = out.iloc[0]
    assert r["status_1"] == "ok" and r["status_5"] == "immature" and np.isnan(r["fret_5"])
    assert not is_final(out)


def test_first_passage_reads_a_both_touch_bar_as_adverse():
    # ATR% = 4%: a session-2 bar trading 95..106 touches both barriers.
    path = {(2, 15): (100.0, 106.0, 95.0, 100.0)}
    bars = _bars(path, base=100.0)
    bars.loc[bars["timestamp"] < pd.Timestamp("2026-10-03", tz="UTC"), ["open", "high", "low", "close"]] = 100.0
    out = ticker_outcomes(_dec("2026-10-01T13:00:00Z"), bars, calendar=CAL, last_complete=CAL[-1],
                          spy=bars, daily=_daily())
    assert out.iloc[0]["fp_5"] == -1.0


def test_corporate_action_inside_the_window_voids_it():
    bars = _bars()
    out = ticker_outcomes(_dec("2026-10-01T16:30:00Z"), bars, calendar=CAL, last_complete=CAL[-1],
                          spy=bars, daily=_daily(), flagged={CAL[3]})
    r = out.iloc[0]
    assert r["status_1"] == "ok" and r["status_5"] == "void_corporate_action" and np.isnan(r["fret_5"])


def test_missing_bars_and_gaps_are_named():
    bars = _bars()
    no = ticker_outcomes(_dec("2026-10-01T16:30:00Z"), None, calendar=CAL, last_complete=CAL[-1], spy=bars)
    assert no.iloc[0]["status_1"] == "no_bars"
    gappy = bars[(bars["timestamp"] < pd.Timestamp("2026-10-01T17:00Z")) |
                 (bars["timestamp"] > pd.Timestamp("2026-10-08", tz="UTC"))]
    out = ticker_outcomes(_dec("2026-10-01T16:30:00Z"), gappy, calendar=CAL, last_complete=CAL[-1], spy=bars)
    assert out.iloc[0]["status_1"] == "entry_gap"


def test_last_complete_session_waits_for_the_close():
    cal = pd.DatetimeIndex(pd.bdate_range("2026-11-09", "2026-11-13"))
    midday = datetime(2026, 11, 12, 17, 0, tzinfo=timezone.utc)   # 12:00 ET
    evening = datetime(2026, 11, 12, 23, 0, tzinfo=timezone.utc)  # 18:00 ET
    assert last_complete_session(cal, midday) == pd.Timestamp("2026-11-11")
    assert last_complete_session(cal, evening) == pd.Timestamp("2026-11-12")
