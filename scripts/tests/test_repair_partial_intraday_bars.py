import pandas as pd

from scripts.repair_partial_intraday_bars import compare, splice

SINCE = pd.Timestamp("2026-11-20 00:00", tz="UTC")
END = pd.Timestamp("2026-11-21 00:00", tz="UTC")


def _bars(rows):
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.insert(0, "symbol", "AAPL")
    return df


CACHED = _bars([
    ("2026-11-19 18:00", 1, 2, 0.5, 1.5, 100.0),      # outside the window: never touched
    ("2026-11-20 17:00", 1, 2, 0.5, 1.5, 3910584.0),  # identical in the fresh pull
    ("2026-11-20 18:00", 1, 2, 0.5, 1.4, 3153886.0),  # partial
    ("2026-11-20 19:00", 1, 2, 0.5, 1.3, 50.0),       # fresh pull lacks it: kept
])
FRESH = _bars([
    ("2026-11-20 17:00", 1, 2, 0.5, 1.5, 3910584.0),
    ("2026-11-20 18:00", 1, 2, 0.4, 1.45, 5889040.0),
    ("2026-11-20 20:00", 1, 2, 0.5, 1.2, 16497382.0),  # never cached: added
])


def test_compare_counts_partial_changed_added_and_kept_rows():
    r = compare(CACHED, FRESH, SINCE, END)
    assert r["window_rows_cached"] == 3
    assert r["rows_matched"] == 2
    assert r["rows_changed"] == 1
    assert r["rows_partial"] == 1
    assert r["rows_added"] == 1
    assert r["rows_kept_missing_from_fresh"] == 1


def test_splice_replaces_window_rows_only():
    out = splice(CACHED, FRESH, SINCE, END).set_index("timestamp")
    assert list(out.index.strftime("%m-%d %H")) == ["11-19 18", "11-20 17", "11-20 18", "11-20 19", "11-20 20"]
    assert out.loc[pd.Timestamp("2026-11-19 18:00", tz="UTC"), "volume"] == 100.0
    assert out.loc[pd.Timestamp("2026-11-20 18:00", tz="UTC"), "volume"] == 5889040.0
    assert out.loc[pd.Timestamp("2026-11-20 19:00", tz="UTC"), "volume"] == 50.0
    assert list(out.columns) == ["symbol", "open", "high", "low", "close", "volume"]
