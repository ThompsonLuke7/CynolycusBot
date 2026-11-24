"""30m daily context: live rows win, stale rows are not served, upsert replaces by key."""
import pandas as pd

from strategies.multi_ticker_swing.live import context_features as cf
from strategies.multi_ticker_swing.live import refresh_daily_context as rd


def _rows(dates, value):
    return pd.DataFrame({"ticker": "AAA", "date": pd.to_datetime(dates), "treasury_10y": value})


def test_live_file_wins_and_stale_rows_are_not_served(tmp_path):
    export, live = tmp_path / "export.parquet", tmp_path / "live.parquet"
    _rows(["2026-06-02", "2026-06-03"], 4.5).to_parquet(export, index=False)
    _rows(["2026-06-03", "2026-11-24"], 4.1).to_parquet(live, index=False)
    lk = cf.DailyContextLookup(export, live)
    assert lk.get("AAA", pd.Timestamp("2026-06-03"))["treasury_10y"] == 4.1     # live wins on a duplicate
    assert lk.get("AAA", pd.Timestamp("2026-11-24 15:00", tz="UTC"))["treasury_10y"] == 4.1
    assert lk.get("AAA", pd.Timestamp("2026-11-24")) != {}
    only_export = cf.DailyContextLookup(export, tmp_path / "missing.parquet")
    assert only_export.get("AAA", pd.Timestamp("2026-11-23")) == {}           # June row, 173 days old
    assert only_export.get("AAA", pd.Timestamp("2026-06-05"))["treasury_10y"] == 4.5


def test_upsert_replaces_same_ticker_date(tmp_path):
    out = tmp_path / "live.parquet"
    rd.upsert(_rows(["2026-11-20", "2026-11-23"], 1.0), out)
    total = rd.upsert(_rows(["2026-11-23", "2026-11-24"], 2.0), out)
    got = pd.read_parquet(out).set_index("date")["treasury_10y"]
    assert total == 3 and got.loc["2026-11-20"] == 1.0 and got.loc["2026-11-23"] == 2.0


def test_session_dates_are_trading_days_plus_next():
    ds = rd.session_dates(pd.Timestamp("2026-11-23"), 3)        # Monday
    assert [d.strftime("%m-%d") for d in ds] == ["11-19", "11-20", "11-23", "11-24"]
    ds = rd.session_dates(pd.Timestamp("2026-11-25"), 1)        # day before Thanksgiving
    assert [d.strftime("%m-%d") for d in ds] == ["11-25", "11-27"]
