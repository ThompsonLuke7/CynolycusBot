import json
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from core.personal_portfolio import data as d
from core.personal_portfolio.data import ET


def test_watchlist_add_update_remove(tmp_path):
    d.add_to_watchlist("nvda", "AI leader", tmp_path, datetime(2026, 11, 1, 12, tzinfo=ET))
    d.add_to_watchlist("$amd", "", tmp_path)
    d.add_to_watchlist("NVDA", "updated", tmp_path, datetime(2026, 11, 5, 12, tzinfo=ET))
    rows = d.read_watchlist(tmp_path)
    assert [r["ticker"] for r in rows] == ["AMD", "NVDA"]
    assert rows[1] == {"ticker": "NVDA", "note": "updated", "added_at": "2026-11-01"}
    d.remove_from_watchlist("amd", tmp_path)
    assert [r["ticker"] for r in d.read_watchlist(tmp_path)] == ["NVDA"]
    with pytest.raises(KeyError):
        d.remove_from_watchlist("AMD", tmp_path)
    for bad in ("", "1ABC", "TOO-LONG-TICKER", "A B", "<script>"):
        with pytest.raises(ValueError):
            d.add_to_watchlist(bad, "", tmp_path)


def test_config_and_overrides_fail_fast(tmp_path):
    assert d.load_config(tmp_path) == d.PersonalConfig()
    (tmp_path / "config.json").write_text(json.dumps({"max_underlying_pct": 0.3}))
    assert d.load_config(tmp_path).max_underlying_pct == 0.3
    (tmp_path / "config.json").write_text(json.dumps({"max_underlying": 0.3}))
    with pytest.raises(ValueError, match="unknown config keys"):
        d.load_config(tmp_path)
    (tmp_path / "look_through.json").write_text(json.dumps({"soxl": {"underlying": "soxx", "leverage": 3}}))
    assert d.load_look_through_overrides(tmp_path) == {"SOXL": {"underlying": "SOXX", "leverage": 3.0}}
    (tmp_path / "look_through.json").write_text(json.dumps({"SOXL": {"underlying": "SOXX"}}))
    with pytest.raises(ValueError):
        d.load_look_through_overrides(tmp_path)


def test_last_completed_session():
    assert d.last_completed_session(datetime(2026, 11, 12, 16, 29, tzinfo=ET)) == date(2026, 11, 11)
    assert d.last_completed_session(datetime(2026, 11, 12, 16, 31, tzinfo=ET)) == date(2026, 11, 12)
    assert d.last_completed_session(datetime(2026, 11, 15, 12, 0, tzinfo=ET)) == date(2026, 11, 13)   # Sunday


def _frame(end: str, n: int = 30):
    days = pd.bdate_range(end=end, periods=n).tz_localize(ET).tz_convert("UTC")
    return pd.DataFrame({"timestamp": days, "close": np.linspace(10, 11, n), "volume": 1_000})


def test_refresh_bars_fetches_only_stale_files(tmp_path):
    calls = []

    def fetch(t, start, end):
        calls.append(t)
        if t == "BAD":
            raise RuntimeError("boom")
        return _frame("2026-11-12")

    (tmp_path / "bars_1d").mkdir()
    _frame("2026-11-12").to_parquet(tmp_path / "bars_1d" / "CUR.parquet")
    _frame("2026-11-05").to_parquet(tmp_path / "bars_1d" / "OLD.parquet")
    st = d.refresh_bars(["CUR", "OLD", "NEW", "BAD"], datetime(2026, 11, 12, 18, 0, tzinfo=ET), tmp_path, fetch)
    assert st["CUR"] == "current" and st["OLD"] == "fetched" and st["NEW"] == "fetched"
    assert st["BAD"].startswith("error") and sorted(calls) == ["BAD", "NEW", "OLD"]
    closes = d.load_bars(["CUR", "OLD", "NEW", "MISSING"], tmp_path)
    assert list(closes.columns) == ["CUR", "NEW", "OLD"] and closes.index.max() == pd.Timestamp("2026-11-12")


def test_refresh_earnings_once_per_day_and_remembers_empties(tmp_path):
    calls = []

    def fetch(t):
        calls.append(t)
        if t == "ETF":
            return None
        return pd.DataFrame({"ticker": t, "date": pd.to_datetime(["2026-10-28", "2027-01-27"]),
                             "eps_estimate": [1.0, 1.1], "reported_eps": [0.8, np.nan], "surprise_pct": [-20.0, np.nan]})

    now = datetime(2026, 11, 12, 10, tzinfo=ET)
    assert d.refresh_earnings(["XYZ", "ETF"], now, tmp_path, fetch) == {"ETF": "empty", "XYZ": "fetched"}
    assert d.refresh_earnings(["XYZ", "ETF"], now, tmp_path, fetch) == {}
    assert sorted(calls) == ["ETF", "XYZ"]
    d.refresh_earnings(["XYZ"], datetime(2026, 11, 13, 10, tzinfo=ET), tmp_path, fetch)
    ev = d.load_earnings(tmp_path)
    assert len(ev[ev.ticker == "XYZ"]) == 2 and ev.loc[ev.ticker == "ETF", "date"].isna().all()


def test_momentum_context_reads_the_newest_ranking(tmp_path):
    xs_dir, shadow = tmp_path / "xs", tmp_path / "shadow"
    xs_dir.mkdir(), shadow.mkdir()
    for day, top in (("2026-10-02", "OLD"), ("2026-10-30", "NEW")):
        pd.DataFrame({"ticker": [top], "mom_12_1": [0.5], "dv_rank": [1.0]}).to_parquet(xs_dir / f"{day}.parquet")
    pd.DataFrame([{"arm": "mom_top300", "ticker": "NEW", "weight": 0.05, "decision_date": "2026-10-30"},
                  {"arm": "mom_top300", "ticker": "OLD", "weight": 0.05, "decision_date": "2026-10-02"},
                  {"arm": "spy", "ticker": "SPY", "weight": 1.0, "decision_date": "2026-10-30"}]).to_csv(shadow / "ledger.csv")
    (shadow / "marks_latest.txt").write_text(
        "momentum shadow ledger | marked through 2026-11-02 | 2 decision(s) | next scheduled decision 2026-11-27\n"
        "cumulative since the first entry: core_sat_300 -11.03%  mom_top300 -30.41%  spy -2.29%\n")
    ctx = d.load_momentum_context(xs_dir, shadow)
    assert ctx["decision_date"] == "2026-10-30" and ctx["sleeve"] == ["NEW"]
    assert list(ctx["cross_section"]["ticker"]) == ["NEW"]
    assert ctx["record"] == {"marked_through": "2026-11-02", "next_decision": "2026-11-27",
                             "cumulative": {"core_sat_300": "-11.03", "mom_top300": "-30.41", "spy": "-2.29"}}
    assert d.load_momentum_context(tmp_path / "none", shadow)["cross_section"] is None


def test_flag_log_once_per_day_and_equity_upsert(tmp_path):
    assert d.append_flag_log([{"ticker": "A"}, {"ticker": "B"}], "2026-11-12", tmp_path)
    assert not d.append_flag_log([{"ticker": "A"}], "2026-11-12", tmp_path)
    assert d.append_flag_log([{"ticker": "A"}], "2026-11-13", tmp_path)
    lines = (tmp_path / "flag_log.jsonl").read_text().splitlines()
    assert [json.loads(x)["date"] for x in lines] == ["2026-11-12", "2026-11-12", "2026-11-13"]
    d.upsert_equity_log("2026-11-12", 100.0, 500.0, tmp_path)
    d.upsert_equity_log("2026-11-12", 101.0, 501.0, tmp_path)
    d.upsert_equity_log("2026-11-13", 102.0, None, tmp_path)
    log = pd.read_csv(tmp_path / "equity_log.csv", dtype={"date": str})
    assert log["date"].tolist() == ["2026-11-12", "2026-11-13"] and log["net_liq"].tolist() == [101.0, 102.0]
