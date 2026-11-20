from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from core.personal_portfolio import trade_analysis as ta
from core.personal_portfolio import trade_bars as tb
from core.personal_portfolio import trade_chart as tc
from core.personal_portfolio.data import ET

ENTRY_FEATURES = ("ret_5d", "ret_20d", "ret_60d", "vs_sma20", "vs_sma50", "vs_sma200", "dist_high_20",
                  "dist_high_252", "vol_20", "spy_ret_20d", "spy_vs_sma50", "spy_vs_sma200", "move_into_entry",
                  "moneyness_entry", "under_entry", "known_session")


def _daily(start="2025-01-02", n=320, drift=0.001, split_on=None, ratio=1.0, seed=1):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    adj = 100 * np.cumprod(1 + drift + 0.01 * rng.standard_normal(n))
    raw = adj.copy()
    if split_on is not None:
        raw[idx < pd.Timestamp(split_on)] *= ratio          # pre-split raw prices are `ratio` times higher
    mk = lambda c: pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6}, index=idx)
    return mk(adj), mk(raw)


def _ep(open_time, close_time, kind="option", put_call="CALL", strike=100.0, fills=None, **kw):
    fills = fills or [{"time": open_time, "role": "entry", "qty": 1, "price": 2.0, "source": "trade"},
                      {"time": close_time, "role": "exit", "qty": -1, "price": 3.0, "source": "trade"}]
    return {"id": "e1", "kind": kind, "put_call": put_call if kind == "option" else None, "strike": strike,
            "direction": "long", "status": "closed", "open_time": open_time, "close_time": close_time,
            "open_date": open_time[:10], "close_date": close_time[:10], "fills": fills, "chart_symbol": "XYZ",
            "underlying": "XYZ", "symbol": "XYZ", "standard_deliverable": True, "description": "XYZ INC",
            "expiry": "2026-03-20", "avg_entry": 2.0, "n_entries": 1, "exit_reason": "closed", "cost": 200.0,
            "hold_days": 10, "dte_at_entry": 60, **kw}


def test_entry_features_ignore_every_bar_after_the_entry():
    split, raw = _daily()
    spy, _ = _daily(seed=2)
    ep = _ep("2026-01-15T15:00:00+00:00", "2026-02-10T15:00:00+00:00")
    base = ta.episode_features(ep, split, raw, spy, None, None)
    later = split.index > pd.Timestamp("2026-01-14")
    split2, raw2, spy2 = split.copy(), raw.copy(), spy.copy()
    for df in (split2, raw2, spy2):
        df.loc[later, ["open", "high", "low", "close"]] *= 3.0
    moved = ta.episode_features(ep, split2, raw2, spy2, None, None)
    assert base["known_session"] == "2026-01-14"
    for k in ENTRY_FEATURES:
        assert base.get(k) == moved.get(k), k
    assert base["under_move_dir"] != moved["under_move_dir"]          # outcomes do look forward


def test_entry_after_the_close_can_use_that_sessions_bar():
    split, raw = _daily()
    out = ta.episode_features(_ep("2026-01-15T21:30:00+00:00", "2026-02-10T15:00:00+00:00"), split, raw, None, None, None)
    assert out["known_session"] == "2026-01-15"


def test_split_inside_the_lookback_is_not_a_crash():
    split, raw = _daily(split_on="2025-12-01", ratio=10.0, drift=0.0)
    out = ta.episode_features(_ep("2026-01-15T15:00:00+00:00", "2026-02-10T15:00:00+00:00"), split, raw, None, None, None)
    assert abs(out["ret_60d"]) < 0.5 and abs(out["vs_sma200"]) < 0.5
    # an option fill priced off raw intraday bars is put on the split-adjusted scale
    bars = pd.DataFrame({"time": [pd.Timestamp("2026-01-15T15:00Z")], "open": [raw.at[pd.Timestamp("2026-01-15"), "close"]],
                         "high": [1.0], "low": [1.0], "close": [1.0]})
    out = ta.episode_features(_ep("2026-01-15T15:00:00+00:00", "2026-02-10T15:00:00+00:00"), split, raw, None, bars, "15Min")
    assert abs(out["move_into_entry"]) < 0.05 and out["under_entry_source"] == "15Min bar open"


def test_direction_sign_and_put_outcome():
    split, raw = _daily(drift=0.002)
    put = _ep("2026-01-15T15:00:00+00:00", "2026-02-10T15:00:00+00:00", put_call="PUT")
    out = ta.episode_features(put, split, raw, None, None, None)
    assert ta.direction_sign(put) == -1 and out["under_move_dir"] < 0   # rising stock hurts a long put
    assert ta.direction_sign({**put, "direction": "short"}) == 1


def test_underlying_class_from_issuer_words():
    assert ta.underlying_class({"kind": "option", "chart_symbol": "TQQQ", "description": "PROSHR QQQ 3X 12/19/2025 $60 Call"}) == "leveraged ETF"
    assert ta.underlying_class({"kind": "option", "chart_symbol": "SVIX", "description": "1X SHRT VIX FUTRS 05/16/2025 $13 Call"}) == "ETF"
    assert ta.underlying_class({"kind": "option", "chart_symbol": "PPLT", "description": "ABRDN PLATINUM PHYS 02/20/2026 $270 Call"}) == "ETF"
    assert ta.underlying_class({"kind": "option", "chart_symbol": "LLY", "description": "ELI LILLY & CO 11/21/2025 $820 Call"}) == "stock"


def test_earnings_context_and_bucket():
    ep = _ep("2026-01-15T15:00:00+00:00", "2026-02-10T15:00:00+00:00")
    dates = [pd.Timestamp("2025-10-30"), pd.Timestamp("2026-02-05"), pd.Timestamp("2026-05-01")]
    ctx = ta.earnings_context(ep, dates)
    assert ctx == {"earnings_known": True, "days_to_earnings": 21, "next_earnings_at_entry": "2026-02-05",
                   "held_through_earnings": True}
    assert ta._earnings_bucket(ctx) == "held through earnings"
    early = ta.earnings_context({**ep, "close_date": "2026-02-04"}, dates)
    assert ta._earnings_bucket(early) == "no report while held"
    assert ta.earnings_context({**ep, "kind": "etf"}, dates) == {"earnings_known": False}


def test_summarize_and_breakdown():
    rows = [{"pnl": p, "return_pct": p / 100, "hold_days": 2, "b_type": t} for p, t in
            [(100, "Call"), (-50, "Call"), (-50, "Put"), (30, "Put"), (-10, "Put"), (20, "Call")]]
    s = ta.summarize(rows)
    assert s["n"] == 6 and s["wins"] == 3 and s["total_pnl"] == 40 and s["profit_factor"] == pytest.approx(150 / 110)
    groups = {g["label"]: g for g in ta.breakdown(rows, "b_type")}
    assert groups["Call"]["total_pnl"] == 70 and groups["Put"]["n"] == 3 and groups["Put"]["ci_avg_pnl"] is None


# ---------- bars ----------
def _fake_fetch(calls):
    def fetch(sym, start, end, tf, adj):
        calls.append((sym, tf, adj))
        idx = pd.bdate_range(start.date(), end.date(), tz=ET)[:-1] + pd.Timedelta(hours=9, minutes=30)
        if tf != "1Day":
            idx = pd.date_range(start, end, freq="15min", tz="UTC")
        n = len(idx)
        return pd.DataFrame({"timestamp": idx, "open": np.full(n, 10.0), "high": 11.0, "low": 9.0, "close": 10.5,
                             "volume": 100.0})
    return fetch


def test_bar_plan_merges_ranges_and_marks_active_symbols():
    now = datetime(2026, 11, 16, 22, tzinfo=timezone.utc)
    eps = [{"chart_symbol": "AAA", "open_date": "2025-01-10", "close_date": "2025-02-01", "status": "closed"},
           {"chart_symbol": "AAA", "open_date": "2026-11-02", "close_date": None, "status": "open"},
           {"chart_symbol": "BBB", "open_date": "2023-03-01", "close_date": "2023-03-05", "status": "closed"}]
    plan = tb.bar_plan(eps, ["CCC"], now)
    assert plan["AAA"]["active"] and not plan["BBB"]["active"] and plan["CCC"]["active"]
    assert str(plan["BBB"]["end"]) == "2023-05-04" and str(plan["AAA"]["start"]) == "2023-12-07"


def test_refresh_daily_fetches_once_for_closed_trades_and_again_when_active(tmp_path):
    calls = []
    fetch = _fake_fetch(calls)
    eps = [{"chart_symbol": "OLD", "open_date": "2023-03-01", "close_date": "2023-03-05", "status": "closed"}]
    day1 = datetime(2026, 11, 13, 22, tzinfo=timezone.utc)
    plan = tb.bar_plan(eps, ["HELD"], day1)
    assert tb.refresh_daily(plan, day1, tmp_path, fetch) == {"HELD": "fetched", "OLD": "fetched"}
    assert len(calls) == 4 and {c[2] for c in calls} == {"raw", "split"}
    day2 = datetime(2026, 11, 16, 22, tzinfo=timezone.utc)
    assert tb.refresh_daily(tb.bar_plan(eps, ["HELD"], day2), day2, tmp_path, fetch) == {"HELD": "fetched", "OLD": "current"}
    assert tb.load_daily("OLD", "raw", tmp_path).index[0] == pd.Timestamp("2022-01-25")


def test_empty_symbols_are_recorded_and_not_refetched(tmp_path):
    calls = []
    now = datetime(2026, 11, 16, 22, tzinfo=timezone.utc)
    plan = tb.bar_plan([{"chart_symbol": "DEAD", "open_date": "2021-03-01", "close_date": "2021-03-05",
                         "status": "closed"}], [], now)
    empty = lambda *a: (calls.append(a), pd.DataFrame())[1]  # noqa: E731
    assert tb.refresh_daily(plan, now, tmp_path, empty) == {"DEAD": "empty"}
    assert tb.refresh_daily(plan, now, tmp_path, empty) == {"DEAD": "current"} and len(calls) == 2


def test_intraday_timeframe_depends_on_the_window():
    now = datetime(2026, 11, 16, 22, tzinfo=timezone.utc)
    short = {"open_date": "2026-10-01", "close_date": "2026-10-03", "status": "closed"}
    mid = {"open_date": "2026-06-01", "close_date": "2026-08-01", "status": "closed"}
    long = {"open_date": "2025-01-02", "close_date": "2026-01-02", "status": "closed"}
    assert tb.intraday_spec(short, now)["timeframe"] == "15Min"
    assert tb.intraday_spec(mid, now)["timeframe"] == "1Hour"
    assert tb.intraday_spec(long, now) is None


def test_price_at_never_reads_past_the_fill():
    bars = pd.DataFrame({"time": pd.to_datetime(["2026-01-15T14:30Z", "2026-01-15T14:45Z"]),
                         "open": [10.0, 11.0], "close": [10.8, 11.5]})
    assert tb.price_at(bars, datetime(2026, 1, 15, 14, 50, tzinfo=timezone.utc), "15Min") == (11.0, "15Min bar open")
    assert tb.price_at(bars, datetime(2026, 1, 15, 21, 0, tzinfo=timezone.utc), "15Min") == (11.5, "last close before the fill")
    assert tb.price_at(bars, datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc), "15Min")[0] is None


# ---------- charts ----------
def test_episode_chart_daily_markers_points_and_strike(tmp_path):
    calls = []
    now = datetime(2026, 11, 16, 22, tzinfo=timezone.utc)
    stock_ep = _ep("2026-10-05T15:00:00+00:00", "2026-10-09T19:00:00+00:00", kind="equity", strike=None,
                   fills=[{"time": "2026-10-05T15:00:00+00:00", "role": "entry", "qty": 10, "price": 10.2, "source": "trade",
                           "cash": -102, "fees": 0, "symbol": "XYZ"},
                          {"time": "2026-10-05T16:00:00+00:00", "role": "entry", "qty": 10, "price": 10.4, "source": "trade",
                           "cash": -104, "fees": 0, "symbol": "XYZ"},
                          {"time": "2026-10-09T19:00:00+00:00", "role": "exit", "qty": -20, "price": 11.0, "source": "trade",
                           "cash": 220, "fees": 0, "symbol": "XYZ"}], avg_entry=10.3)
    tb.refresh_daily(tb.bar_plan([stock_ep], [], now), now, tmp_path, _fake_fetch(calls))
    d = tc.episode_chart(stock_ep, "1d", now, tmp_path)
    assert [m["time"] for m in d["markers"]] == ["2026-10-05", "2026-10-05", "2026-10-09"]
    assert d["points"]["entry"] == [{"time": "2026-10-05", "value": pytest.approx(10.3)}]
    assert d["points"]["exit"] == [{"time": "2026-10-09", "value": 11.0}]
    assert d["price_lines"][0]["title"] == "avg entry" and d["visible"]["from"] < "2026-10-05" < d["visible"]["to"]
    opt = _ep("2026-10-05T15:00:00+00:00", "2026-10-09T19:00:00+00:00")
    assert [p["title"] for p in tc.episode_chart(opt, "1d", now, tmp_path)["price_lines"]] == ["strike", "breakeven at expiry"]
    assert tc.ticker_chart("XYZ", [stock_ep], tmp_path)["n_episodes"] == 1


def test_intraday_chart_times_are_et_wall_clock():
    assert tc._et_seconds(pd.Timestamp("2026-07-01T13:30Z")) == int(pd.Timestamp("2026-07-01T09:30").timestamp())
    assert tc._et_seconds(pd.Timestamp("2026-12-01T14:30Z")) == int(pd.Timestamp("2026-12-01T09:30").timestamp())


def test_trade_review_report_builds_from_episodes():
    from core.personal_portfolio import trade_history as th
    from core.personal_portfolio import trade_review as tr
    from core.personal_portfolio.tests.test_trade_history import _opt_inst, stock, trade

    raw = []
    for i in range(12):
        inst = _opt_inst(f"XYZ   2612{10 + i:02d}C00010000", "XYZ", f"2026-12-{10 + i:02d}", "CALL", 10.0)
        raw += [trade(f"2026-0{1 + i % 9}-05T15:00:00+0000", inst, 1, 1.0, "OPENING"),
                trade(f"2026-0{1 + i % 9}-08T15:00:00+0000", inst, -1, 0.5 + (i % 3) * 0.5, "CLOSING")]
    raw.append(trade("2025-06-02T15:00:00+0000", stock("ABC"), 10, 10.0, "OPENING"))
    raw.append(trade("2025-06-09T15:00:00+0000", stock("ABC"), -10, 12.0, "CLOSING"))
    fills, issues = th.normalize_fills(raw)
    eps, checks = th.build_episodes(fills)
    for e in eps:
        e["ctx"] = {"underlying_class": "stock"}
        e.update(ta.buckets(e, e["ctx"]))
    text = tr.build_report(eps, checks, issues, None, datetime(2026, 11, 16, 22, tzinfo=timezone.utc))
    assert "13 resolved round trips" in text and "## Entry conditions" in text and "Held to expiry: 0 trades" in text
