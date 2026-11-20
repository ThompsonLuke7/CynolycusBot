import json
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from core.personal_portfolio import analysis as a
from core.personal_portfolio.data import ET, PersonalConfig

CFG = PersonalConfig()


@pytest.mark.parametrize("desc,sym,want", [
    ("DEFIANCE DAILY TARGET 2X LONG MSTR ETF NEW", "MSTX", (2.0, "MSTR")),
    ("TRADR 2XL NVDA DLY", "NVDX", (2.0, "NVDA")),
    ("TRDR 2X LG TSLA DLY", "TSLX", (2.0, "TSLA")),
    ("Roundhill T-REX 2X Long COIN Daily Target ETF", "CONX", (2.0, "COIN")),
    ("INVESTMENT MGRS SER TR II TRADR 2X LONG PLTR DAILY ETF", "PLTX", (2.0, "PLTR")),
    ("DIREXION DAILY TSLA BEAR 1X SHARES", "TSLS", (-1.0, "TSLA")),
    ("DRXN SEMICN BULL 3X", "SOXL", (3.0, None)),
    ("ROUNDHILL MEMORY ETF", "DRAM", (1.0, None)),
    (None, "SPY", (1.0, None)),
])
def test_parse_leveraged_etf(desc, sym, want):
    p = a.parse_leveraged_etf(desc, sym)
    assert (p["leverage"], p["underlying"]) == want


def _series(n=300, seed=1, scale=0.03):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, scale, n), index=pd.bdate_range("2025-06-02", periods=n))


def test_validate_look_through_accepts_a_real_2x_and_rejects_noise():
    und = _series()
    etf = 2 * und + _series(seed=9, scale=0.002)
    ok = a.validate_look_through(etf, und, 2.0)
    assert ok["verified"] and ok["beta"] == pytest.approx(2.0, abs=0.05) and ok["corr"] > 0.99
    assert not a.validate_look_through(_series(seed=4), und, 2.0)["verified"]
    assert not a.validate_look_through(etf, und, 3.0)["verified"]           # wrong stated leverage
    assert not a.validate_look_through(etf.tail(20), und, 2.0)["verified"]  # too little history


def test_vol_drag():
    assert a.vol_drag(2.0, 0.5) == pytest.approx(0.25)
    assert a.vol_drag(1.0, 0.5) == 0.0
    assert a.vol_drag(-1.0, 0.3) == pytest.approx(0.09)
    assert a.vol_drag(2.0, None) is None


def _px(days, values):
    return pd.Series(values, index=days, dtype=float)


def test_shrugged_miss_uses_the_two_day_reaction_vs_spy():
    days = pd.bdate_range("2026-07-01", "2026-11-11")
    spy = _px(days, np.linspace(500, 520, len(days)))
    stock = spy * 0.2
    events = pd.DataFrame({"ticker": "XYZ", "date": pd.to_datetime(["2026-10-25", "2027-01-28"]),
                           "reported_eps": [0.9, np.nan], "surprise_pct": [-10.0, np.nan]})   # Sunday report
    st = a.earnings_status(events, stock, spy, date(2026, 11, 11), date(2026, 11, 12), CFG)
    assert st["shrugged_miss"] is True and st["next_earnings"] == "2027-01-28" and st["days_to_next"] == 77
    rep = st["last_report"]
    assert rep["surprise_group"] == "miss" and rep["reaction_group"] == "flat" and abs(rep["reaction_pct"]) < 0.01
    assert rep["known_date"] == "2026-10-27"   # report session Mon 10-26 (Sunday rolls forward) + 1

    moved = stock.copy()
    moved.loc["2026-10-27":] *= 1.05
    assert a.earnings_status(events, moved, spy, date(2026, 11, 11), date(2026, 11, 12), CFG)["shrugged_miss"] is False
    late = a.earnings_status(events, stock, spy, date(2027, 2, 1), date(2027, 2, 1), CFG)
    assert late["shrugged_miss"] is None and late["pending_report"] == "2027-01-28"   # newer report not in data yet


def test_shrugged_miss_expires_after_the_window():
    days = pd.bdate_range("2026-01-02", "2026-11-11")
    spy = _px(days, np.linspace(500, 520, len(days)))
    events = pd.DataFrame({"ticker": "XYZ", "date": pd.to_datetime(["2026-07-20"]),
                           "reported_eps": [0.9], "surprise_pct": [-10.0]})
    st = a.earnings_status(events, spy * 0.1, spy, date(2026, 11, 11), date(2026, 11, 12), CFG)
    assert st["last_report"]["reaction_group"] == "flat" and st["shrugged_miss"] is False


def _option(strike=50.0, put_call="CALL", qty=1.0, dte=3, s=55.0, avg=2.0, bid=4.9, ask=5.1):
    return {"symbol": f"XYZ   2611{dte:02d}C000{int(strike):02d}000", "underlying": "XYZ", "put_call": put_call,
            "strike": strike, "expiry": "2026-11-20", "quantity": qty, "multiplier": 100.0, "average_price": avg,
            "quote": {"underlying_price": s, "dte": dte, "mark": (bid + ask) / 2, "bid": bid, "ask": ask,
                      "delta": 0.8, "theta": -0.1}}


def test_itm_option_into_expiry_flags_the_exercise_cash():
    out, flags = a.option_review(_option(), None, 1_000.0, CFG, date(2026, 11, 17))
    assert out["exercise_notional"] == 5_000 and out["moneyness"] == pytest.approx(0.1)
    assert out["theta_per_day"] == pytest.approx(-10.0)
    f = next(f for f in flags if f["code"] == "itm_into_expiry")
    assert f["severity"] == "high" and "100 XYZ shares ($5,000)" in f["text"]
    _, flags = a.option_review(_option(), None, 50_000.0, CFG, date(2026, 11, 17))
    assert next(f for f in flags if f["code"] == "itm_into_expiry")["severity"] == "warn"


def test_otm_option_near_expiry_reports_the_move_to_breakeven_and_earnings():
    out, flags = a.option_review(_option(strike=60.0, s=50.0, bid=0.1, ask=0.3), "2026-11-18", 1e6, CFG,
                                 date(2026, 11, 17))
    assert out["move_to_breakeven"] == pytest.approx(62.0 / 50.0 - 1)
    codes = {f["code"] for f in flags}
    assert {"otm_near_expiry", "earnings_before_expiry", "wide_spread"} <= codes
    _, far = a.option_review(_option(strike=60.0, s=50.0, dte=40), None, 1e6, CFG, date(2026, 11, 17))
    assert far == []


def test_portfolio_risk_hist_sim_clusters_and_effective_bets():
    a_r = _series(seed=1)
    rets = pd.DataFrame({"AAA": a_r, "BBB": a_r * 1.0, "CCC": _series(seed=2)})
    risk = a.portfolio_risk({"AAA": 1_000.0, "BBB": 1_000.0, "CCC": 2_000.0}, rets, 10_000.0, CFG)
    pnl = rets.tail(252) @ pd.Series({"AAA": 1_000.0, "BBB": 1_000.0, "CCC": 2_000.0})
    assert risk["hist_sim"]["q05"] == pytest.approx(pnl.quantile(0.05))
    assert risk["hist_sim"]["worst_pct"] == pytest.approx(pnl.min() / 10_000.0)
    assert risk["clusters"][0] == ["AAA", "BBB"]
    assert risk["n_eff"] == pytest.approx(2.0, abs=0.15)   # AAA+BBB are one bet, CCC another


def _bars(n=300):
    days = pd.bdate_range(end="2026-11-11", periods=n)
    rng = np.random.default_rng(7)
    und = rng.normal(0.001, 0.04, n)
    spy = und * 0.3 + rng.normal(0, 0.008, n)
    closes = pd.DataFrame({"XYZ": 50 * np.cumprod(1 + und), "XYZL": 20 * np.cumprod(1 + 2 * und),
                           "SPY": 500 * np.cumprod(1 + spy), "QQQ": 400 * np.cumprod(1 + spy * 1.2),
                           "IWM": 200 * np.cumprod(1 + spy * 1.1), "WWW": 30 * np.cumprod(1 + rng.normal(0, 0.02, n))},
                          index=days)
    return closes


def _holdings():
    return {"fetched_at": "2026-11-12T20:00:00+00:00", "quotes": {"WWW": {"mark": 31.0, "description": "WWW CORP"}},
            "accounts": [{"account": "…678", "type": "MARGIN", "is_day_trader": False, "closing_only": False,
                          "balances": {"liquidation_value": 10_000.0, "available_funds": 1_000.0},
                          "positions": [
                              {"symbol": "XYZL", "raw_symbol": "XYZL", "kind": "etf", "quantity": 100.0,
                               "market_value": 3_000.0, "description": "TRADR 2X LONG XYZ DAILY ETF",
                               "average_price": 25.0, "open_pl": 500.0, "day_pl": 10.0, "multiplier": 1.0},
                              {**_option(strike=50.0, s=55.0, dte=6), "kind": "option", "raw_symbol": "XYZOPT",
                               "market_value": 500.0, "open_pl": 300.0, "day_pl": -5.0}]}]}


def test_build_report_combines_look_through_and_option_exposure():
    closes = _bars()
    earnings = pd.DataFrame({"ticker": ["XYZ"], "date": pd.to_datetime(["2026-11-18"]),
                             "reported_eps": [np.nan], "surprise_pct": [np.nan], "fetched_on": ["2026-11-12"]})
    xs = pd.DataFrame({"ticker": ["XYZ", "WWW", "BIG"], "mom_12_1": [0.5, 0.1, 0.9], "dv_rank": [10, 900, 1]})
    now = datetime(2026, 11, 12, 15, 0, tzinfo=ET)
    rep = a.build_report(_holdings(), closes, earnings, {"cross_section": xs, "sleeve": ["BIG", "XYZ"],
                                                         "decision_date": "2026-10-30"},
                         {}, [{"ticker": "WWW", "note": "idea", "added_at": "2026-11-01"}], {}, CFG, now)
    lt = rep["look_through"]["XYZL"]
    assert lt["verified"] and lt["underlying"] == "XYZ" and lt["drag"] > 0
    xyz = next(i for i in rep["instruments"] if i["instrument"] == "XYZ")
    assert xyz["exposure"] == pytest.approx(3_000 * 2 + 0.8 * 100 * 55.0)
    assert set(xyz["sources"]) == {"XYZL", "XYZ   261106C00050000"}
    assert xyz["momentum"] == {"ranked": True, "mom_12_1": 0.5, "dv_rank": 10, "in_liquid_300": True,
                               "in_sleeve": True, "rank_in_300": 2}
    codes = {f["code"] for f in rep["flags"]}
    assert {"name_concentration", "leveraged_decay", "itm_into_expiry", "earnings_before_expiry",
            "earnings_soon"} <= codes
    assert rep["flags"][0]["severity"] == "high"
    assert rep["totals"]["gross_exposure"] == pytest.approx(xyz["exposure"] / 10_000.0)
    w = rep["watchlist"][0]
    assert w["ticker"] == "WWW" and w["mark"] == 31.0 and w["momentum"]["in_liquid_300"] is False
    assert rep["data"]["bars_current"] is True
    json.dumps(a.sanitize(rep), allow_nan=False)
    rows = a.flag_log_records(a.sanitize(rep))
    assert {(r["ticker"], r["role"]) for r in rows} == {("XYZ", "held"), ("WWW", "watch")}
    assert "name_concentration" in next(r for r in rows if r["ticker"] == "XYZ")["flags"]


def test_unverified_mapping_counts_the_etf_as_itself():
    closes = _bars()
    closes["XYZL"] = 20 * np.cumprod(1 + np.random.default_rng(99).normal(0, 0.03, len(closes)))   # not tracking
    rep = a.build_report(_holdings(), closes, pd.DataFrame(), {}, {}, [], {}, CFG,
                         datetime(2026, 11, 12, 15, 0, tzinfo=ET))
    assert not rep["look_through"]["XYZL"]["verified"]
    assert {i["instrument"] for i in rep["instruments"]} == {"XYZL", "XYZ"}
    assert any(f["code"] == "look_through_unverified" for f in rep["flags"])


def test_hist_sim_caps_a_long_option_loss_at_its_premium():
    r = _series(seed=3)
    rets = pd.DataFrame({"AAA": r})
    legs = [{"instrument": "AAA", "exposure": 5_000.0, "max_loss": 100.0}]
    risk = a.portfolio_risk({"AAA": 5_000.0}, rets, 1_000.0, CFG, legs)
    assert risk["hist_sim"]["worst"] == pytest.approx(max((r.tail(252) * 5_000).min(), -100.0))
    assert risk["hist_sim"]["worst"] >= -100.0
    uncapped = a.portfolio_risk({"AAA": 5_000.0}, rets, 1_000.0, CFG)
    assert uncapped["hist_sim"]["worst"] < -100.0


def test_a_report_already_in_the_data_is_not_upcoming():
    days = pd.bdate_range("2026-07-01", "2026-11-12")
    spy = _px(days, np.linspace(500, 520, len(days)))
    events = pd.DataFrame({"ticker": "XYZ", "date": pd.to_datetime(["2026-11-12", "2027-02-11"]),
                           "reported_eps": [1.2, np.nan], "surprise_pct": [30.0, np.nan]})
    st = a.earnings_status(events, spy * 0.2, spy, date(2026, 11, 11), date(2026, 11, 12), CFG)
    assert st["next_earnings"] == "2027-02-11"


def test_flat_year_loss_is_the_log_drag_in_price_terms():
    assert a.flat_year_loss(a.vol_drag(2.0, 1.0)) == pytest.approx(1 - np.exp(-1.0))
    assert a.flat_year_loss(None) is None


def test_build_report_works_with_copy_on_write_on():
    """The live combined_server imports pandas_ta, which turns copy-on-write on process-wide:
    to_numpy() is then read-only (2026-11-13: 'underlying array is read-only' on the live page)."""
    with pd.option_context("mode.copy_on_write", True):
        rep = a.build_report(_holdings(), _bars(), pd.DataFrame(), {}, {}, [], {}, CFG,
                             datetime(2026, 11, 12, 15, 0, tzinfo=ET))
    assert rep["risk"]["n_eff"] is not None and rep["risk"]["hist_sim"] is not None


def test_long_option_at_half_value_late_in_life_is_flagged_with_history_odds():
    stats = {"n": 117, "max_dte": 14, "breakeven_share": 0.05, "worthless_share": 0.85, "avg_expiry_value": 0.17,
             "n_stopped": 83, "stop_better_share": 0.78}
    _, flags = a.option_review(_option(strike=60.0, s=50.0, dte=10, avg=2.0, bid=0.5, ask=0.7), None, 1e6, CFG,
                               date(2026, 11, 10), stats)
    f = next(f for f in flags if f["code"] == "losing_option_late")
    assert f["severity"] == "warn" and "worth 30% of what you paid with 10d left" in f["text"]
    assert "breakeven by expiry 5%" in f["text"] and "beat holding on 78% of the 83" in f["text"]
    _, early = a.option_review(_option(strike=60.0, s=50.0, dte=20, avg=2.0, bid=0.5, ask=0.7), None, 1e6, CFG,
                               date(2026, 10, 31), stats)
    assert "losing_option_late" not in {f["code"] for f in early}
    _, short = a.option_review(_option(strike=60.0, s=50.0, dte=10, qty=-1.0, avg=2.0, bid=0.5, ask=0.7), None, 1e6,
                               CFG, date(2026, 11, 10), stats)
    assert "losing_option_late" not in {f["code"] for f in short}
