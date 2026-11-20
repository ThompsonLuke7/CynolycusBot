"""Stage-3 point-in-time build: life windows, the per-bar universe, delisting exits, pre-registered sets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.momentum_expansion.ablation import build_stage3_pit as B
from strategies.momentum_expansion.ablation.export_stage2 import CALENDAR_FEATURES, EARNINGS_FEATURES, TRAINER


def _hourly(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Live-style RTH hourly bars: 10:00..16:00 ET starts (14:00..20:00 UTC in summer)."""
    ts = [d + pd.Timedelta(hours=h) for d in days for h in range(14, 21)]
    n = len(ts)
    return pd.DataFrame({"timestamp": pd.DatetimeIndex(ts, tz="UTC"), "open": 10.0, "high": 10.5, "low": 9.5,
                         "close": np.linspace(10, 12, n), "volume": 1000.0})


def test_life_window_keeps_one_life_and_drops_padding():
    days = pd.bdate_range("2024-06-03", periods=20)
    h = _hourly(days)
    h.loc[h["timestamp"].dt.date == days[5].date(), "volume"] = 0.0     # Alpaca's zero-volume padding
    first = pd.Timestamp(days[2], tz="America/New_York").tz_convert("UTC")
    last = pd.Timestamp(days[9], tz="America/New_York").tz_convert("UTC")
    b4 = B.life_4h(h, first, last)
    sess = pd.DatetimeIndex(b4.index).tz_convert("America/New_York").normalize().tz_localize(None)
    assert sess.min() == days[2] and sess.max() == days[9]
    assert days[5] not in set(sess)
    assert (b4.groupby(sess).size() == 2).all()                           # two 4H bars per session
    assert len(B.life_4h(h, first, None)) > len(b4)                       # a live security keeps its tail


def test_a_series_with_non_positive_adjusted_prices_is_refused():
    ok = pd.DataFrame({"open": [1.0, 2.0], "high": [1.5, 2.5], "low": [0.9, 1.8], "close": [1.2, 2.2]})
    assert B.valid_prices(ok) and B.valid_prices(None)
    bad = ok.copy()
    bad.loc[0, "low"] = -0.4                        # Alpaca's dividend adjustment overshooting
    assert not B.valid_prices(bad)


def test_universe_takes_the_most_traded_names_above_the_price_floor():
    df = pd.DataFrame({"timestamp": ["t1"] * 5 + ["t2"] * 3, "ticker": list("abcdeabc"),
                       "adv20_lag": [5, 4, 3, 2, 1, 1, 9, np.nan], "close_lag": [0.5, 2, 2, 2, 2, 2, 2, 2]})
    out = B.select_universe(df, size=2)
    assert sorted(out.loc[out["timestamp"] == "t1", "ticker"]) == ["b", "c"]     # a is below $1
    assert sorted(out.loc[out["timestamp"] == "t2", "ticker"]) == ["a", "b"]     # c has no volume history


def test_a_delisted_window_is_priced_against_spy_over_the_same_window():
    days = pd.bdate_range("2024-01-02", periods=80)
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in (15, 19)], tz="UTC")
    spy = pd.DataFrame({"open": np.linspace(400, 430, len(idx)), "high": 440.0, "low": 390.0,
                        "close": np.linspace(401, 431, len(idx)), "volume": 1e6}, index=idx)
    cal = pd.DatetimeIndex(sorted(set(B._session(idx))))
    spy_fwd, spy_entry, spy_close = B.spy_frames(spy, cal)
    t = idx[41]                                                           # afternoon of session 20
    row = {"timestamp": t, "close_lag": 5.0, "adv20_lag": 1e7, "daily_atr_pct": 0.03, "l0_mfe": 0.1,
           "piv_long": 0.0, "last_sess": 23.0}
    for h in B.HORIZONS:
        row.update({f"fret_{h}": -0.5, f"fmfe_{h}": 0.1, f"clamped_{h}": 1.0})
    live = dict(row, ticker="LIVE", last_sess=np.nan, **{f"clamped_{h}": 0.0 for h in B.HORIZONS})
    df = pd.DataFrame([dict(row, ticker="DEAD"), live])
    out = B.cross_section(df, spy_fwd, spy_entry, spy_close, seed=0).set_index("ticker")
    expect = spy_close[23] / spy["open"].iloc[42] - 1                       # SPY: same entry bar, exit session 23
    assert out.loc["DEAD", "spy_fret_10"] == pytest.approx(expect, rel=1e-6)
    assert out.loc["LIVE", "spy_fret_10"] == pytest.approx(spy_fwd.loc[t, "spy_fret_10"], rel=1e-6)


def test_feature_set_and_arms_follow_the_preregistration():
    feats = B.pit_features()
    assert len(feats) == B.N_PIT_FEATURES == 82
    assert not set(feats) & (set(CALENDAR_FEATURES) | set(EARNINGS_FEATURES) | set(B.SURVIVOR_ONLY))
    assert "low_price_flag" in feats                                      # point-in-time price, kept
    arms = B.arms()
    assert [a["name"] for a in arms] == ["L0__pit", "L0__pit__perm", "L2_10__pit", "L2_10__pit__perm",
                                         "L2_20__pit", "L2_20__pit__perm"]
    for real, ctrl in zip(arms[::2], arms[1::2]):
        assert ctrl["target"] == real["target"] + "__perm" and ctrl["control"] and not real["control"]
    assert TRAINER.read_text().count('STAGE = "stage2"') == 1             # the line the export pins
