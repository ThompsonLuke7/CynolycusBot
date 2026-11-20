"""At the live edge (today's daily bar not cached yet) daily features equal what history computes.

Regression for 2026-11-19: the daily block's .shift(1) lag plus a backward as-of
made live daily_* features one session staler than every training row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.momentum_expansion.data.session_open import append_session_open, session_day
from strategies.momentum_expansion.features.feature_matrix_4h import build_ticker_features_4h
from strategies.momentum_expansion.live import runner as mr

DAILY = ["daily_ret_1", "daily_ret_5", "daily_atr_pct", "daily_rsi_14", "daily_trend_state",
         "daily_dist_52w_atr", "daily_ema_stack", "daily_dist_200dma_atr", "daily_new_high_252",
         "daily_gap_pct", "daily_gap_z_60", "weekly_ret_1", "weekly_ret_4", "weekly_trend_state"]
GAP = ["daily_gap_pct", "daily_gap_z_60"]


def _bars(n_days: int = 320, seed: int = 0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-06-02", periods=n_days)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
    opn = close * np.exp(rng.normal(0, 0.01, n_days))
    daily = pd.DataFrame({"open": opn, "high": np.maximum(opn, close) * 1.01,
                          "low": np.minimum(opn, close) * 0.99, "close": close,
                          "volume": rng.integers(1e6, 2e6, n_days).astype(float)},
                         index=(days + pd.Timedelta(hours=5)).tz_localize("UTC"))
    rows = []
    for d, o, c in zip(days, opn, close):
        for hour, (bo, bc) in ((15, (o, (o + c) / 2)), (19, ((o + c) / 2, c))):   # 10:00 / 14:00 ET (EST-ish)
            rows.append({"ts": pd.Timestamp(d) + pd.Timedelta(hours=hour), "open": bo,
                         "high": max(bo, bc) * 1.002, "low": min(bo, bc) * 0.998, "close": bc,
                         "volume": 5e5})
    b4 = pd.DataFrame(rows).set_index("ts")
    b4.index = b4.index.tz_localize("UTC")
    return daily, b4


def _feats(b4, daily):
    out = build_ticker_features_4h(ticker="TST", df_4h=b4, df_1d=daily, ctx_4h={}, ticker_meta={})
    return out[DAILY]


def _last_day_rows(frame, b4):
    last = b4.index.tz_convert("America/New_York").normalize().max()
    return frame.loc[frame.index.tz_convert("America/New_York").normalize() == last]


def test_live_edge_matches_history_for_lagged_features():
    daily, b4 = _bars()
    full = _last_day_rows(_feats(b4, daily), b4)
    edge = _last_day_rows(_feats(b4, daily.iloc[:-1]), b4)      # live: today's daily bar absent
    lagged = [f for f in DAILY if f not in GAP]
    pd.testing.assert_frame_equal(edge[lagged], full[lagged])
    # Without today's open the gap keeps its old meaning: the previous session's gap.
    prev = _feats(b4, daily.iloc[:-1])
    prev_day = prev.loc[prev.index.tz_convert("America/New_York").normalize()
                        == daily.index[-2].tz_convert("America/New_York").normalize()]
    assert np.allclose(edge[GAP].iloc[0], prev_day[GAP].iloc[0])


def test_official_open_makes_every_daily_feature_match():
    daily, b4 = _bars(seed=3)
    full = _last_day_rows(_feats(b4, daily), b4)
    day = daily.index[-1].tz_convert("America/New_York").tz_localize(None).normalize()
    partial = append_session_open(daily.iloc[:-1], float(daily["open"].iloc[-1]), day)
    edge = _last_day_rows(_feats(b4, partial), b4)
    pd.testing.assert_frame_equal(edge, full)


def test_history_rows_are_untouched_by_the_edge_logic():
    daily, b4 = _bars(seed=5)
    full = _feats(b4, daily)
    edge = _feats(b4, daily.iloc[:-1])
    hist = full.index[full.index.tz_convert("America/New_York").normalize()
                      < daily.index[-1].tz_convert("America/New_York").normalize()]
    pd.testing.assert_frame_equal(edge.loc[hist], full.loc[hist])


def test_a_cache_two_sessions_behind_gives_nan_not_stale_values():
    daily, b4 = _bars(seed=7)
    edge = _last_day_rows(_feats(b4, daily.iloc[:-2]), b4)
    assert edge["daily_ret_1"].isna().all() and edge["daily_atr_pct"].isna().all()


def test_append_session_open_only_adds_a_missing_day():
    daily, _ = _bars(n_days=70)
    day = daily.index[-1].tz_convert("America/New_York").tz_localize(None).normalize()
    assert append_session_open(daily, 10.0, day) is daily                       # already covered
    out = append_session_open(daily.iloc[:-1], 10.0, day)
    assert len(out) == len(daily) and out["open"].iloc[-1] == 10.0 and np.isnan(out["close"].iloc[-1])
    assert out.index[-1].tz_convert("America/New_York").date() == daily.index[-1].tz_convert("America/New_York").date()
    assert append_session_open(daily.iloc[:-1], None, day) is not None
    assert session_day(pd.Timestamp("2026-11-19 03:00", tz="UTC").to_pydatetime()) == pd.Timestamp("2026-11-18")


def test_loader_fetches_opens_once_and_only_when_needed():
    daily, _ = _bars(n_days=70)
    day = daily.index[-1].tz_convert("America/New_York").tz_localize(None).normalize()
    calls = []
    covered = mr.SessionOpenDailyLoader(["A", "B"], base=lambda t: daily, day=day,
                                        fetch=lambda ts: calls.append(ts) or {"A": 9.0})
    for t in ("A", "B"):
        covered(t)
    assert len(calls) == 1                                    # lazy, once per panel
    loader = mr.SessionOpenDailyLoader(["A", "B"], base=lambda t: daily.iloc[:-1], day=day,
                                       fetch=lambda ts: {"A": 9.0})
    assert loader("A")["open"].iloc[-1] == 9.0
    assert len(loader("B")) == len(daily) - 1                 # no open known: unchanged
    broken = mr.SessionOpenDailyLoader(["A"], base=lambda t: daily.iloc[:-1], day=day,
                                       fetch=lambda ts: (_ for _ in ()).throw(RuntimeError("down")))
    assert len(broken("A")) == len(daily) - 1 and broken.opens == {}


def test_feature_universe_adds_the_research_universe(monkeypatch):
    import core.shared_universe.universe as su
    monkeypatch.setattr(su, "shared_tickers", lambda **k: ["AAA", "BBB", "SPY", "CCC"])
    out = mr.feature_universe(["BBB", "DDD"])
    assert set(out) == {"AAA", "BBB", "CCC", "DDD"} or set(out) >= {"AAA", "BBB", "CCC", "DDD"}
    assert "DDD" in out and len(out) == len(set(out))


def test_panel_passes_research_metadata_when_asked(monkeypatch):
    import strategies.momentum_expansion.features.live_feature_panel_4h as lp

    seen = {}

    def fake_build(*, ticker, df_4h, df_1d, ctx_4h, ticker_meta=None):
        seen[ticker] = ticker_meta
        out = df_4h[["close"]].copy()
        out.index.name = None
        out["sector_id"] = (ticker_meta or {}).get("sector_id", 0.0)
        return out

    monkeypatch.setattr(lp, "build_ticker_features_4h", fake_build)
    monkeypatch.setattr(lp, "_add_cross_sectional_features", lambda p: p)
    monkeypatch.setattr(lp, "_add_earnings_features_4h", lambda p: p)
    _, b4 = _bars(n_days=5)
    res = lp.build_live_feature_panel_4h(tickers=["MZZ1", "MZZ2"], load_4h_bars=lambda t: b4,
                                         load_1d_bars=lambda t: None, ctx_4h={}, tail=1,
                                         ticker_meta=lambda t: {"sector_id": 7.0 if t == "MZZ1" else 3.0})
    assert seen == {"MZZ1": {"sector_id": 7.0}, "MZZ2": {"sector_id": 3.0}}
    assert sorted(res.panel["sector_id"].tolist()) == [3.0, 7.0]
