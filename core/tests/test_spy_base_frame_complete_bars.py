"""The SPY live base frame never caches a forming 10m bar (2026-11-23).

A bucket close fires when the next bucket's first minute arrives, so the 1m buffer always
ends one minute into a forming bar. `_build_base_frame` appended that row to its cached
frame and never recomputed rows at or before the cache's end, so every decision on
2026-11-20 scored its bar as it stood after one minute (archived OHLC == the bucket's
first 1m bar on 28/29 decisions)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import core.API.Alpaca_API.inference.live_inference as li

NY = "America/New_York"


def _minutes(start: str, n: int) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start, tz=NY), periods=n, freq="1min").tz_convert("UTC")
    close = 500.0 + np.arange(n) * 0.01
    return pd.DataFrame({"open": close, "high": close + 0.05, "low": close - 0.05, "close": close,
                         "volume": 1_000.0}, index=idx)


def _resample_stub(df_1m, *, rule, label, closed, tz, assume_tz, **_):
    return li.build_15m(df_1m, rule=rule, label=label, closed=closed, tz=tz, assume_tz=assume_tz)


def _agent(pre: pd.DataFrame):
    a = li.LiveMetaXGBAgent.__new__(li.LiveMetaXGBAgent)
    a._precomputed_base_frame = pre
    a._assume_tz, a._tz = "UTC", NY
    a._label_timeframe_rule, a._resample_label, a._resample_closed = "10min", "left", "left"
    a._precomputed_append_lookback_days = 20
    a._include_pivot_probs = a._include_tb_probs = a._include_vix_features = False
    a._fill_missing_prob, a._session_open, a._session_close = 0.0, "09:30", "16:00"
    a._ga_predictor = a._ga_probs_frame = None
    a._ga_probs_mode = "xgb"
    a._remember_runtime_status = lambda frame: frame
    return a


def test_forming_bar_is_not_cached_and_closed_bar_is_complete(monkeypatch):
    monkeypatch.setattr(li, "build_meta_feature_frame_from_1m", _resample_stub)
    day = _minutes("2026-11-20 09:30", 60)                       # 09:30 .. 10:29
    pre = li.build_15m(day.iloc[:30], rule="10min", label="left", closed="left", tz=NY, assume_tz="UTC")
    agent = _agent(pre)                                           # cached through the 09:50 bar

    # 10:00 bucket closes: the buffer holds 10:00..10:09 plus the 10:10 minute
    frame = agent._build_base_frame(df_1m=day.iloc[:41])
    t1000, t1010 = (pd.Timestamp(f"2026-11-20 {h}", tz=NY) for h in ("10:00", "10:10"))
    assert t1010 not in frame.index                               # one-minute bar not cached
    assert frame.loc[t1000, "close"] == day.iloc[39]["close"]     # 10:09 close: complete bar

    # 10:10 bucket closes: its row must be recomputed from all ten minutes
    frame = agent._build_base_frame(df_1m=day.iloc[:51])
    assert frame.loc[t1010, "close"] == day.iloc[49]["close"]
    assert frame.loc[t1010, "open"] == day.iloc[40]["open"]


def test_session_last_bar_counts_on_its_last_minute():
    frame = li.build_15m(_minutes("2026-11-20 15:40", 20), rule="10min", label="left", closed="left",
                         tz=NY, assume_tz="UTC")
    last = pd.Timestamp("2026-11-20 15:59", tz=NY)
    kept = li.complete_rows_only(frame, last_minute=last, rule="10min")
    assert list(kept.index.strftime("%H:%M")) == ["15:40", "15:50"]
    kept = li.complete_rows_only(frame, last_minute=last - pd.Timedelta(minutes=1), rule="10min")
    assert list(kept.index.strftime("%H:%M")) == ["15:40"]
