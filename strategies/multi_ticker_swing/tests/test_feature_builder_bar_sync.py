"""LiveSwingFeatureBuilder bars match the training bars (2026-11-23 parity fix).

Live 30m parity failed on ~70% of rows for core features because the stream-built bars
were not the research bars: end-of-bucket stamps, partial/duplicated buckets at restarts,
no 16:00 bucket. These pin the stamp rule, the restart merge and the REST splice."""
from collections import deque
from datetime import datetime, timezone

import pandas as pd

from strategies.multi_ticker_swing.live import feature_builder as fbm


def _builder():
    b = fbm.LiveSwingFeatureBuilder.__new__(fbm.LiveSwingFeatureBuilder)
    import threading
    b._bars, b._last_ts, b._synced_through, b.last_sync = {}, {}, {}, {}
    b._lock = threading.Lock()
    return b


def _bar(ts, o=10.0, h=11.0, lo=9.0, c=10.5, v=100.0):
    return {"timestamp": ts, "open": o, "high": h, "low": lo, "close": c, "volume": v}


def _utc(s):
    return pd.Timestamp(s, tz="America/New_York").tz_convert("UTC").to_pydatetime()


def test_append_stamps_bucket_start_not_last_minute():
    b = _builder()
    b.append_bar("AAA", _bar(_utc("2026-11-20 15:29")))
    b.append_bar("BBB", _bar(_utc("2026-11-20 15:27")))     # no print at :29
    for t in ("AAA", "BBB"):
        assert pd.Timestamp(b._bars[t][-1]["timestamp"]) == pd.Timestamp(_utc("2026-11-20 15:00"))


def test_restart_bucket_merges_into_prefilled_forming_bar():
    b = _builder()
    b._bars["AAA"] = deque([_bar(pd.Timestamp(_utc("2026-11-20 11:30")), o=10, h=10.4, lo=9.8, c=10.2, v=300)],
                           maxlen=fbm.WINDOW_SIZE)
    b.append_bar("AAA", _bar(_utc("2026-11-20 11:59"), o=10.2, h=10.9, lo=10.1, c=10.7, v=500))
    assert len(b._bars["AAA"]) == 1
    m = b._bars["AAA"][-1]
    assert (m["open"], m["high"], m["low"], m["close"], m["volume"]) == (10, 10.9, 9.8, 10.7, 800)


def test_sync_replaces_covered_buckets_and_keeps_newer_stream_bar():
    b = _builder()
    for ts in ("2026-11-20 14:29", "2026-11-20 14:59", "2026-11-20 15:29"):
        b.append_bar("AAA", _bar(_utc(ts), c=1.0))
    rest = pd.DataFrame({"symbol": "AAA", "timestamp": [_utc("2026-11-20 14:00"), _utc("2026-11-20 14:30")],
                         "open": 2.0, "high": 2.5, "low": 1.5, "close": 2.0, "volume": 900.0})
    seen = {}

    def fetch(symbols, start, end):
        seen["symbols"], seen["start"] = symbols, start
        return rest

    st = b.sync_recent_bars(["AAA", "NOPE"], now=_utc("2026-11-20 16:00:30"), fetch=fetch)
    closes = [x["close"] for x in b._bars["AAA"]]
    assert closes == [2.0, 2.0, 1.0]                 # 14:00/14:30 from REST, 15:00 kept (REST lag)
    assert st["synced"] == 1 and st["kept_newer"] == 1 and st["error"] is None
    assert seen["symbols"] == ["AAA"]                # no deque -> not requested
    # a later stream bar for a bucket REST already holds is ignored, not double counted
    b.append_bar("AAA", _bar(_utc("2026-11-20 14:59"), c=9.0, v=1.0))
    assert [x["close"] for x in b._bars["AAA"]] == [2.0, 2.0, 1.0]


def test_sync_drops_forming_rest_bar_and_is_incremental():
    b = _builder()
    b.append_bar("AAA", _bar(_utc("2026-11-20 14:59")))
    rest = pd.DataFrame({"symbol": "AAA", "timestamp": [_utc("2026-11-20 14:30"), _utc("2026-11-20 15:00")],
                         "open": 2.0, "high": 2.5, "low": 1.5, "close": 2.0, "volume": 900.0})
    starts = []

    def fetch(symbols, start, end):
        starts.append(pd.Timestamp(start))
        return rest

    b.sync_recent_bars(["AAA"], now=_utc("2026-11-20 15:10"), fetch=fetch)
    assert [pd.Timestamp(x["timestamp"]) for x in b._bars["AAA"]] == [pd.Timestamp(_utc("2026-11-20 14:30"))]
    assert b._synced_through["AAA"] == pd.Timestamp(_utc("2026-11-20 14:30"))
    b.sync_recent_bars(["AAA"], now=_utc("2026-11-20 15:20"), fetch=fetch)   # nothing new closed
    assert len(starts) == 1
    b.sync_recent_bars(["AAA"], now=_utc("2026-11-20 15:31"), fetch=fetch)
    assert starts[1] == pd.Timestamp(_utc("2026-11-20 15:00"))               # only after the last synced


def test_sync_failure_leaves_deque_and_reports():
    b = _builder()
    b.append_bar("AAA", _bar(_utc("2026-11-20 14:59")))

    def boom(*a, **k):
        raise RuntimeError("rate limited")

    st = b.sync_recent_bars(["AAA"], now=_utc("2026-11-20 15:31"), fetch=boom)
    assert "rate limited" in st["error"] and len(b._bars["AAA"]) == 1


def test_save_parquet_writes_only_complete_bars(tmp_path, monkeypatch):
    monkeypatch.setattr(fbm, "RAW_30M_DIR", tmp_path)
    b = _builder()
    b._bars["AAA"] = deque([_bar(pd.Timestamp(_utc("2026-11-20 11:00"))),
                            _bar(pd.Timestamp(_utc("2026-11-20 11:30")))], maxlen=fbm.WINDOW_SIZE)
    b._save_parquet("AAA", complete_before=_utc("2026-11-20 11:39"))
    saved = pd.read_parquet(tmp_path / "AAA.parquet")
    assert list(pd.to_datetime(saved.index, utc=True)) == [pd.Timestamp(_utc("2026-11-20 11:00"))]


def test_bucket_start_naive_is_utc():
    assert fbm.bucket_start(datetime(2026, 11, 20, 20, 29)) == pd.Timestamp("2026-11-20 20:00", tz="UTC")
    assert fbm.bucket_start(datetime(2026, 11, 20, 20, 29, tzinfo=timezone.utc)) == pd.Timestamp(
        "2026-11-20 20:00", tz="UTC")


def test_daily_scope_from_synthetic_history_equals_full_history():
    """daily_* and 52w/60d/gap features from (daily history + 500-bar deque) equal the
    research functions run on the full intraday history."""
    import numpy as np
    from signals.location_features import add_daily_high_low_gap_features
    from strategies.multi_ticker_swing.features.build_features import _add_daily_context

    rng = np.random.default_rng(7)
    days = pd.bdate_range("2025-01-02", periods=320)
    stamps = [pd.Timestamp(f"{d.date()} 09:30", tz="America/New_York") + pd.Timedelta(minutes=30 * i)
              for d in days for i in range(14)]                              # 09:30..16:00 buckets
    idx = pd.DatetimeIndex(stamps).tz_convert("UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx))))
    gap = np.repeat(rng.normal(0, 0.01, len(days)), 14)                     # overnight gaps
    close = close * np.exp(np.cumsum(np.where(np.arange(len(idx)) % 14 == 0, gap, 0)))
    full = pd.DataFrame({"open": close * (1 + rng.normal(0, 0.001, len(idx))), "close": close,
                         "volume": rng.integers(1_000, 9_000, len(idx)).astype(float)}, index=idx)
    full["high"] = full[["open", "close"]].max(axis=1) * 1.002
    full["low"] = full[["open", "close"]].min(axis=1) * 0.998

    sd = pd.Series(full.index.tz_convert("America/New_York").date, index=full.index)
    want_daily = _add_daily_context(full.copy(), sd).iloc[-1]
    want_loc, cols = add_daily_high_low_gap_features(full.copy())
    deque_frame = full.tail(fbm.WINDOW_SIZE)                                 # starts mid-session
    first = deque_frame.index.tz_convert("America/New_York").normalize().min()
    hist = fbm.daily_ohlcv(full)
    got = fbm.daily_scope_features(deque_frame, hist[hist.index <= first])
    for c in [c for c in want_daily.index if c.startswith("daily_")] + cols:
        want = want_daily[c] if c.startswith("daily_") else want_loc[c].iloc[-1]
        assert np.isclose(got[c], want, rtol=1e-9, equal_nan=True), (c, got[c], want)


def test_daily_scope_cache_matches_uncached_within_a_session():
    import numpy as np
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2025-06-02", periods=80)
    idx = pd.DatetimeIndex([pd.Timestamp(f"{d.date()} 09:30", tz="America/New_York") + pd.Timedelta(minutes=30 * i)
                            for d in days for i in range(14)]).tz_convert("UTC")
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.005, len(idx))))
    full = pd.DataFrame({"open": c, "high": c * 1.003, "low": c * 0.997, "close": c, "volume": 1e4}, index=idx)
    hist = fbm.daily_ohlcv(full)
    cache = {}
    for n in range(len(full) - 6, len(full) + 1):              # successive bars of the last session
        frame = full.iloc[:n].tail(fbm.WINDOW_SIZE)
        first = frame.index.tz_convert("America/New_York").normalize().min()
        h = hist[hist.index <= first]
        _same(fbm.daily_scope_features(frame, h, cache=cache, cache_key="X"), fbm.daily_scope_features(frame, h))
    # a changed prior-session bar invalidates the cached daily values
    frame = full.tail(fbm.WINDOW_SIZE).copy()
    frame.iloc[-15, frame.columns.get_loc("close")] *= 1.5
    first = frame.index.tz_convert("America/New_York").normalize().min()
    h = hist[hist.index <= first]
    _same(fbm.daily_scope_features(frame, h, cache=cache, cache_key="X"), fbm.daily_scope_features(frame, h))


def _same(a: dict, b: dict) -> None:
    import numpy as np
    assert a.keys() == b.keys()
    for k in a:
        assert np.isclose(a[k], b[k], rtol=0, atol=0, equal_nan=True), (k, a[k], b[k])


def test_spy_frame_is_reused_until_spy_or_context_bars_change(monkeypatch):
    b = _builder()
    calls = []
    monkeypatch.setattr(b, "_feature_frame", lambda t: calls.append(t) or f"frame{len(calls)}", raising=False)
    b.append_bar("SPY", _bar(_utc("2026-11-20 14:29")))
    assert b._spy_feature_frame() == "frame1" and b._spy_feature_frame() == "frame1"
    b.append_bar("QQQ", _bar(_utc("2026-11-20 14:29")))          # a context ticker moved
    assert b._spy_feature_frame() == "frame2"
    b.append_bar("SPY", _bar(_utc("2026-11-20 14:59")))
    assert b._spy_feature_frame() == "frame3" and calls == ["SPY", "SPY", "SPY"]
