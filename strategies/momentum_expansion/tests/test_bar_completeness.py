"""Partial intraday bars must never be cached, and the 4H tail must be rebuilt.

Regression for 2026-11-22: intraday catch-ups (end = now - 20 min) cached the still-forming
hour and resumed at ``last + 1 min``, so 13:00-15:00 ET hourly bars stayed partial on ~75%
of sessions since 2026-07. Separately, ``build_4h_for`` skipped the rebuild while hours were
still arriving, so the 2026-11-20 14:32 ET momentum pass scored a 10:00 ET 4H bar holding
only its first hour. Volumes below are AAPL 2026-11-20 (SIP, complete vs as cached).
"""
from __future__ import annotations

import pandas as pd
import pytest

from strategies.momentum_expansion.data import bars

HOURS_UTC = ["15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00"]
FULL_VOL = [6843295, 5490499, 3910584, 5889040, 7968777, 16497382, 22230464]


def _hourly(day: str, hours: list[str], vols: list[float], symbol: str = "AAPL") -> pd.DataFrame:
    ts = pd.to_datetime([f"{day} {h}" for h in hours], utc=True)
    n = len(ts)
    return pd.DataFrame({
        "symbol": [symbol] * n, "timestamp": ts,
        "open": [100.0 + i for i in range(n)], "high": [101.0 + i for i in range(n)],
        "low": [99.0 + i for i in range(n)], "close": [100.5 + i for i in range(n)],
        "volume": [float(v) for v in vols],
    })


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setattr(bars, "_path_for", lambda ticker, kind: tmp_path / f"{ticker}_{kind}.parquet")
    return tmp_path


def _fake_alpaca(monkeypatch, frames_by_end):
    """fetch_intraday stub: returns the frame registered for the requested end, filtered to start."""
    calls = []

    def fake(*, ticker, start, end, **_kw):
        calls.append({"start": pd.Timestamp(start), "end": pd.Timestamp(end)})
        df = frames_by_end[str(end)]
        return df[df["timestamp"] >= pd.Timestamp(start)].reset_index(drop=True)

    monkeypatch.setattr(bars, "fetch_intraday", fake)
    return calls


def test_intraday_bar_length_parses_alpaca_timeframes():
    assert bars._intraday_bar_length("1Hour") == pd.Timedelta(hours=1)
    assert bars._intraday_bar_length("15Min") == pd.Timedelta(minutes=15)
    assert bars._intraday_bar_length("1Day") is None


def test_complete_bars_only_drops_the_forming_bucket():
    df = _hourly("2026-11-20", HOURS_UTC[:4], FULL_VOL[:4])
    # 14:12 ET: the 13:00 ET (18:00 UTC) hour closed at 14:00 ET -> keep all four.
    kept = bars._complete_bars_only(df, timeframe="1Hour", end="2026-11-20T19:12:00+00:00")
    assert len(kept) == 4
    # 13:52 ET: the 13:00 ET hour is still forming -> dropped.
    kept = bars._complete_bars_only(df, timeframe="1Hour", end="2026-11-20T18:52:00+00:00")
    assert kept["timestamp"].max() == pd.Timestamp("2026-11-20 17:00", tz="UTC")
    # Daily bars are untouched.
    assert len(bars._complete_bars_only(df, timeframe="1Day", end="2026-11-20T00:00:00+00:00")) == 4


def test_fetch_one_never_caches_partial_and_heals_old_partials(monkeypatch, cache):
    # The cache as the old code left it: 18:00 and 19:00 UTC hours partial.
    stale = _hourly("2026-11-20", HOURS_UTC[:5], FULL_VOL[:3] + [3153886, 1250724])
    stale.to_parquet(cache / "AAPL_1h.parquet", index=False)

    end_mid = "2026-11-20T20:40:00+00:00"   # 15:40 ET: the 15:00 ET hour is forming
    forming = _hourly("2026-11-20", HOURS_UTC[:6], FULL_VOL[:5] + [4000000])
    calls = _fake_alpaca(monkeypatch, {end_mid: forming})

    out = bars.fetch_one(ticker="AAPL", kind="1h", end=end_mid)

    # Re-fetched from the NY session start, not last + 1 min.
    assert calls[0]["start"] == pd.Timestamp("2026-11-20 05:00", tz="UTC")
    vol = out.set_index("timestamp")["volume"]
    # Old partial hours replaced with complete ones; the forming 20:00 UTC hour not cached.
    assert vol.loc[pd.Timestamp("2026-11-20 18:00", tz="UTC")] == 5889040
    assert vol.loc[pd.Timestamp("2026-11-20 19:00", tz="UTC")] == 7968777
    assert pd.Timestamp("2026-11-20 20:00", tz="UTC") not in vol.index
    on_disk = pd.read_parquet(cache / "AAPL_1h.parquet")
    assert len(on_disk) == 5 and on_disk["volume"].tolist() == [float(v) for v in FULL_VOL[:5]]


def test_fetch_one_fixed_past_end_still_skips_when_fresh(monkeypatch, cache):
    _hourly("2026-06-01", HOURS_UTC, FULL_VOL).to_parquet(cache / "AAPL_1h.parquet", index=False)
    calls = _fake_alpaca(monkeypatch, {})
    bars.fetch_one(ticker="AAPL", kind="1h", end="2026-06-01T21:00:00+00:00")
    assert calls == []   # nothing can follow the 21:00 bar before end


def test_build_4h_refreshes_bar_built_from_its_first_hour(cache):
    # 4H cache built when only the 10:00 ET hour existed (the 2026-11-20 state).
    first_only = _hourly("2026-11-20", HOURS_UTC[:1], FULL_VOL[:1])
    bars.resample_1h_to_4h(first_only).reset_index().to_parquet(cache / "AAPL_4h.parquet", index=False)
    # Hourly cache now holds 10:00-13:00 ET: last_4h (15:00) >= last_1h (18:00) - 4h,
    # which the old gate treated as "fresh".
    four = _hourly("2026-11-20", HOURS_UTC[:4], FULL_VOL[:4])
    four.to_parquet(cache / "AAPL_1h.parquet", index=False)

    out = bars.build_4h_for("AAPL")

    bar_a = out.loc[pd.Timestamp("2026-11-20 15:00", tz="UTC")]
    assert bar_a["volume"] == sum(FULL_VOL[:4])
    assert bar_a["close"] == four["close"].iloc[-1]


def test_incremental_4h_matches_full_rebuild_across_sessions(cache):
    day1 = _hourly("2026-11-19", HOURS_UTC, [1e6] * 7)
    day2_early = _hourly("2026-11-20", HOURS_UTC[:2], FULL_VOL[:2])
    day2_full = _hourly("2026-11-20", HOURS_UTC, FULL_VOL)
    pd.concat([day1, day2_early]).to_parquet(cache / "AAPL_1h.parquet", index=False)
    bars.build_4h_for("AAPL", force=True)
    pd.concat([day1, day2_full]).to_parquet(cache / "AAPL_1h.parquet", index=False)

    incremental = bars.build_4h_for("AAPL")
    full = bars.resample_1h_to_4h(pd.concat([day1, day2_full]))

    pd.testing.assert_frame_equal(
        incremental[["open", "high", "low", "close", "volume"]].sort_index(),
        full[["open", "high", "low", "close", "volume"]].sort_index(),
        check_freq=False, check_names=False,
    )


class _FakeBarsClient:
    """Multi-symbol get_stock_bars stub returning a (symbol, timestamp)-indexed frame."""

    def __init__(self, frames: dict[str, pd.DataFrame], fail: bool = False):
        self.frames, self.fail, self.requests = frames, fail, []

    def get_stock_bars(self, req):
        self.requests.append(req)
        if self.fail:
            raise RuntimeError("boom")
        start = pd.Timestamp(req.start).tz_convert("UTC") if pd.Timestamp(req.start).tzinfo else pd.Timestamp(req.start, tz="UTC")
        parts = [f[f["timestamp"] >= start] for s, f in self.frames.items() if s in req.symbol_or_symbols]
        df = pd.concat(parts) if parts else pd.DataFrame()

        class R:
            pass
        r = R()
        r.df = df.set_index(["symbol", "timestamp"]) if not df.empty else df
        return r


def test_batched_fetch_matches_fetch_one_semantics(cache):
    # AAPL: stale partial 13:00 ET hour cached; MSFT: cache fresh through end; NVDA: no cache.
    _hourly("2026-11-20", HOURS_UTC[:4], FULL_VOL[:3] + [3153886]).to_parquet(cache / "AAPL_1h.parquet", index=False)
    _hourly("2026-11-20", HOURS_UTC[:6], FULL_VOL[:6], "MSFT").to_parquet(cache / "MSFT_1h.parquet", index=False)
    client = _FakeBarsClient({
        "AAPL": _hourly("2026-11-20", HOURS_UTC[:6], FULL_VOL[:5] + [4000000]),   # 20:00 UTC forming
    })
    end = "2026-11-20T20:00:30+00:00"
    status = bars.fetch_intraday_batched(["AAPL", "MSFT", "NVDA"], kind="1h", end=end, client=client)

    assert status == {"AAPL": "updated", "MSFT": "fresh", "NVDA": "uncached"}
    assert len(client.requests) == 1 and client.requests[0].symbol_or_symbols == ["AAPL"]
    # alpaca-py stores request times as naive UTC.
    assert pd.Timestamp(client.requests[0].start) == pd.Timestamp("2026-11-20 05:00")
    vol = pd.read_parquet(cache / "AAPL_1h.parquet").set_index("timestamp")["volume"]
    assert vol.tolist() == [float(v) for v in FULL_VOL[:5]]          # healed, forming hour dropped


def test_batched_fetch_reports_failed_batches_for_fallback(cache):
    _hourly("2026-11-20", HOURS_UTC[:2], FULL_VOL[:2]).to_parquet(cache / "AAPL_1h.parquet", index=False)
    status = bars.fetch_intraday_batched(["AAPL"], kind="1h", end="2026-11-20T21:00:00+00:00",
                                         client=_FakeBarsClient({}, fail=True))
    assert status == {"AAPL": "error"}


def test_batched_refetch_of_identical_rows_does_not_rewrite(cache):
    day = _hourly("2026-11-20", HOURS_UTC[:5], FULL_VOL[:5])
    day.to_parquet(cache / "AAPL_1h.parquet", index=False)
    before = (cache / "AAPL_1h.parquet").stat().st_mtime_ns
    client = _FakeBarsClient({"AAPL": day})
    status = bars.fetch_intraday_batched(["AAPL"], kind="1h", end="2026-11-21T00:00:00+00:00", client=client)
    assert status == {"AAPL": "unchanged"}
    assert (cache / "AAPL_1h.parquet").stat().st_mtime_ns == before


def test_batched_fetch_never_requests_delisted_symbols(cache, monkeypatch):
    _hourly("2026-11-20", HOURS_UTC[:2], FULL_VOL[:2], "DEAD").to_parquet(cache / "DEAD_1h.parquet", index=False)
    monkeypatch.setattr(bars, "is_delisted", lambda t: t == "DEAD")
    client = _FakeBarsClient({"DEAD": _hourly("2026-11-20", HOURS_UTC[:5], FULL_VOL[:5], "DEAD")})
    status = bars.fetch_intraday_batched(["DEAD"], kind="1h", end="2026-11-21T00:00:00+00:00", client=client)
    assert status == {"DEAD": "delisted"} and client.requests == []
