"""Rebasing repaired hourly bars back to the cache's as-fetched price basis (2026-11-23)."""
import pandas as pd

from scripts.repair_partial_bars_rebase import rebase, session_factors


def _session(day, closes, vols):
    ts = pd.to_datetime([f"{day} {h}:00" for h in range(14, 21)], utc=True)   # 10:00-16:00 EDT
    return pd.DataFrame({"symbol": "X", "timestamp": ts, "open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": vols, "trade_count": 1.0, "vwap": closes})


def test_split_basis_restored_and_partial_bars_stay_healed():
    full_vol = [100.0] * 7
    # Old cache: raw 400/share before a 4:1 split on 07-02, partial afternoon hours.
    old = pd.concat([_session("2026-07-01", [400.0] * 7, [100.0] * 4 + [40.0] * 3),
                     _session("2026-07-02", [100.0] * 7, [400.0] * 4 + [160.0] * 3)], ignore_index=True)
    # Fresh pull: adjusted as of today (07-01 divided by 4, volume x4), complete hours.
    new = pd.concat([_session("2026-07-01", [100.0] * 7, [400.0] * 7),
                     _session("2026-07-02", [100.0] * 7, [400.0] * 7)], ignore_index=True)
    since = pd.Timestamp("2026-07-01", tz="UTC")

    f = session_factors(old, new, since)
    assert list(f.round(6)) == [4.0, 1.0]
    out, _ = rebase(old, new, since)
    d1 = out[out["timestamp"].dt.day == 1]
    assert (d1["close"] == 400.0).all() and (d1["vwap"] == 400.0).all()
    assert d1["volume"].tolist() == full_vol                     # afternoon healed, old basis
    d2 = out[out["timestamp"].dt.day == 2]
    assert (d2["close"] == 100.0).all() and (d2["volume"] == 400.0).all()


def test_small_morning_differences_are_not_a_basis_change():
    old = _session("2026-07-01", [100.0, 100.5, 100.0, 99.8, 100, 100, 100], [100.0] * 7)
    new = _session("2026-07-01", [100.0, 100.0, 100.0, 100.0, 100, 100, 100], [100.0] * 7)
    f = session_factors(old, new, pd.Timestamp("2026-07-01", tz="UTC"))
    assert list(f) == [1.0]
