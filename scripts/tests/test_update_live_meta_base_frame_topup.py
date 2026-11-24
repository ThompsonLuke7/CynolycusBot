"""Nightly SPY matrix input: fresh complete minutes after the runtime cache's end (2026-11-23)."""
import pandas as pd

from scripts.update_live_meta_base_frame import fresh_topup


def _frame(start, n):
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=n, freq="1min")
    return pd.DataFrame({"timestamp": idx, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 10.0})


def test_topup_appends_only_new_complete_minutes():
    cache = _frame("2026-11-20 20:55", 5).set_index("timestamp")          # ends 20:59
    asked = {}

    def fetch(start, end):
        asked["start"] = start
        return _frame("2026-11-20 20:58", 4)                              # 20:58..21:01, overlaps

    out, added = fresh_topup(cache, symbol="SPY", now=pd.Timestamp("2026-11-20 21:01:30", tz="UTC"), fetch=fetch)
    assert asked["start"].startswith("2026-11-20T21:00")
    assert added == 1 and out.index.max() == pd.Timestamp("2026-11-20 21:00", tz="UTC")   # 21:01 still forming
    assert out.index.is_unique and len(out) == 6
