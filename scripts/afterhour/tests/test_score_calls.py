from __future__ import annotations

import pandas as pd
import pytest

from scripts.afterhour.collect import post_hash
from scripts.afterhour.score_calls import base_rates, entry_index, score_call

DATES = pd.Index(["2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06", "2026-03-09"])


def bars(opens, closes, highs=None, lows=None, scale=1.0):
    adj = pd.DataFrame({"open": opens, "close": closes, "high": highs or closes, "low": lows or closes,
                        "volume": 1}, index=DATES[:len(opens)])
    raw = adj * scale
    return {"split": adj, "raw": raw}


def test_post_hash_from_share_url() -> None:
    assert post_hash("https://afterhour.com/RoyaltyTradez/bBG4/some-slug") == "bBG4"
    assert post_hash("") is None


@pytest.mark.parametrize("posted,expected", [
    ("2026-03-03T14:00:00Z", 1),   # 09:00 ET, before the open -> same day
    ("2026-03-03T15:00:00Z", 2),   # 10:00 ET, intraday -> next session (no fill at a pre-post price)
    ("2026-03-03T22:00:00Z", 2),   # after the close -> next session
    ("2026-03-07T15:00:00Z", 5),   # Saturday -> Monday
    ("2026-03-09T21:00:00Z", None),  # after the last bar -> censored
])
def test_entry_is_first_open_after_post(posted, expected) -> None:
    assert entry_index(DATES, pd.Timestamp(posted)) == expected


def test_direction_hit_and_short_sign() -> None:
    b = bars([10, 10, 10, 10, 10, 10], [10, 11, 9, 9, 9, 9])
    spy = bars([1] * 6, [1] * 6)["split"]
    call = {"call_id": "c", "username": "u", "posted_at_utc": "2026-03-02T22:00:00Z", "symbol": "X",
            "direction": "short", "instrument": "stock"}
    out = score_call(call, b, spy, {1: 0.6, 5: 0.6, 20: 0.6})
    assert out["entry_date"] == "2026-03-03"
    assert out["hit_1d"] is False          # 10 -> 11 is a miss for a short
    assert out["base_1d"] == pytest.approx(0.4)  # short base rate = 1 - P(up)
    assert out["hit_20d"] is None          # not enough bars: censored, not a miss


def test_strike_compared_in_post_date_basis_after_split() -> None:
    # 2:1 split-adjusted series; raw at entry is 2x adjusted. Strike 210 in raw terms = 105 adjusted.
    b = bars([100, 100, 100, 100, 100, 100], [100, 104, 106, 104, 103, 103], scale=2.0)
    spy = bars([1] * 6, [1] * 6)["split"]
    call = {"call_id": "c", "username": "u", "posted_at_utc": "2026-03-02T22:00:00Z", "symbol": "X",
            "direction": "long", "instrument": "call", "strike": 210.0, "expiry": "2026-03-05"}
    out = score_call(call, b, spy, {1: 0.5, 5: 0.5, 20: 0.5})
    assert out["strike_status"] == "scored"
    assert out["strike_hit_lenient"] is True   # 106 * 2 = 212 >= 210
    assert out["strike_hit_strict"] is False   # expiry close 104 * 2 = 208 < 210


def test_unexpired_strike_is_censored() -> None:
    b = bars([100] * 6, [100] * 6)
    call = {"call_id": "c", "username": "u", "posted_at_utc": "2026-03-02T22:00:00Z", "symbol": "X",
            "direction": "long", "instrument": "call", "strike": 101.0, "expiry": "2026-04-17"}
    out = score_call(call, b, b["split"], {1: 0.5, 5: 0.5, 20: 0.5})
    assert out["strike_status"] == "censored_unexpired"


def test_base_rate_counts_up_windows() -> None:
    adj = pd.DataFrame({"open": [10, 10, 10, 10], "close": [11, 9, 11, 11]},
                       index=pd.Index(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]))
    assert base_rates(adj)[1] == pytest.approx(0.75)


def test_jsonl_roundtrip_survives_unicode_line_separators(tmp_path) -> None:
    from scripts.afterhour.collect import read_jsonl, write_jsonl
    rows = [{"title": "line sep\u0085next"}, {"title": "ok"}]
    write_jsonl(tmp_path / "x.jsonl", rows)
    assert read_jsonl(tmp_path / "x.jsonl") == rows


def test_zero_volume_padding_is_dropped() -> None:
    from scripts.afterhour.score_calls import clean_bars
    df = pd.DataFrame({"open": [495.55, 29.5], "high": [495.55, 30], "low": [495.55, 29], "close": [495.55, 28.95],
                       "volume": [0, 930933]}, index=pd.Index(["2024-07-30", "2024-07-31"]))
    out = clean_bars({"split": df, "raw": df})
    assert list(out["split"].index) == ["2024-07-31"]


def test_clustered_t_is_smaller_than_naive_when_calls_cluster() -> None:
    import math
    from scripts.afterhour.score_calls import clustered_t
    # three weeks, each a block of identical outcomes: the effective sample is 3, not 30
    e = pd.Series([0.5] * 10 + [-0.5] * 10 + [0.5] * 10 + [0.5] * 10 + [-0.5] * 10)
    wk = pd.Series(["w1"] * 10 + ["w2"] * 10 + ["w3"] * 10 + ["w4"] * 10 + ["w5"] * 10)
    naive = e.mean() / (e.std(ddof=0) / math.sqrt(len(e)))
    assert abs(clustered_t(e, wk)) < abs(naive)


def test_reused_symbol_after_long_gap_is_censored() -> None:
    idx = pd.Index(["2024-12-16", "2024-12-17", "2026-01-09", "2026-01-12"])
    adj = pd.DataFrame({"open": [0.045, 0.04, 27.0, 22.45], "high": [0.05, 0.04, 27, 23], "low": [0.04, 0.03, 22, 21],
                        "close": [0.0409, 0.0372, 22.4, 21.75], "volume": [1, 1, 1, 1]}, index=idx)
    b = {"split": adj, "raw": adj}
    call = {"call_id": "c", "username": "u", "posted_at_utc": "2024-12-16T22:00:00Z", "symbol": "AKTS",
            "direction": "long", "instrument": "stock"}
    out = score_call(call, b, adj, {1: 0.5, 5: 0.5, 20: 0.5})
    assert out["entry_date"] == "2024-12-17"
    assert out["hit_5d"] is None and out["series_ends_at_gap"] == "2026-01-09"


def test_strike_base_counts_windows_reaching_the_same_ratio() -> None:
    from scripts.afterhour.score_calls import strike_base
    idx = pd.Index([f"2024-01-{d:02d}" for d in range(2, 8)])
    adj = pd.DataFrame({"open": [100.0] * 6, "high": [100, 112, 100, 100, 100, 100], "low": [100.0] * 6,
                        "close": [100, 100, 100, 100, 100, 111]}, index=idx)
    strict, lenient = strike_base(adj, 1.10, 1, 1, "call")
    # 1-session windows (t, t+1): closes 100,100,100,100,111 -> only the last window ends above 110
    assert strict == pytest.approx(1 / 5)
    # highs over (t, t+1): window 0 and 1 contain the 112 high
    assert lenient == pytest.approx(2 / 5)


def test_confidence_treats_nan_strike_as_missing() -> None:
    from scripts.afterhour.score_calls import confidence
    assert confidence({"tier": "B", "strike": float("nan")}) == "low"
    assert confidence({"tier": "B", "strike": 755.0}) == "high"
    assert confidence({"tier": "A", "strike": None}) == "high"
