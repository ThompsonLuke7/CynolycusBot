"""Synthetic-data regression tests for the segmentation study's simulation and
entry-gating helpers (scripts/capstone/exit_policy_segmentation.py / _entry_side.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/capstone"))

import exit_policy_segmentation as seg  # noqa: E402
import exit_policy_entry_side as ent  # noqa: E402


def _mk_bars(closes, highs=None, lows=None):
    idx = pd.date_range("2025-08-01", periods=len(closes), freq="4h", tz="UTC")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "close": c,
        "high": np.asarray(highs, dtype=float) if highs is not None else c,
        "low": np.asarray(lows, dtype=float) if lows is not None else c,
    }, index=idx)


@pytest.fixture(autouse=True)
def _clean_bar_cache():
    seg._BAR_CACHE.clear()
    yield
    seg._BAR_CACHE.clear()


def _member(bars, in_top_flags, ticker="TST"):
    return pd.DataFrame({"timestamp": bars.index[:len(in_top_flags)],
                         "ticker": ticker, "in_top": in_top_flags})


def test_full_target_exit_books_exact_target():
    # +10% high on bar 2 must exit the full position at exactly +7% under g284-style policy
    bars = _mk_bars([100, 100, 100, 100], highs=[100, 100, 110, 100], lows=[100, 99, 99, 99])
    seg._BAR_CACHE["TST"] = bars
    tr = seg.all_trades(_member(bars, [True, False, False, False]),
                        stop=None, trail=None, target=0.07, scale_frac=1.0, horizon=60, grace=None)
    assert len(tr) == 1
    assert tr.iloc[0]["ret"] == pytest.approx(0.07)


def test_partial_trim_banks_scale_frac_and_rides_remainder():
    # id4-style: trim 16% at +30%, remainder exits on horizon close
    closes = [100] * 10
    highs = [100, 131, 100, 100, 100, 100, 100, 100, 100, 100]
    bars = _mk_bars(closes, highs=highs, lows=[99] * 10)
    bars.iloc[5, bars.columns.get_loc("close")] = 120.0
    seg._BAR_CACHE["TST"] = bars
    tr = seg.all_trades(_member(bars, [True] + [False] * 9),
                        stop=None, trail=None, target=0.30, scale_frac=0.16, horizon=5, grace=None)
    assert len(tr) == 1
    expected = 0.16 * 0.30 + 0.84 * (120.0 / 100.0 - 1)
    assert tr.iloc[0]["ret"] == pytest.approx(expected)


def test_stop_binds_at_stop_level():
    bars = _mk_bars([100, 100, 100], lows=[100, 40, 100])
    seg._BAR_CACHE["TST"] = bars
    tr = seg.all_trades(_member(bars, [True, False, False]),
                        stop=0.50, trail=None, target=None, scale_frac=1.0, horizon=60, grace=None)
    assert tr.iloc[0]["ret"] == pytest.approx(-0.50)


def test_potential_is_max_forward_high():
    bars = _mk_bars([100, 100, 100, 100], highs=[100, 105, 140, 100], lows=[99] * 4)
    seg._BAR_CACHE["TST"] = bars
    tr = seg.all_trades(_member(bars, [True, False, False, False]),
                        stop=None, trail=None, target=None, scale_frac=1.0, horizon=60, grace=None)
    assert tr.iloc[0]["potential"] == pytest.approx(0.40)


def test_member_variant_score_gate_and_cohort_extension():
    ts = pd.Timestamp("2025-08-01 14:00", tz="UTC")
    stream = pd.DataFrame({
        "timestamp": [ts] * 4,
        "ticker": ["A", "B", "C", "D"],
        "score": [0.9, 0.2, 0.8, 0.7],
        "rk": [1.0, 10.0, 15.0, 30.0],
    })
    start, end = pd.Timestamp("2025-07-01", tz="UTC"), pd.Timestamp("2025-09-01", tz="UTC")
    # score gate: B is rank<=10 but below threshold -> excluded
    gated = ent.member_variant(stream, start, end, score_min=0.5)
    assert gated.set_index("ticker")["in_top"].to_dict() == {"A": True, "B": False, "C": False, "D": False}
    # cohort extension: C (rank 15, in ext set) admitted; D (rank 30) beyond depth
    ext = ent.member_variant(stream, start, end, ext_tickers={"C", "D"}, ext_depth=25)
    assert ext.set_index("ticker")["in_top"].to_dict() == {"A": True, "B": True, "C": True, "D": False}


def test_tail_cohort_young_fallback_for_short_history():
    long_bars = _mk_bars(list(100 + np.sin(np.arange(600)) * 5))
    long_bars.index = pd.date_range("2023-01-01", periods=600, freq="4h", tz="UTC")
    short_bars = _mk_bars([100] * 50)
    short_bars.index = pd.date_range("2025-06-01", periods=50, freq="4h", tz="UTC")
    seg._BAR_CACHE.update({"LONG1": long_bars, "LONG2": long_bars * 1.1, "LONG3": long_bars * 0.9,
                           "SHORTY": short_bars})
    out = seg.tail_propensity_cohorts(["LONG1", "LONG2", "LONG3", "SHORTY"])
    assert out.set_index("ticker").loc["SHORTY", "tail_cohort"] == "young"
    assert set(out["tail_cohort"]) <= {"grinder", "moderate", "explosive", "young"}
