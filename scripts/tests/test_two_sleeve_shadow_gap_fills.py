"""`evaluate_exit` must measure the exit, not restate the policy.

Through 2026-08-07 a stop or target exit returned the policy THRESHOLD as
`underlying_ret` regardless of where the bar actually opened. Across all 257
shadow exits ever logged that produced exactly two values — +0.07 on 256
`target` exits and -0.39 on the single `stop` — so every sleeve comparison and
both hypothetical spread P&Ls downstream were constants. A bar that gaps through
a level cannot fill at that level.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.shadow.two_sleeve_shadow_tracker import (
    HARVEST_POLICY,
    TAIL_POLICY,
    evaluate_exit,
)


def _bars(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp").sort_index()


ENTRY_TS = pd.Timestamp("2026-08-03T14:00:00Z")


def test_target_hit_intrabar_fills_at_the_target():
    # Bar opens below the +7% target and trades up through it: the level was
    # tradeable, so the fill is the level.
    bars = _bars([("2026-08-04T14:00:00Z", 100.0, 110.0, 99.0, 108.0)])
    out = evaluate_exit(100.0, bars, ENTRY_TS, HARVEST_POLICY)
    assert out["reason"] == "target"
    assert out["underlying_ret"] == pytest.approx(0.07)


def test_target_gap_up_fills_at_the_open_not_the_target():
    # Opens +22% — the +7% level never traded. Recording 0.07 here understated
    # every gap-up winner the harvest sleeve ever had.
    bars = _bars([("2026-08-04T14:00:00Z", 122.0, 125.0, 121.0, 124.0)])
    out = evaluate_exit(100.0, bars, ENTRY_TS, HARVEST_POLICY)
    assert out["reason"] == "target"
    assert out["underlying_ret"] == pytest.approx(0.22)


def test_stop_gap_down_fills_at_the_open_not_the_stop():
    # Opens -55% against a -39% stop: the fill is the open, and it is worse.
    bars = _bars([("2026-08-04T14:00:00Z", 45.0, 46.0, 40.0, 44.0)])
    out = evaluate_exit(100.0, bars, ENTRY_TS, TAIL_POLICY)
    assert out["reason"] == "stop"
    assert out["underlying_ret"] == pytest.approx(-0.55)


def test_stop_hit_intrabar_still_fills_at_the_stop():
    bars = _bars([("2026-08-04T14:00:00Z", 98.0, 99.0, 55.0, 60.0)])
    out = evaluate_exit(100.0, bars, ENTRY_TS, TAIL_POLICY)
    assert out["reason"] == "stop"
    assert out["underlying_ret"] == pytest.approx(-0.39)


def test_exits_are_no_longer_a_constant_across_paths():
    # The regression that motivated this: different price paths must produce
    # different numbers.
    paths = [
        [("2026-08-04T14:00:00Z", 100.0, 108.0, 99.0, 107.0)],   # target intrabar
        [("2026-08-04T14:00:00Z", 115.0, 118.0, 114.0, 117.0)],  # target on a gap
        [("2026-08-04T14:00:00Z", 130.0, 131.0, 129.0, 130.0)],  # bigger gap
    ]
    rets = {evaluate_exit(100.0, _bars(p), ENTRY_TS, HARVEST_POLICY)["underlying_ret"] for p in paths}
    assert len(rets) == 3


def test_partial_scale_out_books_the_actual_fill():
    # scale_frac < 1 books the realized leg; it must book what filled, not the
    # target, or the trimmed portion is mispriced on every gap.
    policy = dict(HARVEST_POLICY, scale_frac=0.5, target=0.07, stop=None, horizon=2)
    bars = _bars([
        ("2026-08-04T14:00:00Z", 120.0, 121.0, 119.0, 120.0),  # gap through target
        ("2026-08-04T18:00:00Z", 120.0, 121.0, 119.0, 130.0),  # horizon
    ])
    out = evaluate_exit(100.0, bars, ENTRY_TS, policy)
    assert out["reason"] == "horizon"
    # 0.5 booked at the +20% gap open, 0.5 rides to the +30% close.
    assert out["underlying_ret"] == pytest.approx(0.5 * 0.20 + 0.5 * 0.30)


def test_still_open_returns_none():
    bars = _bars([("2026-08-04T14:00:00Z", 100.0, 101.0, 99.0, 100.0)])
    assert evaluate_exit(100.0, bars, ENTRY_TS, HARVEST_POLICY) is None
