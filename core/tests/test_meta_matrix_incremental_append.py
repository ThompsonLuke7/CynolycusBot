"""The cached 10m meta matrix grows by new rows; it is never rewritten.

The live agent rebuilt every feature from the whole 1m buffer on every 10-minute
bar (2026-09-10: ~55s base frame + ~52s setup probabilities, against a 600s bar,
so the loop fell up to 92 minutes behind). The cache that was supposed to prevent
that exists, but its merge replaced the entire overlap window with rows computed
from just that window -- so each pass overwrote months of full-history values
with short-history ones, and restarted `day_id` at zero.

What an append must preserve, measured 2026-09-12 against a full 554,655-row
build: cached rows untouched, `day_id` continuous, and the probability
derivatives (`_lag1`, `_lag2`, `_max_last_4`, `_delta_1`) rebuilt from the cached
predecessors rather than the window's own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.API.Alpaca_API.inference.live_inference import recompute_prob_derivatives

_ET = "America/New_York"


def _frame(values, start="2026-09-10 09:30"):
    index = pd.date_range(pd.Timestamp(start, tz=_ET), periods=len(values), freq="10min")
    base = pd.Series(values, index=index, dtype=float)
    return pd.DataFrame({
        "p_pivot_long": base,
        "p_pivot_long_lag1": np.nan,
        "p_pivot_long_lag2": np.nan,
        "p_pivot_long_max_last_4": np.nan,
        "p_pivot_long_delta_1": np.nan,
    })


def test_the_derivatives_are_rebuilt_from_the_frames_own_history():
    frame = _frame([0.10, 0.40, 0.20, 0.30, 0.25])

    out = recompute_prob_derivatives(frame)

    assert out["p_pivot_long_lag1"].tolist()[1:] == [0.10, 0.40, 0.20, 0.30]
    assert np.isnan(out["p_pivot_long_lag1"].iloc[0])
    assert out["p_pivot_long_lag2"].tolist()[2:] == [0.10, 0.40, 0.20]
    assert out["p_pivot_long_max_last_4"].tolist() == [0.10, 0.40, 0.40, 0.40, 0.40]
    assert out["p_pivot_long_delta_1"].round(10).tolist()[1:] == [0.30, -0.20, 0.10, -0.05]


def test_an_appended_row_sees_the_cached_predecessors():
    """The whole point: the new row's lag/max/delta read cached history.

    Recomputing the tail alone gives the window's own predecessors, which is how
    these columns drifted up to 0.06 from full-history values at every window
    size tried.
    """
    cached = _frame([0.10, 0.90, 0.20, 0.30])
    appended = _frame([0.25], start="2026-09-10 10:10")

    merged = recompute_prob_derivatives(pd.concat([cached, appended]).sort_index())
    tail_only = recompute_prob_derivatives(appended)

    last = merged.iloc[-1]
    assert last["p_pivot_long_lag1"] == 0.30
    assert last["p_pivot_long_lag2"] == 0.20
    assert last["p_pivot_long_max_last_4"] == 0.90, "the 0.90 four rows back is only visible with history"
    assert np.isnan(tail_only.iloc[-1]["p_pivot_long_lag1"]), "a lone tail has no predecessor"


def test_cached_rows_keep_their_values():
    cached = _frame([0.10, 0.90, 0.20, 0.30])
    seeded = recompute_prob_derivatives(cached)
    appended = _frame([0.25], start="2026-09-10 10:10")

    merged = recompute_prob_derivatives(pd.concat([seeded, appended]).sort_index())

    pd.testing.assert_frame_equal(merged.iloc[: len(seeded)], seeded)


def test_columns_are_never_added_to_a_frame_that_lacks_them():
    """A frame built without TB probabilities must not gain TB columns."""
    frame = _frame([0.1, 0.2]).drop(columns=["p_pivot_long_lag2"])

    out = recompute_prob_derivatives(frame)

    assert "p_pivot_long_lag2" not in out.columns
    assert "p_tb_long_lag1" not in out.columns
    assert out["p_pivot_long_lag1"].tolist()[1:] == [0.1]


def test_a_frame_without_the_base_column_is_returned_unchanged():
    frame = pd.DataFrame({"close": [1.0, 2.0]})

    assert recompute_prob_derivatives(frame) is frame
