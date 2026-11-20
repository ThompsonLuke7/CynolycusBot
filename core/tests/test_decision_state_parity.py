"""Parity compare: float32 archive vs research file, NaN handled, mismatches counted."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.decision_state.parity_check import compare_vectors

BAR = pd.Timestamp("2026-11-12 19:00", tz="UTC")


def test_compare_vectors_counts_real_disagreements_only():
    archived = pd.DataFrame({
        "bar_ts": [BAR] * 4, "ticker": ["A", "B", "C", "D"],
        "f__x": np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32),
        "f__y": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "f__only_live": [1.0, 1.0, 1.0, 1.0],
    })
    reference = pd.DataFrame({
        "bar_ts": [BAR] * 3, "ticker": ["A", "B", "C"],
        "x": [1.0, 2.5, np.nan],                    # B disagrees, C NaN on both sides
        "y": [0.1, 0.2, 0.3 + 1e-9],                # float32 rounding is not a mismatch
    })
    table, cov = compare_vectors(archived, reference, keys=["bar_ts", "ticker"])
    t = table.set_index("feature")
    assert cov["matched_rows"] == 3 and cov["features_not_in_reference"] == ["only_live"]
    assert t.loc["x", "mismatch"] == 1 and t.loc["y", "mismatch"] == 0
    assert np.isclose(t.loc["x", "share"], 1 / 3)
