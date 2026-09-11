"""The WOLF case, plus the cases the guard must NOT fire on."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.corporate_actions import mask_windows, suspect_sessions


def _frame(closes, opens=None, volumes=None):
    n = len(closes)
    return pd.DataFrame({
        "open": opens if opens is not None else closes,
        "close": closes,
        "volume": volumes if volumes is not None else [1_000_000] * n,
    })


def test_flags_the_wolf_recapitalisation():
    """Real numbers: WOLF 2025-09-26 -> 2025-09-29, Chapter 11 emergence."""
    closes = [1.475, 1.88, 2.345, 1.20, 21.51, 28.95]
    opens = [1.66, 1.67, 2.03, 1.545, 17.88, 28.96]
    vols = [482_267, 650_127, 1_731_644, 2_303_766, 180_927, 898_532]
    flags = suspect_sessions(_frame(closes, opens, vols))
    assert len(flags) == 1
    assert flags.loc[0, "idx"] == 4
    assert flags.loc[0, "gap_ratio"] == pytest.approx(17.88 / 1.20, rel=1e-3)
    # price up ~15x, volume down ~13x -> traded VALUE barely moved. That ratio
    # near 1 is the tell that the share count changed, not the company's value.
    assert flags.loc[0, "dollar_volume_ratio"] < 3.0


def test_does_not_fire_on_a_real_explosive_move():
    """A 120% day on 20x volume is signal, not a corporate action."""
    closes = [10.0] * 20 + [22.0]
    opens = [10.0] * 20 + [21.0]
    vols = [1_000_000] * 20 + [20_000_000]
    assert suspect_sessions(_frame(closes, opens, vols)).empty


def test_does_not_fire_on_ordinary_bars():
    rng = np.random.default_rng(0)
    closes = list(100 * np.cumprod(1 + rng.normal(0, 0.02, 200)))
    assert suspect_sessions(_frame(closes)).empty


def test_window_mask_covers_every_entry_whose_hold_spans_the_gap():
    closes = [1.0] * 10 + [30.0] * 5
    opens = [1.0] * 10 + [30.0] * 5
    df = _frame(closes, opens)
    bad = mask_windows(df, hold=5)
    # the gap is at index 10; a 5-session hold entered at 6..10 spans it
    assert bad.iloc[6:11].all()
    assert not bad.iloc[:6].any()
    assert not bad.iloc[11:].any()


def test_empty_and_missing_columns_are_safe():
    assert suspect_sessions(pd.DataFrame()).empty
    assert suspect_sessions(pd.DataFrame({"close": [1.0, 2.0]})).empty
    assert not mask_windows(pd.DataFrame(), hold=5).any()
