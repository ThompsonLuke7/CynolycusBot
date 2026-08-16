"""The float32 downcast must not move a cross-sectional rank.

`build_all_features_4h` stores feature columns as float32 so the combined matrix
halves in size. `pd.concat` holds every input alive while materialising the
output, and at float64 that peak was OOM-killed above a 16 GB cap on 2026-08-03,
which is what kept the 4H entry gate shut; at float32 the same build completes
at 15.74 GiB.

The downcast is lossless for the models (XGBoost stores a DMatrix as float32
regardless) but NOT for comparisons: two tickers whose ret_5 differs in the 8th
decimal are distinct in float64 and exactly tied in float32, and `rank(pct=True)`
averages ties. So every column feeding a cross-sectional rank is held back at
full width. These tests pin that set to the ranks that actually consume it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.momentum_expansion.features import feature_matrix_4h as fm


def test_every_rank_source_is_precision_protected():
    """A new xsec rank must not silently start ranking float32 values."""
    unprotected = set(fm._XSEC_RANK_SPECS) - set(fm._PRECISION_SENSITIVE_COLS)
    assert not unprotected, f"rank sources missing float64 protection: {sorted(unprotected)}"

    # xsec_near_high_rank ranks this one without going through _XSEC_RANK_SPECS.
    assert "dist_to_52w_high_atr" in fm._PRECISION_SENSITIVE_COLS


def _frame_with_near_ties(n_tickers: int = 120) -> pd.DataFrame:
    """One timestamp, values separated by less than float32 can represent."""
    base = 0.1234567891
    vals = base + np.arange(n_tickers) * 1e-9
    idx = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(["2026-08-03 14:00:00+00:00"] * n_tickers, name="timestamp"),
         [f"T{i:03d}" for i in range(n_tickers)]],
        names=["timestamp", "ticker"],
    )
    return pd.DataFrame({"ret_5": vals, "dist_to_52w_high_atr": vals}, index=idx)


def test_downcasting_a_rank_source_would_change_the_rank():
    """Demonstrates the hazard the protection exists to prevent."""
    df = _frame_with_near_ties()

    full = fm._add_cross_sectional_features(df.copy())["xsec_ret_5_rank"]
    downcast = fm._add_cross_sectional_features(
        df.astype({"ret_5": "float32", "dist_to_52w_high_atr": "float32"})
    )["xsec_ret_5_rank"]

    # float64 keeps all 120 values distinct. float32 resolves ~1.5e-8 near 0.12,
    # so a 1e-9 step collapses them into ties — far fewer distinct ranks, and
    # every member of a tie group is pulled to that group's average percentile.
    assert full.nunique() == len(full)
    assert downcast.nunique() < full.nunique() / 5
    assert (full - downcast).abs().max() > 0.02


def test_protected_columns_survive_the_downcast_rule():
    """The rule build_all_features_4h applies must leave the rank sources alone."""
    df = _frame_with_near_ties()
    df["some_other_feature"] = 1.5

    to_cast = [
        c for c in df.select_dtypes("float64").columns
        if c not in fm._PRECISION_SENSITIVE_COLS
    ]
    out = df.astype({c: "float32" for c in to_cast})

    assert out["ret_5"].dtype == np.float64
    assert out["dist_to_52w_high_atr"].dtype == np.float64
    assert out["some_other_feature"].dtype == np.float32

    ranks_before = fm._add_cross_sectional_features(df.copy())["xsec_ret_5_rank"]
    ranks_after = fm._add_cross_sectional_features(out)["xsec_ret_5_rank"]
    pd.testing.assert_series_equal(ranks_before, ranks_after)
