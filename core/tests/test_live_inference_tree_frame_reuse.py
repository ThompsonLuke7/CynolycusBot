"""One tree-feature frame per decision bar, not two.

2026-09-10: the SPY loop built the tree feature frame twice per 10-minute bar --
once for GA pivot/TB probabilities inside `build_meta_feature_frame_from_1m`, and
once for swing setup probabilities in `_build_setup_feature_frame` -- from the
same 1m buffer with the same arguments. Each build measured 49.5s over the live
50,000-row buffer, against a 600s bar budget that the loop was already missing.

Sharing one frame is exact by construction: same function, same inputs, and
`LiveGAXGBPredictor.predict_frame` reindexes into a new frame rather than
mutating what it is handed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import core.API.Alpaca_API.inference.live_inference as li

_TREE_KWARGS = dict(
    label_timeframe="10min", resample_label="left", resample_closed="left",
    tz="America/New_York", assume_tz="UTC", include_vix_features=True,
)


class _StubGA:
    """Deterministic in the tree frame it is handed, so a swapped frame shows up."""

    def predict_frame(self, x_df: pd.DataFrame) -> pd.DataFrame:
        close = pd.to_numeric(x_df.get("close"), errors="coerce") if "close" in x_df else None
        base = close.to_numpy(dtype=float) if close is not None else np.arange(len(x_df), dtype=float)
        scaled = np.mod(np.nan_to_num(base), 1.0)
        return pd.DataFrame(
            {"p_pivot_long": scaled, "p_pivot_short": 1.0 - scaled}, index=x_df.index
        )


def _one_minute_frame(sessions: int = 8) -> pd.DataFrame:
    parts = []
    for day in pd.bdate_range("2026-06-01", periods=sessions):
        start = pd.Timestamp(day.date(), tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
        parts.append(pd.date_range(start, periods=390, freq="1min"))
    index = parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]
    rng = np.random.default_rng(11)
    close = 500.0 + np.cumsum(rng.normal(0.0, 0.05, len(index)))
    return pd.DataFrame(
        {"open": close, "high": close + 0.05, "low": close - 0.05,
         "close": close, "volume": 1_000.0},
        index=index.tz_convert("UTC"),
    )


def test_a_supplied_tree_frame_replaces_the_rebuild_and_changes_nothing(monkeypatch):
    df_1m = _one_minute_frame()
    kwargs = dict(rule="10min", label="left", closed="left", tz="America/New_York",
                  assume_tz="UTC", include_pivot_probs=True, include_tb_probs=False,
                  include_vix_features=False, ga_predictor=_StubGA())

    real_builder = li.build_tree_feature_frame_from_1m
    calls = {"n": 0}

    def counting(*args, **kw):
        calls["n"] += 1
        return real_builder(*args, **kw)

    monkeypatch.setattr(li, "build_tree_feature_frame_from_1m", counting)

    rebuilt = li.build_meta_feature_frame_from_1m(df_1m, **kwargs)
    assert calls["n"] == 1, "the GA path builds the tree frame itself when none is supplied"

    tree = real_builder(df_1m, **_TREE_KWARGS)
    calls["n"] = 0
    shared = li.build_meta_feature_frame_from_1m(df_1m, x_tree=tree, **kwargs)

    assert calls["n"] == 0, "a supplied tree frame must not be rebuilt"
    pd.testing.assert_frame_equal(shared, rebuilt)


def test_the_supplied_frame_is_what_the_ga_predictor_scores():
    """Guard against the parameter being accepted and then ignored."""
    df_1m = _one_minute_frame()
    kwargs = dict(rule="10min", label="left", closed="left", tz="America/New_York",
                  assume_tz="UTC", include_pivot_probs=True, include_tb_probs=False,
                  include_vix_features=False, ga_predictor=_StubGA())

    tree = li.build_tree_feature_frame_from_1m(df_1m, **_TREE_KWARGS)
    doctored = tree.copy()
    doctored["close"] = doctored["close"] + 0.25

    baseline = li.build_meta_feature_frame_from_1m(df_1m, x_tree=tree, **kwargs)
    altered = li.build_meta_feature_frame_from_1m(df_1m, x_tree=doctored, **kwargs)

    assert not baseline["p_pivot_long"].equals(altered["p_pivot_long"])
