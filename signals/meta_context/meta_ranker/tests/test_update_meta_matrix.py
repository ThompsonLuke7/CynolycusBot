from __future__ import annotations

import pandas as pd
import pytest

import signals.meta_context.meta_ranker.update_meta_matrix as updater
from strategies.momentum_expansion.features.feature_matrix_4h import (
    _add_cross_sectional_features,
    _add_earnings_features_4h,
)
from strategies.momentum_expansion.inference.ranker import ExpansionRanker
from strategies.multi_ticker_swing_htf.inference.scorer import HTFSwingScorer


pytestmark = pytest.mark.safe


def test_latest_reference_bar_timestamp_uses_newest_reference(monkeypatch):
    def fake_read(path):
        ticker = path.stem
        stamp = "2026-07-16 18:00:00+00:00" if ticker == "SPY" else "2026-07-16 17:00:00+00:00"
        return pd.DataFrame({"timestamp": [pd.Timestamp(stamp)]})

    monkeypatch.setattr(updater, "_read_bars", fake_read)

    assert updater._latest_reference_bar_timestamp() == pd.Timestamp("2026-07-16 18:00:00+00:00")


def test_live_postprocessing_covers_full_deployed_feature_manifest():
    """Regression test for the 2026-07-19 audit: build_ticker_features_4h alone
    does NOT produce xsec_* (cross-sectional rank) or earnings-proximity
    columns -- those only exist via _add_cross_sectional_features /
    _add_earnings_features_4h, which were historically batch-only (never
    called from update_meta_matrix.main() or momentum's
    live/runner.py::evaluate_now(), both fixed this session to route through
    strategies.momentum_expansion.features.live_feature_panel_4h, which bundles
    both steps unconditionally). Both deployed boosters' own
    feature_manifest.json require every one of these columns; silently
    missing them degrades live scoring with no error (see LIVING_SUMMARY.md
    2026-07-19). This asserts the two post-processing steps, applied to a
    live-shaped multi-ticker frame, supply every such column either manifest
    lists -- so a future feature-set change that isn't wired into the shared
    builder fails a test instead of failing silently.
    """
    ts = pd.Timestamp("2026-01-02 14:00", tz="UTC")
    idx = pd.MultiIndex.from_tuples(
        [(ts, "AAA"), (ts, "BBB"), (ts, "CCC")], names=["timestamp", "ticker"])
    raw = pd.DataFrame({
        "ret_5": [0.01, 0.02, -0.01], "ret_20": [0.05, -0.02, 0.03],
        "rs_spy_20": [0.01, 0.02, 0.03], "rvol_20": [1.1, 0.9, 1.5],
        "dollar_vol_surge_20": [1.0, 1.2, 0.8], "atr_expand_14_60": [1.0, 1.1, 0.9],
        "atr_pct_14": [0.02, 0.03, 0.01], "dist_to_52w_high_atr": [2.0, 5.0, 1.0],
    }, index=idx)

    out = _add_cross_sectional_features(raw.copy())
    out = _add_earnings_features_4h(out)

    live_only_prefixes = ("xsec_", "days_to_earnings", "days_since_earnings",
                          "is_pre_earnings_3d", "is_post_earnings_3d", "earnings_in_fwd_window")
    for scorer_cls, label in ((ExpansionRanker, "momentum"), (HTFSwingScorer, "htf")):
        expected = scorer_cls().feature_columns
        missing = [c for c in expected if c.startswith(live_only_prefixes) and c not in out.columns]
        assert not missing, f"{label} manifest columns not produced by live post-processing: {missing}"
