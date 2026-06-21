"""Smoke tests for RankerSwingScanner directional-probability reconstruction."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from strategies.multi_ticker_swing.config.pipeline_config import MODULE_ROOT
from strategies.multi_ticker_swing.live.ranker_scanner import RankerSwingScanner

BUNDLE = MODULE_ROOT / "models" / "oof_ranker_20260618"
pytestmark = pytest.mark.skipif(
    not (BUNDLE / "model_ranker_long.joblib").exists(),
    reason="OOF ranker bundle not present",
)


def _bare_scanner() -> RankerSwingScanner:
    """Construct without touching the feature builder / universe loader."""
    sc = RankerSwingScanner.__new__(RankerSwingScanner)
    sc._long = joblib.load(BUNDLE / "model_ranker_long.joblib")
    sc._short = joblib.load(BUNDLE / "model_ranker_short.joblib")
    sc._calib_long = joblib.load(BUNDLE / "calib_long.joblib")
    sc._calib_short = joblib.load(BUNDLE / "calib_short.joblib")
    sc._feats = json.loads((BUNDLE / "meta.json").read_text())["feature_names"]
    sc._warned_missing = False
    return sc


def _synthetic(n: int, feats: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((n, len(feats))).astype("float32"), columns=feats)


def test_raw_probs_valid():
    sc = _bare_scanner()
    pl, ps = sc._raw_p(_synthetic(256, sc._feats))
    assert pl.shape == ps.shape == (256,)
    assert np.all((pl >= 0) & (pl <= 1)) and np.all((ps >= 0) & (ps <= 1))   # calibrated P(win)


def test_nan_served_features_are_handled():
    """Missing context features (live builder gap) must not crash; on REAL feature rows the
    NaN-served raw P stays well-correlated with the full-feature output."""
    sc = _bare_scanner()
    mx = MODULE_ROOT / "backtest" / "competition_20260615_test_matrix.parquet"
    if mx.exists():
        full = pd.read_parquet(mx, columns=sc._feats).head(400)
    else:                                   # fallback: synthetic, no-crash guarantee only
        full = _synthetic(256, sc._feats)
    base = [c for c in sc._feats if not c.startswith(
        ("news_", "theme", "treasury", "days_to_earn", "days_since_earn", "is_earnings",
         "is_pre_earn", "is_post_earn", "macro", "primary_theme", "related_theme", "membership",
         "distance_to_", "gap_fill", "breakout_proximity", "support_proximity",
         "resistance_proximity", "inside_", "failed_break", "spy_distance", "spy_inside", "spy_regime"))]
    pl_full, _ = sc._raw_p(full)
    pl_live, _ = sc._raw_p(full[base])                       # scanner reindexes the rest to NaN
    assert np.all(np.isfinite(pl_live))
    if mx.exists():
        assert np.corrcoef(pl_full, pl_live)[0, 1] > 0.9     # NaN-serving preserves ranking
