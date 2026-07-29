from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.parabolic_filter import (
    ParabolicFilter, atr_rule_rank, recommended_selector,
    DEFAULT_HORIZON_BARS, DEFAULT_THRESHOLD_PCT,
)


def _frame(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "entry_px_underlying": rng.uniform(5, 100, n),
        "atr_at_entry": rng.uniform(0.05, 6.0, n),
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC"),
    })


# --------------------------------------------------------------- ATR rule (HTF)

def test_atr_rule_ranks_higher_volatility_higher():
    df = pd.DataFrame({"atr_at_entry": [1.0, 5.0, 0.1], "entry_px_underlying": [100.0, 100.0, 100.0]})
    r = atr_rule_rank(df)
    assert r.iloc[1] > r.iloc[0] > r.iloc[2], "higher ATR% must rank higher"


def test_atr_rule_is_scale_free():
    """A $10 stock with 5% ATR must rank equal to a $1000 stock with 5% ATR."""
    df = pd.DataFrame({"atr_at_entry": [0.5, 50.0], "entry_px_underlying": [10.0, 1000.0]})
    r = atr_rule_rank(df)
    assert r.iloc[0] == r.iloc[1]


def test_atr_rule_missing_column_raises():
    with pytest.raises(KeyError):
        atr_rule_rank(pd.DataFrame({"atr_at_entry": [1.0]}))


def test_atr_rule_handles_bad_values_as_nan_not_zero():
    df = pd.DataFrame({"atr_at_entry": [1.0, np.nan, 2.0], "entry_px_underlying": [100.0, 100.0, 0.0]})
    r = atr_rule_rank(df)
    assert r.isna().sum() == 2, "NaN and divide-by-zero must be NaN, never coerced to a rank"


# ------------------------------------------------------- per-module recommendation

def test_recommended_selector_matches_validated_evidence():
    assert recommended_selector("momentum_expansion") == "model"
    assert recommended_selector("multi_ticker_swing_htf") == "atr_rule"


def test_recommended_selector_refuses_unvalidated_module():
    """No recommendation exists for these -- must fail loudly, not guess."""
    for mod in ("meta_ranker", "dealer_ranker", "intraday_structure", "multi_ticker_swing"):
        with pytest.raises(ValueError):
            recommended_selector(mod)


# --------------------------------------------------------------- model scorer

class _Stub:
    """Minimal stand-in for a fitted XGBClassifier."""

    def __init__(self, n_feat=2):
        self.n_feat = n_feat

    def predict_proba(self, X):
        s = 1.0 / (1.0 + np.exp(-np.nan_to_num(np.asarray(X, dtype=float)).sum(axis=1)))
        return np.column_stack([1 - s, s])


def _filter(feats=("f1", "f2")):
    return ParabolicFilter(booster=_Stub(len(feats)), feature_names=list(feats),
                           module="momentum_expansion")


def test_predict_proba_returns_probabilities():
    p = _filter().predict_proba(_frame())
    assert p.shape == (50,)
    assert ((p >= 0) & (p <= 1)).all()


def test_missing_feature_raises_rather_than_silently_scoring():
    df = _frame().drop(columns=["f2"])
    with pytest.raises(KeyError, match="missing"):
        _filter().predict_proba(df)


def test_lookahead_guard_raises():
    df = _frame()
    asof = df.timestamp.iloc[10]
    with pytest.raises(ValueError, match="lookahead"):
        _filter().predict_proba(df, asof=asof)


def test_lookahead_guard_passes_when_features_precede_asof():
    df = _frame()
    asof = df.timestamp.max() + pd.Timedelta("1h")
    p = _filter().predict_proba(df, asof=asof)
    assert len(p) == len(df)


def test_select_returns_requested_fraction():
    df = _frame(n=100)
    mask = _filter().select(df, top_fraction=0.20)
    assert 15 <= int(mask.sum()) <= 25, "top 20% of 100 should select ~20"


def test_select_uses_validated_default_for_momentum():
    df = _frame(n=100)
    mask = _filter().select(df)          # momentum default = 0.20
    assert 15 <= int(mask.sum()) <= 25


def test_select_refuses_unknown_module_without_explicit_fraction():
    f = ParabolicFilter(booster=_Stub(), feature_names=["f1", "f2"], module="some_new_module")
    with pytest.raises(ValueError):
        f.select(_frame())


def test_defaults_match_the_validated_label_definition():
    """The evidence in 09_shares_parabolic_filter.md is for +25% within 20 bars."""
    assert DEFAULT_THRESHOLD_PCT == 0.25
    assert DEFAULT_HORIZON_BARS == 20


def test_roundtrip_save_load(tmp_path):
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 2)), columns=["f1", "f2"])
    y = (X.f1 + X.f2 > 0).astype(int)
    m = xgb.XGBClassifier(n_estimators=10, max_depth=2, eval_metric="logloss")
    m.fit(X, y)
    f = ParabolicFilter(booster=m, feature_names=["f1", "f2"], module="momentum_expansion",
                        trained_through="2025-12-01")
    f.save(tmp_path / "m")
    g = ParabolicFilter.load(tmp_path / "m")
    assert g.feature_names == ["f1", "f2"]
    assert g.module == "momentum_expansion"
    assert g.trained_through == "2025-12-01"
    assert np.allclose(f.predict_proba(X), g.predict_proba(X))
