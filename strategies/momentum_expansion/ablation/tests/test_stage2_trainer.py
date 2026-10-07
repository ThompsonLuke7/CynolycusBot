"""Control-flow tests for the Stage-2 Colab trainer. No model is trained here:
``fit_predict`` is replaced by a stub (project rule: training runs on Colab)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategies.model_training.colab_competition import date_folds as harness_date_folds
from strategies.momentum_expansion.ablation import export_stage2 as E

_PATH = Path(__file__).resolve().parents[1] / "colab" / "stage2_train_colab.py"
_spec = importlib.util.spec_from_file_location("stage2_train_colab", _PATH)
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)


def _matrix(tmp_path: Path, n_tickers: int = 60, n_days: int = 1500) -> tuple[Path, dict]:
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2020-01-02", periods=n_days, tz="UTC")
    ts = np.repeat(days.to_numpy(), n_tickers)
    df = pd.DataFrame({"timestamp": pd.to_datetime(ts, utc=True),
                       "ticker": np.tile([f"T{i:03d}" for i in range(n_tickers)], n_days)})
    for c in ("f1", "f2", "f3"):
        df[c] = rng.normal(size=len(df)).astype("float32")
    df.loc[df.sample(frac=0.02, random_state=1).index, "f2"] = np.nan
    df["y_A"] = (rng.random(len(df)) < 0.2).astype("float32")
    df.loc[df["timestamp"] > days[-11], "y_A"] = np.nan          # last sessions have no label yet
    path = tmp_path / "m.parquet"
    df.to_parquet(path, index=False)
    manifest = {"feature_sets": {"base": ["f1", "f2", "f3"], "two": ["f1", "f3"]},
                "walk_forward": {"train_months": 24, "test_months": 6, "inner_val_frac": 0.2},
                "xgb_params": {"n_estimators": 5}}
    return path, manifest


def _stub(calls: list):
    def fit_predict(X_tr, y_tr, X_va, y_va, X_te, *, seed, params, device):
        assert not np.isnan(X_tr).any() and not np.isnan(y_tr).any() and not np.isnan(X_te).any()
        calls.append((len(X_tr), len(X_va), len(X_te), seed))
        return X_te[:, 0].astype(np.float32) + seed, 7, np.ones(X_tr.shape[1], dtype=np.float32)
    return fit_predict


def test_folds_match_the_deployed_harness():
    ts = pd.Series(pd.bdate_range("2020-01-02", "2026-09-30", tz="UTC"))
    for emb in (21, 35, 63):
        assert T.date_folds(ts, train_months=24, embargo_days=emb, test_months=6) == \
            harness_date_folds(ts, train_months=24, embargo_days=emb, test_months=6)


def test_no_training_row_sits_inside_the_embargo(tmp_path):
    path, manifest = _matrix(tmp_path)
    data = T.load_matrix(path, ["f1", "f2", "f3"], ["y_A"])
    ok = T.complete_rows(data["X"], [0, 1, 2]) & ~np.isnan(data["y"]["y_A"])
    emb = 35
    for fold in T.date_folds(data["ts"], train_months=24, embargo_days=emb, test_months=6):
        tr, va, te = T.fold_indices(data["ts"], ok, fold, embargo_days=emb, inner_val_frac=0.2)
        ts = data["ts"]
        assert ts.iloc[tr].max() < ts.iloc[va].min() - pd.Timedelta(days=emb - 1)
        assert ts.iloc[va].max() <= ts.iloc[te].min() - pd.Timedelta(days=emb)
        assert ok[tr].all() and ok[va].all() and ok[te].all()
        assert not (set(tr) & set(va)) and not (set(va) & set(te))


def test_run_arm_writes_oof_and_resumes_without_refitting(tmp_path):
    path, manifest = _matrix(tmp_path)
    data = T.load_matrix(path, ["f1", "f2", "f3"], ["y_A"])
    out = tmp_path / "out"; out.mkdir()
    arm = {"name": "A", "target": "y_A", "features": "base", "embargo_days": 21, "control": False}
    calls: list = []
    S = T
    small = dict(min_train=500, min_val=50)      # the synthetic matrix is far below the production floor
    res = S.run_arm(arm, data, manifest, out, seeds=[42, 43], fit_predict=_stub(calls), log=lambda *_: None, **small)
    oof = pd.read_parquet(out / "oof_A.parquet")
    assert res["rows"] == len(oof) > 0 and set(oof.columns) == {"timestamp", "ticker", "score", "fold"}
    assert not oof.duplicated(["timestamp", "ticker"]).any()
    assert len(calls) == 2 * oof["fold"].nunique()                     # two seeds per fold
    # scores are the seed average of the stub: f1 + (42 + 43) / 2
    merged = oof.merge(pd.read_parquet(path)[["timestamp", "ticker", "f1"]], on=["timestamp", "ticker"])
    np.testing.assert_allclose(merged["score"], merged["f1"] + 42.5, rtol=1e-5)
    # rows with a missing feature or label never reach the model or the output
    full = pd.read_parquet(path)
    bad = full[full["f2"].isna() | full["y_A"].isna()][["timestamp", "ticker"]]
    assert oof.merge(bad, on=["timestamp", "ticker"]).empty
    assert json.loads((out / "summary_A.json").read_text())["arm"]["name"] == "A"

    # a finished arm is skipped; a half-finished one only fits the missing folds
    n = len(calls)
    assert S.run_arm(arm, data, manifest, out, seeds=[42, 43], fit_predict=_stub(calls), log=lambda *_: None, **small) is None
    assert len(calls) == n
    (out / "oof_A.parquet").unlink()
    first = sorted((out / "folds").glob("A__f*.parquet"))[0]
    first.unlink()
    S.run_arm(arm, data, manifest, out, seeds=[42, 43], fit_predict=_stub(calls), log=lambda *_: None, **small)
    assert len(calls) == n + 2
    pd.testing.assert_frame_equal(pd.read_parquet(out / "oof_A.parquet").sort_values(["timestamp", "ticker"]).reset_index(drop=True),
                                  oof.sort_values(["timestamp", "ticker"]).reset_index(drop=True))
    assert S.pack_results(out).exists()


def test_arm_spec_is_consistent():
    arms = E.arms()
    names = [a["name"] for a in arms]
    assert len(names) == len(set(names))
    fsets = E.feature_sets(["a", "week_sin", "b", "dow", "days_to_earnings"])
    assert fsets["no_cal"] == ["a", "b", "days_to_earnings"] and fsets["no_cal_leader"][:2] == ["a", "b"]
    assert fsets["no_cal_noearn"] == ["a", "b"]
    for a in arms:
        assert a["features"] in fsets
        assert a["embargo_days"] >= max(21, a["horizon_sessions"] * 7 / 5)   # embargo covers the label window
        if a["control"]:
            assert a["target"].endswith("__perm") and a["name"].endswith("__perm")
    assert {"L0", "L1_10", "L2_10", "L2_20", "L2_40", "PIV"} <= set(names)
    assert E.embargo_days(40) == 63 and E.embargo_days(10) == 21


def test_thin_bars_are_excluded_from_every_arm(tmp_path):
    path, manifest = _matrix(tmp_path)
    df = pd.read_parquet(path)
    stray = df[df["timestamp"] == df["timestamp"].iloc[len(df) // 2]].head(3).copy()
    stray["timestamp"] = stray["timestamp"] + pd.Timedelta(hours=1)        # a late-open bar, 3 names
    pd.concat([df, stray], ignore_index=True).to_parquet(path, index=False)
    data = T.load_matrix(path, ["f1", "f2", "f3"], ["y_A"], min_names_per_bar=10)
    assert (~data["full_bar"]).sum() == 3 and T.full_bar_rows(data["ts"], 1).all()
    # the smoke subset keeps the SAME bars: counted on the whole matrix, not on the 1-in-4 tickers kept
    sub = T.load_matrix(path, ["f1", "f2", "f3"], ["y_A"], min_names_per_bar=30, ticker_frac=0.25)
    assert len(np.unique(sub["ticker"])) == 15 and sub["full_bar"].mean() > 0.99
    out = tmp_path / "out"; out.mkdir()
    arm = {"name": "A", "target": "y_A", "features": "base", "embargo_days": 21, "control": False}
    T.run_arm(arm, data, manifest, out, seeds=[42], fit_predict=_stub([]),
              log=lambda *_: None, min_train=500, min_val=50)
    oof = pd.read_parquet(out / "oof_A.parquet")
    assert not oof["timestamp"].isin(stray["timestamp"]).any()


def test_bundle_is_found_as_loose_files(tmp_path, monkeypatch):
    (tmp_path / T.MANIFEST_NAME).write_text("{}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(T, "DRIVE_DIR", tmp_path / "no_drive" / "cynolycus_stage2")
    work, out = T.locate_bundle()
    assert work.resolve() == tmp_path.resolve() and out.is_dir()
