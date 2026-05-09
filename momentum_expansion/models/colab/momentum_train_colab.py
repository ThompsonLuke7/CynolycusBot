# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # momentum_expansion — Colab training notebook (4H expansion model)
#
# Run this on Colab (or any GPU/CPU box). Local execution is intentionally
# avoided per project rules.
#
# Inputs (upload `momentum_colab_bundle.tgz`):
#   - training_matrix_4h.parquet   (features + expansion_target)
#   - feature_manifest.json
#   - label_manifest.json
#
# Outputs (download back to repo):
#   - expansion_xgb.json           (booster)
#   - feature_manifest.json        (echo of input)
#   - eval_metrics.json            (OOF metrics)
#   - oof_preds.parquet            (out-of-fold predictions, optional)

# %%
# Cell 1 — install / import
# !pip install -q xgboost==2.* lightgbm==4.* pandas pyarrow scikit-learn

import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

# %%
# Cell 2 — unpack bundle (uploaded as momentum_colab_bundle.tgz)
BUNDLE = Path("momentum_colab_bundle.tgz")
WORK   = Path("momentum_work")
WORK.mkdir(exist_ok=True)
with tarfile.open(BUNDLE, "r:gz") as tar:
    tar.extractall(WORK)

manifest = json.loads((WORK / "feature_manifest.json").read_text())
FEATURES = manifest["feature_columns"]
print("features:", len(FEATURES))
print("rows:", manifest["n_rows"], "tickers:", manifest["n_tickers"])

df = pd.read_parquet(WORK / "training_matrix_4h.parquet")
print(df.shape, df.head().T)

# %%
# Cell 3 — walk-forward CV split
# Each fold trains on train_years of history, leaves embargo_days, then
# tests on test_months. Step size = test window so folds are non-overlapping.
from datetime import timedelta

WF = manifest["walk_forward"]
TRAIN_YEARS  = float(WF["train_years"])
EMBARGO_DAYS = int(WF["embargo_days"])
TEST_MONTHS  = int(WF["test_months"])

ts = pd.to_datetime(df.index.get_level_values(0))
date_min, date_max = ts.min(), ts.max()
print("range", date_min, "->", date_max)

folds = []
test_end = date_max
while True:
    test_start = test_end - pd.DateOffset(months=TEST_MONTHS)
    train_end  = test_start - timedelta(days=EMBARGO_DAYS)
    train_start = train_end - pd.DateOffset(years=int(TRAIN_YEARS))
    if train_start < date_min:
        break
    folds.append({
        "train_start": train_start, "train_end": train_end,
        "test_start": test_start,   "test_end": test_end,
    })
    test_end = test_start
folds = list(reversed(folds))
print(f"folds: {len(folds)}")
for f in folds:
    print(f)

# %%
# Cell 4 — fit XGBoost per fold, gather OOF predictions
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="aucpr",
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=20,
    subsample=0.80,
    colsample_bytree=0.70,
    reg_alpha=0.10,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=80,
)

oof_rows = []
fold_metrics = []
for fi, f in enumerate(folds):
    m_train = (ts >= f["train_start"]) & (ts <= f["train_end"])
    m_test  = (ts >= f["test_start"])  & (ts <= f["test_end"])

    Xtr, ytr = df.loc[m_train, FEATURES], df.loc[m_train, "expansion_target"].astype(int)
    Xte, yte = df.loc[m_test,  FEATURES], df.loc[m_test,  "expansion_target"].astype(int)
    if len(Xtr) < int(WF["min_train_rows"]) or len(Xte) < 1000:
        continue

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    p = model.predict_proba(Xte)[:, 1]

    auc = roc_auc_score(yte, p) if yte.nunique() > 1 else float("nan")
    ap  = average_precision_score(yte, p) if yte.nunique() > 1 else float("nan")
    fold_metrics.append({"fold": fi, "n_train": len(Xtr), "n_test": len(Xte), "auc": auc, "ap": ap})
    print(fold_metrics[-1])

    fold_oof = pd.DataFrame({"score": p, "y": yte.values}, index=Xte.index)
    oof_rows.append(fold_oof)

if oof_rows:
    oof = pd.concat(oof_rows)
    oof.to_parquet(WORK / "oof_preds.parquet")
    print("aggregate OOF AUC:", roc_auc_score(oof["y"], oof["score"]))
    print("aggregate OOF AP :", average_precision_score(oof["y"], oof["score"]))

# %%
# Cell 5 — final fit on all data, save booster
final_model = xgb.XGBClassifier(**{**XGB_PARAMS, "early_stopping_rounds": None})
final_model.fit(df[FEATURES], df["expansion_target"].astype(int))
booster_path = WORK / "expansion_xgb.json"
final_model.get_booster().save_model(str(booster_path))
print("saved booster ->", booster_path)

# %%
# Cell 6 — write metrics + final manifest
metrics = {
    "fold_metrics": fold_metrics,
    "n_folds": len(fold_metrics),
    "params":  XGB_PARAMS,
    "feature_columns": FEATURES,
}
(WORK / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
(WORK / "feature_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

# %%
# Cell 7 — bundle outputs for download
import shutil
out_bundle = Path("momentum_model_bundle.tgz")
with tarfile.open(out_bundle, "w:gz") as tar:
    for name in ("expansion_xgb.json", "feature_manifest.json", "eval_metrics.json"):
        p = WORK / name
        if p.exists():
            tar.add(p, arcname=name)
    if (WORK / "oof_preds.parquet").exists():
        tar.add(WORK / "oof_preds.parquet", arcname="oof_preds.parquet")
print("download:", out_bundle)
