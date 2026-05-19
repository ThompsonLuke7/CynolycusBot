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
import os
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
# Cell 4 — tune XGBoost, fit per fold, gather clean OOF predictions
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import ParameterSampler

BASE_XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    early_stopping_rounds=100,
)

PARAM_SPACE = {
    "n_estimators": [1200, 1800, 2400, 3200],
    "learning_rate": [0.015, 0.02, 0.03, 0.05],
    "max_depth": [3, 4, 5, 6],
    "min_child_weight": [10, 20, 40, 80],
    "subsample": [0.70, 0.80, 0.90],
    "colsample_bytree": [0.55, 0.70, 0.85],
    "reg_alpha": [0.0, 0.05, 0.10, 0.30],
    "reg_lambda": [0.75, 1.0, 2.0, 4.0],
    "max_delta_step": [0, 1],
}
N_PARAM_TRIALS = int(os.environ.get("MOMENTUM_XGB_PARAM_TRIALS", "8"))
VAL_FRACTION = 0.20
RANDOM_STATE = 42


def _time_train_val_split(index, fraction=VAL_FRACTION):
    fold_ts = pd.to_datetime(index.get_level_values(0))
    unique_days = pd.Series(fold_ts.normalize().unique()).sort_values().to_numpy()
    split_at = max(1, int(len(unique_days) * (1.0 - fraction)))
    if split_at >= len(unique_days):
        split_at = len(unique_days) - 1
    val_start = pd.Timestamp(unique_days[split_at])
    train_mask = fold_ts < val_start
    val_mask = fold_ts >= val_start
    return train_mask, val_mask


def _scale_pos_weight(y):
    pos = int(np.sum(np.asarray(y) == 1))
    neg = int(np.sum(np.asarray(y) == 0))
    return float(neg / max(pos, 1))


def _fit_xgb(Xtr, ytr, Xval, yval, params):
    model = xgb.XGBClassifier(
        **BASE_XGB_PARAMS,
        **params,
        scale_pos_weight=_scale_pos_weight(ytr),
    )
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return model


def _predict(model, X):
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        return model.predict_proba(X)[:, 1]
    return model.predict_proba(X, iteration_range=(0, int(best_iteration) + 1))[:, 1]


INT_PARAM_KEYS = {"n_estimators", "max_depth", "min_child_weight", "max_delta_step"}


def _coerce_xgb_params(params):
    out = {}
    for key, value in params.items():
        if hasattr(value, "item"):
            value = value.item()
        if key in INT_PARAM_KEYS:
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out


BEST_PARAMS_JSON = os.environ.get("MOMENTUM_XGB_BEST_PARAMS_JSON", "").strip()
if BEST_PARAMS_JSON:
    BEST_PARAMS = _coerce_xgb_params(json.loads(BEST_PARAMS_JSON))
    tuning = pd.DataFrame([{"trial": "env", "mean_val_ap": np.nan, "n_folds": 0, **BEST_PARAMS}])
    tuning.to_csv(WORK / "xgb_param_search.csv", index=False)
    print("using env best params:", BEST_PARAMS)
else:
    candidate_params = list(ParameterSampler(PARAM_SPACE, n_iter=N_PARAM_TRIALS, random_state=RANDOM_STATE))
    tuning_rows = []
    for pi, params in enumerate(candidate_params):
        fold_scores = []
        for fi, f in enumerate(folds):
            m_train_full = (ts >= f["train_start"]) & (ts <= f["train_end"])
            train_full = df.loc[m_train_full]
            if len(train_full) < int(WF["min_train_rows"]):
                continue

            inner_train_mask, inner_val_mask = _time_train_val_split(train_full.index)
            Xtr = train_full.loc[inner_train_mask, FEATURES]
            ytr = train_full.loc[inner_train_mask, "expansion_target"].astype(int)
            Xval = train_full.loc[inner_val_mask, FEATURES]
            yval = train_full.loc[inner_val_mask, "expansion_target"].astype(int)
            if len(Xtr) < int(WF["min_train_rows"]) // 2 or len(Xval) < 1000 or yval.nunique() < 2:
                continue

            model = _fit_xgb(Xtr, ytr, Xval, yval, params)
            pval = _predict(model, Xval)
            fold_scores.append(average_precision_score(yval, pval))

        mean_ap = float(np.mean(fold_scores)) if fold_scores else float("nan")
        tuning_rows.append({"trial": pi, "mean_val_ap": mean_ap, "n_folds": len(fold_scores), **params})
        print("tune", tuning_rows[-1])

    tuning = pd.DataFrame(tuning_rows).sort_values("mean_val_ap", ascending=False)
    tuning.to_csv(WORK / "xgb_param_search.csv", index=False)
    BEST_PARAMS = _coerce_xgb_params({k: tuning.iloc[0][k] for k in PARAM_SPACE})
print("best params:", BEST_PARAMS)

oof_rows = []
fold_metrics = []
for fi, f in enumerate(folds):
    m_train_full = (ts >= f["train_start"]) & (ts <= f["train_end"])
    m_test       = (ts >= f["test_start"])  & (ts <= f["test_end"])

    train_full = df.loc[m_train_full]
    Xte, yte = df.loc[m_test,  FEATURES], df.loc[m_test,  "expansion_target"].astype(int)
    if len(train_full) < int(WF["min_train_rows"]) or len(Xte) < 1000:
        continue

    inner_train_mask, inner_val_mask = _time_train_val_split(train_full.index)
    Xtr = train_full.loc[inner_train_mask, FEATURES]
    ytr = train_full.loc[inner_train_mask, "expansion_target"].astype(int)
    Xval = train_full.loc[inner_val_mask, FEATURES]
    yval = train_full.loc[inner_val_mask, "expansion_target"].astype(int)
    if len(Xtr) < int(WF["min_train_rows"]) // 2 or len(Xval) < 1000:
        continue

    model = _fit_xgb(Xtr, ytr, Xval, yval, BEST_PARAMS)
    p = _predict(model, Xte)

    auc = roc_auc_score(yte, p) if yte.nunique() > 1 else float("nan")
    ap  = average_precision_score(yte, p) if yte.nunique() > 1 else float("nan")
    fold_metrics.append({
        "fold": fi,
        "n_train": len(Xtr),
        "n_val": len(Xval),
        "n_test": len(Xte),
        "best_iteration": int(model.best_iteration) if getattr(model, "best_iteration", None) is not None else None,
        "auc": auc,
        "ap": ap,
    })
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
final_train_mask, final_val_mask = _time_train_val_split(df.index)
round_selector = _fit_xgb(
    df.loc[final_train_mask, FEATURES],
    df.loc[final_train_mask, "expansion_target"].astype(int),
    df.loc[final_val_mask, FEATURES],
    df.loc[final_val_mask, "expansion_target"].astype(int),
    BEST_PARAMS,
)
best_n_estimators = (
    int(round_selector.best_iteration) + 1
    if getattr(round_selector, "best_iteration", None) is not None
    else int(BEST_PARAMS["n_estimators"])
)
print("final best_n_estimators:", best_n_estimators)

final_params = {**BEST_PARAMS, "n_estimators": best_n_estimators}
final_model = xgb.XGBClassifier(
    **{k: v for k, v in BASE_XGB_PARAMS.items() if k != "early_stopping_rounds"},
    **final_params,
    scale_pos_weight=_scale_pos_weight(df["expansion_target"].astype(int)),
)
final_model.fit(df[FEATURES], df["expansion_target"].astype(int), verbose=False)
booster_path = WORK / "expansion_xgb.json"
final_model.get_booster().save_model(str(booster_path))
print("saved booster ->", booster_path)

# %%
# Cell 6 — write metrics + final manifest
metrics = {
    "fold_metrics": fold_metrics,
    "n_folds": len(fold_metrics),
    "base_params": BASE_XGB_PARAMS,
    "best_params": BEST_PARAMS,
    "best_n_estimators": best_n_estimators,
    "n_param_trials": N_PARAM_TRIALS,
    "validation_fraction": VAL_FRACTION,
    "feature_columns": FEATURES,
}
(WORK / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
(WORK / "feature_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

# %%
# Cell 7 — bundle outputs for download
import shutil
out_bundle = Path("momentum_model_bundle.tgz")
with tarfile.open(out_bundle, "w:gz") as tar:
    for name in ("expansion_xgb.json", "feature_manifest.json", "eval_metrics.json", "xgb_param_search.csv"):
        p = WORK / name
        if p.exists():
            tar.add(p, arcname=name)
    if (WORK / "oof_preds.parquet").exists():
        tar.add(WORK / "oof_preds.parquet", arcname="oof_preds.parquet")
print("download:", out_bundle)
