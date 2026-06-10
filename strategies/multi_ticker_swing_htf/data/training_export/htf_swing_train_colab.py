# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # multi_ticker_swing_htf — Colab training notebook (4H pivot-swing model)
#
# Upload `htf_swing_colab_bundle.tgz`, then run the cells top to bottom.
#
# Inputs:
#   - htf_training_matrix_4h.parquet   (features + htf_swing_score)
#   - feature_manifest.json
#   - label_manifest.json
#
# Outputs (download via htf_swing_model_bundle.tgz):
#   - htf_swing_xgb.json
#   - eval_metrics.json
#   - oof_preds.parquet           (out-of-fold predictions for backtesting)
#   - oof_score_buckets.csv
#   - feature_importance.csv

# %%
# !pip install -q xgboost==2.* pandas pyarrow scikit-learn

import json
import os
import shutil
import tarfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# %%
BUNDLE = Path("htf_swing_colab_bundle.tgz")
WORK = Path("htf_work")
WORK.mkdir(exist_ok=True)
with tarfile.open(BUNDLE, "r:gz") as tar:
    try:
        tar.extractall(WORK, filter="data")
    except TypeError:
        tar.extractall(WORK)

manifest = json.loads((WORK / "feature_manifest.json").read_text())
FEATURES = manifest["feature_columns"]
TARGET = manifest["target_column"]
TARGET_KIND = manifest.get("target_kind", "regression")
if TARGET_KIND != "regression":
    raise ValueError(f"HTF trainer expects regression target, got {TARGET_KIND}")
WF = manifest["walk_forward"]
print("features:", len(FEATURES), "| target:", TARGET, "| rows:", manifest["n_rows"])

df = pd.read_parquet(WORK / "htf_training_matrix_4h.parquet")
df = df[df[TARGET].notna()].copy()
print(df.shape)

DIAGNOSTIC_COLUMNS = [c for c in manifest.get("label_columns", []) if c in df.columns]

# %%
# Walk-forward folds: train_years window, embargo gap, non-overlapping test_months.
TRAIN_YEARS = float(WF["train_years"])
EMBARGO_DAYS = int(WF["embargo_days"])
TEST_MONTHS = int(WF["test_months"])

ts = pd.to_datetime(df.index.get_level_values(0))
date_min, date_max = ts.min(), ts.max()
print("range", date_min, "->", date_max)

folds = []
test_end = date_max
while True:
    test_start = test_end - pd.DateOffset(months=TEST_MONTHS)
    train_end = test_start - timedelta(days=EMBARGO_DAYS)
    train_start = train_end - pd.DateOffset(years=int(TRAIN_YEARS))
    if train_start < date_min:
        break
    folds.append({"train_start": train_start, "train_end": train_end,
                  "test_start": test_start, "test_end": test_end})
    test_end = test_start
folds = list(reversed(folds))
print("folds:", len(folds))
for f in folds:
    print(f)

# %%
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterSampler

# Use the Colab GPU when present (nvidia-smi on PATH), else CPU. Override with XGB_DEVICE.
XGB_DEVICE = os.environ.get("XGB_DEVICE") or ("cuda" if shutil.which("nvidia-smi") else "cpu")
print("xgb device:", XGB_DEVICE)

BASE_PARAMS = dict(
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    device=XGB_DEVICE,
    early_stopping_rounds=75,
)
PARAM_SPACE = {
    "n_estimators": [800, 1200, 1800, 2400],
    "learning_rate": [0.015, 0.025, 0.04, 0.06],
    "max_depth": [3, 4, 5, 6],
    "min_child_weight": [10, 20, 40, 80],
    "subsample": [0.70, 0.80, 0.90],
    "colsample_bytree": [0.60, 0.75, 0.90],
    "reg_alpha": [0.0, 0.05, 0.20],
    "reg_lambda": [1.0, 2.0, 4.0],
}
N_TRIALS = int(os.environ.get("HTF_XGB_PARAM_TRIALS", "25"))
VAL_FRACTION = 0.20
INT_KEYS = {"n_estimators", "max_depth", "min_child_weight"}


def time_train_val_split(index, fraction=VAL_FRACTION):
    fold_ts = pd.to_datetime(index.get_level_values(0))
    days = pd.Series(fold_ts.normalize().unique()).sort_values().to_numpy()
    split_at = min(max(1, int(len(days) * (1.0 - fraction))), len(days) - 1)
    val_start = pd.Timestamp(days[split_at])
    return fold_ts < val_start, fold_ts >= val_start


def fit_xgb(Xtr, ytr, Xval, yval, params):
    model = xgb.XGBRegressor(**BASE_PARAMS, **params)
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return model


def predict(model, X):
    bi = getattr(model, "best_iteration", None)
    return model.predict(X) if bi is None else model.predict(X, iteration_range=(0, int(bi) + 1))


def coerce(params):
    return {k: (int(v) if k in INT_KEYS else float(v)) for k, v in params.items()}


# %%
# Hyperparameter search: score by mean validation Spearman across folds' inner-val.
candidates = list(ParameterSampler(PARAM_SPACE, n_iter=N_TRIALS, random_state=42))
tuning_rows = []
for pi, params in enumerate(candidates):
    scores = []
    for f in folds:
        m_train = (ts >= f["train_start"]) & (ts <= f["train_end"])
        train_full = df.loc[m_train]
        if len(train_full) < int(WF["min_train_rows"]):
            continue
        itr, ival = time_train_val_split(train_full.index)
        Xtr, ytr = train_full.loc[itr, FEATURES], train_full.loc[itr, TARGET].astype(float)
        Xval, yval = train_full.loc[ival, FEATURES], train_full.loc[ival, TARGET].astype(float)
        if len(Xtr) < int(WF["min_train_rows"]) // 2 or len(Xval) < 500:
            continue
        model = fit_xgb(Xtr, ytr, Xval, yval, params)
        scores.append(pd.Series(predict(model, Xval), index=yval.index).corr(yval, method="spearman"))
    tuning_rows.append({"trial": pi, "mean_val_spearman": float(np.mean(scores)) if scores else float("nan"),
                        "n_folds": len(scores), **params})
    print(tuning_rows[-1])

tuning = pd.DataFrame(tuning_rows).sort_values("mean_val_spearman", ascending=False)
tuning.to_csv(WORK / "xgb_param_search.csv", index=False)
BEST_PARAMS = coerce({k: tuning.iloc[0][k] for k in PARAM_SPACE})
print("best params:", BEST_PARAMS)

# %%
# Walk-forward OOF predictions (each fold trained only on prior data).
oof_rows, fold_metrics = [], []
for fi, f in enumerate(folds):
    m_train = (ts >= f["train_start"]) & (ts <= f["train_end"])
    m_test = (ts >= f["test_start"]) & (ts <= f["test_end"])
    train_full = df.loc[m_train]
    Xte, yte = df.loc[m_test, FEATURES], df.loc[m_test, TARGET].astype(float)
    if len(train_full) < int(WF["min_train_rows"]) or len(Xte) < 500:
        continue
    itr, ival = time_train_val_split(train_full.index)
    Xtr, ytr = train_full.loc[itr, FEATURES], train_full.loc[itr, TARGET].astype(float)
    Xval, yval = train_full.loc[ival, FEATURES], train_full.loc[ival, TARGET].astype(float)
    if len(Xtr) < int(WF["min_train_rows"]) // 2 or len(Xval) < 500:
        continue
    model = fit_xgb(Xtr, ytr, Xval, yval, BEST_PARAMS)
    p = predict(model, Xte)
    fold_metrics.append({
        "fold": fi, "n_train": len(Xtr), "n_val": len(Xval), "n_test": len(Xte),
        "best_iteration": int(model.best_iteration) if getattr(model, "best_iteration", None) is not None else None,
        "rmse": float(np.sqrt(mean_squared_error(yte, p))),
        "mae": float(mean_absolute_error(yte, p)),
        "spearman": float(pd.Series(p, index=yte.index).corr(yte, method="spearman")),
    })
    print(fold_metrics[-1])
    fold_oof = pd.DataFrame({"score": p, "y": yte.values}, index=Xte.index)
    if DIAGNOSTIC_COLUMNS:
        fold_oof = fold_oof.join(df.loc[m_test, DIAGNOSTIC_COLUMNS], how="left")
    oof_rows.append(fold_oof)

if oof_rows:
    oof = pd.concat(oof_rows)
    oof.to_parquet(WORK / "oof_preds.parquet")
    print("aggregate OOF RMSE:", float(np.sqrt(mean_squared_error(oof["y"], oof["score"]))))
    print("aggregate OOF Spearman:", oof["score"].corr(oof["y"], method="spearman"))

    bucket_rows = []
    ranked = oof["score"].rank(method="first")
    oof_diag = oof.copy()
    oof_diag["decile"] = pd.qcut(ranked, 10, labels=False) + 1
    for decile, g in oof_diag.groupby("decile"):
        rec = {"bucket": f"decile_{int(decile)}", "n": int(len(g)),
               "score_mean": float(g["score"].mean()), "target_mean": float(g["y"].mean())}
        for col in ("fwd_best_high_return", "fwd_worst_low_return", "fwd_close_return"):
            if col in g.columns:
                rec[f"avg_{col}"] = float(g[col].mean())
        bucket_rows.append(rec)
    pd.DataFrame(bucket_rows).to_csv(WORK / "oof_score_buckets.csv", index=False)
    print(pd.DataFrame(bucket_rows).to_string(index=False))

# %%
# Final fit on all rows: pick n_estimators via a held-out tail, then refit on everything.
final_tr, final_val = time_train_val_split(df.index)
selector = fit_xgb(df.loc[final_tr, FEATURES], df.loc[final_tr, TARGET].astype(float),
                   df.loc[final_val, FEATURES], df.loc[final_val, TARGET].astype(float), BEST_PARAMS)
best_n = int(selector.best_iteration) + 1 if getattr(selector, "best_iteration", None) is not None else int(BEST_PARAMS["n_estimators"])
print("final best_n_estimators:", best_n)

final_params = {**BEST_PARAMS, "n_estimators": best_n}
final_model = xgb.XGBRegressor(
    **{k: v for k, v in BASE_PARAMS.items() if k != "early_stopping_rounds"}, **final_params,
)
final_model.fit(df[FEATURES], df[TARGET].astype(float), verbose=False)
final_model.get_booster().save_model(str(WORK / "htf_swing_xgb.json"))
print("saved booster")

importance_rows = []
for kind in ("gain", "weight", "cover"):
    for feature, value in final_model.get_booster().get_score(importance_type=kind).items():
        importance_rows.append({"importance_type": kind, "feature": feature, "value": float(value)})
pd.DataFrame(importance_rows).to_csv(WORK / "feature_importance.csv", index=False)

metrics = {
    "fold_metrics": fold_metrics,
    "n_folds": len(fold_metrics),
    "base_params": BASE_PARAMS,
    "best_params": BEST_PARAMS,
    "best_n_estimators": best_n,
    "n_param_trials": N_TRIALS,
    "target_column": TARGET,
    "feature_columns": FEATURES,
}
(WORK / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
(WORK / "feature_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

# %%
out_bundle = Path("htf_swing_model_bundle.tgz")
with tarfile.open(out_bundle, "w:gz") as tar:
    for name in ("htf_swing_xgb.json", "feature_manifest.json", "eval_metrics.json",
                 "xgb_param_search.csv", "oof_score_buckets.csv", "feature_importance.csv"):
        p = WORK / name
        if p.exists():
            tar.add(p, arcname=name)
    if (WORK / "oof_preds.parquet").exists():
        tar.add(WORK / "oof_preds.parquet", arcname="oof_preds.parquet")
print("download:", out_bundle)
