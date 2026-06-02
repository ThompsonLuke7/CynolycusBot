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
#   - training_matrix_4h.parquet   (features + expansion_survival_score)
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
    try:
        tar.extractall(WORK, filter="data")
    except TypeError:
        tar.extractall(WORK)

manifest = json.loads((WORK / "feature_manifest.json").read_text())
FEATURES = manifest["feature_columns"]
TARGET = manifest.get("target_column", "expansion_survival_score")
TARGET_KIND = manifest.get("target_kind", "regression")
if TARGET_KIND != "regression":
    raise ValueError(f"momentum_expansion trainer now expects regression target, got {TARGET_KIND}")
print("features:", len(FEATURES))
print("target:", TARGET, TARGET_KIND)
print("rows:", manifest["n_rows"], "tickers:", manifest["n_tickers"])

df = pd.read_parquet(WORK / "training_matrix_4h.parquet")
if TARGET not in df.columns:
    raise ValueError(f"Missing target column {TARGET}; rebuild/export the training matrix")
print(df.shape, df.head().T)

DIAGNOSTIC_COLUMNS = [
    c for c in [
        "fwd_max_return",
        "fwd_max_alpha",
        "fwd_atr_adj_return",
        "fwd_max_drawdown",
        "fwd_close_return",
        "trend_persistence",
        "expansion_score",
        "expansion_target",
    ]
    if c in df.columns
]

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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterSampler

BASE_XGB_PARAMS = dict(
    objective="reg:squarederror",
    eval_metric="rmse",
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


def _fit_xgb(Xtr, ytr, Xval, yval, params):
    model = xgb.XGBRegressor(
        **BASE_XGB_PARAMS,
        **params,
    )
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return model


def _predict(model, X):
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        return model.predict(X)
    return model.predict(X, iteration_range=(0, int(best_iteration) + 1))


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
    tuning = pd.DataFrame([{"trial": "env", "mean_val_spearman": np.nan, "n_folds": 0, **BEST_PARAMS}])
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
            ytr = train_full.loc[inner_train_mask, TARGET].astype(float)
            Xval = train_full.loc[inner_val_mask, FEATURES]
            yval = train_full.loc[inner_val_mask, TARGET].astype(float)
            if len(Xtr) < int(WF["min_train_rows"]) // 2 or len(Xval) < 1000:
                continue

            model = _fit_xgb(Xtr, ytr, Xval, yval, params)
            pval = _predict(model, Xval)
            fold_scores.append(pd.Series(pval, index=yval.index).corr(yval, method="spearman"))

        mean_ap = float(np.mean(fold_scores)) if fold_scores else float("nan")
        tuning_rows.append({"trial": pi, "mean_val_spearman": mean_ap, "n_folds": len(fold_scores), **params})
        print("tune", tuning_rows[-1])

    tuning = pd.DataFrame(tuning_rows).sort_values("mean_val_spearman", ascending=False)
    tuning.to_csv(WORK / "xgb_param_search.csv", index=False)
    BEST_PARAMS = _coerce_xgb_params({k: tuning.iloc[0][k] for k in PARAM_SPACE})
print("best params:", BEST_PARAMS)

oof_rows = []
fold_metrics = []
for fi, f in enumerate(folds):
    m_train_full = (ts >= f["train_start"]) & (ts <= f["train_end"])
    m_test       = (ts >= f["test_start"])  & (ts <= f["test_end"])

    train_full = df.loc[m_train_full]
    Xte, yte = df.loc[m_test,  FEATURES], df.loc[m_test,  TARGET].astype(float)
    if len(train_full) < int(WF["min_train_rows"]) or len(Xte) < 1000:
        continue

    inner_train_mask, inner_val_mask = _time_train_val_split(train_full.index)
    Xtr = train_full.loc[inner_train_mask, FEATURES]
    ytr = train_full.loc[inner_train_mask, TARGET].astype(float)
    Xval = train_full.loc[inner_val_mask, FEATURES]
    yval = train_full.loc[inner_val_mask, TARGET].astype(float)
    if len(Xtr) < int(WF["min_train_rows"]) // 2 or len(Xval) < 1000:
        continue

    model = _fit_xgb(Xtr, ytr, Xval, yval, BEST_PARAMS)
    p = _predict(model, Xte)

    rmse = mean_squared_error(yte, p, squared=False)
    mae = mean_absolute_error(yte, p)
    spearman = pd.Series(p, index=yte.index).corr(yte, method="spearman")
    fold_metrics.append({
        "fold": fi,
        "n_train": len(Xtr),
        "n_val": len(Xval),
        "n_test": len(Xte),
        "best_iteration": int(model.best_iteration) if getattr(model, "best_iteration", None) is not None else None,
        "rmse": rmse,
        "mae": mae,
        "spearman": spearman,
    })
    print(fold_metrics[-1])

    fold_oof = pd.DataFrame({"score": p, "y": yte.values}, index=Xte.index)
    if DIAGNOSTIC_COLUMNS:
        fold_oof = fold_oof.join(df.loc[m_test, DIAGNOSTIC_COLUMNS], how="left")
    oof_rows.append(fold_oof)

if oof_rows:
    oof = pd.concat(oof_rows)
    oof.to_parquet(WORK / "oof_preds.parquet")
    print("aggregate OOF RMSE:", mean_squared_error(oof["y"], oof["score"], squared=False))
    print("aggregate OOF MAE :", mean_absolute_error(oof["y"], oof["score"]))
    print("aggregate OOF Spearman:", oof["score"].corr(oof["y"], method="spearman"))

    def _diagnostic_record(bucket, g):
        rec = {
            "bucket": bucket,
            "n": int(len(g)),
            "score_min": float(g["score"].min()),
            "score_mean": float(g["score"].mean()),
            "score_max": float(g["score"].max()),
            "target_mean": float(g["y"].mean()),
        }
        if "fwd_max_return" in g.columns:
            rec["avg_fwd_max_return"] = float(g["fwd_max_return"].mean())
            rec["median_fwd_max_return"] = float(g["fwd_max_return"].median())
            rec["pct_gt_20"] = float((g["fwd_max_return"] >= 0.20).mean())
            rec["pct_gt_25"] = float((g["fwd_max_return"] >= 0.25).mean())
            rec["pct_gt_40"] = float((g["fwd_max_return"] >= 0.40).mean())
        if "fwd_max_alpha" in g.columns:
            rec["avg_fwd_alpha"] = float(g["fwd_max_alpha"].mean())
        if "fwd_max_drawdown" in g.columns:
            rec["avg_drawdown"] = float(g["fwd_max_drawdown"].mean())
            rec["median_drawdown"] = float(g["fwd_max_drawdown"].median())
            if "fwd_max_return" in g.columns:
                rec["pct_clean_gt20_dd_lte_15"] = float(
                    ((g["fwd_max_return"] >= 0.20) & (g["fwd_max_drawdown"] <= 0.15)).mean()
                )
        if "fwd_close_return" in g.columns:
            rec["avg_fwd_close_return"] = float(g["fwd_close_return"].mean())
            rec["close_win_rate"] = float((g["fwd_close_return"] > 0).mean())
        return rec

    bucket_rows = []
    ranked = oof["score"].rank(method="first")
    oof_diag = oof.copy()
    oof_diag["decile"] = pd.qcut(ranked, 10, labels=False) + 1
    for decile, g in oof_diag.groupby("decile"):
        bucket_rows.append(_diagnostic_record(f"decile_{int(decile)}", g))
    for pct in (0.01, 0.02, 0.05, 0.10, 0.20):
        threshold = float(oof["score"].quantile(1.0 - pct))
        g = oof[oof["score"] >= threshold]
        bucket_rows.append(_diagnostic_record(f"top_{int(pct * 100)}pct", g))
    if isinstance(oof.index, pd.MultiIndex):
        ts_level = "timestamp" if "timestamp" in oof.index.names else oof.index.names[0]
        top5_idx = (
            oof.reset_index()
            .sort_values([ts_level, "score"], ascending=[True, False])
            .groupby(ts_level)
            .head(5)
            .set_index(oof.index.names)
            .index
        )
        bucket_rows.append(_diagnostic_record("top5_per_4h_bar", oof.loc[oof.index.isin(top5_idx)]))
    bucket_metrics = pd.DataFrame(bucket_rows)
    bucket_metrics.to_csv(WORK / "oof_score_buckets.csv", index=False)
    print(bucket_metrics.to_string(index=False))

# %%
# Cell 5 — final fit on all data, save booster
final_train_mask, final_val_mask = _time_train_val_split(df.index)
round_selector = _fit_xgb(
    df.loc[final_train_mask, FEATURES],
    df.loc[final_train_mask, TARGET].astype(float),
    df.loc[final_val_mask, FEATURES],
    df.loc[final_val_mask, TARGET].astype(float),
    BEST_PARAMS,
)
best_n_estimators = (
    int(round_selector.best_iteration) + 1
    if getattr(round_selector, "best_iteration", None) is not None
    else int(BEST_PARAMS["n_estimators"])
)
print("final best_n_estimators:", best_n_estimators)

final_params = {**BEST_PARAMS, "n_estimators": best_n_estimators}
final_model = xgb.XGBRegressor(
    **{k: v for k, v in BASE_XGB_PARAMS.items() if k != "early_stopping_rounds"},
    **final_params,
)
final_model.fit(df[FEATURES], df[TARGET].astype(float), verbose=False)
booster_path = WORK / "expansion_xgb.json"
final_model.get_booster().save_model(str(booster_path))
print("saved booster ->", booster_path)

importance_rows = []
booster = final_model.get_booster()
for importance_type in ("gain", "weight", "cover"):
    scores = booster.get_score(importance_type=importance_type)
    for feature, value in scores.items():
        importance_rows.append({
            "importance_type": importance_type,
            "feature": feature,
            "value": float(value),
        })
importance = pd.DataFrame(importance_rows)
if not importance.empty:
    importance.to_csv(WORK / "feature_importance.csv", index=False)
    for importance_type in ("gain", "weight", "cover"):
        top = (
            importance[importance["importance_type"] == importance_type]
            .sort_values("value", ascending=False)
            .head(20)
        )
        print(f"top feature importance ({importance_type})")
        print(top[["feature", "value"]].to_string(index=False))

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
    "target_column": TARGET,
    "target_kind": TARGET_KIND,
    "feature_columns": FEATURES,
}
(WORK / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
(WORK / "feature_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

# %%
# Cell 7 — bundle outputs for download
import shutil
out_bundle = Path("momentum_model_bundle.tgz")
with tarfile.open(out_bundle, "w:gz") as tar:
    for name in (
        "expansion_xgb.json",
        "feature_manifest.json",
        "eval_metrics.json",
        "xgb_param_search.csv",
        "oof_score_buckets.csv",
        "feature_importance.csv",
    ):
        p = WORK / name
        if p.exists():
            tar.add(p, arcname=name)
    if (WORK / "oof_preds.parquet").exists():
        tar.add(WORK / "oof_preds.parquet", arcname="oof_preds.parquet")
print("download:", out_bundle)
