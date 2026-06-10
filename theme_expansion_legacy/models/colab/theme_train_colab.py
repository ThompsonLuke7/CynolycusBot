# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # theme_expansion - Colab training notebook
#
# Upload `theme_colab_bundle.tgz`, then run the cells top to bottom.
#
# Outputs:
# - `theme_xgb.json`
# - `eval_metrics.json`
# - `oof_preds.parquet`
# - `oof_score_buckets.csv`
# - `feature_importance.csv`
# - `theme_model_bundle.tgz`

# %%
# !pip install -q xgboost==2.* pandas pyarrow scikit-learn

import json
import os
import tarfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# %%
BUNDLE = Path("theme_colab_bundle.tgz")
WORK = Path("theme_work")
WORK.mkdir(exist_ok=True)
with tarfile.open(BUNDLE, "r:gz") as tar:
    try:
        tar.extractall(WORK, filter="data")
    except TypeError:
        tar.extractall(WORK)

manifest = json.loads((WORK / "feature_manifest.json").read_text())
TARGET = manifest["target_column"]
NUMERIC_FEATURES = manifest["feature_columns"]
CATEGORICAL_FEATURES = manifest.get("categorical_columns", [])
WF = manifest["walk_forward"]

df = pd.read_parquet(WORK / "theme_training_matrix.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "theme"]).reset_index(drop=True)
print(df.shape)
print("target:", TARGET)
print("numeric features:", len(NUMERIC_FEATURES))
print("categorical features:", CATEGORICAL_FEATURES)
print("date range:", df["date"].min(), "->", df["date"].max())

# %%
def build_xy(frame, category_levels=None):
    X_num = frame[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan).astype(float)
    X_cat = pd.get_dummies(frame[CATEGORICAL_FEATURES].fillna("unknown").astype(str), dummy_na=False)
    if category_levels is not None:
        X_cat = X_cat.reindex(columns=category_levels, fill_value=0)
    X = pd.concat([X_num, X_cat], axis=1).fillna(0.0)
    y = frame[TARGET].astype(float)
    return X, y, list(X_cat.columns)


def walk_forward_folds(frame):
    date_min = frame["date"].min()
    test_end = frame["date"].max()
    folds = []
    while True:
        test_start = test_end - pd.DateOffset(months=int(WF["test_months"]))
        train_end = test_start - timedelta(days=int(WF["embargo_days"]))
        train_start = train_end - pd.DateOffset(years=int(WF["train_years"]))
        if train_start < date_min:
            break
        folds.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        test_end = test_start
    return list(reversed(folds))


folds = walk_forward_folds(df)
print("folds:", len(folds))
for fold in folds:
    print(fold)

# %%
import shutil

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterSampler

# Use the Colab GPU when present (nvidia-smi on PATH), else CPU. Override with XGB_DEVICE.
XGB_DEVICE = os.environ.get("XGB_DEVICE") or ("cuda" if shutil.which("nvidia-smi") else "cpu")
print("xgb device:", XGB_DEVICE)

BASE_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "device": XGB_DEVICE,
    "early_stopping_rounds": 75,
}
PARAM_SPACE = {
    "n_estimators": [800, 1200, 1800, 2400],
    "learning_rate": [0.015, 0.025, 0.04, 0.06],
    "max_depth": [2, 3, 4, 5],
    "min_child_weight": [10, 25, 50, 100],
    "subsample": [0.70, 0.85, 1.0],
    "colsample_bytree": [0.60, 0.80, 1.0],
    "reg_alpha": [0.0, 0.05, 0.20],
    "reg_lambda": [1.0, 2.0, 5.0],
}
N_TRIALS = int(os.environ.get("THEME_XGB_PARAM_TRIALS", "30"))


def time_train_val_split(frame, val_fraction=0.20):
    dates = pd.Series(frame["date"].dt.normalize().unique()).sort_values().to_numpy()
    split_at = max(1, int(len(dates) * (1.0 - val_fraction)))
    if split_at >= len(dates):
        split_at = len(dates) - 1
    val_start = pd.Timestamp(dates[split_at])
    return frame["date"] < val_start, frame["date"] >= val_start


def fit_model(train_frame, params):
    train_mask, val_mask = time_train_val_split(train_frame)
    train_part = train_frame.loc[train_mask]
    val_part = train_frame.loc[val_mask]
    Xtr, ytr, cat_levels = build_xy(train_part)
    Xval, yval, _ = build_xy(val_part, cat_levels)
    model = xgb.XGBRegressor(**BASE_PARAMS, **params)
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return model, cat_levels


def predict(model, X):
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        return model.predict(X)
    return model.predict(X, iteration_range=(0, int(best_iteration) + 1))


candidate_params = list(ParameterSampler(PARAM_SPACE, n_iter=N_TRIALS, random_state=42))
tuning_rows = []
for i, params in enumerate(candidate_params):
    scores = []
    for fold in folds:
        train_frame = df[(df["date"] >= fold["train_start"]) & (df["date"] <= fold["train_end"])]
        if len(train_frame) < int(WF["min_train_rows"]):
            continue
        inner_train_mask, inner_val_mask = time_train_val_split(train_frame)
        inner_train = train_frame.loc[inner_train_mask]
        inner_val = train_frame.loc[inner_val_mask]
        Xtr, ytr, cat_levels = build_xy(inner_train)
        Xval, yval, _ = build_xy(inner_val, cat_levels)
        model = xgb.XGBRegressor(**BASE_PARAMS, **params)
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        p = predict(model, Xval)
        scores.append(pd.Series(p).corr(pd.Series(yval.values), method="spearman"))
    row = {"trial": i, "mean_val_spearman": float(np.nanmean(scores)), "n_folds": len(scores), **params}
    tuning_rows.append(row)
    print(row)

tuning = pd.DataFrame(tuning_rows).sort_values("mean_val_spearman", ascending=False)
tuning.to_csv(WORK / "xgb_param_search.csv", index=False)
BEST_PARAMS = {k: tuning.iloc[0][k].item() if hasattr(tuning.iloc[0][k], "item") else tuning.iloc[0][k] for k in PARAM_SPACE}
BEST_PARAMS["n_estimators"] = int(BEST_PARAMS["n_estimators"])
BEST_PARAMS["max_depth"] = int(BEST_PARAMS["max_depth"])
BEST_PARAMS["min_child_weight"] = int(BEST_PARAMS["min_child_weight"])
print("best:", BEST_PARAMS)

# %%
oof_rows = []
fold_metrics = []
for i, fold in enumerate(folds):
    train_frame = df[(df["date"] >= fold["train_start"]) & (df["date"] <= fold["train_end"])]
    test_frame = df[(df["date"] >= fold["test_start"]) & (df["date"] <= fold["test_end"])]
    if len(train_frame) < int(WF["min_train_rows"]) or len(test_frame) < int(WF["min_test_rows"]):
        continue
    model, cat_levels = fit_model(train_frame, BEST_PARAMS)
    Xte, yte, _ = build_xy(test_frame, cat_levels)
    p = predict(model, Xte)
    metric = {
        "fold": i,
        "n_train": int(len(train_frame)),
        "n_test": int(len(test_frame)),
        "rmse": float(np.sqrt(mean_squared_error(yte, p))),
        "mae": float(mean_absolute_error(yte, p)),
        "spearman": float(pd.Series(p).corr(pd.Series(yte.values), method="spearman")),
    }
    fold_metrics.append(metric)
    print(metric)
    oof_rows.append(
        test_frame[["date", "theme", TARGET]].assign(score=p).rename(columns={TARGET: "y"})
    )

oof = pd.concat(oof_rows, ignore_index=True)
oof.to_parquet(WORK / "oof_preds.parquet", index=False)
print("OOF RMSE:", np.sqrt(mean_squared_error(oof["y"], oof["score"])))
print("OOF MAE:", mean_absolute_error(oof["y"], oof["score"]))
print("OOF Spearman:", oof["score"].corr(oof["y"], method="spearman"))

bucket_rows = []
ranked = oof["score"].rank(method="first")
oof["decile"] = pd.qcut(ranked, 10, labels=False) + 1
for decile, group in oof.groupby("decile"):
    bucket_rows.append(
        {
            "bucket": f"decile_{int(decile)}",
            "n": int(len(group)),
            "score_mean": float(group["score"].mean()),
            "target_mean": float(group["y"].mean()),
            "target_median": float(group["y"].median()),
        }
    )
for pct in (0.01, 0.02, 0.05, 0.10, 0.20):
    threshold = float(oof["score"].quantile(1.0 - pct))
    group = oof[oof["score"] >= threshold]
    bucket_rows.append(
        {
            "bucket": f"top_{int(pct * 100)}pct",
            "n": int(len(group)),
            "score_mean": float(group["score"].mean()),
            "target_mean": float(group["y"].mean()),
            "target_median": float(group["y"].median()),
        }
    )
bucket_metrics = pd.DataFrame(bucket_rows)
bucket_metrics.to_csv(WORK / "oof_score_buckets.csv", index=False)
print(bucket_metrics.to_string(index=False))

# %%
final_train_mask, final_val_mask = time_train_val_split(df)
final_train = df.loc[final_train_mask]
final_val = df.loc[final_val_mask]
Xtr, ytr, cat_levels = build_xy(final_train)
Xval, yval, _ = build_xy(final_val, cat_levels)
selector = xgb.XGBRegressor(**BASE_PARAMS, **BEST_PARAMS)
selector.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
best_n = int(selector.best_iteration) + 1 if getattr(selector, "best_iteration", None) is not None else int(BEST_PARAMS["n_estimators"])

Xall, yall, cat_levels = build_xy(df, cat_levels)
final_params = {**BEST_PARAMS, "n_estimators": best_n}
final_model = xgb.XGBRegressor(
    **{k: v for k, v in BASE_PARAMS.items() if k != "early_stopping_rounds"},
    **final_params,
)
final_model.fit(Xall, yall, verbose=False)
final_model.get_booster().save_model(str(WORK / "theme_xgb.json"))

importance_rows = []
for kind in ("gain", "weight", "cover"):
    for feature, value in final_model.get_booster().get_score(importance_type=kind).items():
        importance_rows.append({"importance_type": kind, "feature": feature, "value": float(value)})
importance = pd.DataFrame(importance_rows)
importance.to_csv(WORK / "feature_importance.csv", index=False)

metrics = {
    "fold_metrics": fold_metrics,
    "n_folds": len(fold_metrics),
    "base_params": BASE_PARAMS,
    "best_params": BEST_PARAMS,
    "best_n_estimators": best_n,
    "target_column": TARGET,
    "feature_columns": list(Xall.columns),
    "manifest": manifest,
}
(WORK / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
(WORK / "feature_manifest.json").write_text(json.dumps({**manifest, "expanded_feature_columns": list(Xall.columns)}, indent=2, default=str))

# %%
out_bundle = Path("theme_model_bundle.tgz")
with tarfile.open(out_bundle, "w:gz") as tar:
    for name in (
        "theme_xgb.json",
        "feature_manifest.json",
        "eval_metrics.json",
        "xgb_param_search.csv",
        "oof_score_buckets.csv",
        "feature_importance.csv",
        "oof_preds.parquet",
    ):
        p = WORK / name
        if p.exists():
            tar.add(p, arcname=name)
print("download:", out_bundle)
