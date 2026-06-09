# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # Meta Ranker — Colab training notebook
#
# Upload `meta_ranker_colab_bundle.tgz`, then run cells top to bottom.
#
# Inputs (inside the bundle):
#   - meta_ranker_matrix.parquet   (features + meta_label, leakage-controlled)
#   - manifest.json
#
# Outputs (download via meta_ranker_model_bundle.tgz):
#   - meta_ranker_xgb.json
#   - eval_metrics.json
#   - oof_preds.parquet            (out-of-fold meta scores for backtesting)
#   - oof_score_buckets.csv, feature_importance.csv, xgb_param_search.csv

# %%
# !pip install -q "xgboost==2.*" pandas pyarrow scikit-learn

import json
import os
import shutil
import tarfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# %%
BUNDLE = Path("meta_ranker_colab_bundle.tgz")
WORK = Path("meta_work")
WORK.mkdir(exist_ok=True)
with tarfile.open(BUNDLE, "r:gz") as tar:
    try:
        tar.extractall(WORK, filter="data")
    except TypeError:
        tar.extractall(WORK)

manifest = json.loads((WORK / "manifest.json").read_text())
FEATURES = manifest["feature_columns"]
TARGET = manifest["label_column"]
CATS = [c for c in manifest.get("categorical_columns", []) if c in FEATURES]
print("features:", len(FEATURES), "| target:", TARGET, "| categoricals:", CATS)

df = pd.read_parquet(WORK / "meta_ranker_matrix.parquet")
df = df.reset_index()                      # (timestamp, ticker) -> columns
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.dropna(subset=[TARGET]).sort_values("timestamp")
for c in CATS:
    df[c] = df[c].astype("category")
print(df.shape, "| range", df["timestamp"].min(), "->", df["timestamp"].max())

# Diagnostic columns kept in the OOF for backtest-style evaluation.
DIAG = [c for c in ("fwd_close_return", "fwd_max_drawdown", "fwd_atr_adj_return") if c in df.columns]

# %%
# Walk-forward folds: 18m train, 21d embargo (> label horizon), 4m non-overlap
# test. The meta matrix only spans ~3.5y (base OOF range), so a shorter window
# is used to get enough folds for a stable read.
WF = {"train_months": 18, "embargo_days": 21, "test_months": 4, "min_train_rows": 50000}
ts = df["timestamp"]
date_min, date_max = ts.min(), ts.max()
folds, test_end = [], date_max
while True:
    test_start = test_end - pd.DateOffset(months=WF["test_months"])
    train_end = test_start - timedelta(days=WF["embargo_days"])
    train_start = train_end - pd.DateOffset(months=WF["train_months"])
    if train_start < date_min:
        break
    folds.append(dict(train_start=train_start, train_end=train_end,
                      test_start=test_start, test_end=test_end))
    test_end = test_start
folds = list(reversed(folds))
print("folds:", len(folds))

# %%
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterSampler

XGB_DEVICE = os.environ.get("XGB_DEVICE") or ("cuda" if shutil.which("nvidia-smi") else "cpu")
print("device:", XGB_DEVICE)

BASE_PARAMS = dict(objective="reg:squarederror", eval_metric="rmse", random_state=42,
                   n_jobs=-1, tree_method="hist", device=XGB_DEVICE,
                   enable_categorical=True, early_stopping_rounds=75)
PARAM_SPACE = {
    "n_estimators": [600, 1000, 1600, 2400],
    "learning_rate": [0.015, 0.025, 0.04, 0.06],
    "max_depth": [3, 4, 5, 6],
    "min_child_weight": [20, 40, 80, 160],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.75, 0.9],
    "reg_alpha": [0.0, 0.1, 0.5],
    "reg_lambda": [1.0, 2.0, 4.0],
}
N_TRIALS = int(os.environ.get("META_XGB_PARAM_TRIALS", "25"))
VAL_FRACTION = 0.20
INT_KEYS = {"n_estimators", "max_depth", "min_child_weight"}
MIN_FINAL_ESTIMATORS = 50   # never ship a degenerate shallow model (see HTF lesson)


def time_val_split(frame, fraction=VAL_FRACTION):
    days = np.sort(frame["timestamp"].dt.normalize().unique())
    split = min(max(1, int(len(days) * (1 - fraction))), len(days) - 1)
    cut = pd.Timestamp(days[split])
    return frame["timestamp"] < cut, frame["timestamp"] >= cut


def fit_xgb(Xtr, ytr, Xval, yval, params):
    m = xgb.XGBRegressor(**BASE_PARAMS, **params)
    m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return m


def predict(m, X):
    bi = getattr(m, "best_iteration", None)
    return m.predict(X) if bi is None else m.predict(X, iteration_range=(0, int(bi) + 1))


def coerce(p):
    return {k: (int(v) if k in INT_KEYS else float(v)) for k, v in p.items()}


# %%
# Hyperparameter search by mean validation Spearman across fold inner-splits.
cands = list(ParameterSampler(PARAM_SPACE, n_iter=N_TRIALS, random_state=42))
rows = []
for pi, params in enumerate(cands):
    scs = []
    for f in folds:
        tr = df[(ts >= f["train_start"]) & (ts <= f["train_end"])]
        if len(tr) < WF["min_train_rows"]:
            continue
        itr, ival = time_val_split(tr)
        Xtr, ytr = tr.loc[itr, FEATURES], tr.loc[itr, TARGET]
        Xval, yval = tr.loc[ival, FEATURES], tr.loc[ival, TARGET]
        if len(Xtr) < WF["min_train_rows"] // 2 or len(Xval) < 2000:
            continue
        m = fit_xgb(Xtr, ytr, Xval, yval, coerce(params))
        scs.append(pd.Series(predict(m, Xval), index=yval.index).corr(yval, method="spearman"))
    rows.append({"trial": pi, "mean_val_spearman": float(np.mean(scs)) if scs else float("nan"), **params})
    print(rows[-1])
tuning = pd.DataFrame(rows).sort_values("mean_val_spearman", ascending=False)
tuning.to_csv(WORK / "xgb_param_search.csv", index=False)
BEST = coerce({k: tuning.iloc[0][k] for k in PARAM_SPACE})
print("best:", BEST)

# %%
# Walk-forward OOF predictions.
oof_rows, fold_metrics = [], []
for fi, f in enumerate(folds):
    tr = df[(ts >= f["train_start"]) & (ts <= f["train_end"])]
    te = df[(ts >= f["test_start"]) & (ts <= f["test_end"])]
    if len(tr) < WF["min_train_rows"] or len(te) < 2000:
        continue
    itr, ival = time_val_split(tr)
    m = fit_xgb(tr.loc[itr, FEATURES], tr.loc[itr, TARGET],
                tr.loc[ival, FEATURES], tr.loc[ival, TARGET], BEST)
    p = predict(m, te[FEATURES])
    fold_metrics.append({"fold": fi, "n_train": int(itr.sum()), "n_test": int(len(te)),
                         "best_iteration": int(getattr(m, "best_iteration", 0) or 0),
                         "rmse": float(np.sqrt(mean_squared_error(te[TARGET], p))),
                         "mae": float(mean_absolute_error(te[TARGET], p)),
                         "spearman": float(pd.Series(p, index=te.index).corr(te[TARGET], method="spearman"))})
    print(fold_metrics[-1])
    block = pd.DataFrame({"score": p, "y": te[TARGET].values}, index=te.index)
    block["timestamp"] = te["timestamp"].values
    block["ticker"] = te["ticker"].values
    for c in DIAG:
        block[c] = te[c].values
    oof_rows.append(block)

oof = pd.concat(oof_rows)
oof.set_index(["timestamp", "ticker"]).to_parquet(WORK / "oof_preds.parquet")
print("OOF Spearman:", oof["score"].corr(oof["y"], method="spearman"))

ranked = oof["score"].rank(method="first")
oof["decile"] = pd.qcut(ranked, 10, labels=False) + 1
buckets = oof.groupby("decile").agg(n=("score", "size"), score_mean=("score", "mean"),
                                    y_mean=("y", "mean"),
                                    **({"fwd_close_mean": ("fwd_close_return", "mean")} if "fwd_close_return" in oof else {}))
buckets.to_csv(WORK / "oof_score_buckets.csv")
print(buckets)

# %%
# Final fit on all rows with a robust (non-degenerate) tree count.
final_tr, final_val = time_val_split(df)
selector = fit_xgb(df.loc[final_tr, FEATURES], df.loc[final_tr, TARGET],
                   df.loc[final_val, FEATURES], df.loc[final_val, TARGET], BEST)
selector_n = int(getattr(selector, "best_iteration", 0) or 0) + 1
fold_best = [m["best_iteration"] + 1 for m in fold_metrics if m.get("best_iteration") is not None]
fold_floor = int(np.median(fold_best)) if fold_best else selector_n
best_n = max(selector_n, fold_floor, MIN_FINAL_ESTIMATORS)
print(f"final n_estimators: selector={selector_n} fold_floor={fold_floor} -> {best_n}")

final = xgb.XGBRegressor(**{k: v for k, v in BASE_PARAMS.items() if k != "early_stopping_rounds"},
                         **{**BEST, "n_estimators": best_n})
final.fit(df[FEATURES], df[TARGET], verbose=False)
final.get_booster().save_model(str(WORK / "meta_ranker_xgb.json"))

imp = []
for kind in ("gain", "weight", "cover"):
    for feat, val in final.get_booster().get_score(importance_type=kind).items():
        imp.append({"importance_type": kind, "feature": feat, "value": float(val)})
pd.DataFrame(imp).to_csv(WORK / "feature_importance.csv", index=False)

(WORK / "eval_metrics.json").write_text(json.dumps({
    "fold_metrics": fold_metrics, "n_folds": len(fold_metrics), "best_params": BEST,
    "best_n_estimators": best_n, "target_column": TARGET, "feature_columns": FEATURES,
    "categorical_columns": CATS, "label_definition": manifest.get("label_definition"),
}, indent=2, default=str))
(WORK / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

# %%
out = Path("meta_ranker_model_bundle.tgz")
with tarfile.open(out, "w:gz") as tar:
    for name in ("meta_ranker_xgb.json", "manifest.json", "eval_metrics.json",
                 "xgb_param_search.csv", "oof_score_buckets.csv", "feature_importance.csv",
                 "oof_preds.parquet"):
        p = WORK / name
        if p.exists():
            tar.add(p, arcname=name)
print("download:", out)
