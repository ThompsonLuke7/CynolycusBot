# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # multi_ticker_swing — Colab trainer (30m multiclass swing model)
#
# Upload `swing_colab_bundle.tgz`, then run this file/cells top to bottom.
#
# Output: `swing_model_bundle.tgz`, containing the model and probability artifacts
# expected by the local sweep scripts.

# %%
# !pip install -q "xgboost==2.*" pandas pyarrow scikit-learn

import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# %%
BUNDLE = Path("swing_colab_bundle.tgz")
WORK = Path("swing_work")
WORK.mkdir(exist_ok=True)
with tarfile.open(BUNDLE, "r:gz") as tar:
    try:
        tar.extractall(WORK, filter="data")
    except TypeError:
        tar.extractall(WORK)

manifest = json.loads((WORK / "feature_manifest.json").read_text())
FEATURES = manifest["feature_columns"]
TARGET = manifest["target_column"]
TRAIN_FRAC = float(manifest["train_frac"])
VAL_FRAC = float(manifest["val_frac"])
NEUTRAL_WEIGHT_FACTOR = float(manifest["neutral_weight_factor"])
XGB_CONFIG = dict(manifest["xgboost_config"])

df = pd.read_parquet(WORK / "training_matrix_30m.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)
print("matrix", df.shape, "tickers", df["ticker"].nunique() if "ticker" in df.columns else None)
print("features", len(FEATURES), "range", df["timestamp"].min(), "->", df["timestamp"].max())

# %%
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, log_loss

device = os.environ.get("XGB_DEVICE") or ("cuda" if shutil.which("nvidia-smi") else "cpu")
print("xgb device:", device)
XGB_CONFIG["tree_method"] = "hist"
XGB_CONFIG["device"] = device

early = XGB_CONFIG.pop("early_stopping_rounds", 60)


def split_time(frame):
    n = len(frame)
    t1 = int(n * TRAIN_FRAC)
    t2 = int(n * (TRAIN_FRAC + VAL_FRAC))
    return frame.iloc[:t1].copy(), frame.iloc[t1:t2].copy(), frame.iloc[t2:].copy()


def weights(frame):
    y = frame[TARGET].to_numpy(int)
    soft = frame.get("sample_weight", pd.Series(1.0, index=frame.index)).to_numpy(float)
    return (soft * np.where(y == 1, NEUTRAL_WEIGHT_FACTOR, 1.0)).astype(np.float32)


def metrics(y_true, proba, split):
    pred = np.argmax(proba, axis=1)
    rep = classification_report(
        y_true, pred, labels=[0, 1, 2], target_names=["short", "neutral", "long"],
        output_dict=True, zero_division=0,
    )
    out = {
        f"{split}_accuracy": float(accuracy_score(y_true, pred)),
        f"{split}_log_loss": float(log_loss(y_true, proba, labels=[0, 1, 2])),
    }
    for name in ["short", "neutral", "long"]:
        out[f"{split}_{name}_precision"] = float(rep[name]["precision"])
        out[f"{split}_{name}_recall"] = float(rep[name]["recall"])
        out[f"{split}_{name}_f1"] = float(rep[name]["f1-score"])
    return out


train_df, val_df, test_df = split_time(df)
X_train = train_df[FEATURES].to_numpy(np.float32)
y_train = train_df[TARGET].to_numpy(int)
X_val = val_df[FEATURES].to_numpy(np.float32)
y_val = val_df[TARGET].to_numpy(int)
X_test = test_df[FEATURES].to_numpy(np.float32)
y_test = test_df[TARGET].to_numpy(int)

print("splits", len(train_df), len(val_df), len(test_df))
print("class counts train", np.bincount(y_train, minlength=3))

model = xgb.XGBClassifier(**XGB_CONFIG, early_stopping_rounds=early, verbosity=1)
model.fit(X_train, y_train, sample_weight=weights(train_df), eval_set=[(X_val, y_val)], verbose=50)

# %%
OUT = WORK / "artifacts"
OUT.mkdir(exist_ok=True)
model.save_model(OUT / "swing_xgb_model.json")

val_proba = model.predict_proba(X_val)
test_proba = model.predict_proba(X_test)
all_proba = model.predict_proba(df[FEATURES].to_numpy(np.float32))

eval_metrics = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "features_used": len(FEATURES),
    "n_rows": int(len(df)),
    "n_tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else None,
    "best_iteration": int(getattr(model, "best_iteration", -1)) + 1,
    "xgb_device": device,
    **metrics(y_val, val_proba, "val"),
    **metrics(y_test, test_proba, "test"),
}
(OUT / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2))
(OUT / "selected_features.txt").write_text("\n".join(FEATURES))

fi = (
    pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
fi.to_csv(OUT / "feature_importance.csv", index=False)

proba = df[["timestamp", "ticker"]].copy()
proba["p_short"] = all_proba[:, 0]
proba["p_neutral"] = all_proba[:, 1]
proba["p_long"] = all_proba[:, 2]
proba["split"] = "train"
proba.loc[val_df.index, "split"] = "val"
proba.loc[test_df.index, "split"] = "test"
proba = proba[["timestamp", "ticker", "split", "p_long", "p_short", "p_neutral"]]
proba.to_parquet(OUT / "p_swing_probs.parquet", index=False)

meta = {
    "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_swing_30m_colab"),
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "dataset_name": "30m_multi_ticker_shared_universe",
    "classes": ["short", "neutral", "long"],
    "feature_names": FEATURES,
    "manifest": manifest,
    "artifact_paths": {p.name: p.name for p in OUT.iterdir()},
}
(OUT / "meta.json").write_text(json.dumps(meta, default=str, indent=2))

bundle_out = Path("swing_model_bundle.tgz")
with tarfile.open(bundle_out, "w:gz") as tar:
    for path in OUT.iterdir():
        tar.add(path, arcname=path.name)
print("download:", bundle_out)
