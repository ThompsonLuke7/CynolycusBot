# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # SPY Day Trader — Colab model competition trainer
#
# Upload `spy_daytrader_context_colab_bundle.tgz`, then run this file/cells top to bottom.
#
# Default target is `tb_label`. Override with:
# `SPY_TARGET=spyopt_target_up`, `SPY_TARGET=spyopt_target_big_move_075pct`, or another discrete label.

# %%
# !pip install -q "xgboost==2.*" lightgbm joblib pandas pyarrow scikit-learn

import json
import os
import shutil
from pathlib import Path

import pandas as pd

from colab_competition import (
    CompetitionConfig,
    load_bundle,
    parse_families,
    parse_seeds,
    run_competition,
    write_artifact_bundle,
)

# %%
BUNDLE = Path(os.environ.get("SPY_BUNDLE", "spy_daytrader_context_colab_bundle.tgz"))
WORK = Path("spy_daytrader_work")
manifest = load_bundle(BUNDLE, WORK, "spy_daytrader_context_manifest.json")

FEATURES = manifest["feature_columns"]
TARGET_RAW = os.environ.get("SPY_TARGET", "tb_label")
MATRIX_FILE = manifest.get("matrix_file", "spy_daytrader_context_matrix.parquet")
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", "0.70"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.15"))
SEEDS = parse_seeds(os.environ.get("MODEL_SEEDS"), int(os.environ.get("N_SEEDS", "5")))
FAMILIES = parse_families(os.environ.get("MODEL_FAMILIES"))
TOP_K = int(os.environ.get("TOP_K", "5"))
RANK_GROUP = os.environ.get("RANK_GROUP", "date")
POSITIVE_LABEL = os.environ.get("POSITIVE_LABEL")
POSITIVE_LABEL = int(POSITIVE_LABEL) if POSITIVE_LABEL not in {None, ""} else None

df = pd.read_parquet(WORK / MATRIX_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
if TARGET_RAW not in df.columns:
    raise ValueError(f"Target {TARGET_RAW!r} is not in the SPY matrix. Available labels include: {manifest.get('label_columns', [])[:30]}")

target_series = pd.to_numeric(df[TARGET_RAW], errors="coerce")
unique_labels = sorted(target_series.dropna().unique().tolist())
if len(unique_labels) > 20:
    raise ValueError(
        f"{TARGET_RAW} has {len(unique_labels)} unique values. Pick a discrete classifier/ranker label, "
        "for example tb_label, atr_leg_label, spyopt_target_up, spyopt_target_big_move_075pct, "
        "or spyopt_target_big_move_100pct."
    )
label_to_encoded = {label: idx for idx, label in enumerate(unique_labels)}
encoded_to_label = {idx: label for label, idx in label_to_encoded.items()}
df["_competition_target"] = target_series.map(label_to_encoded)

print("matrix", df.shape)
print("raw target", TARGET_RAW, "labels", unique_labels, "encoded", encoded_to_label)
print("features", len(FEATURES), "range", df["timestamp"].min(), "->", df["timestamp"].max())
print("families", FAMILIES, "seeds", SEEDS)

# %%
device = os.environ.get("XGB_DEVICE") or ("cuda" if shutil.which("nvidia-smi") else "cpu")
print("xgb device:", device)

cfg = CompetitionConfig(
    task_name=f"spy_daytrader_{TARGET_RAW}",
    target_column="_competition_target",
    feature_columns=FEATURES,
    train_frac=TRAIN_FRAC,
    val_frac=VAL_FRAC,
    seeds=SEEDS,
    families=FAMILIES,
    output_dir=WORK / "artifacts",
    sample_weight_column=None,
    neutral_weight_factor=1.0,
    positive_label=POSITIVE_LABEL,
    rank_group=RANK_GROUP,
    top_k=TOP_K,
    xgb_config={"n_estimators": 900, "learning_rate": 0.035, "max_depth": 5, "early_stopping_rounds": 70},
    device=device,
    timestamp_column="timestamp",
    id_columns=("timestamp",),
)
result = run_competition(df, cfg)

print("best")
print(result["best"])
print("family summary")
print(result["summary"])
print("top-pick overlap")
print(result["pick_overlap"])

# %%
OUT = cfg.output_dir
meta_path = OUT / "competition_meta.json"
meta = json.loads(meta_path.read_text())
meta["raw_target_column"] = TARGET_RAW
meta["encoded_label_map"] = {str(k): v for k, v in encoded_to_label.items()}
meta["available_label_columns"] = manifest.get("label_columns", [])
meta_path.write_text(json.dumps(meta, indent=2, default=str))

bundle_out = Path(f"spy_daytrader_{TARGET_RAW}_model_competition_bundle.tgz")
write_artifact_bundle(OUT, bundle_out)
print("download:", bundle_out)
