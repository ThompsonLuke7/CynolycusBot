# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # multi_ticker_swing — Colab model competition trainer
#
# Upload `swing_context_colab_bundle.tgz`, then run this file/cells top to bottom.
#
# Trains 5+ seeds across:
# - XGBoost classifier
# - XGBoost ranker
# - LightGBM classifier
# - LightGBM ranker
#
# Outputs averaged metrics, top-feature stability, and top-pick overlap.

# %%
# !pip install -q "xgboost==2.*" lightgbm joblib pandas pyarrow scikit-learn

import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
BUNDLE = Path(os.environ.get("SWING_BUNDLE", "swing_context_colab_bundle.tgz"))
if not BUNDLE.exists() and Path("swing_colab_bundle.tgz").exists():
    BUNDLE = Path("swing_colab_bundle.tgz")
WORK = Path("swing_work")
manifest = load_bundle(BUNDLE, WORK, "feature_manifest.json")

FEATURES = manifest["feature_columns"]
TARGET = os.environ.get("SWING_TARGET", manifest["target_column"])
MATRIX_FILE = manifest.get("matrix_file", "training_matrix_30m.parquet")
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", manifest["train_frac"]))
VAL_FRAC = float(os.environ.get("VAL_FRAC", manifest["val_frac"]))
NEUTRAL_WEIGHT_FACTOR = float(os.environ.get("NEUTRAL_WEIGHT_FACTOR", manifest["neutral_weight_factor"]))
SEEDS = parse_seeds(os.environ.get("MODEL_SEEDS"), int(os.environ.get("N_SEEDS", "5")))
FAMILIES = parse_families(os.environ.get("MODEL_FAMILIES"))
TOP_K = int(os.environ.get("TOP_K", "10"))
RANK_GROUP = os.environ.get("RANK_GROUP", "timestamp")
POSITIVE_LABEL = os.environ.get("POSITIVE_LABEL")
POSITIVE_LABEL = int(POSITIVE_LABEL) if POSITIVE_LABEL not in {None, ""} else None

df = pd.read_parquet(WORK / MATRIX_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["ticker"] = df["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)

bar_context_file = manifest.get("bar_context_file")
if bar_context_file:
    bar_context = pd.read_parquet(WORK / bar_context_file)
    bar_context["timestamp"] = pd.to_datetime(bar_context["timestamp"], utc=True)
    bar_context["ticker"] = bar_context["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    df = df.merge(bar_context, on=["timestamp", "ticker"], how="left")

context_file = manifest.get("context_file")
if context_file:
    context = pd.read_parquet(WORK / context_file)
    context["date"] = pd.to_datetime(context["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    context["ticker"] = context["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    df["_context_date"] = df["timestamp"].dt.tz_convert(None).dt.normalize()
    df = df.merge(
        context.rename(columns={"date": "_context_date"}),
        on=["ticker", "_context_date"],
        how="left",
    ).drop(columns=["_context_date"])

df = df.sort_values("timestamp").reset_index(drop=True)
print("matrix", df.shape, "tickers", df["ticker"].nunique())
print("target", TARGET, "features", len(FEATURES), "range", df["timestamp"].min(), "->", df["timestamp"].max())
print("families", FAMILIES, "seeds", SEEDS)

# %%
device = os.environ.get("XGB_DEVICE") or ("cuda" if shutil.which("nvidia-smi") else "cpu")
print("xgb device:", device)

cfg = CompetitionConfig(
    task_name="multi_ticker_swing_30m",
    target_column=TARGET,
    feature_columns=FEATURES,
    train_frac=TRAIN_FRAC,
    val_frac=VAL_FRAC,
    seeds=SEEDS,
    families=FAMILIES,
    output_dir=WORK / "artifacts",
    sample_weight_column=manifest.get("sample_weight_column", "sample_weight"),
    neutral_weight_factor=NEUTRAL_WEIGHT_FACTOR,
    positive_label=POSITIVE_LABEL,
    rank_group=RANK_GROUP,
    top_k=TOP_K,
    xgb_config=dict(manifest.get("xgboost_config", {})),
    device=device,
    timestamp_column="timestamp",
    id_columns=("timestamp", "ticker"),
)
result = run_competition(df, cfg)

print("best")
print(result["best"])
print("family summary")
print(result["summary"])
print("top-pick overlap")
print(result["pick_overlap"])

# %%
# Backward-compatible swing artifacts for local scripts that expect the old XGB-classifier bundle shape.
OUT = cfg.output_dir
xgb_rows = result["results"][result["results"]["family"] == "xgb_classifier"].copy()
if not xgb_rows.empty:
    best_xgb = xgb_rows.sort_values(["val_log_loss", "test_positive_f1"], ascending=[True, False], na_position="last").iloc[0]
    import joblib

    model = joblib.load(OUT / f"model_xgb_classifier_seed{int(best_xgb['seed'])}.joblib")
    if hasattr(model, "save_model"):
        model.save_model(OUT / "swing_xgb_model.json")
    fi_path = OUT / f"feature_importance_xgb_classifier_seed{int(best_xgb['seed'])}.csv"
    if fi_path.exists():
        shutil.copy2(fi_path, OUT / "feature_importance.csv")
    proba_chunks = []
    chunk_size = int(os.environ.get("PROBA_CHUNK_SIZE", "1000000"))
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start : start + chunk_size]
        p = model.predict_proba(chunk[FEATURES].to_numpy(np.float32))
        out = chunk[["timestamp", "ticker"]].copy()
        out["p_short"] = p[:, 0] if p.shape[1] > 0 else np.nan
        out["p_neutral"] = p[:, 1] if p.shape[1] > 1 else np.nan
        out["p_long"] = p[:, 2] if p.shape[1] > 2 else np.nan
        out["split"] = "train"
        out.loc[result["val_df"].index.intersection(out.index), "split"] = "val"
        out.loc[result["test_df"].index.intersection(out.index), "split"] = "test"
        proba_chunks.append(out[["timestamp", "ticker", "split", "p_long", "p_short", "p_neutral"]])
    pd.concat(proba_chunks, ignore_index=True).to_parquet(OUT / "p_swing_probs.parquet", index=False)
    if str(result["best"]["family"]) != "xgb_classifier":
        print("overall best was not xgb_classifier; compatibility model is documented in seed_results.csv")

selected = "\n".join(FEATURES)
(OUT / "selected_features.txt").write_text(selected)
(OUT / "eval_metrics.json").write_text(json.dumps(result["best"].to_dict(), indent=2, default=str))

meta = {
    "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_swing_30m_colab_competition"),
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "dataset_name": "30m_multi_ticker_shared_universe",
    "classes": ["short", "neutral", "long"],
    "feature_names": FEATURES,
    "manifest": manifest,
    "competition_artifacts": sorted(p.name for p in OUT.iterdir() if p.is_file()),
}
(OUT / "meta.json").write_text(json.dumps(meta, default=str, indent=2))

bundle_out = Path("swing_model_competition_bundle.tgz")
write_artifact_bundle(OUT, bundle_out)
print("download:", bundle_out)
