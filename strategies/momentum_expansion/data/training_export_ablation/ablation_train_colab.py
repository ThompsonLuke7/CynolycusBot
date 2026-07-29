# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # momentum_expansion — WS-E ablation-arm Colab trainer
#
# Upload `ablation_colab_bundle.tgz`, then run top to bottom. Local execution is
# intentionally avoided per project rules (this is GPU/Colab work; plan defect D6 —
# see docs/superpowers/plans/2026-07-26-market-regime-and-sector-context.md WS-E).
#
# Trains the SAME model competition (xgboost/lightgbm x classifier/ranker, per
# `colab_competition.py`) independently for each of the five feature-block arms:
#   baseline / +risk / +liquidity / +sector / +all
# (`ablation_feature_manifest.json`'s "arms" — sourced from
# `strategies/momentum_expansion/ablation/feature_blocks.py`, not hand-typed here),
# on the SAME walk-forward fold spec (WALK_FORWARD_CONFIG) and the SAME rows, so the
# only thing that varies between arms is which columns the model can see. This is
# the actual ablation the local screen (run_screen.py) could not perform (D6: no
# local training) — it only screened whether the new columns show a *training-free*
# signal (rank IC / regime-conditional expectancy); this notebook is what actually
# tells you whether a trained model benefits from them.
#
# IMPORTANT (plan D3): the bundled matrix (`training_matrix_4h_with_regime.parquet`)
# is NOT the production training_matrix_4h.parquet — it is a separate export that
# adds the regime/sector columns on top. The baseline arm's rows are therefore
# whatever the production dropna already produced (unaffected by the new columns);
# +risk/+liquidity/+sector/+all arms will additionally drop any row where their own
# added columns are NaN (pre-2021-07 z-score warmup, or an unresolved sector) —
# EXPECT the non-baseline arms to train and evaluate on somewhat fewer, more recent
# rows than baseline. Report row counts per arm; do not compare arms on a metric
# without also reporting how many rows each arm actually trained/evaluated on.
#
# Inputs (inside the bundle):
#   - training_matrix_4h_with_regime.parquet
#   - ablation_feature_manifest.json   (has "arms": {name: [feature_columns]})
#   - colab_competition.py             (shared harness, same one momentum's own
#                                        trainer uses)
#
# Outputs (download via ablation_model_bundle.tgz):
#   - ablation_arm_comparison.csv      (per-arm: n_rows, best family/seed, primary
#                                        metric, walk-forward OOF Spearman)
#   - oof_preds_<arm>.parquet          (per-arm walk-forward OOF, for a post-hoc
#                                        rank-IC/NDCG/turnover/Sharpe read using
#                                        strategies/momentum_expansion/ablation/metrics.py
#                                        locally afterwards — that harness code does
#                                        not need to run in Colab, only the training)
#   - eval_metrics_<arm>.json, feature_stability_<arm>.csv

# %%
# !pip install -q "xgboost==2.*" lightgbm joblib pandas pyarrow scikit-learn

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from colab_competition import (
    CompetitionConfig,
    load_bundle,
    parse_seeds,
    primary_metric_name,
    run_competition,
    walk_forward_oof,
    write_artifact_bundle,
)

# %%
BUNDLE = Path(os.environ.get("ABLATION_BUNDLE", "ablation_colab_bundle.tgz"))
WORK = Path("ablation_work")
manifest = load_bundle(BUNDLE, WORK, "ablation_feature_manifest.json")

ARMS = manifest["arms"]  # {name: [feature_columns]} -- sourced from feature_blocks.py
REG_TARGET = manifest.get("regression_target_column", manifest.get("target_column"))
RELEVANCE_COL = manifest.get("relevance_column", "is_strong_setup")
STRONG_SOURCE = manifest.get("strong_setup_source_column", "fwd_max_return")
STRONG_THRESH = float(os.environ.get("STRONG_SETUP_THRESHOLD", manifest.get("strong_setup_threshold", 0.20)))
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", manifest.get("train_frac", 0.6)))
VAL_FRAC = float(os.environ.get("VAL_FRAC", manifest.get("val_frac", 0.2)))
RANK_GROUP = os.environ.get("RANK_GROUP", manifest.get("rank_group", "timestamp"))
TOP_K = int(os.environ.get("TOP_K", manifest.get("top_k", 20)))
# Keep the per-arm competition modest by default (5 arms x N families x M seeds all
# multiply) -- override via env if Colab compute allows a fuller sweep.
SEEDS = parse_seeds(os.environ.get("MODEL_SEEDS"), int(os.environ.get("N_SEEDS", "3")))
FAMILIES = os.environ.get("MODEL_FAMILIES", "xgb_classifier,xgb_ranker").split(",")
WF = manifest["walk_forward"]
BANNED_TARGET = manifest.get("banned_target_column", "trend_persistence")

df = pd.read_parquet(WORK / "training_matrix_4h_with_regime.parquet")
if "timestamp" not in df.columns:
    df = df.reset_index().rename(columns={df.columns[0]: "timestamp"})
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["ticker"] = df["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
df = df.sort_values("timestamp").reset_index(drop=True)

if STRONG_SOURCE not in df.columns:
    raise ValueError(f"strong-setup source column {STRONG_SOURCE!r} missing from matrix")
df[RELEVANCE_COL] = (pd.to_numeric(df[STRONG_SOURCE], errors="coerce") >= STRONG_THRESH).astype(int)

assert REG_TARGET != BANNED_TARGET, (
    f"{BANNED_TARGET} is a known forward-looking/leaky label (see manifest's "
    "banned_target_reason) -- refusing to train against it."
)

print("matrix", df.shape, "tickers", df["ticker"].nunique(), "range", df["timestamp"].min(), "->", df["timestamp"].max())
print("arms:", {name: len(cols) for name, cols in ARMS.items()})
print("families", FAMILIES, "| seeds", SEEDS)

# %%
device = os.environ.get("XGB_DEVICE") or ("cuda" if shutil.which("nvidia-smi") else "cpu")
print("xgb device:", device)

DIAGNOSTIC_COLUMNS = [c for c in manifest.get("diagnostic_label_columns", []) if c in df.columns and c != BANNED_TARGET]

arm_summaries = []
train_months = int(round(float(WF["train_years"]) * 12))

for arm_name, feature_cols in ARMS.items():
    print(f"\n=== arm: {arm_name} ({len(feature_cols)} features) ===")
    OUT = WORK / "artifacts" / arm_name.replace("+", "plus_")
    OUT.mkdir(parents=True, exist_ok=True)

    required = list(dict.fromkeys([RELEVANCE_COL, REG_TARGET] + feature_cols))
    arm_df = df.dropna(subset=[c for c in required if c in df.columns]).copy()
    print(f"  rows after arm-specific dropna: {len(arm_df)} (full matrix: {len(df)})")
    if len(arm_df) < int(WF.get("min_train_rows", 20000)):
        print(f"  SKIPPING {arm_name}: below min_train_rows ({WF.get('min_train_rows')}) after dropna")
        arm_summaries.append({"arm": arm_name, "n_features": len(feature_cols), "n_rows": len(arm_df), "skipped": True})
        continue

    cfg = CompetitionConfig(
        task_name=f"momentum_expansion_4h_ablation_{arm_name}",
        target_column=RELEVANCE_COL,
        regression_target_column=REG_TARGET,
        relevance_column=RELEVANCE_COL,
        feature_columns=feature_cols,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        seeds=SEEDS,
        families=FAMILIES,
        output_dir=OUT,
        rank_group=RANK_GROUP,
        top_k=TOP_K,
        device=device,
        timestamp_column="timestamp",
        id_columns=("timestamp", "ticker"),
    )
    result = run_competition(arm_df, cfg)
    best = result["best"]
    metric = primary_metric_name(result["results"])
    print(f"  best: {best.get('family')} seed={best.get('seed')} {metric}={best.get(metric) if metric else float('nan')}")

    oof = walk_forward_oof(
        arm_df, cfg, str(best["family"]), int(best["seed"]),
        train_months=train_months,
        embargo_days=int(WF["embargo_days"]),
        test_months=int(WF["test_months"]),
        min_train_rows=int(WF.get("min_train_rows", 20000)),
        diagnostic_columns=DIAGNOSTIC_COLUMNS,
    )
    oof_spearman = float("nan")
    if not oof.empty:
        oof.to_parquet(WORK / f"oof_preds_{arm_name.replace('+', 'plus_')}.parquet")
        oof_spearman = float(oof["score"].corr(oof["y"], method="spearman"))
        print(f"  walk-forward OOF rows={len(oof)} Spearman(score,y)={oof_spearman:.4f}")

    # NOTE: written to WORK directly (not the per-arm OUT subdir) so
    # write_artifact_bundle -- which only tars top-level files, not
    # subdirectories -- actually includes these in the downloaded bundle.
    (WORK / f"eval_metrics_{arm_name.replace('+', 'plus_')}.json").write_text(json.dumps({
        "arm": arm_name, "best": best.to_dict(), "primary_metric": metric,
        "oof_rows": int(len(oof)), "oof_spearman": oof_spearman,
        "n_rows_after_dropna": int(len(arm_df)), "n_features": len(feature_cols),
    }, indent=2, default=str))
    result["stability"].to_csv(WORK / f"feature_stability_{arm_name.replace('+', 'plus_')}.csv", index=False)

    arm_summaries.append({
        "arm": arm_name, "n_features": len(feature_cols), "n_rows": len(arm_df),
        "best_family": best.get("family"), "best_seed": best.get("seed"),
        "primary_metric": metric, "primary_metric_value": best.get(metric) if metric else np.nan,
        "oof_rows": len(oof), "oof_spearman": oof_spearman, "skipped": False,
    })

# %%
comparison = pd.DataFrame(arm_summaries)
comparison.to_csv(WORK / "ablation_arm_comparison.csv", index=False)
print("\n=== ablation arm comparison ===")
print(comparison.to_string(index=False))
print(
    "\nRead this table alongside the local training-free screen "
    "(strategies/momentum_expansion/ablation/run_screen.py results): a higher "
    "primary_metric_value / oof_spearman for a non-baseline arm is the actual "
    "trained-model evidence the screen could not produce. Per the plan's acceptance "
    "criteria (§5), do not promote a block into the live manifest without this table "
    "AND a positive incremental read in >=3 walk-forward periods AND a week-block "
    "bootstrap CI excluding zero AND BH-FDR survival."
)

out_bundle = Path("ablation_model_bundle.tgz")
write_artifact_bundle(WORK, out_bundle)
print("download:", out_bundle)
