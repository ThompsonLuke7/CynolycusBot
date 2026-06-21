# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # Meta Ranker — Colab model-competition trainer
#
# Upload `meta_ranker_colab_bundle.tgz`, then run top to bottom (GPU/Colab work).
#
# Trains a competition across model families × seeds on the leakage-controlled meta
# matrix (out-of-fold base scores + theme/news/calendar context):
#   - XGBoost / LightGBM **regressor** on the continuous `trade_quality`
#   - XGBoost / LightGBM **classifier** on the binary `meta_good` flag
#   - XGBoost / LightGBM **ranker** (`rank:ndcg`, query groups = candidates per 4H bar)
# Reports averaged metrics, top-feature stability, and top-pick overlap so a single
# lucky seed is exposed as noise, then regenerates walk-forward OOF for the winner.
#
# Inputs (inside the bundle):
#   - meta_ranker_matrix.parquet   (features + trade_quality + meta_good, leakage-controlled)
#   - manifest.json
#   - colab_competition.py         (shared harness)
#
# Outputs (download via meta_ranker_model_bundle.tgz):
#   - meta_ranker_xgb.json         (best XGB booster — live-inference compatible)
#   - best_model.joblib            (overall winner, any family)
#   - oof_preds.parquet            (walk-forward OOF meta scores for backtesting)
#   - seed_results.csv, model_family_summary.csv, feature_stability.csv,
#     top_pick_overlap.csv, competition_meta.json, eval_metrics.json

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
    ALL_FAMILIES,
    CompetitionConfig,
    load_bundle,
    parse_families,
    parse_seeds,
    primary_metric_name,
    run_competition,
    fit_family,
    walk_forward_oof,
    write_artifact_bundle,
)

# %%
BUNDLE = Path(os.environ.get("META_BUNDLE", "meta_ranker_colab_bundle.tgz"))
WORK = Path("meta_work")
# Default target = "quality" (meta_good). For the upside variant (bigger raw winners) set
# META_MANIFEST=manifest_upside.json to train the second target in the same session.
manifest = load_bundle(BUNDLE, WORK, os.environ.get("META_MANIFEST", "manifest.json"))

FEATURES = manifest["feature_columns"]
REG_TARGET = manifest.get("regression_target_column", manifest.get("label_column", "trade_quality"))
RELEVANCE_COL = manifest.get("relevance_column") or manifest.get("target_column", "meta_good")
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", manifest.get("train_frac", 0.6)))
VAL_FRAC = float(os.environ.get("VAL_FRAC", manifest.get("val_frac", 0.2)))
RANK_GROUP = os.environ.get("RANK_GROUP", manifest.get("rank_group", "timestamp"))
TOP_K = int(os.environ.get("TOP_K", manifest.get("top_k", 20)))
SEEDS = parse_seeds(os.environ.get("MODEL_SEEDS"), int(os.environ.get("N_SEEDS", "7")))
FAMILIES = parse_families(os.environ.get("MODEL_FAMILIES")) if os.environ.get("MODEL_FAMILIES") else list(ALL_FAMILIES)
WF = manifest.get("walk_forward", {"train_months": 18, "embargo_days": 21, "test_months": 4, "min_train_rows": 50000})
CATS = [c for c in manifest.get("categorical_columns", []) if c in FEATURES]

# Forward outcomes carried through the OOF for backtest-style evaluation.
DIAGNOSTIC_COLUMNS = ["fwd_close_return", "fwd_max_return", "fwd_max_drawdown", "fwd_atr_adj_return", "meta_good"]

df = pd.read_parquet(WORK / "meta_ranker_matrix.parquet").reset_index()
df = df.rename(columns={df.columns[0]: "timestamp"}) if "timestamp" not in df.columns else df
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["ticker"] = df["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
for c in CATS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)
DIAGNOSTIC_COLUMNS = [c for c in DIAGNOSTIC_COLUMNS if c in df.columns]

print("matrix", df.shape, "tickers", df["ticker"].nunique(), "range", df["timestamp"].min(), "->", df["timestamp"].max())
print("reg target", REG_TARGET, "| relevance/classifier", RELEVANCE_COL,
      "| positive rate", round(float(df[RELEVANCE_COL].mean()), 4) if RELEVANCE_COL in df.columns else "n/a")
print("families", FAMILIES, "| seeds", SEEDS)

# %%
device = os.environ.get("XGB_DEVICE") or ("cuda" if shutil.which("nvidia-smi") else "cpu")
print("xgb device:", device)

cfg = CompetitionConfig(
    task_name="meta_ranker_4h",
    target_column=RELEVANCE_COL,
    regression_target_column=REG_TARGET,
    relevance_column=RELEVANCE_COL,
    feature_columns=FEATURES,
    train_frac=TRAIN_FRAC,
    val_frac=VAL_FRAC,
    seeds=SEEDS,
    families=FAMILIES,
    output_dir=WORK / "artifacts",
    rank_group=RANK_GROUP,
    top_k=TOP_K,
    xgb_config=dict(manifest.get("xgboost_config", {"early_stopping_rounds": 75})),
    lgbm_config=dict(manifest.get("lightgbm_config", {})),
    device=device,
    timestamp_column="timestamp",
    id_columns=("timestamp", "ticker"),
)
result = run_competition(df, cfg)
print("best\n", result["best"])
print("family summary\n", result["summary"])
print("feature stability (head)\n", result["stability"].head(20))
print("top-pick overlap\n", result["pick_overlap"])

OUT = cfg.output_dir

# %%
# Walk-forward OOF of the winner → backtestable meta scores.
best = result["best"]
best_family, best_seed = str(best["family"]), int(best["seed"])
oof = walk_forward_oof(
    df, cfg, best_family, best_seed,
    train_months=int(WF.get("train_months", 18)),
    embargo_days=int(WF.get("embargo_days", 21)),
    test_months=int(WF.get("test_months", 4)),
    min_train_rows=int(WF.get("min_train_rows", 50000)),
    diagnostic_columns=DIAGNOSTIC_COLUMNS,
)
if not oof.empty:
    oof.to_parquet(OUT / "oof_preds.parquet")
    print("OOF rows", len(oof), "| OOF Spearman(score,y)", round(float(oof["score"].corr(oof["y"], method="spearman")), 4))
    # Decile diagnostics vs forward outcomes.
    diag = oof.copy()
    diag["decile"] = pd.qcut(diag["score"].rank(method="first"), 10, labels=False) + 1
    agg = {"n": ("score", "size"), "score_mean": ("score", "mean"), "y_mean": ("y", "mean")}
    if "fwd_close_return" in diag.columns:
        agg["fwd_close_mean"] = ("fwd_close_return", "mean")
    if "meta_good" in diag.columns:
        agg["good_rate"] = ("meta_good", "mean")
    buckets = diag.groupby("decile").agg(**agg)
    buckets.to_csv(OUT / "oof_score_buckets.csv")
    print(buckets)

# %%
# Final full-data fit for the shippable boosters (live inference loads meta_ranker_xgb.json).
def _final_fit(family: str, seed: int):
    days = np.sort(df["timestamp"].dt.normalize().unique())
    cut = pd.Timestamp(days[min(max(1, int(len(days) * (1 - VAL_FRAC))), len(days) - 1)])
    tr, va = df[df["timestamp"] < cut], df[df["timestamp"] >= cut]
    return fit_family(family, seed, tr, va, cfg)

results = result["results"]
xgb_rows = results[results["family"].str.startswith("xgb")]
if not xgb_rows.empty:
    metric = primary_metric_name(results)
    best_xgb = xgb_rows.sort_values(metric, ascending=False, na_position="last").iloc[0] if metric else xgb_rows.iloc[0]
    xgb_model = _final_fit(str(best_xgb["family"]), int(best_xgb["seed"]))
    booster = xgb_model.get_booster() if hasattr(xgb_model, "get_booster") else xgb_model
    booster.save_model(str(OUT / "meta_ranker_xgb.json"))
    print("saved live-compat booster from", best_xgb["family"], "seed", int(best_xgb["seed"]))

import joblib
winner_full = _final_fit(best_family, best_seed)
joblib.dump(winner_full, OUT / "best_model_full.joblib")

# %%
(OUT / "eval_metrics.json").write_text(json.dumps({
    "best": best.to_dict(),
    "winner_family": best_family,
    "winner_seed": best_seed,
    "primary_metric": primary_metric_name(results),
    "target_column": REG_TARGET,
    "relevance_column": RELEVANCE_COL,
    "label_definition": manifest.get("label_definition"),
    "feature_columns": FEATURES,
    "categorical_columns": CATS,
}, indent=2, default=str))
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

meta = {
    "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_meta_ranker_4h_competition"),
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "winner": {"family": best_family, "seed": best_seed},
    "feature_names": FEATURES,
    "manifest": manifest,
    "competition_artifacts": sorted(p.name for p in OUT.iterdir() if p.is_file()),
}
(OUT / "meta.json").write_text(json.dumps(meta, default=str, indent=2))

out_bundle = Path("meta_ranker_model_bundle.tgz")
write_artifact_bundle(OUT, out_bundle)
print("download:", out_bundle)
