# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # momentum_expansion — Colab model-competition trainer (4H expansion model)
#
# Upload `momentum_colab_bundle.tgz`, then run top to bottom. Local execution is
# intentionally avoided per project rules (this is GPU/Colab work).
#
# Trains a competition across model families × seeds:
#   - XGBoost / LightGBM **regressor** on the continuous `expansion_survival_score`
#   - XGBoost / LightGBM **classifier** on the binary strong-setup flag
#   - XGBoost / LightGBM **ranker** (`rank:ndcg`, query groups = candidates per 4H bar)
# and reports averaged metrics, top-feature stability, and top-pick overlap so a
# single lucky seed is exposed as noise.
#
# Inputs (inside the bundle):
#   - training_matrix_4h.parquet   (features + expansion_survival_score + fwd_* labels)
#   - feature_manifest.json
#   - colab_competition.py         (shared harness)
#
# Outputs (download via momentum_model_bundle.tgz):
#   - expansion_xgb.json           (best XGB booster — live-inference compatible)
#   - best_model.joblib            (overall winner, any family)
#   - oof_preds.parquet            (walk-forward OOF of the winner — feeds the meta matrix)
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
BUNDLE = Path(os.environ.get("MOMENTUM_BUNDLE", "momentum_colab_bundle.tgz"))
WORK = Path("momentum_work")
manifest = load_bundle(BUNDLE, WORK, "feature_manifest.json")

FEATURES = manifest["feature_columns"]
REG_TARGET = manifest.get("regression_target_column", manifest.get("target_column", "expansion_survival_score"))
RELEVANCE_COL = manifest.get("relevance_column", "is_strong_setup")
STRONG_SOURCE = manifest.get("strong_setup_source_column", "fwd_max_return")
STRONG_THRESH = float(os.environ.get("STRONG_SETUP_THRESHOLD", manifest.get("strong_setup_threshold", 0.20)))
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", manifest.get("train_frac", 0.6)))
VAL_FRAC = float(os.environ.get("VAL_FRAC", manifest.get("val_frac", 0.2)))
RANK_GROUP = os.environ.get("RANK_GROUP", manifest.get("rank_group", "timestamp"))
TOP_K = int(os.environ.get("TOP_K", manifest.get("top_k", 20)))
SEEDS = parse_seeds(os.environ.get("MODEL_SEEDS"), int(os.environ.get("N_SEEDS", "7")))
FAMILIES = parse_families(os.environ.get("MODEL_FAMILIES")) if os.environ.get("MODEL_FAMILIES") else list(ALL_FAMILIES)
WF = manifest["walk_forward"]

# Diagnostic columns carried through the OOF so the meta-ranker spine + backtests
# can join forward outcomes. (The meta matrix reads score + these from MOM_OOF.)
DIAGNOSTIC_COLUMNS = [
    c for c in ["fwd_max_return", "fwd_max_alpha", "fwd_atr_adj_return", "fwd_max_drawdown",
                "fwd_close_return", "trend_persistence", "expansion_score", "expansion_target"]
]

df = pd.read_parquet(WORK / "training_matrix_4h.parquet").reset_index()
df = df.rename(columns={df.columns[0]: "timestamp"}) if "timestamp" not in df.columns else df
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["ticker"] = df["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
df = df.sort_values("timestamp").reset_index(drop=True)

# Binary strong-setup relevance/classifier target derived from the forward label.
if STRONG_SOURCE not in df.columns:
    raise ValueError(f"strong-setup source column {STRONG_SOURCE!r} missing from matrix")
df[RELEVANCE_COL] = (pd.to_numeric(df[STRONG_SOURCE], errors="coerce") >= STRONG_THRESH).astype(int)
DIAGNOSTIC_COLUMNS = [c for c in DIAGNOSTIC_COLUMNS if c in df.columns]

print("matrix", df.shape, "tickers", df["ticker"].nunique(), "range", df["timestamp"].min(), "->", df["timestamp"].max())
print("reg target", REG_TARGET, "| relevance", RELEVANCE_COL, f"(>= {STRONG_THRESH} of {STRONG_SOURCE})",
      "| positive rate", round(float(df[RELEVANCE_COL].mean()), 4))
print("families", FAMILIES, "| seeds", SEEDS)

# %%
device = os.environ.get("XGB_DEVICE") or ("cuda" if shutil.which("nvidia-smi") else "cpu")
print("xgb device:", device)

cfg = CompetitionConfig(
    task_name="momentum_expansion_4h",
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
    xgb_config=dict(manifest.get("xgboost_config", {"early_stopping_rounds": 100})),
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
# Walk-forward OOF of the winning family/seed → feeds the meta-ranker spine.
best = result["best"]
best_family, best_seed = str(best["family"]), int(best["seed"])
train_months = int(round(float(WF["train_years"]) * 12))
oof = walk_forward_oof(
    df, cfg, best_family, best_seed,
    train_months=train_months,
    embargo_days=int(WF["embargo_days"]),
    test_months=int(WF["test_months"]),
    min_train_rows=int(WF.get("min_train_rows", 20000)),
    diagnostic_columns=DIAGNOSTIC_COLUMNS,
)
if not oof.empty:
    oof.to_parquet(OUT / "oof_preds.parquet")
    print("OOF rows", len(oof), "| OOF Spearman(score,y)", round(float(oof["score"].corr(oof["y"], method="spearman")), 4))

# %%
# Final full-data fit for the shippable boosters. Live inference loads an XGBoost
# JSON booster (expansion_xgb.json), so always produce one from the best XGB family;
# also persist the overall winner's full-data model.
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
    booster.save_model(str(OUT / "expansion_xgb.json"))
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
    "strong_setup_threshold": STRONG_THRESH,
    "feature_columns": FEATURES,
}, indent=2, default=str))
(OUT / "feature_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

meta = {
    "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_momentum_4h_competition"),
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "winner": {"family": best_family, "seed": best_seed},
    "feature_names": FEATURES,
    "manifest": manifest,
    "competition_artifacts": sorted(p.name for p in OUT.iterdir() if p.is_file()),
}
(OUT / "meta.json").write_text(json.dumps(meta, default=str, indent=2))

out_bundle = Path("momentum_model_bundle.tgz")
write_artifact_bundle(OUT, out_bundle)
print("download:", out_bundle)
