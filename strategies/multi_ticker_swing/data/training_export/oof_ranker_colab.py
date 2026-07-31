# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # multi_ticker_swing 30m — walk-forward OOF RANKER trainer (long + short)
#
# Produces leak-free out-of-fold ranker scores for the meta ranker, plus a final
# full-history model per side for live inference. This is NOT the competition (that was a
# single-split bake-off to pick the winner). Here we take the winning family — **xgb_ranker**
# — and train it walk-forward OOF, exactly the way momentum / HTF feed the meta ranker.
#
# Upload `swing_context_colab_bundle.tgz` (already contains the matrix + context parquets +
# feature_manifest.json + colab_competition.py), drop this file next to it, and run top to bottom.
#
# Env knobs (all optional):
#   SIDES="long,short"   # which rankers to train (relevance: long=label 2, short=label 0)
#   N_FOLDS=5            # walk-forward folds; fold 1 is unscored (no prior data)
#   SEED=44             # winning long seed
#   SWING_BUNDLE=swing_context_colab_bundle.tgz

# %%
# !pip install -q "xgboost==2.*" joblib pandas pyarrow scikit-learn

import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from colab_competition import load_bundle  # shipped inside the context bundle

# Use GPU when available (Colab GPU runtime) — falls back to CPU hist otherwise.
DEVICE = os.environ.get("XGB_DEVICE") or ("cuda" if shutil.which("nvidia-smi") else "cpu")
print("xgb device:", DEVICE)

# %%
BUNDLE = Path(os.environ.get("SWING_BUNDLE", "swing_context_colab_bundle.tgz"))
WORK = Path("swing_oof_work")
manifest = load_bundle(BUNDLE, WORK, "feature_manifest.json")

FEATURES: list[str] = manifest["feature_columns"]
TARGET = manifest.get("target_column", "target")
WEIGHT_COL = manifest.get("sample_weight_column", "sample_weight")
XGB_CFG = dict(manifest.get("xgboost_config", {}))          # reuse the long winner's hyperparams
SIDES = [s.strip() for s in os.environ.get("SIDES", "long,short").split(",") if s.strip()]
N_FOLDS = int(os.environ.get("N_FOLDS", "5"))
SEED = int(os.environ.get("SEED", "44"))
TOP_K = int(os.environ.get("TOP_K", "10"))
POS_LABEL = {"long": 2, "short": 0}                         # relevance target per side

OUT = WORK / "oof_artifacts"
OUT.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Build the merged 192-feature frame (identical merge to swing_train_colab.py)

# %%
df = pd.read_parquet(WORK / manifest["matrix_file"])
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["ticker"] = df["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)

bar = pd.read_parquet(WORK / manifest["bar_context_file"])
bar["timestamp"] = pd.to_datetime(bar["timestamp"], utc=True)
bar["ticker"] = bar["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
df = df.merge(bar, on=["timestamp", "ticker"], how="left")
del bar

ctx = pd.read_parquet(WORK / manifest["context_file"])
ctx["date"] = pd.to_datetime(ctx["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
ctx["ticker"] = ctx["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
df["_cd"] = df["timestamp"].dt.tz_convert(None).dt.normalize()
df = df.merge(ctx.rename(columns={"date": "_cd"}), on=["ticker", "_cd"], how="left").drop(columns=["_cd"])
del ctx

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing {len(missing)} features after merge: {missing[:20]}")
for c in FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)

df = df.dropna(subset=[TARGET]).sort_values("timestamp").reset_index(drop=True)
print("merged frame", df.shape, "tickers", df["ticker"].nunique(),
      "range", df["timestamp"].min(), "->", df["timestamp"].max())


# %% [markdown]
# ## Ranker config + helpers

# %%
def ranker_params(seed: int) -> dict:
    p = dict(XGB_CFG)
    for k in ("objective", "num_class", "eval_metric", "early_stopping_rounds"):
        p.pop(k, None)
    p.update(
        objective="rank:ndcg",
        eval_metric=f"ndcg@{TOP_K}",
        tree_method="hist",
        device=DEVICE,
        random_state=seed,
        n_jobs=-1,
    )
    return p


def group_sizes(frame: pd.DataFrame) -> list[int]:
    # frame is sorted by timestamp, so each timestamp's rows are contiguous
    return frame.groupby("timestamp", sort=False).size().astype(int).tolist()


def fit_ranker(train: pd.DataFrame, pos_label: int, seed: int, early: int) -> xgb.XGBRanker:
    es = int(len(train) * 0.85)
    fit_df, es_df = train.iloc[:es], train.iloc[es:]
    model = xgb.XGBRanker(**ranker_params(seed), early_stopping_rounds=early)
    model.fit(
        fit_df[FEATURES].to_numpy(np.float32),
        (fit_df[TARGET].to_numpy() == pos_label).astype(np.float32),
        group=group_sizes(fit_df),
        eval_set=[(es_df[FEATURES].to_numpy(np.float32), (es_df[TARGET].to_numpy() == pos_label).astype(np.float32))],
        eval_group=[group_sizes(es_df)],
        verbose=False,
    )
    return model


# %% [markdown]
# ## Walk-forward OOF + final full-history model, per side

# %%
EARLY = int(XGB_CFG.get("early_stopping_rounds", 60))
n = len(df)
fold = n // N_FOLDS
summary = {}

for side in SIDES:
    pos = POS_LABEL[side]
    oof = np.full(n, np.nan, dtype=np.float32)
    for k in range(1, N_FOLDS + 1):
        vs = (k - 1) * fold
        ve = k * fold if k < N_FOLDS else n
        if vs < 100:
            print(f"[{side}] fold {k}: no prior data — left unscored")
            continue
        model = fit_ranker(df.iloc[:vs], pos, SEED, EARLY)
        oof[vs:ve] = model.predict(df.iloc[vs:ve][FEATURES].to_numpy(np.float32))
        print(f"[{side}] fold {k}/{N_FOLDS}  train={vs}  pred={ve - vs}  best_iter={getattr(model, 'best_iteration', '?')}")

    preds = df[["timestamp", "ticker", TARGET]].copy()
    preds["y"] = (df[TARGET].to_numpy() == pos).astype(np.int8)   # relevance for this side
    preds["score"] = oof
    preds = preds.dropna(subset=["score"])
    preds.to_parquet(OUT / f"oof_preds_{side}.parquet", index=False)

    # final model on ALL data for live inference
    final = fit_ranker(df, pos, SEED, EARLY)
    joblib.dump(final, OUT / f"model_ranker_{side}.joblib")
    final.save_model(OUT / f"model_ranker_{side}.native")
    summary[side] = {"oof_rows_scored": int(len(preds)), "final_best_iter": int(getattr(final, "best_iteration", -1) or -1)}
    print(f"[{side}] done — oof scored {len(preds)} rows")

# %%
meta = {
    "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_swing_30m_oof_ranker"),
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "task": "multi_ticker_swing_30m_oof_ranker",
    "family": "xgb_ranker",
    "seed": SEED,
    "n_folds": N_FOLDS,
    "sides": SIDES,
    "positive_label": POS_LABEL,
    "feature_count": len(FEATURES),
    "xgboost_config_source": "feature_manifest.json (long winner)",
    "summary": summary,
    "feature_names": FEATURES,
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

bundle_out = Path("swing_oof_ranker_bundle.tgz")
with tarfile.open(bundle_out, "w:gz") as tar:
    for p in sorted(OUT.iterdir()):
        if p.is_file():
            tar.add(p, arcname=p.name)
print("download:", bundle_out, "->", [p.name for p in sorted(OUT.iterdir())])
