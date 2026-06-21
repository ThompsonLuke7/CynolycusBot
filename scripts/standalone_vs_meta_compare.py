"""
Standalone module performance vs the meta-ranker on a recent window.

For each per-bar ranking signal we take the top-20 names per 4H bar in the test window
and measure how often they were actually good trades. Compares the individual modules
(momentum, HTF swing, theme, news) against a meta model trained on everything before the
window — to decide whether confluence adds value or the strong modules should stay standalone.

"good trade" = meta_good (fwd_max_return>=12% & low MAE & +alpha & liquid); also report the
raw +10% forward-move hit-rate and mean forward max return.

Run: PYTHONPATH=. .venv/bin/python scripts/standalone_vs_meta_compare.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

MATRIX = "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet"
WIN_START = pd.Timestamp("2026-04-01", tz="UTC")
TOPK = 20

MODULES = {  # display name -> ranking column (higher = better)
    "momentum (mom_score)": "mom_score",
    "HTF swing (htf_score)": "htf_score",
    "theme (theme_heat)": "theme_heat_score",
    "news (catalyst)": "news_catalyst_score",
}


def topk_stats(df, score_col, k=TOPK):
    d = df.dropna(subset=[score_col])
    if d.empty:
        return None
    cov = len(d) / len(df)
    picks = d.groupby("timestamp", group_keys=False).apply(
        lambda g: g.nlargest(k, score_col), include_groups=True)
    return {
        "coverage": cov,
        "hit_meta_good": picks["meta_good"].mean(),
        "hit_+10%": (picks["fwd_max_return"] >= 0.10).mean(),
        "mean_fwd_max": picks["fwd_max_return"].mean(),
        "mean_fwd_close": picks["fwd_close_return"].mean(),
        "n_bars": picks["timestamp"].nunique(),
    }


def main():
    df = pd.read_parquet(MATRIX).reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["meta_good", "fwd_max_return", "fwd_close_return"])
    train, test = df[df.timestamp < WIN_START], df[df.timestamp >= WIN_START]
    print(f"train rows={len(train):,}  test rows={len(test):,}  "
          f"test window {test.timestamp.min().date()}..{test.timestamp.max().date()}")
    print(f"test bars={test.timestamp.nunique()}  test base meta_good={test.meta_good.mean():.3f}  "
          f"base +10%={ (test.fwd_max_return>=0.10).mean():.3f}\n")

    # ---- train a meta model on everything before the window ----
    # Use the manifest's real feature_columns (these EXCLUDE all forward/label fields, e.g.
    # fwd_max_drawdown, so there is no leakage — the production bundle trains on exactly these).
    import json
    manifest = json.load(open("signals/meta_context/meta_ranker/manifest.json"))
    feats = [c for c in manifest["feature_columns"] if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    assert not any(c.startswith("fwd_") for c in feats), "forward leak in features!"
    bst = xgb.train({"objective": "binary:logistic", "eval_metric": "auc", "max_depth": 5,
                     "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "verbosity": 0},
                    xgb.DMatrix(train[feats].astype("float32").values, label=train["meta_good"].values),
                    num_boost_round=300)
    test = test.copy()
    test["meta_model"] = bst.predict(xgb.DMatrix(test[feats].astype("float32").values))

    # also a cheap mom+htf rank-average ensemble (no training)
    test["mom_htf_avg"] = (test.groupby("timestamp")["mom_score"].rank(pct=True)
                           + test.groupby("timestamp")["htf_score"].rank(pct=True)) / 2

    print(f"{'signal':28s} {'cov':>5s} {'bars':>5s} {'meta_good':>9s} {'+10%hit':>8s} {'fwd_max':>8s} {'fwd_close':>9s} {'AUC':>6s}")
    rows = {**MODULES, "mom+htf avg": "mom_htf_avg", "META MODEL (78f)": "meta_model"}
    for name, col in rows.items():
        s = topk_stats(test, col)
        if s is None:
            print(f"{name:28s}  n/a"); continue
        d = test.dropna(subset=[col])
        auc = roc_auc_score(d["meta_good"], d[col]) if d["meta_good"].nunique() > 1 else float("nan")
        print(f"{name:28s} {s['coverage']*100:4.0f}% {s['n_bars']:5d} {s['hit_meta_good']*100:8.1f}% "
              f"{s['hit_+10%']*100:7.1f}% {s['mean_fwd_max']*100:7.1f}% {s['mean_fwd_close']*100:8.2f}% {auc:6.3f}")


if __name__ == "__main__":
    main()
