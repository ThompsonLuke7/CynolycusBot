"""Train and persist the deployable parabolic filter for momentum_expansion.

Only momentum gets a model. multi_ticker_swing_htf uses `atr_rule_rank`, because a
plain ATR% sort matched the model within noise there (+2.43pp vs +2.92pp) and an
XGBoost dependency is not justified for a difference the CIs cannot separate.

Trains on the first 60% of history by time, reports held-out performance, then
persists to Data/models/parabolic_filter/<module>/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from signals.parabolic_filter.filter import (
    DEFAULT_HORIZON_BARS, DEFAULT_THRESHOLD_PCT, ParabolicFilter,
)

REPO = Path(__file__).resolve().parents[1]
FEATS = REPO / "research/options_experiment/data/parabolic_dataset.parquet"
LABELS = REPO / "research/options_experiment/data/forward_excursion.parquet"
OUTDIR = REPO / "Data/models/parabolic_filter"

DROP = {
    "module", "ticker", "timestamp", "decision_ts", "entry_ts", "exit_ts", "signal_ts",
    "entry_px_underlying", "exit_px_underlying", "exit_reason", "provenance", "cadence",
    "source_file", "parabolic", "realized_move_atr", "direction", "bars_held",
    "tp_price", "sl_price", "y", "trade_id",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="momentum_expansion")
    ap.add_argument("--threshold-pct", type=float, default=DEFAULT_THRESHOLD_PCT)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_BARS)
    args = ap.parse_args()

    d = pd.read_parquet(FEATS)
    d = d[d.module == args.module].copy()
    lab = pd.read_parquet(LABELS)
    lab = lab[lab.module == args.module].copy()
    for x in (d, lab):
        x["decision_ts"] = pd.to_datetime(x.decision_ts, utc=True)
    mfe = f"mfe_atr_{args.horizon}"
    lab["mfe_pct"] = lab[mfe] * lab["atr"] / lab["entry_px"]
    d = d.merge(lab[["ticker", "decision_ts", "mfe_pct"]].drop_duplicates(["ticker", "decision_ts"]),
                on=["ticker", "decision_ts"], how="inner")
    d["y"] = (d.mfe_pct >= args.threshold_pct).astype(int)
    d = d.sort_values("decision_ts")

    fc = [c for c in d.columns
          if c not in DROP and c != "mfe_pct"
          and not c.startswith(("mfe_", "mae_", "parabolic_"))
          and pd.api.types.is_numeric_dtype(d[c])]
    i1 = int(len(d) * 0.6)
    tr, oos = d.iloc[:i1], d.iloc[i1:]
    print(f"{args.module}: n={len(d)} feats={len(fc)} base_rate={d.y.mean():.1%} "
          f"(>= +{args.threshold_pct:.0%} within {args.horizon} bars)")
    print(f"  train {tr.decision_ts.min().date()}..{tr.decision_ts.max().date()} n={len(tr)}")
    print(f"  OOS   {oos.decision_ts.min().date()}..{oos.decision_ts.max().date()} n={len(oos)}")

    m = xgb.XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=20, reg_lambda=2.0,
                          eval_metric="logloss", n_jobs=8, random_state=7)
    X = lambda s: s[fc].replace([np.inf, -np.inf], np.nan)
    m.fit(X(tr), tr.y, verbose=False)
    p = m.predict_proba(X(oos))[:, 1]
    print(f"  OOS AUC = {roc_auc_score(oos.y, p):.3f}")
    for k in (0.20, 0.30):
        cut = np.quantile(p, 1 - k)
        sel = oos.y[p >= cut]
        print(f"    top {k:.0%}: n={len(sel)} precision {sel.mean():.1%} "
              f"(base {oos.y.mean():.1%}, lift {sel.mean()/oos.y.mean():.2f}x)")

    f = ParabolicFilter(booster=m, feature_names=fc, module=args.module,
                        threshold_pct=args.threshold_pct, horizon_bars=args.horizon,
                        trained_through=str(tr.decision_ts.max().date()))
    dest = OUTDIR / args.module
    f.save(dest)
    print(f"\nsaved -> {dest}")


if __name__ == "__main__":
    main()
