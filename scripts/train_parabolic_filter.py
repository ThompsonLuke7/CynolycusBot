"""Train and honestly evaluate the parabolic-likelihood filter.

Target: does this signal reach >= N ATR of favorable excursion within H bars?
(True forward MFE from underlying bars -- NOT the exit-rule-capped realized move.)

Evaluation discipline (AGENTS.md + the experiment pre-registration):
  * WALK-FORWARD by time. Train on the past, test on the future. Never shuffled.
  * The final test window is scored ONCE, after thresholds are fixed on validation.
  * Reported against the base rate, because a 40%-base-rate problem makes an
    unconditional "always yes" look deceptively good.
  * Precision at the operating point is what matters -- that is the hit rate an
    options program would actually experience.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FEATS = REPO / "research/options_experiment/data/parabolic_dataset.parquet"
LABELS = REPO / "research/options_experiment/data/forward_excursion.parquet"
OUT = REPO / "research/options_experiment/data/parabolic_filter_eval.json"

DROP = {
    "module", "ticker", "timestamp", "decision_ts", "entry_ts", "exit_ts", "signal_ts",
    "entry_px_underlying", "exit_px_underlying", "exit_reason", "provenance", "cadence",
    "source_file", "parabolic", "realized_move_atr", "direction", "bars_held",
    "tp_price", "sl_price", "entry_px", "atr", "score_y", "trade_id",
}


def load(module: str, horizon: int, thresh: float):
    f = pd.read_parquet(FEATS)
    f = f[f.module == module].copy()
    lab = pd.read_parquet(LABELS)
    lab = lab[lab.module == module].copy()
    key = ["ticker", "decision_ts"]
    for d in (f, lab):
        d["decision_ts"] = pd.to_datetime(d["decision_ts"], utc=True)
    lab = lab[key + [f"mfe_atr_{horizon}"]].drop_duplicates(key)
    d = f.merge(lab, on=key, how="inner")
    d["y"] = (d[f"mfe_atr_{horizon}"] >= thresh).astype(int)
    d = d[d[f"mfe_atr_{horizon}"].notna()].sort_values("decision_ts")
    feat_cols = [c for c in d.columns
                 if c not in DROP and c != "y" and not c.startswith("mfe_atr")
                 and not c.startswith("mae_atr") and not c.startswith("parabolic_")
                 and pd.api.types.is_numeric_dtype(d[c])]
    return d, feat_cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="momentum_expansion")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--thresh", type=float, default=4.0)
    args = ap.parse_args()

    d, fc = load(args.module, args.horizon, args.thresh)
    print(f"{args.module}: n={len(d)} features={len(fc)} "
          f"base_rate={d.y.mean():.1%} (MFE_{args.horizon} >= {args.thresh} ATR)")

    # walk-forward: 60% train / 20% validation / 20% test, strictly by time
    n = len(d)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    tr, va, te = d.iloc[:i1], d.iloc[i1:i2], d.iloc[i2:]
    print(f"  train {tr.decision_ts.min().date()}..{tr.decision_ts.max().date()} n={len(tr)} base={tr.y.mean():.1%}")
    print(f"  valid {va.decision_ts.min().date()}..{va.decision_ts.max().date()} n={len(va)} base={va.y.mean():.1%}")
    print(f"  test  {te.decision_ts.min().date()}..{te.decision_ts.max().date()} n={len(te)} base={te.y.mean():.1%}")

    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, average_precision_score

    X = lambda s: s[fc].replace([np.inf, -np.inf], np.nan)
    m = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        reg_lambda=2.0, eval_metric="logloss", early_stopping_rounds=40,
        n_jobs=4, random_state=7,
    )
    m.fit(X(tr), tr.y, eval_set=[(X(va), va.y)], verbose=False)

    res = {"module": args.module, "horizon": args.horizon, "thresh": args.thresh,
           "n": int(len(d)), "n_features": len(fc)}
    for name, s in (("valid", va), ("test", te)):
        p = m.predict_proba(X(s))[:, 1]
        base = s.y.mean()
        auc = roc_auc_score(s.y, p)
        ap_ = average_precision_score(s.y, p)
        print(f"\n=== {name.upper()} ===  base rate {base:.1%}  AUC {auc:.3f}  AP {ap_:.3f}")
        print(f"  {'top-k%':>8}{'n':>7}{'precision':>11}{'lift':>8}")
        row = {"base_rate": float(base), "auc": float(auc), "ap": float(ap_), "topk": {}}
        for k in (0.05, 0.10, 0.20, 0.30, 0.50):
            cut = np.quantile(p, 1 - k)
            sel = s.y[p >= cut]
            if len(sel) == 0:
                continue
            prec = sel.mean()
            print(f"  {k*100:>7.0f}%{len(sel):>7}{prec:>10.1%}{prec/base:>8.2f}x")
            row["topk"][str(k)] = {"n": int(len(sel)), "precision": float(prec),
                                   "lift": float(prec / base)}
        res[name] = row

    imp = sorted(zip(fc, m.feature_importances_), key=lambda x: -x[1])[:12]
    print("\ntop features:", ", ".join(f"{k}({v:.3f})" for k, v in imp))
    res["top_features"] = [[k, float(v)] for k, v in imp]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev[f"{args.module}_h{args.horizon}_t{args.thresh}"] = res
    OUT.write_text(json.dumps(prev, indent=1))
    print(f"\nwrote -> {OUT}")


if __name__ == "__main__":
    main()
