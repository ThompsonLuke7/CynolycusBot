"""Apply the parabolic-likelihood filter to SHARE trades.

Shares pay no option spread, so the filter's value shows up undiluted here. The
question is simply: if we had only taken the trades the filter ranks highest, would
share performance have improved?

Discipline:
  * the filter is trained on the first 60% of each module's history and only ever
    SCORES the remaining 40% -- every number below is out-of-sample.
  * significance via week-block bootstrap (trades cluster heavily in time).
  * reported against taking ALL trades, which is the current behavior.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
FEATS = REPO / "research/options_experiment/data/parabolic_dataset.parquet"
LABELS = REPO / "research/options_experiment/data/forward_excursion.parquet"
SPINE = REPO / "research/options_experiment/data/signal_spine.parquet"
OUT = REPO / "research/options_experiment/data/shares_filter_eval.parquet"

MODULES = ("momentum_expansion", "multi_ticker_swing_htf")
DROP = {
    "module", "ticker", "timestamp", "decision_ts", "entry_ts", "exit_ts", "signal_ts",
    "entry_px_underlying", "exit_px_underlying", "exit_reason", "provenance", "cadence",
    "source_file", "parabolic", "realized_move_atr", "direction", "bars_held",
    "tp_price", "sl_price", "y", "mfe_atr_20", "trade_id",
}


def score_module(mod: str, horizon: int, thresh: float) -> pd.DataFrame:
    d = pd.read_parquet(FEATS)
    d = d[d.module == mod].copy()
    lab = pd.read_parquet(LABELS)
    lab = lab[lab.module == mod].copy()
    for x in (d, lab):
        x["decision_ts"] = pd.to_datetime(x.decision_ts, utc=True)
    d = d.merge(
        lab[["ticker", "decision_ts", f"mfe_atr_{horizon}"]].drop_duplicates(["ticker", "decision_ts"]),
        on=["ticker", "decision_ts"], how="inner")
    d["y"] = (d[f"mfe_atr_{horizon}"] >= thresh).astype(int)
    d = d.sort_values("decision_ts")
    fc = [c for c in d.columns
          if c not in DROP and not c.startswith(("mfe_", "mae_", "parabolic_"))
          and pd.api.types.is_numeric_dtype(d[c])]
    i1 = int(len(d) * 0.6)
    tr, oos = d.iloc[:i1], d.iloc[i1:].copy()
    m = xgb.XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=20, reg_lambda=2.0,
                          eval_metric="logloss", n_jobs=8, random_state=7)
    X = lambda s: s[fc].replace([np.inf, -np.inf], np.nan)
    m.fit(X(tr), tr.y, verbose=False)
    oos["p"] = m.predict_proba(X(oos))[:, 1]
    print(f"  {mod}: trained {len(tr)}, scored OOS {len(oos)}, AUC={roc_auc_score(oos.y, oos.p):.3f}")
    return oos[["module", "ticker", "decision_ts", "entry_ts", "p", "y"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--thresh", type=float, default=4.0)
    args = ap.parse_args()
    rng = np.random.default_rng(11)

    print("training parabolic filter per module (first 60% only):")
    scored = pd.concat([score_module(m, args.horizon, args.thresh) for m in MODULES])

    spine = pd.read_parquet(SPINE)
    spine = spine[spine.module.isin(MODULES)].copy()
    spine["decision_ts"] = pd.to_datetime(spine.signal_ts.fillna(spine.entry_ts), utc=True)
    # actual SHARE return of each trade, as the modules trade them
    spine["ret"] = (spine.direction * (spine.exit_px_underlying - spine.entry_px_underlying)
                    / spine.entry_px_underlying)
    spine["week_key"] = spine.decision_ts.dt.to_period("W").astype(str)
    j = spine.merge(scored[["module", "ticker", "decision_ts", "p", "y"]],
                    on=["module", "ticker", "decision_ts"], how="inner")
    j = j[np.isfinite(j.ret)]
    j.to_parquet(OUT, index=False)
    print(f"\nOOS share trades scored by the filter: {len(j)}")

    for mod, g in j.groupby("module"):
        print(f"\n=== {mod} — share returns by filter percentile (OOS) ===")
        g = g.copy()
        g["bucket"] = pd.qcut(g.p, 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 HIGH"], duplicates="drop")
        t = g.groupby("bucket", observed=True).apply(lambda x: pd.Series({
            "n": len(x),
            "parabolic_rate": x.y.mean(),
            "mean_ret_%": 100 * x.ret.mean(),
            "median_ret_%": 100 * x.ret.median(),
            "win_rate": (x.ret > 0).mean(),
        }), include_groups=False)
        print(t.round(3).to_string())

        base = g.ret.mean()
        wks = g.week_key.unique()
        grp = {w: g[g.week_key == w] for w in wks}
        print(f"  ALL trades (current behavior): mean {100*base:+.2f}% over n={len(g)}")
        for k in (0.50, 0.30, 0.20, 0.10):
            sel = g.nlargest(max(int(len(g) * k), 20), "p")
            obs = sel.ret.mean() - base
            boot = []
            for _ in range(2000):
                pick = rng.choice(wks, size=len(wks), replace=True)
                sub = pd.concat([grp[w] for w in pick])
                s2 = sub.nlargest(max(int(len(sub) * k), 20), "p")
                boot.append(s2.ret.mean() - sub.ret.mean())
            lo, hi = np.percentile(boot, [2.5, 97.5])
            flag = "SIGNIFICANT" if lo > 0 else ("worse" if hi < 0 else "n.s.")
            print(f"    top {k:.0%}: n={len(sel):4d} mean {100*sel.ret.mean():+.2f}% "
                  f"(lift {100*obs:+.2f}pp, CI [{100*lo:+.2f},{100*hi:+.2f}]) {flag}")


if __name__ == "__main__":
    main()
