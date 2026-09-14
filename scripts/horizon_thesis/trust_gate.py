"""Ranker-trust gate: predict whether TODAY's top-3 will beat its own cross-section.

From "When Alpha Breaks" (arXiv 2603.13252): a second model decides whether to
deploy the ranker at all, rather than scaling position size continuously -- the
paper's own ablation found inverse-uncertainty SIZING made things worse, and only
the on/off gate plus a tail cap helped.

Target per decision day: did the top-3 beat that day's scored universe over the
next 10 sessions (and 20, reported alongside)?

Gate features, all known before the decision:
  * the regime panel the matrix already carries (PIT)
  * cross-sectional shape: universe size, score dispersion, top-3 score margin,
    and how crowded the top-3 is in recent momentum
  * the ranker's own TRAILING record, lagged by the full hold so that every
    window feeding it had already closed -- the obvious place to leak, so the
    lag is explicit and tested below

Policies: always-on; on/off at a threshold fixed on TRAIN; and the three-level
hybrid (full / half / off) at the train quantiles. Prior work here found the
entry-day breadth gate unforecastable, so the null is the expected result.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/horizon_thesis"))

from build_decision_panel import REGIME_COLS, load_panel  # noqa: E402
from run_horizon_grid import block_boot  # noqa: E402

DATA = REPO / "research/execution_quality/data"
TOP_K = 3
MIN_XS = 200
HOLDS = [10, 20]
TRAIL_WINDOWS = [10, 20, 60]
PARAMS = dict(max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
              min_child_weight=20, tree_method="hist", reg_lambda=2.0,
              objective="binary:logistic", eval_metric="auc", seed=5)
N_ROUNDS = 300
SEED = 5


def daily_frame(panel: pd.DataFrame, hold: int) -> pd.DataFrame:
    """One row per decision day: the realised top-3 excess and the PIT gate features."""
    tgt = f"fwdret_{hold}"
    p = panel.dropna(subset=[tgt]).copy()
    p = p[p["xs_size"] >= MIN_XS]
    p = p.sort_values(["decision_day", "rank_mom"]).drop_duplicates(["decision_day", "ticker"])
    g = p.groupby("decision_day")
    uni = g[tgt].mean()
    top = p[p["rank_mom"] <= TOP_K].groupby("decision_day")[tgt].mean()
    out = pd.DataFrame({"excess": (top - uni).dropna()})
    out["universe_ret"] = uni.reindex(out.index)
    out["xs_size"] = g["xs_size"].first().reindex(out.index)
    out["score_std"] = g["mom_score"].std().reindex(out.index)
    top_score = p[p["rank_mom"] <= TOP_K].groupby("decision_day")["mom_score"].mean()
    out["top_margin"] = (top_score - g["mom_score"].mean()).reindex(out.index)
    out["top_past_ret_20"] = p[p["rank_mom"] <= TOP_K].groupby("decision_day")["past_ret_20"].mean().reindex(out.index)
    out["top_past_vol_20"] = p[p["rank_mom"] <= TOP_K].groupby("decision_day")["past_vol_20"].mean().reindex(out.index)
    for c in REGIME_COLS:
        out[c] = g[c].first().reindex(out.index)
    # Trailing record of the ranker, lagged so every window in it had closed.
    lag = hold + 1
    for w in TRAIL_WINDOWS:
        out[f"trail_mean_{w}"] = out["excess"].shift(lag).rolling(w).mean()
        out[f"trail_hit_{w}"] = (out["excess"] > 0).astype(float).shift(lag).rolling(w).mean()
    out[f"trail_std_{max(TRAIL_WINDOWS)}"] = out["excess"].shift(lag).rolling(max(TRAIL_WINDOWS)).std()
    return out.dropna().reset_index()


def main() -> None:
    rng = np.random.default_rng(SEED)
    panel = load_panel(require=["mom_score"])
    results: dict = {}
    for hold in HOLDS:
        d = daily_frame(panel, hold)
        feats = [c for c in d.columns if c not in {"decision_day", "excess", "universe_ret"}]
        n = len(d)
        i_tr, i_va = int(n * 0.60), int(n * 0.78)
        tr, va, te = d.iloc[:i_tr], d.iloc[i_tr:i_va], d.iloc[i_va:]
        ytr = (tr["excess"] > 0).astype(int)
        yva = (va["excess"] > 0).astype(int)
        print(f"\n{'=' * 96}\nHOLD {hold} sessions — {n} decision days "
              f"({d['decision_day'].min().date()} .. {d['decision_day'].max().date()})")
        print(f"  train={len(tr)} val={len(va)} test={len(te)}   base rate (top3 beats universe) "
              f"train {ytr.mean():.1%} / test {(te['excess'] > 0).mean():.1%}")
        bst = xgb.train(PARAMS, xgb.DMatrix(tr[feats], label=ytr), num_boost_round=N_ROUNDS,
                        evals=[(xgb.DMatrix(va[feats], label=yva), "v")],
                        early_stopping_rounds=30, verbose_eval=False)
        p_tr = bst.predict(xgb.DMatrix(tr[feats]))
        p_te = bst.predict(xgb.DMatrix(te[feats]))
        y_te = (te["excess"] > 0).astype(int).to_numpy()
        order = np.argsort(p_te)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(p_te)) + 1
        pos, neg = y_te.sum(), (1 - y_te).sum()
        auroc = float((ranks[y_te == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)) if pos and neg else np.nan
        thr_on = float(np.median(p_tr))
        thr_lo, thr_hi = float(np.quantile(p_tr, 0.40)), float(np.quantile(p_tr, 0.70))
        ex = te["excess"].to_numpy()
        weights = {
            "always_on": np.ones(len(te)),
            "gate_on_off": (p_te >= thr_on).astype(float),
            "hybrid_full_half_off": np.where(p_te >= thr_hi, 1.0, np.where(p_te >= thr_lo, 0.5, 0.0)),
        }
        print(f"  gate AUROC (test) = {auroc:.3f}   thresholds: on/off {thr_on:.3f}, "
              f"hybrid {thr_lo:.3f}/{thr_hi:.3f} (all fixed on train)")
        print(f"\n  {'policy':22s} {'exposure':>9s} {'mean excess':>12s} {'per unit risk':>14s} {'95% CI':>20s}")
        rows = []
        base = None
        for name, w in weights.items():
            r = ex * w
            lo, hi, p = block_boot(pd.Series(r, index=np.arange(len(r))), hold, rng, n=1000)
            ratio = float(r.mean() / r.std()) if r.std() > 0 else np.nan
            print(f"  {name:22s} {w.mean():8.1%} {r.mean() * 100:11.3f}% {ratio:14.3f} "
                  f"[{lo * 100:+8.3f},{hi * 100:+8.3f}]")
            rows.append(dict(policy=name, exposure=float(w.mean()), mean=float(r.mean()),
                             ratio=ratio, ci=[lo, hi], p=p))
            if name == "always_on":
                base = r
        for name in ("gate_on_off", "hybrid_full_half_off"):
            diff = pd.Series(ex * weights[name] - base)
            lo, hi, p = block_boot(diff, hold, rng, n=1000)
            print(f"  {name} - always_on: {diff.mean() * 100:+.3f}pp [{lo * 100:+.3f},{hi * 100:+.3f}] p={p:.3f}")
            rows.append(dict(compare=f"{name}_minus_always_on", diff=float(diff.mean()), ci=[lo, hi], p=p))
        imp = sorted(bst.get_score(importance_type="gain").items(), key=lambda kv: -kv[1])[:8]
        print(f"  top gate features by gain: {[(k, round(v)) for k, v in imp]}")
        results[f"hold_{hold}"] = dict(auroc=auroc, rows=rows, n_days=int(n),
                                       base_rate=float((te['excess'] > 0).mean()),
                                       importance=[(k, float(v)) for k, v in imp])
    path = DATA / "trust_gate.json"
    path.write_text(json.dumps(results, indent=1, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
