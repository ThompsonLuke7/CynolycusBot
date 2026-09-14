"""'Keep it only while it is still working' -- tested at a day-5 checkpoint.

The thesis trade runs 2-6 weeks, but a fixed 30-session hold also sits through
the ones that die early. So: enter the top-k momentum names, look again at the
close of session 5, and decide whether to keep each one to session 30.

Policies, all scored on the SAME test-period entries:

  hold_30          ride every entry to session 30 (the thesis, unmanaged)
  exit_5           take whatever is there at session 5 (roughly today's behaviour)
  rule_working     keep if the trade is up at the checkpoint
  rule_beats_spy   keep if it is beating SPY over the same five sessions
  ml_keep          keep if a model predicts positive remaining return
  ml_keep_top_half keep the better half by predicted remaining return
                   (threshold fixed on TRAIN, never on test)

A kept trade compounds: (1+ret to day 5) x (1+ret day 5 -> day 30). A dropped one
stops at the day-5 return. The separation test -- does the rule/model actually
sort the remaining return? -- is reported next to the policy P&L, because a
policy can win on P&L while sorting nothing.

Leakage: checkpoint features use only sessions 1-5, the target is sessions 6-30,
the split is time-ordered with a 30-session embargo, and corporate-action-flagged
windows are dropped by the panel loader.
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

from build_decision_panel import load_panel  # noqa: E402

DATA = REPO / "research/execution_quality/data"
CHECKPOINT_FEATURES = ["cp_ret", "cp_mfe", "cp_mae", "cp_close_vs_high", "cp_up_day_share",
                       "cp_volume_ratio", "cp_vs_spy"]
ENTRY_FEATURES = ["mom_score", "rank_mom", "past_ret_20", "past_ret_60", "past_vol_20",
                  "beta_60", "dist_252_high", "log_dollar_vol_20",
                  "regime_spy_trend", "regime_spy_ret_20", "regime_vix_z", "breadth_z"]
TARGET = "rem_5_30"
EMBARGO_SESSIONS = 30
PARAMS = dict(max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=30, tree_method="hist", device="cuda",
              reg_lambda=2.0, objective="reg:squarederror", seed=7)
N_ROUNDS = 400
SEED = 7


def entries(panel: pd.DataFrame, k: int) -> pd.DataFrame:
    e = panel[panel["rank_mom"] <= k].copy()
    e["cp_vs_spy"] = e["cp_ret"] - e["spyret_5"]
    # two 4H bars a day can pick the same name twice; one entry per name per day
    return e.sort_values(["decision_day", "rank_mom"]).drop_duplicates(["decision_day", "ticker"])


def split(e: pd.DataFrame):
    days = np.array(sorted(e["decision_day"].unique()))
    i_tr, i_va = int(len(days) * 0.60), int(len(days) * 0.78)
    tr_end = days[i_tr]
    va_start, va_end = days[min(i_tr + EMBARGO_SESSIONS, len(days) - 1)], days[i_va]
    te_start = days[min(i_va + EMBARGO_SESSIONS, len(days) - 1)]
    d = e["decision_day"].to_numpy()
    return e[d <= tr_end], e[(d >= va_start) & (d <= va_end)], e[d >= te_start], (tr_end, va_end, te_start)


def policy_returns(e: pd.DataFrame, keep: np.ndarray) -> np.ndarray:
    held = (1.0 + e["fwdret_5"].to_numpy()) * (1.0 + e[TARGET].to_numpy()) - 1.0
    return np.where(keep, held, e["fwdret_5"].to_numpy())


def day_block_boot(values: np.ndarray, days: np.ndarray, rng, n=2000) -> tuple[float, float]:
    """Bootstrap whole decision days: entries on one day share market moves."""
    uniq, inv = np.unique(days, return_inverse=True)
    by_day = [values[inv == i] for i in range(len(uniq))]
    means = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(by_day), len(by_day))
        means[i] = np.concatenate([by_day[j] for j in pick]).mean()
    return tuple(np.percentile(means, [2.5, 97.5]))


def main() -> None:
    rng = np.random.default_rng(SEED)
    panel = load_panel(require=["mom_score", "cp_ret", TARGET, "fwdret_5", "fwdret_30", "spyret_5"])
    out: dict = {}
    for k in (3, 10):
        e = entries(panel, k)
        tr, va, te, bounds = split(e)
        print(f"\n{'=' * 96}\nTOP-{k} ENTRIES   total={len(e):,}  train={len(tr):,} val={len(va):,} test={len(te):,}")
        print(f"  train<= {bounds[0].date()}   val<= {bounds[1].date()}   test>= {bounds[2].date()} "
              f"(embargo {EMBARGO_SESSIONS} sessions)")
        feats = CHECKPOINT_FEATURES + ENTRY_FEATURES
        bst = xgb.train(PARAMS, xgb.DMatrix(tr[feats], label=tr[TARGET]), num_boost_round=N_ROUNDS,
                        evals=[(xgb.DMatrix(va[feats], label=va[TARGET]), "v")],
                        early_stopping_rounds=40, verbose_eval=False)
        pred_tr = bst.predict(xgb.DMatrix(tr[feats]))
        pred_te = bst.predict(xgb.DMatrix(te[feats]))
        thr_half = float(np.median(pred_tr))

        keeps = {
            "hold_30": np.ones(len(te), bool),
            "exit_5": np.zeros(len(te), bool),
            "rule_working": te["cp_ret"].to_numpy() > 0,
            "rule_beats_spy": te["cp_vs_spy"].to_numpy() > 0,
            "ml_keep": pred_te > 0,
            "ml_keep_top_half": pred_te > thr_half,
        }
        days = te["decision_day"].to_numpy()
        print(f"\n  {'policy':17s} {'kept':>6s} {'mean/trade':>11s} {'median':>8s} {'win%':>6s} {'95% CI mean':>18s}")
        rows = []
        for name, keep in keeps.items():
            r = policy_returns(te, keep)
            lo, hi = day_block_boot(r, days, rng)
            print(f"  {name:17s} {keep.mean():6.1%} {r.mean() * 100:10.2f}% {np.median(r) * 100:7.2f}% "
                  f"{(r > 0).mean():5.1%} [{lo * 100:+7.2f},{hi * 100:+7.2f}]")
            rows.append(dict(k=k, policy=name, kept=float(keep.mean()), mean=float(r.mean()),
                             median=float(np.median(r)), win=float((r > 0).mean()),
                             ci=[float(lo), float(hi)], n=int(len(r))))

        print(f"\n  separation on remaining return ({TARGET}, test):")
        for name in ("rule_working", "rule_beats_spy", "ml_keep_top_half"):
            keep = keeps[name]
            if keep.all() or not keep.any():
                continue
            a, b = te[TARGET].to_numpy()[keep], te[TARGET].to_numpy()[~keep]
            d = a.mean() - b.mean()
            pooled = np.concatenate([a, b])
            lab = np.concatenate([np.ones(len(a)), np.zeros(len(b))])
            null = np.array([pooled[rng.permutation(len(pooled))][: len(a)].mean()
                             - pooled[rng.permutation(len(pooled))][len(a):].mean() for _ in range(1000)])
            p = float((np.abs(null) >= abs(d)).mean())
            print(f"    {name:17s} kept {a.mean() * 100:+6.2f}%  dropped {b.mean() * 100:+6.2f}%  "
                  f"diff {d * 100:+6.2f}pp  perm p={p:.3f}")
            rows.append(dict(k=k, separation=name, kept_mean=float(a.mean()), dropped_mean=float(b.mean()),
                             diff=float(d), perm_p=p))
        rho = float(pd.Series(te["cp_ret"].to_numpy()).corr(pd.Series(te[TARGET].to_numpy()), method="spearman"))
        rho_ml = float(pd.Series(pred_te).corr(pd.Series(te[TARGET].to_numpy()), method="spearman"))
        print(f"    day-5 return vs remaining return: Spearman {rho:+.4f}   model vs remaining: {rho_ml:+.4f}")
        out[f"top{k}"] = dict(rows=rows, rho_cp=rho, rho_ml=rho_ml, thr_half=thr_half,
                             bounds=[str(b) for b in bounds], best_iter=int(bst.best_iteration or 0))
    path = DATA / "continuation_model.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
