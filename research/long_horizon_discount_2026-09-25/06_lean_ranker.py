"""Step (b) from Part 2: a leaner walk-forward ranker, run 2026-09-29.

Two changes from 03, and each one tests a single diagnosis:
  1. Market-level features are removed. spy_dist_sma200 and spy_ret_63 take the same value
     for every row on a date, so they can only shift the whole cross-section, and a
     within-date ranking ignores such shifts.
  2. Each feature becomes its within-date percentile. Otherwise a regime shift in feature
     LEVELS (e.g. every vol doubling in 2020) can stand in for a date effect.
Arms: lambdarank (per-date query, decile grades); regression on the same inputs (checks the
objective); permuted-label lambdarank (noise control). Folds, purge, metrics and CIs match 03.
Also prints each arm's mean top-20 vol percentile, because a vol tilt is the usual source of
fake top-k edges in this repo.

Scores are written for EVERY panel date in each test year, including dates too recent to have
a label, so 05_rotation_backtest.py can trade them. Metrics use labeled rows only.

    .venv/bin/python research/long_horizon_discount_2026-09-25/06_lean_ranker.py --hold 126
"""
from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = spec_from_file_location("wf", HERE / "03_walkforward.py")
wf = module_from_spec(_spec)
_spec.loader.exec_module(wf)

LEAN = ["mom_12_1", "ret_126", "ret_63", "ret_21", "ret_5", "dist_sma200", "min_low_vs_sma200_20",
        "sma200_slope_60", "dd_52w_high", "rv_63", "tmo_main", "days_since_tmo_turn",
        "pct_above_sma200_252", "dv_rank"]
BASE = dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=400, feature_fraction=0.7,
            bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, n_estimators=300, verbose=-1,
            seed=wf.SEED, deterministic=True, num_threads=8)


def rank_features(p: pd.DataFrame) -> pd.DataFrame:
    """Within-date percentile of each LEAN feature. NaN stays NaN (LightGBM routes it)."""
    return p.groupby("date")[LEAN].rank(pct=True)


def fit_ranker(tr: pd.DataFrame, X: pd.DataFrame, grade: pd.Series) -> lgb.LGBMRanker:
    order = np.argsort(tr["date"].values, kind="stable")
    group = tr["date"].iloc[order].value_counts(sort=False).reindex(tr["date"].iloc[order].unique()).values
    return lgb.LGBMRanker(objective="lambdarank", **BASE).fit(
        X.iloc[order], grade.iloc[order], group=group)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=126)
    h = ap.parse_args().hold
    y_raw = f"fwd_xs_{h}"
    p = pd.read_parquet(wf.PANEL)
    R = rank_features(p)
    p["y_rank"] = p.groupby("date")[y_raw].rank(pct=True)
    p["grade"] = np.floor(p["y_rank"] * 10).clip(0, 9)
    labeled = pd.DatetimeIndex(sorted(p.loc[p[y_raw].notna(), "date"].unique()))
    rng = np.random.default_rng(wf.SEED)
    preds = []
    for year, tr_d, _ in wf.folds(labeled, h):
        tr_mask = p["date"].isin(tr_d) & p[y_raw].notna()
        te_mask = p["date"].dt.year == year
        tr, Xtr, Xte = p[tr_mask], R[tr_mask], R[te_mask]
        te = p.loc[te_mask, ["date", "ticker", y_raw, f"fwd_ret_{h}", "rv_63"]].copy()
        grade = tr["grade"].astype(int)
        te["s_lean_rank"] = fit_ranker(tr, Xtr, grade).predict(Xte)
        te["s_lean_reg"] = lgb.LGBMRegressor(objective="regression", **BASE).fit(Xtr, tr["y_rank"]).predict(Xte)
        perm = grade.groupby(tr["date"]).transform(lambda s: pd.Series(rng.permutation(s.values), index=s.index))
        te["s_lean_perm"] = fit_ranker(tr, Xtr, perm).predict(Xte)
        preds.append(te)
        print(f"fold {year}: train {tr['date'].nunique()} dates / {len(tr):,} rows, "
              f"test {te['date'].nunique()} dates ({te[y_raw].notna().groupby(te['date']).any().sum()} labeled)")
    pr = pd.concat(preds)
    pr["s_mom_12_1"] = p.loc[pr.index, "mom_12_1"]
    pr["rv_pct"] = pr.groupby("date")["rv_63"].rank(pct=True)
    lab = pr[pr[y_raw].notna()]

    lines = [f"=== lean ranker | hold {h} | {lab.date.nunique()} labeled test dates, "
             f"{lab.date.min().date()}..{lab.date.max().date()} ==="]
    lines.append(f"{'arm':12s} {'IC':>7s} {'top20 xs':>9s} {'[95% 6mo-block CI]':>22s} {'top10%':>8s} "
                 f"{'top20 vol pct':>13s}")
    by_year = {}
    for arm in ["lean_rank", "lean_reg", "lean_perm", "mom_12_1"]:
        s = f"s_{arm}"
        m = wf.per_date_metrics(lab, s, y_raw)
        ci = wf.block_ci(m["top20"])
        volp = lab.groupby("date", group_keys=False).apply(lambda g: g.nlargest(20, s)["rv_pct"].mean()).mean()
        lines.append(f"{arm:12s} {m['ic'].mean():+7.3f} {m['top20'].mean():+9.2%} "
                     f"[{ci[0]:+7.2%}, {ci[1]:+7.2%}] {m['top10pct'].mean():+8.2%} {volp:13.2f}")
        by_year[arm] = m["top20"].groupby(m.index.year).mean()
    lines.append("\ntop-20 excess over universe, by test year:")
    lines.append(pd.DataFrame(by_year).map(lambda v: f"{v:+.1%}").to_string())

    tr = p[p["date"] < labeled.max() - pd.tseries.offsets.BDay(h + 5)]
    tr = tr[tr[y_raw].notna()]
    m = fit_ranker(tr, R.loc[tr.index], tr["grade"].astype(int))
    imp = pd.Series(m.booster_.feature_importance("gain"), index=LEAN).sort_values(ascending=False)
    lines.append("\nfeature gain share (final lambdarank):\n" + (imp / imp.sum()).round(3).to_string())
    txt = "\n".join(lines)
    print(txt)
    (HERE / f"06_results_h{h}.txt").write_text(txt)
    pr[["date", "ticker", "s_lean_rank", "s_lean_reg", "s_lean_perm"]].to_parquet(
        HERE / f"06_oof_scores_h{h}.parquet", index=False)


if __name__ == "__main__":
    main()
