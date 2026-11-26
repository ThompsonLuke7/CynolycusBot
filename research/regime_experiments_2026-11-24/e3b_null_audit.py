"""E3b: was E3's null an artifact of how it was tested? (PLAN.md section 4, pre-registered
2026-11-25 ~00:47 ET before any result here was computed).

Reuses E3's features, spans and walk-forward (e3_factor_completeness.py) and relaxes one choice at
a time:
  * standalone (family alone vs climatology) next to incremental (base+family vs base), plus the
    out-of-sample correlation between the family-only and base-only probabilities ("overlap");
  * six targets: Nasdaq-100 20d <= -5% (E3's), 60d <= -10%, 120d <= -15%, 20d <= -10% (tail);
    S&P 500 20d <= -5% and 60d <= -10% (base features on the S&P itself);
    purge = horizon, permutation block = max(60, horizon);
  * two slow families: rates_slow (share of the last 252 sessions with the curve inverted, 250d
    curve change) and credit_slow (120d Baa spread change);
  * logistic regression with features clipped to the training window's 1st/99th percentiles
    (primary, 100 permutation draws for incremental, 50 for standalone) and LightGBM with its own
    20-draw permutation null.
Decision per target and family: ADDS BEYOND PRICE/VOL if the clipped logistic beats its permuted
95th percentile in >= 2 of the last 3 decades (all decades it has if fewer, minimum 2). MATTERS ON
ITS OWN if the family-alone model beats its permuted 95th percentile over all test years.

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e3b_null_audit.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_factor_completeness as e3  # noqa: E402
import regime_common as rc  # noqa: E402

e3.LGB_PARAMS = {**e3.LGB_PARAMS, "n_jobs": 1}  # one thread per worker process
SEED = 20261125
WORKERS = int(os.environ.get("E3B_WORKERS", 5))
PERMS_INC = int(os.environ.get("E3B_PERMS", 100))
PERMS_ALONE = PERMS_INC // 2
PERMS_LGB = int(os.environ.get("E3B_PERMS_LGB", 20))
TARGETS = {  # name -> (index, horizon, drawdown threshold)
    "T1_ndx_20d_5": ("ndx", 20, -0.05), "T2_ndx_60d_10": ("ndx", 60, -0.10),
    "T3_ndx_120d_15": ("ndx", 120, -0.15), "T4_ndx_20d_10": ("ndx", 20, -0.10),
    "T5_spx_20d_5": ("spx", 20, -0.05), "T6_spx_60d_10": ("spx", 60, -0.10),
}
ONLY = [t for t in os.environ.get("E3B_TARGETS", "").split(",") if t]
FAMILIES = ("vix", "term", "credit", "credit_slow", "rates", "rates_slow", "dollar", "oil",
            "relative", "sectors_breadth", "sectors_corr", "claims")


def features_for(index: str, h: int, thr: float):
    F, fam, _, panel_name = e3.build_features(index)
    p, _ = rc.latest_panel()
    curve = (p["ust10y"] - p["tbill3m"]).reindex(F.index)
    F["rates_inv_share252"] = (curve < 0).astype(float).where(curve.notna()).rolling(252, min_periods=252).mean()
    F["rates_curve_chg250"] = curve.diff(250)
    fam["rates_slow"] = ["rates_inv_share252", "rates_curve_chg250"]
    F["credit_chg120"] = p["baa_10y_spread"].reindex(F.index).diff(120)
    fam["credit_slow"] = ["credit_chg120"]
    close = p[index].dropna()
    mdd = rc.forward_targets(close, h).reindex(F.index)["fwd_mdd20"]  # column name is fixed; horizon is h
    y = (mdd <= thr).astype(float).where(mdd.notna())
    return F, fam, y, panel_name


def climatology(y: np.ndarray, years: np.ndarray, test_years: list[int], purge: int) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    for yr in test_years:
        test = np.where(years == yr)[0]
        if len(test) == 0:
            continue
        tr = np.arange(0, max(test[0] - purge, 0))
        yt = y[tr][~np.isnan(y[tr])]
        if len(yt) >= 500:
            pred[test] = yt.mean()
    return pred


def _job(args):
    X, y, years, test_years, perm_cols, seed, model, purge, block = args
    warnings.simplefilter("ignore")
    Xp = X.copy()
    if seed is not None:
        idx = e3.block_permute_rows(len(X), np.random.default_rng(seed), block)
        Xp[:, perm_cols] = X[idx][:, perm_cols]
    return e3.walk_forward(Xp, y, years, test_years, model=model, purge=purge, winsor=(model == "lr"))


def run_target(tname: str, pool, rng) -> tuple[pd.DataFrame, dict]:
    index, h, thr = TARGETS[tname]
    F, fam, y, panel_name = features_for(index, h, thr)
    dec_series = rc.decade_of(F.index)
    purge, block = h, max(60, h)
    rows, info = [], {"panel": panel_name, "purge": purge, "block": block}
    for name in FAMILIES:
        base_cols, fcols = fam["base"], fam[name]
        cols = base_cols + fcols
        sub, ys, years, test_years, lost = e3.span_setup(F, y, cols)
        dec = dec_series.reindex(sub.index).to_numpy()
        Xa = sub[cols].to_numpy()
        Xb = sub[base_cols].to_numpy()
        Xf = sub[fcols].to_numpy()
        inc_cols = list(range(len(base_cols), len(cols)))
        alone_cols = list(range(len(fcols)))
        common = (years, test_years)
        jobs = [(Xb, ys, *common, [], None, "lr", purge, block), (Xa, ys, *common, [], None, "lr", purge, block),
                (Xf, ys, *common, [], None, "lr", purge, block), (Xb, ys, *common, [], None, "lgb", purge, block),
                (Xa, ys, *common, [], None, "lgb", purge, block)]
        jobs += [(Xa, ys, *common, inc_cols, int(s), "lr", purge, block) for s in rng.integers(0, 2**31, PERMS_INC)]
        jobs += [(Xf, ys, *common, alone_cols, int(s), "lr", purge, block) for s in rng.integers(0, 2**31, PERMS_ALONE)]
        jobs += [(Xa, ys, *common, inc_cols, int(s), "lgb", purge, block) for s in rng.integers(0, 2**31, PERMS_LGB)]
        out = list(pool.map(_job, jobs))
        pb, pa, pf, gb, ga = out[:5]
        inc_null = out[5:5 + PERMS_INC]
        alone_null = out[5 + PERMS_INC:5 + PERMS_INC + PERMS_ALONE]
        lgb_null = out[5 + PERMS_INC + PERMS_ALONE:]
        pc = climatology(ys, years, test_years, purge)
        real_inc = e3.decade_improvement(ys, pb, pa, dec)
        real_alone = e3.decade_improvement(ys, pc, pf, dec)
        real_lgb = e3.decade_improvement(ys, gb, ga, dec)
        n_inc = pd.DataFrame([e3.decade_improvement(ys, pb, q, dec) for q in inc_null])
        n_alone = pd.DataFrame([e3.decade_improvement(ys, pc, q, dec) for q in alone_null])
        n_lgb = pd.DataFrame([e3.decade_improvement(ys, gb, q, dec) for q in lgb_null])
        ok = ~np.isnan(pb) & ~np.isnan(pf) & ~np.isnan(ys)
        overlap = float(np.corrcoef(pb[ok], pf[ok])[0, 1])
        base_skill = e3.decade_improvement(ys, pc, pb, dec)
        for d in real_inc:
            rows.append({"target": tname, "family": name, "decade": d,
                         "inc": real_inc[d], "inc_p95": float(n_inc[d].quantile(0.95)),
                         "alone": real_alone.get(d, np.nan), "alone_p95": float(n_alone[d].quantile(0.95)),
                         "lgb_inc": real_lgb.get(d, np.nan), "lgb_p95": float(n_lgb[d].quantile(0.95)),
                         "base_skill": base_skill.get(d, np.nan), "overlap_corr": overlap,
                         "first_row": str(sub.index[0].date()), "first_test_year": test_years[0],
                         "base_rate": float(np.nanmean(ys))})
        print(tname, name, {k: round(v * 1000, 1) for k, v in real_inc.items()}, "alone",
              {k: round(v * 1000, 1) for k, v in real_alone.items()}, "overlap", round(overlap, 2), flush=True)
    return pd.DataFrame(rows), info


def verdicts(res: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (t, f), g in res.groupby(["target", "family"]):
        g = g.set_index("decade")
        have = [d for d in ("2000s", "2010s", "2020s") if d in g.index]
        need = 2 if len(have) >= 3 else max(2, len(have))
        inc_pass = int(sum(g.loc[d, "inc"] > g.loc[d, "inc_p95"] for d in have))
        lgb_pass = int(sum(g.loc[d, "lgb_inc"] > g.loc[d, "lgb_p95"] for d in have))
        out.append({"target": t, "family": f, "decades": len(have),
                    "inc_passes": inc_pass, "ADDS_BEYOND_PRICE_VOL": inc_pass >= need,
                    "lgb_passes": lgb_pass, "lgb_adds": lgb_pass >= need,
                    "alone_all": g.loc["all", "alone"], "alone_p95_all": g.loc["all", "alone_p95"],
                    "MATTERS_ALONE": bool(g.loc["all", "alone"] > g.loc["all", "alone_p95"]),
                    "inc_all": g.loc["all", "inc"], "base_skill_all": g.loc["all", "base_skill"],
                    "overlap_corr": g["overlap_corr"].iloc[0]})
    return pd.DataFrame(out)


def main() -> int:
    rc.OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    results, infos = [], {}
    with ProcessPoolExecutor(WORKERS) as pool:
        for tname in (ONLY or list(TARGETS)):
            res, info = run_target(tname, pool, rng)
            results.append(res)
            infos[tname] = info
            pd.concat(results).to_csv(rc.OUT / "e3b_results.csv", index=False)  # checkpoint per target
    res = pd.concat(results)
    ver = verdicts(res)
    ver.to_csv(rc.OUT / "e3b_verdicts.csv", index=False)
    (rc.OUT / "e3b_summary.json").write_text(json.dumps(
        {"targets": infos, "perms": {"lr_inc": PERMS_INC, "lr_alone": PERMS_ALONE, "lgb": PERMS_LGB}}, indent=2))
    pd.set_option("display.width", 250)
    print(ver.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
