"""E3: are we missing a factor? (PLAN.md, pre-registered 2026-11-24 15:19 ET)

Target: Nasdaq-100 path max drawdown over the next 20 sessions <= -5% (fwd_mdd20, regime_common).
Walk-forward, expanding window, refit every calendar year; a year's test rows are scored by a
model trained on rows whose 20-session label window ends before the year starts (purge = 20).
Model: logistic regression (C=1, StandardScaler fit on the training rows only). Secondary: a
shallow LightGBM (fixed hyperparameters below), real arms only.

Feature families (all trailing, known at the close of t; FRED with its E0 availability lag):
  base  F1 trend + F2 realised vol, NDX/SPX price only (the arm every family is added to)
  vix   F2 implied: log VIX, VIX z, log(VIX / SPX RV20), VIX 5d change               1991+
  term  F2 VIX3M/VIX log ratio and its 5d change                                     2007+
  credit F3 Baa-10y spread z and 20d change                                          1987+
  rates F4 10y-3m curve, its 60d change, 10y 20d change, 3m bill 60d change,
        fed funds 120d change                                                       1987+
  dollar F5 DXY 20d log return and z                                                1987+
  oil   F5 WTI 20d log return and its 252d z                                         2001+
  relative F6 NDX-SPX and RUT-SPX 20d relative return                                1988+
  sectors F6 share of 9 sector SPDRs above SMA50, mean pairwise 60d correlation      2000+
  claims macro: initial claims 4-week average, 13-week log change                     1987+
NFCI is not used (latest vintage, not point-in-time). F7 sentiment/positioning, F8 event calendar
and F9 reaction-to-news have no long history and are E4's.

Each family is evaluated on its OWN span: baseline and baseline+family are fit on exactly the
same rows (where the family's features exist), and the first test year is the first full year
after 4 years of training. Arms: baseline; +each family; all long-span families
(vix, credit, rates, dollar, relative, claims; 1991+ span); leave-one-family-out from that.

Control: for each family, its feature rows block-permuted in 60-session blocks across its span
(100 draws), the whole walk-forward refit. Improvement I = baseline log-loss - arm log-loss
(positive = better), per decade. The permuted I's 95th percentile is the null and the MDE.

Decision (fixed): a family is CRUCIAL AND MISSING if its real I beats its permuted 95th
percentile in at least 2 of the last 3 decades (2000s, 2010s, 2020s; a family whose span starts
later needs every decade it has, minimum 2). A family that helps only in specific episodes is
CONTEXT. Episode audit (descriptive): for every NDX drawdown of 10% or more, each family arm's
mean probability minus the baseline's over the 20 sessions before the price votes flagged (3 of
fast_down, below_sma50, spx_trend, rv), and the baseline's log-loss over the episode.

Follow-ups after the first run (2026-11-24 22:05 ET), labelled, not part of the decision above:
  * E3_WINSOR=1: the logistic model's features clipped to each training window's 1st/99th
    percentiles (fit on training rows only). The first run showed raw levels extrapolated out of
    range (2020 claims, 1990s VIX) wrecking the linear model; this is the next variant. Same rule.
  * sectors_breadth / sectors_corr: the passing "sectors" family split into its two features.
  * The episode audit also reports each family's mean lift over ALL out-of-sample days, so a lift
    before episodes can be told apart from a constant offset.

    .venv/bin/python research/regime_experiments_2026-11-24/e3_factor_completeness.py
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
import regime_common as rc  # noqa: E402

SEED = 20261124
PERMS = int(os.environ.get("E3_PERMS", 100))
WORKERS = int(os.environ.get("E3_WORKERS", 3))
WINSOR = os.environ.get("E3_WINSOR", "0") == "1"
SUFFIX = "_winsor" if WINSOR else ""
BLOCK = 60
PURGE = rc.H
TRAIN_YEARS_MIN = 4
LGB_PARAMS = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, max_depth=3,
                  min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  random_state=SEED, verbose=-1, n_jobs=2)
LONG_SPAN = ("vix", "credit", "rates", "dollar", "relative", "claims")
SECTORS = ("xlk", "xlf", "xle", "xlv", "xly", "xlp", "xli", "xlb", "xlu")


def build_features(target: str = "ndx") -> tuple[pd.DataFrame, dict[str, list[str]], pd.Series, str]:
    """E3 features; ``target`` (E3b) is the index whose price/vol forms the base. E3 = "ndx"."""
    p, name = rc.latest_panel()
    c, spx = p[target], p["spx"]
    lc = np.log(c)
    f = pd.DataFrame(index=p.index)
    fam: dict[str, list[str]] = {}
    # base: F1 trend + F2 realised
    ema20 = c.ewm(span=20, adjust=False).mean()
    f["b_dist_sma50"] = lc - np.log(c.rolling(50, min_periods=50).mean())
    f["b_dist_sma200"] = lc - np.log(c.rolling(200, min_periods=200).mean())
    f["b_ema20_slope5"] = np.log(ema20 / ema20.shift(5))
    f["b_ret20"] = lc.diff(20)
    f["b_ret60"] = lc.diff(60)
    f["b_dd252"] = lc - np.log(c.rolling(252, min_periods=252).max())
    f["b_spx_trend"] = np.log(spx.ewm(span=20, adjust=False).mean() / spx.ewm(span=50, adjust=False).mean())
    rv_n = rc.rv20(c)
    f["b_log_rv20"] = np.log(rv_n)
    f["b_log_rv5_rv20"] = np.log(np.log(c).diff().rolling(5, min_periods=5).std() / rv_n)
    f["b_log_rv60"] = np.log(np.log(c).diff().rolling(60, min_periods=60).std())
    f["b_spx_rv20_z"] = rc.zscore(rc.rv20(spx))
    fam["base"] = [x for x in f.columns if x.startswith("b_")]
    spx_rv_ann = rc.rv20(spx) * np.sqrt(252) * 100
    f["vix_log"] = np.log(p["vix"])
    f["vix_z"] = rc.zscore(p["vix"])
    f["vix_vrp"] = np.log(p["vix"] / spx_rv_ann)
    f["vix_chg5"] = np.log(p["vix"]).diff(5)
    fam["vix"] = ["vix_log", "vix_z", "vix_vrp", "vix_chg5"]
    f["term_log"] = np.log(p["vix3m"] / p["vix"])
    f["term_chg5"] = f["term_log"].diff(5)
    fam["term"] = ["term_log", "term_chg5"]
    f["credit_z"] = rc.zscore(p["baa_10y_spread"])
    f["credit_chg20"] = p["baa_10y_spread"].diff(20)
    fam["credit"] = ["credit_z", "credit_chg20"]
    curve = p["ust10y"] - p["tbill3m"]
    f["rates_curve"] = curve
    f["rates_curve_chg60"] = curve.diff(60)
    f["rates_10y_chg20"] = p["ust10y"].diff(20)
    f["rates_bill_chg60"] = p["tbill3m"].diff(60)
    f["rates_ff_chg120"] = p["fed_funds"].diff(120)
    fam["rates"] = ["rates_curve", "rates_curve_chg60", "rates_10y_chg20", "rates_bill_chg60", "rates_ff_chg120"]
    f["dollar_ret20"] = np.log(p["dxy"]).diff(20)
    f["dollar_z"] = rc.zscore(p["dxy"])
    fam["dollar"] = ["dollar_ret20", "dollar_z"]
    wti = p["wti"].where(p["wti"] > 0)  # 2020-04-20 settled negative; no log of it
    f["oil_ret20"] = np.log(wti).diff(20)
    f["oil_z"] = rc.zscore(p["wti"])  # level z keeps the negative print instead of losing 252 rows
    fam["oil"] = ["oil_ret20", "oil_z"]
    f["rel_ndx_spx20"] = lc.diff(20) - np.log(spx).diff(20)
    f["rel_rut_spx20"] = np.log(p["rut"]).diff(20) - np.log(spx).diff(20)
    fam["relative"] = ["rel_ndx_spx20", "rel_rut_spx20"]
    sec = p[list(SECTORS)]
    above = (sec > sec.rolling(50, min_periods=50).mean()).astype(float).where(sec.rolling(50, min_periods=50).count() == 50)
    f["sec_breadth"] = above.mean(axis=1, skipna=False)
    sret = np.log(sec).diff()
    corr = []
    for i in range(len(SECTORS)):
        for j in range(i + 1, len(SECTORS)):
            corr.append(sret.iloc[:, i].rolling(60, min_periods=60).corr(sret.iloc[:, j]))
    f["sec_avg_corr60"] = pd.concat(corr, axis=1).mean(axis=1, skipna=False)
    fam["sectors"] = ["sec_breadth", "sec_avg_corr60"]
    cl = p["initial_claims"].rolling(20, min_periods=20).mean()
    f["claims_chg13w"] = np.log(cl / cl.shift(65))
    fam["claims"] = ["claims_chg13w"]
    fam["sectors_breadth"] = ["sec_breadth"]  # post-hoc split of "sectors"
    fam["sectors_corr"] = ["sec_avg_corr60"]
    f = f.replace([np.inf, -np.inf], np.nan)
    y = (rc.forward_targets(c.dropna()).reindex(p.index)["fwd_mdd20"] <= -0.05).astype(float).where(
        rc.forward_targets(c.dropna()).reindex(p.index)["fwd_mdd20"].notna())
    keep = c.notna() & (p.index >= "1985-10-01")
    return f[keep], fam, y[keep], name


def walk_forward(X: np.ndarray, y: np.ndarray, years: np.ndarray, test_years: list[int],
                 model: str = "lr", purge: int = PURGE, winsor: bool | None = None,
                 window_years: int | None = None) -> np.ndarray:
    """``window_years`` (E10): train only on the most recent N calendar years; None = expanding."""
    winsor = WINSOR if winsor is None else winsor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    pred = np.full(len(y), np.nan)
    for yr in test_years:
        test = np.where(years == yr)[0]
        if len(test) == 0:
            continue
        tr = np.arange(0, max(test[0] - purge, 0))
        if window_years is not None:
            tr = tr[years[tr] >= yr - window_years]
        tr = tr[~np.isnan(y[tr])]
        if len(tr) < 500 or len(np.unique(y[tr])) < 2:
            continue
        if model == "lr":
            Xtr, Xte = X[tr], X[test]
            if winsor:
                lo, hi = np.percentile(Xtr, 1, axis=0), np.percentile(Xtr, 99, axis=0)
                Xtr, Xte = np.clip(Xtr, lo, hi), np.clip(Xte, lo, hi)
            sc = StandardScaler().fit(Xtr)
            m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), y[tr])
            pred[test] = m.predict_proba(sc.transform(Xte))[:, 1]
        else:
            import lightgbm as lgb
            m = lgb.LGBMClassifier(**LGB_PARAMS).fit(X[tr], y[tr])
            pred[test] = m.predict_proba(X[test])[:, 1]
    return pred


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def decade_improvement(y, p_base, p_arm, decades) -> dict[str, float]:
    out = {}
    ok = ~np.isnan(y) & ~np.isnan(p_base) & ~np.isnan(p_arm)
    for d in np.unique(decades[ok]):
        m = ok & (decades == d)
        out[d] = logloss(y[m], p_base[m]) - logloss(y[m], p_arm[m])
    out["all"] = logloss(y[ok], p_base[ok]) - logloss(y[ok], p_arm[ok])
    return out


def span_setup(F: pd.DataFrame, y: pd.Series, cols: list[str]):
    m = F[cols].notna().all(axis=1) & y.reindex(F.index).notna()
    first, last = F.index[m.to_numpy()][0], F.index[m.to_numpy()][-1]
    sub = F.loc[first:]
    inside = sub.loc[:last]
    lost = int((~inside[cols].notna().all(axis=1)).sum())
    if lost > 0.02 * len(inside):
        raise ValueError(f"{cols[-1]}: {lost} of {len(inside)} rows inside the span lack features")
    sub = sub[sub[cols].notna().all(axis=1)]
    ysub = y.reindex(sub.index).to_numpy()
    years = sub.index.year.to_numpy()
    test_years = [yr for yr in range(first.year + TRAIN_YEARS_MIN + 1, years.max() + 1)]
    return sub, ysub, years, test_years, lost


def block_permute_rows(n: int, rng, block: int = BLOCK) -> np.ndarray:
    starts = np.arange(0, n, block)
    order = rng.permutation(len(starts))
    return np.concatenate([np.arange(starts[k], min(starts[k] + block, n)) for k in order])


def _perm_job(args):
    X, y, years, test_years, perm_cols, seed = args
    rng = np.random.default_rng(seed)
    Xp = X.copy()
    idx = block_permute_rows(len(X), rng)
    Xp[:, perm_cols] = X[idx][:, perm_cols]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return walk_forward(Xp, y, years, test_years)


def evaluate_family(F, y, fam, name, dec_series, pool, rng) -> tuple[dict, pd.DataFrame]:
    base_cols = fam["base"]
    cols = base_cols + fam[name]
    sub, ys, years, test_years, lost = span_setup(F, y, cols)
    dec = dec_series.reindex(sub.index).to_numpy()
    Xb, Xa = sub[base_cols].to_numpy(), sub[cols].to_numpy()
    p_base = walk_forward(Xb, ys, years, test_years)
    p_arm = walk_forward(Xa, ys, years, test_years)
    real = decade_improvement(ys, p_base, p_arm, dec)
    perm_cols = list(range(len(base_cols), len(cols)))
    seeds = rng.integers(0, 2**31, PERMS)
    nulls = list(pool.map(_perm_job, [(Xa, ys, years, test_years, perm_cols, int(s)) for s in seeds]))
    null_imp = pd.DataFrame([decade_improvement(ys, p_base, pn, dec) for pn in nulls])
    lgb_b = walk_forward(Xb, ys, years, test_years, "lgb")
    lgb_a = walk_forward(Xa, ys, years, test_years, "lgb")
    lgb_imp = decade_improvement(ys, lgb_b, lgb_a, dec)
    rows = []
    for d in real:
        q95 = float(null_imp[d].quantile(0.95)) if d in null_imp else np.nan
        rows.append({"family": name, "decade": d, "improvement": real[d], "null_p95_mde": q95,
                     "null_mean": float(null_imp[d].mean()), "beats_p95": real[d] > q95,
                     "placebo_pct": float((null_imp[d] < real[d]).mean() * 100),
                     "lgb_improvement": lgb_imp.get(d, np.nan)})
    res = pd.DataFrame(rows)
    span = {"first_row": str(sub.index[0].date()), "first_test_year": test_years[0], "rows": len(sub),
            "rows_lost_inside_span": lost,
            "base_rate": float(np.nanmean(ys))}
    preds = pd.DataFrame({"y": ys, f"p_base__{name}": p_base, f"p_arm__{name}": p_arm}, index=sub.index)
    return {"span": span, "res": res}, preds


def episode_audit(close: pd.Series, preds: dict[str, pd.DataFrame], votes: pd.DataFrame) -> pd.DataFrame:
    c = close.dropna()
    peak, peak_date, in_ep, eps = c.iloc[0], c.index[0], False, []
    trough, trough_date = None, None
    for d, v in c.items():
        if not in_ep:
            if v > peak:
                peak, peak_date = v, d
            elif v / peak - 1 <= -0.10:
                in_ep, trough, trough_date = True, v, d
        else:
            if v < trough:
                trough, trough_date = v, d
            if v >= peak:
                eps.append((peak_date, trough_date, trough / peak - 1))
                in_ep, peak, peak_date = False, v, d
    if in_ep:
        eps.append((peak_date, trough_date, trough / peak - 1))
    price_flag = votes[["fast_down", "below_sma50", "spx_trend", "rv"]].sum(axis=1, min_count=4) >= 3
    rows = []
    for pk, tr, dd in eps:
        after = price_flag.loc[pk:tr]
        flag = after.index[after.to_numpy()][0] if after.any() else None
        row = {"peak": str(pk.date()), "trough": str(tr.date()), "drawdown": dd,
               "price_flag": str(flag.date()) if flag is not None else None}
        for name, pr in preds.items():
            w = pr.loc[:flag].iloc[-21:-1] if flag is not None else pr.iloc[:0]
            w = w.drop(columns="y")
            if len(w) == 20 and w.notna().all().all():
                row[f"dp_{name}"] = float((w[f"p_arm__{name}"] - w[f"p_base__{name}"]).mean())
            seg = pr.loc[pk:tr].dropna()
            if name == "credit" and len(seg.drop(columns="y")):
                p = np.clip(seg[f"p_base__{name}"], 1e-6, 1 - 1e-6)
                row["base_logloss_episode"] = float(-(seg["y"] * np.log(p) + (1 - seg["y"]) * np.log(1 - p)).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    rc.OUT.mkdir(exist_ok=True)
    F, fam, y, panel_name = build_features()
    dec_series = rc.decade_of(F.index)
    rng = np.random.default_rng(SEED)
    summary = {"panel": panel_name, "perms": PERMS, "families": {}}
    all_res, preds = [], {}
    with ProcessPoolExecutor(WORKERS) as pool:
        for name in [k for k in fam if k != "base"]:
            out, pr = evaluate_family(F, y, fam, name, dec_series, pool, rng)
            res = out["res"]
            all_res.append(res)
            preds[name] = pr
            have = [d for d in ("2000s", "2010s", "2020s") if d in set(res["decade"])]
            passes = int(res[res["decade"].isin(have)]["beats_p95"].sum())
            need = 2 if len(have) >= 3 else max(2, len(have))
            summary["families"][name] = {**out["span"], "decades_judged": have, "passes": passes,
                                         "needed": need, "CRUCIAL_MISSING": passes >= need,
                                         "all_years_improvement": float(res.loc[res["decade"] == "all", "improvement"].iloc[0])}
            print(name, summary["families"][name], flush=True)
            print(res.round(5).to_string(index=False), flush=True)
    # all long-span families together and leave-one-family-out (real arms, LR + LGB)
    long_cols = fam["base"] + [c for k in LONG_SPAN for c in fam[k]]
    sub, ys, years, test_years, _ = span_setup(F, y, long_cols)
    dec = dec_series.reindex(sub.index).to_numpy()
    p_base = walk_forward(sub[fam["base"]].to_numpy(), ys, years, test_years)
    combos = {"all_long": long_cols}
    for k in LONG_SPAN:
        combos[f"all_minus_{k}"] = [c for c in long_cols if c not in fam[k]]
    rows = []
    for label, cols in combos.items():
        p = walk_forward(sub[cols].to_numpy(), ys, years, test_years)
        rows.append({"arm": label, **decade_improvement(ys, p_base, p, dec)})
        if label == "all_long":
            pl_b = walk_forward(sub[fam["base"]].to_numpy(), ys, years, test_years, "lgb")
            pl_a = walk_forward(sub[cols].to_numpy(), ys, years, test_years, "lgb")
            rows.append({"arm": "all_long_lgb", **decade_improvement(ys, pl_b, pl_a, dec)})
    combo = pd.DataFrame(rows)
    combo.to_csv(rc.OUT / f"e3_combined_arms{SUFFIX}.csv", index=False)
    summary["mean_lift_all_oos_days"] = {
        k: float((pr[f"p_arm__{k}"] - pr[f"p_base__{k}"]).mean()) for k, pr in preds.items()}
    summary["combined_span"] = {"first_row": str(sub.index[0].date()), "first_test_year": test_years[0]}
    ok = ~np.isnan(p_base) & ~np.isnan(ys)
    summary["baseline_logloss_vs_climatology"] = {
        "baseline": logloss(ys[ok], p_base[ok]),
        "base_rate_only": logloss(ys[ok], np.full(ok.sum(), ys[ok].mean()))}
    pd.concat(all_res).to_csv(rc.OUT / f"e3_family_tests{SUFFIX}.csv", index=False)
    v = rc.lh_votes(rc.latest_panel()[0], "ndx")
    audit = episode_audit(rc.latest_panel()[0]["ndx"], preds, v)
    audit.to_csv(rc.OUT / f"e3_episode_audit{SUFFIX}.csv", index=False)
    summary["winsorized"] = WINSOR
    (rc.OUT / f"e3_summary{SUFFIX}.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.set_option("display.width", 250)
    print(combo.round(5).to_string(index=False))
    print(audit.round(3).to_string(index=False))
    print(json.dumps({k: (v["passes"], v["needed"], v["CRUCIAL_MISSING"]) for k, v in summary["families"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
