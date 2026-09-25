"""Walk-forward ML ranker for 63/126-session holds, with the controls that have
caught false positives in this repo before.

Target: the per-date percentile rank of forward excess return vs SPY. This is robust to
the fat right tail. P&L is still measured on raw excess returns.
Folds: test = one calendar year (2019..latest). Train = every decision date whose
label window ends before the test year starts (purge = H sessions + 5 embargo).
Arms:
  all / tech / fund     feature-group ablations (does fundamentals add anything?)
  permuted              the SAME model on labels shuffled within date. This is the
                        noise-trained control. A fake top-k edge from a vol/beta tilt
                        shows up here too (see memory: top-k null control).
  mom_12_1 / hand_rule  non-ML baselines
Metrics per test date, then averaged over dates:
  IC        Spearman(score, fwd excess)
  top20     mean fwd excess of the top 20 names minus the universe mean that date
  top10pct  same, for the top decile
CIs: bootstrap over 6-month blocks of decision dates, because labels overlap for ~6 months.
Portfolio: non-overlapping rebalances every H sessions, equal-weight top 20, net of 20 bp
round-trip. CAGR and max drawdown vs SPY.

    .venv/bin/python research/long_horizon_discount_2026-09-25/03_walkforward.py --hold 126
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PANEL = REPO / "Data" / "research" / "long_horizon" / "panel_weekly.parquet"
OUT = Path(__file__).resolve().parent
SEED = 11
COST = 0.002

TECH = ["dist_sma200", "min_low_vs_sma200_20", "dist_sma50", "sma200_slope_60", "sma50_vs_sma200",
        "dd_52w_high", "up_52w_low", "dd_max_63", "ret_5", "ret_21", "ret_63", "ret_126", "mom_12_1",
        "rs_spy_63", "rv_21", "rv_63", "rv_ratio", "tmo_main", "tmo_min_10", "tmo_minus_sig",
        "days_since_tmo_turn", "vol_trend", "dv_rank", "dist_ema200", "min_low_vs_ema200_20",
        "pct_above_sma200_252", "sma200_slope_252", "max_dd_504", "spy_dist_sma200", "spy_ret_63"]
FUND = ["rev_yoy", "rev_yoy_prev", "rev_accel", "rev_ttm_yoy", "log_rev_ttm", "gm_q", "gm_yoy_chg",
        "opm_q", "opm_yoy_chg", "eps_yoy", "ni_positive", "days_since_filing"]
SURPRISE = ["surprise_last", "surprise_mean4", "days_since_earnings"]
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=400,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
              n_estimators=400, verbose=-1, seed=SEED, deterministic=True, num_threads=8)


def folds(dates: pd.DatetimeIndex, hold: int):
    years = sorted({d.year for d in dates if d.year >= 2019})
    for y in years:
        test = dates[(dates.year == y)]
        cutoff = pd.Timestamp(f"{y}-01-01")
        # label of date d ends ~hold sessions later; purge by trading-calendar distance
        train = dates[dates < cutoff - pd.tseries.offsets.BDay(hold + 5)]
        if len(train) >= 52 and len(test):
            yield y, train, test


def per_date_metrics(df: pd.DataFrame, score: str, y: str) -> pd.DataFrame:
    def f(g):
        g = g.dropna(subset=[score, y])
        if len(g) < 50:
            return pd.Series(dtype=float)
        ic = g[score].rank().corr(g[y].rank())
        top = g.nlargest(20, score)[y].mean()
        dec = g[g[score] >= g[score].quantile(0.9)][y].mean()
        return pd.Series({"ic": ic, "top20": top - g[y].mean(), "top10pct": dec - g[y].mean(),
                          "top20_abs": top, "univ": g[y].mean()})
    return df.groupby("date").apply(f)


def block_ci(m: pd.Series, n=2000):
    blocks = m.groupby((m.index.year * 2 + (m.index.month > 6)))
    keys = list(blocks.groups)
    sums = blocks.sum().values
    cnts = blocks.count().values
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(n):
        idx = rng.integers(0, len(keys), len(keys))
        boots.append(sums[idx].sum() / cnts[idx].sum())
    return np.percentile(boots, [2.5, 97.5])


def portfolio(df: pd.DataFrame, score: str, hold: int, spy_ret: pd.Series) -> dict:
    """Rebalance every `hold` sessions (~hold/5 weeks), equal-weight top 20."""
    dates = sorted(df["date"].unique())
    step = max(1, round(hold / 5))
    rets, bench = [], []
    for d in dates[::step]:
        g = df[df["date"] == d].dropna(subset=[score, f"fwd_ret_{hold}"])
        if len(g) < 50:
            continue
        rets.append(g.nlargest(20, score)[f"fwd_ret_{hold}"].mean() - COST)
        bench.append(spy_ret.get(d, np.nan))
    r, b = np.array(rets), np.array(bench)
    yrs = len(r) * hold / 252
    eq, beq = np.cumprod(1 + r), np.cumprod(1 + np.nan_to_num(b))
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    return {"periods": len(r), "cagr": eq[-1] ** (1 / yrs) - 1, "spy_cagr": beq[-1] ** (1 / yrs) - 1,
            "period_mdd": mdd, "hit_vs_spy": np.mean(r > b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=126)
    args = ap.parse_args()
    h = args.hold
    y_raw, y_rank = f"fwd_xs_{h}", "y_rank"
    p = pd.read_parquet(PANEL)
    p = p[p[y_raw].notna() | (p["date"] >= p["date"].max() - pd.Timedelta(days=10))].copy()
    p["y_rank"] = p.groupby("date")[y_raw].rank(pct=True)
    surprise = [c for c in SURPRISE if c in p.columns and p[c].notna().mean() > 0.2]
    arms = {"all": TECH + FUND + surprise, "tech": TECH, "fund": FUND + surprise,
            "all_no_surprise": TECH + FUND}
    dates = pd.DatetimeIndex(sorted(p.loc[p[y_raw].notna(), "date"].unique()))
    rng = np.random.default_rng(SEED)
    preds = []
    for year, tr_d, te_d in folds(dates, h):
        tr = p[p["date"].isin(tr_d)]
        te = p[p["date"].isin(te_d)].copy()
        for arm, feats in arms.items():
            m = lgb.LGBMRegressor(**PARAMS).fit(tr[feats], tr[y_rank])
            te[f"s_{arm}"] = m.predict(te[feats])
        perm = tr.groupby("date")[y_rank].transform(lambda s: pd.Series(rng.permutation(s.values), index=s.index))
        te["s_permuted"] = lgb.LGBMRegressor(**PARAMS).fit(tr[arms["all"]], perm).predict(te[arms["all"]])
        preds.append(te)
        print(f"fold {year}: train {tr['date'].nunique()} dates / {len(tr):,} rows, test {te['date'].nunique()} dates")
    pr = pd.concat(preds)
    pr["s_mom_12_1"] = pr["mom_12_1"]
    # hand rule from 01: oversold TMO turn while price has held within ~5% of / above the 200 SMA
    pr["s_hand_rule"] = ((pr["min_low_vs_sma200_20"] > -0.05) & (pr["tmo_min_10"] <= -9)
                         & (pr["days_since_tmo_turn"] <= 5)).astype(float) + 1e-6 * pr["mom_12_1"].fillna(0)

    # the user's AMZN shape: persistent uptrend, rising long trend, recent tag of the 200 EMA (+-5%)
    pr["s_compounder_dip"] = ((pr["pct_above_sma200_252"] >= 0.75) & (pr["sma200_slope_252"] > 0)
                              & (pr["min_low_vs_ema200_20"].between(-0.05, 0.05))).astype(float) \
        + 1e-6 * pr["mom_12_1"].fillna(0)
    spy_ret = (pr[f"fwd_ret_{h}"] - pr[y_raw]).groupby(pr["date"]).median()  # SPY over same window

    lines = [f"=== hold {h} sessions | test years {sorted(pr.date.dt.year.unique())} | "
             f"{pr.date.nunique()} test dates, {pr.ticker.nunique()} tickers ==="]
    lines.append(f"{'arm':16s} {'IC':>7s} {'top20 xs':>9s} {'[95% 6mo-block CI]':>22s} {'top10%':>8s} "
                 f"{'CAGR':>7s} {'SPY':>7s} {'MDD':>7s} {'hit':>5s}")
    by_year = {}
    for arm in ["all", "all_no_surprise", "tech", "fund", "permuted", "mom_12_1", "hand_rule", "compounder_dip"]:
        s = f"s_{arm}"
        m = per_date_metrics(pr, s, y_raw)
        ci = block_ci(m["top20"])
        pf = portfolio(pr, s, h, spy_ret)
        lines.append(f"{arm:16s} {m['ic'].mean():+7.3f} {m['top20'].mean():+9.2%} "
                     f"[{ci[0]:+7.2%}, {ci[1]:+7.2%}] {m['top10pct'].mean():+8.2%} "
                     f"{pf['cagr']:+7.1%} {pf['spy_cagr']:+7.1%} {pf['period_mdd']:+7.1%} {pf['hit_vs_spy']:5.0%}")
        by_year[arm] = m["top20"].groupby(m.index.year).mean()
    lines.append("\ntop-20 excess over universe, by test year:")
    lines.append(pd.DataFrame(by_year).map(lambda v: f"{v:+.1%}").to_string())

    # what did the model learn? gain importance on the last fold's 'all' model
    tr = p[p["date"].isin(dates[dates < dates.max() - pd.tseries.offsets.BDay(h + 5)])]
    m = lgb.LGBMRegressor(**PARAMS).fit(tr[arms["all"]], tr[y_rank])
    imp = pd.Series(m.booster_.feature_importance("gain"), index=arms["all"]).sort_values(ascending=False)
    lines.append("\nfeature gain share (final model):\n" + (imp / imp.sum()).head(15).round(3).to_string())
    txt = "\n".join(lines)
    print(txt)
    (OUT / f"03_results_h{h}.txt").write_text(txt)
    pr[["date", "ticker", y_raw, f"fwd_ret_{h}"] + [c for c in pr.columns if c.startswith("s_")]].to_parquet(
        OUT / f"03_oof_scores_h{h}.parquet", index=False)


if __name__ == "__main__":
    main()
