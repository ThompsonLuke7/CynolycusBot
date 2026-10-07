"""ML inside the liquid universe: can a model beat plain 12-1 momentum among the 300 most liquid
stocks? (2026-10-04)

Point-in-time universe (02 --pit). The rebalance is monthly, so the label is the forward 21-session
return (next open -> close 21 sessions later), ranked within the training universe on each date.
Features are the 14 stock-level technical features of 06, as within-date percentiles. There are no
fundamentals (dead names have none) and no market-level features.
Walk-forward: one test year at a time, 2019-2026. Training uses every earlier date whose label
window closed at least 5 sessions before the test year.

Arms, fixed before the first run. Each is the equal-weight top 20 within the liquid-300:
  ml300        LightGBM regression trained on liquid-300 rows
  ml1000       LightGBM regression trained on liquid-1000 rows, picking within the liquid-300
  ridge300     ridge regression on the same inputs, liquid-300 (the interpretable baseline)
  leaders_ml   the 60 highest-momentum liquid-300 names (top quintile), re-ranked by the ml1000 score
  perm300      ml300 trained on labels shuffled within date (noise control)
  mom_top300   plain 12-1 momentum, the baseline to beat; spy is the benchmark
Scoring uses 05's simulator (daily marks, 4 monthly phases, 20 bp round trip). The verdict is each
arm's PAIRED active return against mom_top300, with a 126-session block-bootstrap CI.

--label (added after the rank run, so the two extra labels are EXPLORATORY):
  rank   within-date percentile of the forward return (the pre-registered run -> 13_results.txt)
  raw    forward return clipped to [-50%, +100%], minus the date's universe mean. It keeps the SIZE of
         the win, which the rank label throws away (a +200% month and a +15% month can share a rank)
  top    1 if the forward return is in the date's top decile, else 0 (classifier; the score is P(top decile))
Why: on the rank label every ML arm lost to mom_top300 by 16-29%/yr. Momentum's IC across the
liquid-300 is only +0.011, but its top 20 beat the group by 2.2% a month, so the payoff is in the tail.

    .venv/bin/python research/long_horizon_discount_2026-09-25/13_ml_top300.py [--label raw|top]
"""
from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = spec_from_file_location(name, HERE / file)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rb = _load("rb", "05_rotation_backtest.py")
lr = _load("lr", "06_lean_ranker.py")
pb, wf = rb.pb, lr.wf
HOLD, N_TOP, N_LIQ, N_LEADERS = 21, 20, 300, 60
FEATS = lr.LEAN
ARMS = ["mom_top300", "ml300", "ml1000", "ridge300", "leaders_ml", "perm300"]


def make_label(fwd: pd.Series, dates: pd.Series, kind: str) -> pd.Series:
    if kind == "rank":
        return fwd.groupby(dates).rank(pct=True)
    if kind == "raw":
        c = fwd.clip(-0.5, 1.0)
        return c - c.groupby(dates).transform("mean")
    return (fwd.groupby(dates).rank(pct=True) >= 0.9).astype(float).where(fwd.notna())


def fit_gbm(X: pd.DataFrame, y: pd.Series, params: dict, kind: str):
    """Returns a scoring function. `top` uses a classifier and scores by P(top decile)."""
    if kind == "top":
        m = lgb.LGBMClassifier(objective="binary", **params).fit(X, y.astype(int))
        return m, lambda Z: m.predict_proba(Z)[:, 1]
    m = lgb.LGBMRegressor(objective="regression", **params).fit(X, y)
    return m, m.predict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", choices=["rank", "raw", "top"], default="rank")
    kind = ap.parse_args().label
    pb.BARS, pb.OUT = pb.PIT_BARS, pb.PIT_OUT
    p = pd.read_parquet(pb.OUT, columns=["date", "ticker"] + FEATS).reset_index(drop=True)
    O_df, C_df = rb.price_matrices(sorted(set(p["ticker"]) | set(rb.BENCH)),
                                   pb.PIT_OUT.parent / "px_open_pit.parquet", pb.PIT_OUT.parent / "px_close_pit.parquet")
    C_df = C_df.ffill()
    O_df = O_df.fillna(C_df.shift(1))
    O_df["CASH"] = C_df["CASH"] = 1.0
    O_df = O_df[C_df.columns]
    cal, col = C_df.index, {t: i for i, t in enumerate(C_df.columns)}
    O, C = O_df.values, C_df.values
    end = len(cal) - 1

    # label: next open -> close HOLD sessions after the decision (the same convention as 02)
    e = cal.searchsorted(p["date"].values, side="right")
    x = e + HOLD - 1
    ok = x <= end
    j = p["ticker"].map(col).values
    p["fwd"] = np.nan
    p.loc[ok, "fwd"] = C[x[ok], j[ok]] / O[e[ok], j[ok]] - 1
    liq = p["dv_rank"] <= N_LIQ
    R1000 = p.groupby("date")[FEATS].rank(pct=True)
    R300 = p[liq].groupby("date")[FEATS].rank(pct=True)
    y1000 = make_label(p["fwd"], p["date"], kind)
    y300 = make_label(p.loc[liq, "fwd"], p.loc[liq, "date"], kind)

    labeled = pd.DatetimeIndex(sorted(p.loc[p["fwd"].notna(), "date"].unique()))
    rng = np.random.default_rng(wf.SEED)
    p300 = lr.BASE | {"min_data_in_leaf": 150}
    S = pd.DataFrame(index=p.index[liq], columns=["s_ml300", "s_ml1000", "s_ridge300", "s_perm300"], dtype=float)
    for year, tr_d, _ in wf.folds(labeled, HOLD):
        tr = p["date"].isin(tr_d) & p["fwd"].notna()
        te = (p["date"].dt.year == year) & liq
        tr3 = tr & liq
        m300, score300 = fit_gbm(R300.loc[tr3[tr3].index], y300.loc[tr3[tr3].index], p300, kind)
        S.loc[te[te].index, "s_ml300"] = score300(R300.loc[te[te].index])
        _, score1000 = fit_gbm(R1000[tr], y1000[tr], lr.BASE, kind)
        S.loc[te[te].index, "s_ml1000"] = score1000(R1000.loc[te[te].index])
        ridge = Ridge(alpha=10.0).fit(R300.loc[tr3[tr3].index].fillna(0.5) - 0.5, y300.loc[tr3[tr3].index])
        S.loc[te[te].index, "s_ridge300"] = ridge.predict(R300.loc[te[te].index].fillna(0.5) - 0.5)
        perm = y300.loc[tr3[tr3].index].groupby(p.loc[tr3[tr3].index, "date"]).transform(
            lambda s: pd.Series(rng.permutation(s.values), index=s.index))
        S.loc[te[te].index, "s_perm300"] = fit_gbm(R300.loc[tr3[tr3].index], perm, p300, kind)[1](R300.loc[te[te].index])
        print(f"fold {year}: train {tr3.sum():,} liquid-300 rows / {tr.sum():,} liquid-1000 rows, test {te.sum():,}")

    u = p[liq & (p["date"] >= rb.START)].join(S)
    u["s_mom_top300"] = u["mom_12_1"].fillna(-9)
    lead = u.groupby("date")["s_mom_top300"].rank(ascending=False, method="first") <= N_LEADERS
    u["s_leaders_ml"] = u["s_ml1000"].where(lead)
    u["vol_pct"] = u.groupby("date")["rv_63"].rank(pct=True)

    # --- portfolio test: 4 monthly phases, same simulator and costs as 05 ---
    dec = pd.DatetimeIndex(sorted(u["date"].unique()))
    by_date = dict(tuple(u.groupby("date")))
    k = rb.FREQS["monthly"]
    rets = {a: [] for a in ARMS + ["spy"]}
    stats = {a: [] for a in ARMS}
    for ph in range(k):
        ds = dec[ph::k]
        entries = [i for i in (cal.searchsorted(d, side="right") for d in ds) if i <= end]
        ds = ds[:len(entries)]
        b, _ = rb.simulate([{"SPY": 1.0}], entries[:1], O, C, col, end)
        idx = cal[entries[0]:]
        rets["spy"].append(pd.Series(np.diff(np.r_[1.0, b]) / np.r_[1.0, b][:-1], index=idx))
        for a in ARMS:
            ws = []
            for d in ds:
                g = by_date[d]
                names = g[g[f"s_{a}"].notna()].nlargest(N_TOP, f"s_{a}")["ticker"]
                ws.append({t: 1 / len(names) for t in names})
            v, _ = rb.simulate(ws, entries, O, C, col, end)
            stats[a].append(rb.stats(v, b, idx))
            rets[a].append(pd.Series(np.diff(np.r_[1.0, v]) / np.r_[1.0, v][:-1], index=idx))

    lab = u[u["fwd"].notna()]
    mom_top = lab[lab.groupby("date")["s_mom_top300"].rank(ascending=False, method="first") <= N_TOP]
    lines = [f"=== ML within the liquid-{N_LIQ} | point-in-time universe | label: forward {HOLD}-session return, '{kind}' | "
             f"decisions {dec.min().date()}..{dec.max().date()} | monthly, top {N_TOP}, {rb.COST_RT:.1%} round trip ==="]
    lines.append(f"{'arm':11s} {'CAGR med [min, max]':>24s} {'Sharpe':>6s} {'MDD med/worst':>14s} {'vs SPY /yr [95% CI]':>24s} "
                 f"{'vs mom_top300 /yr [95% CI]':>28s} {'IC':>7s} {'top20 xs':>8s} {'vol pct':>7s} {'shared w/ mom':>13s}")
    base = pd.concat(rets["mom_top300"], axis=1)
    spy = pd.concat(rets["spy"], axis=1)
    for a in ARMS:
        df = pd.DataFrame(stats[a])
        r = pd.concat(rets[a], axis=1)
        act = (r - spy).dropna().mean(axis=1)
        lo, hi = rb.block_ci(act)
        pair = (r - base).dropna().mean(axis=1)
        plo, phi = rb.block_ci(pair) if a != "mom_top300" else (0.0, 0.0)
        s = f"s_{a}"
        m = wf.per_date_metrics(lab.rename(columns={"fwd": "y"}), s, "y")
        top = lab[lab.groupby("date")[s].rank(ascending=False, method="first") <= N_TOP]
        shared = top.merge(mom_top[["date", "ticker"]], on=["date", "ticker"]).groupby("date").size().reindex(
            lab["date"].unique(), fill_value=0).mean()
        lines.append(
            f"{a:11s} {df.cagr.median():+7.1%} [{df.cagr.min():+6.1%},{df.cagr.max():+6.1%}] {df.sharpe.median():6.2f} "
            f"{df.mdd.median():+6.1%}/{df.mdd.min():+6.1%} {act.mean() * 252:+7.1%} [{lo:+6.1%},{hi:+6.1%}] "
            f"{pair.mean() * 252:+9.1%} [{plo:+6.1%},{phi:+6.1%}] {m['ic'].mean():+7.3f} {m['top20'].mean():+8.2%} "
            f"{top['vol_pct'].mean():7.2f} {shared:10.1f}/20")
    lines.append(f"spy         {pd.DataFrame(stats['mom_top300']).bcagr.median():+7.1%}"
                 f"{'':17s} {pd.DataFrame(stats['mom_top300']).bsharpe.median():6.2f}")
    yr = pd.DataFrame({a: (1 + rets[a][0]).groupby(rets[a][0].index.year).prod() - 1 for a in ["spy"] + ARMS})
    lines.append("\ncalendar-year return, phase 0:\n" + yr.map(lambda v: f"{v:+.0%}").to_string())
    imp = pd.Series(m300.booster_.feature_importance("gain"), index=FEATS)
    coef = pd.Series(ridge.coef_, index=FEATS)
    lines.append("\nwhat the last-fold models use (ml300 gain share | ridge300 coefficient on the feature's percentile):\n"
                 + pd.DataFrame({"gain_share": (imp / imp.sum()).round(3), "ridge_coef": coef.round(3)})
                 .sort_values("gain_share", ascending=False).to_string())
    txt = "\n".join(lines)
    print(txt)
    (HERE / ("13_results.txt" if kind == "rank" else f"13_results_{kind}.txt")).write_text(txt)


if __name__ == "__main__":
    main()
