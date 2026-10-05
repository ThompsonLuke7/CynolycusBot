"""Does the deployed (volatility-tilted) 4H model work inside any subset?

Question (2026-09-29): both deployed 4H labels reward forward MFE, so the model
leans on volatility and low price. Is there a tradeable subset -- low price,
high beta, high ATR, a liquidity band, a regime -- where its picks CONSISTENTLY
beat that subset's own names?

Per decision bar and subset (all bucketing inputs are decision-time values):
  top3_ex     mean 10-session SPY-excess return of the model's top 3 in the
              subset minus the subset's equal-weight mean;
  volctl_ex   same for the top 3 by lagged daily ATR% -- "just buy the most
              volatile names", the tilt a permuted-label model reproduces;
  edge        top3_ex - volctl_ex (does the model beat its own tilt?);
  resid_ex    top 3 by the score residualised on ATR%/beta/price, minus EW;
  ic          within-subset Spearman(score, 10-session excess return);
  top3_net    the top 3's own excess return after an ASSUMED round-trip cost;
  w_*         the same on returns winsorised at the pooled 1st/99th percentile
              (low-price means are carried by a few +100% names);
  hit         share of the top 3 beating the subset median, minus 0.5.
Aggregated with a 2-week-block bootstrap (10-session windows overlap), per
6-month fold, and an MDE (80% power, 5% two-sided) for every null.

Entry = next session open, exit = close 10 sessions later, corporate-action
guarded (panel from run_feature_screen). Gross unless labelled net.

Usage:
    .venv/bin/python -m strategies.momentum_expansion.ablation.run_subset_eval --model momentum
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.momentum_expansion.ablation.bootstrap import bh_fdr, week_block_bootstrap_ci
from strategies.momentum_expansion.ablation.run_feature_screen import FOLD_EDGES, RESULTS_DIR, build_panel

logger = logging.getLogger(__name__)

H = 10
TOP_K = 3
MIN_SUBSET_NAMES = 15
# ASSUMED equity round trip (spread + slippage) by share price. Not measured in
# this repo for equities -- a sensitivity input, reported next to gross.
COST_BANDS = [(0, 5, 0.010), (5, 10, 0.005), (10, 20, 0.003), (20, np.inf, 0.0015)]


def _cost(price: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=price.index)
    for lo, hi, c in COST_BANDS:
        out[(price >= lo) & (price < hi)] = c
    return out


def _xs_q(panel: pd.DataFrame, col: str, n: int = 5) -> pd.Series:
    pct = panel.groupby("timestamp")[col].rank(pct=True)
    return np.minimum(np.ceil(pct * n), n).astype("Int64")


def define_subsets(p: pd.DataFrame) -> dict[str, pd.Series]:
    price = p["close_d"]
    beta_q = _xs_q(p, "beta_spy_60")
    atr_q = _xs_q(p, "daily_atr_pct")
    adv_q = _xs_q(p, "adv20_d")
    s = {"all": pd.Series(True, index=p.index)}
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 50), (50, 150), (150, np.inf)]:
        s[f"price_{lo}-{hi if np.isfinite(hi) else 'up'}"] = (price >= lo) & (price < hi)
    for q in range(1, 6):
        s[f"beta_q{q}"] = beta_q == q
        s[f"atr_q{q}"] = atr_q == q
        s[f"adv_q{q}"] = adv_q == q
    s["lowpx<10_x_beta_q4-5"] = (price < 10) & (beta_q >= 4)
    s["lowpx<10_x_atr_q4-5"] = (price < 10) & (atr_q >= 4)
    s["lowpx<20_x_beta_q5"] = (price < 20) & (beta_q == 5)
    s["beta_q5_x_adv_q4-5"] = (beta_q == 5) & (adv_q >= 4)
    s["atr_q5_x_adv_q4-5"] = (atr_q == 5) & (adv_q >= 4)
    s["spy_uptrend"] = p["regime_spy_trend"] > 0
    s["spy_downtrend"] = p["regime_spy_trend"] <= 0
    s["vix_z>1"] = p["regime_vix_z"] > 1
    return {k: v.fillna(False).astype(bool) for k, v in s.items()}


def _resid_score(g: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([np.ones(len(g))] + [g[c].rank(pct=True).to_numpy()
                                              for c in ("daily_atr_pct", "beta_spy_60", "log_price")])
    y = g["score"].rank(pct=True).to_numpy()
    X = np.nan_to_num(X, nan=0.5)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def per_bar(p: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    sub = p.loc[mask, ["timestamp", "score", "daily_atr_pct", "beta_spy_60", "log_price", "close_d",
                       f"fxret_{H}", "w_y", "cost"]].dropna(subset=["score", f"fxret_{H}", "daily_atr_pct"])
    rows = []
    y = f"fxret_{H}"
    for ts, g in sub.groupby("timestamp", sort=True):
        if len(g) < MIN_SUBSET_NAMES:
            continue
        ew = g[y].mean()
        top = g.nlargest(TOP_K, "score")
        vol = g.nlargest(TOP_K, "daily_atr_pct")
        res = g.iloc[np.argsort(-_resid_score(g))[:TOP_K]]
        med = g[y].median()
        wy = g["w_y"]
        rows.append({
            "timestamp": ts, "n": len(g),
            "w_top3_ex": wy.loc[top.index].mean() - wy.mean(),
            "w_edge": wy.loc[top.index].mean() - wy.loc[vol.index].mean(),
            "hit": (top[y] > med).mean() - 0.5,
            "vol_hit": (vol[y] > med).mean() - 0.5,
            "top3_ex": top[y].mean() - ew,
            "volctl_ex": vol[y].mean() - ew,
            "edge": top[y].mean() - vol[y].mean(),
            "resid_ex": res[y].mean() - ew,
            "ic": g["score"].corr(g[y], method="spearman"),
            "top3_gross": top[y].mean(),
            "top3_net": (top[y] - top["cost"]).mean(),
        })
    return pd.DataFrame(rows)


def _biweek(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True)
    return (t.dt.tz_convert(None).dt.to_period("W").apply(lambda w: w.ordinal // 2)).astype(str)


def summarize_metric(bars: pd.DataFrame, col: str, n_boot: int) -> dict:
    v = bars[col].astype(float)
    b = week_block_bootstrap_ci(v.reset_index(drop=True), _biweek(bars["timestamp"]).reset_index(drop=True),
                                n_boot=n_boot, alpha=0.05)
    se = (b["ci_hi"] - b["ci_lo"]) / (2 * 1.96) if np.isfinite(b["ci_hi"]) else np.nan
    folds = pd.cut(pd.to_datetime(bars["timestamp"], utc=True), FOLD_EDGES, right=False)
    fm = v.groupby(folds.to_numpy(), observed=True).mean()
    return {"mean": b["point"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"], "mde": 2.8 * se,
            "p_approx": float(2 * (1 - _phi(abs(b["point"]) / se))) if se and se > 0 else np.nan,
            "folds_pos": int((fm > 0).sum()), "n_folds": int(len(fm)), "worst_fold": float(fm.min()),
            "best_fold": float(fm.max())}


def _phi(z: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1 + erf(z / sqrt(2)))


def run(model: str, *, n_boot: int = 1000) -> pd.DataFrame:
    p = build_panel(model)
    p["cost"] = _cost(p["close_d"])
    lo, hi = p[f"fxret_{H}"].quantile([0.01, 0.99])
    p["w_y"] = p[f"fxret_{H}"].clip(lo, hi)
    subsets = define_subsets(p)
    rows, per_fold = [], []
    for name, mask in subsets.items():
        bars = per_bar(p, mask)
        if len(bars) < 50:
            logger.info("skip %s: %d bars", name, len(bars))
            continue
        rec = {"subset": name, "n_bars": len(bars), "names_per_bar": float(bars["n"].median())}
        for col in ("top3_ex", "volctl_ex", "edge", "resid_ex", "ic", "top3_gross", "top3_net",
                    "w_top3_ex", "w_edge", "hit", "vol_hit"):
            s = summarize_metric(bars, col, n_boot)
            rec.update({f"{col}_{k}": v for k, v in s.items()})
        rows.append(rec)
        f = bars.assign(fold=pd.cut(pd.to_datetime(bars["timestamp"], utc=True), FOLD_EDGES,
                                    right=False).astype(str))
        per_fold.append(f.groupby("fold")[["top3_ex", "volctl_ex", "edge", "resid_ex", "ic", "top3_net",
                                     "w_top3_ex", "w_edge", "hit"]]
                        .mean().assign(subset=name).reset_index())
        logger.info("%-24s bars=%4d top3_ex=%+.4f w=%+.4f edge=%+.4f w_edge=%+.4f hit=%+.3f ic=%+.3f net=%+.4f",
                    name, len(bars), rec["top3_ex_mean"], rec["w_top3_ex_mean"], rec["edge_mean"],
                    rec["w_edge_mean"], rec["hit_mean"], rec["ic_mean"], rec["top3_net_mean"])
    out = pd.DataFrame(rows)
    out["edge_q"] = bh_fdr(out["edge_p_approx"])
    out["top3_ex_q"] = bh_fdr(out["top3_ex_p_approx"])
    out["w_edge_q"] = bh_fdr(out["w_edge_p_approx"])
    d = RESULTS_DIR / model
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / "subset_eval.csv", index=False)
    pd.concat(per_fold, ignore_index=True).to_csv(d / "subset_eval_by_fold.csv", index=False)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="momentum", choices=["momentum", "htf"])
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    run(a.model, n_boot=a.n_boot)


if __name__ == "__main__":
    main()
