"""Stage 2 -- is the ENTER/PASS decision decidable, and on which instrument?

Thread's Experiment B, with the plan's correction C1 folded in: the option route's
~35%-of-premium round trip is ~5x any measured selection edge, so the ROUTE is a
first-class arm here, not a downstream detail.

Full counterfactual feedback (plan C4): the market is exogenous and history replayable, so
for every candidate we know the outcome of ENTER and of PASS exactly. ENTER/PASS therefore
reduces to cost-sensitive supervised learning on the R-unit outcome -- there is no partial
feedback and so no bandit, and no exploration. That is what the arms below are.

Arms, all on identical rows / features / split:
  incumbent   deployed walk-forward OOF mom_score (and htf_score)
  xgb         XGBoost regression on r_10 in R units
  lgbm        LightGBM, same target
  mlp         small sklearn MLP, same target
  *_null      IDENTICAL model trained on WITHIN-BAR PERMUTED labels. Mandatory:
              25_matched_control_and_depth_null.md section B.1 showed a permuted-label
              model produces +2.1..+7.6pp of FAKE top-k excess via a vol/beta tilt.

Metric: within-bar top-k excess in R units against the bar's own equal-weight mean, with a
decision-day block bootstrap. Plus the R distribution and the tail-capture rate (share of
picks that are a P1 event), because the mean is exactly what hides the tail.

Route arm: MODEL-BASED Black-Scholes reprice of a 30-DTE call on the realized underlying
path, IV held CONSTANT at trailing realized vol, then the MEASURED round-trip spread
charged. Same machinery and the same limits as
research/regime_coverage_2026-09-21/exp_b_dte_reprice.py: it isolates theta+delta, it is
NOT a claim about real fills, and only the equity-vs-option COMPARISON is the output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import panel as P  # noqa: E402
from research.options_lab.pricing import bsm_price  # noqa: E402

DATA = HERE / "data"
SEED = 17
TOP_K = [1, 2, 3, 5, 10]
EVAL_H = 10                 # sessions; matches the modules' own 25-bar (~10 day) horizon
MIN_XS = 200
N_BOOT = 5000
FEATURES = ["mom_score", "htf_score", "regime_spy_trend", "regime_spy_ret_20",
            "regime_vix_z", "regime_vix_high", "breadth_z", "sector_dispersion_z",
            "spy_rv20_z", "risk_appetite_z", "liquidity_stress_z", "credit_risk_z",
            "past_ret_20", "past_ret_60", "past_ret_120", "past_vol_20",
            "log_dollar_vol_20", "dist_252_high", "beta_60", "xs_size"]
TARGET = f"r_{EVAL_H}"

# --- route arm constants, all stated, none fitted -------------------------------
OPT_DTE = 30
OPT_MONEYNESS = 1.00        # ATM; adaptive_delta in momentum_config sits 0.35-0.55 delta
OPT_R, OPT_Q = 0.045, 0.0
# measured round-trip spread on our own option fills: 23.6pp of premium
# (23_rank_depth_and_options.md section 5). Charged once.
OPT_SPREAD_RT = 0.236


# --------------------------------------------------------------------- utilities

def bar_topk_excess(df: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    """Per decision bar: mean target of the top-k by `score_col` minus the bar's own
    equal-weight mean. The bar is its own control, so universe drift, sector mix and
    market direction are all held fixed by construction."""
    g = df.groupby("timestamp", sort=True)
    rank = g[score_col].rank(ascending=False, method="first")
    pick = df[rank <= k]
    top = pick.groupby("timestamp")[TARGET].mean()
    allm = g[TARGET].mean()
    out = pd.DataFrame({"top": top, "bar": allm}).dropna()
    out["excess"] = out["top"] - out["bar"]
    return out


def block_boot(per_bar: pd.Series, rng: np.random.Generator) -> tuple[float, float, float]:
    """Decision-day block bootstrap on the per-bar excess series."""
    v = per_bar.to_numpy(float)
    n = len(v)
    if n < 10:
        return float(np.mean(v)), np.nan, np.nan
    draws = np.array([v[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    obs = float(v.mean())
    p = float(2.0 * min((draws <= 0).mean(), (draws >= 0).mean()))
    return obs, float(np.percentile(draws, 2.5)), min(1.0, p)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30:
        return float("nan")
    xv, yv = x[ok], y[ok]
    var = float(np.var(xv))
    return float(np.cov(xv, yv)[0, 1] / var) if var > 0 else float("nan")


def dist(x: pd.Series) -> dict:
    x = x.dropna()
    if x.empty:
        return {}
    return {"n": int(len(x)), "mean": float(x.mean()), "p1": float(x.quantile(0.01)),
            "p25": float(x.quantile(0.25)), "median": float(x.median()),
            "p75": float(x.quantile(0.75)), "p99": float(x.quantile(0.99)),
            "share_gt_3R": float((x >= 3).mean()), "share_lt_neg1R": float((x <= -1).mean())}


# ------------------------------------------------------------------- route arm

def option_return(under_ret: np.ndarray, iv_ann: np.ndarray, hold_sessions: int) -> np.ndarray:
    """Gross return of a 30-DTE ATM call held `hold_sessions`, repriced on the realized
    underlying return with IV HELD CONSTANT. S is normalised to 1."""
    T0 = OPT_DTE / 365.0
    T1 = max(0.0, (OPT_DTE - hold_sessions * 7.0 / 5.0) / 365.0)
    sig = np.clip(iv_ann, 0.08, 3.0)
    c0 = bsm_price(1.0, OPT_MONEYNESS, T0, OPT_R, OPT_Q, sig, "call")
    c1 = bsm_price(1.0 + under_ret, OPT_MONEYNESS, T1, OPT_R, OPT_Q, sig, "call")
    c0 = np.asarray(c0, float)
    return np.where(c0 > 1e-8, np.asarray(c1, float) / c0 - 1.0, np.nan)


# ------------------------------------------------------------------------- main

def fit_arm(train: pd.DataFrame, evalset: pd.DataFrame, kind: str, permute: bool,
            rng: np.random.Generator) -> np.ndarray:
    import xgboost as xgb
    from lightgbm import LGBMRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    y = train[TARGET].to_numpy(float).copy()
    if permute:
        # permute WITHIN each decision bar: destroys the label's link to the row while
        # leaving the bar's own return distribution untouched.
        idx = np.arange(len(train))
        for _, g in train.groupby("timestamp", sort=False).indices.items():
            pos = idx[g] if isinstance(g, np.ndarray) else np.asarray(g)
            y[pos] = y[rng.permutation(pos)]
    X, Xe = train[FEATURES].to_numpy(float), evalset[FEATURES].to_numpy(float)
    if kind == "xgb":
        m = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=SEED, n_jobs=4, tree_method="hist")
        m.fit(X, y)
    elif kind == "lgbm":
        m = LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                          n_jobs=4, verbose=-1)
        m.fit(X, y)
    elif kind == "mlp":
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=60,
                                       early_stopping=True, random_state=SEED))
        m.fit(X, y)
    else:
        raise ValueError(kind)
    return np.asarray(m.predict(Xe), float)


def main() -> None:
    rng = np.random.default_rng(SEED)
    out: dict = {}
    print("loading panel ...", flush=True)
    pan = P.load(require=[TARGET, "mom_score"], drop_flagged=True)
    pan = pan[pan["xs_size"] >= MIN_XS].dropna(subset=[TARGET])
    tr, va, te, bounds = P.split(pan)
    out["bounds"] = bounds
    print(json.dumps(bounds, indent=2))
    print(f"rows train={len(tr):,} val={len(va):,} test={len(te):,}  "
          f"target={TARGET} (R units, 1R = {P.K_RISK} x ATR{P.ATR_WINDOW}%)", flush=True)
    print("\nTEST IS NOT READ IN THIS SCRIPT. Selection happens on VALIDATION; the test\n"
          "block was consumed by the horizon grid (2024-12-26..2026-05-14) and by\n"
          "confluence (2026-03-11..05-14) -- see 00_preregistration.md section 3.2.\n")

    # ---------------------------------------------------------- arms
    arms: dict[str, pd.Series] = {"incumbent_mom": va["mom_score"],
                                  "incumbent_htf": va["htf_score"]}
    for kind in ("xgb", "lgbm", "mlp"):
        for permute in (False, True):
            name = f"{kind}_null" if permute else kind
            print(f"fitting {name} ...", flush=True)
            arms[name] = pd.Series(fit_arm(tr, va, kind, permute, rng), index=va.index)

    # ---------------------------------------------------------- top-k excess
    print("\n" + "=" * 78 + f"\nA. within-bar top-k excess in R units, hold {EVAL_H}d, "
          "VALIDATION\n" + "=" * 78)
    rows = []
    for name, s in arms.items():
        v = va.assign(_s=s.to_numpy())
        for k in TOP_K:
            pb = bar_topk_excess(v, "_s", k)
            obs, lo, p = block_boot(pb["excess"], rng)
            rows.append(dict(arm=name, k=k, bars=int(len(pb)), excess_R=obs,
                             boot_p2_5=lo, p=p, bar_mean_R=float(pb["bar"].mean())))
    ex = pd.DataFrame(rows)
    out["topk_excess"] = ex.to_dict("records")
    piv = ex.pivot_table(index="arm", columns="k", values="excess_R").round(4)
    print("\nexcess over the bar's own equal-weight mean (R units):")
    print(piv.to_string())
    print("\ntwo-sided block-bootstrap p (decision-day blocks, "
          f"{N_BOOT} draws):")
    print(ex.pivot_table(index="arm", columns="k", values="p").round(4).to_string())

    # ---------------------------------------------------------- real vs null
    print("\n" + "=" * 78 + "\nB. real arm minus its OWN permuted-label null\n" + "=" * 78)
    net = []
    for kind in ("xgb", "lgbm", "mlp"):
        for k in TOP_K:
            r = ex[(ex.arm == kind) & (ex.k == k)]["excess_R"].iloc[0]
            n = ex[(ex.arm == f"{kind}_null") & (ex.k == k)]["excess_R"].iloc[0]
            net.append(dict(arm=kind, k=k, real_R=r, null_R=n, net_R=r - n))
    nd = pd.DataFrame(net)
    out["real_minus_null"] = nd.to_dict("records")
    print(nd.pivot_table(index="arm", columns="k", values="net_R").round(4).to_string())
    print("\nA net at or below zero means the arm's apparent selectivity is the "
          "vol/beta tilt\nthat fitting noise produces, not information.")

    # ---------------------------------------------------------- distributions
    print("\n" + "=" * 78 + f"\nC. R distribution of the picks (k=3) and tail capture\n"
          + "=" * 78)
    dd = {}
    for name, s in arms.items():
        v = va.assign(_s=s.to_numpy())
        rank = v.groupby("timestamp")["_s"].rank(ascending=False, method="first")
        pick = v[rank <= 3]
        d = dist(pick[TARGET])
        d["tail_capture_P1"] = float((pick[f"rmfe_{EVAL_H}"] >= 4.0).mean())
        dd[name] = d
    base_tail = float((va[f"rmfe_{EVAL_H}"] >= 4.0).mean())
    out["pick_distribution"] = dd
    out["universe_tail_rate_P1"] = base_tail
    print(pd.DataFrame(dd).T.round(4).to_string())
    print(f"\nuniverse P1 rate on the same rows: {base_tail:.3%} "
          "(tail_capture_P1 above it = the arm concentrates monsters)")

    # ---------------------------------------------------------- route arm
    print("\n" + "=" * 78 + "\nD. ROUTE: equity vs a MODEL-REPRICED 30-DTE ATM call\n"
          + "=" * 78)
    print("MODEL-BASED. IV = trailing realised vol, held CONSTANT (no vega, no IV "
          "expansion),\nno commission, spread charged once at the measured 23.6pp. "
          "Isolates theta+delta.\nOnly the equity-vs-option COMPARISON is the output, "
          "never the option levels.\n")
    route = []
    for name in ("incumbent_mom", "xgb", "xgb_null"):
        v = va.assign(_s=arms[name].to_numpy())
        rank = v.groupby("timestamp")["_s"].rank(ascending=False, method="first")
        pick = v[rank <= 3].copy()
        und = pick[f"fwdret_{EVAL_H}"].to_numpy(float)
        iv = (pick["past_vol_20"].to_numpy(float) * np.sqrt(252.0))
        gross = option_return(und, iv, EVAL_H)
        netr = gross - OPT_SPREAD_RT
        route.append(dict(arm=name, n=int(len(pick)),
                          equity_mean_pct=float(np.nanmean(und) * 100),
                          equity_mean_R=float(pick[TARGET].mean()),
                          option_gross_pct=float(np.nanmean(gross) * 100),
                          option_net_pct=float(np.nanmean(netr) * 100),
                          option_win=float(np.nanmean(netr > 0)),
                          option_median_net_pct=float(np.nanmedian(netr) * 100),
                          option_p99_pct=float(np.nanpercentile(gross, 99) * 100),
                          # leverage as a REGRESSION slope, not mean(gross/und): the ratio
                          # explodes for underlying moves near zero and its mean is
                          # meaningless.
                          leverage_slope=_slope(und, gross)))
    rt = pd.DataFrame(route)
    out["route"] = rt.to_dict("records")
    print(rt.round(3).to_string(index=False))

    DATA.mkdir(exist_ok=True)
    (DATA / "stage2_decidability.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {DATA / 'stage2_decidability.json'}")


if __name__ == "__main__":
    main()
