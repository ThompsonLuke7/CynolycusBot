"""Stage 6 -- sizing. First the reduction, then the only part that is actually new.

THE REDUCTION. Under full counterfactual feedback (plan C4) and a reward LINEAR in size,
sizing is not a new learning problem. Expected value of a size policy is sum_i m(s_i) * R_i,
which is maximised pointwise by m = max wherever E[R|s] > 0 and m = 0 elsewhere -- i.e.
exactly Stage 2's selection problem restated. Stage 2 answered it: on this state, nothing
fitted beats its own permuted-label null. So a learned linear-objective sizer inherits that
null and there is nothing further to test.

WHAT IS NEW is the NON-LINEAR objective. Log growth, Kelly, or any drawdown-constrained
objective depends on the predictive DISTRIBUTION, not only its mean: f* ~ mean / variance.
That makes DISPERSION prediction the live question -- and dispersion is the one thing this
repo has already shown to be highly predictable (forward 20d realised vol, OOS R^2 = 0.742,
`research/regime_coverage_2026-09-21/forward_rv_predictability.py`) while direction is not
(R^2 = 0.003).

So this script measures three things on the incumbent's picks, validation only:
  A  is R-unit DISPERSION predictable from the state, against a permuted-state null?
  B  what is the realised Kelly fraction by score decile, and is it monotone?
  C  does a dispersion-scaled size policy beat flat size in LOG GROWTH, against a
     permuted-score null?
No RL. The kill criteria from Stage 3 and Stage 4 forbid another optimiser on these features.
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

DATA = HERE / "data"
SEED = 17
MIN_XS = 200
TOP_K = 3
EVAL_H = 10
N_BOOT = 5000
FEATURES = ["mom_score", "htf_score", "regime_spy_trend", "regime_spy_ret_20",
            "regime_vix_z", "regime_vix_high", "breadth_z", "sector_dispersion_z",
            "spy_rv20_z", "risk_appetite_z", "liquidity_stress_z", "credit_risk_z",
            "past_ret_20", "past_ret_60", "past_ret_120", "past_vol_20",
            "log_dollar_vol_20", "dist_252_high", "beta_60", "xs_size"]
TARGET = f"r_{EVAL_H}"
# fixed-fraction grid: fraction of capital risked per 1R
F_GRID = np.round(np.arange(0.02, 0.62, 0.02), 4)


def log_growth(r: np.ndarray, f: float) -> float:
    """Mean log growth per trade at fixed fraction f of capital risked per 1R."""
    g = 1.0 + f * r
    g = np.where(g <= 1e-6, 1e-6, g)      # a wipe-out is bounded, not infinite
    return float(np.mean(np.log(g)))


def kelly_f(r: np.ndarray) -> float:
    vals = [log_growth(r, f) for f in F_GRID]
    return float(F_GRID[int(np.argmax(vals))])


def boot(v: np.ndarray, rng) -> tuple[float, float, float]:
    v = v[np.isfinite(v)]
    if len(v) < 30:
        return (float(np.mean(v)) if len(v) else np.nan), np.nan, np.nan
    d = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)])
    return float(v.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    import xgboost as xgb
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(SEED)
    out: dict = {}
    pan = P.load(require=[TARGET, "mom_score"], drop_flagged=True)
    pan = pan[pan["xs_size"] >= MIN_XS].dropna(subset=[TARGET])
    tr, va, _, bounds = P.split(pan)
    out["bounds"] = bounds
    print(json.dumps(bounds, indent=2))
    print("TEST IS NEVER READ.\n")

    print("=" * 78 + "\n0. the reduction\n" + "=" * 78)
    print("With full counterfactual feedback and a reward linear in size, the optimal\n"
          "discrete-size policy is max-size wherever E[R|s] > 0 -- Stage 2's selection\n"
          "problem restated, which came back null for every fitted arm. A learned\n"
          "LINEAR-objective sizer therefore has nothing left to test, and only the\n"
          "NON-LINEAR (log-growth) objective introduces a genuinely new quantity:\n"
          "the predictive DISPERSION.\n")

    # ---------------------------------------------------------------- A
    print("=" * 78 + "\nA. is R-unit DISPERSION predictable from the state?\n" + "=" * 78)
    ytr = np.abs(tr[TARGET].to_numpy(float))
    yva = np.abs(va[TARGET].to_numpy(float))
    Xtr, Xva = tr[FEATURES].to_numpy(float), va[FEATURES].to_numpy(float)
    rows = []
    for permute in (False, True):
        y = ytr.copy()
        if permute:
            for _, pos in tr.groupby("timestamp", sort=False).indices.items():
                pos = np.asarray(pos)
                y[pos] = y[rng.permutation(pos)]
        m = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                             n_jobs=4, tree_method="hist")
        m.fit(Xtr, y)
        pred = m.predict(Xva)
        rows.append(dict(arm="permuted_label" if permute else "real",
                         oos_r2=float(r2_score(yva, pred)),
                         spearman=float(pd.Series(pred).corr(pd.Series(yva), method="spearman"))))
        if not permute:
            va = va.assign(pred_disp=pred)
    ad = pd.DataFrame(rows)
    out["dispersion"] = ad.to_dict("records")
    print(ad.round(4).to_string(index=False))
    print("\nfor reference: the SAME features on the SIGNED target (Stage 2) produced a\n"
          "top-k excess that did not beat its own null, and `forward_rv_predictability.py`\n"
          "measured forward 20d realised vol at OOS R^2 = 0.742 vs 0.003 for forward return.")

    # ---------------------------------------------------------------- B
    print("\n" + "=" * 78 + "\nB. realised Kelly fraction by incumbent score decile "
          "(validation)\n" + "=" * 78)
    va = va.copy()
    va["dec"] = va.groupby("timestamp")["mom_score"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    rowsb = []
    for d, g in va.groupby("dec"):
        r = g[TARGET].to_numpy(float)
        r = r[np.isfinite(r)]
        rowsb.append(dict(decile=int(d) + 1, n=len(r), mean_R=float(r.mean()),
                          sd_R=float(r.std()), kelly_f=kelly_f(r),
                          growth_at_kelly=log_growth(r, kelly_f(r)),
                          share_gt_3R=float((r >= 3).mean())))
    bd = pd.DataFrame(rowsb)
    out["kelly_by_decile"] = bd.to_dict("records")
    print(bd.round(4).to_string(index=False))

    # ---------------------------------------------------------------- C
    print("\n" + "=" * 78 + "\nC. dispersion-scaled size vs flat size, in LOG GROWTH\n"
          + "=" * 78)
    r3 = va.groupby("timestamp")["mom_score"].rank(ascending=False, method="first")
    pick = va[r3 <= TOP_K].dropna(subset=[TARGET, "pred_disp"]).copy()
    r = pick[TARGET].to_numpy(float)
    f_flat = kelly_f(r)
    print(f"picks n={len(pick):,}  flat Kelly f*={f_flat:.2f}  "
          f"log growth {log_growth(r, f_flat):.5f}/trade")

    rowsc = []
    for name, scaler in (("flat", np.ones(len(pick))),
                         ("inverse_pred_dispersion",
                          1.0 / np.clip(pick["pred_disp"].to_numpy(float), 1e-3, None)),
                         ("inverse_realised_atr",
                          1.0 / np.clip(pick["atr_pct"].to_numpy(float), 1e-3, None)),
                         ("permuted_dispersion",
                          1.0 / np.clip(rng.permutation(pick["pred_disp"].to_numpy(float)),
                                        1e-3, None))):
        w = np.asarray(scaler, float)
        w = w / np.mean(w)                        # same AVERAGE size as flat
        best = max(((log_growth(r * w, f), f) for f in F_GRID), key=lambda t: t[0])
        rowsc.append(dict(policy=name, f_star=best[1], log_growth=best[0],
                          mean_sized_R=float(np.mean(r * w)),
                          sd_sized_R=float(np.std(r * w)),
                          worst_sized_R=float(np.min(r * w))))
    cd = pd.DataFrame(rowsc)
    out["sizing_policies"] = cd.to_dict("records")
    print()
    print(cd.round(5).to_string(index=False))
    gain = (cd.loc[cd.policy == "inverse_pred_dispersion", "log_growth"].iloc[0]
            - cd.loc[cd.policy == "flat", "log_growth"].iloc[0])
    gnull = (cd.loc[cd.policy == "permuted_dispersion", "log_growth"].iloc[0]
             - cd.loc[cd.policy == "flat", "log_growth"].iloc[0])
    out["log_growth_gain"] = float(gain)
    out["log_growth_gain_null"] = float(gnull)
    out["log_growth_net_of_null"] = float(gain - gnull)
    print(f"\ninverse-dispersion vs flat        {gain:+.5f} log-growth/trade")
    print(f"permuted-dispersion vs flat       {gnull:+.5f}")
    print(f"NET OF NULL                       {gain - gnull:+.5f}")
    print("\nSizes are normalised to the same AVERAGE size, so this is a shape comparison,\n"
          "not a leverage comparison. A gain the permuted arm reproduces is the\n"
          "mechanical effect of down-weighting volatile rows, not a learned sizer.")

    DATA.mkdir(exist_ok=True)
    (DATA / "stage6_sizing.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {DATA / 'stage6_sizing.json'}")


if __name__ == "__main__":
    main()
