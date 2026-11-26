"""E11: adaptive horizon -- first-passage targets vs a fixed 20 days (PLAN.md section 5,
pre-registered 2026-11-25 21:31 ET before running).

Targets on the Nasdaq-100 from the close of t (purge 60 for all, so spans and purges match):
  T1   fixed: 20-session path max drawdown <= -5%
  FP5  (primary) first passage at fixed +-5%, at most 60 sessions: 1 if -5% is touched before +5%
       (closes); 0 if +5% first or neither within 60. Resolves fast in a fast tape, slowly in a calm one.
  FPs  (secondary) barriers at +-1 x RV20(daily) x sqrt(20), vol-scaled, at most 60 sessions.
Models: clipped logistic and LightGBM on base + long-span families (E3 features), walk-forward.
Books: E9's sizing map with cutoffs from inner out-of-fold scores (the E9 follow-up method); E2
execution. Benchmarks on the same span: buy-and-hold, LH vote x live multipliers, 15% vol target.
Decision: FP5 is PREFERRED over T1 if its book beats T1's (Sharpe-difference 95% CI above 0, or
>= 3/4 crash wins with the CI not below 0), per model. Predictability (log-loss vs each target's own
climatology) and days-to-resolution are reported.

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e11_adaptive_horizon.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_general_policy as e2  # noqa: E402
import e3_factor_completeness as e3  # noqa: E402
import e9_regime_model as e9  # noqa: E402
import regime_common as rc  # noqa: E402

SEED = 20261125
PURGE = 60
MAX_H = 60


def first_passage(close: pd.Series, up: np.ndarray, dn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label 1 if the down barrier (fraction dn<0) is touched before up (>0) within MAX_H closes."""
    c = close.to_numpy(dtype=float)
    n = len(c)
    lab = np.full(n, np.nan)
    days = np.full(n, np.nan)
    for i in range(n - MAX_H):
        if not (np.isfinite(up[i]) and np.isfinite(dn[i])):
            continue
        path = c[i + 1:i + 1 + MAX_H] / c[i] - 1
        if np.isnan(path).any():
            continue
        hit_dn = np.flatnonzero(path <= dn[i])
        hit_up = np.flatnonzero(path >= up[i])
        a = hit_dn[0] if len(hit_dn) else MAX_H + 1
        b = hit_up[0] if len(hit_up) else MAX_H + 1
        lab[i] = 1.0 if a < b else 0.0
        days[i] = min(a, b) + 1 if min(a, b) <= MAX_H else np.nan
    return lab, days


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main() -> int:
    warnings.simplefilter("ignore")
    rc.OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    p, panel = rc.latest_panel()
    F, fam, _, _ = e3.build_features("ndx")
    close = p["ndx"].reindex(F.index)
    n = len(F)
    t1 = rc.forward_targets(close.dropna(), 20).reindex(F.index)["fwd_mdd20"]
    y = {"T1": (t1 <= -0.05).astype(float).where(t1.notna())}
    fp5, d5 = first_passage(close, np.full(n, 0.05), np.full(n, -0.05))
    sig = (rc.rv20(close) * np.sqrt(20)).to_numpy()
    fps, ds = first_passage(close, sig, -sig)
    y["FP5"] = pd.Series(fp5, index=F.index)
    y["FPs"] = pd.Series(fps, index=F.index)
    resolution = {"FP5": pd.Series(d5).describe(percentiles=[0.1, 0.5, 0.9]).round(1).to_dict(),
                  "FPs": pd.Series(ds).describe(percentiles=[0.1, 0.5, 0.9]).round(1).to_dict(),
                  "FP5_unresolved_share": float(np.mean(np.isnan(d5[~np.isnan(fp5)]))),
                  "FP5_by_vol_tercile_median_days": {}}
    rvq = pd.Series(sig, index=F.index).rank(pct=True)
    for lo, hi, nm in ((0, 1 / 3, "calm"), (1 / 3, 2 / 3, "mid"), (2 / 3, 1.01, "fast")):
        m = (rvq >= lo) & (rvq < hi) & pd.Series(~np.isnan(d5), index=F.index)
        resolution["FP5_by_vol_tercile_median_days"][nm] = float(np.median(d5[m.to_numpy()]))
    print("resolution", json.dumps(resolution, default=str), flush=True)
    cols = fam["base"] + [c for k in e9.LONG_FAMS for c in fam[k]]
    weights, skill = {}, {}
    for tname, yy in y.items():
        for model in ("lr", "lgb"):
            out = e9.sup_walk_forward(F, yy, cols, model, PURGE, "oof").dropna()
            weights[f"{tname}_{model}_size"] = out["size"]
            weights[f"{tname}_{model}_longflat"] = (out["size"] >= 1.0).astype(float)
            sub, ys, years, ty, _ = e3.span_setup(F, yy, cols)
            sc = out["score"].reindex(sub.index).to_numpy()
            pc = np.full(len(ys), np.nan)
            for yr in ty:
                te = np.where(years == yr)[0]
                if len(te):
                    pc[te] = np.nanmean(ys[:max(te[0] - PURGE, 0)])
            ok = ~np.isnan(sc) & ~np.isnan(ys) & ~np.isnan(pc)
            dec = rc.decade_of(sub.index).to_numpy()
            skill[f"{tname}_{model}"] = {"base_rate": float(np.nanmean(ys)),
                                         **{k: float(v) * 1000 for k, v in pd.Series(ll(ys[ok], pc[ok]) - ll(ys[ok], sc[ok])).groupby(dec[ok]).mean().items()},
                                         "all": float((ll(ys[ok], pc[ok]) - ll(ys[ok], sc[ok])).mean()) * 1000}
            print(tname, model, "skill (millinats)", {k: round(v, 1) for k, v in skill[f'{tname}_{model}'].items()}, flush=True)
    # books on the common span
    start = max(v.index[0] for v in weights.values())
    fr, _ = rc.build_frame("ndx")
    idx = fr.loc[start:].index
    vote = e2.target_weights(fr, p["ndx"]).reindex(idx)
    w = pd.DataFrame({"bh": 1.0, "vote_size": vote["size_regime"], "voltarget15": vote["voltarget15"]}, index=idx)
    for k, v in weights.items():
        w[k] = v.reindex(idx)
    w = w.dropna()
    c = p["ndx"].reindex(w.index)
    r = c.pct_change().fillna(0.0).to_numpy()
    rf = (p["tbill3m"].shift(1).reindex(w.index) / 100 / 252).to_numpy()
    rets = pd.DataFrame({k: e2.run_book(w[k].to_numpy(), r, rf) for k in w.columns}, index=w.index)
    rets["bh"] = r
    crashes = e2.crash_table(c, rets, e2.CRASH_WINDOWS)
    crashes.to_csv(rc.OUT / "e11_crashes.csv", index=False)
    books = {}
    for k in w.columns:
        sigw = w[k].to_numpy()
        pos = np.concatenate([[sigw[0]] * 2, sigw[:-2]])
        books[k] = e2.metrics(rets[k].to_numpy(), rf, pos)
    decisions = {}
    for model in ("lr", "lgb"):
        for alt in ("FP5", "FPs"):
            a, b = f"{alt}_{model}_size", f"T1_{model}_size"
            lo, hi = e2.block_ci_sharpe_diff(rets[a].to_numpy() - rf, rets[b].to_numpy() - rf, rng)
            wins = int((crashes[f"{a}_mdd"] > crashes[f"{b}_mdd"]).sum())
            decisions[f"{alt}_vs_T1_{model}"] = {"sharpe_diff_ci": [lo, hi], "crash_wins": f"{wins}/{len(crashes)}",
                                                 "PREFERRED": bool(lo > 0 or (wins >= 0.75 * len(crashes) and hi >= 0))}
    (rc.OUT / "e11_summary.json").write_text(json.dumps({"panel": panel, "span_start": str(start.date()),
                                                        "resolution": resolution, "skill_millinats": skill,
                                                        "books": books, "decisions": decisions}, indent=2, default=str))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(books).T[["cagr", "sharpe", "mdd", "invested", "turnover_per_yr"]].round(3).to_string())
    print(json.dumps(decisions, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
