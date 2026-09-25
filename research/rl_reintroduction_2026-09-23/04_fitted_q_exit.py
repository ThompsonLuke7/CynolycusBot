"""Stage 4 -- the first RL: Fitted Q-Iteration on HOLD / EXIT.

Why FQI and not DQN/PPO (plan correction C4): the market is exogenous and the replay is
complete, so this is offline optimal stopping with FULL counterfactual feedback. There is no
exploration problem. Two consequences that shape the code:

  * Q(s, EXIT) needs no model at all. Exiting pays r_close[t] - cost, which we OBSERVE.
  * Only Q(s, HOLD) is learned, by bootstrapping the Bellman target
        target(s) = gamma * max( r_exit(s'), Q_{k-1}(s', HOLD) )
    over the next position-day s' of the same entry, terminal states taking r_exit(s').

So FQI here is a genuine RL estimator (it bootstraps through the Bellman operator) of the
same quantity Stage 3's imitation estimates by hindsight regression. Comparing the two is
the point: if bootstrapping adds nothing over imitating the oracle, RL is not earning its
complexity on this problem.

THE CONTROL THAT MATTERS is not a label permutation, it is the OFFSET-ONLY arm: an
identically-trained FQI whose only feature is position age. If the full-feature agent does
not beat it, then everything it learned was a holding horizon, and the honest summary is
"hold N days", not "a learned exit policy". A permuted-feature arm is also run.

TRAIN -> policy; VALIDATION -> grading. The test block is never read.
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
PATHS = DATA / "stage3_paths.parquet"
SEED = 17
COST_RT = 0.002
FIXED_H = [5, 10, 15, 20, 30]
N_ITER = 6
GAMMAS = [1.0, 0.99]
N_BOOT = 5000
MIN_XS = 200

REGIME = ["regime_spy_trend", "regime_spy_ret_20", "regime_vix_z", "regime_vix_high",
          "breadth_z", "sector_dispersion_z", "spy_rv20_z", "risk_appetite_z",
          "liquidity_stress_z", "credit_risk_z"]
FULL_FEATS = ["off", "r_close", "run_max", "run_min", "ret_5", "atr_pct",
              "entry_score"] + REGIME
OFFSET_FEATS = ["off"]


def build_state(paths: pd.DataFrame) -> pd.DataFrame:
    p = paths.sort_values(["eid", "offset"]).copy()
    g = p.groupby("eid", sort=False)["r_close"]
    p["run_max"] = g.cummax()
    p["run_min"] = g.cummin()
    p["ret_5"] = g.transform(lambda s: s - s.shift(5))
    p["off"] = p["offset"]
    p["cost_R"] = COST_RT / (P.K_RISK * p["atr_pct"])
    p["r_exit"] = p["r_close"] - p["cost_R"]
    # next position-day of the same entry
    p["r_exit_next"] = g.shift(-1) - p["cost_R"]
    p["terminal"] = p.groupby("eid", sort=False)["offset"].transform("max").eq(p["offset"])
    return p[p["offset"] >= 1].copy()


def fqi(train: pd.DataFrame, evalset: pd.DataFrame, feats: list[str], gamma: float,
        *, permute: bool, rng: np.random.Generator) -> tuple[np.ndarray, list[float]]:
    """Returns Q(s,HOLD) on `evalset` plus the per-iteration mean training target."""
    import xgboost as xgb

    tr = train.copy()
    if permute:
        # decouple the STATE from the path within each offset. Path rewards, offsets and
        # the marginal feature distribution all survive; only "which state goes with which
        # continuation" is destroyed.
        for _, pos in tr.groupby("off", sort=False).indices.items():
            pos = np.asarray(pos)
            perm = rng.permutation(pos)
            for c in feats:
                if c == "off":
                    continue
                tr.loc[tr.index[pos], c] = tr[c].to_numpy()[perm]

    Xtr = tr[feats].to_numpy(float)
    Xev = evalset[feats].to_numpy(float)
    # bootstrap needs Q_{k-1}(s', HOLD); s' is the NEXT row of the same eid, so build an
    # index map from each row to its successor's position inside `tr`.
    tr = tr.reset_index(drop=True)
    nxt = np.full(len(tr), -1, dtype=int)
    pos_of = {(e, o): i for i, (e, o) in enumerate(zip(tr["eid"], tr["off"]))}
    for i, (e, o) in enumerate(zip(tr["eid"], tr["off"])):
        nxt[i] = pos_of.get((e, o + 1), -1)
    r_exit_next = tr["r_exit_next"].to_numpy(float)
    terminal = tr["terminal"].to_numpy(bool)

    q_hold_tr = np.zeros(len(tr))
    means: list[float] = []
    model = None
    for _ in range(N_ITER):
        nxt_best = np.where(np.isfinite(r_exit_next), r_exit_next, 0.0)
        has_next = nxt >= 0
        cand = np.where(has_next, q_hold_tr[np.clip(nxt, 0, len(tr) - 1)], -np.inf)
        nxt_best = np.maximum(nxt_best, np.where(has_next & ~terminal, cand, -np.inf))
        target = gamma * np.where(np.isfinite(nxt_best), nxt_best, 0.0)
        model = xgb.XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.06,
                                 subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                                 n_jobs=4, tree_method="hist")
        model.fit(Xtr, target)
        q_hold_tr = np.asarray(model.predict(Xtr), float)
        means.append(float(np.mean(target)))
    return np.asarray(model.predict(Xev), float), means


def run_policy(evalset: pd.DataFrame, q_hold: np.ndarray) -> pd.DataFrame:
    """EXIT at the first day exiting is worth at least holding."""
    e = evalset.assign(q_hold=q_hold)
    e["exit_now"] = e["r_exit"] >= e["q_hold"]
    rows = []
    for eid, g in e.groupby("eid", sort=False):
        g = g.sort_values("off")
        hit = g[g["exit_now"]]
        if len(hit):
            r = float(hit["r_exit"].iloc[0]); k = int(hit["off"].iloc[0]); early = True
        else:
            r = float(g["r_exit"].iloc[-1]); k = int(g["off"].iloc[-1]); early = False
        rows.append({"eid": eid, "policy_R": r, "policy_day": k, "early": early})
    return pd.DataFrame(rows)


def boot(v: np.ndarray, rng) -> tuple[float, float, float]:
    v = v[np.isfinite(v)]
    if len(v) < 30:
        return (float(np.mean(v)) if len(v) else np.nan), np.nan, np.nan
    d = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)])
    return float(v.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    rng = np.random.default_rng(SEED)
    out: dict = {}
    if not PATHS.exists():
        raise SystemExit("run 03_exit_oracle.py first -- it builds data/stage3_paths.parquet")
    paths = pd.read_parquet(PATHS)
    st = build_state(paths)

    pan = P.load(require=["mom_score"], drop_flagged=True)
    pan = pan[pan["xs_size"] >= MIN_XS]
    tr_s, va_s, _, bounds = P.split(pan)
    tr_end = tr_s["timestamp"].max()
    va_lo, va_hi = va_s["timestamp"].min(), va_s["timestamp"].max()
    out["bounds"] = bounds
    print(json.dumps(bounds, indent=2))

    tr = st[st["timestamp"] <= tr_end]
    ev = st[(st["timestamp"] >= va_lo) & (st["timestamp"] <= va_hi)]
    print(f"position-days train={len(tr):,} val={len(ev):,} "
          f"(entries {tr['eid'].nunique():,} / {ev['eid'].nunique():,})", flush=True)
    print("TEST IS NEVER READ.\n")

    # benchmarks on the same validation entries, recomputed from the paths
    bench = []
    for eid, g in ev.groupby("eid", sort=False):
        g = g.sort_values("off")
        rec = {"eid": eid, "arm": g["arm"].iloc[0]}
        for h in FIXED_H:
            sel = g[g["off"] <= h]
            rec[f"fixed_{h}"] = float(sel["r_exit"].iloc[-1]) if len(sel) else np.nan
        rec["oracle"] = float(g["r_exit"].max())
        bench.append(rec)
    bd = pd.DataFrame(bench)

    results: dict[str, pd.DataFrame] = {}
    for gamma in GAMMAS:
        for label, feats, perm in (("fqi_full", FULL_FEATS, False),
                                   ("fqi_offset_only", OFFSET_FEATS, False),
                                   ("fqi_perm_state", FULL_FEATS, True)):
            key = f"{label}_g{gamma}"
            print(f"fitting {key} ...", flush=True)
            q, means = fqi(tr, ev, feats, gamma, permute=perm, rng=rng)
            results[key] = run_policy(ev, q).rename(columns={"policy_R": key,
                                                             "policy_day": f"{key}_day",
                                                             "early": f"{key}_early"})
            out.setdefault("target_means", {})[key] = means

    j = bd
    for k, f in results.items():
        j = j.merge(f, on="eid", how="inner")

    print("\n" + "=" * 78 + "\nFQI vs the offset-only control, VALIDATION, mean R\n" + "=" * 78)
    rows = []
    for arm, g in j.groupby("arm"):
        best_fixed_col = max((f"fixed_{h}" for h in FIXED_H),
                             key=lambda c: float(g[c].mean()))
        base = float(g[best_fixed_col].mean())
        rec = {"arm": arm, "n": int(len(g)), "best_fixed": best_fixed_col,
               "best_fixed_R": base, "oracle_R": float(g["oracle"].mean())}
        for k in results:
            mu, lo, hi = boot((g[k] - g[best_fixed_col]).to_numpy(float), rng)
            rec[f"{k}_R"] = float(g[k].mean())
            rec[f"{k}_gain"] = mu
            rec[f"{k}_ci"] = f"[{lo:.3f},{hi:.3f}]"
            rec[f"{k}_day"] = float(g[f"{k}_day"].mean())
        rows.append(rec)
    rd = pd.DataFrame(rows)
    out["results"] = rd.to_dict("records")

    for gamma in GAMMAS:
        print(f"\n-- gamma = {gamma} --")
        cols = ["arm", "n", "best_fixed", "best_fixed_R", "oracle_R"]
        for label in ("fqi_full", "fqi_offset_only", "fqi_perm_state"):
            cols += [f"{label}_g{gamma}_R", f"{label}_g{gamma}_gain",
                     f"{label}_g{gamma}_ci", f"{label}_g{gamma}_day"]
        print(rd[cols].round(3).to_string(index=False))
        print("\n  net over the offset-only control (the number that decides this stage):")
        for _, r in rd.iterrows():
            net = r[f"fqi_full_g{gamma}_gain"] - r[f"fqi_offset_only_g{gamma}_gain"]
            netp = r[f"fqi_full_g{gamma}_gain"] - r[f"fqi_perm_state_g{gamma}_gain"]
            print(f"    {r['arm']:<11} vs offset-only {net:+.3f}R   "
                  f"vs permuted-state {netp:+.3f}R")

    DATA.mkdir(exist_ok=True)
    (DATA / "stage4_fitted_q.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {DATA / 'stage4_fitted_q.json'}")


if __name__ == "__main__":
    main()
