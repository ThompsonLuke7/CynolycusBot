"""Stage 5 -- entry timing: WAIT / ENTER / REJECT on a candidate the score already likes.

The thread's V1, ported from the design that already exists in
`strategies/spy_intraday/Policy/Execution_Agent/` (direction-gated, one-shot per event
window, reward measured against a deterministic enter-immediately baseline).

Scope, fixed by what came before:
  * Stage 2 ruled out improving WHICH name to take on this state representation. This asks
    only WHEN, with the pick set held to the incumbent's certified top-3 ordering.
  * Direction is immutable: every candidate is a long, inherited from the score.
  * Once ENTER fires, a deterministic exit takes over (hold 30 sessions -- the best fixed
    policy Stage 3 measured), so this isolates the entry decision alone.

Reward, exactly the Execution_Agent shape (env.py:59-61):
      reward = R_agent - R_enter_immediately          (zero on WAIT steps)
so REJECT has a real opportunity cost and "never trade" cannot win. Everything is in R units
with 1R fixed at entry-day-0 ATR for ALL arms, so a later entry cannot flatter itself by
being measured against a smaller risk unit.

Action space and window:
      A = {WAIT, ENTER, REJECT}, evaluated over a W-session entry window after the signal.
      WAIT advances one session; ENTER locks that session's open; REJECT terminates at 0.

THE CONTROL THAT DECIDES THIS STAGE is not a permutation but the set of DETERMINISTIC timing
rules on the same candidates: enter-immediately (the baseline), enter-on-day-k for each k,
and enter-on-first-pullback. A learned policy that cannot beat the best of those has learned
a fixed delay, not a timing rule. A permuted-state arm is also run, after Stage 3 showed a
noise twin reproducing a confident-looking gain.
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

BARS_1D = REPO / "Data/shared/bars/1d"
DATA = HERE / "data"
ENTRY_PATHS = DATA / "stage5_entry_paths.parquet"
SEED = 17
TOP_K = 3
WINDOW = 5          # sessions in which the entry decision may be made
HOLD = 30           # deterministic post-entry hold, = Stage 3's best fixed policy
COST_RT = 0.002
MIN_XS = 200
N_BOOT = 5000

REGIME = ["regime_spy_trend", "regime_spy_ret_20", "regime_vix_z", "regime_vix_high",
          "breadth_z", "sector_dispersion_z", "spy_rv20_z", "risk_appetite_z",
          "liquidity_stress_z", "credit_risk_z"]
STATE = ["wait_days", "ret_since_signal", "mfe_since_signal", "mae_since_signal",
         "close_vs_signal_high", "atr_pct", "entry_score", "rank"] + REGIME


def build_entry_paths(picks: pd.DataFrame) -> pd.DataFrame:
    """For each candidate: what entering on session d (d = 0..WINDOW-1) would have returned.

    All returns are divided by the SAME 1R -- the ATR known at the signal -- so a delayed
    entry is not rewarded for having a different denominator.
    """
    rows = []
    tick = sorted(picks["ticker"].unique())
    for i, t in enumerate(tick, 1):
        path = BARS_1D / f"{t}.parquet"
        if not path.exists():
            continue
        d = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
        d["session_date"] = (pd.to_datetime(d["timestamp"], utc=True)
                             .dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None))
        d = d.sort_values("session_date").reset_index(drop=True)
        pos = pd.Series(np.arange(len(d)), index=d["session_date"])
        o, h, lo, c = (d[x].to_numpy(float) for x in ("open", "high", "low", "close"))
        for r in picks[picks["ticker"] == t].itertuples():
            j = pos.get(r.entry_session)
            if j is None:
                continue
            unit = P.K_RISK * r.atr_pct
            if not np.isfinite(unit) or unit < P.K_RISK * P.MIN_ATR_PCT:
                continue
            p0 = o[j]
            if not np.isfinite(p0) or p0 <= 0:
                continue
            for dd in range(WINDOW):
                k = j + dd
                ex = k + HOLD
                if ex >= len(d) or not np.isfinite(o[k]) or o[k] <= 0:
                    continue
                # state observable at the OPEN of session k (uses sessions j..k-1 only)
                prior = slice(j, k) if k > j else slice(j, j + 1)
                rows.append({
                    "eid": f"{t}|{r.entry_session.date()}", "ticker": t,
                    "timestamp": r.timestamp, "signal_session": r.entry_session,
                    "wait_days": dd, "atr_pct": r.atr_pct,
                    "entry_score": r.entry_score, "rank": r.rank,
                    "entry_px_R": (o[k] / p0 - 1.0) / unit,
                    "ret_since_signal": (c[k - 1] / p0 - 1.0) / unit if k > j else 0.0,
                    "mfe_since_signal": (np.nanmax(h[prior]) / p0 - 1.0) / unit if k > j else 0.0,
                    "mae_since_signal": (np.nanmin(lo[prior]) / p0 - 1.0) / unit if k > j else 0.0,
                    "close_vs_signal_high": ((c[k - 1] / np.nanmax(h[prior]) - 1.0) / unit
                                             if k > j and np.nanmax(h[prior]) > 0 else 0.0),
                    # outcome of ENTERING here: hold HOLD sessions from this open
                    "outcome_R": (c[ex] / o[k] - 1.0) / unit - COST_RT / unit,
                })
        if i % 200 == 0:
            print(f"  entry paths {i}/{len(tick)}", flush=True)
    e = pd.DataFrame(rows)
    reg = picks.drop_duplicates(["ticker", "entry_session"])[
        ["ticker", "entry_session"] + REGIME].rename(columns={"entry_session": "signal_session"})
    e = e.merge(reg, on=["ticker", "signal_session"], how="left")
    DATA.mkdir(exist_ok=True)
    e.to_parquet(ENTRY_PATHS, index=False)
    return e


def boot(v: np.ndarray, rng) -> tuple[float, float, float]:
    v = v[np.isfinite(v)]
    if len(v) < 30:
        return (float(np.mean(v)) if len(v) else np.nan), np.nan, np.nan
    d = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)])
    return float(v.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def learned_policy(tr: pd.DataFrame, ev: pd.DataFrame, *, permute: bool,
                   rng: np.random.Generator) -> pd.DataFrame:
    """Predict the ADVANTAGE of entering here over entering immediately; act on its sign.

    Because feedback is full (plan C4) the advantage is observable in training, so this is
    cost-sensitive regression, not RL -- which is the honest algorithm for V1.
    """
    import xgboost as xgb

    t = tr.copy()
    if permute:
        for _, pos in t.groupby("wait_days", sort=False).indices.items():
            pos = np.asarray(pos)
            perm = rng.permutation(pos)
            for c in STATE:
                if c == "wait_days":
                    continue
                t.loc[t.index[pos], c] = t[c].to_numpy()[perm]
    m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, random_state=SEED, n_jobs=4,
                         tree_method="hist")
    m.fit(t[STATE].to_numpy(float), t["advantage"].to_numpy(float))
    e = ev.assign(pred=m.predict(ev[STATE].to_numpy(float)))
    rows = []
    for eid, g in e.groupby("eid", sort=False):
        g = g.sort_values("wait_days")
        # walk the window: ENTER at the first day predicted better than entering now-vs-
        # baseline, REJECT if the best remaining prediction is negative at the last step.
        taken = None
        for _, row in g.iterrows():
            if row["pred"] >= 0:
                taken = row
                break
        if taken is None:
            rows.append({"eid": eid, "action": "REJECT", "policy_R": 0.0, "wait": np.nan})
        else:
            rows.append({"eid": eid, "action": "ENTER", "policy_R": float(taken["outcome_R"]),
                         "wait": float(taken["wait_days"])})
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    out: dict = {}
    pan = P.load(require=["mom_score", "r_10"], drop_flagged=True)
    pan = pan[pan["xs_size"] >= MIN_XS]
    tr_s, va_s, _, bounds = P.split(pan)
    tr_end = tr_s["timestamp"].max()
    va_lo, va_hi = va_s["timestamp"].min(), va_s["timestamp"].max()
    out["bounds"] = bounds
    print(json.dumps(bounds, indent=2))

    trval = pan[pan["timestamp"] <= va_hi].copy()
    trval["rank"] = trval.groupby("timestamp")["mom_score"].rank(ascending=False, method="first")
    picks = trval[trval["rank"] <= TOP_K].copy()
    picks["entry_score"] = picks["mom_score"]
    picks = picks.dropna(subset=["entry_session", "atr_pct"])
    # The panel has two 4H bars per day and both map to the same next-session open, so a
    # name picked at both bars would appear twice for ONE tradeable position. Live opens one
    # position per name per day: keep the better-ranked bar.
    picks = (picks.sort_values(["ticker", "entry_session", "rank"])
             .drop_duplicates(["ticker", "entry_session"], keep="first"))
    print(f"candidates {len(picks):,} over {picks['ticker'].nunique()} tickers "
          f"(incumbent mom_score top-{TOP_K}); TEST IS NEVER READ.", flush=True)

    if ENTRY_PATHS.exists():
        print(f"reading cached {ENTRY_PATHS}", flush=True)
        e = pd.read_parquet(ENTRY_PATHS)
    else:
        e = build_entry_paths(picks)
    e = e.sort_values(["eid", "wait_days"]).drop_duplicates(["eid", "wait_days"], keep="first")
    print(f"entry-decision rows {len(e):,} over {e['eid'].nunique():,} candidates", flush=True)

    # baseline = enter immediately (wait_days == 0)
    base = e[e["wait_days"] == 0].set_index("eid")["outcome_R"]
    e = e[e["eid"].isin(base.index)].copy()
    e["baseline_R"] = e["eid"].map(base)
    e["advantage"] = e["outcome_R"] - e["baseline_R"]

    tr = e[e["timestamp"] <= tr_end]
    ev = e[(e["timestamp"] >= va_lo) & (e["timestamp"] <= va_hi)]
    print(f"rows train={len(tr):,} val={len(ev):,} "
          f"(candidates {tr['eid'].nunique():,} / {ev['eid'].nunique():,})\n", flush=True)

    print("=" * 78 + "\nA. DETERMINISTIC timing rules on the same candidates (validation)\n"
          + "=" * 78)
    ev_base = ev[ev["wait_days"] == 0]
    det = []
    for d in range(WINDOW):
        g = ev[ev["wait_days"] == d]
        mu, lo, hi = boot((g["outcome_R"] - g["baseline_R"]).to_numpy(float), rng)
        det.append(dict(rule=f"enter_day_{d}", n=int(len(g)), mean_R=float(g["outcome_R"].mean()),
                        vs_baseline_R=mu, ci_lo=lo, ci_hi=hi))
    # enter on the first session that closes below the signal open (a pullback rule)
    pull = (ev[ev["ret_since_signal"] < 0].sort_values(["eid", "wait_days"])
            .groupby("eid", as_index=False).first())
    miss = set(ev["eid"].unique()) - set(pull["eid"])
    pull_R = pd.concat([pull["outcome_R"], pd.Series([0.0] * len(miss))])
    det.append(dict(rule="first_pullback_else_reject", n=int(len(pull_R)),
                    mean_R=float(pull_R.mean()),
                    vs_baseline_R=float(pull_R.mean() - ev_base["outcome_R"].mean()),
                    ci_lo=np.nan, ci_hi=np.nan))
    dd = pd.DataFrame(det)
    out["deterministic"] = dd.to_dict("records")
    print(dd.round(4).to_string(index=False))
    best_det = dd.loc[dd["vs_baseline_R"].idxmax()]
    print(f"\nbest deterministic rule: {best_det['rule']} "
          f"({best_det['vs_baseline_R']:+.4f}R vs enter-immediately)")

    print("\n" + "=" * 78 + "\nB. LEARNED policy vs its permuted-state twin\n" + "=" * 78)
    real = learned_policy(tr, ev, permute=False, rng=rng)
    null = learned_policy(tr, ev, permute=True, rng=rng)
    base_v = ev_base.set_index("eid")["outcome_R"]
    rows = []
    for name, f in (("learned", real), ("permuted_state", null)):
        f = f.assign(baseline_R=f["eid"].map(base_v)).dropna(subset=["baseline_R"])
        mu, lo, hi = boot((f["policy_R"] - f["baseline_R"]).to_numpy(float), rng)
        rows.append(dict(arm=name, n=int(len(f)), mean_R=float(f["policy_R"].mean()),
                         vs_baseline_R=mu, ci_lo=lo, ci_hi=hi,
                         reject_rate=float((f["action"] == "REJECT").mean()),
                         mean_wait=float(f["wait"].mean(skipna=True))))
    rr = pd.DataFrame(rows)
    out["learned"] = rr.to_dict("records")
    out["baseline_mean_R"] = float(ev_base["outcome_R"].mean())
    print(f"enter-immediately baseline: {ev_base['outcome_R'].mean():.4f}R "
          f"on {len(ev_base):,} candidates\n")
    print(rr.round(4).to_string(index=False))
    net = rr.loc[rr.arm == "learned", "vs_baseline_R"].iloc[0] - \
        rr.loc[rr.arm == "permuted_state", "vs_baseline_R"].iloc[0]
    net_det = rr.loc[rr.arm == "learned", "vs_baseline_R"].iloc[0] - best_det["vs_baseline_R"]
    out["net_of_permuted"] = float(net)
    out["net_of_best_deterministic"] = float(net_det)
    print(f"\nlearned net of permuted-state twin : {net:+.4f}R")
    print(f"learned net of best deterministic  : {net_det:+.4f}R")
    print("\nBoth must be positive for a learned TIMING rule to exist. Beating only the\n"
          "enter-immediately baseline proves a fixed delay, not a policy.")

    DATA.mkdir(exist_ok=True)
    (DATA / "stage5_entry_timing.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {DATA / 'stage5_entry_timing.json'}")


if __name__ == "__main__":
    main()
