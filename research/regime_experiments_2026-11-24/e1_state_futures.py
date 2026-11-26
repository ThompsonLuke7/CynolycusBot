"""E1: does a regime state describe different futures? (PLAN.md, pre-registered 2026-11-24 15:19 ET)

Primary: Nasdaq-100, states from the long-history vote (regime_common.py), decades 1980s..2020s.
Secondary: S&P 500 1962-2026 (all trend votes on the S&P itself), decades 1960s..2020s.

Targets (from the close of t, regime_common.forward_targets): fwd_mdd20 (primary), fwd_rv20,
fwd_ret20 (reported only). E1 is descriptive: the window starts at the close the state is
computed from, which is not an executable price (E2 trades one session later).

Statistic per decade and target: eta^2 of the 5-state label (share of the target's variance the
state explains; direction-free). Null: circular shift of the state series within the decade
(1,000 draws, shift >= 60 sessions), which keeps both series' autocorrelation.
Per-state means carry a 95% circular 20-session block-bootstrap CI (500 draws).

Decision (fixed in PLAN.md): the state definition is USEFUL if fwd_mdd20 or fwd_rv20 eta^2 exceeds
the placebo 95th percentile in at least 4 of 5 Nasdaq-100 decades. Added before running
(2026-11-24 21:30 ET): the S&P secondary passes at >= 6 of 7 decades.

Secondary diagnostics, added 2026-11-24 21:30 ET before any result was computed. They are not
part of the decision:
  (a) beyond vol clustering: eta^2 of the state on the target's residual after an in-decade OLS on
      log trailing RV20, with the same placebo. Forward vol is persistent, so a state containing an
      RV vote will "describe" forward vol trivially; (a) asks whether it adds anything beyond that.
  (b) comparators: BEAR flag (vote7 >= 4), SMA50-only and RV-only, each 2-state, same placebo.
  (c) agreement with the live 7-vote (built in memory by build_daily_regime, 2021+ rows).

    .venv/bin/python research/regime_experiments_2026-11-24/e1_state_futures.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import regime_common as rc  # noqa: E402

DRAWS = 1000
BOOT = 500
MIN_SHIFT = 60
TARGETS = ("fwd_mdd20", "fwd_rv20", "fwd_ret20")
SEED = 20261124


def residual_on_trail_vol(y: np.ndarray, trail: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones_like(trail), np.log(trail)])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


def analyse(target: str, start: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fr, panel_name = rc.build_frame(target)
    fr = fr.loc[start:].dropna(subset=["state", "fwd_mdd20", "trail_rv20"])
    fr = fr[fr["trail_rv20"] > 0]
    fr["decade"] = rc.decade_of(fr.index)
    fr["bear"] = (fr["vote7"] >= 4).astype(int)
    rng = np.random.default_rng(SEED)
    code_of = {s: i for i, s in enumerate(rc.STATES)}
    tests, states, comps = [], [], []
    for dec, g in fr.groupby("decade"):
        codes = g["state"].map(code_of).to_numpy()
        for t in TARGETS:
            y = g[t].to_numpy()
            obs = rc.eta2(codes, y)
            null = rc.circular_shift_null(codes, y, rc.eta2, DRAWS, rng, MIN_SHIFT)
            yr = residual_on_trail_vol(y, g["trail_rv20"].to_numpy())
            obs_r = rc.eta2(codes, yr)
            null_r = rc.circular_shift_null(codes, yr, rc.eta2, DRAWS, rng, MIN_SHIFT)
            tests.append({"decade": dec, "target": t, "n": len(g), "first": str(g.index[0].date()),
                          "last": str(g.index[-1].date()), "eta2": obs, "null_p95": np.percentile(null, 95),
                          "placebo_pct": (null < obs).mean() * 100, "beats_p95": obs > np.percentile(null, 95),
                          "eta2_beyond_vol": obs_r, "null_beyond_vol_p95": np.percentile(null_r, 95),
                          "beats_p95_beyond_vol": obs_r > np.percentile(null_r, 95)})
            for label, codes2 in (("bear_flag", g["bear"].to_numpy()),
                                  ("sma50_only", g["below_sma50"].astype(int).to_numpy()),
                                  ("rv_only", g["rv"].astype(int).to_numpy())):
                o2 = rc.eta2(codes2, y)
                n2 = rc.circular_shift_null(codes2, y, rc.eta2, DRAWS, rng, MIN_SHIFT)
                comps.append({"decade": dec, "target": t, "state_def": label, "eta2": o2,
                              "null_p95": np.percentile(n2, 95), "beats_p95": o2 > np.percentile(n2, 95)})
            comps.append({"decade": dec, "target": t, "state_def": "lh_vote_5state", "eta2": obs,
                          "null_p95": np.percentile(null, 95), "beats_p95": obs > np.percentile(null, 95)})
        for s in rc.STATES:
            m = (g["state"] == s).to_numpy()
            row = {"decade": dec, "state": s, "n_days": int(m.sum()), "share": m.mean()}
            if m.sum() >= 20:
                for t in TARGETS:
                    y = g[t].to_numpy()
                    lo, hi = rc.block_bootstrap_mean(y, m, BOOT, rng)
                    row[f"{t}_mean"], row[f"{t}_lo"], row[f"{t}_hi"] = y[m].mean(), lo, hi
                row["p_mdd_le_5pct"] = (g["fwd_mdd20"].to_numpy()[m] <= -0.05).mean()
            states.append(row)
    return (pd.DataFrame(tests).assign(index=target, panel=panel_name),
            pd.DataFrame(states).assign(index=target), pd.DataFrame(comps).assign(index=target))


def live_agreement() -> dict:
    from signals.market_regime.daily_regime import build_daily_regime
    from signals.market_regime.regime_vote import classify_frame

    live = build_daily_regime()
    live = live.set_index(pd.to_datetime(live["date"])) if "date" in live.columns else live
    lv = classify_frame(live).dropna(subset=["regime_vote_count"])
    fr, _ = rc.build_frame("ndx")
    both = lv.join(fr[["vote7", "state"]], how="inner").dropna(subset=["vote7"])
    both = both.loc["2021-01-01":]
    if both.empty:
        raise ValueError("no overlap between the live vote and the LH vote")
    bear_live = both["regime_vote_count"] >= 4
    bear_lh = both["vote7"] >= 4
    return {"rows": len(both), "first": str(both.index[0].date()), "last": str(both.index[-1].date()),
            "state_agreement": float((both["regime"] == both["state"]).mean()),
            "bear_flag_agreement": float((bear_live == bear_lh).mean()),
            "bear_share_live": float(bear_live.mean()), "bear_share_lh": float(bear_lh.mean()),
            "vote_count_corr": float(both["regime_vote_count"].corr(both["vote7"])),
            "confusion_live_rows_lh_cols": pd.crosstab(both["regime"], both["state"]).to_dict(orient="index")}


def main() -> int:
    rc.OUT.mkdir(exist_ok=True)
    results = {}
    all_t, all_s, all_c = [], [], []
    for target, start, need in (("ndx", "1986-01-01", 4), ("spx", "1962-01-01", 6)):
        t, s, c = analyse(target, start)
        all_t.append(t), all_s.append(s), all_c.append(c)
        dec = t[t["target"].isin(["fwd_mdd20", "fwd_rv20"])].groupby("decade")["beats_p95"].any()
        dec_mdd = t[t["target"] == "fwd_mdd20"].set_index("decade")["beats_p95"]
        dec_bv = t[t["target"].isin(["fwd_mdd20", "fwd_rv20"])].groupby("decade")["beats_p95_beyond_vol"].any()
        results[target] = {"decades": len(dec), "decades_passing": int(dec.sum()), "needed": need,
                           "USEFUL": bool(dec.sum() >= need),
                           "decades_passing_mdd_only": int(dec_mdd.sum()),
                           "decades_passing_beyond_vol": int(dec_bv.sum())}
    tests = pd.concat(all_t)
    tests.to_csv(rc.OUT / "e1_tests.csv", index=False)
    pd.concat(all_s).to_csv(rc.OUT / "e1_state_means.csv", index=False)
    pd.concat(all_c).to_csv(rc.OUT / "e1_comparators.csv", index=False)
    try:
        results["live_agreement"] = live_agreement()
    except Exception as exc:  # reported, not hidden
        results["live_agreement"] = f"FAILED {type(exc).__name__}: {exc}"
    (rc.OUT / "e1_summary.json").write_text(json.dumps(results, indent=2, default=str))
    pd.set_option("display.width", 220)
    print(tests[["index", "decade", "target", "n", "eta2", "null_p95", "placebo_pct", "beats_p95",
                 "eta2_beyond_vol", "null_beyond_vol_p95", "beats_p95_beyond_vol"]].round(4).to_string(index=False))
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
