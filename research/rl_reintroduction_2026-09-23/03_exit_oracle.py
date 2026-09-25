"""Stage 3 -- how much R is a PERFECT exit worth, and how much of it is learnable?

This is the gate for every sequential-RL idea downstream. Plan correction C4: because the
market is exogenous and history replayable, the exit problem is FULL-INFORMATION OPTIMAL
STOPPING, not an exploration problem. So the honest first question is not "can an agent
learn to exit" but "how much is there to win at all, and how much survives generalisation".

Three things are measured, on identical entries:

  A  HEADROOM    oracle (hindsight-best close) minus the live-approximate rule, in R units.
                 An arithmetic property of price paths. No model, no fitting.
  B  RECOVERY    a feature-based imitation of the oracle, trained on TRAIN and graded on
                 VALIDATION: what share of the headroom survives out of sample. Headroom is
                 an upper bound; recovery is the realistic number.
  C  CONTROL     the same two on RANDOM entries from the same bars, so "headroom" can be
                 separated from "headroom our entries specifically create".

Exit convention: every policy decides AT A DAILY CLOSE and exits at THAT close. The oracle
is therefore attainable in principle by a daily policy -- it is not the intraday high, which
no policy could take. The stop/trail arm is the one exception and uses intraday low/high,
because that is how a real stop fills.

Entries: top-3 per decision bar by the deployed walk-forward OOF `mom_score` and
`htf_score`, k=3 per RANKING_CONFIG. TEST IS NEVER READ.
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
PATHS = DATA / "stage3_paths.parquet"
SEED = 17
TOP_K = 3
MAX_HOLD = 30            # sessions of path kept
FIXED_H = [5, 10, 15, 20, 30]
MIN_XS = 200
COST_RT = 0.002          # 10bp each way, from momentum BACKTEST_CONFIG commission_pct
N_BOOT = 5000

# ---------------------------------------------------------------------------
# TWO exit arms, because the first one I wrote was a DEAD config.
#
# `old_config` = momentum_config's RISK block (1.5-ATR stop, 2.4-ATR ratchet trail arming at
# 1.0 ATR, 15-session cap). Its only consumers are momentum_option_policy.py and
# backtest/simulate.py; live/runner.py:516 says the governed engine "Replaces
# MomentumOptionPolicy execution". Kept as the SUPERSEDED baseline so the value of the change
# the repo already made stays visible.
#
# `live_policy` = core.live_4h_exec.ExitPolicy as actually wired at live/runner.py:679-681.
STOP_ATR = 1.5
TRAIL_ARM_ATR = 1.0
TRAIL_DIST_ATR = 2.4
OLD_MAX_SESSIONS = 15          # max_holding_4h_bars = 30 at 2 bars/day

# live ExitPolicy, equity path (the one these bar paths can actually represent):
LIVE_STOP_ATR = 1.5           # underlying_stop_atr -- options; equity keeps the premium stop
LIVE_EQUITY_STOP_FRAC = 0.39  # stop_loss, applied to the SHARE price for equity
LIVE_TRAIL = None             # trail_stop = None -- NO TRAIL
LIVE_TAKE_PROFIT = 0.30       # +30% gain -> trim
LIVE_SCALE_FRAC = 0.16        # fraction sold at the trim
LIVE_HORIZON_SESSIONS = 27    # horizon_bars = 53 at 2 4H bars/day -> ~26.5 sessions

REGIME = ["regime_spy_trend", "regime_spy_ret_20", "regime_vix_z", "regime_vix_high",
          "breadth_z", "sector_dispersion_z", "spy_rv20_z", "risk_appetite_z",
          "liquidity_stress_z", "credit_risk_z"]


# ------------------------------------------------------------------ entry picks

def pick_entries(pan: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for score in ("mom_score", "htf_score"):
        r = pan.groupby("timestamp")[score].rank(ascending=False, method="first")
        p = pan[r <= TOP_K].copy()
        p["arm"] = score
        p["entry_score"] = p[score]
        frames.append(p)
    # random-k control: same bars, same eligible universe, k draws per bar.
    # groupby.sample is safe here because MIN_XS guarantees every bar has >= 200 rows.
    ctrl = pan.groupby("timestamp", sort=False).sample(n=TOP_K, random_state=SEED).copy()
    ctrl["arm"] = "random_k"
    ctrl["entry_score"] = np.nan
    frames.append(ctrl)
    e = pd.concat(frames, ignore_index=True)
    keep = ["timestamp", "ticker", "entry_session", "arm", "entry_score", "atr_pct"] + REGIME
    return e[keep].dropna(subset=["entry_session", "atr_pct"])


# ----------------------------------------------------------------------- paths

def build_paths(entries: pd.DataFrame) -> pd.DataFrame:
    """One row per (entry, session offset 0..MAX_HOLD): the realised path in R units."""
    rows = []
    tickers = sorted(entries["ticker"].unique())
    for i, t in enumerate(tickers, 1):
        path = BARS_1D / f"{t}.parquet"
        if not path.exists():
            continue
        d = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
        d["session_date"] = (pd.to_datetime(d["timestamp"], utc=True)
                             .dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None))
        d = d.sort_values("session_date").reset_index(drop=True)
        pos = pd.Series(np.arange(len(d)), index=d["session_date"])
        o, h, lo, c = (d[x].to_numpy(float) for x in ("open", "high", "low", "close"))
        sub = entries[entries["ticker"] == t]
        for r in sub.itertuples():
            j = pos.get(r.entry_session)
            if j is None or j + 1 >= len(d):
                continue
            entry_px = o[j]
            if not np.isfinite(entry_px) or entry_px <= 0:
                continue
            end = min(len(d) - 1, j + MAX_HOLD)
            n = end - j + 1
            unit = P.K_RISK * r.atr_pct
            if not np.isfinite(unit) or unit < P.K_RISK * P.MIN_ATR_PCT:
                continue
            rows.append(pd.DataFrame({
                "eid": f"{r.arm}|{t}|{r.entry_session.date()}",
                "arm": r.arm, "ticker": t, "entry_session": r.entry_session,
                "timestamp": r.timestamp, "entry_score": r.entry_score,
                "atr_pct": r.atr_pct, "offset": np.arange(n),
                "r_close": (c[j:end + 1] / entry_px - 1.0) / unit,
                "r_high": (h[j:end + 1] / entry_px - 1.0) / unit,
                "r_low": (lo[j:end + 1] / entry_px - 1.0) / unit,
            }))
        if i % 200 == 0:
            print(f"  paths {i}/{len(tickers)}", flush=True)
    p = pd.concat(rows, ignore_index=True)
    reg = entries.drop_duplicates(["arm", "ticker", "entry_session"])[
        ["arm", "ticker", "entry_session"] + REGIME]
    p = p.merge(reg, on=["arm", "ticker", "entry_session"], how="left")
    DATA.mkdir(exist_ok=True)
    p.to_parquet(PATHS, index=False)
    return p


# -------------------------------------------------------------------- policies

def cost_R(atr_pct: np.ndarray) -> np.ndarray:
    return COST_RT / (P.K_RISK * atr_pct)


def evaluate(paths: pd.DataFrame) -> pd.DataFrame:
    """Per entry: realised R under each policy, net of the round-trip cost."""
    out = []
    for eid, g in paths.groupby("eid", sort=False):
        g = g.sort_values("offset")
        rc = g["r_close"].to_numpy(float)
        rh = g["r_high"].to_numpy(float)
        rl = g["r_low"].to_numpy(float)
        n = len(rc)
        atr = float(g["atr_pct"].iloc[0])
        cst = COST_RT / (P.K_RISK * atr)
        rec = {"eid": eid, "arm": g["arm"].iloc[0], "ticker": g["ticker"].iloc[0],
               "entry_session": g["entry_session"].iloc[0],
               "timestamp": g["timestamp"].iloc[0], "atr_pct": atr,
               "entry_score": g["entry_score"].iloc[0], "path_len": n, "cost_R": cst}
        # fixed horizons
        for h in FIXED_H:
            k = min(h, n - 1)
            rec[f"fixed_{h}"] = rc[k] - cst if k >= 1 else np.nan
        # oracle: best CLOSE over offsets 1..n-1
        if n >= 2:
            k = int(np.nanargmax(rc[1:])) + 1
            rec["oracle"] = rc[k] - cst
            rec["oracle_day"] = k
        else:
            rec["oracle"] = np.nan
            rec["oracle_day"] = np.nan
        # --- arm 1: OLD (superseded) config -- ATR stop + ratchet trail + 15-session cap.
        # r_* are already in R = (px/entry - 1)/(K_RISK*atr), so an X-ATR price move is
        # X/K_RISK in R units.
        stop_level = -STOP_ATR / P.K_RISK
        arm_level = TRAIL_ARM_ATR / P.K_RISK
        trail_gap = TRAIL_DIST_ATR / P.K_RISK
        armed, best, exit_r, exit_day = False, -np.inf, None, None
        for k in range(1, n):
            if rh[k] > best:
                best = rh[k]
            if not armed and best >= arm_level:
                armed = True
            level = (best - trail_gap) if armed else stop_level
            if rl[k] <= level:
                exit_r, exit_day = level, k      # filled at the stop level
                break
            if k >= OLD_MAX_SESSIONS:
                exit_r, exit_day = rc[k], k
                break
        if exit_r is None:
            exit_r, exit_day = rc[n - 1], n - 1
        rec["old_config"] = exit_r - cst
        rec["old_day"] = exit_day

        # --- arm 2: the ACTUAL live ExitPolicy. No trail. 1.5-ATR stop. +30% trim of 16%.
        # 27-session horizon. The trim means the position exits in two pieces, so the
        # realised R is a weighted blend.
        stop_level = -LIVE_STOP_ATR / P.K_RISK
        tp_level = LIVE_TAKE_PROFIT / (P.K_RISK * atr)   # +30% of PRICE, in R units
        trimmed_at = None
        exit_r, exit_day = None, None
        for k in range(1, n):
            if rl[k] <= stop_level:                      # hard stop first, as live does
                exit_r, exit_day = stop_level, k
                break
            if trimmed_at is None and rh[k] >= tp_level:
                trimmed_at = tp_level                    # trim fills at the trigger
            if k >= LIVE_HORIZON_SESSIONS:
                exit_r, exit_day = rc[k], k
                break
        if exit_r is None:
            exit_r, exit_day = rc[n - 1], n - 1
        if trimmed_at is None:
            rec["live_policy"] = exit_r - cst
        else:
            rec["live_policy"] = (LIVE_SCALE_FRAC * trimmed_at
                                  + (1.0 - LIVE_SCALE_FRAC) * exit_r) - cst
        rec["live_day"] = exit_day
        rec["live_trimmed"] = trimmed_at is not None
        out.append(rec)
    return pd.DataFrame(out)


def boot_mean(v: np.ndarray, rng) -> tuple[float, float, float]:
    v = v[np.isfinite(v)]
    if len(v) < 30:
        return float(np.mean(v)) if len(v) else np.nan, np.nan, np.nan
    d = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)])
    return float(v.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def dist(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if not len(x):
        return {}
    return {"n": int(len(x)), "mean": float(x.mean()), "p5": float(np.percentile(x, 5)),
            "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "share_gt_3R": float((x >= 3).mean()), "share_lt_neg1R": float((x <= -1).mean())}


# ------------------------------------------------------------------- imitation

def imitation(paths: pd.DataFrame, res: pd.DataFrame, tr_end: pd.Timestamp,
              va: tuple[pd.Timestamp, pd.Timestamp], *, permute: bool = False,
              seed: int = SEED) -> dict:
    """Train a continuation-value model on the ORACLE and grade it out of sample.

    At every open position-day the target is the oracle's remaining value:
        cont = max(r_close[t+1:]) - r_close[t]
    i.e. how much more R a perfect holder gets by NOT exiting now. The policy exits at the
    first day the model predicts cont <= 0. Trained on TRAIN entries, graded on VALIDATION.

    `permute=True` is the MANDATORY null arm (00_preregistration.md section 6.1): the target
    is shuffled WITHIN each position-day offset, so the model keeps whatever the offset
    structure gives it and loses only the row-level signal. Without this arm the recovery
    number cannot be distinguished from the fact that the policy's DEFAULT is "hold to the
    end", which is itself a good policy here.
    """
    import xgboost as xgb
    rng = np.random.default_rng(seed)

    p = paths.sort_values(["eid", "offset"]).copy()
    g = p.groupby("eid", sort=False)["r_close"]
    # running max of FUTURE closes, strictly after t
    rev_max = g.transform(lambda s: s[::-1].shift(1).cummax()[::-1])
    p["cont"] = rev_max - p["r_close"]
    p["run_max"] = g.cummax()
    p["run_min"] = g.cummin()
    p["ret_5"] = g.transform(lambda s: s - s.shift(5))
    p["off"] = p["offset"]
    p = p[p["offset"] >= 1].dropna(subset=["cont"])

    feats = ["off", "r_close", "run_max", "run_min", "ret_5", "atr_pct",
             "entry_score"] + REGIME
    tr = p[p["timestamp"] <= tr_end].copy()
    ev = p[(p["timestamp"] >= va[0]) & (p["timestamp"] <= va[1])]
    if tr.empty or ev.empty:
        return {"error": "empty split"}
    y = tr["cont"].to_numpy(float).copy()
    if permute:
        # shuffle within each offset: the offset-conditional distribution of `cont` is
        # preserved, only the link to THIS position-day is destroyed.
        for _, pos in tr.groupby("off", sort=False).indices.items():
            pos = np.asarray(pos)
            y[pos] = y[rng.permutation(pos)]
    m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, random_state=SEED, n_jobs=4,
                         tree_method="hist")
    m.fit(tr[feats].to_numpy(float), y)
    ev = ev.assign(pred=m.predict(ev[feats].to_numpy(float)))

    rows = []
    for eid, gg in ev.groupby("eid", sort=False):
        gg = gg.sort_values("offset")
        hit = gg[gg["pred"] <= 0]
        k = int(hit["offset"].iloc[0]) if len(hit) else int(gg["offset"].iloc[-1])
        r = float(gg.loc[gg["offset"] == k, "r_close"].iloc[0])
        atr = float(gg["atr_pct"].iloc[0])
        cst = COST_RT / (P.K_RISK * atr)
        rows.append({"eid": eid, "imitation": r - cst, "imitation_day": k,
                     "early_exit": bool(len(hit))})
    im = pd.DataFrame(rows)
    return {"frame": im, "n_train_rows": int(len(tr)), "n_eval_rows": int(len(ev)),
            "feature_importance": dict(zip(feats, [float(x) for x in m.feature_importances_]))}


# ------------------------------------------------------------------------ main

def main() -> None:
    rng = np.random.default_rng(SEED)
    out: dict = {}
    pan = P.load(require=["mom_score", "r_10"], drop_flagged=True)
    pan = pan[pan["xs_size"] >= MIN_XS]
    tr, va, te, bounds = P.split(pan)
    out["bounds"] = bounds
    # take the edges from the slices P.split actually produced: bounds carries DATES, and
    # localising a date to midnight would drop that day's intraday bars.
    tr_end = tr["timestamp"].max()
    va_lo, va_hi = va["timestamp"].min(), va["timestamp"].max()
    trval = pan[pan["timestamp"] <= va_hi]
    print(json.dumps(bounds, indent=2))
    print(f"train+val bars {trval['timestamp'].nunique():,} "
          f"(TEST from {bounds['test'][0]} is never read)", flush=True)

    if PATHS.exists():
        print(f"reading cached paths {PATHS}", flush=True)
        paths = pd.read_parquet(PATHS)
    else:
        entries = pick_entries(trval)
        print(f"entries {len(entries):,} over {entries['ticker'].nunique()} tickers; "
              f"building paths ...", flush=True)
        paths = build_paths(entries)
    print(f"path rows {len(paths):,} over {paths['eid'].nunique():,} entries", flush=True)

    res = evaluate(paths)
    out["entries"] = {a: int(n) for a, n in res["arm"].value_counts().items()}
    print("\n" + "=" * 78 + "\nA. HEADROOM -- realised R by policy (train+val entries)\n"
          + "=" * 78)
    print("Every policy decides at a daily CLOSE and exits at that close, so the oracle is\n"
          "attainable in principle. Stop arms use intraday low/high, as a real stop would.\n"
          "`old_config` is momentum_config's SUPERSEDED RISK block (1.5-ATR stop + 2.4-ATR\n"
          "ratchet trail + 15-session cap) -- it does NOT bind the live path. `live_policy`\n"
          "is core.live_4h_exec.ExitPolicy as wired at live/runner.py:679-681: NO trail,\n"
          "1.5-ATR stop, +30% trim of 16%, 27-session horizon. This is the EQUITY path; the\n"
          "option premium path is not represented here.\n")
    pol = [f"fixed_{h}" for h in FIXED_H] + ["old_config", "live_policy", "oracle"]
    tab = []
    for arm, g in res.groupby("arm"):
        for c in pol:
            mu, lo, hi = boot_mean(g[c].to_numpy(float), rng)
            tab.append(dict(arm=arm, policy=c, n=int(g[c].notna().sum()), mean_R=mu,
                            ci_lo=lo, ci_hi=hi, median_R=float(g[c].median())))
    td = pd.DataFrame(tab)
    out["policies"] = td.to_dict("records")
    print(td.pivot_table(index="policy", columns="arm", values="mean_R").round(3).to_string())
    print("\nmedian R:")
    print(td.pivot_table(index="policy", columns="arm", values="median_R").round(3).to_string())

    print("\nheadroom = oracle - policy (mean R, with a bootstrap CI on the PAIRED diff):")
    hr = []
    for arm, g in res.groupby("arm"):
        for c in pol[:-1]:
            d = (g["oracle"] - g[c]).to_numpy(float)
            mu, lo, hi = boot_mean(d, rng)
            hr.append(dict(arm=arm, vs=c, headroom_R=mu, ci_lo=lo, ci_hi=hi))
    hd = pd.DataFrame(hr)
    out["headroom"] = hd.to_dict("records")
    print(hd.pivot_table(index="vs", columns="arm", values="headroom_R").round(3).to_string())

    print("\nexit-day distribution (median session of exit):")
    print(res.groupby("arm")[["oracle_day", "old_day", "live_day"]].median().round(1).to_string())
    print("\ntrim rate under the live policy (+30% take-profit):")
    print(res.groupby("arm")["live_trimmed"].mean().round(3).to_string())

    print("\nR distribution, momentum arm:")
    mom = res[res["arm"] == "mom_score"]
    out["distribution_mom"] = {c: dist(mom[c].to_numpy(float)) for c in pol}
    print(pd.DataFrame(out["distribution_mom"]).T.round(3).to_string())

    # ------------------------------------------------------------ B. recovery
    print("\n" + "=" * 78 + "\nB. RECOVERY -- oracle imitation, trained on TRAIN, graded on "
          "VALIDATION\n" + "=" * 78)
    real = imitation(paths, res, tr_end, (va_lo, va_hi), permute=False)
    null = imitation(paths, res, tr_end, (va_lo, va_hi), permute=True)
    if "error" in real or "error" in null:
        print("  empty split")
    else:
        fr, fn = real.pop("frame"), null.pop("frame")
        j = (res.merge(fr, on="eid", how="inner")
                .merge(fn.rename(columns={"imitation": "imitation_null",
                                          "imitation_day": "imitation_null_day",
                                          "early_exit": "early_exit_null"}),
                       on="eid", how="inner"))
        j = j[(j["timestamp"] >= va_lo) & (j["timestamp"] <= va_hi)]
        print(f"  train position-days {real['n_train_rows']:,}  "
              f"eval position-days {real['n_eval_rows']:,}  graded entries {len(j):,}")
        print("  BEST FIXED HORIZON is the benchmark that matters: the policy's DEFAULT is\n"
              "  hold-to-the-end, so beating live_approx is not evidence of a learned exit.\n")
        rec = []
        for arm, g in j.groupby("arm"):
            base = float(g["live_policy"].mean())
            orc = float(g["oracle"].mean())
            bestfix = max((float(g[f"fixed_{h}"].mean()) for h in FIXED_H))
            room = orc - bestfix
            mu, lo, hi = boot_mean((g["imitation"] - g["fixed_30"]).to_numpy(float), rng)
            mun, _, _ = boot_mean((g["imitation_null"] - g["fixed_30"]).to_numpy(float), rng)
            rec.append(dict(arm=arm, n=int(len(g)), live_R=base,
                            old_config_R=float(g["old_config"].mean()),
                            fixed30_R=float(g["fixed_30"].mean()),
                            best_fixed_R=bestfix, imitation_R=float(g["imitation"].mean()),
                            null_R=float(g["imitation_null"].mean()), oracle_R=orc,
                            room_over_best_fixed=room,
                            gain_vs_fixed30=mu, gain_ci_lo=lo, gain_ci_hi=hi,
                            null_gain_vs_fixed30=mun, net_of_null=mu - mun,
                            early_exit_rate=float(g["early_exit"].mean()),
                            null_early_exit_rate=float(g["early_exit_null"].mean()),
                            mean_exit_day=float(g["imitation_day"].mean())))
        rd = pd.DataFrame(rec)
        out["recovery"] = rd.to_dict("records")
        out["imitation_importance"] = real["feature_importance"]
        print(rd.round(3).to_string(index=False))
        print("\nnet_of_null is the number that counts. A gain that the PERMUTED-label\n"
              "model reproduces is the offset structure, not a learned exit rule.")
        print("\ntop imitation features:")
        for k, v in sorted(real["feature_importance"].items(), key=lambda x: -x[1])[:8]:
            print(f"  {k:<22} {v:.4f}")

    DATA.mkdir(exist_ok=True)
    (DATA / "stage3_exit_oracle.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {DATA / 'stage3_exit_oracle.json'}")


if __name__ == "__main__":
    main()
