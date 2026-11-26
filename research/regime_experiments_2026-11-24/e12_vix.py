"""E12: VIX -- predicted vs actual regime, can a spike be predicted, and does fading spikes pay?
(PLAN.md section 5, pre-registered 2026-11-25 21:31 ET before running.)

E12a predicted vs actual (descriptive): predicted turbulent = LH vote >= 4 / HMM-R2 P(turb) > 0.5
     (E9 walk-forward, filtered) / live vote >= 4 (2021+). Actual turbulent = next-20-session
     realised vol above the trailing-252 75th percentile of RV20 (threshold known at t); second
     definition: next-20-session path drawdown <= -5%. Precision, recall, base rate by decade.
E12b spike prediction: S5 = max VIX over t+1..t+5 >= 1.25 VIX_t (primary); S10 = max over t+1..t+10
     >= 1.40 VIX_t. Features: E3 base on the S&P + VIX curl (1d/5d log change, VIX z, VRP) + Baa 20d
     change (1991+); secondary spans add sector correlation (1999+) and VIX3M/VIX (2007+).
     Walk-forward clipped logistic + LightGBM. Log-loss vs climatology; top-decile precision (by
     each decade's out-of-sample scores) minus base rate with a 20-session block-bootstrap CI;
     "curl adds" = curl block permuted inside the full model (100 draws, logistic).
     Decision: PREDICTABLE if the model beats climatology AND top-decile precision > base rate
     (CI > 0) in >= 2 of 3 decades.
E12c fading spikes: M1 VIX_t / min(VIX[t-10..t]) >= 1.5; M2 M1 within t-4..t AND VIX_t < VIX_t-1 AND
     VIX_t >= 1.3 x that 10-day min. Signal at the close of t, entry at the close of t+1, hold H in
     {1,5,10,20}, no overlapping trades. Instruments: S&P 500 and Nasdaq-100 long (1990+); short vol
     = -1x daily VIXY return minus 1.35%/yr (2011+); SVIX itself (2022+, sanity). Control: random
     entries, same count and holding, 1,000 draws; plus always-short-vol carry.
     Decision (H = 10): an E8 shadow candidate if mean AND hit rate beat the random p95 on both the
     equity and the short-vol instrument. Worst trade always reported. Not a live recommendation.

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e12_vix.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_factor_completeness as e3  # noqa: E402
import e9_regime_model as e9  # noqa: E402
import regime_common as rc  # noqa: E402

SEED = 20261125
ET = ZoneInfo("America/New_York")
VOL_RAW = rc.PANEL_DIR / "raw_vol"
SHORT_VOL_FEE = 0.0135
HOLDS = (1, 5, 10, 20)
DRAWS = 1000


# ---------------------------------------------------------------- data
def fetch_vol_etps() -> pd.DataFrame:
    """VIXY / SVXY / SVIX daily closes (Yahoo, split-adjusted), immutable raw folder per run."""
    existing = sorted(VOL_RAW.glob("*/vol_etps.parquet"))
    now = dt.datetime.now(ET)
    if existing and existing[-1].parent.name[:8] == now.strftime("%Y%m%d"):
        return pd.read_parquet(existing[-1])
    import yfinance as yf

    cutoff = pd.Timestamp(now.date() if now.time() >= dt.time(17, 0) else now.date() - dt.timedelta(days=1))
    out = {}
    for sym in ("VIXY", "SVXY", "SVIX"):
        d = yf.Ticker(sym).history(period="max", interval="1d", auto_adjust=False, timeout=30)
        if d.empty:
            raise RuntimeError(f"Yahoo returned nothing for {sym}")
        idx = pd.DatetimeIndex(d.index.tz_localize(None) if d.index.tz is not None else d.index).normalize()
        out[sym.lower()] = pd.Series(d["Close"].to_numpy(), index=idx)
    f = pd.DataFrame(out)
    f = f[f.index <= cutoff]
    run = VOL_RAW / now.strftime("%Y%m%dT%H%M")
    run.mkdir(parents=True, exist_ok=False)
    f.to_parquet(run / "vol_etps.parquet")
    (run / "manifest.json").write_text(json.dumps({"fetched_at_et": now.isoformat(), "cutoff": str(cutoff.date()),
                                                   "first_valid": {c: str(f[c].first_valid_index().date()) for c in f}}))
    return f


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ---------------------------------------------------------------- E12a
def predicted_vs_actual(p: pd.DataFrame) -> pd.DataFrame:
    fr, _ = rc.build_frame("ndx")
    thr = fr["trail_rv20"].rolling(252, min_periods=252).quantile(0.75)
    act_vol = (fr["fwd_rv20"] > thr).where(fr["fwd_rv20"].notna() & thr.notna())
    act_dd = (fr["fwd_mdd20"] <= -0.05).where(fr["fwd_mdd20"].notna())
    r = (np.log(p["ndx"]).diff() * 100).dropna()
    ph, _ = e9.hmm_walk_forward(r.to_numpy()[:, None], r.index, 2, 1991)
    hmm = pd.Series(ph, index=r.index)
    preds = {"lh_vote_ge4": (fr["vote7"] >= 4).where(fr["vote7"].notna()),
             "hmm_p_gt_0.5": (hmm > 0.5).where(hmm.notna()).reindex(fr.index)}
    try:
        from signals.market_regime.daily_regime import build_daily_regime
        from signals.market_regime.regime_vote import classify_frame

        live = build_daily_regime()
        lv = classify_frame(live)["regime_vote_count"]
        lv.index = pd.to_datetime(live["date"])
        preds["live_vote_ge4"] = (lv >= 4).where(lv.notna()).reindex(fr.index)
    except Exception as exc:  # reported
        print("live vote unavailable:", exc)
    rows = []
    for pname, pr in preds.items():
        for aname, act in (("vol_top_quartile", act_vol), ("drawdown_5pct", act_dd)):
            d = pd.DataFrame({"p": pr, "a": act}).dropna()
            d = d.loc["1991-01-01":]
            for dec, g in list(d.groupby(rc.decade_of(d.index).to_numpy())) + [("all", d)]:
                pp, aa = g["p"].astype(bool), g["a"].astype(bool)
                rows.append({"predictor": pname, "actual": aname, "decade": dec, "days": len(g),
                             "predicted_share": pp.mean(), "base_rate": aa.mean(),
                             "precision": aa[pp].mean() if pp.any() else np.nan,
                             "recall": pp[aa].mean() if aa.any() else np.nan,
                             "false_alarm_rate": (~aa[pp]).mean() if pp.any() else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- E12b
def spike_features(p: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.Series, pd.Series]:
    F, fam, _, _ = e3.build_features("spx")
    vix = p["vix"].reindex(F.index)
    lv = np.log(vix)
    F["curl_1d"] = lv.diff(1)
    F["curl_5d"] = lv.diff(5)
    F["curl_vix_z"] = rc.zscore(vix)
    F["curl_vrp"] = np.log(vix / (rc.rv20(p["spx"]).reindex(F.index) * np.sqrt(252) * 100))
    fam["curl"] = ["curl_1d", "curl_5d", "curl_vix_z", "curl_vrp"]
    F["credit_chg20_b"] = p["baa_10y_spread"].reindex(F.index).diff(20)
    fam["credit20"] = ["credit_chg20_b"]
    F["term_vix3m"] = np.log(p["vix3m"].reindex(F.index) / vix)
    fam["term_e12"] = ["term_vix3m"]
    v = vix.to_numpy()
    n = len(v)
    s5 = np.full(n, np.nan)
    s10 = np.full(n, np.nan)
    for i in range(n - 10):
        if np.isnan(v[i]) or np.isnan(v[i + 1:i + 11]).any():
            continue
        s5[i] = float(v[i + 1:i + 6].max() >= 1.25 * v[i])
        s10[i] = float(v[i + 1:i + 11].max() >= 1.40 * v[i])
    F = F.replace([np.inf, -np.inf], np.nan)
    return F, fam, pd.Series(s5, index=F.index), pd.Series(s10, index=F.index)


def top_decile(y, s, dec, rng) -> dict:
    out = {}
    for d in np.unique(dec):
        m = dec == d
        if m.sum() < 500:
            continue
        yy, ss = y[m], s[m]
        cut = np.quantile(ss, 0.9)
        top = ss >= cut
        diff = yy[top].mean() - yy.mean()
        n = len(yy)
        nb = int(np.ceil(n / 20))
        boot = np.empty(500)
        for k in range(500):
            idx = ((rng.integers(0, n, nb)[:, None] + np.arange(20)[None, :]) % n).ravel()[:n]
            y2, s2 = yy[idx], ss[idx]
            t2 = s2 >= np.quantile(s2, 0.9)
            boot[k] = y2[t2].mean() - y2.mean()
        out[d] = {"precision_top10": float(yy[top].mean()), "base_rate": float(yy.mean()), "lift": float(diff),
                  "lift_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    return out


def spike_prediction(p: pd.DataFrame, rng) -> dict:
    F, fam, s5, s10 = spike_features(p)
    res = {}
    specs = {"full_1991": fam["base"] + fam["curl"] + fam["credit20"],
             "full_plus_sectors_1999": fam["base"] + fam["curl"] + fam["credit20"] + fam["sectors_corr"],
             "full_plus_term_2007": fam["base"] + fam["curl"] + fam["credit20"] + fam["term_e12"]}
    for tname, y in (("S5", s5), ("S10", s10)):
        for spec, cols in specs.items():
            sub, ys, years, ty, _ = e3.span_setup(F, y, cols)
            X = sub[cols].to_numpy()
            dec = rc.decade_of(sub.index).to_numpy()
            pc = np.full(len(ys), np.nan)
            for yr in ty:
                te = np.where(years == yr)[0]
                if len(te):
                    tr = np.arange(0, max(te[0] - 10, 0))
                    pc[te] = np.nanmean(ys[tr])
            for model in ("lr", "lgb"):
                pr = e3.walk_forward(X, ys, years, ty, model=model, purge=10, winsor=(model == "lr"))
                ok = ~np.isnan(pr) & ~np.isnan(ys) & ~np.isnan(pc)
                skill = pd.Series(ll(ys[ok], pc[ok]) - ll(ys[ok], pr[ok])).groupby(dec[ok]).mean().to_dict()
                td = top_decile(ys[ok], pr[ok], dec[ok], rng)
                have = [d for d in td if d in ("1990s", "2000s", "2010s", "2020s")]
                passes = sum(skill.get(d, -1) > 0 and td[d]["lift_ci"][0] > 0 for d in have)
                entry = {"skill_vs_climatology": skill, "top_decile": td, "decades": len(have),
                         "passes": int(passes), "PREDICTABLE": bool(passes >= 2)}
                if model == "lr" and spec == "full_1991":
                    curl_idx = [cols.index(c) for c in fam["curl"]]
                    real = float((ll(ys[ok], e3.walk_forward(np.delete(X, curl_idx, axis=1), ys, years, ty, purge=10,
                                                             winsor=True)[ok]) - ll(ys[ok], pr[ok])).mean())
                    null = []
                    for _ in range(100):
                        Xp = X.copy()
                        idx = e3.block_permute_rows(len(X), rng)
                        Xp[:, curl_idx] = X[idx][:, curl_idx]
                        q = e3.walk_forward(Xp, ys, years, ty, purge=10, winsor=True)
                        null.append(float((ll(ys[ok], q[ok]) - ll(ys[ok], pr[ok])).mean()))
                    entry["curl_drop_cost"] = real  # loss increase when curl is removed (positive = curl helps)
                    entry["curl_perm_cost_mean"] = float(np.mean(null))
                    entry["curl_perm_cost_p05"] = float(np.percentile(null, 5))
                    entry["CURL_ADDS"] = bool(np.percentile(null, 5) > 0)
                res[f"{tname}|{spec}|{model}"] = {"base_rate": float(np.nanmean(ys)), "first_test_year": ty[0], **entry}
                print(tname, spec, model, "passes", passes, "of", len(have), flush=True)
    return res


# ---------------------------------------------------------------- E12c
def signals(vix: pd.Series) -> dict[str, pd.Series]:
    m10 = vix.rolling(11, min_periods=11).min()
    m1 = vix / m10 >= 1.5
    recent = m1.rolling(5, min_periods=1).max().astype(bool)
    m2 = recent & (vix < vix.shift(1)) & (vix >= 1.3 * m10)
    return {"M1_spike": m1.fillna(False), "M2_spike_then_down_day": m2.fillna(False)}


def trades_for(sig: pd.Series, ret: pd.Series, h: int) -> pd.DataFrame:
    """Signal at the close of t -> enter at the close of t+1 -> exit at the close of t+1+h."""
    idx = ret.index
    s = sig.reindex(idx).fillna(False).to_numpy()
    r = ret.to_numpy()
    out, i, n = [], 0, len(idx)
    while i < n:
        if s[i] and i + 1 + h < n and np.isfinite(r[i + 2:i + 2 + h]).all():
            gross = np.prod(1 + r[i + 2:i + 2 + h]) - 1
            out.append({"signal": idx[i], "entry": idx[i + 1], "exit": idx[i + 1 + h], "ret": gross})
            i = i + 1 + h
        else:
            i += 1
    return pd.DataFrame(out)


def random_control(ret: pd.Series, h: int, n_trades: int, rng) -> tuple[np.ndarray, np.ndarray]:
    r = ret.to_numpy()
    valid = np.array([np.isfinite(r[i + 2:i + 2 + h]).all() and i + 1 + h < len(r) for i in range(len(r))])
    cand = np.flatnonzero(valid)
    cum = np.concatenate([[0.0], np.cumsum(np.log1p(np.where(np.isfinite(r), r, 0.0)))])
    means, hits = np.empty(DRAWS), np.empty(DRAWS)
    for k in range(DRAWS):
        e = rng.choice(cand, n_trades, replace=False)
        g = np.expm1(cum[e + 2 + h] - cum[e + 2])
        means[k], hits[k] = g.mean(), (g > 0).mean()
    return means, hits


def fade_spikes(p: pd.DataFrame, etps: pd.DataFrame, rng) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    vix = p["vix"].dropna()
    sig = signals(vix)
    inst = {"spx_long": p["spx"].pct_change().loc["1990-01-01":],
            "ndx_long": p["ndx"].pct_change().loc["1990-01-01":]}
    vx = etps["vixy"].dropna()
    inst["short_vol_vixy"] = (-vx.pct_change() - SHORT_VOL_FEE / 252).dropna()
    sv = etps["svix"].dropna()
    inst["svix_actual"] = sv.pct_change().dropna()
    rows, tlist = [], []
    for sname, sg in sig.items():
        for iname, ret in inst.items():
            ret = ret.dropna()
            for h in HOLDS:
                t = trades_for(sg, ret, h)
                if len(t) < 5:
                    continue
                mu, hit = random_control(ret, h, len(t), rng)
                row = {"signal": sname, "instrument": iname, "hold": h, "trades": len(t),
                       "first": str(t["entry"].iloc[0].date()), "mean": t["ret"].mean(), "median": t["ret"].median(),
                       "hit_rate": (t["ret"] > 0).mean(), "worst": t["ret"].min(), "best": t["ret"].max(),
                       "random_mean_p95": float(np.percentile(mu, 95)), "random_hit_p95": float(np.percentile(hit, 95)),
                       "random_mean_med": float(np.median(mu)), "random_hit_med": float(np.median(hit)),
                       "beats_random": bool(t["ret"].mean() > np.percentile(mu, 95) and (t["ret"] > 0).mean() > np.percentile(hit, 95))}
                rows.append(row)
                if h == 10:
                    tlist.append(t.assign(signal_name=sname, instrument=iname))
    res = pd.DataFrame(rows)
    carry = {}
    for iname in ("short_vol_vixy", "svix_actual"):
        r = inst[iname]
        eq = np.cumprod(1 + r.to_numpy())
        yrs = len(r) / 252
        carry[iname] = {"from": str(r.index[0].date()), "cagr": float(eq[-1] ** (1 / yrs) - 1),
                        "mdd": float((eq / np.maximum.accumulate(eq) - 1).min()),
                        "worst_day": float(r.min()), "worst_day_date": str(r.idxmin().date())}
    dec = {}
    for sname in sig:
        h10 = res[(res.signal == sname) & (res.hold == 10)].set_index("instrument")
        eq_ok = h10.loc["spx_long", "beats_random"] if "spx_long" in h10.index else False
        sv_ok = h10.loc["short_vol_vixy", "beats_random"] if "short_vol_vixy" in h10.index else False
        dec[sname] = {"spx_beats_random": bool(eq_ok), "short_vol_beats_random": bool(sv_ok),
                      "SHADOW_CANDIDATE": bool(eq_ok and sv_ok)}
    return res, pd.concat(tlist) if tlist else pd.DataFrame(), {"carry": carry, "decision_h10": dec}


def main() -> int:
    warnings.simplefilter("ignore")
    rc.OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    p, panel = rc.latest_panel()
    etps = fetch_vol_etps()
    print("vol ETPs", {c: str(etps[c].first_valid_index().date()) for c in etps}, "last", etps.index[-1].date(), flush=True)
    pva = predicted_vs_actual(p)
    pva.to_csv(rc.OUT / "e12a_predicted_vs_actual.csv", index=False)
    print(pva[pva.decade == "all"].round(3).to_string(index=False), flush=True)
    res_c, trades, dec_c = fade_spikes(p, etps, rng)
    res_c.to_csv(rc.OUT / "e12c_fade_results.csv", index=False)
    trades.to_csv(rc.OUT / "e12c_trades_h10.csv", index=False)
    print(res_c.round(4).to_string(index=False), flush=True)
    print(json.dumps(dec_c, indent=1), flush=True)
    res_b = spike_prediction(p, rng)
    (rc.OUT / "e12_summary.json").write_text(json.dumps({"panel": panel, "E12b": res_b, "E12c": dec_c},
                                                        indent=2, default=str))
    print(json.dumps({k: (v["passes"], v["decades"], v["PREDICTABLE"], v.get("CURL_ADDS")) for k, v in res_b.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
