"""E5: how fine-grained must the intraday regime be? (PLAN.md, pre-registered 2026-11-24 15:19 ET)

Data: SPY and QQQ 5-minute SIP bars 2016-01..2026-11-20 (DIVIDEND-ADJUSTED, as cached by the 2026-11-20
regime study: 0.987x raw in 2024; every input is a within-session ratio or a gap to the adjusted prior
close, so the factor cancels) and VIXY 5-minute SIP bars (split-adjusted, so a
change since the prior close never spans a reverse split), fetched once into
Data/research/regime_40y/bars_5m/ (immutable). Regular sessions only; the ~3 half-day sessions a
year are excluded (they would need their own clock). 5-minute bars are enough: the finest arm in
the plan IS the 5-minute refresh, so no 1-minute fetch is needed.

Clarified BEFORE running (2026-11-24 22:00 ET). The plan's checkpoints (10:00, 10:30, 11:00, 12:00,
14:30) all sit on the 30-minute grid, where a 30-minute and a 5-minute refresh are identical by
construction. So decisions are made at EVERY 5-minute bar close from 10:00 to 15:30 ET, and the
arms differ only in how stale the shared session state is at the decision:
  (a) daily only: prior-close LH vote (regime_common; S&P for SPY, Nasdaq-100 for QQQ), log VIX
      and log RV20 at the prior close, ATR20 %; no session state
  (b) session state computed once, at 10:00 (30 minutes after the open)
  (c) refreshed every 30 minutes (state as of the last :00/:30 bar close)
  (d) refreshed every 5 minutes (state as of the decision bar)
  (e) (b) plus event-triggered refreshes: on scheduled FOMC days at 14:05 and 14:35 (the bars after
      the 14:00 statement and the 14:30 press conference), and on any day after a shock bar
      (|5-min log return| > 4x the trailing 20-session SD of 5-min returns)
  (b_sched) (b) plus the schedule alone: an FOMC-day indicator and FOMC x after-14:00
All arms include log minutes-to-close at the decision. Session state at time s: gap, return since
the open, close vs session VWAP, opening-range (09:30-10:00) break in ATR units, range so far / ATR,
5-minute realised vol so far / its trailing 20-session level, VIXY change since the prior close.
Targets at decision t: rest-of-day range (log of (max high - min low after t) / close_t) and
rest-of-day direction (close > close_t).

Models: walk-forward per symbol, yearly refit, trained on all earlier years (test 2018..2026).
Logistic regression for direction, OLS for range, scalers fit on training rows only.
Gain of an arm = loss(a) - loss(arm) (log-loss for direction, MSE for range); 95% CIs from a
day-block bootstrap (500 draws).

Decision (fixed): use the coarsest refresh (b -> c -> d) whose gain is at least 80% of (d)'s.
Add event-triggered refreshes only if, on FOMC days after 14:05, (e) beats (b_sched), i.e. the
refreshed state predicts the rest of the day beyond what the schedule alone gives (CI above 0).

Follow-up after the first run (2026-11-24 22:10 ET), labelled, not part of the decision: arm
(c_sched) = (c) plus the FOMC flags, tested against (c) on FOMC days after 14:05 and on all rows,
because the first run picked the 30-minute clock and showed the schedule flag beat the
event-refreshed state. Its bootstrap uses its own seed so the reported CIs are unchanged.

    .venv/bin/python research/regime_experiments_2026-11-24/e5_intraday_granularity.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regime_common as rc  # noqa: E402

REPO = rc.REPO
BARS5 = REPO / "Data" / "research" / "regime_switch_2026-11-20" / "bars_5m"
VIXY_PATH = rc.PANEL_DIR / "bars_5m" / "VIXY.parquet"
SEED = 20261124
BOOT = 500
SHOCK_K = 4.0
# Scheduled FOMC statement days (14:00 ET). 2016-2021 read from federalreserve.gov historical pages
# (2021 from fomccalendars.htm); 2022+ from signals/events/economic_calendar.py. Unscheduled actions
# (2019-10-11, 2020-03-03, 2020-03-15, 2020-03-23) are not 14:00 events and are excluded.
FOMC_2016_2021 = [
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
]
STATE_COLS = ["s_gap", "s_ret_open", "s_vwap", "s_or_break", "s_range_atr", "s_rv_ratio", "s_vixy"]
DAILY_COLS = ["d_vote7", "d_log_vix", "d_log_rv20", "d_atr_pct"]


def fomc_days() -> set[pd.Timestamp]:
    from signals.events.economic_calendar import _FOMC_DATES

    return {pd.Timestamp(d) for d in FOMC_2016_2021 + list(_FOMC_DATES)}


def fetch_vixy() -> pd.DataFrame:
    if VIXY_PATH.exists():
        return pd.read_parquet(VIXY_PATH)
    from core.API.Alpaca_API.market_data.fetch_intraday import fetch_intraday

    VIXY_PATH.parent.mkdir(parents=True, exist_ok=True)
    d = fetch_intraday(ticker="VIXY", start="2016-01-01", end="2026-11-21", timeframe="5Min",
                       limit=None, adjustment="split", feed="SIP", skip_delisted=False)
    if d.empty:
        raise RuntimeError("VIXY 5m fetch returned nothing")
    d.to_parquet(VIXY_PATH)
    (VIXY_PATH.parent / "VIXY_manifest.json").write_text(json.dumps(
        {"fetched_at_utc": pd.Timestamp.now("UTC").isoformat(), "source": "alpaca sip 5Min adjustment=split",
         "rows": len(d)}))
    return d


def to_sessions(d: pd.DataFrame) -> pd.DataFrame:
    d = d.reset_index() if "timestamp" not in d.columns else d.copy()
    d["et"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("America/New_York")
    d["session"] = d["et"].dt.normalize().dt.tz_localize(None)
    mins = d["et"].dt.hour * 60 + d["et"].dt.minute
    d["bar_min"] = mins  # bar start, minutes after midnight ET
    return d[(mins >= 570) & (mins < 960)].reset_index(drop=True)


def session_rows(sym: str, fomc: set, vixy: pd.DataFrame, daily_state: pd.DataFrame) -> pd.DataFrame:
    bars = to_sessions(pd.read_parquet(BARS5 / f"{sym}.parquet"))
    # daily inputs from ALL sessions (half days included) so prior close / ATR never skip a day
    full = bars.groupby("session")["bar_min"].agg(["min", "max", "count"])
    full = set(full[(full["min"] == 570) & (full["max"] == 955) & (full["count"] >= 76)].index)
    vx = to_sessions(vixy)
    vx_by_sess = {k: g.set_index("bar_min")["close"] for k, g in vx.groupby("session")}
    vx_close = vx.groupby("session")["close"].last()
    vx_prev = vx_close.shift(1)
    lr5 = np.log(bars["close"]).groupby(bars["session"]).diff()
    sd5 = lr5.groupby(bars["session"]).std().shift(1).rolling(20, min_periods=20).mean()
    day = bars.groupby("session").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    tr = pd.concat([day["h"] - day["l"], (day["h"] - day["c"].shift()).abs(), (day["l"] - day["c"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(20, min_periods=20).mean().shift(1)
    rv20 = np.log(day["c"]).diff().rolling(20, min_periods=20).std().shift(1)
    out = []
    for sess, g in bars.groupby("session"):
        if sess not in full:
            continue
        if sess not in atr.index or not np.isfinite(atr.get(sess, np.nan)) or not np.isfinite(sd5.get(sess, np.nan)):
            continue
        prev_c = day["c"].shift().get(sess)
        ds = daily_state[daily_state.index < sess].tail(1)  # prior session's close values
        if ds.empty or not np.isfinite(prev_c) or (sess - ds.index[-1]).days > 5:
            continue
        g = g.sort_values("bar_min")
        close = g["close"].to_numpy()
        hi, lo = g["high"].to_numpy(), g["low"].to_numpy()
        end_min = g["bar_min"].to_numpy() + 5  # bar close time
        vw = np.cumsum(g["vwap"].to_numpy() * g["volume"].to_numpy()) / np.cumsum(g["volume"].to_numpy())
        cum_hi, cum_lo = np.maximum.accumulate(hi), np.minimum.accumulate(lo)
        or_idx = np.where(end_min == 600)[0]
        if len(or_idx) == 0:
            continue
        k_or = or_idx[0]
        or_hi, or_lo = cum_hi[k_or], cum_lo[k_or]
        a = atr[sess]
        lr = np.diff(np.log(close), prepend=np.log(g["open"].iloc[0]))
        rv_so_far = np.sqrt(np.cumsum(lr ** 2) / np.arange(1, len(lr) + 1))
        vxs = vx_by_sess.get(sess, pd.Series(dtype=float))
        vx_raw = vxs.reindex(g["bar_min"])
        vx_at = vx_raw.ffill().to_numpy()  # within-session carry of the last VIXY print; share reported
        vxp = vx_prev.get(sess, np.nan)
        st = pd.DataFrame({
            "s_gap": np.log(g["open"].iloc[0] / prev_c),
            "s_ret_open": np.log(close / g["open"].iloc[0]),
            "s_vwap": np.log(close / vw),
            "s_or_break": np.where(close > or_hi, (close - or_hi) / a, np.where(close < or_lo, (close - or_lo) / a, 0.0)),
            "s_range_atr": (cum_hi - cum_lo) / a,
            "s_rv_ratio": np.log(rv_so_far / sd5[sess]),
            "s_vixy": np.log(vx_at / vxp) if np.isfinite(vxp) else np.nan,
        })
        st["end_min"] = end_min
        shock = np.abs(lr) > SHOCK_K * sd5[sess]
        # rest-of-day targets at each decision bar k: bars after k
        rev_hi = np.maximum.accumulate(hi[::-1])[::-1]
        rev_lo = np.minimum.accumulate(lo[::-1])[::-1]
        n = len(g)
        dec = np.where((end_min >= 600) & (end_min <= 930))[0]
        is_fomc = sess in fomc
        state_np = st[STATE_COLS].to_numpy()
        pos_of = {m: i for i, m in enumerate(end_min)}
        refresh_e = [600] + ([845, 875] if is_fomc else []) + [int(m) for m in end_min[shock] if m > 600]
        refresh_e = np.array(sorted(set(refresh_e)))
        for k in dec:
            t = end_min[k]
            if k + 1 >= n:
                continue
            rng_rest = rev_hi[k + 1] - rev_lo[k + 1]
            row = {"session": sess, "t": int(t), "vixy_bar_missing": float(np.isnan(vx_raw.iloc[k])), "y_range": np.log(max(rng_rest, 1e-6) / close[k]),
                   "y_up": float(close[-1] > close[k]), "log_mins_left": np.log(960 - t),
                   "fomc": float(is_fomc), "fomc_post14": float(is_fomc and t >= 845)}
            for arm, s_time in (("b", 600), ("c", t - (t % 30)), ("d", t),
                                ("e", refresh_e[refresh_e <= t].max())):
                idx = pos_of.get(int(s_time))
                vals = state_np[idx] if idx is not None else np.full(len(STATE_COLS), np.nan)
                for cname, v in zip(STATE_COLS, vals):
                    row[f"{arm}_{cname}"] = v
            for cname in DAILY_COLS:
                row[cname] = ds[cname].iloc[-1]
            row["d_atr_pct"] = np.log(a / prev_c)
            out.append(row)
    return pd.DataFrame(out)


def daily_state_for(target: str) -> pd.DataFrame:
    fr, _ = rc.build_frame(target)
    p, _ = rc.latest_panel()
    ds = pd.DataFrame({"d_vote7": fr["vote7"], "d_log_vix": np.log(p["vix"]),
                       "d_log_rv20": np.log(rc.rv20(p[target]))}, index=fr.index)
    ds["d_atr_pct"] = np.nan  # filled per session from the 5-minute bars
    return ds.loc["2015-01-01":].dropna(subset=["d_vote7", "d_log_vix", "d_log_rv20"])


def arm_cols(arm: str) -> list[str]:
    base = DAILY_COLS + ["log_mins_left"]
    if arm == "a":
        return base
    if arm in ("b_sched", "c_sched"):
        return base + [f"{arm[0]}_{c}" for c in STATE_COLS] + ["fomc", "fomc_post14"]
    return base + [f"{arm}_{c}" for c in STATE_COLS]


def walk_forward(df: pd.DataFrame, cols: list[str], target: str) -> np.ndarray:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = df[cols].to_numpy()
    y = df[target].to_numpy()
    years = df["session"].dt.year.to_numpy()
    pred = np.full(len(df), np.nan)
    for yr in range(2018, years.max() + 1):
        te, tr = years == yr, years < yr
        if te.sum() == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        if target == "y_up":
            m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            pred[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        else:
            m = LinearRegression().fit(sc.transform(X[tr]), y[tr])
            pred[te] = m.predict(sc.transform(X[te]))
    return pred


def row_loss(y: np.ndarray, p: np.ndarray, target: str) -> np.ndarray:
    if target == "y_up":
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return (y - p) ** 2


def day_bootstrap(diff: np.ndarray, sess: np.ndarray, rng, draws: int = BOOT) -> np.ndarray:
    """Bootstrap of the mean of per-row differences, resampling whole sessions."""
    codes, uniq = pd.factorize(sess)
    s = np.bincount(codes, weights=diff)
    c = np.bincount(codes)
    out = np.empty(draws)
    for k in range(draws):
        pick = rng.integers(0, len(s), len(s))
        out[k] = s[pick].sum() / c[pick].sum()
    return out


def analyse(df: pd.DataFrame, sym: str, rng) -> tuple[list[dict], dict]:
    df = df.dropna(subset=list({c for a in ("a", "b", "c", "d", "e", "b_sched") for c in arm_cols(a)})).reset_index(drop=True)
    oos = df["session"].dt.year >= 2018
    rows, decision = [], {}
    for target in ("y_up", "y_range"):
        losses = {}
        for arm in ("a", "b", "c", "d", "e", "b_sched", "c_sched"):
            p = walk_forward(df, arm_cols(arm), target)
            losses[arm] = row_loss(df[target].to_numpy(), p, target)
        sess = df["session"].to_numpy()
        m = oos.to_numpy()
        base = losses["a"][m]
        boots = {arm: day_bootstrap(base - losses[arm][m], sess[m], rng) for arm in ("b", "c", "d", "e")}
        gain = {arm: float((base - losses[arm][m]).mean()) for arm in ("b", "c", "d", "e")}
        for arm in ("b", "c", "d", "e"):
            ratio = boots[arm] / boots["d"]
            rows.append({"symbol": sym, "target": target, "arm": arm, "rows": int(m.sum()),
                         "loss_a": float(base.mean()), "gain": gain[arm],
                         "gain_lo": float(np.percentile(boots[arm], 2.5)), "gain_hi": float(np.percentile(boots[arm], 97.5)),
                         "share_of_d": gain[arm] / gain["d"] if gain["d"] != 0 else np.nan,
                         "share_of_d_lo": float(np.percentile(ratio, 2.5)), "share_of_d_hi": float(np.percentile(ratio, 97.5))})
        # by decision time bucket (share of d's gain), descriptive
        for lo_t, hi_t in ((600, 660), (660, 780), (780, 870), (870, 931)):
            mm = m & (df["t"].between(lo_t, hi_t - 1)).to_numpy()
            g = {arm: float((losses["a"][mm] - losses[arm][mm]).mean()) for arm in ("b", "c", "d")}
            rows.append({"symbol": sym, "target": target, "arm": f"bucket_{lo_t}_{hi_t}",
                         "rows": int(mm.sum()), "gain_b": g["b"], "gain_c": g["c"], "gain_d": g["d"]})
        # event test: FOMC days, decisions at/after 14:05
        fm = m & (df["fomc"] == 1).to_numpy() & (df["t"] >= 845).to_numpy()
        ev = day_bootstrap(losses["b_sched"][fm] - losses["e"][fm], sess[fm], rng)
        sched = day_bootstrap(losses["b"][fm] - losses["b_sched"][fm], sess[fm], rng)
        decision[target] = {
            "fomc_rows": int(fm.sum()), "fomc_days": int(pd.Series(sess[fm]).nunique()),
            "e_vs_bsched_gain": float((losses["b_sched"][fm] - losses["e"][fm]).mean()),
            "e_vs_bsched_ci": [float(np.percentile(ev, 2.5)), float(np.percentile(ev, 97.5))],
            "sched_vs_b_gain": float((losses["b"][fm] - losses["b_sched"][fm]).mean()),
            "sched_vs_b_ci": [float(np.percentile(sched, 2.5)), float(np.percentile(sched, 97.5))],
            "d_vs_b_on_fomc_post14": float((losses["b"][fm] - losses["d"][fm]).mean()),
        }
        pick = None
        for arm in ("b", "c", "d"):
            if gain["d"] > 0 and gain[arm] >= 0.8 * gain["d"]:
                pick = arm
                break
        decision[target]["coarsest_ge_80pct"] = pick
        rng2 = np.random.default_rng(SEED + 1)  # follow-up only
        fu = day_bootstrap(losses["c"][fm] - losses["c_sched"][fm], sess[fm], rng2)
        fu_all = day_bootstrap(losses["c"][m] - losses["c_sched"][m], sess[m], rng2)
        decision[target]["followup_csched_vs_c_fomc_post14"] = {
            "gain": float((losses["c"][fm] - losses["c_sched"][fm]).mean()),
            "ci": [float(np.percentile(fu, 2.5)), float(np.percentile(fu, 97.5))]}
        decision[target]["followup_csched_vs_c_all_rows"] = {
            "gain": float((losses["c"][m] - losses["c_sched"][m]).mean()),
            "ci": [float(np.percentile(fu_all, 2.5)), float(np.percentile(fu_all, 97.5))]}
        decision[target]["gains"] = gain
    return rows, decision


def fomc_check(fomc: set) -> pd.DataFrame:
    """In-data check of the calendar: the 14:00-14:05 SPY bar range vs that day's median 5-min range."""
    bars = to_sessions(pd.read_parquet(BARS5 / "SPY.parquet"))
    bars["rng"] = (bars["high"] - bars["low"]) / bars["close"]
    med = bars.groupby("session")["rng"].median()
    at14 = bars[bars["bar_min"] == 840].set_index("session")["rng"]
    ratio = (at14 / med).dropna()
    rows = []
    for d in sorted(fomc):
        if d in ratio.index:
            rows.append({"date": str(d.date()), "bar14_over_median": float(ratio[d]),
                         "pct_rank_all_days": float((ratio < ratio[d]).mean() * 100)})
    return pd.DataFrame(rows)


def main() -> int:
    rc.OUT.mkdir(exist_ok=True)
    warnings.simplefilter("ignore")
    fomc = fomc_days()
    chk = fomc_check(fomc)
    chk.to_csv(rc.OUT / "e5_fomc_calendar_check.csv", index=False)
    print("FOMC calendar check: days", len(chk), "median 14:00 bar / day median",
          round(chk["bar14_over_median"].median(), 2), "share below 1.5x:",
          round((chk["bar14_over_median"] < 1.5).mean(), 3), flush=True)
    vixy = fetch_vixy()
    print("VIXY rows", len(vixy), flush=True)
    rng = np.random.default_rng(SEED)
    all_rows, decisions = [], {}
    for sym, target in (("SPY", "spx"), ("QQQ", "ndx")):
        df = session_rows(sym, fomc, vixy, daily_state_for(target))
        df.to_parquet(rc.OUT / f"e5_rows_{sym}.parquet")
        print(sym, "rows", len(df), "sessions", df["session"].nunique(), df["session"].min().date(),
              df["session"].max().date(), "null share", df.isna().mean().round(3).max(),
              "vixy bar missing (carried)", round(df["vixy_bar_missing"].mean(), 4), flush=True)
        rows, dec = analyse(df, sym, rng)
        all_rows += rows
        decisions[sym] = dec
        print(json.dumps(dec, indent=1, default=str), flush=True)
    res = pd.DataFrame(all_rows)
    res.to_csv(rc.OUT / "e5_results.csv", index=False)
    (rc.OUT / "e5_summary.json").write_text(json.dumps(decisions, indent=2, default=str))
    pd.set_option("display.width", 250)
    print(res.round(6).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
