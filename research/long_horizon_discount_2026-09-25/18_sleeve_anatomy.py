"""Anatomy of the top-300 momentum sleeve: which picks hurt, why whole months go red, and what moves it (2026-10-07).

The user's questions after the Part 7 chart: why was BBAI picked three times while it fell; are the bad months
caused by a few bad stocks or by everything falling together; do pump-and-dump names get in; is there a
common factor. This script is DIAGNOSTIC and tests no rule (19_quality_and_timing.py does that). It runs on
the full point-in-time history 2019-2026, because one year of 13 periods cannot separate a pattern from luck.

Unit: a "pick" is one name in one 4-week holding period (entry open -> next entry open). All four weekly
offsets of the 4-week grid are pooled, and CIs bootstrap 6-month blocks of decision dates.

    .venv/bin/python research/long_horizon_discount_2026-09-25/18_sleeve_anatomy.py
"""
from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from scripts.shadow.momentum_shadow_ledger import ANCHOR  # noqa: E402


def _load(name: str, file: str):
    spec = spec_from_file_location(name, HERE / file)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rb = _load("rb", "05_rotation_backtest.py")
wf = _load("wf", "03_walkforward.py")
pb = rb.pb
N_TOP, N_LIQ, STEP = 20, 300, 4
FACTORS = ["SPY", "QQQ", "SMH", "ARKK"]
FEATS = ["mom_12_1", "dv_rank", "rv_63", "ret_21", "ret_63", "dist_sma200", "pct_above_sma200_252"]


def build_context() -> dict:
    """Point-in-time liquid-300 rows at every weekly decision date, prices, and each row's 4-week forward return."""
    pb.BARS, pb.OUT = pb.PIT_BARS, pb.PIT_OUT
    p = pd.read_parquet(pb.OUT, columns=["date", "ticker"] + FEATS)
    p = p[(p["date"] >= rb.START - pd.Timedelta(days=7)) & (p["dv_rank"] <= N_LIQ)].dropna(subset=["mom_12_1"]).reset_index(drop=True)
    O_df, C_df = rb.price_matrices(sorted(set(p["ticker"]) | set(rb.BENCH) | set(FACTORS)),
                                   pb.PIT_OUT.parent / "px_open_pit.parquet", pb.PIT_OUT.parent / "px_close_pit.parquet")
    raw_close = C_df.copy()
    C_df = C_df.ffill()
    O_df = O_df.fillna(C_df.shift(1))
    O_df["CASH"] = C_df["CASH"] = 1.0
    O_df = O_df[C_df.columns]
    cal, col = C_df.index, {t: i for i, t in enumerate(C_df.columns)}
    O, C, end = O_df.values, C_df.values, len(cal) - 1
    dates = pd.DatetimeIndex(sorted(p["date"].unique()))
    dates = dates[dates >= rb.START]
    p = p[p["date"].isin(dates)].reset_index(drop=True)
    entry = {d: int(cal.searchsorted(d, side="right")) for d in dates}
    nxt = {d: (entry[dates[i + STEP]] if i + STEP < len(dates) and entry[dates[i + STEP]] <= end else None)
           for i, d in enumerate(dates)}
    e = p["date"].map(entry).values
    x = p["date"].map(nxt).values
    j = p["ticker"].map(col).values
    ok = e <= end
    exit_px = np.where(pd.isna(x), C[end, j], O[np.where(pd.isna(x), 0, x).astype(int), j])
    p["fwd"] = np.where(ok, exit_px / O[np.minimum(e, end), j] - 1, np.nan)
    p["complete"] = ~pd.isna(x)
    # features of the path that produced the momentum, from daily closes up to the decision date
    r = raw_close.pct_change(fill_method=None)
    up = (r > 0).where(r.notna()).rolling(231, min_periods=150).mean().shift(21)      # share of up days, t-252..t-21
    vol = r.rolling(231, min_periods=150).std().shift(21) * np.sqrt(252)
    di = cal.get_indexer(p["date"])
    p["up_days_12_1"] = up.values[di, j]
    p["vol_12_1"] = vol.values[di, j]
    g = p.groupby("date")
    p["mom_rank"] = g["mom_12_1"].rank(ascending=False, method="first")
    p["vol_pct"] = g["rv_63"].rank(pct=True)
    p["picked"] = p["mom_rank"] <= N_TOP
    # the weekly offset whose 4-week grid is the live ledger's (it runs into the 2026-10-02 decision)
    anchor_monday = pd.Timestamp(ANCHOR) - pd.Timedelta(days=ANCHOR.weekday())
    weeks_before = (anchor_monday - (dates - pd.to_timedelta(dates.weekday, unit="D"))).days // 7
    return {"p": p, "O": O, "C": C, "cal": cal, "col": col, "end": end, "dates": dates, "entry": entry, "nxt": nxt,
            "C_df": C_df, "raw_close": raw_close, "live": dates[weeks_before % STEP == 0]}


def date_ci(per_date: pd.Series) -> tuple[float, float]:
    return tuple(wf.block_ci(per_date.dropna()))


def main() -> None:
    ctx = build_context()
    p, cal, col, O, C, end, dates = ctx["p"], ctx["cal"], ctx["col"], ctx["O"], ctx["C"], ctx["end"], ctx["dates"]
    picks = p[p["picked"] & p["fwd"].notna()].copy()
    lines = [f"=== anatomy of the top-{N_TOP} momentum sleeve in the liquid-{N_LIQ} | point-in-time | {dates.min().date()}.."
             f"{dates.max().date()} | {len(picks):,} picks (name x 4-week period, all 4 weekly offsets pooled) ==="]

    # ---- A. what a single pick looks like ----
    f = picks["fwd"]
    lines.append("\nA. One pick over its 4-week period")
    lines.append(f"   mean {f.mean():+.1%} | median {f.median():+.1%} | lost money {(f < 0).mean():.0%} | "
                 f"worse than -20% {(f < -0.2).mean():.0%} | better than +20% {(f > 0.2).mean():.0%} | better than +50% {(f > 0.5).mean():.1%}")
    # runs of consecutive losing periods for one name inside one weekly offset
    runs = []
    for off in range(STEP):
        d_off = dates[off::STEP]
        s = picks[picks["date"].isin(d_off)].sort_values(["ticker", "date"])
        pos = {d: i for i, d in enumerate(d_off)}
        for _, g in s.groupby("ticker"):
            run, prev = 0, None
            for kk, lost in zip(g["date"].map(pos).values, (g["fwd"] < 0).values):
                if lost and run and prev is not None and kk == prev + 1:
                    run += 1                      # still held, still losing
                else:
                    if run:
                        runs.append(run)
                    run = 1 if lost else 0
                prev = kk
            if run:
                runs.append(run)
    runs = pd.Series(runs)
    lines.append(f"   losing streaks while held: {len(runs):,} streaks; 1 period {(runs == 1).mean():.0%}, 2 {(runs == 2).mean():.0%}, "
                 f"3 or more {(runs >= 3).mean():.0%} (BBAI's three losing periods in a row is the ordinary case, not an error)")

    # ---- B. do the features known at pick time separate good picks from bad? ----
    lines.append("\nB. Picks split into thirds by a feature known at the decision (thirds are formed within each decision date)")
    lines.append(f"   {'feature':26s} {'third':>6s} {'feature median':>14s} {'mean':>7s} {'median':>7s} {'<-20%':>6s} {'>+20%':>6s}   high-minus-low mean [95% CI]")
    feats = {"rv_63": "63d volatility", "dist_sma200": "distance above 200 SMA", "ret_63": "last 3 months' return",
             "ret_21": "last month's return", "mom_12_1": "12-1 momentum itself", "up_days_12_1": "share of up days (smooth)",
             "dv_rank": "liquidity rank (1 = most)"}
    for c, label in feats.items():
        q = picks.groupby("date")[c].rank(pct=True)
        third = pd.cut(q, [0, 1 / 3, 2 / 3, 1.0], labels=["low", "mid", "high"])
        hi = picks[third == "high"].groupby("date")["fwd"].mean()
        lo = picks[third == "low"].groupby("date")["fwd"].mean()
        ci = date_ci(hi - lo)
        for k in ("low", "mid", "high"):
            s = picks[third == k]
            tail = f"   {(hi - lo).mean():+.2%} [{ci[0]:+.2%}, {ci[1]:+.2%}]" if k == "high" else ""
            lines.append(f"   {label if k == 'low' else '':26s} {k:>6s} {s[c].median():14.2f} {s['fwd'].mean():+7.1%} "
                         f"{s['fwd'].median():+7.1%} {(s['fwd'] < -0.2).mean():6.0%} {(s['fwd'] > 0.2).mean():6.0%}{tail}")
    pump = (picks["ret_63"] > 1.0) | (picks["dist_sma200"] > 1.0)
    lines.append(f"   'pump' picks (more than doubled in 3 months, or price more than 2x its 200-day average): {pump.mean():.1%} of picks")
    for k, m in (("pump", pump), ("the rest", ~pump)):
        s = picks[m]["fwd"]
        lines.append(f"     {k:9s} mean {s.mean():+.1%} | median {s.median():+.1%} | <-20% {(s < -0.2).mean():.0%} | >+20% {(s > 0.2).mean():.0%} | >+50% {(s > 0.5).mean():.1%}")
    dpm = picks[pump].groupby("date")["fwd"].mean() - picks[~pump].groupby("date")["fwd"].mean()
    ci = date_ci(dpm)
    lines.append(f"     pump minus rest, per decision date: {dpm.mean():+.2%} [{ci[0]:+.2%}, {ci[1]:+.2%}] on {dpm.notna().sum()} dates that had both")

    # pump x profitability. Fundamentals exist for today's tickers only, so "unknown" partly means "later delisted".
    fund = pd.read_parquet(pb.PIT_OUT, columns=["date", "ticker", "ni_positive"])
    pk = picks.merge(fund, on=["date", "ticker"], how="left").assign(pump=pump.values)
    pk["prof"] = np.where(pk["ni_positive"].isna(), "unknown", np.where(pk["ni_positive"] == 1, "profitable", "loss-making"))
    lines.append(f"   pump x latest filed net income (known for {pk.ni_positive.notna().mean():.0%} of picks):")
    for is_pump in (True, False):
        for k in ("profitable", "loss-making", "unknown"):
            s = pk[(pk["pump"] == is_pump) & (pk["prof"] == k)]["fwd"]
            lines.append(f"     {'pump' if is_pump else 'not pump':9s} {k:12s} n {len(s):5d} | mean {s.mean():+.1%} | median {s.median():+.1%} | "
                         f"<-20% {(s < -0.2).mean():.0%} | >+20% {(s > 0.2).mean():.0%}")
    lossy = pk["pump"] & (pk["prof"] == "loss-making")
    dl = pk[lossy].groupby("date")["fwd"].mean() - pk[~lossy].groupby("date")["fwd"].mean()
    ci = date_ci(dl)
    lines.append(f"     loss-making pump minus every other pick, per decision date: {dl.mean():+.2%} [{ci[0]:+.2%}, {ci[1]:+.2%}] on {dl.notna().sum()} dates")

    # ---- C. are bad periods a few bad stocks or everything at once? ----
    per = picks.groupby("date").agg(sleeve=("fwd", "mean"), breadth=("fwd", lambda s: (s > 0).mean()), median=("fwd", "median"),
                                    worst=("fwd", "min"), best=("fwd", "max"))
    for fac in FACTORS:
        e = np.array([ctx["entry"][d] for d in per.index])
        x = np.array([ctx["nxt"][d] if ctx["nxt"][d] is not None else -1 for d in per.index])
        ex = np.where(x >= 0, O[np.maximum(x, 0), col[fac]], C[end, col[fac]])
        per[fac] = ex / O[e, col[fac]] - 1
    lines.append(f"\nC. Whole periods ({len(per)} decision dates)")
    lines.append(f"   correlation of the sleeve's period return with breadth (share of its 20 names that rose): {per.sleeve.corr(per.breadth):.2f}")
    q = per["sleeve"].rank(pct=True)
    for label, m in (("worst 10% of periods", q <= 0.1), ("middle 80%", (q > 0.1) & (q < 0.9)), ("best 10% of periods", q >= 0.9)):
        s = per[m]
        lines.append(f"   {label:21s} sleeve {s.sleeve.mean():+6.1%} | names up {s.breadth.mean():4.0%} | median name {s['median'].mean():+6.1%} | "
                     f"SPY {s.SPY.mean():+5.1%} QQQ {s.QQQ.mean():+5.1%} SMH {s.SMH.mean():+5.1%} ARKK {s.ARKK.mean():+5.1%}")
    # leave-out test: remove each period's 3 worst names, how much of the bad periods remains?
    trimmed = picks.groupby("date")["fwd"].apply(lambda s: s.nlargest(N_TOP - 3).mean())
    bad = q <= 0.1
    lines.append(f"   worst 10% of periods with each period's 3 worst names REMOVED in hindsight: {trimmed[bad].mean():+.1%} "
                 f"(was {per.sleeve[bad].mean():+.1%}); so most of a bad period is NOT a few blow-ups")

    # daily co-movement
    live = ctx["live"]
    es = [ctx["entry"][d] for d in live if ctx["entry"][d] <= end]
    ws = [{t: 1 / N_TOP for t in p[(p["date"] == d) & p["picked"]]["ticker"]} for d in live[:len(es)]]
    v = rb.simulate(ws, es, O, C, col, end)[0]
    idx = cal[es[0]:]
    sleeve_r = pd.Series(np.diff(np.r_[1.0, v]) / np.r_[1.0, v][:-1], index=idx)
    fr = ctx["C_df"][FACTORS].pct_change(fill_method=None).reindex(idx)
    lines.append("   daily returns of the sleeve explained by an index (R-squared, beta), full history | last 12 months:")
    for fac in FACTORS:
        out = []
        for sl in (slice(None), slice(-252, None)):
            a, b = sleeve_r.iloc[sl], fr[fac].iloc[sl]
            beta = np.cov(a, b)[0, 1] / b.var()
            out.append(f"R2 {a.corr(b) ** 2:.2f} beta {beta:.2f}")
        lines.append(f"     {fac:5s} {out[0]}  |  {out[1]}")
    # average pairwise correlation among the 20 held names, trailing 60 sessions at each decision
    rr = ctx["raw_close"].pct_change(fill_method=None)
    pc = []
    for d in live:
        names = p[(p["date"] == d) & p["picked"]]["ticker"].tolist()
        i = cal.get_loc(d)
        cm = rr.iloc[max(0, i - 59):i + 1][names].corr().values
        pc.append(cm[np.triu_indices(len(names), 1)].mean())
    pc = pd.Series(pc, index=live)
    uni = []
    rng = np.random.default_rng(11)
    for d in dates[::STEP * 6]:
        names = rng.choice(p[p["date"] == d]["ticker"].values, N_TOP, replace=False)
        i = cal.get_loc(d)
        cm = rr.iloc[max(0, i - 59):i + 1][list(names)].corr().values
        uni.append(np.nanmean(cm[np.triu_indices(N_TOP, 1)]))
    lines.append(f"   average pairwise correlation among the 20 held names (trailing 60 days): median {pc.median():.2f}, "
                 f"range {pc.min():.2f}-{pc.max():.2f}, last 12 months {pc.iloc[-13:].mean():.2f} | 20 random liquid-300 names: {np.mean(uni):.2f}")

    # ---- D. the last 12 months, period by period, against the factors ----
    last = per.loc[[d for d in live if d in per.index][-13:]]
    lines.append("\nD. The last 13 periods on the live schedule (decision date; sleeve; names up; indexes over the same window)")
    for d, r_ in last.iterrows():
        lines.append(f"   {d.date()}  sleeve {r_.sleeve:+6.1%}  up {r_.breadth:4.0%}  median {r_['median']:+6.1%}  best {r_.best:+5.0%} worst {r_.worst:+5.0%}  |  "
                     f"SPY {r_.SPY:+5.1%}  QQQ {r_.QQQ:+5.1%}  SMH {r_.SMH:+5.1%}  ARKK {r_.ARKK:+5.1%}")

    # ---- E. does anything known at the decision warn of a bad period? ----
    st = pd.DataFrame(index=dates)
    eq = pd.Series(v, index=idx)
    st["sleeve trailing 3-month return"] = (eq / eq.shift(63)).reindex(dates, method="ffill") - 1
    st["sleeve drawdown from its peak"] = (eq / eq.cummax()).reindex(dates, method="ffill") - 1
    st["pairwise correlation of picks"] = pc.reindex(dates, method="ffill")
    st["share of picks that are 'pump'"] = p[p["picked"]].assign(pump=lambda d_: (d_["ret_63"] > 1) | (d_["dist_sma200"] > 1)).groupby("date")["pump"].mean()
    st["median volatility of picks"] = p[p["picked"]].groupby("date")["rv_63"].median()
    st["SPY trailing 3-month return"] = (ctx["C_df"]["SPY"] / ctx["C_df"]["SPY"].shift(63) - 1).reindex(dates)
    tgt = (per["sleeve"] - per["SPY"]).reindex(dates)
    lines.append("\nE. Known at the decision vs the NEXT period's sleeve-minus-SPY return (rank correlation; n = "
                 f"{tgt.notna().sum()} overlapping periods, ~{tgt.notna().sum() // STEP} independent)")
    for c in st:
        ok = st[c].notna() & tgt.notna()
        ic = st.loc[ok, c].rank().corr(tgt[ok].rank())
        hi = tgt[ok][st.loc[ok, c] >= st.loc[ok, c].quantile(0.8)].mean()
        lo = tgt[ok][st.loc[ok, c] <= st.loc[ok, c].quantile(0.2)].mean()
        lines.append(f"   {c:34s} rank corr {ic:+.2f} | next period when it is in its top fifth {hi:+.1%}, bottom fifth {lo:+.1%}")
    lines.append(f"   a rank correlation needs about +/-{2 / np.sqrt(tgt.notna().sum() / STEP):.2f} to stand out with ~{tgt.notna().sum() // STEP} independent periods")
    txt = "\n".join(lines)
    print(txt)
    (HERE / "18_results.txt").write_text(txt)


if __name__ == "__main__":
    main()
