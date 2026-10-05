"""Rotation backtest, run 2026-09-29: daily marked-to-market portfolios vs SPY buy-and-hold,
rebalanced monthly / quarterly / semiannually / annually (every 4/13/26/52 weekly panel dates).

Why: the CAGRs in 03/04 come from ONE rebalance phase and period-end marks. That noise is
visible in 03 alone: mom_12_1 shows 44% at the 63d rebalance and 28% at 126d. Here every
phase offset is run (a quarterly schedule has 13 weekly start points). The table reports the
median phase and the worst and best phases, and the drawdown uses daily closes.

Pre-registered before the first run (README Part 3):
  primary new arm  leader_pullback: 12-1 momentum in the top decile of the liquid-1000 AND the
                   hand rule (low held within 5% of the 200 SMA over 20d, TMO <= -9 in the last
                   10d, turned up <= 5d ago). Up to 20 names by momentum; empty slots are held in SPY.
  secondary        leader_dip: top-decile momentum AND >= 10% off the 63d high AND the same
                   200-SMA hold. No TMO condition. Empty slots are held in SPY.
                   mom_trend: mom_12_1 top 20, but 100% cash (0% return) whenever SPY closes
                   below its 200 SMA on the decision date.
  controls         high_vol: top 20 by 63d realized vol. Added after 06 showed momentum's top 20 sits at
                   the 88th vol percentile. If mom_12_1 does no better, its edge is just a vol tilt.
  robustness       mom_top300: mom_12_1 top 20 within the 300 most liquid names, where delistings are rare,
                   so survivorship matters less. core_satellite: 70% SPY + 30% mom_12_1 top 20. It is a
                   sizing blend of two arms above, not a new signal. Both were added after the smoke run.
  references       spy / qqq / rsp buy-and-hold; ew_universe (equal-weight liquid-1000);
                   mom_12_1 top 20; hand_rule top 20 (as defined in 03); lean_ml and lean_perm
                   (06 OOF scores; the 63d model for monthly/quarterly, the 126d model for longer).
Execution: decide at the Friday close, then trade at the next session's open. Equal weight, and
weights drift within a period. 20 bp round trip is charged on one-way turnover.
Price fills: a name with no bar is marked at its last close. A missing open uses the previous
close. The ETFs' adjusted bars include dividends, so SPY/QQQ/RSP are total-return benchmarks.

Added 2026-10-03:
  breadth          mom_top50 / mom_top100 (equal-weight top 50 / 100 by 12-1 momentum) and core_sat_50
                   (70% SPY + 30% top 50). These vary the top-N parameter and add no new signal.
  exclusion        mom_xmiss / core_sat_xmiss: the mom_12_1 and core_satellite arms, skipping any name with a
                   "miss + flat reaction" earnings report (07: surprise < -2%, |reaction| <= 2%) whose reaction
                   window closed within the last 91 calendar days. Pre-registered in README Part 3 next-steps;
                   the rule comes from 07, whose effect was measured on the same years, so this is in-sample.
  --pit            point-in-time universe: every symbol that traded since 2016, including delisted names
                   and names today's universe dropped (02 --pit). A held name whose bars stop is marked at
                   its last close and sits as cash until the next rebalance.
  --delist-zero    a pessimistic bound for --pit: a name that stopped trading in distress (last close < $2,
                   or < 20% of its 252d high, or down > 50% in its final 5 sessions) is worth ZERO from the
                   session after its last bar. Buyouts keep their last price.

    .venv/bin/python research/long_horizon_discount_2026-09-25/05_rotation_backtest.py
    .venv/bin/python research/long_horizon_discount_2026-09-25/05_rotation_backtest.py --pit [--delist-zero]
"""
from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = spec_from_file_location("pb", HERE / "02_build_panel.py")
pb = module_from_spec(_spec)
_spec.loader.exec_module(pb)

PX_OPEN = pb.OUT.parent / "px_open.parquet"     # derived wide caches of the research bars
PX_CLOSE = pb.OUT.parent / "px_close.parquet"
START = pd.Timestamp("2019-01-01")
N_TOP = 20
COST_RT = 0.002
FREQS = {"monthly": 4, "quarterly": 13, "semiannual": 26, "annual": 52}
ML_HOLD = {"monthly": 63, "quarterly": 63, "semiannual": 126, "annual": 126}
BENCH = ["SPY", "QQQ", "RSP"]
SEED = 11


def price_matrices(tickers: list[str], px_open: Path = PX_OPEN, px_close: Path = PX_CLOSE) -> tuple[pd.DataFrame, pd.DataFrame]:
    if px_open.exists() and px_close.exists():
        O, C = pd.read_parquet(px_open), pd.read_parquet(px_close)
        if set(tickers) <= set(C.columns):
            return O, C
    opens, closes = {}, {}
    for t in tickers:
        df = pb.load_bars(t)
        if df is None:
            raise FileNotFoundError(f"no research bars for {t}")
        if not df.index.is_unique:
            raise ValueError(f"{t}: duplicate session dates in research bars")
        opens[t], closes[t] = df["open"], df["close"]
    cal = closes["SPY"].index
    O, C = pd.DataFrame(opens).reindex(cal), pd.DataFrame(closes).reindex(cal)
    O.to_parquet(px_open)
    C.to_parquet(px_close)
    return O, C


def distressed_delistings(C_df: pd.DataFrame, min_gap: int = 10) -> tuple[pd.Series, pd.Series]:
    """(last bar date of every name that stopped trading, the subset that stopped in distress)."""
    last = C_df.apply(pd.Series.last_valid_index)
    gone = last[last < C_df.index[-min_gap]]
    bad = {}
    for t, d in gone.items():
        s = C_df[t].dropna()
        if s.iloc[-1] < 2 or s.iloc[-1] < 0.2 * s.iloc[-252:].max() or (len(s) > 5 and s.iloc[-1] / s.iloc[-6] < 0.5):
            bad[t] = d
    return gone, pd.Series(bad, dtype="datetime64[ns]")


def flag_miss_flat(p: pd.DataFrame) -> pd.DataFrame:
    """miss_flat = a shrugged-off earnings miss became known in the 91 days up to the decision date."""
    spec = spec_from_file_location("er", HERE / "07_earnings_reaction.py")
    er = module_from_spec(spec)
    spec.loader.exec_module(er)
    ev = er.build_events()
    ev = ev[(ev["s_grp"] == "miss") & (ev["r_grp"] == "flat")]
    known = ev.rename(columns={"known_date": "known"})[["ticker", "known"]].sort_values("known")
    out = pd.merge_asof(p.sort_values("date"), known, left_on="date", right_on="known", by="ticker",
                        direction="backward", tolerance=pd.Timedelta(days=91))
    out["miss_flat"] = out["known"].notna()
    return out.drop(columns="known")


def build_scores(p: pd.DataFrame, with_events: bool = False) -> pd.DataFrame:
    mom = p["mom_12_1"].fillna(-9)
    leader = p.groupby("date")["mom_12_1"].rank(pct=True) >= 0.9
    held200 = p["min_low_vs_sma200_20"] > -0.05
    hand = held200 & (p["tmo_min_10"] <= -9) & (p["days_since_tmo_turn"] <= 5)
    p = p.assign(mom=mom, f_leader_pullback=leader & hand,
                 f_leader_dip=leader & held200 & (p["dd_max_63"] <= -0.10),
                 s_hand_rule=hand.astype(float) + 1e-6 * mom)
    p["miss_flat"] = False
    if with_events:
        p = flag_miss_flat(p)
    for h in (63, 126):
        f = HERE / f"06_oof_scores_h{h}.parquet"
        if f.exists():
            s = pd.read_parquet(f).rename(columns={"s_lean_rank": f"s_lean_ml_{h}", "s_lean_perm": f"s_lean_perm_{h}"})
            p = p.merge(s[["date", "ticker", f"s_lean_ml_{h}", f"s_lean_perm_{h}"]], on=["date", "ticker"], how="left")
    return p


def weights_for(arm: str, g: pd.DataFrame, freq: str) -> dict[str, float]:
    def top(col, n=N_TOP):
        names = g[g[col].notna()].nlargest(n, col)["ticker"]
        return {t: 1 / len(names) for t in names} if len(names) else {"CASH": 1.0}

    def flagged(col):
        names = g[g[col]].nlargest(N_TOP, "mom")["ticker"]
        w = {t: 1 / N_TOP for t in names}
        if len(names) < N_TOP:
            w["SPY"] = (N_TOP - len(names)) / N_TOP
        return w

    if arm in ("spy", "qqq", "rsp"):
        return {arm.upper(): 1.0}
    if arm == "ew_universe":
        return {t: 1 / len(g) for t in g["ticker"]}
    if arm == "mom_12_1":
        return top("mom")
    if arm == "high_vol":
        return top("rv_63")
    if arm == "mom_top300":
        names = g[g["dv_rank"] <= 300].nlargest(N_TOP, "mom")["ticker"]
        return {t: 1 / len(names) for t in names}
    if arm == "core_satellite":
        return {"SPY": 0.7, **{t: 0.3 * w for t, w in top("mom").items()}}
    if arm in ("mom_top50", "mom_top100"):
        return top("mom", int(arm[7:]))
    if arm in ("mom_xmiss", "core_sat_xmiss"):
        names = g[~g["miss_flat"]].nlargest(N_TOP, "mom")["ticker"]
        w = {t: 1 / len(names) for t in names}
        return w if arm == "mom_xmiss" else {"SPY": 0.7, **{t: 0.3 * x for t, x in w.items()}}
    if arm == "core_sat_300":   # added after the first --pit run: the blend built on the sturdier top-300 arm
        return {"SPY": 0.7, **{t: 0.3 * w for t, w in weights_for("mom_top300", g, freq).items()}}
    if arm == "core_sat_50":
        return {"SPY": 0.7, **{t: 0.3 * w for t, w in top("mom", 50).items()}}
    if arm == "mom_trend":
        return {"CASH": 1.0} if g["spy_dist_sma200"].iloc[0] < 0 else top("mom")
    if arm == "hand_rule":
        return top("s_hand_rule")
    if arm in ("leader_pullback", "leader_dip"):
        return flagged(f"f_{arm}")
    if arm in ("lean_ml", "lean_perm"):
        col = f"s_{arm}_{ML_HOLD[freq]}"
        return top(col) if col in g and g[col].notna().any() else {}
    raise KeyError(arm)


def simulate(weights: list[dict], entries: list[int], O: np.ndarray, C: np.ndarray, col: dict,
             end: int) -> tuple[np.ndarray, float]:
    """Close-marked equity from entries[0] to `end` (inclusive), and total one-way turnover."""
    vals = np.full(end + 1, np.nan)
    V, w_old, turnover = 1.0, None, 0.0
    for k, (w, e) in enumerate(zip(weights, entries)):
        nxt = entries[k + 1] if k + 1 < len(entries) else None
        unfilled = [x for x in w if O[e, col[x]] == 0]   # zeroed by --delist-zero before the entry (YNDX halt)
        if unfilled:                                      # the order cannot fill, so that weight stays in cash
            w = {x: v for x, v in w.items() if x not in unfilled} | {"CASH": w.get("CASH", 0) + sum(w[x] for x in unfilled)}
        if w_old is None:
            t, V = 1.0, V * (1 - COST_RT / 2)
        else:
            t = 0.5 * sum(abs(w.get(x, 0) - w_old.get(x, 0)) for x in set(w) | set(w_old))
            V *= 1 - t * COST_RT
        turnover += t
        idx = np.array([col[x] for x in w])
        wt = np.array(list(w.values()))
        e0 = O[e, idx]
        if not np.all(np.isfinite(e0) & (e0 > 0)):
            bad = [x for x, v in zip(w, e0) if not (np.isfinite(v) and v > 0)]
            raise ValueError(f"no entry price at session {e} for {bad}")
        stop = nxt - 1 if nxt is not None else end
        vals[e:stop + 1] = V * (C[e:stop + 1, idx] / e0) @ wt
        if nxt is not None:
            g = O[nxt, idx] / e0
            V *= g @ wt
            dw = wt * g
            w_old = dict(zip(w, dw / dw.sum()))
    return vals[entries[0]:], turnover


def stats(v: np.ndarray, b: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    r = np.diff(np.r_[1.0, v]) / np.r_[1.0, v][:-1]
    rb = np.diff(np.r_[1.0, b]) / np.r_[1.0, b][:-1]
    yrs = len(v) / 252
    yearly = lambda x: pd.Series(x, index=dates).groupby(dates.year).last().pct_change()  # noqa: E731
    yv, yb = yearly(v).iloc[1:], yearly(b).iloc[1:]  # drops the partial first year; last year is YTD
    return {"cagr": v[-1] ** (1 / yrs) - 1, "bcagr": b[-1] ** (1 / yrs) - 1,
            "sharpe": r.mean() / r.std() * np.sqrt(252), "bsharpe": rb.mean() / rb.std() * np.sqrt(252),
            "mdd": (v / np.maximum.accumulate(v) - 1).min(), "beta": (beta := np.cov(r, rb)[0, 1] / rb.var()),
            "alpha": (r - beta * rb).mean() * 252, "ir": (r - rb).mean() / (r - rb).std() * np.sqrt(252),
            "yrs_beat": float((yv > yb).mean()), "active": pd.Series(r - rb, index=dates)}


def block_ci(a: pd.Series, block=126, n=2000) -> tuple[float, float]:
    x = a.values
    nb = len(x) // block
    blocks = x[: nb * block].reshape(nb, block)
    rng = np.random.default_rng(SEED)
    boots = blocks[rng.integers(0, nb, (n, nb))].mean(axis=(1, 2)) * 252
    return tuple(np.percentile(boots, [2.5, 97.5]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pit", action="store_true", help="point-in-time universe (run 02 --pit first)")
    ap.add_argument("--delist-zero", action="store_true", help="distressed delistings go to zero (needs --pit)")
    ap.add_argument("--arms", nargs="*", help="run only these arms")
    ap.add_argument("--freqs", nargs="*", default=list(FREQS), choices=list(FREQS))
    ap.add_argument("--tag", default="", help="results file suffix; defaults to the mode")
    args = ap.parse_args()
    if args.delist_zero and not args.pit:
        ap.error("--delist-zero needs --pit")
    px_open, px_close = PX_OPEN, PX_CLOSE
    if args.pit:
        pb.BARS, pb.OUT = pb.PIT_BARS, pb.PIT_OUT
        px_open, px_close = pb.PIT_OUT.parent / "px_open_pit.parquet", pb.PIT_OUT.parent / "px_close_pit.parquet"
    tag = args.tag or ("pit_delist0" if args.delist_zero else "pit" if args.pit else "")
    p = pd.read_parquet(pb.OUT)
    want_events = bool(args.arms) and any(a.endswith("xmiss") for a in args.arms)
    p = build_scores(p[p["date"] >= START - pd.Timedelta(days=7)], with_events=want_events)
    O_df, C_df = price_matrices(sorted(set(p["ticker"]) | set(BENCH)), px_open, px_close)
    gone, bad = distressed_delistings(C_df)
    C_df = C_df.ffill()
    O_df = O_df.fillna(C_df.shift(1))
    if args.delist_zero:
        for t, d in bad.items():
            C_df.loc[C_df.index > d, t] = 0.0
            O_df.loc[O_df.index > d, t] = 0.0
    O_df["CASH"] = C_df["CASH"] = 1.0
    O_df = O_df[C_df.columns]  # one column index for both arrays
    cal = C_df.index
    col = {t: i for i, t in enumerate(C_df.columns)}
    O, C = O_df.values, C_df.values
    end = len(cal) - 1
    dec_dates = pd.DatetimeIndex(sorted(p.loc[p["date"] >= START, "date"].unique()))
    by_date = dict(tuple(p.groupby("date")))
    arms = ["spy", "qqq", "rsp", "ew_universe", "mom_12_1", "mom_top300", "core_satellite", "mom_trend",
            "high_vol", "hand_rule",
            "leader_pullback", "leader_dip", "lean_ml", "lean_perm", "mom_top50", "mom_top100", "core_sat_50",
            "core_sat_300"]
    if want_events:  # only on request: building the earnings events takes ~1 minute
        arms += ["mom_xmiss", "core_sat_xmiss"]
    if args.pit:  # 06 has no OOF scores for names outside today's universe
        arms = [a for a in arms if not a.startswith("lean_")]
    if args.arms:
        arms = [a for a in arms if a in args.arms]

    universe = "point-in-time universe" + (", distressed delistings -> 0" if args.delist_zero else "") \
        if args.pit else "today's-tickers universe"
    lines = [f"=== rotation backtest | {universe} | decisions {dec_dates.min().date()}..{dec_dates.max().date()} | "
             f"marks to {cal[end].date()} | top {N_TOP}, {COST_RT:.1%} round trip | median phase [worst, best] ==="]
    held = p[p["date"] >= START]
    held = held[held.groupby("date")["mom"].rank(ascending=False, method="first") <= N_TOP]
    held_gone = held[held["ticker"].map(gone).notna()]
    stop_soon = held_gone[(held_gone["ticker"].map(gone) - held_gone["date"]).dt.days <= 28]
    lines.append(f"names that stopped trading before the last {10} sessions: {len(gone)} of {C_df.shape[1] - 1}, "
                 f"{len(bad)} in distress | weekly momentum top-{N_TOP} picks that stopped trading within 4 weeks: "
                 f"{len(stop_soon)} ({stop_soon['ticker'].nunique()} names, "
                 f"{stop_soon['ticker'].isin(bad.index).sum()} picks in distress)")
    if want_events:
        top = held  # weekly momentum top-N picks
        lines.append(f"miss+flat flag: {p.loc[p['date'] >= START, 'miss_flat'].mean():.1%} of universe rows, "
                     f"{top['miss_flat'].mean():.1%} of weekly momentum top-{N_TOP} picks")
    yearly_rows = {}
    for freq, k in ((f, FREQS[f]) for f in args.freqs):
        lines.append(f"\n--- {freq}: rebalance every {k} weeks, {k} phases ---")
        lines.append(f"{'arm':16s} {'CAGR med [min, max]':>24s} {'SPY':>6s} {'active/yr [95% CI]':>24s} "
                     f"{'Sharpe':>6s} {'SPY':>5s} {'MDD med/worst':>14s} {'beta':>5s} {'alpha':>6s} {'IR':>5s} {'yrs>SPY':>7s} "
                     f"{'turn/yr':>7s} {'names':>5s}")
        for arm in arms:
            res, actives, turns, names = [], [], [], []
            for ph in range(k):
                ds = dec_dates[ph::k]
                ws = [weights_for(arm, by_date[d], freq) for d in ds]
                keep = [i for i, w in enumerate(ws) if w]
                if len(keep) < len(ws):  # lean_ml has no score before its first test year
                    continue
                entries = [cal.searchsorted(d, side="right") for d in ds]
                entries = [e for e in entries if e <= end]
                ws = ws[:len(entries)]
                v, turn = simulate(ws, entries, O, C, col, end)
                b, _ = simulate([{"SPY": 1.0}], entries[:1], O, C, col, end)
                st = stats(v, b, cal[entries[0]:])
                res.append(st)
                actives.append(st["active"])
                turns.append(turn / (len(v) / 252))
                names.append(np.mean([sum(1 for t in w if t not in ("SPY", "CASH")) for w in ws]))
                if ph == 0:
                    yearly_rows[(freq, arm)] = (pd.Series(v, index=cal[entries[0]:]).groupby(lambda d: d.year).last()
                                                .pipe(lambda s: s / s.shift(1).fillna(1.0) - 1))
            if not res:
                lines.append(f"{arm:16s} (no scores)")
                continue
            df = pd.DataFrame(res)
            act = pd.concat(actives, axis=1).dropna().mean(axis=1)  # tranche-average active return
            lo, hi = block_ci(act)
            lines.append(
                f"{arm:16s} {df.cagr.median():+7.1%} [{df.cagr.min():+6.1%},{df.cagr.max():+6.1%}] "
                f"{df.bcagr.median():+6.1%} {act.mean() * 252:+7.1%} [{lo:+6.1%},{hi:+6.1%}] "
                f"{df.sharpe.median():6.2f} {df.bsharpe.median():5.2f} {df.mdd.median():+6.1%}/{df.mdd.min():+6.1%} "
                f"{df.beta.median():5.2f} {df.alpha.median():+6.1%} {df.ir.median():5.2f} {df.yrs_beat.median():7.0%} {np.median(turns):7.1f} {np.median(names):5.1f}")
    lines.append("\ncalendar-year return, phase 0 (2019 starts at the first January decision; 2026 is YTD):")
    yr = pd.DataFrame({f"{a}|{f[:3]}": s for (f, a), s in yearly_rows.items()
                       if f in ("monthly", "quarterly") and a in
                       ("spy", "qqq", "mom_12_1", "mom_top300", "core_satellite", "hand_rule", "leader_pullback",
                        "lean_ml", "mom_top50", "mom_top100", "core_sat_50", "core_sat_300")})
    lines.append(yr.map(lambda x: f"{x:+.0%}").to_string())
    txt = "\n".join(lines)
    print(txt)
    (HERE / f"05_results{'_' + tag if tag else ''}.txt").write_text(txt)


if __name__ == "__main__":
    main()
