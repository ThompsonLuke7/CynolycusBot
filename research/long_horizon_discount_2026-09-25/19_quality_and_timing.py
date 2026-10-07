"""Can the sleeve avoid pump-and-dump picks, or give back less of its gains? (2026-10-07)

Tests the user's two ideas after the Part 7 chart, on the full point-in-time history (2019-2026, 4 monthly
offsets, 05's simulator and costs). Each arm is compared PAIRED against its own baseline.

A. Quality of momentum (the sleeve alone; baseline = top 20 by 12-1 momentum in the liquid-300).
   A dropped name is replaced by the next one down the momentum ranking.
     no_pump       drop names that more than doubled in 3 months or trade above 2x their 200-day average
     no_lossy_pump drop a "pump" name only if its latest filed quarter showed a net LOSS (added after the 2x2
                   check showed loss-making pumps earn -1.0% a period against +3.8% for profitable ones).
                   A name with no filing on record is KEPT: fundamentals exist only for today's tickers, so
                   "unknown" partly means "later delisted", and excluding on it would use the future
     vol_cap       drop the most volatile 10% of the liquid-300 (63-day volatility)
     no_hot_month  drop names up more than 50% in the last month
     smooth        of the top 40 by momentum, keep the 20 with the largest share of up days (steady climbers)
     riskadj       rank by momentum / its own volatility over the same 12-1 window instead of raw momentum
     sector_cap    at most 6 of the 20 (30%, the system's SizingConfig.per_sector_cap_pct) per sector ETF, with
                   the sector assigned as signals.market_regime.sector_map does (best trailing correlation)
B. Giving back less (the 70/30 blend, reset monthly; baseline = fixed 70/30).
     b_no_pump     the blend with the no_pump sleeve
     b_drift       no monthly reset: buy 70/30 once and let it run (the reset is itself "selling strength")
     b_trim        sleeve share 15% when the sleeve's trailing 3-month return is in the top fifth of its own history
     b_add         sleeve share 45% when it is in the bottom fifth
     b_band        both of the above
     b_trend       sleeve share 15% while the sleeve is below its own 100-day average
     b_tp30/50     sell any name that closes 30% / 50% above its entry, hold SPY until the next rebalance
   The timing signals use the always-invested sleeve's own equity up to the decision close, and the "fifths"
   are quantiles of its history so far (expanding, at least one year), so nothing is fitted to the full sample.

IN-SAMPLE WARNING: no_pump, no_lossy_pump, b_trim and b_add were prompted by diagnostics on this same history. A good
result here is a candidate for the forward shadow, not a finding.

    .venv/bin/python research/long_horizon_discount_2026-09-25/19_quality_and_timing.py
"""
from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from signals.market_regime.config import SECTOR_ETFS_LIST, SECTOR_RESOLVER_MIN_OBS, SECTOR_RESOLVER_WINDOW_DAYS  # noqa: E402
from signals.market_regime.sector_map import correlate_to_sectors  # noqa: E402


def _load(name: str, file: str):
    spec = spec_from_file_location(name, HERE / file)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


an = _load("an", "18_sleeve_anatomy.py")
ro = _load("ro", "16_risk_overlays.py")
rb, pb = an.rb, an.pb
N_TOP, STEP, BLEND, SECTOR_MAX = 20, 4, 0.30, 6
SLEEVE_ARMS = ["base", "no_pump", "no_lossy_pump", "vol_cap", "no_hot_month", "smooth", "riskadj", "sector_cap"]
BLEND_ARMS = ["b_base", "b_no_pump", "b_no_lossy_pump", "b_drift", "b_trim", "b_add", "b_band", "b_trend", "b_tp30", "b_tp50"]


def sector_table(ctx: dict, cands: dict[pd.Timestamp, list[str]]) -> dict:
    """{date: {ticker: sector ETF}} by best trailing correlation, as the live resolver assigns it."""
    logret = np.log(ctx["raw_close"]).diff()
    sectors = [s for s in SECTOR_ETFS_LIST if s in logret.columns]
    if len(sectors) < 9:
        bars = {s: pb.load_bars(s) for s in SECTOR_ETFS_LIST}
        etf = pd.DataFrame({s: np.log(b["close"]).diff() for s, b in bars.items() if b is not None})
    else:
        etf = logret[sectors]
    out = {}
    for d, names in cands.items():
        win = logret.loc[:d, names].tail(SECTOR_RESOLVER_WINDOW_DAYS)
        both = pd.concat([win, etf.reindex(win.index)], axis=1)
        corr = both.corr(min_periods=SECTOR_RESOLVER_MIN_OBS).loc[names, etf.columns]
        out[d] = {t: (corr.loc[t].idxmax() if corr.loc[t].notna().any() else None) for t in names}
    # parity with the system's own function on a sample
    rng = np.random.default_rng(11)
    keys = list(out)
    for d in (keys[i] for i in rng.choice(len(keys), 12, replace=False)):
        t = cands[d][int(rng.integers(0, 10))]
        best, _, _ = correlate_to_sectors(logret[t].dropna(), {s: etf[s].dropna() for s in etf.columns}, asof=d)
        if best != out[d][t]:
            raise AssertionError(f"sector assignment differs from correlate_to_sectors for {t} on {d.date()}: {out[d][t]} vs {best}")
    return out


def main() -> None:
    ctx = an.build_context()
    p, cal, col, O, C, end, dates = ctx["p"], ctx["cal"], ctx["col"], ctx["O"], ctx["C"], ctx["end"], ctx["dates"]
    p = p.merge(pd.read_parquet(pb.PIT_OUT, columns=["date", "ticker", "ni_positive"]), on=["date", "ticker"], how="left")
    p = p.assign(pump=(p["ret_63"] > 1.0) | (p["dist_sma200"] > 1.0), riskadj=p["mom_12_1"] / p["vol_12_1"])
    p["lossy_pump"] = p["pump"] & (p["ni_positive"] == 0)
    by_date = {d: g.sort_values("mom_12_1", ascending=False) for d, g in p.groupby("date")}
    sectors = sector_table(ctx, {d: g["ticker"].head(80).tolist() for d, g in by_date.items()})

    def pick(arm: str, d) -> list[str]:
        g = by_date[d]
        if arm == "no_pump":
            g = g[~g["pump"]]
        elif arm == "no_lossy_pump":
            g = g[~g["lossy_pump"]]
        elif arm == "vol_cap":
            g = g[g["vol_pct"] <= 0.9]
        elif arm == "no_hot_month":
            g = g[~(g["ret_21"] > 0.5)]
        elif arm == "smooth":
            g = g.head(2 * N_TOP).sort_values("up_days_12_1", ascending=False)
        elif arm == "riskadj":
            g = g.dropna(subset=["riskadj"]).sort_values("riskadj", ascending=False)
        elif arm == "sector_cap":
            out, count = [], {}
            for t in g["ticker"]:
                s = sectors[d].get(t)
                if s is not None and count.get(s, 0) >= SECTOR_MAX:
                    continue
                out.append(t)
                count[s] = count.get(s, 0) + 1
                if len(out) == N_TOP:
                    break
            return out
        return g["ticker"].head(N_TOP).tolist()

    ew = {arm: {d: {t: 1 / N_TOP for t in pick(arm, d)} for d in dates} for arm in SLEEVE_ARMS}
    changed = {arm: np.mean([len(set(ew[arm][d]) - set(ew["base"][d])) for d in dates]) for arm in SLEEVE_ARMS}

    def blend(w: dict, share: float) -> dict:
        return {"SPY": 1 - share, **{t: x * share for t, x in w.items()}}

    live0 = ctx["live"][0]
    rets = {a: [] for a in SLEEVE_ARMS + BLEND_ARMS + ["spy"]}
    stats = {a: [] for a in SLEEVE_ARMS + BLEND_ARMS}
    extra = {a: [] for a in SLEEVE_ARMS + BLEND_ARMS}
    last12 = {}
    for ph in range(STEP):
        ds = dates[ph::STEP]
        entries = [i for i in (cal.searchsorted(d, side="right") for d in ds) if i <= end]
        ds = ds[:len(entries)]
        idx = cal[entries[0]:]
        b, _ = rb.simulate([{"SPY": 1.0}], entries[:1], O, C, col, end)
        v_base, _ = ro.simulate([ew["base"][d] for d in ds], entries, O, C, col, end)
        eq = pd.Series(v_base, index=idx)
        r63 = eq / eq.shift(63) - 1
        hi = r63.expanding(min_periods=252).quantile(0.8)
        lo = r63.expanding(min_periods=252).quantile(0.2)
        below = eq < eq.rolling(100).mean()

        def at(series, d, default=False):
            s = series.loc[:d]
            return default if s.empty or pd.isna(s.iloc[-1]) else s.iloc[-1]

        hot = [bool(at(r63, d, np.nan) > at(hi, d, np.inf)) for d in ds]
        cold = [bool(at(r63, d, np.nan) < at(lo, d, -np.inf)) for d in ds]
        weak = [bool(at(below, d)) for d in ds]
        plans = {a: ([ew[a][d] for d in ds], {}) for a in SLEEVE_ARMS}
        plans |= {
            "b_base": ([blend(ew["base"][d], BLEND) for d in ds], {}),
            "b_no_pump": ([blend(ew["no_pump"][d], BLEND) for d in ds], {}),
            "b_no_lossy_pump": ([blend(ew["no_lossy_pump"][d], BLEND) for d in ds], {}),
            "b_trim": ([blend(ew["base"][d], 0.15 if h else BLEND) for d, h in zip(ds, hot)], {}),
            "b_add": ([blend(ew["base"][d], 0.45 if c else BLEND) for d, c in zip(ds, cold)], {}),
            "b_band": ([blend(ew["base"][d], 0.15 if h else 0.45 if c else BLEND) for d, h, c in zip(ds, hot, cold)], {}),
            "b_trend": ([blend(ew["base"][d], 0.15 if w else BLEND) for d, w in zip(ds, weak)], {}),
            "b_tp30": ([blend(ew["base"][d], BLEND) for d in ds], {"tp": 0.30, "park": "SPY"}),
            "b_tp50": ([blend(ew["base"][d], BLEND) for d in ds], {"tp": 0.50, "park": "SPY"}),
        }
        curves = {}
        for a, (ws, kw) in plans.items():
            v, exited = ro.simulate(ws, entries, O, C, col, end, **kw)
            curves[a] = v
            share = np.mean([sum(x for t, x in w.items() if t not in ("SPY", "CASH")) for w in ws])
            extra[a].append((share, exited))
        curves["b_drift"] = (1 - BLEND) * b + BLEND * v_base      # bought once, never reset
        extra["b_drift"].append((np.mean(BLEND * v_base / curves["b_drift"]), 0.0))
        rets["spy"].append(pd.Series(np.diff(np.r_[1.0, b]) / np.r_[1.0, b][:-1], index=idx))
        for a, v in curves.items():
            stats[a].append(rb.stats(v, b, idx))
            rets[a].append(pd.Series(np.diff(np.r_[1.0, v]) / np.r_[1.0, v][:-1], index=idx))
        if ds[0] == live0 or live0 in ds:
            for a, v in {**curves, "spy": b}.items():
                s = pd.Series(v, index=idx)
                w = s.loc["2025-10-03":]
                last12[a] = (w.iloc[-1] / w.iloc[0] - 1, (w / w.cummax() - 1).min())
            flags = pd.DataFrame({"hot": hot, "cold": cold, "weak": weak}, index=ds).loc["2025-10-01":]

    spy = pd.concat(rets["spy"], axis=1)
    lines = [f"=== quality filters and give-back rules | top-{N_TOP} momentum in the liquid-300 | point-in-time, monthly, 4 offsets | "
             f"{dates.min().date()}..{cal[end].date()} ==="]
    lines.append(f"{'arm':15s} {'CAGR med [min, max]':>24s} {'Sharpe':>6s} {'MDD med/worst':>14s} {'vs SPY /yr':>10s} "
                 f"{'vs own baseline /yr [95% CI]':>30s} {'sleeve':>6s} {'changed':>7s} {'exits':>5s} {'last 12m':>8s} {'12m MDD':>7s}")
    for group, baseline in ((SLEEVE_ARMS, "base"), (BLEND_ARMS, "b_base")):
        base_r = pd.concat(rets[baseline], axis=1)
        for a in group:
            df = pd.DataFrame(stats[a])
            r = pd.concat(rets[a], axis=1)
            act = (r - spy).dropna().mean(axis=1)
            pair = (r - base_r).dropna().mean(axis=1)
            plo, phi = rb.block_ci(pair) if a != baseline else (0.0, 0.0)
            share, exited = np.mean([x[0] for x in extra[a]]), np.mean([x[1] for x in extra[a]])
            ch = changed.get(a, changed.get(a[2:], 0.0))
            lines.append(f"{a:15s} {df.cagr.median():+7.1%} [{df.cagr.min():+6.1%},{df.cagr.max():+6.1%}] {df.sharpe.median():6.2f} "
                         f"{df.mdd.median():+6.1%}/{df.mdd.min():+6.1%} {act.mean() * 252:+10.1%} "
                         f"{pair.mean() * 252:+11.1%} [{plo:+6.1%},{phi:+6.1%}] {share:6.0%} {ch:5.1f}/20 {exited:5.0%} "
                         f"{last12[a][0]:+8.1%} {last12[a][1]:+7.1%}")
        lines.append("")
    df = pd.DataFrame(stats["base"])
    lines.append(f"spy             {df.bcagr.median():+7.1%} {'':17s} {df.bsharpe.median():6.2f} {'':46s} {'':27s} "
                 f"{last12['spy'][0]:+8.1%} {last12['spy'][1]:+7.1%}")
    lines.append("\n'changed' = names per rebalance that differ from the plain top 20. 'last 12m' = the live schedule from the "
                 "2025-10-03 close; it is ONE year on one schedule and is shown for context, not as evidence.")
    lines.append("timing flags at the last 13 decisions on the live schedule (hot = trailing 3-month return in its top fifth so far; "
                 "cold = bottom fifth; weak = below its 100-day average):")
    lines.append("  " + "  ".join(f"{d.strftime('%m-%d')}:{'H' if r_.hot else 'C' if r_.cold else '-'}{'w' if r_.weak else ''}" for d, r_ in flags.iterrows()))
    sec = pd.Series([s for d in ctx["live"][-13:] for t, s in sectors[d].items() if t in ew["base"][d]]).value_counts()
    lines.append(f"sector ETF of the plain top-20 picks over the last 13 decisions: {sec.to_dict()}")
    # Is b_band a plateau? Vary the look-back, the quantile and the two shares around the pre-set 63 / 20% / 15-45.
    lines.append("\nb_band sensitivity: paired active vs fixed 70/30 per year [95% CI] and Sharpe (fixed 70/30 = "
                 f"{pd.DataFrame(stats['b_base']).sharpe.median():.2f})")
    for look in (42, 63, 126):
        for q in (0.2, 0.3):
            for lo_s, hi_s in ((0.15, 0.45), (0.20, 0.40)):
                R, B, sharpe = [], [], []
                for ph in range(STEP):
                    ds = dates[ph::STEP]
                    entries = [i for i in (cal.searchsorted(d, side="right") for d in ds) if i <= end]
                    ds = ds[:len(entries)]
                    idx = cal[entries[0]:]
                    eq = pd.Series(ro.simulate([ew["base"][d] for d in ds], entries, O, C, col, end)[0], index=idx)
                    r = eq / eq.shift(look) - 1
                    hi, lo = r.expanding(min_periods=252).quantile(1 - q), r.expanding(min_periods=252).quantile(q)

                    def last(series, d, default):
                        x = series.loc[:d]
                        return default if x.empty or pd.isna(x.iloc[-1]) else x.iloc[-1]

                    shares = [lo_s if last(r, d, np.nan) > last(hi, d, np.inf) else
                              hi_s if last(r, d, np.nan) < last(lo, d, -np.inf) else BLEND for d in ds]
                    v = ro.simulate([blend(ew["base"][d], x) for d, x in zip(ds, shares)], entries, O, C, col, end)[0]
                    v0 = ro.simulate([blend(ew["base"][d], BLEND) for d in ds], entries, O, C, col, end)[0]
                    to_r = lambda x: pd.Series(np.diff(np.r_[1.0, x]) / np.r_[1.0, x][:-1], index=idx)  # noqa: E731
                    R.append(to_r(v)), B.append(to_r(v0))
                    sharpe.append(to_r(v).mean() / to_r(v).std() * np.sqrt(252))
                pair = (pd.concat(R, axis=1) - pd.concat(B, axis=1)).dropna().mean(axis=1)
                plo, phi = rb.block_ci(pair)
                lines.append(f"  look-back {look:3d}d, top/bottom {q:.0%}, shares {lo_s:.0%}/{hi_s:.0%}: "
                             f"{pair.mean() * 252:+.1%} [{plo:+.1%}, {phi:+.1%}]  Sharpe {np.median(sharpe):.2f}")
    txt = "\n".join(lines)
    print(txt)
    (HERE / "19_results.txt").write_text(txt)


if __name__ == "__main__":
    main()
