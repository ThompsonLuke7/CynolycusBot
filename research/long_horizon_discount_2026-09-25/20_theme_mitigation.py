"""Can a theme sell-off be sidestepped: daily exits, puts on the theme index, or rotating out? (2026-10-07)

The user's challenge to Part 8 ("a red month can only be sized"): the June 2026 semiconductor drop looked
overextended beforehand and broke down in its first days, so why hold for the month? Parts 6 and 8 tested
per-name stops and month-end checks only. This tests the three things they did not, on the 70/30 blend,
point-in-time 2019-2026, 4 monthly offsets, paired against the fixed 70/30:

THEME INDEX of a decision = the ETF (45 sector/thematic funds) whose daily returns best match the new 20-name
basket over the trailing 60 sessions. "Extended" = that ETF closes more than 15% above its 100-day average on
the decision date.

A. Leave during the month (signal on a daily close, trade the next open, sleeve money waits in SPY until the
   next rebalance):
     x_stop10     the sleeve is down 10% from this period's entry
     x_tbreak     the theme ETF closes below its 20-day average AND 5% or more off its 20-day high
     x_tbreak_ext x_tbreak, armed only when the theme was extended at the decision
     x_sbreak     the always-invested sleeve closes below its own 20-day average
B. Hedge (the sleeve stays; the hedge is sized at the basket's trailing beta to the theme ETF, clipped 0.5-2.5):
     h_put        every period, buy 5%-out-of-the-money puts on the theme ETF that expire at the next rebalance
     h_put_ext    the same, only when the theme is extended at the decision
     h_short_ext  short the theme ETF instead when extended (IDEALISED: no borrow cost, not possible in a Roth;
                  it shows what removing the theme exposure does, with no option premium)
   PUTS ARE MODELLED, NOT QUOTED: there is no usable historical option data (see the retracted options study).
   Black-Scholes with sigma = the ETF's average of 21- and 63-day realized vol, 2.5% of premium paid each way.
   The repo's own IV surfaces (22 liquid semis/megacaps, 2026-09-11..10-05) put 30-day IV at 0.82x
   max(rv21, rv63), so this sigma is in the right range; the +-20% rows bracket it.
C. Rotate (changes which 20 names are held):
     r_skip_broken  skip a name whose own theme ETF is below its 50-day average at the decision
     r_cap_ext      at most 6 names from the theme ETF's group, only when the theme is extended

    .venv/bin/python research/long_horizon_discount_2026-09-25/20_theme_mitigation.py
"""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = spec_from_file_location(name, HERE / file)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


an = _load("an", "18_sleeve_anatomy.py")
rb, pb = an.rb, an.pb
N_TOP, STEP, SHARE = 20, 4, 0.30
THEMES = ["XLK", "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLU", "SMH", "SOXX", "IGV", "SKYY",
          "CIBR", "HACK", "BOTZ", "ROBO", "AIQ", "ARKK", "ARKG", "ARKQ", "XBI", "IBB", "ICLN", "TAN", "LIT", "URA", "NLR",
          "GRID", "PAVE", "ITA", "UFO", "FINX", "IPAY", "BLOK", "QTUM", "DRIV", "KWEB", "GLD", "SLV", "COPX", "QQQ", "IWM"]
EXT, OTM, HALF_SPREAD = 0.15, 0.95, 0.025
_ncdf = np.vectorize(lambda z: 0.5 * (1 + erf(z / sqrt(2))))


def bs_put(s, k: float, t, sigma: float):
    """Black-Scholes put, zero rate. `s` and `t` may be arrays; t in years."""
    s, t = np.asarray(s, float), np.maximum(np.asarray(t, float), 1e-9)
    d1 = (np.log(s / k) + 0.5 * sigma ** 2 * t) / (sigma * np.sqrt(t))
    return k * _ncdf(-(d1 - sigma * np.sqrt(t))) - s * _ncdf(-d1)


def main() -> None:
    ctx = an.build_context()
    p, cal, col, O, C, end, dates = ctx["p"], ctx["cal"], ctx["col"], ctx["O"], ctx["C"], ctx["end"], ctx["dates"]
    bars = {t: pb.load_bars(t) for t in THEMES}
    Ec = pd.DataFrame({t: b["close"] for t, b in bars.items() if b is not None}).reindex(cal).ffill()
    Eo = pd.DataFrame({t: b["open"] for t, b in bars.items() if b is not None}).reindex(cal).fillna(Ec.shift(1))
    er = np.log(Ec).diff()
    sma20, sma50, sma100, hi20 = Ec.rolling(20).mean(), Ec.rolling(50).mean(), Ec.rolling(100).mean(), Ec.rolling(20).max()
    rv = (er.rolling(21).std() + er.rolling(63).std()) / 2 * np.sqrt(252)
    lr = np.log(ctx["raw_close"]).diff()
    by_date = {d: g.sort_values("mom_12_1", ascending=False) for d, g in p.groupby("date")}
    spy = col["SPY"]

    # theme of the basket, and of each candidate name, at every decision date
    theme, beta, name_theme = {}, {}, {}
    for d, g in by_date.items():
        cands = g["ticker"].head(80).tolist()
        win = lr.loc[:d, cands].tail(120)
        ew = er.reindex(win.index)
        cc = pd.concat([win, ew], axis=1).corr(min_periods=60).loc[cands, ew.columns]
        name_theme[d] = {t: (cc.loc[t].idxmax() if cc.loc[t].notna().any() else None) for t in cands}
        basket = win[cands[:N_TOP]].tail(60).mean(axis=1)
        e60 = ew.tail(60)
        c60 = e60.corrwith(basket)
        theme[d] = c60.idxmax()
        beta[d] = float(np.clip(basket.cov(e60[theme[d]]) / e60[theme[d]].var(), 0.5, 2.5))
    ext = {d: bool(Ec.at[d, theme[d]] / sma100.at[d, theme[d]] - 1 > EXT) for d in by_date}

    def names_for(arm: str, d) -> list[str]:
        g = by_date[d]
        if arm == "r_skip_broken":
            ok = [t for t in g["ticker"].head(80) if name_theme[d].get(t) is None
                  or not (Ec.at[d, name_theme[d][t]] < sma50.at[d, name_theme[d][t]])]
            return (ok + [t for t in g["ticker"] if t not in ok])[:N_TOP] if len(ok) < N_TOP else ok[:N_TOP]
        if arm == "r_cap_ext" and ext[d]:
            out, n_theme = [], 0
            for t in g["ticker"].head(80):
                if name_theme[d].get(t) == theme[d]:
                    if n_theme >= 6:
                        continue
                    n_theme += 1
                out.append(t)
                if len(out) == N_TOP:
                    break
            return out
        return g["ticker"].head(N_TOP).tolist()

    def engine(ds, entries, plan):
        """plan(d, ctxp) -> dict(names, exit_rel, put (bool), short (bool), vol_mult). Returns (values, info)."""
        vals = np.full(end + 1, np.nan)
        V, w_old, n_exit, n_hedge = 1.0, None, 0, 0
        for k, (d, e) in enumerate(zip(ds, entries)):
            nxt = entries[k + 1] if k + 1 < len(entries) else None
            last = nxt - 1 if nxt is not None else end
            pl = plan(d, e, last)
            names = pl["names"]
            idx = np.array([col[t] for t in names])
            wt = np.full(len(names), 1 / len(names))
            w = {"SPY": 1 - SHARE, **{t: SHARE / len(names) for t in names}}
            if w_old is None:
                V *= 1 - rb.COST_RT / 2
            else:
                V *= 1 - 0.5 * sum(abs(w.get(x, 0) - w_old.get(x, 0)) for x in set(w) | set(w_old)) * rb.COST_RT
            spy_c = C[e:last + 1, spy] / O[e, spy]
            slv_c = (C[e:last + 1, idx] / O[e, idx]) @ wt
            spy_n = O[nxt, spy] / O[e, spy] if nxt is not None else None
            name_n = O[nxt, idx] / O[e, idx] if nxt is not None else None
            slv_n = name_n @ wt if nxt is not None else None
            exited = False
            xr = pl.get("exit_rel")
            if xr is not None and e + xr + 1 <= last:
                x = e + xr + 1
                val_x = ((O[x, idx] / O[e, idx]) @ wt) * (1 - rb.COST_RT)
                slv_c = slv_c.copy()
                slv_c[x - e:] = val_x * C[x:last + 1, spy] / O[x, spy]
                if nxt is not None:
                    slv_n = val_x * O[nxt, spy] / O[x, spy]
                exited, n_exit = True, n_exit + 1
            rel_c = (1 - SHARE) * spy_c + SHARE * slv_c
            rel_n = (1 - SHARE) * spy_n + SHARE * slv_n if nxt is not None else None
            if pl.get("put") or pl.get("short"):
                n_hedge += 1
                t_ = theme[d]
                m_c = Ec[t_].values[e:last + 1] / Eo[t_].values[e]
                m_n = Eo[t_].values[nxt] / Eo[t_].values[e] if nxt is not None else None
                h = SHARE * beta[d]
                if pl.get("put"):
                    sigma = float(rv.at[d, t_]) * pl.get("vol_mult", 1.0)
                    n_sess = last - e + 1
                    prem = float(bs_put(1.0, OTM, n_sess / 252, sigma)) * (1 + HALF_SPREAD)
                    rel_c = rel_c + h * (bs_put(m_c, OTM, (last - np.arange(e, last + 1) + 0.5) / 252, sigma) - prem)
                    if nxt is not None:
                        rel_n = rel_n + h * (max(OTM - m_n, 0.0) * (1 - HALF_SPREAD) - prem)
                else:
                    rel_c = rel_c + h * (1 - m_c)
                    if nxt is not None:
                        rel_n = rel_n + h * (1 - m_n)
            vals[e:last + 1] = V * rel_c
            if nxt is not None:
                V *= rel_n
                if exited:
                    w_old = {"SPY": 1.0}
                else:
                    dw = np.r_[(1 - SHARE) * spy_n, SHARE * wt * name_n]
                    w_old = dict(zip(["SPY"] + names, dw / dw.sum()))
        return vals[entries[0]:], {"exits": n_exit / len(ds), "hedged": n_hedge / len(ds)}

    arms = ["b_base", "x_stop10", "x_tbreak", "x_tbreak_ext", "x_sbreak", "h_put", "h_put_ext", "h_short_ext",
            "r_skip_broken", "r_cap_ext"]
    sens = ["x_stop8", "x_stop15", "h_put_ext_cheap", "h_put_ext_dear"]
    rets = {a: [] for a in arms + sens + ["spy"]}
    stats = {a: [] for a in arms + sens}
    info = {a: [] for a in arms + sens}
    last12, case = {}, []
    for ph in range(STEP):
        ds = dates[ph::STEP]
        entries = [i for i in (cal.searchsorted(d, side="right") for d in ds) if i <= end]
        ds = ds[:len(entries)]
        idx = cal[entries[0]:]
        b = rb.simulate([{"SPY": 1.0}], entries[:1], O, C, col, end)[0]
        base_names = {d: names_for("b_base", d) for d in ds}
        v_sleeve = rb.simulate([{t: 1 / N_TOP for t in base_names[d]} for d in ds], entries, O, C, col, end)[0]
        eqs = pd.Series(v_sleeve, index=idx)
        below = (eqs < eqs.rolling(20).mean()).values
        off = entries[0]

        def first(mask):
            hit = np.flatnonzero(mask)
            return int(hit[0]) if len(hit) else None

        def sleeve_path(d, e, last):
            ii = np.array([col[t] for t in base_names[d]])
            return (C[e:last + 1, ii] / O[e, ii]).mean(axis=1)

        def tbreak(d, e, last):
            t_ = theme[d]
            c_, s_, h_ = Ec[t_].values[e:last + 1], sma20[t_].values[e:last + 1], hi20[t_].values[e:last + 1]
            return first((c_ < s_) & (c_ <= 0.95 * h_))

        plans = {
            "b_base": lambda d, e, last: {"names": base_names[d]},
            "x_stop10": lambda d, e, last: {"names": base_names[d], "exit_rel": first(sleeve_path(d, e, last) <= 0.90)},
            "x_stop8": lambda d, e, last: {"names": base_names[d], "exit_rel": first(sleeve_path(d, e, last) <= 0.92)},
            "x_stop15": lambda d, e, last: {"names": base_names[d], "exit_rel": first(sleeve_path(d, e, last) <= 0.85)},
            "x_tbreak": lambda d, e, last: {"names": base_names[d], "exit_rel": tbreak(d, e, last)},
            "x_tbreak_ext": lambda d, e, last: {"names": base_names[d], "exit_rel": tbreak(d, e, last) if ext[d] else None},
            "x_sbreak": lambda d, e, last: {"names": base_names[d], "exit_rel": first(below[e - off:last + 1 - off])},
            "h_put": lambda d, e, last: {"names": base_names[d], "put": True},
            "h_put_ext": lambda d, e, last: {"names": base_names[d], "put": ext[d]},
            "h_put_ext_cheap": lambda d, e, last: {"names": base_names[d], "put": ext[d], "vol_mult": 0.8},
            "h_put_ext_dear": lambda d, e, last: {"names": base_names[d], "put": ext[d], "vol_mult": 1.2},
            "h_short_ext": lambda d, e, last: {"names": base_names[d], "short": ext[d]},
            "r_skip_broken": lambda d, e, last: {"names": names_for("r_skip_broken", d)},
            "r_cap_ext": lambda d, e, last: {"names": names_for("r_cap_ext", d)},
        }
        curves = {}
        for a, plan in plans.items():
            curves[a], inf = engine(ds, entries, plan)
            info[a].append(inf)
        ref = rb.simulate([{"SPY": 1 - SHARE, **{t: SHARE / N_TOP for t in base_names[d]}} for d in ds], entries, O, C, col, end)[0]
        if not np.allclose(curves["b_base"], ref):
            raise AssertionError("period engine disagrees with 05's simulate on the plain 70/30 blend")
        rets["spy"].append(pd.Series(np.diff(np.r_[1.0, b]) / np.r_[1.0, b][:-1], index=idx))
        for a, v in curves.items():
            stats[a].append(rb.stats(v, b, idx))
            rets[a].append(pd.Series(np.diff(np.r_[1.0, v]) / np.r_[1.0, v][:-1], index=idx))
        if ctx["live"][0] in ds:
            for a, v in {**curves, "spy": b}.items():
                w = pd.Series(v, index=idx).loc["2025-10-03":]
                last12[a] = (w.iloc[-1] / w.iloc[0] - 1, (w / w.cummax() - 1).min())
            for k, (d, e) in enumerate(zip(ds, entries)):
                if d < pd.Timestamp("2025-10-01"):
                    continue
                nxt = entries[k + 1] if k + 1 < len(entries) else None
                last = nxt - 1 if nxt is not None else end
                sp = sleeve_path(d, e, last)
                tb, s10 = tbreak(d, e, last), first(sp <= 0.90)
                t_ = theme[d]
                etf_r = (Eo[t_].values[nxt] if nxt is not None else Ec[t_].values[end]) / Eo[t_].values[e] - 1
                case.append(f"   {d.date()}  theme {t_:5s} beta {beta[d]:.1f}  {Ec.at[d, t_] / sma100.at[d, t_] - 1:+5.0%} vs 100d avg"
                            f"{' EXT' if ext[d] else '    '}  theme {etf_r:+6.1%}  sleeve {sp[-1] - 1:+6.1%} (low {sp.min() - 1:+5.0%})  "
                            f"theme-break day {tb if tb is not None else '-':>2}  sleeve -10% day {s10 if s10 is not None else '-':>2}")

    spy_r = pd.concat(rets["spy"], axis=1)
    base_r = pd.concat(rets["b_base"], axis=1)
    lines = [f"=== theme mitigation on the 70/30 blend | top-{N_TOP} momentum in the liquid-300 | point-in-time, monthly, 4 offsets | "
             f"{dates.min().date()}..{cal[end].date()} ==="]
    th = pd.Series(theme).value_counts(normalize=True)
    lines.append("theme index of the basket, share of decisions: " + "  ".join(f"{t} {v:.0%}" for t, v in th.head(10).items())
                 + f" | extended at {np.mean(list(ext.values())):.0%} of decisions")
    lines.append(f"{'arm':16s} {'CAGR med [min, max]':>24s} {'Sharpe':>6s} {'MDD med/worst':>14s} {'vs fixed 70/30 /yr [95% CI]':>30s} "
                 f"{'acted':>6s} {'last 12m':>8s} {'12m MDD':>7s}")
    for a in arms + sens:
        df = pd.DataFrame(stats[a])
        pair = (pd.concat(rets[a], axis=1) - base_r).dropna().mean(axis=1)
        lo, hi = rb.block_ci(pair) if a != "b_base" else (0.0, 0.0)
        acted = np.mean([x["exits"] + x["hedged"] for x in info[a]])
        if a == sens[0]:
            lines.append("  sensitivity:")
        lines.append(f"{a:16s} {df.cagr.median():+7.1%} [{df.cagr.min():+6.1%},{df.cagr.max():+6.1%}] {df.sharpe.median():6.2f} "
                     f"{df.mdd.median():+6.1%}/{df.mdd.min():+6.1%} {pair.mean() * 252:+11.1%} [{lo:+6.1%},{hi:+6.1%}] "
                     f"{acted:6.0%} {last12[a][0]:+8.1%} {last12[a][1]:+7.1%}")
    df = pd.DataFrame(stats["b_base"])
    lines.append(f"{'spy':16s} {df.bcagr.median():+7.1%} {'':17s} {df.bsharpe.median():6.2f} {'':53s} {last12['spy'][0]:+8.1%} {last12['spy'][1]:+7.1%}")
    lines.append("\n'acted' = share of periods in which the rule exited or hedged. 'last 12m' = the live schedule from the 2025-10-03 close.")
    lines.append("the last 13 periods on the live schedule (day numbers count sessions from the entry; '-' = never triggered):")
    lines += case

    # how often does "extended" precede a bad period, and what does a theme break cost when it is a false alarm?
    per = p[p["picked"] & p["fwd"].notna()].groupby("date")["fwd"].mean()
    e_s = pd.Series(ext).reindex(per.index)
    lines.append(f"\nsleeve's next 4 weeks when the theme was extended at the decision: mean {per[e_s].mean():+.1%}, lost >10% in "
                 f"{(per[e_s] < -0.10).mean():.0%} of cases | not extended: mean {per[~e_s].mean():+.1%}, lost >10% in {(per[~e_s] < -0.10).mean():.0%}")
    txt = "\n".join(lines)
    print(txt)
    (HERE / "20_results.txt").write_text(txt)


if __name__ == "__main__":
    main()
