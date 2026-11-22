"""Relative strength across sector/theme ETFs, split by market regime.

The question (user, 2026-11-20): in a selloff, is it the *weak groups* (the AI trade) that keep
falling and the strong groups (defensives, healthcare) that hold up, so that shorting the
weakest and owning the strongest pays even when index direction is unpredictable?
(01_regime_study.py found that QQQ/SPY/SMH trend states do not predict index direction.)

Pre-registered 2026-11-20 ~16:25 ET, before any result was computed.

Data   Alpaca SIP daily bars, split+dividend adjusted, 2016-01-04..2026-11-19
       (Data/research/regime_switch_2026-11-20/bars_1d_etfs, fetched 2026-11-20 21:11 UTC).
Universe  The research cache's equity ETF list minus broad-market funds (SPY VOO VTI DIA RSP
       MDY QQQ IWM): sectors, styles and themes. Eligible at a decision when it has >= 126
       sessions of history and a 20-session median dollar volume >= $5M. Today's ETF list
       (survivor-shaped: closed ETFs are absent). Near-duplicates (SMH/SOXX, HACK/CIBR) are kept.
Timing Decide at the close of each week's last session t. Enter at the next session's open,
       exit at the open of the session after the following week's last session (open to
       open, about 5 sessions). No costs (ETF spreads are ~1-3 bp; reported turnover lets the
       reader charge them).
Signals  R63 (primary) = close_t / close_{t-63} - 1; R21 (secondary).
Books  equal weight, 5 names each side:
       LS        top5 - bottom5
       weak_rel  SPY - bottom5     (short the weakest, hedged with SPY)
       strong_rel top5 - SPY       (own the strongest, hedged)
       weak_short -bottom5         (outright short of the weakest)
Regime QQQ state at t from 01_regime_study.py: D1_sma50 (close < SMA50) and D2_ema20_50.
Null   each week, 5 random eligible names vs 5 others, 1,000 draws; report the percentile of the
       real mean in each regime cell. Weeks are non-overlapping, so t = mean / (sd / sqrt(n)),
       MDE ~= 2.8 x SE.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BARS = REPO / "Data" / "research" / "regime_switch_2026-11-20" / "bars_1d_etfs"
OUT = HERE / "out"
EXCLUDE = {"SPY", "VOO", "VTI", "DIA", "RSP", "MDY", "QQQ", "IWM"}
K, MIN_HIST, MIN_DV = 5, 126, 5e6
NULL_DRAWS = 1000
RNG = np.random.default_rng(20261121)

_spec = importlib.util.spec_from_file_location("rs01", HERE / "01_regime_study.py")
rs01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs01)


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    closes, opens, dvol = {}, {}, {}
    for p in sorted(BARS.glob("*.parquet")):
        t = p.stem
        d = pd.read_parquet(p)
        d["date"] = pd.to_datetime(d["timestamp"]).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        d = d.set_index("date").sort_index().loc[: rs01.LAST_SESSION]
        if d.index.duplicated().any():
            raise ValueError(f"{t}: duplicate dates")
        closes[t], opens[t], dvol[t] = d["close"], d["open"], d["close"] * d["volume"]
    c, o, v = pd.DataFrame(closes), pd.DataFrame(opens), pd.DataFrame(dvol)
    if c.index[-1] != rs01.LAST_SESSION:
        raise ValueError("panel does not end on the last session")
    return c, o, v


def main() -> None:
    c, o, v = load_panel()
    qqq = rs01.load("QQQ")
    if not qqq.index.equals(c.index):
        raise ValueError("ETF calendar differs from QQQ")
    st = rs01.states(qqq)
    names = [t for t in c.columns if t not in EXCLUDE]
    hist = c[names].notna().cumsum()
    dv20 = v[names].rolling(20, min_periods=20).median()
    sig = {"R63": c[names] / c[names].shift(63) - 1, "R21": c[names] / c[names].shift(21) - 1}

    idx = c.index
    week = idx.to_series().dt.to_period("W-FRI")
    last_of_week = idx.to_series().groupby(week.values).max().values
    pos = {d: i for i, d in enumerate(idx)}
    decisions = [d for d in last_of_week if pos[d] >= 200]

    rows = []
    for si, d in enumerate(decisions[:-1]):
        i = pos[d]
        nxt = decisions[si + 1]
        j_in, j_out = i + 1, pos[nxt] + 1
        if j_out >= len(idx):
            break
        elig = [t for t in names if hist.at[d, t] >= MIN_HIST and dv20.at[d, t] >= MIN_DV
                and np.isfinite(o.iat[j_in, c.columns.get_loc(t)]) and np.isfinite(o.iat[j_out, c.columns.get_loc(t)])]
        if len(elig) < 4 * K:
            continue
        fwd = (o.iloc[j_out][elig] / o.iloc[j_in][elig] - 1)
        spy = o.iloc[j_out]["SPY"] / o.iloc[j_in]["SPY"] - 1
        rec = {"date": d, "n_elig": len(elig), "spy": spy,
               "D1": st.at[d, "D1_sma50"], "D2": st.at[d, "D2_ema20_50"]}
        # one null draw set per week, shared by both signals
        fv = fwd.to_numpy()
        perm = np.argsort(RNG.random((NULL_DRAWS, len(fv))), axis=1)
        rec["null_ls"] = (fv[perm[:, :K]].mean(1) - fv[perm[:, K:2 * K]].mean(1))
        for sname, s in sig.items():
            r = s.loc[d, elig].dropna().sort_values()
            if len(r) < 4 * K:
                continue
            bot, top = list(r.index[:K]), list(r.index[-K:])
            rb, rt = fwd[bot].mean(), fwd[top].mean()
            rec.update({f"{sname}_LS": rt - rb, f"{sname}_weak_rel": spy - rb, f"{sname}_strong_rel": rt - spy,
                        f"{sname}_weak_short": -rb, f"{sname}_bottom": ",".join(bot), f"{sname}_top": ",".join(top)})
        rows.append(rec)
    w = pd.DataFrame(rows).set_index("date")

    out = []
    books = ["LS", "weak_rel", "strong_rel", "weak_short"]
    for sname in ("R63", "R21"):
        for reg in ("D1", "D2"):
            for cell, mask in (("all", w[reg].notna()), ("DOWN", w[reg] == 1.0), ("UP", w[reg] == 0.0)):
                sub = w[mask]
                null = np.stack(sub["null_ls"].to_numpy()).mean(0) if len(sub) else np.array([np.nan])
                for b in books:
                    x = sub[f"{sname}_{b}"].dropna()
                    se = x.std(ddof=1) / np.sqrt(len(x))
                    row = {"signal": sname, "regime": reg, "cell": cell, "book": b, "weeks": len(x),
                           "mean_wk": x.mean(), "median_wk": x.median(), "hit": (x > 0).mean(), "t": x.mean() / se,
                           "mde_wk": 2.8 * se, "ann": (1 + x).prod() ** (52 / len(x)) - 1,
                           "worst_wk": x.min(), "best_wk": x.max()}
                    if b == "LS":
                        row["null_pct"] = float((null < x.mean()).mean())
                    out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(OUT / "rs_by_regime.csv", index=False)

    yr = w.copy()
    yr["year"] = yr.index.year
    by_year = yr.groupby(["year"]).agg(weeks=("spy", "size"), spy=("spy", lambda s: (1 + s).prod() - 1),
                                       R63_LS=("R63_LS", lambda s: (1 + s).prod() - 1),
                                       R63_weak_short=("R63_weak_short", lambda s: (1 + s).prod() - 1),
                                       R21_LS=("R21_LS", lambda s: (1 + s).prod() - 1),
                                       down_share=("D1", "mean"))
    by_year.to_csv(OUT / "rs_by_year.csv")
    w.drop(columns="null_ls").to_csv(OUT / "rs_weekly.csv")
    recent = w.loc["2026-09-01":, ["D1", "spy", "R63_bottom", "R63_LS", "R63_weak_short", "R63_top"]]
    recent.to_csv(OUT / "rs_recent.csv")
    print(f"weeks={len(w)} first={w.index[0].date()} last={w.index[-1].date()} "
          f"elig median={int(w['n_elig'].median())}")


if __name__ == "__main__":
    main()
