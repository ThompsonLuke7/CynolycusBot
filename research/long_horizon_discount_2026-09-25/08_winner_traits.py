"""What did the future big winners look like BEFORE they won? 1-year and 3-year lift (2026-09-29).

The user asked whether ML can learn what companies like NVDA had in common. Taken literally
("what do the winners share?"), that question returns traits that the LOSERS share too. The
testable version fixes a trait known at time t and then asks two things:
  win lift   P(top-5% forward return | trait) / P(top-5%)      among the liquid-1000 that date
  loss lift  P(bottom-5% forward return | trait) / P(bottom-5%)
A trait that raises both is only volatility. A useful trait raises win lift and not loss lift,
and its median forward return beats the universe median.

Start dates are the first weekly panel date of each quarter, 2017..(last date - horizon).
Entry is the next open; exit is the close 252 or 756 sessions later. Traits are pre-specified
here and were not searched.
Limits: 3-year windows from 2017-2026 give only ~3 non-overlapping periods, so this describes
the data and is not a validated model. The universe is today's tickers, so dead losers are
missing: loss lift is UNDERSTATED and high-vol traits look better than they are.

    .venv/bin/python research/long_horizon_discount_2026-09-25/08_winner_traits.py
"""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = spec_from_file_location("rb", HERE / "05_rotation_backtest.py")
rb = module_from_spec(_spec)
_spec.loader.exec_module(rb)

HORIZONS = {"1y": 252, "3y": 756}
TAIL = 0.05


def traits(p: pd.DataFrame) -> dict[str, pd.Series]:
    mom_pct = p.groupby("date")["mom_12_1"].rank(pct=True)
    vol_pct = p.groupby("date")["rv_63"].rank(pct=True)
    return {
        "momentum top decile (12-1)": mom_pct >= 0.9,
        "within 10% of 52w high": p["dd_52w_high"] > -0.10,
        "within 10% of 52w low": p["up_52w_low"] < 0.10,
        "revenue growth > 30% y/y": p["rev_yoy"] > 0.30,
        "revenue accelerating (>15%, +5pp)": (p["rev_yoy"] > 0.15) & (p["rev_accel"] > 0.05),
        "gross margin >= 50%": p["gm_q"] >= 0.50,
        "profitable (net income > 0)": p["ni_positive"] == 1,
        "op margin expanding (+2pp y/y)": p["opm_yoy_chg"] > 0.02,
        "EPS beats, 4q avg > 5%": p["surprise_mean4"] > 5,
        "compounder (75% above rising 200)": (p["pct_above_sma200_252"] >= 0.75) & (p["sma200_slope_252"] > 0),
        "high volatility (top quintile)": vol_pct >= 0.8,
        "smaller half of liquid-1000": p["dv_rank"] > 500,
        "growth leader: rev>20%, GM>=50%, mom top 20%": (p["rev_yoy"] > 0.20) & (p["gm_q"] >= 0.50) & (mom_pct >= 0.8),
    }


def main() -> None:
    p = pd.read_parquet(rb.pb.OUT)
    first_of_q = p.groupby(p["date"].dt.to_period("Q"))["date"].transform("min")
    p = p[p["date"] == first_of_q].reset_index(drop=True)
    O, C = rb.price_matrices(sorted(set(p["ticker"]) | {"SPY"}))
    cal = C.index
    col = {t: j for j, t in enumerate(C.columns)}
    ent = cal.searchsorted(p["date"].values, side="right")
    j = p["ticker"].map(col).values
    Ov, Cv = O.values, C.values
    T = traits(p)
    lines = [f"=== winner traits | liquid-1000 at the first weekly date of each quarter | "
             f"top/bottom {TAIL:.0%} of forward return within the start date ==="]
    for name, h in HORIZONS.items():
        ex = ent + h - 1
        valid = ex < len(cal)
        fwd = np.full(len(p), np.nan)
        fwd[valid] = Cv[ex[valid], j[valid]] / Ov[ent[valid], j[valid]] - 1
        d = p.assign(fwd=fwd).dropna(subset=["fwd"])
        d["pct"] = d.groupby("date")["fwd"].rank(pct=True)
        d["win"], d["lose"] = d["pct"] > 1 - TAIL, d["pct"] <= TAIL
        d["xs_med"] = d["fwd"] - d.groupby("date")["fwd"].transform("median")
        starts = d["date"].nunique()
        lines.append(f"\n--- {name} ({h} sessions): {starts} start dates "
                     f"{d.date.min().date()}..{d.date.max().date()}, {len(d):,} stock-starts; "
                     f"exits with no price (delisted/merged) are dropped ---")
        lines.append(f"{'trait':46s} {'share':>6s} {'P(t|win)':>8s} {'win lift':>8s} {'loss lift':>9s} "
                     f"{'med fwd vs univ':>15s} {'win lift by yr min..max':>24s}")
        for tname, mask in T.items():
            m = mask.loc[d.index].fillna(False).astype(bool)
            if m.sum() < 200:
                continue
            wl = d.loc[m, "win"].mean() / d["win"].mean()
            ll = d.loc[m, "lose"].mean() / d["lose"].mean()
            ptw = m[d["win"]].mean()
            by = d.assign(m=m, yr=d["date"].dt.year).groupby("yr")
            yr = by.apply(lambda g: g.loc[g["m"], "win"].mean() / g["win"].mean() if g["m"].sum() >= 30 else np.nan).dropna()
            lines.append(f"{tname:46s} {m.mean():6.1%} {ptw:8.1%} {wl:8.2f} {ll:9.2f} "
                         f"{d.loc[m, 'xs_med'].median():+15.1%} {yr.min():11.2f}..{yr.max():.2f}")
        nv = d[d["ticker"] == "NVDA"][["date", "fwd", "pct", "mom_12_1", "rev_yoy", "gm_q"]]
        lines.append(f"NVDA as a start-date example ({name}): forward return / percentile / traits held at start")
        for i, r in nv.iloc[:: max(1, len(nv) // 6)].iterrows():
            held = [t.split(" (")[0] for t in T if bool(T[t].loc[i])]
            lines.append(f"  {r.date.date()}  fwd {r.fwd:+7.0%}  pct {r.pct:.2f}  traits: {', '.join(held) or '-'}")
    txt = "\n".join(lines)
    print(txt)
    (HERE / "08_results.txt").write_text(txt)


if __name__ == "__main__":
    main()
