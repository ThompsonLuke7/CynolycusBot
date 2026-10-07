"""Is "top 20 momentum among the 300 most liquid" a plateau or a lucky spot? (2026-10-04)

The 300 cutoff was chosen in Part 3 as a survivorship hedge and then held up on the point-in-time
universe, so it is post hoc. This grid varies both parameters: the liquidity cutoff L (most liquid
100 / 200 / 300 / 500 / 1000 by 60-day median dollar volume) and the number of names N (5 / 10 /
20 / 30). Each arm is the equal-weight top N by 12-1 momentum, on the point-in-time universe, with
monthly rebalance, 4 phases, and 05's simulator and costs. A robust rule shows a smooth surface
around its chosen point.

    .venv/bin/python research/long_horizon_discount_2026-09-25/15_liquidity_grid.py
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
pb = rb.pb
CUTOFFS, SIZES = (100, 200, 300, 500, 1000), (5, 10, 20, 30)


def main() -> None:
    pb.BARS, pb.OUT = pb.PIT_BARS, pb.PIT_OUT
    p = pd.read_parquet(pb.OUT, columns=["date", "ticker", "mom_12_1", "dv_rank"])
    p = p[p["date"] >= rb.START - pd.Timedelta(days=7)]
    O_df, C_df = rb.price_matrices(sorted(set(p["ticker"]) | set(rb.BENCH)),
                                   pb.PIT_OUT.parent / "px_open_pit.parquet", pb.PIT_OUT.parent / "px_close_pit.parquet")
    C_df = C_df.ffill()
    O_df = O_df.fillna(C_df.shift(1))
    O_df["CASH"] = C_df["CASH"] = 1.0
    O_df = O_df[C_df.columns]
    cal, col = C_df.index, {t: i for i, t in enumerate(C_df.columns)}
    O, C, end = O_df.values, C_df.values, len(cal) - 1
    dec = pd.DatetimeIndex(sorted(p.loc[p["date"] >= rb.START, "date"].unique()))
    by_date = {d: g.dropna(subset=["mom_12_1"]) for d, g in p.groupby("date")}
    k = rb.FREQS["monthly"]
    rows = []
    for L in CUTOFFS:
        for n in SIZES:
            res, act = [], []
            for ph in range(k):
                ds = dec[ph::k]
                entries = [i for i in (cal.searchsorted(d, side="right") for d in ds) if i <= end]
                ws = []
                for d in ds[:len(entries)]:
                    g = by_date[d]
                    names = g[g["dv_rank"] <= L].nlargest(n, "mom_12_1")["ticker"]
                    ws.append({t: 1 / len(names) for t in names})
                v, _ = rb.simulate(ws, entries, O, C, col, end)
                b, _ = rb.simulate([{"SPY": 1.0}], entries[:1], O, C, col, end)
                st = rb.stats(v, b, cal[entries[0]:])
                res.append(st)
                act.append(st["active"])
            df = pd.DataFrame(res)
            a = pd.concat(act, axis=1).dropna().mean(axis=1)
            lo, hi = rb.block_ci(a)
            rows.append({"L": L, "N": n, "cagr": df.cagr.median(), "cagr_min": df.cagr.min(), "cagr_max": df.cagr.max(),
                         "sharpe": df.sharpe.median(), "mdd": df.mdd.median(), "mdd_worst": df.mdd.min(),
                         "active": a.mean() * 252, "lo": lo, "hi": hi, "spy": df.bcagr.median(), "spy_sharpe": df.bsharpe.median()})
    r = pd.DataFrame(rows)
    lines = [f"=== top-N 12-1 momentum among the L most liquid | point-in-time, monthly, 4 phases | "
             f"SPY {r.spy.iloc[0]:+.1%} CAGR, Sharpe {r.spy_sharpe.iloc[0]:.2f} ==="]
    for title, colname, fmt in (("CAGR, median phase", "cagr", "{:+.1%}"), ("Sharpe, median phase", "sharpe", "{:.2f}"),
                                ("max drawdown, median phase", "mdd", "{:+.0%}")):
        lines.append(f"\n{title} (rows = liquidity cutoff L, columns = names held N):")
        lines.append(r.pivot(index="L", columns="N", values=colname).map(fmt.format).to_string())
    lines.append("\nactive return vs SPY per year [95% CI] and CAGR phase range:")
    for x in r.itertuples():
        lines.append(f"  L={x.L:<5d} N={x.N:<3d} {x.active:+6.1%} [{x.lo:+6.1%},{x.hi:+6.1%}]   "
                     f"CAGR {x.cagr:+.1%} [{x.cagr_min:+.1%},{x.cagr_max:+.1%}]  worst MDD {x.mdd_worst:+.0%}")
    txt = "\n".join(lines)
    print(txt)
    (HERE / "15_results.txt").write_text(txt)


if __name__ == "__main__":
    main()
