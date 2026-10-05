"""The year's top-10 winners in hindsight: what they returned, whether they were findable on
Jan 1, and how much of their run the monthly momentum rule caught (2026-10-03).

The user asked: "picking the 10 best stocks each year would obviously beat SPY, so how do we
find those?" For each calendar year 2019-2025 this script uses the liquid-1000 as of the last
weekly panel date of the prior year, enters at the first open and exits at the last close.
  * perfect-foresight top-10 vs SPY vs the median stock vs 10 random stocks;
  * where the eventual top-10 ranked on Jan 1 (12-1 momentum and 63d vol percentiles);
  * capture: the share of each winner's log return that the 05 monthly momentum top-20 earned
    while holding it. Momentum does not predict the winner on Jan 1; it climbs aboard mid-run.
The universe is today's tickers (survivor-shaped), so the median and random-pick figures are
flattered, which makes the index look relatively WORSE than it really is.

    .venv/bin/python research/long_horizon_discount_2026-09-25/10_top10_winners.py
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
YEARS = range(2019, 2026)
SEED = 11


def main() -> None:
    p = pd.read_parquet(rb.pb.OUT, columns=["date", "ticker", "mom_12_1", "rv_63", "dv_rank"])
    O, C = rb.price_matrices(sorted(set(p["ticker"]) | {"SPY"}))
    cal, col = C.index, {t: j for j, t in enumerate(C.columns)}
    Ov, Cv = O.values, C.ffill().values
    dates = pd.DatetimeIndex(sorted(p["date"].unique()))
    rng = np.random.default_rng(SEED)
    lines = ["=== top-10 winners per calendar year | liquid-1000 at prior year-end | first open -> last close ==="]
    lines.append(f"{'year':5s} {'top10':>7s} {'SPY':>6s} {'median':>7s} {'%>SPY':>6s} {'rand10 med':>10s} "
                 f"{'P(rand10>SPY)':>13s} {'mom20 Jan1':>10s} {'winners in mom20 Jan1':>21s} "
                 f"{'held by monthly mom':>19s} {'run captured':>12s}")
    detail = []
    for y in YEARS:
        d0 = dates[dates < pd.Timestamp(f"{y}-01-01")].max()
        g = p[p["date"] == d0].copy()
        e = cal.searchsorted(d0, side="right")
        x = cal.searchsorted(pd.Timestamp(f"{y + 1}-01-01"), side="left") - 1
        j = g["ticker"].map(col).values
        g["ret"] = Cv[x, j] / Ov[e, j] - 1
        g = g.dropna(subset=["ret"])
        spy = Cv[x, col["SPY"]] / Ov[e, col["SPY"]] - 1
        g["mom_pct"] = g["mom_12_1"].rank(pct=True)
        g["vol_pct"] = g["rv_63"].rank(pct=True)
        top = g.nlargest(10, "ret")
        mom20 = g.nlargest(20, "mom_12_1")
        rand = np.array([g["ret"].values[rng.choice(len(g), 10, replace=False)].mean() for _ in range(10000)])

        # monthly momentum holdings through the year (every 4th weekly date, as in 05)
        ds = dates[(dates >= d0) & (dates < pd.Timestamp(f"{y}-12-31"))][::4]
        ents = [cal.searchsorted(d, side="right") for d in ds] + [x + 1]
        held_logret = dict.fromkeys(top["ticker"], 0.0)
        for k, d in enumerate(ds):
            h = set(p[(p["date"] == d)].nlargest(20, "mom_12_1")["ticker"])
            a, b = ents[k], ents[k + 1]
            for t in top["ticker"]:
                if t in h:
                    end_px = Ov[b, col[t]] if b <= x else Cv[x, col[t]]
                    held_logret[t] += np.log(end_px / Ov[a, col[t]])
        full = np.log1p(top.set_index("ticker")["ret"])
        cap = pd.Series(held_logret) / full
        n_held = int((pd.Series(held_logret) != 0).sum())
        lines.append(f"{y:<5d} {top['ret'].mean():+7.0%} {spy:+6.0%} {g['ret'].median():+7.0%} "
                     f"{(g['ret'] > spy).mean():6.0%} {np.median(rand):+10.0%} {(rand > spy).mean():13.0%} "
                     f"{mom20['ret'].mean():+10.0%} {top['ticker'].isin(mom20['ticker']).sum():>18d}/10 "
                     f"{n_held:>16d}/10 {cap.mean():12.0%}")
        for _, r in top.iterrows():
            detail.append(f"  {y} {r['ticker']:6s} {r['ret']:+7.0%}  Jan-1 momentum pct {r['mom_pct']:.2f}  "
                          f"vol pct {r['vol_pct']:.2f}  run captured by monthly momentum {cap[r['ticker']]:.0%}")
    lines.append("\nper-winner detail (pct = percentile within the liquid-1000 on the prior year-end date):")
    lines.extend(detail)
    txt = "\n".join(lines)
    print(txt)
    (HERE / "10_results.txt").write_text(txt)


if __name__ == "__main__":
    main()
