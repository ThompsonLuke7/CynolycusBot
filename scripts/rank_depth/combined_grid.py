"""top-k crossed with a liquidity filter, on shares, as excess over the universe.

Two independent liquidity notions, because they answer different questions:
  * share dollar-volume tercile  -- what you can trade in size, in shares
  * measured option spread       -- whether the name has an option worth buying
                                    (option_liquidity_by_name.json, live quotes)
Every cell is EXCESS over the equal-weight universe return on the same decision
days, with a decision-day block bootstrap for the interval.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DATA = REPO / "research/execution_quality/data"
BARS = REPO / "Data/shared/bars/1d"
B = 3000
_c: dict[str, float | None] = {}


def dvol(t):
    if t not in _c:
        p = BARS / f"{t}.parquet"
        v = None
        if p.exists():
            d = pd.read_parquet(p, columns=["close", "volume"])
            if len(d) >= 30:
                tail = d.tail(60)
                v = float((tail["close"] * tail["volume"]).median())
        _c[t] = v
    return _c[t]


def boot(s, col, base, rng):
    days = s["decision_day"].unique()
    per = {d: s.loc[s["decision_day"] == d, col].values for d in days}
    obs = s[col].mean() - base
    dr = np.empty(B)
    for b in range(B):
        pick = rng.choice(days, size=len(days), replace=True)
        dr[b] = np.concatenate([per[d] for d in pick]).mean() - base
    return obs, np.percentile(dr, 2.5), np.percentile(dr, 97.5), \
        2 * min((dr <= 0).mean(), (dr >= 0).mean())


def main() -> None:
    df = pd.read_parquet(DATA / "rank_depth_shares.parquet")
    ctl = json.loads((DATA / "rank_depth_control.json").read_text())
    liq = json.loads((DATA / "option_liquidity_by_name.json").read_text())
    spread = {t: v["spread_med"] for t, v in liq.items()
              if v.get("spread_med") is not None}
    df["dvol"] = df["ticker"].map(dvol)
    df["opt_spread"] = df["ticker"].map(spread)
    rng = np.random.default_rng(23)

    for h in [8, 10, 15]:
        col, base = f"ret_{h}", ctl[str(h)]["univ_mean"]
        print(f"\n########## HOLD {h} DAYS -- excess over universe ({base*100:+.2f}%) ##########")
        for module in ["meta_ranker", "momentum_expansion", "multi_ticker_swing_htf",
                       "dealer_ranker"]:
            m = df[(df["module"] == module) & df[col].notna()].copy()
            if len(m) < 50:
                continue
            thin = m["dvol"].quantile(1 / 3)
            filters = {
                "all": m,
                "drop thin 1/3": m[m["dvol"] > thin],
                "opt spread<=15%": m[m["opt_spread"] <= 0.15],
                "opt spread<=10%": m[m["opt_spread"] <= 0.10],
            }
            print(f"\n{module}")
            print(f"  {'filter':18s} {'k':>2s} {'n':>4s} {'excess%':>8s} "
                  f"{'CI':>17s} {'p':>6s}")
            for fname, fdf in filters.items():
                for k in [1, 2, 3, 5, 10]:
                    s = fdf[fdf["rank"] <= k]
                    if len(s) < 15:
                        continue
                    o, lo, hi, p = boot(s, col, base, rng)
                    star = " *" if p < 0.05 else ""
                    print(f"  {fname:18s} {k:2d} {len(s):4d} {o*100:8.2f} "
                          f"[{lo*100:6.2f},{hi*100:6.2f}] {p:6.3f}{star}")


if __name__ == "__main__":
    main()
