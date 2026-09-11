"""Excess return over the universe control, with a day-block bootstrap.

Overlapping forward windows and two decision bars per day mean the signals are
not independent. The bootstrap resamples whole DECISION DAYS with replacement,
which keeps every within-day correlation intact and is the honest unit here.
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
HOLDS = [5, 8, 10, 15, 21]
DEPTHS = [1, 2, 3, 5, 10]
B = 5000


def main() -> None:
    df = pd.read_parquet(DATA / "rank_depth_shares.parquet")
    ctl = json.loads((DATA / "rank_depth_control.json").read_text())
    rng = np.random.default_rng(11)

    print(f"{'module':24s} {'hold':>4s} {'k':>3s} {'n':>4s} {'days':>5s} "
          f"{'excess%':>8s} {'CI low':>8s} {'CI high':>8s} {'p':>6s}")
    for module in ["meta_ranker", "multi_ticker_swing_htf",
                   "momentum_expansion", "dealer_ranker"]:
        for h in HOLDS:
            col = f"ret_{h}"
            base = ctl[str(h)]["univ_mean"]
            m = df[(df["module"] == module) & df[col].notna()]
            for k in DEPTHS:
                s = m[m["rank"] <= k]
                if len(s) < 10:
                    continue
                days = s["decision_day"].unique()
                per_day = {d: s.loc[s["decision_day"] == d, col].values for d in days}
                obs = s[col].mean() - base
                draws = np.empty(B)
                for b in range(B):
                    pick = rng.choice(days, size=len(days), replace=True)
                    draws[b] = np.concatenate([per_day[d] for d in pick]).mean() - base
                lo, hi = np.percentile(draws, [2.5, 97.5])
                p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
                star = " *" if p < 0.05 else ""
                print(f"{module:24s} {h:4d} {k:3d} {len(s):4d} {len(days):5d} "
                      f"{obs*100:8.2f} {lo*100:8.2f} {hi*100:8.2f} {p:6.3f}{star}")
            print()


if __name__ == "__main__":
    main()
