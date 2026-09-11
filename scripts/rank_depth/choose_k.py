"""Which k? Test the k-to-k differences instead of reading the grid by eye.

Picking the best-looking cell out of {1,2,3} on 60-odd bars is how you end up
deploying noise. This bootstraps the PAIRED difference between depths on the
same decision days, so the question is "is k=2 reliably better than k=3" rather
than "which number is bigger".
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
B = 10000


def main() -> None:
    df = pd.read_parquet(DATA / "rank_depth_shares.parquet")
    rng = np.random.default_rng(41)
    print("PAIRED difference between depths, bootstrapped over decision days")
    print(f"{'module':22s} {'hold':>4s} {'pair':>7s} {'diff pp':>8s} {'CI':>17s} {'p':>6s}")
    for module in ["momentum_expansion", "meta_ranker"]:
        for h in [8, 10, 15]:
            col = f"ret_{h}"
            m = df[(df["module"] == module) & df[col].notna()]
            days = sorted(m["decision_day"].unique())
            # per-day mean return of the top-k set, so days weigh equally
            per = {k: {d: m.loc[(m["decision_day"] == d) & (m["rank"] <= k), col].mean()
                       for d in days} for k in (1, 2, 3, 10)}
            for a, b in [(1, 3), (2, 3), (1, 2), (3, 10)]:
                dif = np.array([per[a][d] - per[b][d] for d in days])
                dif = dif[np.isfinite(dif)]
                if len(dif) < 10:
                    continue
                draws = np.array([rng.choice(dif, len(dif), replace=True).mean()
                                  for _ in range(B)])
                lo, hi = np.percentile(draws, [2.5, 97.5])
                p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
                star = " *" if p < 0.05 else ""
                print(f"{module:22s} {h:4d} {f'{a} vs {b}':>7s} {dif.mean()*100:8.2f} "
                      f"[{lo*100:6.2f},{hi*100:6.2f}] {p:6.3f}{star}")
            print()


if __name__ == "__main__":
    main()
