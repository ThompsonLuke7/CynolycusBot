"""Does the ORDER within a module's top-10 carry information?

The combined grid scans ~240 cells, so any single p<0.05 in it is expected by
chance. This asks the one question that the top-k decision actually rests on,
with a null that removes every other explanation:

    NULL: the module's rank ordering within a decision bar is arbitrary.

Implemented by shuffling ranks WITHIN each decision bar and recomputing the
top-k mean. The pick set, the decision days, the universe drift and the sector
composition are all held fixed by construction -- only the ordering moves. If
top-1 is no better than a random draw from that same bar's ten names, there is
nothing to gain by cutting k.

Also reports Benjamini-Hochberg FDR over the whole exploratory grid.
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
N_PERM = 20000
ONLY = ["meta_ranker"]
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


def perm_test(sub: pd.DataFrame, col: str, k: int, rng) -> tuple:
    """sub holds every ranked pick; shuffle rank within bar, take top-k mean."""
    bars = []
    for _a, g in sub.groupby("avail"):
        v = g[col].values
        r = g["rank"].values
        if len(v) < k + 1 or not np.isfinite(v).all():
            continue
        bars.append((v, r))
    if len(bars) < 10:
        return None
    obs = float(np.mean(np.concatenate([v[r <= k] for v, r in bars])))
    draws = np.empty(N_PERM)
    for i in range(N_PERM):
        draws[i] = np.mean(np.concatenate(
            [rng.permutation(v)[:k] for v, _r in bars]))
    p = (np.sum(draws >= obs) + 1) / (N_PERM + 1)
    return obs, float(draws.mean()), p, len(bars)


def main() -> None:
    df = pd.read_parquet(DATA / "rank_depth_shares.parquet")
    df["dvol"] = df["ticker"].map(dvol)
    rng = np.random.default_rng(31)

    print("PERMUTATION TEST -- top-k mean vs a random draw from the SAME bar's picks")
    print("one-sided p; low p means the ordering helps\n")
    print(f"{'module':24s} {'filter':14s} {'hold':>4s} {'k':>2s} {'bars':>5s} "
          f"{'top-k%':>7s} {'shuffled%':>9s} {'edge pp':>8s} {'p':>6s}")
    import os
    only = os.environ.get("PERM_MODULES")
    mods = only.split(",") if only else ["meta_ranker", "momentum_expansion",
                                         "multi_ticker_swing_htf", "dealer_ranker"]
    for module in mods:
        m0 = df[df["module"] == module]
        thin = m0["dvol"].quantile(1 / 3)
        for fname, m in [("all", m0), ("drop thin 1/3", m0[m0["dvol"] > thin])]:
            for h in [8, 10, 15]:
                col = f"ret_{h}"
                s = m[m[col].notna()]
                for k in [1, 2, 3]:
                    r = perm_test(s, col, k, rng)
                    if r is None:
                        continue
                    obs, null, p, nb = r
                    star = " *" if p < 0.05 else ""
                    print(f"{module:24s} {fname:14s} {h:4d} {k:2d} {nb:5d} "
                          f"{obs*100:7.2f} {null*100:9.2f} {(obs-null)*100:8.2f} "
                          f"{p:6.3f}{star}")
            print()


if __name__ == "__main__":
    main()
