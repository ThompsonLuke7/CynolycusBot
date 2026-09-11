"""Is a module's score just a proxy for price / volatility / illiquidity?

Measured WITHIN each decision bar (Spearman over that bar's 10 ranked names), so
a market-wide move cannot manufacture the correlation. A strongly negative
score-vs-price correlation means rank 1 is systematically the cheapest, thinnest
name -- which is a ranking defect, not an edge.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DATA = REPO / "research/execution_quality/data"
BARS = REPO / "Data/shared/bars/1d"
_c: dict[str, dict | None] = {}


def prof(t):
    if t not in _c:
        p = BARS / f"{t}.parquet"
        v = None
        if p.exists():
            d = pd.read_parquet(p, columns=["timestamp", "high", "low", "close", "volume"])
            if len(d) >= 30:
                tail = d.tail(60)
                v = {"price": float(tail["close"].median()),
                     "dvol": float((tail["close"] * tail["volume"]).median()),
                     "rng": float(((tail["high"] - tail["low"]) / tail["close"]).median())}
        _c[t] = v
    return _c[t]


def main() -> None:
    df = pd.read_parquet(DATA / "rank_depth_shares.parquet")
    rows = []
    for r in df.itertuples():
        p = prof(r.ticker)
        if p:
            rows.append({"module": r.module, "avail": r.avail, "rank": r.rank,
                         "score": r.score, "ret_10": r.ret_10, **p})
    d = pd.DataFrame(rows).dropna(subset=["score"])

    print("within-bar Spearman of SCORE against name characteristics")
    print(f"{'module':24s} {'bars':>5s} {'vs price':>9s} {'vs $vol':>9s} {'vs range':>9s}")
    for m in sorted(d["module"].unique()):
        s = d[d["module"] == m]
        acc = {k: [] for k in ("price", "dvol", "rng")}
        n = 0
        for _bar, g in s.groupby("avail"):
            if len(g) < 5 or g["score"].nunique() < 3:
                continue
            n += 1
            for k in acc:
                if g[k].nunique() >= 3:
                    acc[k].append(spearmanr(g["score"], g[k]).statistic)
        print(f"{m:24s} {n:5d} " + " ".join(
            f"{np.nanmean(acc[k]):9.2f}" for k in ("price", "dvol", "rng")))

    print("\nreturn by liquidity tercile, WITHIN each module (10-day, %)")
    print(f"{'module':24s} {'thin':>8s} {'mid':>8s} {'thick':>8s}")
    for m in sorted(d["module"].unique()):
        s = d[d["module"] == m].dropna(subset=["ret_10"])
        if len(s) < 60:
            continue
        q = pd.qcut(s["dvol"], 3, labels=["thin", "mid", "thick"])
        g = s.groupby(q, observed=True)["ret_10"].mean() * 100
        print(f"{m:24s} " + " ".join(f"{g.get(k, np.nan):8.2f}"
                                     for k in ("thin", "mid", "thick")))


if __name__ == "__main__":
    main()
