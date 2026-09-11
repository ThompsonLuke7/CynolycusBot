"""What KIND of name does each module rank, and does it change with depth?

If a module's deep ranks return much worse than its top ranks, the first thing to
check is whether it is ranking a different KIND of instrument down there -- lower
price, thinner volume, higher volatility -- rather than the same kind, worse
chosen.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
                px = float(tail["close"].median())
                v = {"price": px,
                     "dollar_vol": float((tail["close"] * tail["volume"]).median()),
                     "atr_pct": float(((tail["high"] - tail["low"]) / tail["close"]).median())}
        _c[t] = v
    return _c[t]


def main() -> None:
    df = pd.read_parquet(DATA / "rank_depth_shares.parquet")
    rows = []
    for r in df.itertuples():
        p = prof(r.ticker)
        if p:
            rows.append({"module": r.module, "rank": r.rank, "ret_10": r.ret_10,
                         "score": r.score, **p})
    d = pd.DataFrame(rows)
    print(f"{'module':24s} {'rank':>5s} {'n':>4s} {'med px$':>8s} {'med $vol':>11s} "
          f"{'atr%':>6s} {'med score':>10s} {'ret10%':>7s}")
    for m in sorted(d["module"].unique()):
        s = d[d["module"] == m]
        for lo, hi, tag in [(1, 1, "1"), (2, 3, "2-3"), (4, 5, "4-5"), (6, 10, "6-10")]:
            g = s[(s["rank"] >= lo) & (s["rank"] <= hi)]
            if len(g) < 5:
                continue
            print(f"{m:24s} {tag:>5s} {len(g):4d} {g['price'].median():8.2f} "
                  f"{g['dollar_vol'].median():11,.0f} {g['atr_pct'].median()*100:6.2f} "
                  f"{g['score'].median():10.4f} {g['ret_10'].mean()*100:7.2f}")
        print()

    print("--- score spread within a bar: how separated are ranks 1 and 10? ---")
    sp = df.dropna(subset=["score"]).groupby(["module", "avail"])["score"].agg(
        ["max", "min", "count"])
    sp["spread"] = sp["max"] - sp["min"]
    sp["rel"] = sp["spread"] / sp["max"].abs().replace(0, np.nan)
    print(sp.groupby("module")[["spread", "rel"]].median().to_string())


if __name__ == "__main__":
    main()
