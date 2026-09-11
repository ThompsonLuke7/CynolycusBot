"""How much do momentum and HTF actually overlap, and where do they diverge?

The two are believed ~90% correlated. If that is true at the top of the ranking,
their top-1 returns should not differ by 5 percentage points. This measures name
overlap per decision bar and compares the returns of the shared vs unique picks.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DATA = REPO / "research/execution_quality/data"


def main() -> None:
    df = pd.read_parquet(DATA / "rank_depth_shares.parquet")
    picks = defaultdict(dict)
    for line in (DATA / "stage2_signal_spine.jsonl").open():
        r = json.loads(line)
        if not r.get("submit") or r.get("rank") is None:
            continue
        picks[(r["module"], r["available_at"])][r["ticker"]] = int(r["rank"])

    bars = sorted({a for _m, a in picks})
    pairs = [("momentum_expansion", "multi_ticker_swing_htf"),
             ("momentum_expansion", "meta_ranker"),
             ("multi_ticker_swing_htf", "meta_ranker")]
    print(f"{'pair':50s} {'bars':>5s} {'ovl@10':>7s} {'ovl@3':>7s} {'ovl@1':>7s} {'rank corr':>10s}")
    for a, b in pairs:
        o10, o3, o1, rc, n = [], [], [], [], 0
        for bar in bars:
            pa, pb = picks.get((a, bar)), picks.get((b, bar))
            if not pa or not pb:
                continue
            n += 1
            o10.append(len(set(pa) & set(pb)) / max(len(pa), len(pb)))
            t3a = {t for t, r in pa.items() if r <= 3}
            t3b = {t for t, r in pb.items() if r <= 3}
            o3.append(len(t3a & t3b) / 3)
            o1.append(int({t for t, r in pa.items() if r == 1} ==
                          {t for t, r in pb.items() if r == 1}))
            shared = set(pa) & set(pb)
            if len(shared) >= 3:
                rc.append(pd.Series([pa[t] for t in shared]).corr(
                    pd.Series([pb[t] for t in shared]), method="spearman"))
        print(f"{a+' vs '+b:50s} {n:5d} {np.mean(o10):7.1%} {np.mean(o3):7.1%} "
              f"{np.mean(o1):7.1%} {np.nanmean(rc):10.2f}")

    print("\n--- returns of SHARED vs UNIQUE picks (10-day, top-10 sets) ---")
    key = {(r.module, r.avail, r.ticker): r for r in df.itertuples()}
    for a, b in pairs:
        sh, ua, ub = [], [], []
        for bar in bars:
            pa, pb = picks.get((a, bar)), picks.get((b, bar))
            if not pa or not pb:
                continue
            for t in set(pa) | set(pb):
                row = key.get((a, bar, t)) or key.get((b, bar, t))
                if row is None or not np.isfinite(getattr(row, "ret_10", np.nan)):
                    continue
                v = row.ret_10
                if t in pa and t in pb:
                    sh.append(v)
                elif t in pa:
                    ua.append(v)
                else:
                    ub.append(v)
        f = lambda x: f"{np.mean(x)*100:6.2f}% (n={len(x)})" if x else "n/a"  # noqa: E731
        print(f"{a} & {b}")
        print(f"    shared      : {f(sh)}")
        print(f"    {a[:20]:20s}: {f(ua)}")
        print(f"    {b[:20]:20s}: {f(ub)}")


if __name__ == "__main__":
    main()
