"""Do the ML rankers have RECALL? i.e. do the big movers show up in their top-10?

The intraday recall test asked this of a rules engine. This asks it of the three
ML modules, which is the question that decides whether an ML layer has anything
to work with.

RECALL here = of the names that made a large forward move on a given decision
bar, what share did the module rank into its top-10 targets?
PRECISION  = of the top-10 it did rank, what share made a large move?
BASE RATE  = what share of the whole eligible universe made a large move?

Precision above the base rate means the ranker carries information. Recall says
whether an ML filter on top would have anything left to select from — you cannot
filter your way to a trade that was never ranked.

Universe per decision bar = every eligible name in shared_universe.csv that has
daily bars covering the forward window. Forward move = MFE over the next 10
trading days in ATR units, ATR taken from the session BEFORE the decision so the
normaliser carries no look-ahead.
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
BARS_1D = REPO / "Data/shared/bars/1d"
FWD_DAYS = 10
_c: dict[str, pd.DataFrame | None] = {}


def daily(t):
    if t not in _c:
        p = BARS_1D / f"{t}.parquet"
        d = None
        if p.exists():
            d = pd.read_parquet(p, columns=["timestamp", "high", "low", "close"])
            d["date"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(
                "America/New_York").dt.date
            prev = d["close"].shift(1)
            tr = pd.concat([d["high"] - d["low"], (d["high"] - prev).abs(),
                            (d["low"] - prev).abs()], axis=1).max(axis=1)
            d["atr"] = tr.rolling(14, min_periods=5).mean()
            d = d.reset_index(drop=True)
        _c[t] = d
    return _c[t]


def fwd_mfe(t, day):
    d = daily(t)
    if d is None or len(d) < 30:
        return None
    idx = d.index[d["date"] > day]
    if len(idx) == 0:
        return None
    i0 = int(idx[0])
    w = d.iloc[i0:i0 + FWD_DAYS]
    if len(w) < FWD_DAYS // 2:
        return None
    ref = float(d["close"].iloc[i0 - 1]) if i0 > 0 else float(w["close"].iloc[0])
    atr = float(d["atr"].iloc[max(0, i0 - 1)])
    if not (np.isfinite(atr) and atr > 0 and ref > 0):
        return None
    return (float(w["high"].max()) - ref) / atr


def main() -> None:
    u = pd.read_csv(REPO / "Data/shared/universe/shared_universe.csv")
    universe = sorted(set(u.loc[u["is_eligible"] == True, "ticker"].astype(str)))  # noqa: E712
    universe = [t for t in universe if (BARS_1D / f"{t}.parquet").exists()]
    print(f"eligible universe with daily bars: {len(universe)}")

    ranked = defaultdict(set)
    for line in (DATA / "stage2_signal_spine.jsonl").open():
        r = json.loads(line)
        if not r.get("submit"):
            continue
        day = datetime.fromisoformat(r["available_at"].replace("Z", "+00:00")).date()
        ranked[(r["module"], day)].add(r["ticker"])
    print(f"module-days with ranked targets: {len(ranked)}\n")

    days = sorted({d for _m, d in ranked})
    fwd_cache: dict[tuple, float] = {}
    for d in days:
        for t in universe:
            k = (t, d)
            if k not in fwd_cache:
                v = fwd_mfe(t, d)
                if v is not None:
                    fwd_cache[k] = v

    print(f"{'module':24s} {'bars':>5s} {'top-k':>6s} {'base rate':>10s} "
          f"{'precision':>10s} {'lift':>6s} {'RECALL':>8s}")
    for module in sorted({m for m, _d in ranked}):
        rows = []
        for (m, d), names in ranked.items():
            if m != module:
                continue
            pool = {t: fwd_cache[(t, d)] for t in universe if (t, d) in fwd_cache}
            if len(pool) < 200:
                continue
            thr = np.percentile(list(pool.values()), 95)      # top 5% = "big mover"
            movers = {t for t, v in pool.items() if v >= thr}
            picked = names & set(pool)
            if not picked or not movers:
                continue
            rows.append({
                "n_pool": len(pool), "n_pick": len(picked),
                "hit": len(picked & movers),
                "base": len(movers) / len(pool),
            })
        if len(rows) < 5:
            print(f"{module:24s}   too few usable bars ({len(rows)})")
            continue
        pick = sum(r["n_pick"] for r in rows)
        hit = sum(r["hit"] for r in rows)
        movers_tot = sum(int(round(r["base"] * r["n_pool"])) for r in rows)
        base = float(np.mean([r["base"] for r in rows]))
        prec = hit / pick if pick else float("nan")
        rec = hit / movers_tot if movers_tot else float("nan")
        print(f"{module:24s} {len(rows):5d} {pick/len(rows):6.1f} {base:10.1%} "
              f"{prec:10.1%} {prec/base if base else float('nan'):5.2f}x {rec:8.2%}")

    print("\n  precision > base rate -> the ranking carries information")
    print("  RECALL is low BY CONSTRUCTION for a top-10 out of ~2,900: the ceiling is")
    print("  10/2900 = 0.34% if every pick were a mover. Read recall against that ceiling,")
    print("  not against 100%.")


if __name__ == "__main__":
    main()
