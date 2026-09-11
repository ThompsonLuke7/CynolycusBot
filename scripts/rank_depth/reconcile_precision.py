"""Why does MFE-precision by rank disagree with realized return by rank?

The earlier recall test required each decision bar to have >=200 universe names
with a computable forward MFE, which silently dropped roughly half the bars. This
recomputes precision by rank depth on ALL bars, and puts realized return next to
it, so the two metrics can be compared on identical samples.
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
BARS = REPO / "Data/shared/bars/1d"
FWD = 10
DEPTHS = [1, 2, 3, 5, 10]
_c: dict[str, pd.DataFrame | None] = {}


def daily(t):
    if t not in _c:
        p = BARS / f"{t}.parquet"
        d = None
        if p.exists():
            d = pd.read_parquet(p, columns=["timestamp", "open", "high", "low", "close"])
            d["date"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(
                "America/New_York").dt.date
            prev = d["close"].shift(1)
            tr = pd.concat([d["high"] - d["low"], (d["high"] - prev).abs(),
                            (d["low"] - prev).abs()], axis=1).max(axis=1)
            d["atr"] = tr.rolling(14, min_periods=5).mean()
            d = d.reset_index(drop=True)
        _c[t] = d
    return _c[t]


def mfe_and_ret(t, day):
    d = daily(t)
    if d is None or len(d) < 30:
        return None
    idx = d.index[d["date"] > day]
    if len(idx) == 0:
        return None
    i0 = int(idx[0])
    w = d.iloc[i0:i0 + FWD]
    if len(w) < FWD // 2:
        return None
    e = float(d["open"].iloc[i0])
    atr = float(d["atr"].iloc[i0 - 1]) if i0 > 0 else np.nan
    if not (np.isfinite(e) and e > 0 and np.isfinite(atr) and atr > 0):
        return None
    return ((float(w["high"].max()) - e) / atr, float(w["close"].iloc[-1]) / e - 1.0)


def main() -> None:
    u = pd.read_csv(REPO / "Data/shared/universe/shared_universe.csv")
    uni = sorted(set(u.loc[u["is_eligible"] == True, "ticker"].astype(str)))  # noqa: E712
    uni = [t for t in uni if (BARS / f"{t}.parquet").exists()]

    ranked = defaultdict(dict)
    for line in (DATA / "stage2_signal_spine.jsonl").open():
        r = json.loads(line)
        if not r.get("submit") or r.get("rank") is None:
            continue
        day = datetime.fromisoformat(r["available_at"].replace("Z", "+00:00")).date()
        ranked[(r["module"], r["available_at"], day)][r["ticker"]] = int(r["rank"])

    days = sorted({d for _m, _a, d in ranked})
    pool: dict = {}
    for d in days:
        got = {}
        for t in uni:
            v = mfe_and_ret(t, d)
            if v is not None:
                got[t] = v
        pool[d] = got
    print("pool size per decision day: "
          f"min={min(len(v) for v in pool.values())} "
          f"median={int(np.median([len(v) for v in pool.values()]))} "
          f"max={max(len(v) for v in pool.values())}")
    small = sum(1 for v in pool.values() if len(v) < 200)
    print(f"days the old >=200 filter would have DROPPED: {small} of {len(pool)}\n")

    for min_pool, tag in [(200, "OLD filter (pool>=200)"), (1, "ALL bars")]:
        print(f"===== {tag} =====")
        print(f"{'module':24s} {'bars':>5s} {'base%':>6s}  " +
              "  ".join(f"{'p@'+str(k):>7s}" for k in DEPTHS) + "   " +
              "  ".join(f"{'r@'+str(k):>7s}" for k in DEPTHS))
        for module in sorted({m for m, _a, _d in ranked}):
            hit = defaultdict(int); tot = defaultdict(int)
            rets = defaultdict(list)
            bases, nbars = [], 0
            for (m, _a, d), names in ranked.items():
                if m != module:
                    continue
                p = pool[d]
                if len(p) < min_pool:
                    continue
                thr = np.percentile([v[0] for v in p.values()], 95)
                movers = {t for t, v in p.items() if v[0] >= thr}
                bases.append(len(movers) / len(p))
                nbars += 1
                for t, rk in names.items():
                    if t not in p:
                        continue
                    for k in DEPTHS:
                        if rk <= k:
                            tot[k] += 1
                            hit[k] += int(t in movers)
                            rets[k].append(p[t][1])
            if nbars < 5:
                continue
            prec = [hit[k] / tot[k] if tot[k] else np.nan for k in DEPTHS]
            rr = [float(np.mean(rets[k])) if rets[k] else np.nan for k in DEPTHS]
            print(f"{module:24s} {nbars:5d} {np.mean(bases)*100:6.1f}  " +
                  "  ".join(f"{x*100:7.1f}" for x in prec) + "   " +
                  "  ".join(f"{x*100:7.2f}" for x in rr))
        print()
    print("p@k = % of top-k picks that reached the universe's top-5% forward MFE")
    print("r@k = mean realized 10-day open-to-close return of those same picks, %")


if __name__ == "__main__":
    main()
