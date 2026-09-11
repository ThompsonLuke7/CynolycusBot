"""Universe and SPY control for the rank-depth grid.

A -3% mean is meaningless without knowing what the average eligible name did over
the same forward window. This computes, per decision day used by the modules:
  * the equal-weight mean forward return of the whole eligible universe
  * SPY's forward return
so every module cell can be read as excess over both.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DATA = REPO / "research/execution_quality/data"
BARS = REPO / "Data/shared/bars/1d"
HOLDS = [5, 8, 10, 15, 21]
_c: dict[str, pd.DataFrame | None] = {}


def daily(t):
    if t not in _c:
        p = BARS / f"{t}.parquet"
        d = None
        if p.exists():
            d = pd.read_parquet(p, columns=["timestamp", "open", "high", "low", "close"])
            d["date"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(
                "America/New_York").dt.date
            d = d.reset_index(drop=True)
        _c[t] = d
    return _c[t]


def fwd(t, day, h):
    d = daily(t)
    if d is None or len(d) < 30:
        return None
    idx = d.index[d["date"] > day]
    if len(idx) == 0:
        return None
    i0 = int(idx[0])
    w = d.iloc[i0:i0 + h]
    if len(w) < max(2, h // 2):
        return None
    e = float(d["open"].iloc[i0])
    if not (np.isfinite(e) and e > 0):
        return None
    return float(w["close"].iloc[-1]) / e - 1.0


def main() -> None:
    days = sorted({datetime.fromisoformat(json.loads(l)["available_at"].replace(
        "Z", "+00:00")).date()
        for l in (DATA / "stage2_signal_spine.jsonl").open()})
    u = pd.read_csv(REPO / "Data/shared/universe/shared_universe.csv")
    uni = sorted(set(u.loc[u["is_eligible"] == True, "ticker"].astype(str)))  # noqa: E712
    uni = [t for t in uni if (BARS / f"{t}.parquet").exists()]
    rng = np.random.default_rng(7)
    sample = list(rng.choice(uni, size=min(600, len(uni)), replace=False))
    print(f"decision days: {len(days)}  universe sample: {len(sample)} of {len(uni)}\n")

    print(f"{'hold':>5s} {'univ mean%':>11s} {'univ med%':>10s} {'univ hit%':>10s} {'SPY%':>8s}")
    out = {}
    for h in HOLDS:
        vals, spys = [], []
        for d in days:
            got = [v for t in sample if (v := fwd(t, d, h)) is not None]
            if len(got) < 100:
                continue
            vals.append(float(np.mean(got)))
            s = fwd("SPY", d, h)
            if s is not None:
                spys.append(s)
        if not vals:
            continue
        out[h] = {"univ_mean": float(np.mean(vals)), "spy_mean": float(np.mean(spys))}
        allv = np.concatenate([[v] for v in vals])
        print(f"{h:5d} {np.mean(vals)*100:11.2f} {np.median(allv)*100:10.2f} "
              f"{'':>10s} {np.mean(spys)*100:8.2f}")
    (DATA / "rank_depth_control.json").write_text(json.dumps(out, indent=1))
    print("\nRead every module cell as EXCESS over the 'univ mean' column: that is what")
    print("an equal-weight random draw from the same eligible universe returned over the")
    print("same forward windows, averaged across the same decision days.")


if __name__ == "__main__":
    main()
