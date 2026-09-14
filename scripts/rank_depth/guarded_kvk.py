"""k-vs-k on the OOF rank-depth table, screened with core.corporate_actions.

Reads oof_rank_depth_<module>.parquet (written by oof_replication.py) and the
cache-wide flag table (corporate_action_flags.parquet, written by
apply_ca_guard.py). Drops every forward window that contains a flagged session,
reports how many, then prints excess-over-universe by k and paired k-vs-k.

LEVELS here are inflated by survivorship (the bar cache holds almost no delisted
names); the k-vs-k DIFFERENCES are within-bar and are what parameters are set on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "research/execution_quality/data"
HOLDS = [5, 10, 15]
DEPTHS = [1, 2, 3, 5, 10]
PAIRS = [(1, 2), (2, 3), (3, 5), (3, 10), (5, 10)]


def contaminated(s: pd.DataFrame, flags: pd.DataFrame, hold: int) -> np.ndarray:
    by = {t: np.sort(g["date"].to_numpy()) for t, g in flags.groupby("ticker")}
    span = np.timedelta64(int(np.ceil(hold * 7 / 5)) + 4, "D")
    tick, start = s["ticker"].to_numpy(), s["decision_day"].to_numpy()
    bad = np.zeros(len(s), bool)
    for t, dates in by.items():
        mask = tick == t
        if mask.any():
            st = start[mask]
            bad[mask] = (np.searchsorted(dates, st + span, side="right")
                         > np.searchsorted(dates, st, side="left"))
    return bad


def boot(d: pd.Series, rng) -> tuple[float, float, float]:
    dr = np.array([rng.choice(d.values, len(d), replace=True).mean() for _ in range(2000)])
    lo, hi = np.percentile(dr, [2.5, 97.5])
    return lo, hi, 2 * min((dr <= 0).mean(), (dr >= 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", choices=["mom", "htf"], required=True)
    args = ap.parse_args()
    m = pd.read_parquet(DATA / f"oof_rank_depth_{args.module}.parquet")
    m["decision_day"] = pd.to_datetime(m["decision_day"])
    flags = pd.read_parquet(DATA / "corporate_action_flags.parquet")
    if "organic" not in flags.columns:
        raise SystemExit("corporate_action_flags.parquet predates the organic rule -- "
                         "re-run scripts/rank_depth/apply_ca_guard.py")
    n_all = len(flags)
    flags = flags[~flags["organic"]]      # real moves are returns, not defects
    print(f"masking {len(flags)} non-organic flags (of {n_all}; "
          f"{(flags['direction'] == 'down').sum()} down-gaps)")
    rng = np.random.default_rng(23)
    print(f"{args.module}: {m['timestamp'].nunique():,} bars, {m['ticker'].nunique()} tickers")
    for h in HOLDS:
        rc = f"ret_{h}"
        s = m[m[rc].notna()].copy()
        bad = contaminated(s, flags, h)
        s = s[~bad]
        ctrl = s.groupby("timestamp")[rc].mean()
        per = {k: s[s["rank"] <= k].groupby("timestamp")[rc].mean() for k in DEPTHS}
        print(f"\n--- {h}-day hold: dropped {int(bad.sum())} flagged obs ({bad.mean():.3%}) ---")
        for k in DEPTHS:
            d = (per[k] - ctrl.reindex(per[k].index)).dropna()
            lo, hi, p = boot(d, rng)
            print(f"  k={k:2d} excess {d.mean()*100:+6.2f}% [{lo*100:6.2f},{hi*100:6.2f}] "
                  f"p={p:.3f}{'*' if p < 0.05 else ''}")
        for a, b in PAIRS:
            d = (per[a] - per[b]).dropna()
            lo, hi, p = boot(d, rng)
            print(f"  k={a} vs k={b:2d}: {d.mean()*100:+6.2f}pp [{lo*100:6.2f},{hi*100:6.2f}] "
                  f"p={p:.3f}{'*' if p < 0.05 else ''}")


if __name__ == "__main__":
    main()
