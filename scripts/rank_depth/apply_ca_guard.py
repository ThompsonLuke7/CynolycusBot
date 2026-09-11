"""Scan the bar cache for unadjusted corporate actions, then re-run the k-vs-k
test with the principled guard instead of the ad-hoc |ret| <= 200% filter.

The config note written on 2026-09-08 cites numbers from the ad-hoc filter. If
the principled guard disagrees, the note is wrong and must be corrected -- that
is the point of running this.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from core.corporate_actions import suspect_sessions  # noqa: E402

DATA = REPO / "research/execution_quality/data"
BARS = REPO / "Data/shared/bars/1d"
HOLDS = [5, 10, 15]


def main() -> None:
    print("=== scanning the bar cache ===")
    rows = []
    files = sorted(BARS.glob("*.parquet"))
    for i, p in enumerate(files, 1):
        try:
            d = pd.read_parquet(p, columns=["timestamp", "open", "close", "volume"])
        except Exception:
            continue
        f = suspect_sessions(d)
        if f.empty:
            continue
        ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(
            "America/New_York").dt.normalize().dt.tz_localize(None)
        for r in f.itertuples():
            rows.append({"ticker": p.stem, "date": ts.iloc[int(r.idx)],
                         "gap_ratio": r.gap_ratio, "volume_ratio": r.volume_ratio,
                         "dollar_volume_ratio": r.dollar_volume_ratio})
        if i % 1000 == 0:
            print(f"  {i}/{len(files)}  tickers flagged so far: "
                  f"{len({x['ticker'] for x in rows})}")
    flags = pd.DataFrame(rows)
    flags.to_parquet(DATA / "corporate_action_flags.parquet")
    print(f"\nflagged {len(flags)} sessions across {flags['ticker'].nunique()} "
          f"tickers, of {len(files)} files scanned")
    likely = flags[flags["dollar_volume_ratio"] < 3.0]
    print(f"  of those, {len(likely)} have dollar_volume_ratio < 3 "
          f"(share-count change rather than a real move)")
    print("\nlargest 12 by gap:")
    print(flags.nlargest(12, "gap_ratio")[
        ["ticker", "date", "gap_ratio", "volume_ratio", "dollar_volume_ratio"]
    ].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n\n=== k-vs-k with the guard applied ===")
    m = pd.read_parquet(DATA / "oof_rank_depth_mom.parquet")
    bad = {(r.ticker, r.date) for r in flags.itertuples()}
    # a forward window is contaminated if a flagged session falls inside it
    by_ticker: dict[str, np.ndarray] = {}
    for t, g in flags.groupby("ticker"):
        by_ticker[t] = np.sort(g["date"].to_numpy())
    print(f"tickers with a flag that also appear in the study: "
          f"{len(set(by_ticker) & set(m['ticker'].unique()))}")

    rng = np.random.default_rng(23)
    for h in HOLDS:
        rc = f"ret_{h}"
        s = m[m[rc].notna()].copy()
        # window = entry session .. entry + h sessions (calendar-approximated:
        # h trading days is at most ceil(h*7/5)+4 calendar days)
        span = pd.Timedelta(days=int(np.ceil(h * 7 / 5)) + 4)
        contaminated = np.zeros(len(s), dtype=bool)
        tick = s["ticker"].to_numpy()
        start = s["session_date"].to_numpy() if "session_date" in s.columns \
            else s["decision_day"].to_numpy()
        for t, dates in by_ticker.items():
            mask = tick == t
            if not mask.any():
                continue
            st = start[mask]
            lo = np.searchsorted(dates, st, side="left")
            hi = np.searchsorted(dates, st + span.to_timedelta64(), side="right")
            contaminated[mask] = hi > lo
        n_bad = int(contaminated.sum())
        s = s[~contaminated]
        per = {k: s[s["rank"] <= k].groupby("timestamp")[rc].mean()
               for k in (1, 2, 3, 5, 10)}
        print(f"\n--- {rc}  (dropped {n_bad} contaminated observations, "
              f"{n_bad / max(len(contaminated), 1):.3%}) ---")
        ctrl = s.groupby("timestamp")[rc].mean()
        for k in (1, 2, 3, 5, 10):
            d = (per[k] - ctrl.reindex(per[k].index)).dropna()
            draws = np.array([rng.choice(d.values, len(d), replace=True).mean()
                              for _ in range(2000)])
            lo_, hi_ = np.percentile(draws, [2.5, 97.5])
            p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
            print(f"  k={k:2d} excess {d.mean() * 100:+6.2f}% "
                  f"[{lo_ * 100:6.2f},{hi_ * 100:6.2f}] p={p:.3f}"
                  f"{'*' if p < 0.05 else ''}")
        for a, b in [(2, 3), (3, 10), (2, 10), (1, 2)]:
            d = (per[a] - per[b]).dropna()
            draws = np.array([rng.choice(d.values, len(d), replace=True).mean()
                              for _ in range(2000)])
            lo_, hi_ = np.percentile(draws, [2.5, 97.5])
            p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
            print(f"  k={a} vs k={b}: {d.mean() * 100:+6.2f}pp "
                  f"[{lo_ * 100:6.2f},{hi_ * 100:6.2f}] p={p:.3f}"
                  f"{'*' if p < 0.05 else ''}")


if __name__ == "__main__":
    main()
