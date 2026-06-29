"""
Collect catalyst news for a chosen universe scope.

Splits the (slow) news sweep into two cadences so the nightly job stays fast
while the broad corpus still gets refreshed weekly:

  --scope priority  (nightly)  the names the live system actually trades/ranks
                               tightly: swing trading_universe (all tiers) UNION
                               the latest Momentum Expansion universe snapshot.
                               This is the "urgent, act-on-it" set; it feeds the
                               same incremental embed/cluster + news_catalyst
                               signal so breaking news on tradeable names is fresh
                               every night.
  --scope full      (weekly)   every eligible name in shared_universe.csv (the
                               old nightly behavior) — the broad backfill.

collect_company_news merges into news_records.parquet and de-dupes, so the
look-back overlap and the priority⊂full overlap are both safe.

  python -m scripts.collect_news_scope --scope priority
  python -m scripts.collect_news_scope --scope full --lookback-days 7
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

SHARED_UNIVERSE = Path("Data/shared/universe/shared_universe.csv")
SWING_UNIVERSE = Path("strategies/multi_ticker_swing/config/trading_universe.json")
MOMENTUM_SNAPSHOT_DIR = Path("strategies/momentum_expansion/data/universe_snapshots")


def _eligible_universe() -> list[str]:
    uni = pd.read_csv(SHARED_UNIVERSE)
    if "is_eligible" in uni.columns:
        uni = uni[uni["is_eligible"].astype(bool)]
    return sorted(uni["ticker"].astype(str).str.upper().unique().tolist())


def _priority_universe() -> list[str]:
    names: set[str] = set()
    # Swing tradeable universe (all tiers — these are the curated names).
    if SWING_UNIVERSE.exists():
        names |= {str(t).upper() for t in json.loads(SWING_UNIVERSE.read_text()).keys()}
    # Latest Momentum Expansion universe snapshot.
    snaps = sorted(glob.glob(str(MOMENTUM_SNAPSHOT_DIR / "universe_*.csv")))
    if snaps:
        msnap = pd.read_csv(snaps[-1])
        tcol = next((c for c in msnap.columns if c.lower() in ("ticker", "symbol")), None)
        if tcol:
            names |= {str(t).upper() for t in msnap[tcol].dropna().unique()}
    return sorted(names)


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect catalyst news for a universe scope.")
    ap.add_argument("--scope", choices=["priority", "full"], required=True)
    ap.add_argument("--lookback-days", type=int, default=3,
                    help="Look-back window (overlap is de-duped on merge).")
    args = ap.parse_args()

    from signals.news.pipeline import collect_company_news
    from signals.news.config import NEWS_RECORDS_PATH

    tickers = _priority_universe() if args.scope == "priority" else _eligible_universe()
    if not tickers:
        print(f"  no tickers for scope={args.scope} — nothing to collect")
        return 0

    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=args.lookback_days)
    df = collect_company_news(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        output_path=NEWS_RECORDS_PATH,
    )
    print(f"  scope={args.scope}: collected over {len(tickers):,} tickers; "
          f"news_records now {len(df):,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
