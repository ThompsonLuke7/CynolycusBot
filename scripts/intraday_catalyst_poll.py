"""Intraday catalyst polling + scoring (live mode).

Runs as a long-lived process during market hours. Every ``--interval`` seconds:
  1. Pulls fresh records from light sources (Google News RSS, yfinance,
     Fed RSS, Finnhub, ClinicalTrials.gov).
  2. Drops anything already in the live ledger (by content_hash).
  3. Runs the catalyst classifier on each new record (~20-100 ms each
     including BGE+FinBERT inference).
  4. Appends to news/data/processed/live_catalyst_records.parquet with
     fields: ticker, timestamp, source, headline, catalyst_family,
     catalyst_subtype, catalyst_score, scored_at.
  5. Optionally emits a Webhook / shared-stream event for live trade
     entries (left as a hook).

Designed for cron-style invocation in addition to long-running daemon mode:
  # daemon (run during market hours):
  python scripts/intraday_catalyst_poll.py --interval 300 --universe-from Data/shared/universe/shared_universe.csv

  # single shot (for cron):
  python scripts/intraday_catalyst_poll.py --once --universe-from Data/shared/universe/shared_universe.csv

This is intentionally lightweight and CPU-only for the inference step.
Per-record latency on the test box: ~80 ms (BGE) + ~30 ms (FinBERT) + ~3 ms
(booster) = ~115 ms / record. With ~1000 records per poll, a poll takes
~2 minutes, well under the 5-minute target interval.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from news.catalyst_types import classify_catalyst_types
from news.config import NEWS_RECORDS_PATH
from news.dedup import deduplicate_news
from news.live_scorer import CatalystScorer
from news.schema import empty_news_frame
from news.sources import (
    fetch_clinicaltrials_updates,
    fetch_fed_press_releases,
    fetch_finnhub_company_news,
    fetch_google_news_rss,
    fetch_yfinance_news,
)


LIVE_LEDGER_PATH = Path("news/data/processed/live_catalyst_records.parquet")


def load_universe(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        from meta_context.config import CONTEXT_BACKTEST_UNIVERSE_PATH
        path = CONTEXT_BACKTEST_UNIVERSE_PATH
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise SystemExit(f"universe file {path} missing 'ticker' column")
    # If the universe has a per-row `is_eligible` flag, honor it
    if "is_eligible" in df.columns:
        df = df[df["is_eligible"].fillna(True).astype(bool)]
    return df["ticker"].astype(str).str.upper().tolist()


def collect_recent_records(
    tickers: list[str],
    *,
    lookback_minutes: int,
    sources: list[str],
) -> pd.DataFrame:
    """Pull only fresh records (timestamp within lookback)."""
    end = datetime.now(timezone.utc).replace(microsecond=0)
    start = end - pd.Timedelta(minutes=lookback_minutes)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    frames: list[pd.DataFrame] = []

    if "google_news" in sources:
        frames.append(fetch_google_news_rss(tickers, start=start_str, end=end_str, max_per_ticker=10))
    if "yfinance" in sources:
        frames.append(fetch_yfinance_news(tickers, start=start_str, end=end_str, max_per_ticker=5))
    if "finnhub" in sources:
        frames.append(fetch_finnhub_company_news(tickers, start=start_str, end=end_str))
    if "fed_rss" in sources:
        frames.append(fetch_fed_press_releases(start=start_str, end=end_str, max_items=50))
    if "clinicaltrials" in sources:
        frames.append(fetch_clinicaltrials_updates(tickers, start=start_str, end=end_str, max_pages=2))

    raw = pd.concat(frames, ignore_index=True) if frames else empty_news_frame()
    if raw.empty:
        return raw
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    # Keep only the lookback window
    cutoff = pd.Timestamp(start, tz="UTC")
    raw = raw[raw["timestamp"] >= cutoff].copy()
    return deduplicate_news(raw)


def existing_content_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    if NEWS_RECORDS_PATH.exists():
        try:
            nr = pd.read_parquet(NEWS_RECORDS_PATH, columns=["content_hash"])
            hashes |= set(nr["content_hash"].dropna().astype(str))
        except Exception:
            pass
    if path.exists():
        try:
            live = pd.read_parquet(path, columns=["content_hash"])
            hashes |= set(live["content_hash"].dropna().astype(str))
        except Exception:
            pass
    return hashes


def poll_once(
    scorer: CatalystScorer,
    tickers: list[str],
    *,
    lookback_minutes: int,
    sources: list[str],
    ledger_path: Path,
) -> int:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] polling {len(tickers)} tickers, {lookback_minutes}m lookback...")
    new = collect_recent_records(tickers, lookback_minutes=lookback_minutes, sources=sources)
    if new.empty:
        print("  no new records")
        return 0

    seen = existing_content_hashes(ledger_path)
    new = new[~new["content_hash"].astype(str).isin(seen)].copy()
    if new.empty:
        print("  all polled records already seen")
        return 0

    new = classify_catalyst_types(new)
    print(f"  scoring {len(new):,} new records...")
    new["catalyst_score"] = scorer.score(new.to_dict(orient="records"))
    new["scored_at"] = pd.Timestamp.utcnow()

    cols = [
        "record_id", "ticker", "timestamp", "headline", "source",
        "catalyst_family", "catalyst_subtype", "catalyst_score",
        "scored_at", "content_hash",
    ]
    new = new[[c for c in cols if c in new.columns]]

    if ledger_path.exists():
        prior = pd.read_parquet(ledger_path)
        merged = pd.concat([prior, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=["content_hash"], keep="last")
    else:
        merged = new
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(ledger_path, index=False)

    top = new.sort_values("catalyst_score", ascending=False).head(5)
    print(f"  appended {len(new)} records (ledger now {len(merged):,})")
    print(f"  top 5 catalysts this tick:")
    for _, r in top.iterrows():
        ts = r["timestamp"].strftime("%H:%M")
        print(f"    {ts} {r['ticker']:<8} {r['catalyst_score']:.3f} [{r['catalyst_subtype']}] {str(r['headline'])[:80]}")
    return len(new)


def main() -> int:
    parser = argparse.ArgumentParser(description="Intraday catalyst poll + score")
    parser.add_argument("--universe-from", type=Path, default=Path("Data/shared/universe/shared_universe.csv"))
    parser.add_argument("--interval", type=int, default=300, help="seconds between polls")
    parser.add_argument("--lookback-minutes", type=int, default=15)
    parser.add_argument("--once", action="store_true", help="single poll then exit")
    parser.add_argument("--sources", nargs="*", default=["google_news", "yfinance", "fed_rss"])
    parser.add_argument("--ledger", type=Path, default=LIVE_LEDGER_PATH)
    args = parser.parse_args()

    tickers = load_universe(args.universe_from)
    print(f"universe: {len(tickers)} tickers (from {args.universe_from})")
    scorer = CatalystScorer()

    if args.once:
        poll_once(scorer, tickers,
                  lookback_minutes=args.lookback_minutes,
                  sources=args.sources, ledger_path=args.ledger)
        return 0

    while True:
        try:
            poll_once(scorer, tickers,
                      lookback_minutes=args.lookback_minutes,
                      sources=args.sources, ledger_path=args.ledger)
        except Exception as e:
            print(f"  poll error: {e}", file=sys.stderr)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
