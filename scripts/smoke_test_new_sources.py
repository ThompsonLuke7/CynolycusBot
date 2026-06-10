"""Smoke test for the 4 new news/sources.py additions.

Tests each source against live APIs using a tiny ticker set and short date
range. No full data gather — just verifies connectivity + schema shape.

Run from repo root:
    python scripts/smoke_test_new_sources.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SMOKE_TICKERS = ["LMT", "RTX", "BA", "NOC", "MSFT"]
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"


def _check(name: str, df: pd.DataFrame, required_cols: list[str], min_rows: int = 1) -> bool:
    if df is None or not isinstance(df, pd.DataFrame):
        print(f"  [{FAIL}] {name}: returned non-DataFrame")
        return False
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  [{FAIL}] {name}: missing columns {missing}  (got {list(df.columns)})")
        return False
    if len(df) < min_rows:
        print(f"  [{WARN}] {name}: got {len(df)} rows (expected >= {min_rows}) — API may be unavailable or no data")
        print(f"          columns OK: {list(df.columns)}")
        return True  # schema correct, just no data right now
    print(f"  [{PASS}] {name}: {len(df)} rows  cols={list(df.columns)}")
    print(df.head(3).to_string(index=False))
    return True


# ── 1. Earnings Calendar ──────────────────────────────────────────────────────

def test_earnings_calendar() -> bool:
    print("\n[1/4] earnings-calendar (yfinance)")
    try:
        from signals.news.sources import fetch_earnings_calendar

        df = fetch_earnings_calendar(SMOKE_TICKERS)
        return _check(
            "fetch_earnings_calendar",
            df,
            ["ticker", "next_earnings_date", "days_to_earnings", "is_earnings_week", "snapshot_date"],
            min_rows=0,
        )
    except Exception:
        print(f"  [{FAIL}] raised exception:")
        traceback.print_exc()
        return False


# ── 2. Economic Calendar ──────────────────────────────────────────────────────

def test_economic_calendar() -> bool:
    print("\n[2/4] economic-calendar (TradingView)")
    try:
        from signals.news.sources import fetch_economic_calendar

        df = fetch_economic_calendar(
            start=(pd.Timestamp.now() - pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
            end=(pd.Timestamp.now() + pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
        )
        ok = _check(
            "fetch_economic_calendar",
            df,
            ["event", "event_date", "actual", "forecast", "previous", "surprise", "importance", "country"],
            min_rows=0,
        )
        if ok and not df.empty:
            macro_events = df[df["event"].str.contains("CPI|NFP|Fed|FOMC|Payroll|Unemployment", case=False, na=False)]
            print(f"          macro events found: {len(macro_events)}")
        return ok
    except Exception:
        print(f"  [{FAIL}] raised exception:")
        traceback.print_exc()
        return False


# ── 3. NASDAQ Short Interest ──────────────────────────────────────────────────

def test_nasdaq_short_interest() -> bool:
    print("\n[3/4] nasdaq-short-interest (nasdaqtrader.com)")
    try:
        from signals.news.sources import fetch_nasdaq_short_interest

        df = fetch_nasdaq_short_interest(SMOKE_TICKERS)
        ok = _check(
            "fetch_nasdaq_short_interest",
            df,
            ["ticker", "settlement_date", "short_interest", "avg_daily_vol", "days_to_cover", "short_pct_float", "source"],
            min_rows=0,
        )
        if ok and not df.empty:
            # Filter to smoke tickers if present
            hits = df[df["ticker"].isin(SMOKE_TICKERS)]
            if not hits.empty:
                print(f"          sample tickers present: {list(hits['ticker'].unique())}")
                print(hits[["ticker", "settlement_date", "short_interest", "days_to_cover"]].to_string(index=False))
        return ok
    except Exception:
        print(f"  [{FAIL}] raised exception:")
        traceback.print_exc()
        return False


# ── 4. USAspending Contracts ──────────────────────────────────────────────────

def test_usaspending_contracts() -> bool:
    print("\n[4/4] usaspending-contracts (api.usaspending.gov)")
    try:
        from signals.news.sources import fetch_usaspending_contracts

        df = fetch_usaspending_contracts(
            SMOKE_TICKERS,
            start=(pd.Timestamp.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d"),
            end=pd.Timestamp.now().strftime("%Y-%m-%d"),
            min_award_amount=50_000_000,
            max_pages=1,
        )
        ok = _check(
            "fetch_usaspending_contracts",
            df,
            ["ticker", "award_id", "recipient_name", "award_amount", "award_date", "description", "agency", "snapshot_date"],
            min_rows=0,
        )
        if ok and not df.empty:
            print(df[["ticker", "recipient_name", "award_amount", "agency"]].head(5).to_string(index=False))
        return ok
    except Exception:
        print(f"  [{FAIL}] raised exception:")
        traceback.print_exc()
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("Smoke test: new news sources")
    print("=" * 60)

    results = {
        "earnings-calendar": test_earnings_calendar(),
        "economic-calendar": test_economic_calendar(),
        "nasdaq-short-interest": test_nasdaq_short_interest(),
        "usaspending-contracts": test_usaspending_contracts(),
    }

    print("\n" + "=" * 60)
    print("Summary:")
    all_pass = True
    for name, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll smoke tests passed (or warned on no data).")
        return 0
    else:
        print("\nSome tests failed — check output above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
