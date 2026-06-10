"""Audit what fraction of >100% 20-day-forward runners have a news catalyst.

Used to verify that the catalyst-module collection expansion (yfinance,
Google News RSS, EX-99 enrichment) closed the coverage gap measured before
the rerun (4.7% T-7d, 18.3% T-30d on a 600-ticker sample).

Usage:
    .venv/bin/python scripts/audit_runner_news_coverage.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from strategies.momentum_expansion.config.momentum_config import RAW_1D_DIR

BARS_DIR = RAW_1D_DIR
NEWS_RECORDS = Path("signals/news/data/processed/news_records.parquet")
WINNER_LIB = Path("signals/news/data/processed/winner_news_library.parquet")


def find_runners(min_runup: float = 1.0, window: int = 20, start: str = "2023-01-01") -> pd.DataFrame:
    rows: list[dict] = []
    for fname in sorted(os.listdir(BARS_DIR)):
        if not fname.endswith(".parquet"):
            continue
        ticker = fname[:-8]
        try:
            b = pd.read_parquet(BARS_DIR / fname, columns=["timestamp", "close"])
        except Exception:
            continue
        if len(b) < window + 5:
            continue
        b["ts"] = pd.to_datetime(b["timestamp"], utc=True)
        b = b.sort_values("ts").reset_index(drop=True)
        b["fwd_max"] = b["close"].rolling(window).max().shift(-window)
        b["fwd_ret"] = b["fwd_max"] / b["close"] - 1
        big = b[b["fwd_ret"] > min_runup]
        for _, r in big.iterrows():
            rows.append({"ticker": ticker, "event_ts": r["ts"], "fwd_max_ret": r["fwd_ret"]})
    runs = pd.DataFrame(rows)
    if runs.empty:
        return runs
    runs = runs[runs["event_ts"] >= pd.Timestamp(start, tz="UTC")].copy()
    if runs.empty:
        return runs
    runs["ym"] = runs["event_ts"].dt.tz_convert(None).dt.to_period("M")
    # one event per ticker-month
    return (
        runs.sort_values("fwd_max_ret", ascending=False)
        .drop_duplicates(["ticker", "ym"])
        .reset_index(drop=True)
    )


def coverage(runners: pd.DataFrame, nr: pd.DataFrame, window_days: int) -> float:
    if runners.empty or nr.empty:
        return 0.0
    nr = nr.copy()
    nr["ts"] = pd.to_datetime(nr["timestamp"], utc=True).dt.tz_convert(None).astype("datetime64[ns]")
    by_ticker = {t: g["ts"].to_numpy() for t, g in nr.groupby("ticker")}
    hits = 0
    for _, r in runners.iterrows():
        arr = by_ticker.get(r["ticker"])
        if arr is None or len(arr) == 0:
            continue
        ev_ts = r["event_ts"]
        if ev_ts.tzinfo is not None:
            ev_ts = ev_ts.tz_convert(None)
        ev = ev_ts.to_datetime64()
        lo = (ev_ts - pd.Timedelta(days=window_days)).to_datetime64()
        if ((arr >= lo) & (arr <= ev)).any():
            hits += 1
    return hits / len(runners)


def main() -> int:
    print("=== Loading bars and finding runner events ===")
    runners = find_runners()
    print(f"Runner events (>100% 20d forward, one per ticker-month): {len(runners)}")
    print(f"Tickers with at least one runner: {runners['ticker'].nunique()}")

    nr = pd.read_parquet(NEWS_RECORDS)
    print(f"\nnews_records rows: {len(nr):,}")
    print(f"  by source (top 10):")
    print(nr["source"].value_counts().head(10).to_string())

    for window in (7, 14, 30, 60):
        cov = coverage(runners, nr, window)
        print(f"  coverage T-{window:>2}d: {cov * 100:5.1f}%")

    # Per-source coverage
    print(f"\n=== Per-source coverage (T-30d) ===")
    for src in nr["source"].value_counts().head(8).index:
        cov = coverage(runners, nr[nr["source"] == src], 30)
        print(f"  {src:<20}: {cov * 100:5.1f}%")

    if WINNER_LIB.exists():
        wl = pd.read_parquet(WINNER_LIB)
        print(f"\nwinner library rows: {len(wl):,}")
        print(f"  unique tickers in winner library: {wl['ticker'].nunique()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
