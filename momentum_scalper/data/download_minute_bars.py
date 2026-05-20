"""Polygon 1-minute OHLCV downloader.

Requires POLYGON_API_KEY in the environment. Output is partitioned as
data/minute_bars/ticker=XYZ/YYYY-MM.parquet and includes premarket bars.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from momentum_scalper.configs.settings import MINUTE_BARS_DIR, ensure_data_dirs, polygon_api_key
from momentum_scalper.utils.io import clean_ticker, month_starts, write_parquet


BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end}"
FIELDS = ["timestamp", "ticker", "open", "high", "low", "close", "volume", "vwap", "trade_count"]


def _fetch_json(url: str) -> dict:
    with urlopen(url, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_month(ticker: str, month: pd.Timestamp, adjusted: bool = True, api_key: str | None = None) -> pd.DataFrame:
    key = api_key or polygon_api_key()
    if not key:
        raise RuntimeError("Set POLYGON_API_KEY before downloading Polygon minute bars.")
    ticker = clean_ticker(ticker)
    start = month.strftime("%Y-%m-%d")
    end = (month + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    params = urlencode({"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50000, "apiKey": key})
    url = f"{BASE_URL.format(ticker=ticker, start=start, end=end)}?{params}"
    payload = _fetch_json(url)
    rows = payload.get("results", [])
    if not rows:
        return pd.DataFrame(columns=FIELDS)
    df = pd.DataFrame(rows).rename(
        columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "trade_count"}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["ticker"] = ticker
    for col in FIELDS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[FIELDS].sort_values("timestamp").reset_index(drop=True)


def download_range(tickers: list[str], start: str, end: str, output_dir: Path = MINUTE_BARS_DIR) -> list[Path]:
    ensure_data_dirs()
    written: list[Path] = []
    for ticker in tickers:
        symbol = clean_ticker(ticker)
        for month in month_starts(start, end):
            df = download_month(symbol, month)
            path = output_dir / f"ticker={symbol}" / f"{month:%Y-%m}.parquet"
            written.append(write_parquet(df, path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Polygon 1m bars")
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    paths = download_range(args.tickers, args.start, args.end)
    print(f"wrote {len(paths):,} monthly files")


if __name__ == "__main__":
    main()
