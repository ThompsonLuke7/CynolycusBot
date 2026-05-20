"""Polygon news downloader."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from momentum_scalper.configs.settings import NEWS_DIR, ensure_data_dirs, polygon_api_key
from momentum_scalper.utils.io import clean_ticker, write_parquet


FIELDS = ["timestamp", "ticker", "headline", "description", "publisher", "url"]


def _fetch(url: str) -> dict:
    with urlopen(url, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_news(tickers: list[str], start: str, end: str, api_key: str | None = None) -> pd.DataFrame:
    key = api_key or polygon_api_key()
    if not key:
        raise RuntimeError("Set POLYGON_API_KEY before downloading Polygon news.")
    rows: list[dict] = []
    for ticker in [clean_ticker(t) for t in tickers]:
        params = urlencode({"ticker": ticker, "published_utc.gte": start, "published_utc.lte": end, "limit": 1000, "apiKey": key})
        payload = _fetch(f"https://api.polygon.io/v2/reference/news?{params}")
        for item in payload.get("results", []):
            publisher = item.get("publisher") or {}
            rows.append(
                {
                    "timestamp": item.get("published_utc"),
                    "ticker": ticker,
                    "headline": item.get("title"),
                    "description": item.get("description"),
                    "publisher": publisher.get("name"),
                    "url": item.get("article_url"),
                }
            )
    df = pd.DataFrame(rows, columns=FIELDS)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def save_daily_news(df: pd.DataFrame, output_dir: Path = NEWS_DIR) -> list[Path]:
    ensure_data_dirs()
    if df.empty:
        return []
    written: list[Path] = []
    for day, part in df.assign(day=df["timestamp"].dt.strftime("%Y-%m-%d")).groupby("day"):
        written.append(write_parquet(part[FIELDS].sort_values("timestamp"), output_dir / f"{day}.parquet"))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Polygon ticker news")
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    paths = save_daily_news(download_news(args.tickers, args.start, args.end))
    print(f"wrote {len(paths):,} daily news files")


if __name__ == "__main__":
    main()
