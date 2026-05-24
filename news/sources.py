"""Source adapters for company news APIs and SEC 8-K filings."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import date
from typing import Iterable

import pandas as pd

from news.schema import records_from_frame


def _json_url(url: str, timeout: int = 30) -> object:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _clean_tickers(tickers: Iterable[str]) -> list[str]:
    return [str(t).upper().replace("$", "").strip() for t in tickers if str(t).strip()]


def fetch_finnhub_company_news(tickers: Iterable[str], *, start: str, end: str, api_key: str | None = None) -> pd.DataFrame:
    key = api_key or os.getenv("FINNHUB_API_KEY")
    if not key:
        return records_from_frame(pd.DataFrame(), source="finnhub")
    rows = []
    for ticker in _clean_tickers(tickers):
        params = urllib.parse.urlencode({"symbol": ticker, "from": start, "to": end, "token": key})
        data = _json_url(f"https://finnhub.io/api/v1/company-news?{params}")
        for item in data if isinstance(data, list) else []:
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": pd.to_datetime(item.get("datetime"), unit="s", utc=True),
                    "headline": item.get("headline"),
                    "summary": item.get("summary"),
                    "url": item.get("url"),
                    "source": "finnhub",
                    "source_id": str(item.get("id") or ""),
                }
            )
    return records_from_frame(pd.DataFrame(rows), source="finnhub")


def fetch_fmp_stock_news(tickers: Iterable[str], *, start: str, end: str, api_key: str | None = None) -> pd.DataFrame:
    key = api_key or os.getenv("FMP_API_KEY")
    if not key:
        return records_from_frame(pd.DataFrame(), source="fmp")
    rows = []
    for ticker in _clean_tickers(tickers):
        params = urllib.parse.urlencode({"tickers": ticker, "from": start, "to": end, "apikey": key})
        data = _json_url(f"https://financialmodelingprep.com/stable/news/stock?{params}")
        for item in data if isinstance(data, list) else []:
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": item.get("publishedDate") or item.get("date"),
                    "headline": item.get("title"),
                    "summary": item.get("text") or item.get("site"),
                    "url": item.get("url"),
                    "source": "fmp",
                }
            )
    return records_from_frame(pd.DataFrame(rows), source="fmp")


def fetch_alpha_vantage_news(tickers: Iterable[str], *, api_key: str | None = None) -> pd.DataFrame:
    key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
    if not key:
        return records_from_frame(pd.DataFrame(), source="alpha_vantage")
    params = urllib.parse.urlencode({"function": "NEWS_SENTIMENT", "tickers": ",".join(_clean_tickers(tickers)), "apikey": key})
    data = _json_url(f"https://www.alphavantage.co/query?{params}")
    rows = []
    for item in data.get("feed", []) if isinstance(data, dict) else []:
        ticker_rows = item.get("ticker_sentiment") or []
        symbols = [r.get("ticker") for r in ticker_rows if r.get("ticker")]
        for ticker in symbols:
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": item.get("time_published"),
                    "headline": item.get("title"),
                    "summary": item.get("summary"),
                    "url": item.get("url"),
                    "source": "alpha_vantage",
                }
            )
    return records_from_frame(pd.DataFrame(rows), source="alpha_vantage")


def fetch_sec_8k_news(tickers: Iterable[str], *, start: str, end: str) -> pd.DataFrame:
    """Represent SEC 8-K filings as timestamped catalyst records."""
    from events.forward_guidance.data.sec_client import SecClient

    sec = SecClient()
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    rows = []
    for ticker in _clean_tickers(tickers):
        cik = sec.ticker_to_cik(ticker)
        if not cik:
            continue
        sub = sec.submissions(cik)
        recent = pd.DataFrame(sub.get("filings", {}).get("recent", {}))
        if recent.empty or "filingDate" not in recent.columns:
            continue
        recent["filingDate"] = pd.to_datetime(recent["filingDate"], errors="coerce")
        mask = recent["form"].eq("8-K") & recent["filingDate"].between(start_ts, end_ts)
        for _, row in recent.loc[mask].iterrows():
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": row["filingDate"],
                    "headline": f"{ticker} files 8-K",
                    "summary": row.get("primaryDocDescription") or row.get("form"),
                    "url": "",
                    "source": "sec_8k",
                    "source_id": row.get("accessionNumber"),
                }
            )
    return records_from_frame(pd.DataFrame(rows), source="sec_8k")

