"""Source adapters for company news APIs and SEC 8-K filings."""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Iterable

import pandas as pd

from news.config import SEC_ADMIN_DESCRIPTION_PATTERNS, SEC_ALPHA_FORMS, SEC_LOW_SIGNAL_FORMS
from news.schema import records_from_frame


def load_env_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs without requiring python-dotenv."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _json_url(url: str, timeout: int = 30) -> object:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _clean_tickers(tickers: Iterable[str]) -> list[str]:
    return [str(t).upper().replace("$", "").strip() for t in tickers if str(t).strip()]


def is_alpha_relevant_sec_filing(row: pd.Series, wanted_forms: set[str]) -> bool:
    """Filter SEC submissions down to filings with plausible catalyst value."""
    form = str(row.get("form") or "").upper().strip()
    if form not in wanted_forms:
        return False
    if form in set(SEC_LOW_SIGNAL_FORMS):
        return False
    description = " ".join(
        str(row.get(field) or "")
        for field in ("primaryDocDescription", "primaryDocument", "items")
    ).lower()
    if any(pattern in description for pattern in SEC_ADMIN_DESCRIPTION_PATTERNS):
        return False
    if form == "6-K" and description.strip() in {"", "form 6-k", "6-k"}:
        return False
    return True


def fetch_finnhub_company_news(
    tickers: Iterable[str],
    *,
    start: str,
    end: str,
    api_key: str | None = None,
    min_interval_s: float = 1.1,
    max_retries: int = 2,
) -> pd.DataFrame:
    load_env_file()
    key = api_key or os.getenv("FINNHUB_API_KEY")
    if not key:
        return records_from_frame(pd.DataFrame(), source="finnhub")
    rows = []
    last_request = 0.0
    for ticker in _clean_tickers(tickers):
        elapsed = time.monotonic() - last_request
        if elapsed < min_interval_s:
            time.sleep(min_interval_s - elapsed)
        params = urllib.parse.urlencode({"symbol": ticker, "from": start, "to": end, "token": key})
        url = f"https://finnhub.io/api/v1/company-news?{params}"
        data = []
        for attempt in range(max_retries + 1):
            try:
                data = _json_url(url)
                break
            except Exception:
                if attempt >= max_retries:
                    data = []
                    break
                time.sleep(min_interval_s * (attempt + 2))
        last_request = time.monotonic()
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


def fetch_sec_filing_news(
    tickers: Iterable[str],
    *,
    start: str,
    end: str,
    forms: Iterable[str] = ("8-K",),
    include_archives: bool = True,
    full_text_limit: int = 0,
) -> pd.DataFrame:
    """Represent alpha-relevant SEC filings as timestamped catalyst records."""
    load_env_file()
    from events.forward_guidance.data.sec_client import SecClient, SecFiling

    sec = SecClient()
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    wanted_forms = {str(form).upper().strip() for form in forms if str(form).strip()}
    ticker_list = _clean_tickers(tickers)
    try:
        ticker_map = sec.company_tickers()
    except Exception:
        ticker_map = pd.DataFrame(columns=["ticker", "cik_str"])
    cik_by_ticker = {
        str(row.ticker).upper(): str(row.cik_str)
        for row in ticker_map.itertuples(index=False)
        if str(getattr(row, "ticker", "")).strip() and str(getattr(row, "cik_str", "")).strip()
    }
    rows = []
    for ticker in ticker_list:
        try:
            cik = cik_by_ticker.get(ticker)
            if not cik:
                continue
            recent = sec.all_submission_filings(cik, include_archives=include_archives)
        except Exception:
            continue
        if recent.empty:
            continue
        recent["filingDate"] = pd.to_datetime(recent["filingDate"], errors="coerce")
        recent["form"] = recent["form"].astype(str).str.upper().str.strip()
        mask = recent["form"].isin(wanted_forms) & recent["filingDate"].between(start_ts, end_ts)
        filtered = recent.loc[mask].copy()
        if len(wanted_forms) > 1:
            filtered = filtered.loc[filtered.apply(is_alpha_relevant_sec_filing, axis=1, wanted_forms=wanted_forms)]
        for _, row in filtered.iterrows():
            body = ""
            url = ""
            form = str(row.get("form") or "").upper().strip()
            accession = str(row.get("accessionNumber") or "")
            if int(full_text_limit or 0) > 0:
                try:
                    filing = SecFiling(
                        cik=cik,
                        accession_number=accession,
                        filing_date=str(row.get("filingDate").date()),
                        form=form,
                        primary_document=str(row.get("primaryDocument")),
                        description=row.get("primaryDocDescription"),
                    )
                    url = filing.document_url
                    body = sec.download_filing_text(filing)[: int(full_text_limit)]
                except Exception:
                    body = ""
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": row["filingDate"],
                    "headline": f"{ticker} files {form}",
                    "summary": row.get("primaryDocDescription") or form,
                    "body": body,
                    "url": url,
                    "source": f"sec_{form.lower().replace(' ', '_').replace('/', '_')}",
                    "source_id": accession,
                }
            )
    return records_from_frame(pd.DataFrame(rows), source="sec_8k")


def fetch_sec_8k_news(
    tickers: Iterable[str],
    *,
    start: str,
    end: str,
    include_archives: bool = True,
    full_text_limit: int = 0,
) -> pd.DataFrame:
    return fetch_sec_filing_news(
        tickers,
        start=start,
        end=end,
        forms=("8-K",),
        include_archives=include_archives,
        full_text_limit=full_text_limit,
    )


def fetch_sec_alpha_filings(
    tickers: Iterable[str],
    *,
    start: str,
    end: str,
    include_archives: bool = True,
    full_text_limit: int = 0,
) -> pd.DataFrame:
    return fetch_sec_filing_news(
        tickers,
        start=start,
        end=end,
        forms=SEC_ALPHA_FORMS,
        include_archives=include_archives,
        full_text_limit=full_text_limit,
    )
