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
            # Always capture the primary-document URL — it's cheap and required
            # for downstream EX-99 exhibit enrichment. Only download the full
            # text when explicitly requested.
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
                if int(full_text_limit or 0) > 0:
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


# ---------------------------------------------------------------------------
# Free sources (no API key required)
# ---------------------------------------------------------------------------


def fetch_yfinance_news(
    tickers: Iterable[str],
    *,
    start: str | None = None,
    end: str | None = None,
    max_per_ticker: int = 50,
    min_interval_s: float = 0.5,
) -> pd.DataFrame:
    """Pull news per ticker from Yahoo Finance via yfinance.Ticker(t).news.

    No API key needed. Returns ~10-50 articles per ticker covering recent months.
    """
    try:
        import yfinance as yf
    except ImportError:
        return records_from_frame(pd.DataFrame(), source="yfinance")

    start_ts = pd.Timestamp(start, tz="UTC") if start else None
    end_ts = (pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)) if end else None
    rows: list[dict] = []
    last_request = 0.0
    for ticker in _clean_tickers(tickers):
        elapsed = time.monotonic() - last_request
        if elapsed < min_interval_s:
            time.sleep(min_interval_s - elapsed)
        try:
            news = yf.Ticker(ticker).news or []
        except Exception:
            news = []
        last_request = time.monotonic()
        for item in news[:max_per_ticker]:
            content = item.get("content") or item
            title = content.get("title") or item.get("title")
            if not title:
                continue
            pub = (
                content.get("pubDate")
                or content.get("displayTime")
                or item.get("providerPublishTime")
            )
            if pub is None:
                continue
            ts = pd.to_datetime(pub, utc=True, errors="coerce", unit="s" if isinstance(pub, (int, float)) else None)
            if pd.isna(ts):
                ts = pd.to_datetime(pub, utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            summary = content.get("summary") or content.get("description") or item.get("summary") or ""
            url = (
                (content.get("canonicalUrl") or {}).get("url")
                or (content.get("clickThroughUrl") or {}).get("url")
                or item.get("link")
                or ""
            )
            provider_obj = content.get("provider") or {}
            provider = provider_obj.get("displayName") if isinstance(provider_obj, dict) else str(provider_obj)
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": ts,
                    "headline": title,
                    "summary": summary,
                    "url": url,
                    "source": "yfinance",
                    "source_id": str(item.get("id") or item.get("uuid") or ""),
                    "publisher": provider or "",
                }
            )
    return records_from_frame(pd.DataFrame(rows), source="yfinance")


def fetch_google_news_rss(
    tickers: Iterable[str],
    *,
    start: str | None = None,
    end: str | None = None,
    max_per_ticker: int = 80,
    query_template: str = '"{ticker}" stock',
    min_interval_s: float = 1.0,
    timeout: int = 20,
) -> pd.DataFrame:
    """Pull headlines per ticker from Google News RSS.

    No API key needed. Google News indexes PR Newswire, GlobeNewswire, Reuters,
    Bloomberg, Benzinga, and most major financial press, so this single source
    substitutes for several direct RSS subscriptions.
    """
    import xml.etree.ElementTree as ET

    start_ts = pd.Timestamp(start, tz="UTC") if start else None
    end_ts = (pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)) if end else None
    rows: list[dict] = []
    last_request = 0.0
    for ticker in _clean_tickers(tickers):
        elapsed = time.monotonic() - last_request
        if elapsed < min_interval_s:
            time.sleep(min_interval_s - elapsed)
        query = urllib.parse.quote(query_template.format(ticker=ticker))
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (CynolycusBot research)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                xml_text = resp.read().decode("utf-8", errors="ignore")
        except Exception:
            xml_text = ""
        last_request = time.monotonic()
        if not xml_text:
            continue
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            continue
        items = root.findall(".//item")[:max_per_ticker]
        for it in items:
            title_el = it.find("title")
            link_el = it.find("link")
            pub_el = it.find("pubDate")
            src_el = it.find("source")
            desc_el = it.find("description")
            if title_el is None or pub_el is None:
                continue
            ts = pd.to_datetime(pub_el.text, utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": ts,
                    "headline": (title_el.text or "").strip(),
                    "summary": (desc_el.text or "").strip() if desc_el is not None else "",
                    "url": (link_el.text or "").strip() if link_el is not None else "",
                    "source": "google_news_rss",
                    "source_id": (link_el.text or "")[:200] if link_el is not None else "",
                    "publisher": (src_el.text or "").strip() if src_el is not None else "",
                }
            )
    return records_from_frame(pd.DataFrame(rows), source="google_news_rss")


def enrich_sec_8k_ex99_text(
    records: pd.DataFrame,
    *,
    max_chars: int = 30000,
    min_interval_s: float = 0.15,
    timeout: int = 20,
    item_filter: tuple[str, ...] | None = ("2.02", "7.01", "8.01"),
) -> pd.DataFrame:
    """Download the EX-99 press-release exhibit attached to 8-K filings and
    populate the ``body`` field with stripped text.

    This is the free path to actual earnings prepared remarks and guidance
    language: 8-K Item 2.02 (Results of Operations) almost always attaches the
    earnings release as EX-99.1, which contains revenue, EPS, guidance, and
    management commentary in plain prose.
    """
    if records.empty:
        return records

    from bs4 import BeautifulSoup  # type: ignore

    out = records.copy()
    needs_body = (out["source"].astype(str).eq("sec_8-k")) & (out["body"].fillna("").str.len() < 200)
    targets = out.loc[needs_body].copy()
    if targets.empty:
        return out

    last_request = 0.0
    headers = {"User-Agent": "CynolycusBot research@example.com"}

    def _filing_index(accession: str, cik: str) -> str | None:
        acc_no = accession.replace("-", "")
        return f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K&dateb=&owner=include&count=10&action=getcompany"

    def _get(url: str) -> str:
        nonlocal last_request
        elapsed = time.monotonic() - last_request
        if elapsed < min_interval_s:
            time.sleep(min_interval_s - elapsed)
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                txt = resp.read().decode("utf-8", errors="ignore")
        except Exception:
            txt = ""
        last_request = time.monotonic()
        return txt

    updated = 0
    for idx, row in targets.iterrows():
        url = str(row.get("url") or "")
        if not url:
            continue
        # url points to the primary 8-K document; the EX-99 lives in the same accession folder
        # derive the index URL by replacing the primary doc filename with 'index.json'
        try:
            parts = url.rsplit("/", 1)
            base = parts[0]
            index_json_url = f"{base}/index.json"
        except Exception:
            continue
        index_text = _get(index_json_url)
        if not index_text:
            continue
        try:
            index_obj = json.loads(index_text)
            items = index_obj.get("directory", {}).get("item", [])
        except Exception:
            continue
        # Match common EX-99 naming variants. Skip images and ancillary files
        # so we don't grab the company logo or XBRL bookkeeping.
        ex99_substrings = ("ex99", "ex-99", "ex_99", "exhibit99", "exhibit-99", "exhibit_99")
        skip_ext = {".jpg", ".jpeg", ".png", ".gif", ".xml", ".xsd", ".css", ".js", ".zip", ".xlsx"}
        candidates = []
        for it in items:
            name = str(it.get("name", "") or "")
            lower = name.lower()
            if any(lower.endswith(ext) for ext in skip_ext):
                continue
            if not any(s in lower for s in ex99_substrings):
                continue
            try:
                size = int(it.get("size") or 0)
            except (TypeError, ValueError):
                size = 0
            # Prefer ex99.1 (earnings press release) over .2 or .3
            sub_idx = 1 if any(s in lower for s in ("ex991", "ex99-1", "ex99_1", "exhibit991", "exhibit-99-1", "exhibit99-1", "exhibit_99-1")) else 9
            candidates.append((sub_idx, -size, name))
        if not candidates:
            continue
        candidates.sort()
        chosen = candidates[0][2]
        doc_url = f"{base}/{chosen}"
        doc_text = _get(doc_url)
        if not doc_text:
            continue
        if doc_text.lstrip().lower().startswith("<"):
            try:
                soup = BeautifulSoup(doc_text, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
            except Exception:
                text = doc_text
        else:
            text = doc_text
        text = " ".join(text.split())[:max_chars]
        if not text:
            continue
        out.at[idx, "body"] = text
        if "text" in out.columns:
            existing = str(out.at[idx, "text"] or "")
            out.at[idx, "text"] = (existing + " " + text).strip()[: max_chars + 500]
        updated += 1

    out.attrs["ex99_enriched_count"] = updated
    return out


# ---------------------------------------------------------------------------
# Federal Reserve press releases (FOMC statements, speeches, rate decisions)
# ---------------------------------------------------------------------------


def fetch_fed_press_releases(
    *,
    start: str | None = None,
    end: str | None = None,
    max_items: int = 200,
    timeout: int = 20,
) -> pd.DataFrame:
    """Pull all Federal Reserve press releases via the public RSS feed.

    No key, no rate limit (light per-day usage expected). Each release becomes
    a market-wide catalyst with ``ticker='SPY'`` so it joins the same schema as
    company news (FOMC statements move SPY/QQQ/IWM in lockstep).
    """
    import xml.etree.ElementTree as ET

    url = "https://www.federalreserve.gov/feeds/press_all.xml"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (CynolycusBot research)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            xml_text = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return records_from_frame(pd.DataFrame(), source="fed_rss")

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return records_from_frame(pd.DataFrame(), source="fed_rss")

    start_ts = pd.Timestamp(start, tz="UTC") if start else None
    end_ts = (pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)) if end else None
    rows: list[dict] = []
    for item in root.findall(".//item")[:max_items]:
        title_el = item.find("title")
        link_el = item.find("link")
        pub_el = item.find("pubDate")
        desc_el = item.find("description")
        if title_el is None or pub_el is None:
            continue
        ts = pd.to_datetime(pub_el.text, utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        headline = (title_el.text or "").strip()
        rows.append(
            {
                "ticker": "SPY",  # market-wide proxy
                "timestamp": ts,
                "headline": headline,
                "summary": (desc_el.text or "").strip() if desc_el is not None else "",
                "url": (link_el.text or "").strip() if link_el is not None else "",
                "source": "fed_rss",
                "source_id": (link_el.text or "")[:200] if link_el is not None else "",
            }
        )
    return records_from_frame(pd.DataFrame(rows), source="fed_rss")


# ---------------------------------------------------------------------------
# OpenFDA — drug approvals, applications, recalls
# ---------------------------------------------------------------------------

# Conservative manual map of public-biotech sponsor names -> ticker. Expanded
# at runtime via the universe-config file when present (one company per line:
# "Sponsor Display Name=TICKER").
_FDA_SPONSOR_TICKER_HINTS: dict[str, str] = {
    "pfizer": "PFE",
    "moderna": "MRNA",
    "biontech": "BNTX",
    "regeneron": "REGN",
    "vertex": "VRTX",
    "gilead": "GILD",
    "amgen": "AMGN",
    "biogen": "BIIB",
    "novartis": "NVS",
    "astrazeneca": "AZN",
    "glaxosmithkline": "GSK",
    "sanofi": "SNY",
    "merck": "MRK",
    "eli lilly": "LLY",
    "lilly": "LLY",
    "bristol myers": "BMY",
    "bristol-myers": "BMY",
    "johnson & johnson": "JNJ",
    "abbvie": "ABBV",
    "alnylam": "ALNY",
    "ionis": "IONS",
    "incyte": "INCY",
    "alkermes": "ALKS",
    "horizon therapeutics": "HZNP",
    "neurocrine": "NBIX",
    "exelixis": "EXEL",
    "blueprint medicines": "BPMC",
    "ultragenyx": "RARE",
    "intra-cellular": "ITCI",
    "sarepta": "SRPT",
    "vera therapeutics": "VERA",
    "wolfspeed": "WOLF",
}


def _resolve_ticker_from_sponsor(name: str) -> str | None:
    if not name:
        return None
    nl = name.lower()
    for needle, ticker in _FDA_SPONSOR_TICKER_HINTS.items():
        if needle in nl:
            return ticker
    return None


def fetch_openfda_drug_approvals(
    *,
    start: str | None = None,
    end: str | None = None,
    limit: int = 1000,
    min_interval_s: float = 0.6,
    timeout: int = 30,
) -> pd.DataFrame:
    """OpenFDA drugsfda endpoint — drug application approvals.

    Free, no key required (rate-limited to 240/min/IP unauthenticated).
    Returns one record per application action (approval, supplement, etc.)
    that we could map to a public-biotech ticker via sponsor name.
    """
    rows: list[dict] = []
    page_size = 100
    skip = 0
    last_request = 0.0
    start_dt = pd.Timestamp(start).date() if start else pd.Timestamp("2023-01-01").date()
    end_dt = pd.Timestamp(end).date() if end else pd.Timestamp.today().date()

    search = f"submissions.submission_status_date:[{start_dt.strftime('%Y%m%d')}+TO+{end_dt.strftime('%Y%m%d')}]"
    while skip < limit:
        elapsed = time.monotonic() - last_request
        if elapsed < min_interval_s:
            time.sleep(min_interval_s - elapsed)
        url = f"https://api.fda.gov/drug/drugsfda.json?search={search}&limit={page_size}&skip={skip}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CynolycusBot research@example.com"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
        except Exception:
            break
        last_request = time.monotonic()
        results = data.get("results", []) if isinstance(data, dict) else []
        if not results:
            break
        for r in results:
            sponsor = str(r.get("sponsor_name") or "")
            ticker = _resolve_ticker_from_sponsor(sponsor)
            if not ticker:
                continue
            products = r.get("products") or []
            brand = ""
            if products:
                brand = str((products[0] or {}).get("brand_name") or "")
            for sub in r.get("submissions") or []:
                status = str(sub.get("submission_status") or "")
                status_date = sub.get("submission_status_date") or ""
                if not status_date:
                    continue
                try:
                    ts = pd.to_datetime(status_date, format="%Y%m%d", utc=True)
                except Exception:
                    continue
                # OpenFDA's search filter is unreliable for date ranges; post-filter here.
                if ts.date() < start_dt or ts.date() > end_dt:
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "timestamp": ts,
                        "headline": f"FDA {status}: {brand or r.get('application_number', 'application')} ({sponsor})".strip(),
                        "summary": f"Application {r.get('application_number','')} sponsor={sponsor} class={sub.get('submission_class_code_description','')}",
                        "url": f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={r.get('application_number','')}",
                        "source": "openfda",
                        "source_id": f"{r.get('application_number','')}-{sub.get('submission_number','')}-{status_date}",
                    }
                )
        skip += page_size
        if len(results) < page_size:
            break
    return records_from_frame(pd.DataFrame(rows), source="openfda")


# ---------------------------------------------------------------------------
# ClinicalTrials.gov v2 API — biotech trial status changes
# ---------------------------------------------------------------------------


def fetch_clinicaltrials_updates(
    tickers: Iterable[str] | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
    page_size: int = 100,
    max_pages: int = 50,
    min_interval_s: float = 0.5,
    timeout: int = 30,
) -> pd.DataFrame:
    """ClinicalTrials.gov v2 API — recently updated trials by sponsor.

    Free, no key required. Optionally restrict to known biotech sponsors
    via the _FDA_SPONSOR_TICKER_HINTS reverse mapping.
    """
    target_sponsors = {
        ticker: sponsor for sponsor, ticker in _FDA_SPONSOR_TICKER_HINTS.items()
    }
    selected_tickers = (
        {t.upper() for t in (tickers or [])} & set(target_sponsors)
    ) or set(target_sponsors)

    rows: list[dict] = []
    last_request = 0.0
    start_str = (start or "2023-01-01")
    end_str = end or pd.Timestamp.today().strftime("%Y-%m-%d")

    for ticker in selected_tickers:
        sponsor_name = target_sponsors.get(ticker)
        if not sponsor_name:
            continue
        next_token = None
        for page in range(max_pages):
            elapsed = time.monotonic() - last_request
            if elapsed < min_interval_s:
                time.sleep(min_interval_s - elapsed)
            params = {
                "query.spons": sponsor_name,
                "filter.advanced": f"AREA[LastUpdatePostDate]RANGE[{start_str},{end_str}]",
                "pageSize": str(page_size),
                "fields": "NCTId,BriefTitle,OverallStatus,Phase,LeadSponsorName,LastUpdatePostDate,PrimaryCompletionDate,StudyType",
            }
            if next_token:
                params["pageToken"] = next_token
            url = "https://clinicaltrials.gov/api/v2/studies?" + urllib.parse.urlencode(params)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "CynolycusBot research@example.com"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read())
            except Exception:
                break
            last_request = time.monotonic()
            studies = data.get("studies") if isinstance(data, dict) else None
            if not studies:
                break
            for st in studies:
                ps = (st.get("protocolSection") or {})
                ident = (ps.get("identificationModule") or {})
                status = (ps.get("statusModule") or {})
                sponsor = (ps.get("sponsorCollaboratorsModule") or {})
                design = (ps.get("designModule") or {})
                last_update = (status.get("lastUpdatePostDateStruct") or {}).get("date")
                if not last_update:
                    continue
                try:
                    ts = pd.to_datetime(last_update, utc=True)
                except Exception:
                    continue
                phase = ", ".join((design.get("phases") or [])) or "n/a"
                overall_status = status.get("overallStatus") or ""
                title = ident.get("briefTitle") or ""
                nct = ident.get("nctId") or ""
                rows.append(
                    {
                        "ticker": ticker,
                        "timestamp": ts,
                        "headline": f"Trial {nct} {overall_status}: {title}",
                        "summary": f"Phase={phase} sponsor={(sponsor.get('leadSponsor') or {}).get('name', '')}",
                        "url": f"https://clinicaltrials.gov/study/{nct}",
                        "source": "clinicaltrials",
                        "source_id": f"{nct}-{last_update}",
                    }
                )
            next_token = data.get("nextPageToken") if isinstance(data, dict) else None
            if not next_token:
                break
    return records_from_frame(pd.DataFrame(rows), source="clinicaltrials")


# ---------------------------------------------------------------------------
# yfinance options chain — unusual-flow detection
# ---------------------------------------------------------------------------


def fetch_yfinance_unusual_options_activity(
    tickers: Iterable[str],
    *,
    sigma_threshold: float = 3.0,
    min_interval_s: float = 0.5,
) -> pd.DataFrame:
    """Snapshot of today's options chain per ticker, flagging strikes whose
    volume is >sigma_threshold * historical-baseline-OI.

    No key; uses yfinance. Generates one record per unusual strike. Best run
    daily after the close — pairs naturally with a daily cron.
    """
    try:
        import yfinance as yf
    except ImportError:
        return records_from_frame(pd.DataFrame(), source="yf_options_flow")

    rows: list[dict] = []
    last_request = 0.0
    today = pd.Timestamp.utcnow().normalize()
    for ticker in _clean_tickers(tickers):
        elapsed = time.monotonic() - last_request
        if elapsed < min_interval_s:
            time.sleep(min_interval_s - elapsed)
        try:
            t = yf.Ticker(ticker)
            expiries = t.options or []
        except Exception:
            expiries = []
        last_request = time.monotonic()
        unusual_strikes: list[dict] = []
        for exp in expiries[:6]:  # nearest 6 expiries — where flow concentrates
            try:
                chain = t.option_chain(exp)
            except Exception:
                continue
            for side, df in (("call", chain.calls), ("put", chain.puts)):
                if df is None or df.empty:
                    continue
                df = df.copy()
                # Heuristic: volume > sigma_threshold * sqrt(OI). Rough proxy
                # for unusual activity since we don't have a historical baseline
                # in a single snapshot.
                df["volume"] = df["volume"].fillna(0)
                df["openInterest"] = df["openInterest"].fillna(0)
                df["impliedVolatility"] = df["impliedVolatility"].fillna(0)
                df["lastPrice"] = df["lastPrice"].fillna(0)
                df["strike"] = df["strike"].fillna(0)
                df["expected"] = (df["openInterest"].clip(lower=1) ** 0.5) * sigma_threshold
                hits = df[df["volume"] > df["expected"]]
                for _, row in hits.iterrows():
                    unusual_strikes.append(
                        {
                            "side": side,
                            "expiry": exp,
                            "strike": float(row["strike"]),
                            "volume": int(row["volume"]),
                            "oi": int(row["openInterest"]),
                            "iv": float(row["impliedVolatility"]),
                            "last": float(row["lastPrice"]),
                        }
                    )
        if not unusual_strikes:
            continue
        # Aggregate into a single record per ticker per day
        call_premium = sum(s["volume"] * s["last"] * 100 for s in unusual_strikes if s["side"] == "call")
        put_premium = sum(s["volume"] * s["last"] * 100 for s in unusual_strikes if s["side"] == "put")
        bias = "bullish" if call_premium > put_premium * 1.5 else ("bearish" if put_premium > call_premium * 1.5 else "mixed")
        headline = (
            f"Unusual options activity ({bias}): "
            f"{sum(s['volume'] for s in unusual_strikes):,} contracts across "
            f"{len(unusual_strikes)} strikes; call premium ${call_premium:,.0f} put premium ${put_premium:,.0f}"
        )
        rows.append(
            {
                "ticker": ticker,
                "timestamp": today,
                "headline": headline,
                "summary": json.dumps(unusual_strikes[:20]),
                "url": f"https://finance.yahoo.com/quote/{ticker}/options",
                "source": "yf_options_flow",
                "source_id": f"{ticker}-{today.strftime('%Y%m%d')}",
            }
        )
    return records_from_frame(pd.DataFrame(rows), source="yf_options_flow")


# ---------------------------------------------------------------------------
# Financial Modeling Prep — earnings call transcripts
# ---------------------------------------------------------------------------


def fetch_fmp_earnings_transcripts(
    tickers: Iterable[str],
    *,
    api_key: str | None = None,
    year: int | None = None,
    quarter: int | None = None,
    max_per_ticker: int = 8,
    min_interval_s: float = 0.4,
    timeout: int = 30,
) -> pd.DataFrame:
    """Pull earnings call transcripts from financialmodelingprep.com.

    Free tier: 250 calls/day, no per-minute throttle. Set FMP_API_KEY in .env
    or pass api_key=. Each transcript becomes a record with the full
    prepared-remarks + Q&A text in body.
    """
    load_env_file()
    key = api_key or os.getenv("FMP_API_KEY")
    if not key:
        return records_from_frame(pd.DataFrame(), source="fmp_transcripts")

    rows: list[dict] = []
    last_request = 0.0
    for ticker in _clean_tickers(tickers):
        elapsed = time.monotonic() - last_request
        if elapsed < min_interval_s:
            time.sleep(min_interval_s - elapsed)
        # First list available transcripts for the ticker
        list_url = f"https://financialmodelingprep.com/api/v4/earning_call_transcript?symbol={ticker}&apikey={key}"
        try:
            req = urllib.request.Request(list_url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                listing = json.loads(resp.read())
        except Exception:
            listing = []
        last_request = time.monotonic()
        if not isinstance(listing, list) or not listing:
            continue
        # listing items shape: [year, quarter, date]
        selected = listing[:max_per_ticker]
        for entry in selected:
            try:
                yr = int(entry[0])
                qt = int(entry[1])
                date_str = str(entry[2])
            except Exception:
                continue
            if year is not None and yr != year:
                continue
            if quarter is not None and qt != quarter:
                continue
            elapsed = time.monotonic() - last_request
            if elapsed < min_interval_s:
                time.sleep(min_interval_s - elapsed)
            t_url = (
                f"https://financialmodelingprep.com/api/v3/earning_call_transcript/"
                f"{ticker}?year={yr}&quarter={qt}&apikey={key}"
            )
            try:
                req = urllib.request.Request(t_url)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read())
            except Exception:
                data = []
            last_request = time.monotonic()
            if not isinstance(data, list) or not data:
                continue
            t = data[0]
            content = str(t.get("content") or "")
            if not content:
                continue
            ts = pd.to_datetime(t.get("date") or date_str, utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": ts,
                    "headline": f"{ticker} Q{qt} {yr} Earnings Call Transcript",
                    "summary": content[:600],
                    "body": content[:30000],
                    "url": "",
                    "source": "fmp_transcripts",
                    "source_id": f"{ticker}-{yr}-Q{qt}",
                }
            )
    return records_from_frame(pd.DataFrame(rows), source="fmp_transcripts")


# ---------------------------------------------------------------------------
# CBOE delayed options-chain snapshot (per-ticker, no auth)
# ---------------------------------------------------------------------------

def _cboe_options_aggregate(payload: dict, *, sigma_threshold: float = 3.0) -> dict:
    """Reduce a CBOE options chain payload into a per-ticker daily summary
    plus a list of unusual-flow strikes (volume > sigma * sqrt(open_interest))."""
    data = payload.get("data") or {}
    options = data.get("options") or []
    current_price = float(data.get("current_price") or 0)
    snapshot_day = pd.Timestamp.utcnow().normalize()
    summary: dict = {
        "current_price": current_price,
        "stock_volume": int(data.get("volume") or 0),
        "iv30": float(data.get("iv30") or 0),
        "iv30_change_percent": float(data.get("iv30_change_percent") or 0),
        "snapshot_timestamp": payload.get("timestamp"),
    }
    call_vol = put_vol = 0
    call_oi = put_oi = 0
    call_premium = put_premium = 0.0
    unusual_strikes: list[dict] = []
    for opt in options:
        sym = str(opt.get("option") or "")
        # OCC symbol convention: ROOT[date 6 chars]C/P[strike 8 chars]
        if len(sym) < 15:
            continue
        side_char = sym[-9:-8]
        is_call = side_char == "C"
        vol = int(opt.get("volume") or 0)
        oi = int(opt.get("open_interest") or 0)
        last = float(opt.get("last_trade_price") or 0)
        if is_call:
            call_vol += vol
            call_oi += oi
            call_premium += vol * last * 100
        else:
            put_vol += vol
            put_oi += oi
            put_premium += vol * last * 100
        baseline = max(oi, 1) ** 0.5 * sigma_threshold
        if vol > baseline and vol > 50:  # also require minimum absolute volume
            strike = float(opt.get("option") and sym[-8:]) / 1000.0 if sym[-8:].isdigit() else 0.0
            expiry_raw = sym[-15:-9]
            expiry = pd.to_datetime(f"20{expiry_raw}", format="%Y%m%d", errors="coerce") if expiry_raw.isdigit() else pd.NaT
            dte = int((expiry - snapshot_day.tz_localize(None)).days) if pd.notna(expiry) else None
            unusual_strikes.append(
                {
                    "contract": sym,
                    "side": "call" if is_call else "put",
                    "expiry": expiry.date().isoformat() if pd.notna(expiry) else "",
                    "dte": dte,
                    "strike": strike,
                    "volume": vol,
                    "open_interest": oi,
                    "iv": float(opt.get("iv") or 0),
                    "delta": float(opt.get("delta") or 0),
                    "last": last,
                    "premium": vol * last * 100,
                    "ratio": vol / max(oi, 1),
                    "strike_distance_pct": (strike / current_price - 1.0) if current_price > 0 else None,
                }
            )
    summary.update(
        {
            "call_volume": call_vol,
            "put_volume": put_vol,
            "call_open_interest": call_oi,
            "put_open_interest": put_oi,
            "call_premium": call_premium,
            "put_premium": put_premium,
            "put_call_volume_ratio": (put_vol / call_vol) if call_vol > 0 else None,
            "unusual_strike_count": len(unusual_strikes),
            "unusual_total_volume": sum(s["volume"] for s in unusual_strikes),
        }
    )
    # Keep all unusual strikes for the strike-level table; ticker-level
    # catalyst records only serialize the top few for readability.
    unusual_strikes.sort(key=lambda s: -s["ratio"])
    summary["unusual_strikes"] = unusual_strikes
    return summary


def fetch_cboe_options_snapshot(
    tickers: Iterable[str],
    *,
    sigma_threshold: float = 3.0,
    min_interval_s: float = 0.5,
    timeout: int = 45,
    max_retries: int = 3,
    retry_backoff_s: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pull per-ticker delayed options chains from CBOE's free CDN.

    Returns two frames:
    - summary_df: one row per ticker with aggregate flow metrics + iv30 +
      stock volume + put-call ratio + unusual-strike count.
    - records_df: catalyst-shaped records (one per ticker with unusual flow)
      suitable for merging into ``news_records.parquet``.
    """
    summary_rows: list[dict] = []
    strike_rows: list[dict] = []
    record_rows: list[dict] = []
    last_request = 0.0
    today = pd.Timestamp.utcnow().normalize()
    for ticker in _clean_tickers(tickers):
        elapsed = time.monotonic() - last_request
        if elapsed < min_interval_s:
            time.sleep(min_interval_s - elapsed)
        url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json"
        payload = None
        for attempt in range(max_retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (CynolycusBot research)"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    payload = json.loads(resp.read())
                break
            except Exception:
                if attempt >= max_retries:
                    payload = None
                else:
                    time.sleep(retry_backoff_s * (attempt + 1))
        last_request = time.monotonic()
        if not payload or "data" not in payload:
            continue
        try:
            summary = _cboe_options_aggregate(payload, sigma_threshold=sigma_threshold)
        except Exception:
            continue
        summary["ticker"] = ticker
        summary["snapshot_date"] = today
        summary_rows.append(summary)
        for strike in summary.get("unusual_strikes", []):
            strike_rows.append(
                {
                    "ticker": ticker,
                    "snapshot_date": today,
                    "snapshot_timestamp": summary.get("snapshot_timestamp"),
                    "current_price": summary.get("current_price"),
                    **strike,
                }
            )

        # Emit a catalyst news_record only if there's meaningful unusual flow
        if summary["unusual_strike_count"] >= 3 and summary["unusual_total_volume"] >= 1000:
            cp = summary["call_premium"]
            pp = summary["put_premium"]
            bias = "bullish" if cp > pp * 1.5 else ("bearish" if pp > cp * 1.5 else "mixed")
            headline = (
                f"Unusual options activity ({bias}): "
                f"{summary['unusual_total_volume']:,} contracts across "
                f"{summary['unusual_strike_count']} strikes; call premium ${cp:,.0f} put premium ${pp:,.0f}; "
                f"iv30={summary['iv30']:.1f} ({summary['iv30_change_percent']:+.1f}%)"
            )
            record_rows.append(
                {
                    "ticker": ticker,
                    "timestamp": today,
                    "headline": headline,
                    "summary": json.dumps(summary["unusual_strikes"][:10]),
                    "url": f"https://www.cboe.com/delayed_quote/{ticker}/quote_table",
                    "source": "cboe_options_flow",
                    "source_id": f"{ticker}-{today.strftime('%Y%m%d')}",
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        # Drop the heavyweight unusual_strikes column from the summary parquet
        # — it's preserved inside the news_record summary field as JSON.
        if "unusual_strikes" in summary_df.columns:
            summary_df = summary_df.drop(columns=["unusual_strikes"])
    records_df = records_from_frame(pd.DataFrame(record_rows), source="cboe_options_flow")
    strike_df = pd.DataFrame(strike_rows)
    return summary_df, records_df, strike_df


# ---------------------------------------------------------------------------
# FINRA daily short-sale volume backfill
# ---------------------------------------------------------------------------


def fetch_finra_short_volume_day(
    date: str | pd.Timestamp,
    *,
    timeout: int = 20,
) -> pd.DataFrame:
    """Pull one trading-day's Consolidated NMS short-sale volume CSV.

    Returns long-format frame with columns:
    date, ticker, short_volume, short_exempt_volume, total_volume, market.
    """
    ts = pd.Timestamp(date)
    url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ts.strftime('%Y%m%d')}.txt"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (CynolycusBot research)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return pd.DataFrame(columns=["date", "ticker", "short_volume", "short_exempt_volume", "total_volume", "market"])
    rows: list[dict] = []
    for line in text.splitlines()[1:]:  # skip header
        parts = line.split("|")
        if len(parts) < 5 or parts[0] == "" or parts[1] in {"", "Symbol"}:
            continue
        try:
            rows.append(
                {
                    "date": pd.to_datetime(parts[0], format="%Y%m%d"),
                    "ticker": parts[1].upper(),
                    "short_volume": float(parts[2] or 0),
                    "short_exempt_volume": float(parts[3] or 0),
                    "total_volume": float(parts[4] or 0),
                    "market": parts[5] if len(parts) > 5 else "",
                }
            )
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(rows)


def backfill_finra_short_volume(
    *,
    start: str,
    end: str,
    output_path: object,
    min_interval_s: float = 0.3,
    progress_every: int = 50,
) -> pd.DataFrame:
    """Backfill the consolidated NMS short-sale volume CSV for every
    trading day in [start, end] into a single long parquet.

    Skips weekends and silently skips any day where FINRA returns 404
    (market holidays).
    """
    from pathlib import Path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    days = pd.date_range(start_ts, end_ts, freq="B")  # business days
    frames: list[pd.DataFrame] = []
    last_request = 0.0
    for i, d in enumerate(days):
        elapsed = time.monotonic() - last_request
        if elapsed < min_interval_s:
            time.sleep(min_interval_s - elapsed)
        df = fetch_finra_short_volume_day(d)
        last_request = time.monotonic()
        if not df.empty:
            frames.append(df)
        if (i + 1) % progress_every == 0:
            print(f"  finra backfill: {i + 1}/{len(days)} days, rows so far: {sum(len(f) for f in frames):,}", flush=True)
    if not frames:
        out = pd.DataFrame()
        out.to_parquet(output_path, index=False)
        return out
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(output_path, index=False)
    return out


def emit_finra_short_spike_records(
    short_volume_path: object,
    *,
    z_threshold: float = 2.0,
    window: int = 20,
    min_total_volume: float = 100000.0,
) -> pd.DataFrame:
    """Generate news_record-shaped catalyst events from a FINRA short-volume
    backfill parquet. Emits one record per (ticker, date) where the
    z-score of short_ratio (per ticker, rolling-mean centered) exceeds
    ``z_threshold`` in either direction.
    """
    from pathlib import Path
    df = pd.read_parquet(short_volume_path) if Path(short_volume_path).exists() else pd.DataFrame()
    if df.empty:
        return records_from_frame(pd.DataFrame(), source="finra_short_spike")
    df = df[df["total_volume"] >= float(min_total_volume)].copy()
    df["short_ratio"] = df["short_volume"] / df["total_volume"].clip(lower=1)
    df = df.sort_values(["ticker", "date"])
    grp = df.groupby("ticker")
    df["mean_ratio"] = grp["short_ratio"].transform(lambda s: s.rolling(window, min_periods=5).mean())
    df["std_ratio"] = grp["short_ratio"].transform(lambda s: s.rolling(window, min_periods=5).std())
    df["zscore"] = (df["short_ratio"] - df["mean_ratio"]) / df["std_ratio"].replace(0, pd.NA)
    spikes = df[df["zscore"].abs() >= float(z_threshold)].copy()
    if spikes.empty:
        return records_from_frame(pd.DataFrame(), source="finra_short_spike")
    rows: list[dict] = []
    for _, r in spikes.iterrows():
        direction = "elevated" if r["zscore"] > 0 else "depressed"
        headline = (
            f"Short-volume {direction} (z={r['zscore']:+.1f}σ): "
            f"{int(r['short_volume']):,} shorted of {int(r['total_volume']):,} "
            f"({r['short_ratio'] * 100:.1f}% vs {r['mean_ratio'] * 100:.1f}% baseline)"
        )
        ts = pd.Timestamp(r["date"]).tz_localize("UTC")
        rows.append(
            {
                "ticker": r["ticker"],
                "timestamp": ts,
                "headline": headline,
                "summary": f"short_ratio={r['short_ratio']:.4f} mean={r['mean_ratio']:.4f} std={r['std_ratio']:.4f} z={r['zscore']:.2f}",
                "url": "https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files",
                "source": "finra_short_spike",
                "source_id": f"{r['ticker']}-{ts.strftime('%Y%m%d')}",
            }
        )
    return records_from_frame(pd.DataFrame(rows), source="finra_short_spike")


# ---------------------------------------------------------------------------
# yfinance company profile backfill (replaces FMP profile)
# ---------------------------------------------------------------------------

YFINANCE_PROFILE_FIELDS: tuple[str, ...] = (
    "symbol",
    "shortName",
    "longName",
    "sector",
    "sectorDisp",
    "industry",
    "industryDisp",
    "country",
    "marketCap",
    "enterpriseValue",
    "sharesOutstanding",
    "floatShares",
    "exchange",
    "quoteType",
    "fullTimeEmployees",
    "longBusinessSummary",
    "website",
    "beta",
    "averageVolume",
    "averageVolume10days",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "profitMargins",
    "operatingMargins",
    "totalRevenue",
    "grossProfits",
    "earningsGrowth",
    "revenueGrowth",
    "dividendYield",
    "shortRatio",
    "heldPercentInstitutions",
    "heldPercentInsiders",
)


def fetch_yfinance_profiles(
    tickers: Iterable[str],
    *,
    min_interval_s: float = 0.4,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Pull yf.Ticker(t).info for each ticker and reduce to a fixed schema.

    Free, unlimited. ~0.5s per ticker — full 1077 takes ~9 minutes.
    """
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()

    cleaned = _clean_tickers(tickers)
    rows: list[dict] = []
    last_request = 0.0
    for i, ticker in enumerate(cleaned):
        elapsed = time.monotonic() - last_request
        if elapsed < min_interval_s:
            time.sleep(min_interval_s - elapsed)
        info: dict = {}
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}
        last_request = time.monotonic()
        if not info:
            continue
        row: dict = {"ticker": ticker, "snapshot_date": pd.Timestamp.utcnow().normalize()}
        for f in YFINANCE_PROFILE_FIELDS:
            val = info.get(f)
            if isinstance(val, (list, dict)):
                val = json.dumps(val)[:1000]
            # yfinance occasionally returns the string "Infinity" or "-Infinity"
            # in numeric fields (e.g. trailingPE when earnings are negative).
            # pyarrow can't mix str + float in one column, so coerce here.
            if isinstance(val, str) and val.strip().lower() in {"infinity", "-infinity", "nan"}:
                val = None
            row[f] = val
        rows.append(row)
        if (i + 1) % progress_every == 0:
            print(f"  yfinance profile: {i + 1}/{len(cleaned)} tickers ({len(rows)} successful)", flush=True)
    return pd.DataFrame(rows)
