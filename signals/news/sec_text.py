"""Resumable SEC filing text backfill for catalyst records."""

from __future__ import annotations

import re
import time
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd

from signals.events.forward_guidance.data.sec_client import SecClient, SecFiling
from signals.news.earnings import enrich_earnings_catalyst_fields
from signals.news.relations import classify_news_relations
from signals.news.schema import text_fingerprint


# Section markers per form type. We look for the start marker, then read forward
# until we hit the END marker (or a hard char-cap). Multiple markers per section
# because phrasing varies across filers.
SEC_SECTION_MARKERS = {
    "10-K": [
        ("mdna", [r"item\s*7\.?\s*management'?s?\s*discussion"], [r"item\s*7a\.?\s*", r"item\s*8\.?\s*financial\s*statements"]),
        ("risk_factors", [r"item\s*1a\.?\s*risk\s*factors"], [r"item\s*1b\.?\s*", r"item\s*2\.?\s*propert"]),
        ("business", [r"item\s*1\.?\s*business"], [r"item\s*1a\.?\s*risk", r"item\s*2\.?\s*propert"]),
    ],
    "10-Q": [
        ("mdna", [r"item\s*2\.?\s*management'?s?\s*discussion"], [r"item\s*3\.?\s*quantitative", r"item\s*4\.?\s*controls"]),
        ("risk_factors", [r"item\s*1a\.?\s*risk\s*factors"], [r"item\s*2\.?\s*unregistered\s*sales", r"item\s*6\.?\s*exhibits"]),
    ],
}


def _strip_html(text: str) -> str:
    """Quick-and-dirty HTML/SGML scrub. Good enough for downstream embedding."""
    # Drop the EDGAR SGML wrapper if present
    text = re.sub(r"<sec-document>.*?</sec-header>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    # Strip script/style
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    # Strip tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common entities
    for ent, rep in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&#8217;", "'"), ("&#8220;", "\""), ("&#8221;", "\"")):
        text = text.replace(ent, rep)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_section(haystack: str, start_patterns: list[str], end_patterns: list[str], max_chars: int = 12000) -> str | None:
    """Find the earliest matching start_pattern; read forward until the first end_pattern (or max_chars)."""
    lower = haystack.lower()
    start_match = None
    for pat in start_patterns:
        m = re.search(pat, lower)
        if m and (start_match is None or m.start() < start_match.start()):
            start_match = m
    if start_match is None:
        return None
    start = start_match.start()
    end = start + max_chars
    for pat in end_patterns:
        m = re.search(pat, lower[start + 200:])
        if m:
            end = min(end, start + 200 + m.start())
    section = haystack[start:end]
    return section.strip()


def extract_sec_mda(text: str, form: str) -> dict:
    """Extract MD&A + Risk Factors + Business prose from a 10-K or 10-Q full-text dump.

    Returns a dict of {section: prose}. Sections that aren't found are absent.
    """
    if not text or form not in SEC_SECTION_MARKERS:
        return {}
    # First strip HTML wrappers — much of the EDGAR full-text dump is HTML
    stripped = _strip_html(text)
    sections = {}
    for name, starts, ends in SEC_SECTION_MARKERS[form]:
        sec = _extract_section(stripped, starts, ends)
        if sec:
            sections[name] = sec
    return sections


def fetch_and_extract_filing(url: str, form: str, *, timeout: int = 60, max_response_chars: int = 20_000_000) -> dict:
    """Download a filing URL and return its MD&A/Risk Factors/Business sections."""
    req = urllib.request.Request(url, headers={"User-Agent": "CynolycusBot research@example.com"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read(max_response_chars).decode("utf-8", errors="ignore")
    except Exception:
        return {}
    return extract_sec_mda(raw, form)


def backfill_sec_mda(
    news_path: Path | str = "signals/news/data/processed/news_records.parquet",
    *,
    output_path: Path | str | None = None,
    forms: Iterable[str] = ("10-K", "10-Q"),
    min_interval_s: float = 0.2,
    progress_every: int = 25,
    limit: int | None = None,
) -> pd.DataFrame:
    """Replace SEC-filing bodies (currently just EDGAR submission headers) with
    actual MD&A + Risk Factors + Business sections.

    Only processes records with the right source AND non-empty url AND body that
    looks like a filing-header dump (no MD&A-like keywords). Safe to re-run.
    """
    path = Path(news_path)
    out_path = Path(output_path) if output_path else path
    news = pd.read_parquet(path)
    if news.empty:
        return news

    wanted_forms = {f.upper() for f in forms}
    source_to_form = {"sec_10-k": "10-K", "sec_10-q": "10-Q"}

    def looks_like_mda(body: str) -> bool:
        if not body:
            return False
        body_lower = body[:5000].lower()
        return any(k in body_lower for k in ("management's discussion", "risk factors", "results of operations", "liquidity"))

    candidates = []
    for idx, row in news.iterrows():
        src = str(row.get("source") or "")
        form = source_to_form.get(src)
        if form is None or form not in wanted_forms:
            continue
        url = str(row.get("url") or "")
        if not url:
            continue
        body = str(row.get("body") or "")
        if looks_like_mda(body):
            continue
        candidates.append((idx, url, form))
    if limit is not None:
        candidates = candidates[: int(limit)]

    print(f"sec_mda candidates: {len(candidates):,}")
    last_request = 0.0
    updated = 0
    for pos, (idx, url, form) in enumerate(candidates, start=1):
        elapsed = time.monotonic() - last_request
        if elapsed < min_interval_s:
            time.sleep(min_interval_s - elapsed)
        sections = fetch_and_extract_filing(url, form)
        last_request = time.monotonic()
        if not sections:
            continue
        prose = " ".join(sections.values())[:30000]
        if len(prose) < 500:
            continue
        news.at[idx, "body"] = prose
        text_combined = " ".join(
            str(news.at[idx, f] or "") for f in ("headline", "summary", "body")
        ).strip()[:35000]
        news.at[idx, "text"] = text_combined
        news.at[idx, "content_hash"] = text_fingerprint(
            str(news.at[idx, "ticker"] or ""), news.at[idx, "headline"], prose[:500]
        )
        updated += 1
        if pos % int(progress_every) == 0:
            print(f"  sec_mda processed={pos}/{len(candidates)} updated={updated}", flush=True)
            # periodic checkpoint
            news.to_parquet(out_path, index=False)

    news.to_parquet(out_path, index=False)
    print(f"sec_mda done: updated {updated:,} of {len(candidates):,} candidates")
    return news


def _clean_forms(forms: Iterable[str]) -> set[str]:
    return {str(form).upper().strip() for form in forms if str(form).strip()}


def _source_to_form(source: object) -> str:
    return str(source or "").lower().replace("sec_", "").replace("_", " ").upper().replace("SC 13D A", "SC 13D/A")


def backfill_sec_full_text(
    news_path: Path | str,
    *,
    output_path: Path | str | None = None,
    forms: Iterable[str] = ("8-K", "10-Q", "10-K"),
    full_text_limit: int = 20000,
    limit: int | None = None,
    checkpoint_every: int = 25,
) -> pd.DataFrame:
    """Fill empty SEC bodies by resolving accession metadata and downloading text."""
    path = Path(news_path)
    out_path = Path(output_path) if output_path else path
    news = pd.read_parquet(path)
    if news.empty:
        news.to_parquet(out_path, index=False)
        return news

    wanted_forms = _clean_forms(forms)
    sec = SecClient()
    ticker_map = sec.company_tickers()
    cik_by_ticker = {
        str(row.ticker).upper(): str(row.cik_str)
        for row in ticker_map.itertuples(index=False)
        if str(getattr(row, "ticker", "")).strip() and str(getattr(row, "cik_str", "")).strip()
    }

    source_forms = news["source"].map(_source_to_form)
    body_missing = news.get("body", pd.Series("", index=news.index)).fillna("").astype(str).str.len().eq(0)
    is_sec = news["source"].astype(str).str.startswith("sec")
    mask = is_sec & body_missing & source_forms.isin(wanted_forms)
    candidate_idx = list(news.index[mask])
    if limit is not None:
        candidate_idx = candidate_idx[: int(limit)]

    updated = 0
    for pos, idx in enumerate(candidate_idx, start=1):
        row = news.loc[idx]
        ticker = str(row.get("ticker") or "").upper().replace("$", "").strip()
        cik = cik_by_ticker.get(ticker)
        accession = str(row.get("source_id") or "").strip()
        if not cik or not accession:
            continue
        try:
            form = _source_to_form(row.get("source"))
            filing = SecFiling(
                cik=cik,
                accession_number=accession,
                filing_date=str(pd.Timestamp(row.get("timestamp")).date()),
                form=form,
                primary_document=f"{accession}.txt",
                description=row.get("summary"),
            )
            body = sec.download_filing_text(filing)[: int(full_text_limit)]
        except Exception:
            continue
        if not body.strip():
            continue
        news.at[idx, "url"] = filing.document_url
        news.at[idx, "body"] = body
        text = " ".join(str(news.at[idx, field] or "") for field in ("headline", "summary", "body")).strip()
        news.at[idx, "text"] = text
        news.at[idx, "content_hash"] = text_fingerprint(ticker, news.at[idx, "headline"], news.at[idx, "summary"] or body)
        updated += 1
        if checkpoint_every and pos % int(checkpoint_every) == 0:
            news.to_parquet(out_path, index=False)
            print(f"sec_text_progress processed={pos}/{len(candidate_idx)} updated={updated}")

    news = enrich_earnings_catalyst_fields(classify_news_relations(news))
    news.to_parquet(out_path, index=False)
    print(f"sec_text_done candidates={len(candidate_idx)} updated={updated}")
    return news
