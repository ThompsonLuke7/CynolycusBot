"""Resumable SEC filing text backfill for catalyst records."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from events.forward_guidance.data.sec_client import SecClient, SecFiling
from news.earnings import enrich_earnings_catalyst_fields
from news.relations import classify_news_relations
from news.schema import text_fingerprint


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
