"""Background article-body scraper for Google News / yfinance records.

Most catalyst records arrive with only headline + summary (the RSS feed never
sends the full article body). This module fetches the article URL and uses
``trafilatura`` to extract clean prose into the ``body`` field, so embeddings
can see what's actually being said.

Designed to be safe for long-running background invocation:
- Resumable: skips records that already have a non-trivial body
- Rate-limited (configurable, default ~1 req/sec per host)
- Checkpoints to disk every N records
- Handles 403s / paywalls / timeouts without losing progress
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

from news.schema import text_fingerprint


def _extract_body(url: str, *, timeout: int = 25) -> Optional[str]:
    """Fetch URL and return cleaned article body, or None on failure.

    Uses trafilatura's built-in fetcher which follows redirects (essential for
    Google News obfuscated URLs that redirect to the real article).
    """
    import trafilatura
    from trafilatura.settings import use_config

    config = use_config()
    config.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(timeout))
    config.set("DEFAULT", "MIN_OUTPUT_SIZE", "200")

    try:
        raw = trafilatura.fetch_url(url, config=config)
    except Exception:
        raw = None

    if not raw:
        return None

    body = trafilatura.extract(
        raw,
        include_comments=False,
        include_tables=False,
        favor_recall=True,
        no_fallback=False,
        config=config,
    )
    if not body or len(body) < 200:
        return None
    return body[:15000]


def backfill_google_news_bodies(
    news_path: Path | str = "news/data/processed/news_records.parquet",
    *,
    output_path: Path | str | None = None,
    sources: tuple[str, ...] = ("yfinance",),  # google_news skipped — URLs are unscrapable redirects
    min_interval_s: float = 1.0,
    checkpoint_every: int = 100,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Scrape article bodies for records that currently have no body.

    Note: Google News URLs are obfuscated Protobuf redirects that Google
    actively blocks. We skip them by default and focus on yfinance URLs
    (which resolve directly to finance.yahoo.com / fool.com / marketbeat
    / seekingalpha and scrape cleanly). Rate ~1 req/sec, ~6K yfinance records
    means ~1.5h wall time.
    """
    path = Path(news_path)
    out_path = Path(output_path) if output_path else path
    news = pd.read_parquet(path)
    if news.empty:
        return news

    body_len = news.get("body", pd.Series("", index=news.index)).fillna("").astype(str).str.len()
    has_url = news.get("url", pd.Series("", index=news.index)).fillna("").astype(str).str.len().gt(10)
    not_google_redirect = ~news.get("url", pd.Series("", index=news.index)).fillna("").str.contains("news.google.com/rss/articles", regex=False)
    source_match = news["source"].astype(str).isin(set(sources))
    candidate_mask = source_match & has_url & not_google_redirect & body_len.eq(0)
    candidates = list(news.index[candidate_mask])

    if limit is not None:
        candidates = candidates[: int(limit)]
    print(f"scrape candidates: {len(candidates):,} (rate ~{1/min_interval_s:.1f} req/sec)")

    last_request = 0.0
    updated = 0
    failed = 0
    start_time = time.time()
    for pos, idx in enumerate(candidates, start=1):
        elapsed = time.monotonic() - last_request
        if elapsed < min_interval_s:
            time.sleep(min_interval_s - elapsed)
        url = str(news.at[idx, "url"] or "")
        body = _extract_body(url)
        last_request = time.monotonic()
        if not body:
            failed += 1
            continue
        news.at[idx, "body"] = body
        text_combined = " ".join(
            str(news.at[idx, f] or "")
            for f in ("headline", "summary", "body")
        ).strip()[:18000]
        news.at[idx, "text"] = text_combined
        news.at[idx, "content_hash"] = text_fingerprint(
            str(news.at[idx, "ticker"] or ""),
            str(news.at[idx, "headline"] or ""),
            body[:500],
        )
        updated += 1
        if pos % int(checkpoint_every) == 0:
            wall = time.time() - start_time
            rate = pos / wall if wall > 0 else 0
            eta = (len(candidates) - pos) / rate if rate > 0 else 0
            news.to_parquet(out_path, index=False)
            print(
                f"  scrape progress {pos}/{len(candidates)} "
                f"updated={updated} failed={failed} "
                f"rate={rate:.2f}/sec eta={eta/60:.0f}min",
                flush=True,
            )

    news.to_parquet(out_path, index=False)
    print(f"scrape done: updated={updated} failed={failed} of {len(candidates):,} candidates")
    return news
