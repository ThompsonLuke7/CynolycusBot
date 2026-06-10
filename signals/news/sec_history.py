"""Resumable SEC EDGAR history collection."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from signals.news.dedup import deduplicate_news
from signals.news.relations import classify_news_relations
from signals.news.schema import empty_news_frame
from signals.news.sources import fetch_sec_8k_news
from signals.news.sources import fetch_sec_alpha_filings


def _clean_tickers(tickers: Iterable[str]) -> list[str]:
    return [str(t).upper().replace("$", "").strip() for t in tickers if str(t).strip()]


def collect_sec_history_resumable(
    tickers: Iterable[str],
    *,
    start: str,
    end: str,
    output_path: Path | str,
    batch_size: int = 50,
    full_text_limit: int = 0,
    alpha_forms: bool = False,
) -> pd.DataFrame:
    """Collect SEC 8-K history in batches and write after every batch."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ticker_list = _clean_tickers(tickers)
    existing = pd.read_parquet(output) if output.exists() else empty_news_frame()
    done = set(existing["ticker"].astype(str).str.upper()) if not existing.empty and "ticker" in existing.columns else set()
    remaining = [ticker for ticker in ticker_list if ticker not in done]
    combined = existing.copy()
    total = len(ticker_list)
    for offset in range(0, len(remaining), int(batch_size)):
        batch = remaining[offset : offset + int(batch_size)]
        if not batch:
            continue
        fetcher = fetch_sec_alpha_filings if alpha_forms else fetch_sec_8k_news
        frame = fetcher(batch, start=start, end=end, include_archives=True, full_text_limit=full_text_limit)
        combined = pd.concat([combined, frame], ignore_index=True) if not combined.empty else frame
        combined = classify_news_relations(deduplicate_news(combined))
        combined.to_parquet(output, index=False)
        completed = len(done) + min(offset + len(batch), len(remaining))
        print(f"sec_history_progress tickers={completed}/{total} rows={len(combined)}", flush=True)
    return combined
