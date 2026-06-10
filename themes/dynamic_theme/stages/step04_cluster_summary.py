"""Step 4 — Create cluster summaries for Claude labeling.

For each non-noise cluster produce:
  {
    "cluster_id": int,
    "tickers": [...],
    "top_keywords": [...],
    "sample_headlines": [...]
  }

Keywords extracted via simple TF-IDF over cluster member news headlines.
"""
from __future__ import annotations

import logging
import re
import string
from collections import Counter
from typing import Any

import pandas as pd

from themes.dynamic_theme.config import (
    MAX_HEADLINES_PER_TICKER,
    NEWS_RECORDS_PATH,
    TICKER_DOCUMENTS_PATH,
)

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    "a an the and or but in on at to for of with is are was were be been being "
    "have has had do does did will would could should may might shall its it this "
    "that these those from as by into through about after before up down out over "
    "inc corp ltd co company shares stock market says said ceo quarter year "
    "billion million revenue earnings report results s p".split()
)

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _tokenize(text: str) -> list[str]:
    text = text.lower().translate(_PUNCT_TABLE)
    return [w for w in text.split() if len(w) > 2 and w not in _STOPWORDS and not w.isdigit()]


def _top_keywords(texts: list[str], top_n: int = 8) -> list[str]:
    counts: Counter = Counter()
    for text in texts:
        counts.update(_tokenize(text))
    return [w for w, _ in counts.most_common(top_n)]


def _sample_headlines(headlines: list[str], n: int = 3) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for h in headlines:
        h = h.strip()
        if h and h not in seen:
            seen.add(h)
            out.append(h)
        if len(out) >= n:
            break
    return out


def build_cluster_summaries(
    clusters_df: pd.DataFrame,
    docs_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Build a summary dict per cluster for use in the Claude labeling prompt."""
    if docs_df is None and TICKER_DOCUMENTS_PATH.exists():
        docs_df = pd.read_parquet(TICKER_DOCUMENTS_PATH)

    # build {ticker: headlines} from news_summary field of docs
    ticker_headlines: dict[str, list[str]] = {}
    if docs_df is not None and "recent_news_summary" in docs_df.columns:
        for _, row in docs_df.iterrows():
            t = str(row.get("ticker", "")).upper()
            raw = str(row.get("recent_news_summary") or "")
            headlines = [h.strip() for h in raw.split("|") if h.strip()]
            ticker_headlines[t] = headlines

    summaries: list[dict[str, Any]] = []
    for cid, grp in clusters_df[clusters_df["cluster_id"] >= 0].groupby("cluster_id"):
        tickers = grp["ticker"].astype(str).tolist()
        all_headlines: list[str] = []
        all_texts: list[str] = []
        for t in tickers:
            hl = ticker_headlines.get(t, [])
            all_headlines.extend(hl[:MAX_HEADLINES_PER_TICKER])
            all_texts.extend(hl)

        # also pull from docs description
        if docs_df is not None and "description" in docs_df.columns:
            ticker_set = set(tickers)
            desc_texts = docs_df[docs_df["ticker"].isin(ticker_set)]["description"].dropna().tolist()
            all_texts.extend(desc_texts)

        summaries.append(
            {
                "cluster_id": int(cid),
                "tickers": tickers,
                "top_keywords": _top_keywords(all_texts),
                "sample_headlines": _sample_headlines(all_headlines),
            }
        )

    logger.info("Built summaries for %d clusters", len(summaries))
    return summaries
