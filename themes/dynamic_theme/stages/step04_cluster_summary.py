"""Step 4 — Create cluster summaries for Claude labeling.

For each non-noise cluster produce:
  {
    "cluster_id": int,
    "tickers": [...],
    "members": [{"ticker", "name", "industry", "about"}, ...],  # largest first; what the labeler reads
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
    CLAUDE_LABEL_ABOUT_CHARS,
    CLAUDE_LABEL_MAX_MEMBERS,
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


def _first_sentence(text: object, limit: int = CLAUDE_LABEL_ABOUT_CHARS) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean or clean.lower() in {"nan", "none"}:
        return ""
    end = clean.find(". ")
    sentence = clean if end == -1 else clean[: end + 1]
    return sentence if len(sentence) <= limit else sentence[: limit - 1].rstrip() + "…"


def member_profiles(
    tickers: list[str],
    profiles_df: pd.DataFrame | None,
    docs_df: pd.DataFrame | None = None,
    *,
    limit: int = CLAUDE_LABEL_MAX_MEMBERS,
) -> list[dict[str, str]]:
    """Largest members first: ticker, company name, industry, one-line business.

    Ordered by market cap (unknown last, then by ticker) so the companies that
    define a theme in practice are the ones the labeler reads. The business line
    comes from the profile summary, else the ticker document's description.
    """
    wanted = [str(t) for t in tickers]
    prof = pd.DataFrame({"ticker": wanted})
    if profiles_df is not None and not profiles_df.empty:
        cols = [c for c in ("ticker", "shortName", "longName", "industry", "marketCap", "longBusinessSummary")
                if c in profiles_df.columns]
        prof = prof.merge(profiles_df[cols].drop_duplicates("ticker"), on="ticker", how="left")
    if docs_df is not None and "description" in docs_df.columns:
        prof = prof.merge(docs_df[["ticker", "description"]].drop_duplicates("ticker"), on="ticker", how="left")
    for col in ("shortName", "longName", "industry", "marketCap", "longBusinessSummary", "description"):
        if col not in prof.columns:
            prof[col] = None
    prof["_cap"] = pd.to_numeric(prof["marketCap"], errors="coerce")
    prof = prof.sort_values(["_cap", "ticker"], ascending=[False, True], na_position="last")
    out = []
    for row in prof.head(limit).itertuples(index=False):
        about = _first_sentence(row.longBusinessSummary) or _first_sentence(row.description)
        name = next((str(v) for v in (row.shortName, row.longName) if isinstance(v, str) and v.strip()), "")
        entry = {"ticker": row.ticker, "name": name.strip(), "industry": str(row.industry or "").strip(),
                 "about": about}
        out.append({k: v for k, v in entry.items() if v and v.lower() != "nan"})
    return out


def build_cluster_summaries(
    clusters_df: pd.DataFrame,
    docs_df: pd.DataFrame | None = None,
    profiles_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Build a summary dict per cluster for use in the Claude labeling prompt."""
    if docs_df is None and TICKER_DOCUMENTS_PATH.exists():
        docs_df = pd.read_parquet(TICKER_DOCUMENTS_PATH)
    if profiles_df is None:
        from themes.dynamic_theme.profiles import load_profiles

        profiles_df = load_profiles()

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
                "members": member_profiles(tickers, profiles_df, docs_df),
                "top_keywords": _top_keywords(all_texts),
                "sample_headlines": _sample_headlines(all_headlines),
            }
        )

    logger.info("Built summaries for %d clusters", len(summaries))
    return summaries
