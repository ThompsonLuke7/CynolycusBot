"""Step 1 — Build ticker documents.

For each ticker assemble:
  description          : Yahoo Finance business summary
  recent_news_summary  : last N headlines concatenated
  earnings_summary     : forward guidance / 8-K text (if available)
  catalyst_summary     : SEC catalyst text (if available)

Output: ticker_documents.parquet
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from dynamic_theme.config import (
    EVENTS_FG_PATH,
    MAX_HEADLINES_PER_TICKER,
    NEWS_LOOKBACK_DAYS,
    NEWS_RECORDS_PATH,
    TICKER_DOCUMENTS_PATH,
    TICKER_PROFILES_CACHE_DIR,
    TICKER_PROFILES_PATH,
    ensure_outputs,
)

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_descriptions() -> dict[str, str]:
    """Return {ticker: description} preferring the parquet profile, then JSON cache."""
    descs: dict[str, str] = {}

    # 1) parquet ticker profiles (news module)
    if TICKER_PROFILES_PATH.exists():
        try:
            profiles = pd.read_parquet(TICKER_PROFILES_PATH)
            for col in ["business_summary", "longBusinessSummary", "description", "summary"]:
                if col in profiles.columns and "ticker" in profiles.columns:
                    for _, row in profiles[["ticker", col]].dropna().iterrows():
                        descs[str(row["ticker"]).upper()] = str(row[col])[:600]
                    break
        except Exception as exc:
            logger.warning("Could not read ticker_profiles.parquet: %s", exc)

    # 2) per-ticker JSON cache (legacy theme_expansion)
    if TICKER_PROFILES_CACHE_DIR.exists():
        for fp in TICKER_PROFILES_CACHE_DIR.glob("*.json"):
            ticker = fp.stem.upper()
            if ticker in descs:
                continue
            try:
                data = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
                summary = (
                    data.get("longBusinessSummary")
                    or data.get("business_summary")
                    or data.get("summary")
                    or ""
                )
                if summary:
                    descs[ticker] = str(summary)[:600]
            except Exception:
                pass

    return descs


def _load_news_summaries(tickers: list[str], *, as_of: pd.Timestamp) -> dict[str, str]:
    """Return {ticker: recent_news_summary} — last N headlines in lookback window."""
    summaries: dict[str, str] = {t: "" for t in tickers}
    if not NEWS_RECORDS_PATH.exists():
        logger.warning("news_records.parquet not found at %s", NEWS_RECORDS_PATH)
        return summaries

    try:
        news = pd.read_parquet(NEWS_RECORDS_PATH)
    except Exception as exc:
        logger.warning("Could not read news_records.parquet: %s", exc)
        return summaries

    # normalise ticker column name
    ticker_col = next((c for c in news.columns if c.lower() in ("ticker", "symbol")), None)
    headline_col = next((c for c in news.columns if c.lower() in ("headline", "title", "text", "headline_text")), None)
    time_col = next((c for c in news.columns if c.lower() in ("published_at", "timestamp", "date", "published_utc")), None)

    if ticker_col is None or headline_col is None:
        logger.warning("news_records.parquet missing ticker or headline column")
        return summaries

    news[ticker_col] = news[ticker_col].astype(str).str.upper()
    if time_col:
        news[time_col] = pd.to_datetime(news[time_col], utc=True, errors="coerce")
        cutoff = as_of - pd.Timedelta(days=NEWS_LOOKBACK_DAYS)
        news = news[news[time_col] >= cutoff]

    for ticker, grp in news[news[ticker_col].isin(tickers)].groupby(ticker_col):
        headlines = grp[headline_col].dropna().astype(str).tolist()
        if time_col:
            grp = grp.sort_values(time_col, ascending=False)
            headlines = grp[headline_col].dropna().astype(str).tolist()
        summaries[str(ticker)] = " | ".join(headlines[:MAX_HEADLINES_PER_TICKER])

    return summaries


def _load_events_summaries(tickers: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Return ({ticker: earnings_summary}, {ticker: catalyst_summary}) from events module."""
    earnings: dict[str, str] = {t: "" for t in tickers}
    catalyst: dict[str, str] = {t: "" for t in tickers}

    # look for a processed feature matrix from the forward_guidance module
    candidates = list(EVENTS_FG_PATH.glob("*.parquet")) if EVENTS_FG_PATH.exists() else []
    if not candidates:
        return earnings, catalyst

    for fp in candidates:
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        ticker_col = next((c for c in df.columns if c.lower() in ("ticker", "symbol")), None)
        if ticker_col is None:
            continue
        df[ticker_col] = df[ticker_col].astype(str).str.upper()
        df = df[df[ticker_col].isin(tickers)]

        text_cols = [c for c in df.columns if any(k in c.lower() for k in ("text", "summary", "headline", "guidance", "catalyst"))]
        earnings_cols = [c for c in text_cols if any(k in c.lower() for k in ("earnings", "guidance", "beat", "miss"))]
        catalyst_cols = [c for c in text_cols if any(k in c.lower() for k in ("catalyst", "event", "8k", "form_4"))]

        for _, row in df.iterrows():
            t = str(row[ticker_col])
            if earnings_cols and not earnings.get(t):
                earnings[t] = " ".join(str(row[c]) for c in earnings_cols if pd.notna(row.get(c)))[:400]
            if catalyst_cols and not catalyst.get(t):
                catalyst[t] = " ".join(str(row[c]) for c in catalyst_cols if pd.notna(row.get(c)))[:400]

    return earnings, catalyst


# ── main ─────────────────────────────────────────────────────────────────────

def build_ticker_documents(
    tickers: list[str],
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build the ticker document frame and write ticker_documents.parquet."""
    ensure_outputs()
    as_of = as_of or pd.Timestamp.now(tz="UTC")
    tickers = [str(t).upper() for t in tickers]

    logger.info("Building ticker documents for %d tickers as of %s", len(tickers), as_of.date())

    descs = _load_descriptions()
    news_summaries = _load_news_summaries(tickers, as_of=as_of)
    earnings_summaries, catalyst_summaries = _load_events_summaries(tickers)

    rows = []
    for ticker in tickers:
        rows.append(
            {
                "ticker": ticker,
                "description": descs.get(ticker, ""),
                "recent_news_summary": news_summaries.get(ticker, ""),
                "earnings_summary": earnings_summaries.get(ticker, ""),
                "catalyst_summary": catalyst_summaries.get(ticker, ""),
                "date": as_of.normalize().tz_localize(None),
            }
        )

    df = pd.DataFrame(rows)
    df.to_parquet(TICKER_DOCUMENTS_PATH, index=False)
    logger.info("Wrote %s  rows=%d", TICKER_DOCUMENTS_PATH, len(df))
    return df


def load_ticker_documents() -> pd.DataFrame:
    return pd.read_parquet(TICKER_DOCUMENTS_PATH)
