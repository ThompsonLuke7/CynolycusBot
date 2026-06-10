"""Typed records for unscheduled company news."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pandas as pd


def normalize_ticker(value: Any) -> str:
    return str(value or "").upper().replace("$", "").strip()


def normalize_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def text_fingerprint(*parts: str) -> str:
    joined = " ".join(str(p or "").strip().lower() for p in parts)
    return hashlib.sha256(joined.encode("utf-8", errors="ignore")).hexdigest()[:20]


@dataclass(frozen=True)
class NewsRecord:
    ticker: str
    timestamp: str
    headline: str
    summary: str = ""
    body: str = ""
    url: str = ""
    source: str = "unknown"
    source_id: str = ""

    @property
    def clean_ticker(self) -> str:
        return normalize_ticker(self.ticker)

    @property
    def timestamp_utc(self) -> pd.Timestamp:
        return normalize_timestamp(self.timestamp)

    @property
    def text(self) -> str:
        return " ".join(x for x in (self.headline, self.summary, self.body) if x).strip()

    @property
    def record_id(self) -> str:
        source_key = self.source_id or text_fingerprint(self.timestamp_utc.isoformat(), self.url, self.headline)
        return text_fingerprint(self.clean_ticker, source_key)

    def to_record(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "ticker": self.clean_ticker,
            "timestamp": self.timestamp_utc,
            "headline": str(self.headline or "").strip(),
            "summary": str(self.summary or "").strip(),
            "body": str(self.body or "").strip(),
            "url": str(self.url or "").strip(),
            "source": str(self.source or "unknown").strip(),
            "source_id": str(self.source_id or "").strip(),
            "text": self.text,
            "content_hash": text_fingerprint(self.clean_ticker, self.headline, self.summary or self.body),
        }


def records_from_frame(df: pd.DataFrame, *, source: str = "unknown") -> pd.DataFrame:
    if df.empty:
        return empty_news_frame()
    rows = []
    for _, row in df.iterrows():
        ticker = row.get("ticker") or row.get("symbol")
        timestamp = row.get("timestamp") or row.get("datetime") or row.get("published_at") or row.get("time_published")
        headline = row.get("headline") or row.get("title")
        if not ticker or not timestamp or not headline:
            continue
        rows.append(
            NewsRecord(
                ticker=ticker,
                timestamp=timestamp,
                headline=headline,
                summary=row.get("summary") or row.get("description") or "",
                body=row.get("body") or row.get("content") or "",
                url=row.get("url") or "",
                source=row.get("source") or source,
                source_id=row.get("source_id") or row.get("id") or "",
            ).to_record()
        )
    return pd.DataFrame(rows) if rows else empty_news_frame()


def empty_news_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "record_id",
            "ticker",
            "timestamp",
            "headline",
            "summary",
            "body",
            "url",
            "source",
            "source_id",
            "text",
            "content_hash",
        ]
    )
