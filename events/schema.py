"""Typed records for scheduled known-date events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from events.config import ALLOWED_MACRO_EVENT_TYPES, DEFAULT_TIMEZONE, DISALLOWED_EVENT_TYPES


def normalize_event_type(value: Any) -> str:
    raw = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "fomc": "fomc_decision",
        "fomc_statement": "fomc_decision",
        "powell": "fed_speech",
        "powell_speech": "fed_speech",
        "nonfarm_payrolls": "nfp",
        "non_farm_payrolls": "nfp",
        "employment_report": "jobs",
        "options_expiration": "opex",
        "opex_week": "opex",
    }
    return aliases.get(raw, raw)


def normalize_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(DEFAULT_TIMEZONE)
    return ts.tz_convert("UTC")


@dataclass(frozen=True)
class ScheduledEvent:
    event_type: str
    timestamp: str
    title: str = ""
    source: str = "manual"
    ticker: str | None = None
    url: str | None = None

    @property
    def normalized_type(self) -> str:
        return normalize_event_type(self.event_type)

    @property
    def timestamp_utc(self) -> pd.Timestamp:
        return normalize_timestamp(self.timestamp)

    def to_record(self) -> dict[str, Any]:
        event_type = self.normalized_type
        if event_type in DISALLOWED_EVENT_TYPES:
            raise ValueError("Treasury auctions are intentionally excluded.")
        if event_type not in ALLOWED_MACRO_EVENT_TYPES:
            raise ValueError(f"Unsupported scheduled event type: {self.event_type}")
        ticker = str(self.ticker).upper().replace("$", "").strip() if self.ticker else None
        return {
            "event_type": event_type,
            "timestamp": self.timestamp_utc,
            "title": self.title,
            "source": self.source,
            "ticker": ticker or None,
            "url": self.url,
        }


def events_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["event_type", "timestamp", "title", "source", "ticker", "url"])
    rows = []
    for _, row in df.iterrows():
        try:
            rows.append(
                ScheduledEvent(
                    event_type=row.get("event_type") or row.get("type"),
                    timestamp=row.get("timestamp") or row.get("date"),
                    title=str(row.get("title") or ""),
                    source=str(row.get("source") or "manual"),
                    ticker=row.get("ticker"),
                    url=row.get("url"),
                ).to_record()
            )
        except ValueError:
            continue
    out = pd.DataFrame(rows)
    return out.sort_values(["timestamp", "event_type", "ticker"], na_position="last").reset_index(drop=True)

