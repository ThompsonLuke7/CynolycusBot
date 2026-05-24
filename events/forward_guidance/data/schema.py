"""Typed event records used by the earnings-guidance pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from events.forward_guidance.utils.dates import (
    event_available_at,
    event_id,
    normalize_report_time,
    reaction_session,
    session_close_timestamp,
)


def _metadata_from_record(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _nullable_str(value: Any) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return None
    if raw.endswith(".0") and raw[:-2].isdigit():
        raw = raw[:-2]
    return raw


@dataclass(frozen=True)
class EarningsEvent:
    ticker: str
    earnings_date: str
    report_time: str = "UNKNOWN"
    fiscal_period: str | None = None
    sector: str | None = None
    sector_etf: str | None = None
    cik: str | None = None
    source_url: str | None = None
    source_type: str = "free_web_sec"
    available_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def clean_ticker(self) -> str:
        return self.ticker.upper().replace("$", "")

    @property
    def normalized_report_time(self) -> str:
        return normalize_report_time(self.report_time)

    @property
    def event_id(self) -> str:
        return event_id(self.clean_ticker, self.earnings_date)

    @property
    def reaction_date(self) -> pd.Timestamp:
        return reaction_session(self.earnings_date, self.report_time)

    @property
    def signal_timestamp(self) -> pd.Timestamp:
        return session_close_timestamp(self.reaction_date)

    @property
    def available_timestamp(self) -> pd.Timestamp:
        if self.available_at:
            ts = pd.Timestamp(self.available_at)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            return ts.tz_convert("UTC")
        return event_available_at(self.earnings_date, self.report_time)

    def to_record(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "ticker": self.clean_ticker,
            "earnings_date": str(pd.Timestamp(self.earnings_date).date()),
            "report_time": self.normalized_report_time,
            "reaction_date": str(self.reaction_date.date()),
            "signal_timestamp": self.signal_timestamp.isoformat(),
            "fiscal_period": self.fiscal_period,
            "sector": self.sector,
            "sector_etf": self.sector_etf,
            "cik": self.cik,
            "source_url": self.source_url,
            "source_type": self.source_type,
            "available_at": self.available_timestamp.isoformat(),
            "metadata": json.dumps(self.metadata, sort_keys=True),
        }


def event_from_record(record: dict[str, Any] | pd.Series) -> EarningsEvent:
    if isinstance(record, pd.Series):
        record = record.to_dict()
    return EarningsEvent(
        ticker=_nullable_str(record.get("ticker")) or _nullable_str(record.get("symbol")) or "",
        earnings_date=_nullable_str(record.get("earnings_date")) or _nullable_str(record.get("date")) or "",
        report_time=str(record.get("report_time") or record.get("time") or "UNKNOWN"),
        fiscal_period=_nullable_str(record.get("fiscal_period")),
        sector=_nullable_str(record.get("sector")),
        sector_etf=_nullable_str(record.get("sector_etf")),
        cik=_nullable_str(record.get("cik")),
        source_url=_nullable_str(record.get("source_url")),
        source_type=_nullable_str(record.get("source_type")) or "free_web_sec",
        available_at=_nullable_str(record.get("available_at")),
        metadata=_metadata_from_record(record.get("metadata")),
    )


def events_to_frame(events: list[EarningsEvent]) -> pd.DataFrame:
    return pd.DataFrame([e.to_record() for e in events])


def raw_event_dir(root: Path, event: EarningsEvent) -> Path:
    return root / event.clean_ticker / str(pd.Timestamp(event.earnings_date).date())
