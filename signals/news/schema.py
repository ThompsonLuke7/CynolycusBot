"""Typed records for unscheduled company news."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any
from datetime import datetime

import pandas as pd


TIMESTAMP_SEMANTICS_VERSION = "catalyst-time@1"


def normalize_ticker(value: Any) -> str:
    return str(value or "").upper().replace("$", "").strip()


def normalize_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def normalize_observation_timestamp(value: Any) -> pd.Timestamp | None:
    """Normalize an explicit collection/availability timestamp to UTC.

    A source event date may retain the historical timezone default through
    :func:`normalize_timestamp`.  Observation metadata is different: a naive
    collection time is ambiguous and must not be silently interpreted.
    """

    if value is None or value is pd.NaT or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    ts = pd.Timestamp(value)
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError("observation timestamps must be timezone-aware")
    return ts.tz_convert("UTC")


def _preserve_observation_timestamp(value: Any) -> pd.Timestamp | Any | None:
    """Normalize valid metadata but retain invalid explicit values for quarantine."""

    if value is None or value is pd.NaT or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return normalize_observation_timestamp(value)
    except (TypeError, ValueError):
        return value


def _first_present(row: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        value = row.get(name)
        if value is None or value is pd.NaT or value is pd.NA:
            continue
        try:
            if bool(pd.isna(value)):
                continue
        except (TypeError, ValueError):
            pass
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


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
    event_time: Any | None = None
    published_at: Any | None = None
    observed_at: Any | None = None
    available_at: Any | None = None
    source_record_id: str = ""
    source_artifact_hash: str | None = None
    timestamp_semantics_version: str = TIMESTAMP_SEMANTICS_VERSION
    hindsight_evidence: Mapping[str, Any] | None = None

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
        source_key = self.source_record_id or self.source_id or text_fingerprint(
            self.timestamp_utc.isoformat(), self.url, self.headline
        )
        revision = text_fingerprint(self.headline, self.summary, self.body, self.url)
        return text_fingerprint(self.clean_ticker, self.source, source_key, revision)

    def to_record(self) -> dict[str, Any]:
        timestamp = self.timestamp_utc
        event_time = normalize_timestamp(self.event_time) if self.event_time is not None else timestamp
        published_at = (
            _preserve_observation_timestamp(self.published_at)
            if self.published_at is not None
            else None
        )
        observed_at = _preserve_observation_timestamp(self.observed_at)
        available_at = _preserve_observation_timestamp(self.available_at)
        if available_at is None:
            available_at = observed_at
        # An absent provider ID remains absent.  ``record_id`` is a local
        # compatibility identity, not source evidence and must not turn into
        # a false logical source key for deduplication.
        source_record_id = str(self.source_record_id or self.source_id).strip()
        return {
            "record_id": self.record_id,
            "ticker": self.clean_ticker,
            # Compatibility column retained for every existing downstream
            # feature/label consumer.  New causal consumers use the explicit
            # fields below.
            "timestamp": timestamp,
            "event_time": event_time,
            "published_at": published_at,
            "observed_at": observed_at,
            "available_at": available_at,
            "source_record_id": source_record_id,
            "source_artifact_hash": self.source_artifact_hash,
            "timestamp_semantics_version": self.timestamp_semantics_version,
            "hindsight_evidence": dict(self.hindsight_evidence or {}) or None,
            "headline": str(self.headline or "").strip(),
            "summary": str(self.summary or "").strip(),
            "body": str(self.body or "").strip(),
            "url": str(self.url or "").strip(),
            "source": str(self.source or "unknown").strip(),
            "source_id": str(self.source_id or "").strip(),
            "text": self.text,
            "content_hash": text_fingerprint(self.clean_ticker, self.headline, self.summary or self.body),
        }


def add_observation_metadata(
    df: pd.DataFrame,
    *,
    observed_at: Any | None = None,
) -> pd.DataFrame:
    """Add causal observation columns without changing legacy timestamps.

    ``observed_at`` is one batch timestamp captured by the collector.  It is
    also the earliest known availability when a row has no more precise
    ``available_at`` value.  No event/published timestamp or filesystem time
    is ever used as an availability fallback.
    """

    out = df.copy()
    batch_observed = normalize_observation_timestamp(observed_at)
    if "event_time" not in out.columns:
        out["event_time"] = out.get("timestamp", pd.Series(pd.NaT, index=out.index))
    if "published_at" not in out.columns:
        out["published_at"] = pd.NaT
    if "observed_at" not in out.columns:
        out["observed_at"] = pd.NaT
    if "available_at" not in out.columns:
        out["available_at"] = pd.NaT
    # Keep timezone-aware values in an object column across pandas versions;
    # assigning an aware batch timestamp into a naive all-NaT datetime column
    # otherwise emits a dtype warning and can change behavior in pandas 3.
    out["observed_at"] = out["observed_at"].astype(object)
    out["available_at"] = out["available_at"].astype(object)
    if "source_record_id" not in out.columns:
        out["source_record_id"] = ""
    if "source_artifact_hash" not in out.columns:
        out["source_artifact_hash"] = None
    if "timestamp_semantics_version" not in out.columns:
        out["timestamp_semantics_version"] = TIMESTAMP_SEMANTICS_VERSION

    if batch_observed is not None:
        observed_missing = out["observed_at"].isna()
        out.loc[observed_missing, "observed_at"] = batch_observed
    observed_values = []
    available_values = []
    for observed, available in zip(out["observed_at"], out["available_at"]):
        observed_ts = _preserve_observation_timestamp(observed)
        available_ts = _preserve_observation_timestamp(available)
        if available_ts is None:
            available_ts = observed_ts
        observed_values.append(observed_ts)
        available_values.append(available_ts)
    out["observed_at"] = observed_values
    out["available_at"] = available_values

    source_ids = []
    for _, row in out.iterrows():
        existing = _first_present(row, ("source_record_id", "source_id", "id"))
        source_ids.append(str(existing or "").strip())
    out["source_record_id"] = source_ids
    return out


def parquet_safe_causal_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Serialize mixed valid/raw causal metadata without localizing naive values."""
    out = df.copy()
    for column in ("timestamp", "event_time", "published_at", "observed_at", "available_at"):
        if column not in out.columns:
            continue
        out[column] = [
            value.isoformat() if isinstance(value, (pd.Timestamp, datetime)) else value
            for value in out[column]
        ]
    if "raw_ingestion_fields" in out.columns:
        out["raw_ingestion_fields"] = [
            json.dumps(value, sort_keys=True, default=str) if isinstance(value, Mapping) else value
            for value in out["raw_ingestion_fields"]
        ]
    return out


def records_from_frame(
    df: pd.DataFrame,
    *,
    source: str = "unknown",
    observed_at: Any | None = None,
    collection_time: Any | None = None,
) -> pd.DataFrame:
    # Imported here, not at module scope. The catalyst adapter reaches
    # signals.events.forward_guidance.features.build_matrix for two constants,
    # and that module pulls in core.API.Alpaca_API.market_data -- ~0.85s of
    # import work this module otherwise charges to every importer. That cost is
    # paid per process, so signals.news.sources' spawn-based earnings fetch paid
    # it three times over and blew its own timeout budget.
    from signals.catalysts.nervous_system_adapter import hindsight_evidence_fields

    if df.empty:
        return empty_news_frame()
    rows = []
    for _, row in df.iterrows():
        ticker = _first_present(row, ("ticker", "symbol"))
        timestamp = _first_present(row, ("timestamp", "datetime", "published_at", "time_published"))
        published_at = _first_present(row, ("published_at", "time_published"))
        headline = _first_present(row, ("headline", "title"))
        timestamp_invalid = False
        if timestamp is not None:
            try:
                pd.Timestamp(timestamp)
            except (TypeError, ValueError):
                timestamp_invalid = True
        if not ticker or not timestamp or not headline or timestamp_invalid:
            raw = row.to_dict()
            rows.append({
                "record_id": str(raw.get("record_id") or f"news:row:{len(rows)}"),
                "ticker": str(ticker or "").upper().strip(),
                "timestamp": timestamp,
                "event_time": raw.get("event_time", timestamp),
                "published_at": raw.get("published_at"),
                "observed_at": raw.get("observed_at", observed_at or collection_time),
                "available_at": raw.get("available_at"),
                "source_record_id": str(raw.get("source_record_id") or "").strip(),
                "source": str(raw.get("source") or source),
                "headline": headline or "",
                "summary": raw.get("summary", ""), "body": raw.get("body", ""),
                "url": raw.get("url", ""),
                "ingestion_quarantine_code": (
                    "INVALID_EVENT_TIME" if timestamp_invalid else "MISSING_REQUIRED_NEWS_FIELD"
                ),
                "ingestion_quarantine_message": (
                    "explicit timestamp/occurrence is invalid"
                    if timestamp_invalid else "ticker, timestamp, and headline are required"
                ),
                "raw_ingestion_fields": raw,
            })
            continue
        row_observed = _first_present(row, ("observed_at", "collection_time", "captured_at"))
        row_available = _first_present(row, ("available_at",))
        rows.append(
            NewsRecord(
                ticker=ticker,
                timestamp=timestamp,
                headline=headline,
                summary=_first_present(row, ("summary", "description")) or "",
                body=_first_present(row, ("body", "content")) or "",
                url=_first_present(row, ("url",)) or "",
                source=_first_present(row, ("source",)) or source,
                source_id=_first_present(row, ("source_id", "id")) or "",
                event_time=_first_present(row, ("event_time",)) or timestamp,
                published_at=published_at,
                observed_at=row_observed if row_observed is not None else (observed_at or collection_time),
                available_at=row_available,
                source_record_id=_first_present(row, ("source_record_id", "source_id", "id")) or "",
                source_artifact_hash=_first_present(row, ("source_artifact_hash",)) ,
                timestamp_semantics_version=(
                    _first_present(row, ("timestamp_semantics_version",))
                    or TIMESTAMP_SEMANTICS_VERSION
                ),
                hindsight_evidence=hindsight_evidence_fields(row.to_dict()) or None,
            ).to_record()
        )
    return pd.DataFrame(rows) if rows else empty_news_frame()


def empty_news_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "record_id",
            "ticker",
            "timestamp",
            "event_time",
            "published_at",
            "observed_at",
            "available_at",
            "source_record_id",
            "source_artifact_hash",
            "timestamp_semantics_version",
            "hindsight_evidence",
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
