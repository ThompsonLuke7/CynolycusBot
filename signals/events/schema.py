"""Typed records for scheduled known-date events."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import pandas as pd

from signals.events.config import ALLOWED_MACRO_EVENT_TYPES, DEFAULT_TIMEZONE, DISALLOWED_EVENT_TYPES
from signals.catalysts.nervous_system_adapter import hindsight_evidence_fields


TIMESTAMP_SEMANTICS_VERSION = "catalyst-time@1"


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


def normalize_observation_timestamp(value: Any) -> pd.Timestamp | None:
    """Normalize explicit observation metadata and reject naive timestamps."""

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
    """Retain invalid explicit metadata so the catalyst adapter can quarantine it."""

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


@dataclass(frozen=True)
class ScheduledEvent:
    event_type: str
    timestamp: str
    title: str = ""
    source: str = "manual"
    ticker: str | None = None
    url: str | None = None
    event_time: Any | None = None
    published_at: Any | None = None
    observed_at: Any | None = None
    available_at: Any | None = None
    source_record_id: str = ""
    source_artifact_hash: str | None = None
    timestamp_semantics_version: str = TIMESTAMP_SEMANTICS_VERSION
    hindsight_evidence: Mapping[str, Any] | None = None

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
        if event_type not in ALLOWED_MACRO_EVENT_TYPES and event_type != "earnings":
            raise ValueError(f"Unsupported scheduled event type: {self.event_type}")
        ticker = str(self.ticker).upper().replace("$", "").strip() if self.ticker else None
        event_time = normalize_timestamp(self.event_time if self.event_time is not None else self.timestamp)
        published_at = (
            _preserve_observation_timestamp(self.published_at)
            if self.published_at is not None
            else None
        )
        observed_at = _preserve_observation_timestamp(self.observed_at)
        available_at = _preserve_observation_timestamp(self.available_at)
        if available_at is None:
            available_at = observed_at
        source_record_id = str(self.source_record_id or "").strip()
        if not source_record_id:
            material = "|".join(
                (
                    str(self.source or "manual").strip(),
                    ticker or "",
                    event_type,
                    event_time.isoformat(),
                    str(self.title or "").strip(),
                    str(self.url or "").strip(),
                )
            )
            source_record_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
        return {
            "event_type": event_type,
            # ``timestamp`` is retained as the compatibility occurrence
            # column.  Causal consumers use the explicit fields below.
            "timestamp": event_time,
            "event_time": event_time,
            "published_at": published_at,
            "observed_at": observed_at,
            "available_at": available_at,
            "source_record_id": source_record_id,
            "source_artifact_hash": self.source_artifact_hash,
            "timestamp_semantics_version": self.timestamp_semantics_version,
            "hindsight_evidence": dict(self.hindsight_evidence or {}) or None,
            "title": self.title,
            "source": self.source,
            "ticker": ticker or None,
            "url": self.url,
        }


def events_from_frame(
    df: pd.DataFrame,
    *,
    observed_at: Any | None = None,
    collection_time: Any | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "event_type",
                "timestamp",
                "event_time",
                "published_at",
                "observed_at",
                "available_at",
                "source_record_id",
                "source_artifact_hash",
                "timestamp_semantics_version",
                "title",
                "source",
                "ticker",
                "url",
            ]
        )
    rows = []
    for _, row in df.iterrows():
        timestamp_value = _first_present(row, ("timestamp", "date", "event_time"))
        if timestamp_value is None or not _first_present(row, ("event_type", "type")):
            raw = row.to_dict()
            rows.append({
                "event_type": str(_first_present(row, ("event_type", "type")) or ""),
                "timestamp": timestamp_value,
                "event_time": raw.get("event_time", timestamp_value),
                "observed_at": raw.get("observed_at", observed_at or collection_time),
                "available_at": raw.get("available_at"),
                "title": str(_first_present(row, ("title", "headline")) or ""),
                "source": str(_first_present(row, ("source",)) or "manual"),
                "ticker": _first_present(row, ("ticker", "symbol")),
                "source_record_id": str(_first_present(row, ("source_record_id", "source_id", "id")) or ""),
                "ingestion_quarantine_code": "MISSING_REQUIRED_EVENT_FIELD",
                "ingestion_quarantine_message": "event_type and timestamp are required",
                "raw_ingestion_fields": raw,
            })
            continue
        try:
            rows.append(
                ScheduledEvent(
                    event_type=_first_present(row, ("event_type", "type")) or "",
                    timestamp=timestamp_value,
                    title=str(_first_present(row, ("title", "headline")) or ""),
                    source=str(_first_present(row, ("source",)) or "manual"),
                    ticker=_first_present(row, ("ticker", "symbol")),
                    url=_first_present(row, ("url",)),
                    event_time=_first_present(row, ("event_time",)) or timestamp_value,
                    published_at=_first_present(row, ("published_at",)),
                    observed_at=_first_present(row, ("observed_at", "collection_time", "captured_at"))
                    or observed_at
                    or collection_time,
                    available_at=_first_present(row, ("available_at",)),
                    source_record_id=_first_present(row, ("source_record_id", "source_id", "id")) or "",
                    source_artifact_hash=_first_present(row, ("source_artifact_hash",)),
                    timestamp_semantics_version=(
                        _first_present(row, ("timestamp_semantics_version",))
                        or TIMESTAMP_SEMANTICS_VERSION
                    ),
                    hindsight_evidence=hindsight_evidence_fields(row.to_dict()) or None,
                ).to_record()
            )
        except (TypeError, ValueError):
            # Keep a row whose occurrence metadata is malformed so the
            # causal adapter can return a deterministic quarantine result.
            # Unsupported/disallowed event kinds remain excluded by the
            # scheduled-event producer contract.
            normalized_type = normalize_event_type(_first_present(row, ("event_type", "type")) or "")
            if normalized_type in DISALLOWED_EVENT_TYPES:
                continue
            if normalized_type not in ALLOWED_MACRO_EVENT_TYPES and normalized_type != "earnings":
                continue
            rows.append(
                {
                    "event_type": normalized_type,
                    "timestamp": timestamp_value,
                    "event_time": _first_present(row, ("event_time",)) or timestamp_value,
                    "published_at": _first_present(row, ("published_at",)),
                    "observed_at": _first_present(row, ("observed_at", "collection_time", "captured_at")),
                    "available_at": _first_present(row, ("available_at",)),
                    "source_record_id": _first_present(row, ("source_record_id", "source_id", "id")) or "",
                    "source_artifact_hash": _first_present(row, ("source_artifact_hash",)),
                    "hindsight_evidence": hindsight_evidence_fields(row.to_dict()) or None,
                    "timestamp_semantics_version": (
                        _first_present(row, ("timestamp_semantics_version",))
                        or TIMESTAMP_SEMANTICS_VERSION
                    ),
                    "title": str(_first_present(row, ("title", "headline")) or ""),
                    "source": str(_first_present(row, ("source",)) or "manual"),
                    "ticker": _first_present(row, ("ticker", "symbol")),
                    "url": _first_present(row, ("url",)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return events_from_frame(pd.DataFrame())
    out["_sort_timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    return out.sort_values(
        ["_sort_timestamp", "event_type", "ticker"],
        na_position="last",
    ).drop(columns=["_sort_timestamp"]).reset_index(drop=True)


def parquet_safe_causal_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ("timestamp", "event_time", "published_at", "observed_at", "available_at"):
        if column in out.columns:
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
