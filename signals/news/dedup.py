"""News deduplication helpers."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from collections.abc import Mapping

import pandas as pd


def _norm_url(value: str) -> str:
    return re.sub(r"[?#].*$", "", str(value or "").strip().lower())


def _canonical(value: object) -> object:
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        timestamp = pd.Timestamp(value)
        return timestamp.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    return value


def _canonical_source_field_hash(row: pd.Series) -> str:
    fields = {
        column: row.get(column)
        for column in (
            "source",
            "ticker",
            "timestamp",
            "event_time",
            "published_at",
            "headline",
            "summary",
            "body",
            "url",
        )
    }
    fields["url"] = _norm_url(fields["url"])
    encoded = json.dumps(_canonical(fields), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _nonempty_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def deduplicate_news(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "source" not in out.columns:
        out["source"] = "unknown"
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if "ticker" not in out.columns:
        out["ticker"] = ""
    if "source_record_id" not in out.columns:
        out["source_record_id"] = ""
    source_col = out["source"].fillna("unknown").astype(str).str.strip()
    has_source_record_id = _nonempty_series(out["source_record_id"])
    fallback_ids = out.apply(_canonical_source_field_hash, axis=1)
    logical_ids = out["source_record_id"].fillna("").astype(str).str.strip()
    logical_ids = logical_ids.where(has_source_record_id, fallback_ids)

    if "content_hash" in out.columns:
        revisions = out["content_hash"].fillna("").astype(str).str.strip()
    else:
        revisions = pd.Series("", index=out.index)
    missing_revision = revisions.eq("")
    revisions = revisions.where(
        ~missing_revision,
        out.apply(_canonical_source_field_hash, axis=1),
    )
    out["_source_revision_key"] = source_col + "|" + logical_ids + "|" + revisions
    out = out.sort_values(["timestamp", "source"]).drop_duplicates(
        ["_source_revision_key"], keep="first"
    )
    return out.drop(columns=["_source_revision_key"]).sort_values(
        ["timestamp", "ticker"]
    ).reset_index(drop=True)
