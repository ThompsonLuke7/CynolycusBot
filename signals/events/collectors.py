"""Collectors for scheduled event calendars.

The network adapters are intentionally small and optional. They normalize local
CSV/manual data first, and API-backed collection can be added without changing
the downstream feature contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from signals.events.config import EARNINGS_EVENTS_PATH, MACRO_EVENTS_PATH, ensure_data_dirs
from signals.events.schema import events_from_frame, parquet_safe_causal_metadata


def collect_macro_events(
    input_csv: Path | str | None = None,
    *,
    output_path: Path | str | None = None,
    observed_at: datetime | str | None = None,
    collection_time: datetime | str | None = None,
) -> pd.DataFrame:
    """Normalize scheduled macro events from a CSV and exclude treasury auctions."""
    ensure_data_dirs()
    output_path = output_path or MACRO_EVENTS_PATH
    batch_observed_at = observed_at or collection_time or datetime.now(timezone.utc)
    if input_csv is None:
        df = pd.DataFrame()
    else:
        df = pd.read_csv(input_csv)
    out = events_from_frame(df, observed_at=batch_observed_at)
    parquet_safe_causal_metadata(out).to_parquet(output_path, index=False)
    return out


def collect_earnings_dates(
    input_csv: Path | str | None = None,
    *,
    output_path: Path | str | None = None,
    observed_at: datetime | str | None = None,
    collection_time: datetime | str | None = None,
) -> pd.DataFrame:
    """Normalize earnings-date events from CSV or the moved forward-guidance cache."""
    ensure_data_dirs()
    output_path = output_path or EARNINGS_EVENTS_PATH
    batch_observed_at = observed_at or collection_time or datetime.now(timezone.utc)
    if input_csv is not None:
        raw = pd.read_csv(input_csv)
    else:
        try:
            from signals.events.forward_guidance.data.ingest_events import load_events

            raw = load_events()
        except Exception:
            raw = pd.DataFrame()
    if raw.empty:
        out = events_from_frame(pd.DataFrame())
    else:
        df = raw.copy()
        date_col = "earnings_date" if "earnings_date" in df.columns else "date"
        df["event_type"] = "earnings"
        # The date is the scheduled occurrence.  It is never used as the
        # observation/availability timestamp.
        df["timestamp"] = pd.to_datetime(df[date_col], errors="coerce")
        df["title"] = "earnings date"
        df["source"] = df.get("source_type", "earnings_calendar")
        keep = [
            column
            for column in (
                "event_type",
                "timestamp",
                "title",
                "source",
                "ticker",
                "url",
                "observed_at",
                "available_at",
                "published_at",
                "source_record_id",
                "source_artifact_hash",
            )
            if column in df.columns
        ]
        out = events_from_frame(df[keep], observed_at=batch_observed_at)
    parquet_safe_causal_metadata(out).to_parquet(output_path, index=False)
    return out


def load_scheduled_events(
    macro_path: Path | str = MACRO_EVENTS_PATH,
    earnings_path: Path | str = EARNINGS_EVENTS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    macro = pd.read_parquet(macro_path) if Path(macro_path).exists() else pd.DataFrame()
    earnings = pd.read_parquet(earnings_path) if Path(earnings_path).exists() else pd.DataFrame()
    return macro, earnings
