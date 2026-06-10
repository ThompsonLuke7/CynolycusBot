"""Collectors for scheduled event calendars.

The network adapters are intentionally small and optional. They normalize local
CSV/manual data first, and API-backed collection can be added without changing
the downstream feature contract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from signals.events.config import EARNINGS_EVENTS_PATH, MACRO_EVENTS_PATH, ensure_data_dirs
from signals.events.schema import events_from_frame


def collect_macro_events(input_csv: Path | str | None = None, *, output_path: Path | str | None = None) -> pd.DataFrame:
    """Normalize scheduled macro events from a CSV and exclude treasury auctions."""
    ensure_data_dirs()
    output_path = output_path or MACRO_EVENTS_PATH
    if input_csv is None:
        df = pd.DataFrame(columns=["event_type", "timestamp", "title", "source", "ticker", "url"])
    else:
        df = pd.read_csv(input_csv)
    out = events_from_frame(df)
    out.to_parquet(output_path, index=False)
    return out


def collect_earnings_dates(input_csv: Path | str | None = None, *, output_path: Path | str | None = None) -> pd.DataFrame:
    """Normalize earnings-date events from CSV or the moved forward-guidance cache."""
    ensure_data_dirs()
    output_path = output_path or EARNINGS_EVENTS_PATH
    if input_csv is not None:
        raw = pd.read_csv(input_csv)
    else:
        try:
            from signals.events.forward_guidance.data.ingest_events import load_events

            raw = load_events()
        except Exception:
            raw = pd.DataFrame()
    if raw.empty:
        out = pd.DataFrame(columns=["event_type", "timestamp", "title", "source", "ticker", "url"])
    else:
        df = raw.copy()
        date_col = "earnings_date" if "earnings_date" in df.columns else "date"
        df["event_type"] = "earnings"
        df["timestamp"] = pd.to_datetime(df[date_col], errors="coerce")
        df["title"] = "earnings date"
        df["source"] = df.get("source_type", "earnings_calendar")
        out = df[["event_type", "timestamp", "title", "source", "ticker"]].dropna(subset=["timestamp"])
    out.to_parquet(output_path, index=False)
    return out


def load_scheduled_events(
    macro_path: Path | str = MACRO_EVENTS_PATH,
    earnings_path: Path | str = EARNINGS_EVENTS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    macro = pd.read_parquet(macro_path) if Path(macro_path).exists() else pd.DataFrame()
    earnings = pd.read_parquet(earnings_path) if Path(earnings_path).exists() else pd.DataFrame()
    return macro, earnings
