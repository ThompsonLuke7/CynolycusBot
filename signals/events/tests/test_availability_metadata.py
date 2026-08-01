"""Scheduled-event producer timestamp and availability metadata tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from signals.events.collectors import collect_earnings_dates, collect_macro_events
from signals.events.schema import events_from_frame


UTC = timezone.utc


def test_scheduled_schema_separates_future_occurrence_from_calendar_availability() -> None:
    out = events_from_frame(
        pd.DataFrame(
            [
                {
                    "event_type": "FOMC",
                    "timestamp": "2026-08-13 14:00",
                    "event_time": "2026-08-13T18:00:00Z",
                    "observed_at": "2026-07-30T14:00:00Z",
                    "available_at": "2026-07-30T14:00:00Z",
                    "source_record_id": "calendar-1",
                    "title": "FOMC decision",
                }
            ]
        )
    )
    row = out.iloc[0]
    assert row["timestamp"] == pd.Timestamp("2026-08-13T18:00:00Z")
    assert row["event_time"] == pd.Timestamp("2026-08-13T18:00:00Z")
    assert row["published_at"] is None or pd.isna(row["published_at"])
    assert row["observed_at"] == pd.Timestamp("2026-07-30T14:00:00Z")
    assert row["available_at"] == pd.Timestamp("2026-07-30T14:00:00Z")
    assert row["source_record_id"] == "calendar-1"
    assert row["timestamp_semantics_version"] == "catalyst-time@1"


def test_macro_collector_records_explicit_collection_time_not_event_time(tmp_path) -> None:
    input_path = tmp_path / "macro.csv"
    output_path = tmp_path / "macro.parquet"
    pd.DataFrame(
        [{"event_type": "CPI", "timestamp": "2026-08-13 08:30", "title": "CPI"}]
    ).to_csv(input_path, index=False)
    collection_time = datetime(2026, 7, 30, 14, 0, tzinfo=UTC)

    out = collect_macro_events(
        input_path,
        output_path=output_path,
        collection_time=collection_time,
    )

    assert out.iloc[0]["event_time"] > out.iloc[0]["available_at"]
    assert out.iloc[0]["available_at"] == pd.Timestamp(collection_time)


def test_earnings_collector_keeps_known_date_and_collection_time_distinct(tmp_path) -> None:
    input_path = tmp_path / "earnings.csv"
    output_path = tmp_path / "earnings.parquet"
    pd.DataFrame(
        [{"ticker": "ABC", "earnings_date": "2026-08-13", "source_type": "calendar"}]
    ).to_csv(input_path, index=False)
    collection_time = datetime(2026, 7, 30, 14, 0, tzinfo=UTC)

    out = collect_earnings_dates(
        input_path,
        output_path=output_path,
        collection_time=collection_time,
    )

    assert out.iloc[0]["event_time"] > out.iloc[0]["available_at"]
    assert out.iloc[0]["available_at"] == pd.Timestamp(collection_time)
