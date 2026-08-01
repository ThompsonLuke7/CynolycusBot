"""Scheduled-event producer timestamp and availability metadata tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from signals.events.collectors import collect_earnings_dates, collect_macro_events
from signals.events.schema import events_from_frame
from signals.catalysts.nervous_system_adapter import normalize_catalyst_record
from signals.catalysts.pipeline import scheduled_events_to_catalysts
from core.nervous_system.contracts.quality import LineageRef


UTC = timezone.utc
SOURCE = LineageRef(
    source_id="calendar-feed",
    content_hash="b" * 64,
    record_locator="calendar:row:1",
)


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


def test_invalid_scheduled_metadata_reaches_adapter_quarantine() -> None:
    out = events_from_frame(
        pd.DataFrame(
            [
                {
                    "event_type": "CPI",
                    "timestamp": "2026-08-13T13:30:00Z",
                    "observed_at": "2026-07-30 14:00:00",
                    "available_at": "2026-07-30T14:00:00Z",
                    "title": "Naive observation must remain evidence",
                }
            ]
        )
    )

    assert len(out) == 1
    result = normalize_catalyst_record(out.iloc[0].to_dict(), source_artifact=SOURCE)
    assert result.event is None
    assert result.quarantine_code == "NAIVE_OBSERVED_AT"


def test_invalid_scheduled_publication_time_reaches_adapter_quarantine() -> None:
    out = events_from_frame(
        pd.DataFrame(
            [
                {
                    "event_type": "CPI",
                    "timestamp": "2026-08-13T13:30:00Z",
                    "published_at": "2026-07-30 13:00:00",
                    "observed_at": "2026-07-30T14:00:00Z",
                    "available_at": "2026-07-30T14:00:00Z",
                    "title": "Naive publication must remain evidence",
                }
            ]
        )
    )

    result = normalize_catalyst_record(out.iloc[0].to_dict(), source_artifact=SOURCE)
    assert result.event is None
    assert result.quarantine_code == "NAIVE_PUBLISHED_AT"


def test_invalid_scheduled_event_time_reaches_adapter_quarantine() -> None:
    out = events_from_frame(
        pd.DataFrame(
            [
                {
                    "event_type": "CPI",
                    "timestamp": "not-a-scheduled-time",
                    "observed_at": "2026-07-30T14:00:00Z",
                    "available_at": "2026-07-30T14:00:00Z",
                    "title": "Malformed occurrence remains evidence",
                }
            ]
        )
    )

    assert len(out) == 1
    result = normalize_catalyst_record(out.iloc[0].to_dict(), source_artifact=SOURCE)
    assert result.event is None
    assert result.quarantine_code == "INVALID_EVENT_TIME"


def test_scheduled_catalyst_pipeline_does_not_drop_invalid_occurrence_rows() -> None:
    source_rows = events_from_frame(
        pd.DataFrame(
            [
                {
                    "event_type": "CPI",
                    "timestamp": "not-a-scheduled-time",
                    "observed_at": "2026-07-30T14:00:00Z",
                    "available_at": "2026-07-30T14:00:00Z",
                    "title": "Malformed occurrence remains evidence",
                }
            ]
        )
    )

    normalized = scheduled_events_to_catalysts(source_rows, default_kind="scheduled_event")

    assert len(normalized) == 1
    assert normalized.iloc[0]["ingestion_quarantine_code"] == "INVALID_EVENT_TIME"
    assert normalized.iloc[0]["ingestion_quarantine_message"]
    raw = normalized.iloc[0]["raw_ingestion_fields"]
    assert raw["timestamp"] == "not-a-scheduled-time"
    assert raw["observed_at"] == "2026-07-30T14:00:00Z"


def test_scheduled_schema_preserves_hindsight_evidence_for_adapter_quarantine() -> None:
    out = events_from_frame(
        pd.DataFrame(
            [
                {
                    "event_type": "earnings",
                    "timestamp": "2026-08-13T13:30:00Z",
                    "observed_at": "2026-07-30T14:00:00Z",
                    "available_at": "2026-07-30T14:00:00Z",
                    "title": "Result artifact",
                    "guidance_strength_score": 0.8,
                }
            ]
        )
    )
    normalized = scheduled_events_to_catalysts(out, default_kind="earnings")
    result = normalize_catalyst_record(normalized.iloc[0].to_dict(), source_artifact=SOURCE)

    assert result.event is None
    assert result.quarantine_code == "HINDSIGHT_EARNINGS_EVIDENCE"


def test_event_frame_retains_missing_required_rows_for_quarantine() -> None:
    out = events_from_frame(
        pd.DataFrame([{"timestamp": "2026-08-13T13:30:00Z", "title": "Missing type"}])
    )
    assert len(out) == 1
    assert out.iloc[0]["ingestion_quarantine_code"] == "MISSING_REQUIRED_EVENT_FIELD"
