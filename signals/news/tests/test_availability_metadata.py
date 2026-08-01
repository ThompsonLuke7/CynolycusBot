"""News producer timestamp and availability metadata tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from signals.news.pipeline import collect_company_news, collect_news_from_csv
from signals.news.schema import records_from_frame


UTC = timezone.utc


def test_news_schema_preserves_legacy_timestamp_and_explicit_causal_times() -> None:
    out = records_from_frame(
        pd.DataFrame(
            [
                {
                    "ticker": "ABC",
                    "timestamp": "2026-07-30T13:01:00Z",
                    "event_time": "2026-07-30T13:00:00Z",
                    "published_at": "2026-07-30T13:01:00Z",
                    "observed_at": "2026-07-30T13:03:00Z",
                    "available_at": "2026-07-30T13:03:00Z",
                    "source_record_id": "news-1",
                    "headline": "ABC contract",
                }
            ]
        ),
        source="wire",
    )
    row = out.iloc[0]
    assert row["timestamp"] == pd.Timestamp("2026-07-30T13:01:00Z")
    assert row["event_time"] == pd.Timestamp("2026-07-30T13:00:00Z")
    assert row["published_at"] == pd.Timestamp("2026-07-30T13:01:00Z")
    assert row["observed_at"] == pd.Timestamp("2026-07-30T13:03:00Z")
    assert row["available_at"] == pd.Timestamp("2026-07-30T13:03:00Z")
    assert row["source_record_id"] == "news-1"
    assert row["timestamp_semantics_version"] == "catalyst-time@1"


def test_csv_news_without_collection_time_keeps_availability_unknown(tmp_path) -> None:
    input_path = tmp_path / "news.csv"
    output_path = tmp_path / "news.parquet"
    pd.DataFrame(
        [{"ticker": "ABC", "timestamp": "2026-07-30T13:01:00Z", "headline": "ABC"}]
    ).to_csv(input_path, index=False)

    out = collect_news_from_csv(input_path, output_path=output_path)

    assert pd.isna(out.iloc[0]["observed_at"])
    assert pd.isna(out.iloc[0]["available_at"])


def test_csv_news_collection_time_sets_one_aware_batch_observation(tmp_path) -> None:
    input_path = tmp_path / "news.csv"
    output_path = tmp_path / "news.parquet"
    pd.DataFrame(
        [
            {"ticker": "ABC", "timestamp": "2026-07-30T13:01:00Z", "headline": "ABC one"},
            {"ticker": "ABC", "timestamp": "2026-07-30T13:02:00Z", "headline": "ABC two"},
        ]
    ).to_csv(input_path, index=False)
    collection_time = datetime(2026, 7, 30, 13, 3, tzinfo=UTC)

    out = collect_news_from_csv(
        input_path,
        output_path=output_path,
        collection_time=collection_time,
    )

    assert set(out["observed_at"]) == {pd.Timestamp(collection_time)}
    assert set(out["available_at"]) == {pd.Timestamp(collection_time)}
    assert all(value.tzinfo is not None for value in out["available_at"])


def test_api_news_collection_captures_one_batch_observation(monkeypatch, tmp_path) -> None:
    def fake_fetch(*_args, **_kwargs):
        return pd.DataFrame(
            [
                {"ticker": "ABC", "timestamp": "2026-07-30T13:01:00Z", "headline": "one"},
                {"ticker": "ABC", "timestamp": "2026-07-30T13:02:00Z", "headline": "two"},
            ]
        )

    monkeypatch.setattr("signals.news.pipeline.fetch_finnhub_company_news", fake_fetch)
    out = collect_company_news(
        ["ABC"],
        start="2026-07-30",
        end="2026-07-30",
        sources=("finnhub",),
        output_path=tmp_path / "news.parquet",
        merge_with_existing=False,
    )

    assert len(set(out["observed_at"])) == 1
    assert len(set(out["available_at"])) == 1
    assert out.iloc[0]["available_at"].tzinfo is not None
