"""News producer timestamp and availability metadata tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from signals.news.pipeline import collect_company_news, collect_news_from_csv
from signals.catalysts.pipeline import news_to_catalysts
from signals.catalysts.nervous_system_adapter import normalize_catalyst_record
from core.nervous_system.contracts.quality import LineageRef
from signals.news.dedup import deduplicate_news
from signals.news.schema import NewsRecord, add_observation_metadata, records_from_frame


UTC = timezone.utc
SOURCE = LineageRef(
    source_id="news-schema-feed",
    content_hash="d" * 64,
    record_locator="news:row:1",
)


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


def test_news_object_does_not_fabricate_publication_time_from_legacy_timestamp() -> None:
    record = NewsRecord(
        ticker="ABC",
        timestamp="2026-07-30T13:01:00Z",
        headline="Legacy timestamp only",
        observed_at="2026-07-30T13:03:00Z",
        available_at="2026-07-30T13:03:00Z",
    ).to_record()

    assert record["timestamp"] == pd.Timestamp("2026-07-30T13:01:00Z")
    assert record["event_time"] == pd.Timestamp("2026-07-30T13:01:00Z")
    assert record["published_at"] is None


def test_news_frame_does_not_fabricate_publication_time_from_legacy_timestamp() -> None:
    out = records_from_frame(
        pd.DataFrame(
            [
                {
                    "ticker": "ABC",
                    "timestamp": "2026-07-30T13:01:00Z",
                    "observed_at": "2026-07-30T13:03:00Z",
                    "available_at": "2026-07-30T13:03:00Z",
                    "headline": "Legacy timestamp only",
                }
            ]
        ),
        source="wire",
    )

    assert pd.isna(out.iloc[0]["published_at"])


def test_news_object_does_not_localize_naive_explicit_publication_time() -> None:
    record = NewsRecord(
        ticker="ABC",
        timestamp="2026-07-30T13:01:00Z",
        headline="Naive publication metadata",
        published_at="2026-07-30 13:01:00",
        observed_at="2026-07-30T13:03:00Z",
        available_at="2026-07-30T13:03:00Z",
    ).to_record()

    assert record["published_at"] == "2026-07-30 13:01:00"


def test_news_frame_uses_explicit_time_published_field_as_publication_time() -> None:
    out = records_from_frame(
        pd.DataFrame(
            [
                {
                    "ticker": "ABC",
                    "time_published": "2026-07-30T13:01:00Z",
                    "observed_at": "2026-07-30T13:03:00Z",
                    "available_at": "2026-07-30T13:03:00Z",
                    "headline": "Explicit publication field",
                }
            ]
        ),
        source="wire",
    )

    assert out.iloc[0]["published_at"] == pd.Timestamp("2026-07-30T13:01:00Z")


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


def test_news_dedup_uses_source_record_id_and_preserves_revisions() -> None:
    rows = [
        {
            "source": "wire",
            "source_record_id": "wire-1",
            "ticker": "ABC",
            "timestamp": "2026-07-30T13:01:00Z",
            "headline": "Same source content",
            "content_hash": "content-a",
            "url": "https://example.test/a",
        },
        {
            "source": "wire",
            "source_record_id": "wire-2",
            "ticker": "ABC",
            "timestamp": "2026-07-30T13:01:00Z",
            "headline": "Same source content",
            "content_hash": "content-a",
            "url": "https://example.test/a",
        },
        {
            "source": "wire",
            "source_record_id": "wire-1",
            "ticker": "ABC",
            "timestamp": "2026-07-30T13:01:00Z",
            "headline": "Same source content",
            "content_hash": "content-a",
            "url": "https://example.test/a",
        },
        {
            "source": "wire",
            "source_record_id": "wire-1",
            "ticker": "ABC",
            "timestamp": "2026-07-30T13:01:00Z",
            "headline": "Corrected source content",
            "content_hash": "content-b",
            "url": "https://example.test/a",
        },
    ]

    out = deduplicate_news(pd.DataFrame(rows))

    assert len(out) == 3
    assert set(zip(out["source_record_id"], out["content_hash"])) == {
        ("wire-1", "content-a"),
        ("wire-2", "content-a"),
        ("wire-1", "content-b"),
    }


def test_news_dedup_falls_back_to_canonical_source_fields_without_record_id() -> None:
    rows = [
        {
            "source": "wire",
            "ticker": "ABC",
            "timestamp": "2026-07-30T13:01:00Z",
            "headline": "Same fields",
            "summary": "Summary",
            "url": "https://example.test/a",
        },
        {
            "source": "wire",
            "ticker": "ABC",
            "timestamp": "2026-07-30T13:01:00Z",
            "headline": "Same fields",
            "summary": "Summary",
            "url": "https://example.test/a",
        },
        {
            "source": "wire",
            "ticker": "ABC",
            "timestamp": "2026-07-30T13:01:00Z",
            "headline": "Revised fields",
            "summary": "Summary",
            "url": "https://example.test/a",
        },
    ]

    out = deduplicate_news(pd.DataFrame(rows))

    assert len(out) == 2
    assert set(out["headline"]) == {"Same fields", "Revised fields"}


def test_news_record_id_uses_stable_source_record_id_when_present() -> None:
    common = {
        "ticker": "ABC",
        "timestamp": "2026-07-30T13:01:00Z",
        "headline": "Same content",
        "source": "wire",
    }
    first = NewsRecord(**common, source_record_id="wire-1")
    second = NewsRecord(**common, source_record_id="wire-2")
    rerun = NewsRecord(**common, source_record_id="wire-1")

    assert first.record_id != second.record_id
    assert first.record_id == rerun.record_id


def test_news_record_id_contains_provider_namespace_and_revision() -> None:
    common = {
        "ticker": "ABC",
        "timestamp": "2026-07-30T13:01:00Z",
        "headline": "Same provider identifier",
        "source_record_id": "shared-1",
    }
    first = NewsRecord(**common, source="provider-a", summary="v1")
    other_provider = NewsRecord(**common, source="provider-b", summary="v1")
    revision = NewsRecord(**common, source="provider-a", summary="v2")
    assert first.record_id != other_provider.record_id
    assert first.record_id != revision.record_id
    assert first.record_id == NewsRecord(**common, source="provider-a", summary="v1").record_id


def test_news_frame_retains_missing_required_rows_for_quarantine() -> None:
    out = records_from_frame(
        pd.DataFrame(
            [{"ticker": "ABC", "timestamp": "2026-07-30T13:01:00Z"}]
        ),
        observed_at="2026-07-30T13:03:00Z",
    )
    assert len(out) == 1
    assert out.iloc[0]["ingestion_quarantine_code"] == "MISSING_REQUIRED_NEWS_FIELD"


def test_news_pipeline_does_not_promote_local_record_id_to_source_record_id() -> None:
    out = news_to_catalysts(
        pd.DataFrame(
            [
                {
                    "record_id": "local-compat-id",
                    "source": "wire",
                    "ticker": "ABC",
                    "timestamp": "2026-07-30T13:01:00Z",
                    "observed_at": "2026-07-30T13:03:00Z",
                    "available_at": "2026-07-30T13:03:00Z",
                    "headline": "Source ID absent",
                }
            ]
        )
    )

    assert out.iloc[0]["source_record_id"] in ("", None)


def test_news_observation_metadata_does_not_promote_local_record_id_to_source_id() -> None:
    out = add_observation_metadata(
        pd.DataFrame(
            [
                {
                    "record_id": "local-compat-id",
                    "source": "wire",
                    "ticker": "ABC",
                    "timestamp": "2026-07-30T13:01:00Z",
                    "headline": "Source ID absent",
                }
            ]
        ),
        observed_at=datetime(2026, 7, 30, 13, 3, tzinfo=UTC),
    )

    assert out.iloc[0]["source_record_id"] == ""


def test_news_schema_preserves_hindsight_evidence_for_adapter_quarantine() -> None:
    records = records_from_frame(
        pd.DataFrame(
            [
                {
                    "ticker": "ABC",
                    "timestamp": "2026-07-30T13:01:00Z",
                    "observed_at": "2026-07-30T13:03:00Z",
                    "available_at": "2026-07-30T13:03:00Z",
                    "headline": "Result artifact",
                    "guidance_strength_score": 0.8,
                }
            ]
        ),
        source="forward-guidance",
    )
    normalized = news_to_catalysts(records)
    result = normalize_catalyst_record(normalized.iloc[0].to_dict(), source_artifact=SOURCE)

    assert result.event is None
    assert result.quarantine_code == "HINDSIGHT_EARNINGS_EVIDENCE"
