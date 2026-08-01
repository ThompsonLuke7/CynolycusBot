"""Causal timestamp semantics for catalyst state adaptation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import NAMESPACE_URL, uuid5

from core.nervous_system.contracts.quality import LineageRef
import pandas as pd
import pytest

from signals.catalysts.pipeline import (
    build_catalyst_records,
    news_to_catalysts,
    scheduled_events_to_catalysts,
)
from signals.catalysts.nervous_system_adapter import (
    aggregate_catalyst_pressure,
    normalize_catalyst_record,
    normalize_catalyst_records,
    publish_catalyst_states,
)
from signals.events.forward_guidance.features.build_matrix import LABEL_COLUMNS, POST_EVENT_FEATURE_COLUMNS
from signals.events.forward_guidance.features.nlp import (
    extract_forward_sections,
    extract_structured_guidance_features,
)
from signals.events.collectors import collect_macro_events
from signals.news.pipeline import collect_news_from_csv


UTC = timezone.utc
SOURCE = LineageRef(
    source_id="wire-feed",
    content_hash="a" * 64,
    record_locator="wire-feed:record:1",
)

_ACTUAL_FORWARD_GUIDANCE_EXTRACTOR_FIELDS = tuple(
    sorted(
        set(extract_forward_sections("").keys())
        | set(extract_structured_guidance_features("").keys())
        | {
            "alt_finbert_negative",
            "alt_finbert_neutral",
            "alt_finbert_positive",
            "alt_finbert_tone_score",
            "emb_minilm_0000",
            "embedding_available",
            "finbert_available",
            "finbert_negative",
            "finbert_neutral",
            "finbert_positive",
            "finbert_tone_score",
            "metric_reported_eps",
        }
    )
)


def _time(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 7, 30, hour, minute, tzinfo=UTC)


def test_news_preserves_event_publication_observation_and_availability_times() -> None:
    result = normalize_catalyst_record(
        {
            "source": "wire",
            "source_record_id": "news-1",
            "ticker": "ABC",
            "event_type": "company_news",
            "event_time": _time(13, 0),
            "published_at": _time(13, 1),
            "observed_at": _time(13, 3),
            "available_at": _time(13, 3),
            "headline": "ABC announces a new contract",
        },
        source_artifact=SOURCE,
    )

    assert result.event is not None
    assert result.event.event_time == _time(13, 0)
    assert result.event.published_at == _time(13, 1)
    assert result.event.observed_at == _time(13, 3)
    assert result.event.available_at == _time(13, 3)


def test_earnings_date_known_two_weeks_before_occurrence_keeps_availability_early() -> None:
    result = normalize_catalyst_record(
        {
            "source": "earnings-calendar",
            "source_record_id": "earnings-1",
            "ticker": "ABC",
            "event_type": "earnings",
            "event_time": _time(20, 0),
            "observed_at": _time(14, 0),
            "available_at": _time(14, 0),
        },
        source_artifact=SOURCE,
    )

    assert result.event is not None
    assert result.event.event_time == _time(20, 0)
    assert result.event.published_at is None
    assert result.event.observed_at == _time(14, 0)
    assert result.event.available_at == _time(14, 0)


def test_future_scheduled_event_is_eligible_when_schedule_is_observed() -> None:
    result = normalize_catalyst_record(
        {
            "source": "macro-calendar",
            "source_record_id": "macro-1",
            "event_type": "fomc_decision",
            "event_time": _time(20, 0),
            "observed_at": _time(14, 0),
            "available_at": _time(14, 0),
            "score": -0.2,
        },
        source_artifact=SOURCE,
    )

    assert result.event is not None
    pressure = aggregate_catalyst_pressure(
        (result.event,),
        entity_id="US",
        decision_time=_time(14, 0),
        valid_until=_time(16, 0),
        config_version="catalyst@1",
    )
    assert result.event.event_time == _time(20, 0)
    assert result.event.available_at == _time(14, 0)
    assert pressure.event_ids == (result.event.event_id,)


def test_ambiguous_legacy_timestamp_is_quarantined_without_an_event() -> None:
    result = normalize_catalyst_record(
        {
            "source": "legacy-news",
            "source_record_id": "legacy-1",
            "ticker": "ABC",
            "event_type": "company_news",
            "timestamp": "2026-07-30 13:01:00",
            "headline": "Legacy row with unknown timestamp semantics",
        },
        source_artifact=SOURCE,
    )

    assert result.event is None
    assert result.quarantine_code == "MISSING_AVAILABLE_AT"


def test_missing_publication_time_is_explicitly_warned_without_fabrication() -> None:
    result = normalize_catalyst_record(
        {
            "source": "wire",
            "source_record_id": "missing-publication-1",
            "ticker": "ABC",
            "event_type": "company_news",
            "timestamp": _time(13, 0),
            "observed_at": _time(13, 3),
            "available_at": _time(13, 3),
        },
        source_artifact=SOURCE,
    )

    assert result.event is not None
    assert result.event.published_at is None
    assert "MISSING_PUBLICATION_TIME" in result.warnings


def test_late_observation_is_not_eligible_at_an_earlier_decision() -> None:
    result = normalize_catalyst_record(
        {
            "source": "wire",
            "source_record_id": "news-late",
            "ticker": "ABC",
            "event_type": "company_news",
            "event_time": _time(13, 0),
            "published_at": _time(13, 1),
            "observed_at": _time(14, 30),
            "available_at": _time(14, 30),
            "score": 0.75,
        },
        source_artifact=SOURCE,
    )

    assert result.event is not None
    pressure = aggregate_catalyst_pressure(
        (result.event,),
        entity_id="ABC",
        decision_time=_time(14, 0),
        valid_until=_time(16, 0),
        config_version="catalyst@1",
    )
    assert result.event.published_at == _time(13, 1)
    assert result.event.observed_at == _time(14, 30)
    assert result.event.available_at == _time(14, 30)
    assert pressure.event_ids == ()


def test_news_pipeline_quarantines_naive_explicit_observation_and_availability() -> None:
    frame = pd.DataFrame(
        [
            {
                "record_id": "naive-news",
                "source_record_id": "naive-news",
                "ticker": "ABC",
                "source": "wire",
                "event_type": "company_news",
                "timestamp": "2026-07-30T13:01:00Z",
                "event_time": "2026-07-30T13:00:00Z",
                "observed_at": "2026-07-30 13:03:00",
                "available_at": "2026-07-30 13:03:00",
                "headline": "Naive metadata must not become UTC",
            }
        ]
    )

    normalized = news_to_catalysts(frame)
    result = normalize_catalyst_record(normalized.iloc[0].to_dict(), source_artifact=SOURCE)

    assert result.event is None
    assert result.quarantine_code in {"NAIVE_OBSERVED_AT", "NAIVE_AVAILABLE_AT"}


def test_scheduled_pipeline_preserves_mixed_aware_and_naive_metadata_for_adapter_quarantine() -> None:
    frame = pd.DataFrame(
        [
            {
                "source_record_id": "aware-calendar",
                "source": "calendar",
                "event_type": "cpi",
                "timestamp": "2026-08-13T13:30:00Z",
                "observed_at": "2026-07-30T14:00:00Z",
                "available_at": "2026-07-30T14:00:00Z",
                "title": "CPI",
            },
            {
                "source_record_id": "naive-calendar",
                "source": "calendar",
                "event_type": "fomc_decision",
                "timestamp": "2026-08-13T18:00:00Z",
                "observed_at": "2026-07-30 14:00:00",
                "available_at": "2026-07-30T14:00:00Z",
                "title": "FOMC",
            },
        ]
    )

    normalized = scheduled_events_to_catalysts(frame, default_kind="scheduled_event")
    results = [
        normalize_catalyst_record(row, source_artifact=SOURCE)
        for row in normalized.to_dict(orient="records")
    ]

    assert results[0].event is not None
    assert results[1].event is None
    assert results[1].quarantine_code == "NAIVE_OBSERVED_AT"


def test_identical_revision_is_idempotent_but_conflicting_revision_is_preserved() -> None:
    base = {
        "source": "wire",
        "source_record_id": "revision-1",
        "ticker": "ABC",
        "event_type": "company_news",
        "event_time": _time(13, 0),
        "published_at": _time(13, 1),
        "observed_at": _time(13, 3),
        "available_at": _time(13, 3),
        "headline": "Original headline",
    }
    identical = normalize_catalyst_records((base, dict(base)), source_artifact=SOURCE)
    assert len([result.event for result in identical if result.event is not None]) == 1

    revised = dict(base, headline="Corrected headline")
    conflicting = normalize_catalyst_records((base, revised), source_artifact=SOURCE)
    events = [result.event for result in conflicting if result.event is not None]
    assert len(events) == 2
    assert events[0].event_id != events[1].event_id
    assert all("CONFLICTING_SOURCE_VALUES" in result.warnings for result in conflicting)


def test_batch_fallback_identity_deduplicates_rows_without_source_record_id() -> None:
    batch_source = SOURCE.model_copy(update={"record_locator": "catalyst_records:batch"})
    row = {
        "source": "wire",
        "ticker": "ABC",
        "event_type": "company_news",
        "event_time": _time(13, 0),
        "observed_at": _time(13, 3),
        "available_at": _time(13, 3),
        "headline": "No provider ID",
    }

    results = normalize_catalyst_records(
        (dict(row, record_id="local-one"), dict(row, record_id="local-two")),
        source_artifact=batch_source,
    )

    assert len([result.event for result in results if result.event is not None]) == 1


def test_source_lineage_is_exact_and_deterministic_without_runtime_timestamps() -> None:
    row = {
        "source": "wire",
        "source_record_id": "stable-1",
        "ticker": "ABC",
        "event_type": "company_news",
        "event_time": _time(13, 0),
        "published_at": _time(13, 1),
        "observed_at": _time(13, 3),
        "available_at": _time(13, 3),
        "headline": "Stable source identity",
    }
    first = normalize_catalyst_record(row, source_artifact=SOURCE).event
    second = normalize_catalyst_record(
        dict(
            row,
            observed_at=_time(15, 0),
            available_at=_time(15, 0),
            source_path="/different/local/worktree/news.parquet",
            generated_at=_time(15, 1),
        ),
        source_artifact=SOURCE,
    ).event
    assert first is not None and second is not None
    assert first.state_id == second.state_id == first.event_id
    assert first.lineage_ids == second.lineage_ids
    assert SOURCE.source_id in first.lineage_ids[0]
    assert SOURCE.content_hash in first.lineage_ids[0]
    assert SOURCE.record_locator in first.lineage_ids[0]


def test_local_compatibility_record_id_does_not_change_fallback_event_identity() -> None:
    base = {
        "source": "wire",
        "ticker": "ABC",
        "event_type": "company_news",
        "event_time": _time(13, 0),
        "observed_at": _time(13, 3),
        "available_at": _time(13, 3),
        "headline": "Fallback source fields",
    }
    first = normalize_catalyst_record(
        dict(base, record_id="local-one"), source_artifact=SOURCE
    ).event
    second = normalize_catalyst_record(
        dict(base, record_id="local-two"), source_artifact=SOURCE
    ).event

    assert first is not None and second is not None
    assert first.event_id == second.event_id


def test_event_identity_canonicalizes_nested_unordered_values_across_hash_seeds() -> None:
    script = """
from datetime import datetime, timezone
from core.nervous_system.contracts.quality import LineageRef
from signals.catalysts.nervous_system_adapter import normalize_catalyst_record

source = LineageRef(source_id='wire-feed', content_hash='a' * 64, record_locator='row:1')
row = {
    'source': 'wire',
    'source_record_id': 'seed-stable-1',
    'ticker': 'ABC',
    'event_type': 'company_news',
    'event_time': datetime(2026, 7, 30, 13, tzinfo=timezone.utc),
    'observed_at': datetime(2026, 7, 30, 13, 3, tzinfo=timezone.utc),
    'available_at': datetime(2026, 7, 30, 13, 3, tzinfo=timezone.utc),
    'nested_metadata': {'unordered': frozenset({'alpha', 'beta', 'gamma'})},
}
print(normalize_catalyst_record(row, source_artifact=source).event.event_id)
"""
    ids = []
    repo_root = Path(__file__).resolve().parents[3]
    for seed in ("1", "2"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        ids.append(completed.stdout.strip())

    assert ids[0] == ids[1]


def test_raw_directional_score_is_not_promoted_to_probability() -> None:
    result = normalize_catalyst_record(
        {
            "source": "wire",
            "source_record_id": "scored-1",
            "ticker": "ABC",
            "event_type": "company_news",
            "event_time": _time(13, 0),
            "published_at": _time(13, 1),
            "observed_at": _time(13, 3),
            "available_at": _time(13, 3),
            "score": 4.5,
        },
        source_artifact=SOURCE,
    )
    assert result.event is not None
    assert result.event.relation_confidence is None
    pressure = publish_pressure_for_test(result.event, score=4.5)
    assert pressure.aggregate_score == 4.5
    assert pressure.transition_probabilities == {}


def test_public_pressure_aggregation_uses_typed_event_raw_scores() -> None:
    result = normalize_catalyst_record(
        {
            "source": "wire",
            "source_record_id": "public-score-1",
            "ticker": "ABC",
            "event_type": "company_news",
            "event_time": _time(13, 0),
            "published_at": _time(13, 1),
            "observed_at": _time(13, 3),
            "available_at": _time(13, 3),
            "directional_score": 4.5,
        },
        source_artifact=SOURCE,
    )
    assert result.event is not None

    pressure = aggregate_catalyst_pressure(
        (result.event,),
        entity_id="ABC",
        decision_time=_time(14, 0),
        valid_until=_time(16, 0),
        config_version="catalyst@1",
    )

    assert result.event.raw_score == 4.5
    assert pressure.aggregate_score == 4.5
    assert pressure.channel_scores == {result.event.channel: 4.5}
    assert pressure.transition_probabilities == {}


def test_earnings_result_hindsight_is_quarantined() -> None:
    result = normalize_catalyst_record(
        {
            "source": "forward-guidance",
            "source_record_id": "result-1",
            "ticker": "ABC",
            "event_type": "earnings_result_guidance",
            "event_time": _time(20, 0),
            "observed_at": _time(14, 0),
            "available_at": _time(14, 0),
            "fwd_ret_5d": 0.25,
            "max_drawdown": -0.1,
        },
        source_artifact=SOURCE,
    )
    assert result.event is None
    assert result.quarantine_code == "HINDSIGHT_EARNINGS_EVIDENCE"


def test_hindsight_boundary_quarantines_adversarial_result_aliases_regardless_of_event_name() -> None:
    base = {
        "source": "calendar-or-label-store",
        "source_record_id": "adversarial-1",
        "ticker": "ABC",
        "event_type": "earnings",
        "event_time": _time(20, 0),
        "observed_at": _time(14, 0),
        "available_at": _time(14, 0),
    }
    for field_name in (
        "guidance_strength_score",
        "forward_return_5d",
        "fwd_ret_1d",
        "actual_eps",
        "realized_catalyst_score",
        "beat_miss",
        "post_event_return",
        "post_event_drawdown",
        "label_value",
        "target_return",
    ):
        result = normalize_catalyst_record(
            dict(base, event_type="calendar", **{field_name: 0.25}),
            source_artifact=SOURCE,
        )
        assert result.event is None, field_name
        assert result.quarantine_code == "HINDSIGHT_EARNINGS_EVIDENCE", field_name


def test_pre_event_calendar_metadata_is_not_hindsight_evidence() -> None:
    result = normalize_catalyst_record(
        {
            "source": "earnings-calendar",
            "source_record_id": "calendar-1",
            "ticker": "ABC",
            "event_type": "earnings",
            "event_time": _time(20, 0),
            "observed_at": _time(14, 0),
            "available_at": _time(14, 0),
            "consensus_eps": 1.25,
            "expected_move": 0.04,
            "guidance_date": "2026-08-13",
        },
        source_artifact=SOURCE,
    )

    assert result.event is not None


def test_actual_forward_guidance_alias_matrix_is_quarantined() -> None:
    base = {
        "source": "forward-guidance",
        "source_record_id": "actual-alias-matrix",
        "ticker": "ABC",
        "event_type": "earnings",
        "event_time": _time(20, 0),
        "observed_at": _time(14, 0),
        "available_at": _time(14, 0),
    }
    aliases = set(LABEL_COLUMNS) | set(POST_EVENT_FEATURE_COLUMNS) | {
        "realized_return",
        "actual_result",
        "result_score",
        "drawdown_60d",
    }
    for field_name in sorted(aliases):
        result = normalize_catalyst_record(
            dict(base, event_type="calendar", **{field_name: 0.25}),
            source_artifact=SOURCE,
        )
        assert result.event is None, field_name
        assert result.quarantine_code == "HINDSIGHT_EARNINGS_EVIDENCE", field_name


@pytest.mark.parametrize(
    "field_name",
    tuple(
        alias
        for emitted in _ACTUAL_FORWARD_GUIDANCE_EXTRACTOR_FIELDS
        for alias in (emitted, emitted.upper().replace("_", "-"))
    ),
)
def test_every_actual_forward_guidance_extractor_output_and_alias_is_quarantined(
    field_name: str,
) -> None:
    result = normalize_catalyst_record(
        {
            "source": "forward-guidance",
            "source_record_id": f"extractor:{field_name}",
            "ticker": "ABC",
            "event_type": "calendar",
            "event_time": _time(20, 0),
            "observed_at": _time(14, 0),
            "available_at": _time(14, 0),
            field_name: 0.25,
        },
        source_artifact=SOURCE,
    )

    assert result.event is None, field_name
    assert result.quarantine_code == "HINDSIGHT_EARNINGS_EVIDENCE", field_name


def test_source_lineage_identity_normalizes_logical_feed_variants() -> None:
    row = {
        "source": "wire",
        "source_record_id": "identity-1",
        "ticker": "ABC",
        "event_type": "company_news",
        "event_time": _time(13),
        "observed_at": _time(13, 1),
        "available_at": _time(13, 1),
        "headline": "Stable identity",
    }
    first = normalize_catalyst_record(
        row,
        source_artifact=SOURCE.model_copy(update={"source_id": " Feed-X ", "content_hash": "A" * 64}),
    )
    second = normalize_catalyst_record(
        row,
        source_artifact=SOURCE.model_copy(update={"source_id": "feed-x", "content_hash": "a" * 64}),
    )
    assert first.event is not None and second.event is not None
    assert first.event.event_id == second.event.event_id
    assert first.event.state_id == second.event.state_id
    assert first.event.lineage_ids == second.event.lineage_ids


def test_explicit_invalid_occurrence_is_not_treated_as_missing() -> None:
    result = normalize_catalyst_record(
        {
            "source": "calendar",
            "source_record_id": "invalid-occurrence",
            "event_type": "cpi",
            "event_time": "not-a-time",
            "observed_at": _time(14, 0),
            "available_at": _time(14, 0),
        },
        source_artifact=SOURCE,
    )
    assert result.event is None
    assert result.quarantine_code == "INVALID_EVENT_TIME"


def test_batch_lineage_expands_any_batch_locator() -> None:
    source = SOURCE.model_copy(update={"record_locator": "task11:postgres:batch"})
    results = normalize_catalyst_records(
        [
            {
                "source": "wire",
                "source_record_id": "row-a",
                "ticker": "ABC",
                "event_type": "company_news",
                "event_time": _time(13),
                "observed_at": _time(13, 3),
                "available_at": _time(13, 3),
            },
            {
                "source": "wire",
                "source_record_id": "row-b",
                "ticker": "XYZ",
                "event_type": "company_news",
                "event_time": _time(13),
                "observed_at": _time(13, 3),
                "available_at": _time(13, 3),
            },
        ],
        source_artifact=source,
    )
    assert results[0].event is not None and results[1].event is not None
    assert results[0].event.lineage_ids != results[1].event.lineage_ids


def test_batch_lineage_honors_explicit_rows_and_sanitizes_path_fallbacks() -> None:
    source = SOURCE.model_copy(update={"record_locator": "/tmp/task11/catalyst_records:batch"})
    common = {
        "source": "wire",
        "source_record_id": "same-provider-row",
        "ticker": "ABC",
        "event_type": "company_news",
        "event_time": _time(13),
        "observed_at": _time(13, 3),
        "available_at": _time(13, 3),
        "headline": "Identical source content",
    }
    rows = (
        dict(common, record_locator=" provider:row:left "),
        dict(common, record_locator="provider:row:right"),
        dict(
            common,
            source_record_id="path-source-id",
            source_id="/tmp/provider/source-row.json",
            record_locator="/tmp/provider/explicit-row.json",
        ),
        dict(
            common,
            source_record_id="bad-row",
            record_locator="provider:row:bad",
            observed_at="2026-07-30 13:03:00",
        ),
    )
    uow = _RecordingUow()

    first = publish_catalyst_states(rows, unit_of_work=uow, source_artifact=source)
    canonical_rows = tuple(
        dict(row, record_locator=str(row["record_locator"]).strip()) for row in rows
    )
    second = publish_catalyst_states(
        canonical_rows, unit_of_work=uow, source_artifact=source
    )

    first_events = [result.event for result in first if result.event is not None]
    second_events = [result.event for result in second if result.event is not None]
    assert len(first_events) == len(second_events) == 3
    assert [event.event_id for event in first_events] == [event.event_id for event in second_events]
    event_lineages = {
        str(event.event_id): json.loads(event.lineage_ids[0]) for event in first_events
    }
    locators = {payload["record_locator"] for payload in event_lineages.values()}
    assert {"provider:row:left", "provider:row:right"} <= locators
    assert len(locators) == 3
    assert all("/tmp" not in locator and "\\" not in locator for locator in locators)

    items = {item.target_id: item for item in uow.registry.items if item.target_id is not None}
    assert set(items) == set(event_lineages)
    for event_id, payload in event_lineages.items():
        assert items[event_id].record_locator == payload["record_locator"]
    quarantine = uow.registry.quarantines[0]
    assert quarantine.record_locator == "provider:row:bad"
    assert all("/tmp" not in item.record_locator for item in uow.registry.items)
    assert {
        (edge.target_id, edge.relationship) for edge in uow.registry.edges
    } == {(event_id, "IMPORTED_AS") for event_id in event_lineages}
    assert len(uow.registry.items) == 4
    assert len(uow.registry.quarantines) == 1
    assert len(uow.registry.edges) == 3
    assert uow.registry.runs[-1].counts == {
        "attempted": 4,
        "parsed": 4,
        "imported": 0,
        "quarantined": 0,
        "duplicates": 4,
    }
    assert uow.commits == uow.rollbacks == 0


def test_news_pipeline_carries_hindsight_fields_to_adapter_quarantine() -> None:
    normalized = news_to_catalysts(
        pd.DataFrame(
            [
                {
                    "record_id": "label-artifact",
                    "source_record_id": "label-artifact",
                    "source": "forward-guidance",
                    "ticker": "ABC",
                    "event_type": "earnings",
                    "timestamp": "2026-07-30T13:00:00Z",
                    "observed_at": "2026-07-30T13:03:00Z",
                    "available_at": "2026-07-30T13:03:00Z",
                    "headline": "Result artifact mislabeled as a calendar row",
                    "guidance_strength_score": 0.8,
                }
            ]
        )
    )

    result = normalize_catalyst_record(normalized.iloc[0].to_dict(), source_artifact=SOURCE)

    assert result.event is None
    assert result.quarantine_code == "HINDSIGHT_EARNINGS_EVIDENCE"


def test_pipeline_excludes_forward_guidance_result_evidence_and_publishes_via_caller_uow(tmp_path) -> None:
    news_path = tmp_path / "news.parquet"
    macro_path = tmp_path / "macro.parquet"
    earnings_path = tmp_path / "earnings.parquet"
    features_path = tmp_path / "result_features.parquet"
    labels_path = tmp_path / "result_labels.parquet"
    output_path = tmp_path / "catalyst_records.parquet"
    pd.DataFrame(
        [
            {
                "record_id": "news-1",
                "ticker": "ABC",
                "timestamp": _time(13, 1),
                "event_time": _time(13, 0),
                "published_at": _time(13, 1),
                "observed_at": _time(13, 3),
                "available_at": _time(13, 3),
                "source_record_id": "news-1",
                "headline": "ABC contract",
                "source": "wire",
            }
        ]
    ).to_parquet(news_path, index=False)
    pd.DataFrame(columns=["event_type", "timestamp"]).to_parquet(macro_path, index=False)
    pd.DataFrame(columns=["event_type", "timestamp"]).to_parquet(earnings_path, index=False)
    pd.DataFrame(
        [{"event_id": "result-1", "ticker": "ABC", "signal_timestamp": _time(14, 0), "metric_eps_actual": 2.0}]
    ).to_parquet(features_path, index=False)
    pd.DataFrame(
        [{"event_id": "result-1", "ticker": "ABC", "signal_timestamp": _time(14, 0), "fwd_ret_5d": 0.3}]
    ).to_parquet(labels_path, index=False)
    uow = _RecordingUow()

    records = build_catalyst_records(
        news_path=news_path,
        macro_path=macro_path,
        earnings_path=earnings_path,
        earnings_result_features_path=features_path,
        earnings_result_labels_path=labels_path,
        output_path=output_path,
        unit_of_work=uow,
        source_artifact=SOURCE,
    )

    assert set(records["catalyst_kind"]) == {"news"}
    assert [state.state_type.value for state in uow.states.saved] == ["CATALYST_EVENT"]
    assert uow.commits == 0
    assert uow.rollbacks == 0


def test_publication_rerun_converges_by_stable_state_identity() -> None:
    class IdempotentStates(_RecordingStates):
        def __init__(self) -> None:
            super().__init__()
            self.by_id = {}

        def insert_states_idempotently(self, states):
            for state in states:
                self.by_id.setdefault(state.state_id, state)
            self.saved = list(self.by_id.values())
            return {state.state_id: state.state_id for state in states}

    uow = _RecordingUow()
    uow.states = IdempotentStates()
    row = {
        "source": "wire",
        "source_record_id": "rerun-1",
        "ticker": "ABC",
        "event_type": "company_news",
        "event_time": _time(13, 0),
        "published_at": _time(13, 1),
        "observed_at": _time(13, 3),
        "available_at": _time(13, 3),
        "headline": "same revision",
    }
    publish_catalyst_states((row,), unit_of_work=uow, source_artifact=SOURCE)
    publish_catalyst_states(
        (dict(row, observed_at=_time(15, 0), available_at=_time(15, 0)),),
        unit_of_work=uow,
        source_artifact=SOURCE,
    )

    assert len(uow.states.by_id) == 1
    assert uow.commits == 0
    assert uow.rollbacks == 0


def test_publication_groups_eligible_events_by_ticker_and_explicit_market_scope() -> None:
    common = {
        "source": "wire",
        "event_type": "company_news",
        "event_time": _time(13, 0),
        "published_at": _time(13, 1),
        "observed_at": _time(13, 3),
        "available_at": _time(13, 3),
    }
    rows = (
        dict(common, source_record_id="abc-1", ticker="ABC", score=1.0),
        dict(common, source_record_id="msft-1", ticker="MSFT", score=3.0),
        dict(common, source="macro-calendar", source_record_id="market-1", score=5.0),
    )
    uow = _RecordingUow()

    results = publish_catalyst_states(
        rows,
        unit_of_work=uow,
        source_artifact=SOURCE,
        decision_time=_time(14, 0),
        valid_until=_time(16, 0),
    )

    events = [state for state in uow.states.saved if state.state_type.value == "CATALYST_EVENT"]
    pressures = [state for state in uow.states.saved if state.state_type.value == "CATALYST_PRESSURE"]
    event_by_id = {event.event_id: event for event in events}
    assert len([result.event for result in results if result.event is not None]) == 3
    assert len(events) == 3
    assert {pressure.scope_id for pressure in pressures} == {"ABC", "MSFT", "MARKET"}
    assert len(pressures) == 3
    for pressure in pressures:
        selected = [event_by_id[event_id] for event_id in pressure.event_ids]
        if pressure.scope_id == "MARKET":
            assert [event.ticker for event in selected] == [None]
        else:
            assert {event.ticker for event in selected} == {pressure.scope_id}


def test_scheduled_projection_groups_null_ticker_as_market_and_keeps_scores_aligned() -> None:
    projected = scheduled_events_to_catalysts(
        pd.DataFrame(
            [
                {
                    "source_record_id": "abc-macro",
                    "ticker": "ABC",
                    "event_type": "earnings",
                    "timestamp": _time(20),
                    "observed_at": _time(14),
                    "available_at": _time(14),
                    "score": 1.0,
                },
                {
                    "source_record_id": "msft-macro",
                    "ticker": "MSFT",
                    "event_type": "earnings",
                    "timestamp": _time(20),
                    "observed_at": _time(14),
                    "available_at": _time(14),
                    "score": 3.0,
                },
                {
                    "source_record_id": "market-macro",
                    "ticker": None,
                    "event_type": "cpi",
                    "timestamp": _time(20),
                    "observed_at": _time(14),
                    "available_at": _time(14),
                    "score": -0.5,
                },
            ]
        ),
        default_kind="scheduled_event",
    )
    uow = _RecordingUow()

    results = publish_catalyst_states(
        projected.to_dict(orient="records"),
        unit_of_work=uow,
        source_artifact=SOURCE.model_copy(update={"record_locator": "macro:batch"}),
        decision_time=_time(14),
        valid_until=_time(16),
    )

    events = [result.event for result in results if result.event is not None]
    pressures = {
        state.scope_id: state
        for state in uow.states.saved
        if state.state_type.value == "CATALYST_PRESSURE"
    }
    assert len(events) == len(pressures) == 3
    assert [event.ticker for event in events] == ["ABC", "MSFT", None]
    assert [event.is_direct for event in events] == [True, True, False]
    assert set(pressures) == {"ABC", "MSFT", "MARKET"}
    assert "NONE" not in pressures
    assert pressures["ABC"].aggregate_score == pytest.approx(1.0)
    assert pressures["MSFT"].aggregate_score == pytest.approx(3.0)
    assert pressures["MARKET"].aggregate_score == pytest.approx(-0.5)
    assert all(len(pressure.event_ids) == 1 for pressure in pressures.values())


def test_publication_keeps_scores_aligned_after_duplicate_revision_convergence() -> None:
    common = {
        "source": "wire",
        "ticker": "ABC",
        "event_type": "company_news",
        "event_time": _time(13, 0),
        "published_at": _time(13, 1),
        "observed_at": _time(13, 3),
        "available_at": _time(13, 3),
    }
    rows = (
        dict(common, source_record_id="duplicate-1", headline="same", score=1.0),
        dict(common, source_record_id="duplicate-1", headline="same", score=1.0),
        dict(common, source_record_id="distinct-1", headline="different", score=5.0),
    )
    uow = _RecordingUow()

    publish_catalyst_states(
        rows,
        unit_of_work=uow,
        source_artifact=SOURCE,
        decision_time=_time(14, 0),
        valid_until=_time(16, 0),
    )

    pressure = next(
        state for state in uow.states.saved if state.state_type.value == "CATALYST_PRESSURE"
    )
    assert pressure.aggregate_score == 3.0
    assert pressure.channel_scores == {"WIRE": 3.0}


def test_publication_registers_source_import_items_and_quarantines_in_caller_uow() -> None:
    valid = {
        "source": "wire",
        "source_record_id": "registered-1",
        "ticker": "ABC",
        "event_type": "company_news",
        "event_time": _time(13, 0),
        "observed_at": _time(13, 3),
        "available_at": _time(13, 3),
        "headline": "valid row",
    }
    quarantined = dict(
        valid,
        source_record_id="registered-bad",
        observed_at="2026-07-30 13:03:00",
    )
    uow = _RecordingUow()

    results = publish_catalyst_states(
        (valid, quarantined),
        unit_of_work=uow,
        source_artifact=SOURCE,
    )

    assert [result.quarantine_code for result in results] == [None, "NAIVE_OBSERVED_AT"]
    assert len(uow.registry.artifacts) == 1
    artifact = uow.registry.artifacts[0]
    assert artifact.uri == SOURCE.source_id
    assert artifact.sha256 == SOURCE.content_hash
    assert len(uow.registry.runs) == 1
    assert {item.status for item in uow.registry.items} == {"IMPORTED", "QUARANTINED"}
    assert len(uow.registry.quarantines) == 1
    quarantine = uow.registry.quarantines[0]
    assert quarantine.record_locator == SOURCE.record_locator
    assert quarantine.error_code == "NAIVE_OBSERVED_AT"
    assert quarantine.raw_payload["source_record_id"] == "registered-bad"
    assert quarantine.raw_text
    assert uow.commits == 0
    assert uow.rollbacks == 0
    assert uow.registry.runs[-1].status == "COMPLETED"
    assert uow.registry.runs[-1].counts == {
        "attempted": 2, "parsed": 2, "imported": 1, "quarantined": 1, "duplicates": 0
    }

    publish_catalyst_states(
        (valid, quarantined),
        unit_of_work=uow,
        source_artifact=SOURCE,
    )
    assert len(uow.registry.items) == 2
    assert len(uow.registry.quarantines) == 1
    assert len(uow.registry.edges) == 1
    assert uow.registry.runs[-1].counts == {
        "attempted": 2, "parsed": 2, "imported": 0, "quarantined": 0, "duplicates": 2
    }


def test_registry_accounting_deduplicates_same_batch_quarantine_identity() -> None:
    bad = {
        "source": "wire",
        "source_record_id": "same-bad",
        "ticker": "ABC",
        "event_type": "company_news",
        "event_time": _time(13),
        "observed_at": "2026-07-30 13:03:00",
        "available_at": _time(13, 3),
        "headline": "bad row",
    }
    uow = _RecordingUow()

    publish_catalyst_states((bad, dict(bad)), unit_of_work=uow, source_artifact=SOURCE)
    run = uow.registry.runs[-1]

    assert len(uow.registry.items) == 1
    assert len(uow.registry.quarantines) == 1
    assert len(uow.registry.edges) == 0
    assert run.counts == {
        "attempted": 2,
        "parsed": 2,
        "imported": 0,
        "quarantined": 1,
        "duplicates": 1,
    }

    publish_catalyst_states((bad,), unit_of_work=uow, source_artifact=SOURCE)
    assert len(uow.registry.items) == 1
    assert len(uow.registry.quarantines) == 1
    assert uow.registry.runs[-1].counts == {
        "attempted": 1,
        "parsed": 1,
        "imported": 0,
        "quarantined": 0,
        "duplicates": 1,
    }


def test_registry_converges_normalized_logical_feed_lineage_across_calls() -> None:
    row = {
        "source": "wire",
        "source_record_id": "converge-1",
        "ticker": "ABC",
        "event_type": "company_news",
        "event_time": _time(13),
        "observed_at": _time(13, 3),
        "available_at": _time(13, 3),
        "headline": "Convergence",
    }
    first_lineage = LineageRef(
        source_id="  FEED-X  ", content_hash="A" * 64, record_locator="  feed-x:row:1  "
    )
    second_lineage = LineageRef(
        source_id="feed-x", content_hash="a" * 64, record_locator="feed-x:row:1"
    )
    uow = _RecordingUow()

    publish_catalyst_states((row,), unit_of_work=uow, source_artifact=first_lineage)
    second = publish_catalyst_states((row,), unit_of_work=uow, source_artifact=second_lineage)

    assert len(uow.registry.artifacts) == 1
    artifact = uow.registry.artifacts[0]
    assert artifact.uri == "feed-x"
    assert artifact.sha256 == "a" * 64
    assert artifact.source_id == uuid5(NAMESPACE_URL, "catalyst-source:feed-x:" + "a" * 64)
    assert artifact.metadata["record_locator"] == "feed-x:row:1"
    assert len(uow.registry.items) == 1
    assert len(uow.registry.edges) == 1
    assert uow.registry.items[0].source_id == artifact.source_id
    assert uow.registry.edges[0].source_id == artifact.source_id
    assert second[0].event is not None
    assert uow.registry.runs[-1].counts == {
        "attempted": 1,
        "parsed": 1,
        "imported": 0,
        "quarantined": 0,
        "duplicates": 1,
    }


def test_catalyst_id_only_change_does_not_change_event_or_state_identity() -> None:
    row = {
        "source": "wire",
        "source_record_id": "stable-source-id",
        "ticker": "ABC",
        "event_type": "company_news",
        "event_time": _time(13),
        "observed_at": _time(13, 3),
        "available_at": _time(13, 3),
        "headline": "stable identity",
    }
    first = normalize_catalyst_record(dict(row, catalyst_id="local-a"), source_artifact=SOURCE).event
    second = normalize_catalyst_record(dict(row, catalyst_id="local-b"), source_artifact=SOURCE).event

    assert first is not None and second is not None
    assert first.event_id == second.event_id
    assert first.state_id == second.state_id


def test_downstream_news_label_stage_keeps_provider_and_revision_identities(tmp_path) -> None:
    from signals.news.pipeline import label_news_forward_returns

    frame = pd.DataFrame(
        [
            {"source": "provider-a", "source_record_id": "same-id", "ticker": "ABC", "timestamp": "2026-07-30T13:00:00Z", "headline": "same"},
            {"source": "provider-b", "source_record_id": "same-id", "ticker": "ABC", "timestamp": "2026-07-30T13:00:00Z", "headline": "same"},
            {"source": "provider-a", "source_record_id": "same-id", "ticker": "ABC", "timestamp": "2026-07-30T13:00:00Z", "headline": "revised"},
        ]
    )
    news_path = tmp_path / "news.parquet"
    labels_path = tmp_path / "labels.parquet"
    records = __import__("signals.news.schema", fromlist=["records_from_frame"]).records_from_frame(frame)
    records.to_parquet(news_path, index=False)
    bars = pd.DataFrame(
        {"ticker": ["ABC"] * 3, "timestamp": pd.date_range("2026-07-30 14:00", periods=3, freq="h", tz="UTC"), "close": [100.0, 101.0, 102.0]}
    )

    out = label_news_forward_returns(news_path, bars, bars_per_day=1, output_path=labels_path, incremental=False)
    assert len(out) == 3
    assert len(set(out["record_id"])) == 3


def test_no_uow_publication_returns_quarantine_results_without_database_access() -> None:
    result = publish_catalyst_states(
        (
            {
                "source": "wire",
                "source_record_id": "no-uow-good",
                "ticker": "ABC",
                "event_type": "company_news",
                "event_time": _time(13),
                "observed_at": _time(13, 3),
                "available_at": _time(13, 3),
            },
            {
                "source": "wire",
                "source_record_id": "no-uow-bad",
                "ticker": "ABC",
                "event_type": "company_news",
                "event_time": _time(13),
                "observed_at": "2026-07-30 13:03:00",
                "available_at": _time(13, 3),
            },
        ),
        unit_of_work=None,
        source_artifact=SOURCE,
    )

    assert len(result) == 2
    assert result[0].event is not None
    assert result[1].event is None
    assert result[1].quarantine_code == "NAIVE_OBSERVED_AT"


def test_collector_parquet_round_trip_preserves_malformed_rows_for_both_publish_modes(tmp_path) -> None:
    news_input = tmp_path / "news.csv"
    news_path = tmp_path / "news.parquet"
    pd.DataFrame(
        [
            {
                "source": "provider-a",
                "source_record_id": "news-valid",
                "ticker": "ABC",
                "timestamp": "2026-07-30T13:00:00Z",
                "observed_at": "2026-07-30T13:03:00Z",
                "available_at": "2026-07-30T13:03:00Z",
                "headline": "valid",
            },
            {
                "source": "provider-a",
                "source_record_id": "news-missing",
                "ticker": "ABC",
                "timestamp": "2026-07-30T13:01:00Z",
                "observed_at": "2026-07-30 13:03:00",
            },
        ]
    ).to_csv(news_input, index=False)
    news = collect_news_from_csv(news_input, output_path=news_path)
    news_round_trip = pd.read_parquet(news_path)
    news_records = news_to_catalysts(news_round_trip)
    malformed_news = news_records.loc[
        news_records["ingestion_quarantine_code"].eq("MISSING_REQUIRED_NEWS_FIELD")
    ].iloc[0]
    malformed_news_raw = json.loads(malformed_news["raw_ingestion_fields"])
    assert malformed_news_raw["observed_at"] == "2026-07-30 13:03:00"
    assert malformed_news_raw["timestamp"] == "2026-07-30T13:01:00Z"

    event_input = tmp_path / "events.csv"
    event_path = tmp_path / "events.parquet"
    pd.DataFrame(
        [
            {"event_type": "cpi", "timestamp": "2026-08-13T13:30:00Z", "observed_at": "2026-07-30T14:00:00Z", "title": "CPI"},
            {"event_type": "fomc", "timestamp": "not-a-time", "observed_at": "2026-07-30 14:00:00", "title": "FOMC"},
            {"event_type": "cpi", "timestamp": "2026-08-13T14:30:00Z", "observed_at": "2026-07-30 14:00:00", "title": "CPI naive metadata"},
        ]
    ).to_csv(event_input, index=False)
    events = collect_macro_events(event_input, output_path=event_path)
    event_round_trip = pd.read_parquet(event_path)
    event_records = scheduled_events_to_catalysts(event_round_trip, default_kind="scheduled_event")
    malformed_event = event_records.loc[
        event_records["ingestion_quarantine_code"].eq("INVALID_EVENT_TIME")
    ].iloc[0]
    malformed_event_raw = json.loads(malformed_event["raw_ingestion_fields"])
    assert malformed_event["ingestion_quarantine_message"]
    assert malformed_event["event_time"] == "not-a-time"
    assert malformed_event_raw["timestamp"] == "not-a-time"
    assert malformed_event_raw["observed_at"] == "2026-07-30 14:00:00"

    combined = pd.concat([news_records, event_records], ignore_index=True, sort=False)
    no_uow = publish_catalyst_states(combined.to_dict(orient="records"), source_artifact=SOURCE)
    recording = _RecordingUow()
    with_uow = publish_catalyst_states(
        combined.to_dict(orient="records"), unit_of_work=recording, source_artifact=SOURCE
    )

    assert len(no_uow) == len(with_uow) == 5
    assert {result.quarantine_code for result in no_uow if result.event is None} >= {
        "MISSING_REQUIRED_NEWS_FIELD", "INVALID_EVENT_TIME", "NAIVE_OBSERVED_AT"
    }
    assert len(recording.registry.quarantines) == 3
    assert all(quarantine.raw_text for quarantine in recording.registry.quarantines)
    quarantines = {
        quarantine.error_code: quarantine for quarantine in recording.registry.quarantines
    }
    news_raw_payload = json.loads(
        quarantines["MISSING_REQUIRED_NEWS_FIELD"].raw_payload["raw_ingestion_fields"]
    )
    event_raw_payload = json.loads(
        quarantines["INVALID_EVENT_TIME"].raw_payload["raw_ingestion_fields"]
    )
    assert news_raw_payload["observed_at"] == "2026-07-30 13:03:00"
    assert event_raw_payload["timestamp"] == "not-a-time"
    assert quarantines["NAIVE_OBSERVED_AT"].raw_payload["observed_at"] == (
        "2026-07-30 14:00:00"
    )
    assert recording.commits == recording.rollbacks == 0


class _RecordingStates:
    def __init__(self) -> None:
        self.saved = []

    def insert_states_idempotently(self, states):
        self.saved.extend(states)
        return {state.state_id: state.state_id for state in states}


class _RecordingRegistry:
    def __init__(self) -> None:
        self.artifacts = []
        self.runs = []
        self.items = []
        self.quarantines = []
        self.edges = []
        self.seen = set()

    def save_source_artifact(self, artifact):
        for existing in self.artifacts:
            if existing.source_id == artifact.source_id:
                return existing
        self.artifacts.append(artifact)
        return artifact

    def save_import_run(self, run):
        self.runs.append(run)
        return run

    def insert_import_items_if_absent(self, items):
        inserted = {
            (item.source_id, item.record_locator, item.importer_version, item.normalized_hash)
            for item in items
            if (
                item.source_id,
                item.record_locator,
                item.importer_version,
                item.normalized_hash,
            ) not in self.seen
        }
        self.seen.update(inserted)
        self.items.extend(
            item
            for item in items
            if (
                item.source_id,
                item.record_locator,
                item.importer_version,
                item.normalized_hash,
            ) in inserted
        )
        return inserted

    def save_import_quarantines(self, quarantines):
        self.quarantines.extend(quarantines)

    def save_lineage_edges(self, edges):
        self.edges.extend(edges)

    def finish_import_run(self, import_run_id, *, finished_at, status, counts):
        run = next(run for run in self.runs if run.import_run_id == import_run_id)
        index = self.runs.index(run)
        self.runs[index] = run.__class__(
            import_run_id=run.import_run_id,
            importer_version=run.importer_version,
            started_at=run.started_at,
            finished_at=finished_at,
            status=status,
            counts=counts,
        )
        return self.runs[index]


class _RecordingUow:
    def __init__(self) -> None:
        self.states = _RecordingStates()
        self.registry = _RecordingRegistry()
        self.commits = 0
        self.rollbacks = 0

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def publish_pressure_for_test(event, *, score: float):
    uow = _RecordingUow()
    row = {
        "source": event.source,
        "source_record_id": "pressure-1",
        "ticker": event.ticker,
        "event_type": event.event_type,
        "event_time": event.event_time,
        "published_at": event.published_at,
        "observed_at": event.observed_at,
        "available_at": event.available_at,
        "score": score,
    }
    publish_catalyst_states(
        (row,),
        unit_of_work=uow,
        source_artifact=SOURCE,
        decision_time=event.available_at,
        valid_until=event.available_at.replace(hour=16),
    )
    assert uow.commits == 0
    assert uow.rollbacks == 0
    return next(state for state in uow.states.saved if state.state_type.value == "CATALYST_PRESSURE")
