"""Causal timestamp semantics for catalyst state adaptation."""

from __future__ import annotations

from datetime import datetime, timezone

from core.nervous_system.contracts.quality import LineageRef
import pandas as pd

from signals.catalysts.pipeline import build_catalyst_records
from signals.catalysts.nervous_system_adapter import (
    aggregate_catalyst_pressure,
    normalize_catalyst_record,
    normalize_catalyst_records,
    publish_catalyst_states,
)


UTC = timezone.utc
SOURCE = LineageRef(
    source_id="wire-feed",
    content_hash="a" * 64,
    record_locator="wire-feed:record:1",
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


class _RecordingStates:
    def __init__(self) -> None:
        self.saved = []

    def insert_states_idempotently(self, states):
        self.saved.extend(states)
        return {state.state_id: state.state_id for state in states}


class _RecordingUow:
    def __init__(self) -> None:
        self.states = _RecordingStates()
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
