from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from core.nervous_system.contracts.enums import StateType, ThemeRegime
from core.nervous_system.contracts.quality import LineageRef
from core.nervous_system.contracts.states import ThemeState
from themes.dynamic_theme.nervous_system_adapter import (
    adapt_theme_states,
    persist_theme_states,
)
from themes.dynamic_theme.stages import step08_memberships as step08


UTC = timezone.utc
AS_OF = date(2026, 7, 30)
AVAILABLE_AT = datetime(2026, 7, 30, 20, 5, tzinfo=UTC)
GENERATED_AT = datetime(2026, 7, 30, 20, 6, tzinfo=UTC)
VALID_UNTIL = datetime(2026, 7, 31, 20, 5, tzinfo=UTC)
ARTIFACT_HASH = "a" * 64


def _membership_frame(
    *,
    as_of: date = AS_OF,
    taxonomy_version: str = "taxonomy-v1",
    scores: tuple[tuple[str, str, float], ...] = (
        ("AAA", "alpha_theme", 0.82),
        ("BBB", "alpha_theme", 0.61),
    ),
) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {"ticker": ticker, "theme": theme, "membership_score": score, "date": as_of}
            for ticker, theme, score in scores
        ]
    )
    frame.attrs["taxonomy_version"] = taxonomy_version
    return frame


def _history_frame(
    *,
    as_of: date = AS_OF,
    taxonomy_version: str = "taxonomy-v1",
    available_at: datetime = AVAILABLE_AT,
    generated_at: datetime = GENERATED_AT,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "as_of": as_of,
                "available_at": available_at,
                "generated_at": generated_at,
                "ticker": "AAA",
                "theme": "alpha_theme",
                "membership_score": 0.82,
                "taxonomy_version": taxonomy_version,
                "producer_version": "dynamic-theme@1",
            }
        ]
    )


def _features_frame(*, as_of: date = AS_OF) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "BBB",
                "date": as_of,
                "primary_theme": "alpha_theme",
                "theme_breadth": 0.75,
                "theme_heat_score": 0.20,
                "theme_crowding": 0.40,
                "theme_persistence": 0.80,
            },
            {
                "ticker": "AAA",
                "date": as_of,
                "primary_theme": "alpha_theme",
                "theme_breadth": 0.75,
                "theme_heat_score": 0.20,
                "theme_crowding": 0.40,
                "theme_persistence": 0.80,
            },
        ]
    )


def _lineage() -> tuple[LineageRef, ...]:
    return (
        LineageRef(
            source_id="ticker_theme_membership_history.parquet",
            content_hash=ARTIFACT_HASH,
            record_locator="ticker_theme_membership_history:row:17",
        ),
        LineageRef(
            source_id="ticker_theme_features.parquet",
            content_hash="b" * 64,
            record_locator="ticker_theme_features:row:4",
        ),
    )


def test_history_preserves_days_and_same_taxonomy_reruns(tmp_path):
    history_path = tmp_path / "ticker_theme_membership_history.parquet"
    day_one = _membership_frame(as_of=date(2026, 7, 29))
    day_two = _membership_frame(as_of=AS_OF)

    first = step08.append_membership_history(
        day_one,
        history_path=history_path,
        as_of=date(2026, 7, 29),
        generated_at=GENERATED_AT - timedelta(days=1),
    )
    second = step08.append_membership_history(
        day_two,
        history_path=history_path,
        as_of=AS_OF,
        generated_at=GENERATED_AT,
    )
    same_run = step08.append_membership_history(
        day_two,
        history_path=history_path,
        as_of=AS_OF,
        generated_at=GENERATED_AT,
    )
    rerun = step08.append_membership_history(
        _membership_frame(as_of=AS_OF, scores=(("AAA", "alpha_theme", 0.11), ("BBB", "alpha_theme", 0.22))),
        history_path=history_path,
        as_of=AS_OF,
        generated_at=GENERATED_AT + timedelta(minutes=4),
    )

    assert set(first["as_of"]) == {date(2026, 7, 29)}
    assert set(second["as_of"]) == {date(2026, 7, 29), AS_OF}
    pd.testing.assert_frame_equal(second, same_run)
    keys = ["as_of", "ticker", "theme", "taxonomy_version"]
    assert not rerun.duplicated(keys).any()
    preserved = rerun[(rerun["as_of"] == AS_OF) & (rerun["ticker"] == "AAA")]
    assert preserved["membership_score"].tolist() == [0.82]
    assert rerun["available_at"].dt.tz is not None
    assert rerun["generated_at"].dt.tz is not None


def test_history_revised_taxonomy_is_new_immutable_evidence(tmp_path):
    history_path = tmp_path / "history.parquet"
    step08.append_membership_history(
        _membership_frame(taxonomy_version="taxonomy-v1"),
        history_path=history_path,
        as_of=AS_OF,
        generated_at=GENERATED_AT,
    )
    revised = step08.append_membership_history(
        _membership_frame(taxonomy_version="taxonomy-v2", scores=(("AAA", "alpha_theme", 0.11),)),
        history_path=history_path,
        as_of=AS_OF,
        generated_at=GENERATED_AT + timedelta(minutes=1),
    )

    assert set(revised["taxonomy_version"]) == {"taxonomy-v1", "taxonomy-v2"}
    original = revised[revised["taxonomy_version"] == "taxonomy-v1"]
    assert original.loc[original["ticker"] == "AAA", "membership_score"].item() == 0.82


def test_history_has_exact_schema_and_uses_atomic_same_directory_replace(tmp_path, monkeypatch):
    history_path = tmp_path / "history.parquet"
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def record_replace(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        replacements.append((source_path, destination_path))
        return real_replace(source, destination)

    monkeypatch.setattr(step08.os, "replace", record_replace)
    monkeypatch.setattr(step08, "_utc_now", lambda: AVAILABLE_AT)
    out = step08.append_membership_history(
        _membership_frame(),
        history_path=history_path,
        as_of=AS_OF,
        generated_at=GENERATED_AT,
    )

    assert list(out.columns) == [
        "as_of",
        "available_at",
        "generated_at",
        "ticker",
        "theme",
        "membership_score",
        "taxonomy_version",
        "producer_version",
    ]
    assert replacements
    source, destination = replacements[-1]
    assert source.parent == history_path.parent
    assert destination == history_path
    assert not list(tmp_path.glob("*.tmp"))


def test_compatibility_view_remains_latest_current_view_and_history_is_separate(tmp_path, monkeypatch):
    current_path = tmp_path / "ticker_theme_membership.parquet"
    history_path = tmp_path / "ticker_theme_membership_history.parquet"
    monkeypatch.setattr(step08, "TICKER_MEMBERSHIP_PATH", current_path)
    monkeypatch.setattr(step08, "TICKER_MEMBERSHIP_HISTORY_PATH", history_path)
    monkeypatch.setattr(step08, "ensure_outputs", lambda: None)
    monkeypatch.setattr(step08, "_utc_now", lambda: AVAILABLE_AT)

    embeddings = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
        }
    )
    clusters = pd.DataFrame({"ticker": ["AAA", "BBB"], "cluster_id": [0, 1]})
    registry = pd.DataFrame(
        {
            "cluster_id": [0, 1],
            "theme_name": ["alpha_theme", "beta_theme"],
        }
    )
    current = step08.compute_memberships(
        embeddings_df=embeddings,
        clusters_df=clusters,
        registry_df=registry,
        as_of=pd.Timestamp(AS_OF),
        generated_at=GENERATED_AT,
    )

    persisted_current = pd.read_parquet(current_path)
    persisted_history = pd.read_parquet(history_path)
    assert list(persisted_current.columns) == ["ticker", "theme", "membership_score", "date"]
    pd.testing.assert_frame_equal(
        persisted_current.sort_values(list(persisted_current.columns)).reset_index(drop=True),
        current.sort_values(list(current.columns)).reset_index(drop=True),
    )
    assert list(persisted_history.columns) == [
        "as_of",
        "available_at",
        "generated_at",
        "ticker",
        "theme",
        "membership_score",
        "taxonomy_version",
        "producer_version",
    ]
    assert persisted_history["available_at"].dt.tz is not None
    assert (persisted_history["available_at"] > pd.Timestamp(AS_OF, tz="UTC")).all()
    assert (persisted_history["available_at"] != persisted_history["generated_at"]).all()


def test_theme_state_preserves_sorted_scores_metrics_and_exact_lineage():
    states = adapt_theme_states(
        _history_frame().iloc[[0]].copy().assign(
            ticker="BBB", membership_score=0.61
        ).pipe(
            lambda first: pd.concat(
                [
                    first,
                    _history_frame().assign(ticker="AAA", membership_score=0.82),
                ],
                ignore_index=True,
            )
        ),
        _features_frame(),
        available_at=AVAILABLE_AT,
        valid_until=VALID_UNTIL,
        taxonomy_version="taxonomy-v1",
        lineage=_lineage(),
    )

    assert len(states) == 1
    state = states[0]
    assert isinstance(state, ThemeState)
    assert state.state_type is StateType.THEME
    assert state.theme_regime is ThemeRegime.UNKNOWN
    assert list(state.membership_scores) == ["AAA", "BBB"]
    assert state.membership_scores == {"AAA": 0.82, "BBB": 0.61}
    assert state.breadth == 0.75
    assert state.momentum == 0.20
    assert state.crowding == 0.40
    assert state.persistence == 0.80
    assert state.transition_probabilities == {}
    lineage_ids = [json.loads(value) for value in state.lineage_ids]
    assert lineage_ids[0]["content_hash"] == ARTIFACT_HASH
    assert lineage_ids[0]["record_locator"] == "ticker_theme_membership_history:row:17"
    assert state.available_at == AVAILABLE_AT
    assert state.generated_at == GENERATED_AT


def test_theme_state_uses_quality_warning_when_features_are_missing():
    states = adapt_theme_states(
        _history_frame(),
        pd.DataFrame(),
        available_at=AVAILABLE_AT,
        valid_until=VALID_UNTIL,
        taxonomy_version="taxonomy-v1",
        lineage=_lineage(),
    )

    assert len(states) == 1
    assert states[0].membership_scores == {"AAA": 0.82}
    assert states[0].data_quality.issues
    assert all(issue.severity.value == "WARNING" for issue in states[0].data_quality.issues)


def test_theme_state_uses_quality_warning_when_memberships_are_missing():
    states = adapt_theme_states(
        pd.DataFrame(columns=_history_frame().columns),
        _features_frame(),
        available_at=AVAILABLE_AT,
        valid_until=VALID_UNTIL,
        taxonomy_version="taxonomy-v1",
        lineage=_lineage(),
    )

    assert len(states) == 1
    assert states[0].membership_scores == {}
    assert any(issue.code == "THEME_MEMBERSHIPS_MISSING" for issue in states[0].data_quality.issues)


def test_theme_state_rejects_missing_or_synthetic_lineage():
    with pytest.raises(ValueError, match="lineage"):
        adapt_theme_states(
            _history_frame(),
            _features_frame(),
            available_at=AVAILABLE_AT,
            valid_until=VALID_UNTIL,
            taxonomy_version="taxonomy-v1",
            lineage=(),
        )
    with pytest.raises(ValueError, match="SHA-256"):
        adapt_theme_states(
            _history_frame(),
            _features_frame(),
            available_at=AVAILABLE_AT,
            valid_until=VALID_UNTIL,
            taxonomy_version="taxonomy-v1",
            lineage=(LineageRef(source_id="x", content_hash="synthetic", record_locator="row:0"),),
        )


def test_theme_state_ids_are_deterministic_and_taxonomy_revision_is_new_evidence():
    kwargs = {
        "available_at": AVAILABLE_AT,
        "valid_until": VALID_UNTIL,
        "lineage": _lineage(),
    }
    first = adapt_theme_states(_history_frame(taxonomy_version="taxonomy-v1"), _features_frame(), taxonomy_version="taxonomy-v1", **kwargs)
    rerun = adapt_theme_states(_history_frame(taxonomy_version="taxonomy-v1"), _features_frame(), taxonomy_version="taxonomy-v1", **kwargs)
    revised = adapt_theme_states(_history_frame(taxonomy_version="taxonomy-v2"), _features_frame(), taxonomy_version="taxonomy-v2", **kwargs)

    assert first[0].state_id == rerun[0].state_id
    assert first[0].state_id != revised[0].state_id


class _FakeStates:
    def __init__(self, *, error: Exception | None = None):
        self.error = error
        self.inserted = []

    def insert_states_idempotently(self, states):
        if self.error:
            raise self.error
        self.inserted.extend(states)
        return {state.state_id: state.state_id for state in states}


class _FakeUow:
    def __init__(self, states: _FakeStates):
        self.states = states
        self.commit_called = False
        self.rollback_called = False

    def commit(self):
        self.commit_called = True

    def rollback(self):
        self.rollback_called = True


def test_optional_publication_uses_caller_owned_uow_without_transaction_ownership():
    states_repo = _FakeStates()
    uow = _FakeUow(states_repo)
    count = persist_theme_states(
        _history_frame(),
        _features_frame(),
        unit_of_work=uow,
        available_at=AVAILABLE_AT,
        valid_until=VALID_UNTIL,
        taxonomy_version="taxonomy-v1",
        lineage=_lineage(),
    )

    assert count == 1
    assert len(states_repo.inserted) == 1
    assert not uow.commit_called
    assert not uow.rollback_called


def test_optional_publication_propagates_persistence_failure():
    uow = _FakeUow(_FakeStates(error=RuntimeError("persistence failed")))
    with pytest.raises(RuntimeError, match="persistence failed"):
        persist_theme_states(
            _history_frame(),
            _features_frame(),
            unit_of_work=uow,
            available_at=AVAILABLE_AT,
            valid_until=VALID_UNTIL,
            taxonomy_version="taxonomy-v1",
            lineage=_lineage(),
        )
    assert not uow.commit_called
    assert not uow.rollback_called


def test_new_membership_state_type_is_distinct_from_theme_aggregate():
    from core.nervous_system.contracts.states import ThemeMembership

    assert StateType.THEME_MEMBERSHIP is not StateType.THEME
    assert ThemeMembership.model_fields["state_type"].default is StateType.THEME_MEMBERSHIP


def test_pipeline_publishes_only_after_artifacts_and_propagates_failure(monkeypatch, tmp_path):
    import themes.dynamic_theme.pipeline as pipeline

    events: list[str] = []
    history_path = tmp_path / "membership_history.parquet"
    features_path = tmp_path / "features.parquet"
    monkeypatch.setattr(pipeline, "TICKER_MEMBERSHIP_HISTORY_PATH", history_path)
    monkeypatch.setattr(pipeline, "TICKER_THEME_FEATURES_PATH", features_path)

    monkeypatch.setattr(pipeline, "ensure_outputs", lambda: events.append("outputs"))
    monkeypatch.setattr(
        pipeline,
        "build_ticker_documents",
        lambda *args, **kwargs: events.append("documents") or pd.DataFrame(),
    )
    monkeypatch.setattr(
        pipeline,
        "generate_embeddings",
        lambda *args, **kwargs: events.append("embeddings") or pd.DataFrame(),
    )

    def write_memberships(*args, **kwargs):
        events.append("membership_parquet")
        _history_frame().to_parquet(history_path, index=False)
        return _membership_frame()

    def write_features(*args, **kwargs):
        events.append("feature_parquet")
        frame = _features_frame()
        frame.to_parquet(features_path, index=False)
        return frame

    def fail_publication(*args, **kwargs):
        events.append("publish")
        raise RuntimeError("persistence failed after parquet")

    monkeypatch.setattr(pipeline, "compute_memberships", write_memberships)
    monkeypatch.setattr(pipeline, "build_meta_features", write_features)
    monkeypatch.setattr(pipeline, "persist_theme_states", fail_publication)

    with pytest.raises(RuntimeError, match="persistence failed after parquet"):
        pipeline.daily_run(
            as_of=pd.Timestamp(AS_OF, tz="UTC"),
            tickers=["AAA"],
            unit_of_work=object(),
            valid_until_for=lambda available: available + timedelta(days=1),
        )
    assert events[-3:] == ["membership_parquet", "feature_parquet", "publish"]


def test_pipeline_research_run_does_not_publish_without_uow(monkeypatch):
    import themes.dynamic_theme.pipeline as pipeline

    events: list[str] = []
    monkeypatch.setattr(pipeline, "ensure_outputs", lambda: None)
    monkeypatch.setattr(pipeline, "build_ticker_documents", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(pipeline, "generate_embeddings", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(pipeline, "compute_memberships", lambda *a, **k: _membership_frame())
    monkeypatch.setattr(pipeline, "build_meta_features", lambda *a, **k: _features_frame())
    monkeypatch.setattr(pipeline, "persist_theme_states", lambda *a, **k: events.append("publish"))

    pipeline.daily_run(as_of=pd.Timestamp(AS_OF, tz="UTC"), tickers=["AAA"])

    assert events == []
