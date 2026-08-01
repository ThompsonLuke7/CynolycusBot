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
GENERATED_AT = datetime(2026, 7, 30, 20, 4, tzinfo=UTC)
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


def _lineage_from_root(root: str, *, content_hash: str = ARTIFACT_HASH, locator: str = "row:17") -> tuple[LineageRef, ...]:
    return (
        LineageRef(
            source_id=f"{root}/ticker_theme_membership_history.parquet",
            content_hash=content_hash,
            record_locator=locator,
        ),
        LineageRef(
            source_id=f"{root}/ticker_theme_features.parquet",
            content_hash="b" * 64,
            record_locator="row:4",
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
    assert any(issue.code == "MISSING_MEMBERSHIPS" for issue in states[0].data_quality.issues)


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


def test_theme_state_identity_excludes_completion_time_and_local_source_path():
    first = adapt_theme_states(
        _history_frame(),
        _features_frame(),
        available_at=AVAILABLE_AT,
        valid_until=VALID_UNTIL,
        taxonomy_version="taxonomy-v1",
        lineage=_lineage_from_root("/worktree/one"),
    )[0]
    rerun = adapt_theme_states(
        _history_frame(),
        _features_frame(),
        available_at=AVAILABLE_AT + timedelta(minutes=1),
        valid_until=VALID_UNTIL + timedelta(minutes=1),
        taxonomy_version="taxonomy-v1",
        lineage=_lineage_from_root("/worktree/two"),
    )[0]
    revised_artifact = adapt_theme_states(
        _history_frame(),
        _features_frame(),
        available_at=AVAILABLE_AT + timedelta(minutes=1),
        valid_until=VALID_UNTIL + timedelta(minutes=1),
        taxonomy_version="taxonomy-v1",
        lineage=_lineage_from_root("/worktree/two", content_hash="c" * 64),
    )[0]
    revised_locator = adapt_theme_states(
        _history_frame(),
        _features_frame(),
        available_at=AVAILABLE_AT + timedelta(minutes=1),
        valid_until=VALID_UNTIL + timedelta(minutes=1),
        taxonomy_version="taxonomy-v1",
        lineage=_lineage_from_root("/worktree/two", locator="row:18"),
    )[0]

    assert first.state_id == rerun.state_id
    assert first.available_at != rerun.available_at
    assert first.state_id != revised_artifact.state_id
    assert first.state_id != revised_locator.state_id


def test_raw_theme_labels_remain_distinct_and_membership_output_preserves_them(tmp_path, monkeypatch):
    monkeypatch.setattr(step08, "TICKER_MEMBERSHIP_PATH", tmp_path / "current.parquet")
    monkeypatch.setattr(step08, "TICKER_MEMBERSHIP_HISTORY_PATH", tmp_path / "history.parquet")
    monkeypatch.setattr(step08, "ensure_outputs", lambda: None)
    monkeypatch.setattr(step08, "_utc_now", lambda: AVAILABLE_AT)
    embeddings = pd.DataFrame(
        {"ticker": ["AAA", "BBB"], "embedding": [[1.0, 0.0], [0.0, 1.0]]}
    )
    clusters = pd.DataFrame({"ticker": ["AAA", "BBB"], "cluster_id": [0, 1]})
    registry = pd.DataFrame(
        {"cluster_id": [0, 1], "theme_name": ["AI & ML", "AI/ML"]}
    )

    output = step08.compute_memberships(
        embeddings_df=embeddings,
        clusters_df=clusters,
        registry_df=registry,
        as_of=pd.Timestamp(AS_OF),
        generated_at=GENERATED_AT,
    )

    assert set(output["theme"]) == {"AI & ML", "AI/ML"}
    assert set(pd.read_parquet(tmp_path / "history.parquet")["theme"]) == {
        "AI & ML",
        "AI/ML",
    }
    assert step08.canonical_theme_id("AI & ML") != step08.canonical_theme_id("AI/ML")


def test_schema_less_empty_memberships_emit_feature_identified_warning_state():
    states = adapt_theme_states(
        pd.DataFrame(),
        _features_frame(),
        available_at=AVAILABLE_AT,
        valid_until=VALID_UNTIL,
        taxonomy_version="taxonomy-v1",
        lineage=_lineage(),
    )

    assert len(states) == 1
    assert states[0].membership_scores == {}
    assert any(issue.code == "MISSING_MEMBERSHIPS" for issue in states[0].data_quality.issues)


def test_history_rejects_generated_at_after_actual_completion(monkeypatch, tmp_path):
    monkeypatch.setattr(step08, "_utc_now", lambda: AVAILABLE_AT)

    with pytest.raises(ValueError, match="generated_at"):
        step08.append_membership_history(
            _membership_frame(),
            history_path=tmp_path / "history.parquet",
            as_of=AS_OF,
            generated_at=AVAILABLE_AT + timedelta(seconds=1),
        )


def test_history_rejects_naive_existing_availability_timestamp(monkeypatch, tmp_path):
    monkeypatch.setattr(step08, "_utc_now", lambda: AVAILABLE_AT)
    existing = _history_frame()
    existing["available_at"] = existing["available_at"].dt.tz_localize(None)
    history_path = tmp_path / "history.parquet"
    existing.to_parquet(history_path, index=False)

    with pytest.raises(ValueError, match="timezone-aware"):
        step08.append_membership_history(
            _membership_frame(),
            history_path=history_path,
            as_of=AS_OF,
            generated_at=GENERATED_AT,
        )


def test_history_captures_availability_after_existing_history_normalization(
    monkeypatch,
    tmp_path,
):
    history_path = tmp_path / "history.parquet"
    _history_frame().to_parquet(history_path, index=False)
    completion = AVAILABLE_AT + timedelta(minutes=10)
    events: list[str] = []
    real_normalize = step08._normalize_existing_history
    real_replace = os.replace

    def normalize_existing(path):
        events.append("history_normalized")
        return real_normalize(path)

    def capture_completion():
        events.append("availability_captured")
        return completion

    def replace(source, destination):
        events.append("history_replaced")
        return real_replace(source, destination)

    monkeypatch.setattr(step08, "_normalize_existing_history", normalize_existing)
    monkeypatch.setattr(step08, "_utc_now", capture_completion)
    monkeypatch.setattr(step08.os, "replace", replace)

    out = step08.append_membership_history(
        _membership_frame(taxonomy_version="taxonomy-v2"),
        history_path=history_path,
        as_of=AS_OF,
        generated_at=GENERATED_AT,
    )

    assert events == [
        "history_normalized",
        "availability_captured",
        "history_replaced",
    ]
    new_rows = out[out["taxonomy_version"] == "taxonomy-v2"]
    assert set(new_rows["available_at"]) == {pd.Timestamp(completion)}


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
    assert history_path.exists()
    assert features_path.exists()


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


def test_pipeline_uses_feature_completion_for_effective_availability(monkeypatch, tmp_path):
    import themes.dynamic_theme.pipeline as pipeline

    history_path = tmp_path / "membership_history.parquet"
    features_path = tmp_path / "features.parquet"
    feature_completion = AVAILABLE_AT + timedelta(minutes=5)
    captured: dict[str, object] = {}
    monkeypatch.setattr(pipeline, "TICKER_MEMBERSHIP_HISTORY_PATH", history_path)
    monkeypatch.setattr(pipeline, "TICKER_THEME_FEATURES_PATH", features_path)
    monkeypatch.setattr(pipeline, "ensure_outputs", lambda: None)
    monkeypatch.setattr(pipeline, "build_ticker_documents", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(pipeline, "generate_embeddings", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(
        pipeline,
        "compute_memberships",
        lambda *a, **k: (lambda frame: (frame.to_parquet(history_path, index=False), _membership_frame())[1])(_history_frame()),
    )

    def write_features(*args, **kwargs):
        frame = _features_frame()
        frame.to_parquet(features_path, index=False)
        return frame

    def capture_publication(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(pipeline, "build_meta_features", write_features)
    monkeypatch.setattr(pipeline, "persist_theme_states", capture_publication)
    monkeypatch.setattr(pipeline, "_utc_now", lambda: feature_completion, raising=False)

    pipeline.daily_run(
        as_of=pd.Timestamp(AS_OF, tz="UTC"),
        tickers=["AAA"],
        unit_of_work=object(),
        valid_until_for=lambda available: available + timedelta(days=1),
    )

    assert captured["available_at"] == feature_completion
    assert captured["valid_until"] == feature_completion + timedelta(days=1)


def test_historical_daily_rerun_publishes_only_its_exact_date_taxonomy_and_lineage(
    monkeypatch,
    tmp_path,
):
    import themes.dynamic_theme.pipeline as pipeline

    history_path = tmp_path / "membership_history.parquet"
    features_path = tmp_path / "features.parquet"
    newer_date = AS_OF + timedelta(days=1)
    represented_as_of = pd.Timestamp("2026-07-30T23:30:00-04:00")
    feature_completion = AVAILABLE_AT + timedelta(days=2)
    history = pd.concat(
        [
            _history_frame(
                as_of=newer_date,
                taxonomy_version="taxonomy-v2",
                generated_at=GENERATED_AT + timedelta(days=1),
                available_at=AVAILABLE_AT + timedelta(days=1),
            ),
            _history_frame(as_of=AS_OF, taxonomy_version="taxonomy-v1"),
        ],
        ignore_index=True,
    )
    features = pd.concat(
        [
            _features_frame(as_of=newer_date),
            _features_frame(as_of=AS_OF),
        ],
        ignore_index=True,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(pipeline, "TICKER_MEMBERSHIP_HISTORY_PATH", history_path)
    monkeypatch.setattr(pipeline, "TICKER_THEME_FEATURES_PATH", features_path)
    monkeypatch.setattr(pipeline, "ensure_outputs", lambda: None)
    monkeypatch.setattr(pipeline, "build_ticker_documents", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(pipeline, "generate_embeddings", lambda *a, **k: pd.DataFrame())

    def write_memberships(*args, **kwargs):
        history.to_parquet(history_path, index=False)
        return _membership_frame(as_of=AS_OF, taxonomy_version="taxonomy-v1")

    def write_features(*args, **kwargs):
        features.to_parquet(features_path, index=False)
        return features

    def capture_publication(memberships, selected_features, **kwargs):
        captured["memberships"] = memberships
        captured["features"] = selected_features
        captured.update(kwargs)

    monkeypatch.setattr(pipeline, "compute_memberships", write_memberships)
    monkeypatch.setattr(pipeline, "build_meta_features", write_features)
    monkeypatch.setattr(pipeline, "persist_theme_states", capture_publication)
    monkeypatch.setattr(pipeline, "_utc_now", lambda: feature_completion)

    pipeline.daily_run(
        as_of=represented_as_of,
        tickers=["AAA"],
        unit_of_work=object(),
        valid_until_for=lambda available: available + timedelta(days=1),
    )

    selected_history = captured["memberships"]
    selected_features = captured["features"]
    assert set(selected_history["as_of"]) == {AS_OF}
    assert set(selected_history["taxonomy_version"]) == {"taxonomy-v1"}
    assert set(selected_features["date"]) == {AS_OF}
    lineage = captured["lineage"]
    assert {ref.record_locator for ref in lineage} == {
        "ticker_theme_membership_history:row:1",
        "ticker_theme_features:row:2",
        "ticker_theme_features:row:3",
    }
    assert {ref.content_hash for ref in lineage if "membership_history" in ref.source_id} == {
        pipeline._artifact_sha256(history_path)
    }
    assert {ref.content_hash for ref in lineage if "features" in ref.source_id} == {
        pipeline._artifact_sha256(features_path)
    }
    assert captured["taxonomy_version"] == "taxonomy-v1"


def test_publication_fails_clearly_without_exact_date_taxonomy_evidence(
    monkeypatch,
    tmp_path,
):
    import themes.dynamic_theme.pipeline as pipeline

    history_path = tmp_path / "membership_history.parquet"
    monkeypatch.setattr(pipeline, "TICKER_MEMBERSHIP_HISTORY_PATH", history_path)
    _history_frame(as_of=AS_OF + timedelta(days=1)).to_parquet(
        history_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="no membership history evidence.*2026-07-30.*taxonomy-v1",
    ):
        pipeline._publish_completed_theme_outputs(
            _membership_frame(),
            pd.DataFrame(),
            unit_of_work=object(),
            valid_until_for=lambda available: available + timedelta(days=1),
            represented_as_of=pd.Timestamp(AS_OF, tz="UTC"),
            feature_completion_at=AVAILABLE_AT,
        )


def test_publication_requires_taxonomy_version_from_current_membership_attrs(
    monkeypatch,
    tmp_path,
):
    import themes.dynamic_theme.pipeline as pipeline

    history_path = tmp_path / "membership_history.parquet"
    features_path = tmp_path / "features.parquet"
    memberships = _membership_frame()
    memberships.attrs.clear()

    monkeypatch.setattr(pipeline, "TICKER_MEMBERSHIP_HISTORY_PATH", history_path)
    monkeypatch.setattr(pipeline, "TICKER_THEME_FEATURES_PATH", features_path)
    monkeypatch.setattr(pipeline, "ensure_outputs", lambda: None)
    monkeypatch.setattr(pipeline, "build_ticker_documents", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(pipeline, "generate_embeddings", lambda *a, **k: pd.DataFrame())

    def write_memberships(*args, **kwargs):
        _history_frame().to_parquet(history_path, index=False)
        return memberships

    def write_features(*args, **kwargs):
        features = _features_frame()
        features.to_parquet(features_path, index=False)
        return features

    monkeypatch.setattr(pipeline, "compute_memberships", write_memberships)
    monkeypatch.setattr(pipeline, "build_meta_features", write_features)
    monkeypatch.setattr(pipeline, "persist_theme_states", lambda *a, **k: None)

    with pytest.raises(ValueError, match="attrs.*taxonomy_version"):
        pipeline.daily_run(
            as_of=pd.Timestamp(AS_OF, tz="UTC"),
            tickers=["AAA"],
            unit_of_work=object(),
            valid_until_for=lambda available: available + timedelta(days=1),
        )


def test_publication_rejects_ambiguous_exact_date_taxonomy_history(
    monkeypatch,
    tmp_path,
):
    import themes.dynamic_theme.pipeline as pipeline

    history_path = tmp_path / "membership_history.parquet"
    features_path = tmp_path / "features.parquet"
    history = pd.concat(
        [
            _history_frame(),
            _history_frame().assign(membership_score=0.11),
        ],
        ignore_index=True,
    )

    monkeypatch.setattr(pipeline, "TICKER_MEMBERSHIP_HISTORY_PATH", history_path)
    monkeypatch.setattr(pipeline, "TICKER_THEME_FEATURES_PATH", features_path)
    monkeypatch.setattr(pipeline, "ensure_outputs", lambda: None)
    monkeypatch.setattr(pipeline, "build_ticker_documents", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(pipeline, "generate_embeddings", lambda *a, **k: pd.DataFrame())

    def write_memberships(*args, **kwargs):
        history.to_parquet(history_path, index=False)
        return _membership_frame()

    def write_features(*args, **kwargs):
        features = _features_frame()
        features.to_parquet(features_path, index=False)
        return features

    monkeypatch.setattr(pipeline, "compute_memberships", write_memberships)
    monkeypatch.setattr(pipeline, "build_meta_features", write_features)
    monkeypatch.setattr(pipeline, "persist_theme_states", lambda *a, **k: None)

    with pytest.raises(ValueError, match="ambiguous"):
        pipeline.daily_run(
            as_of=pd.Timestamp(AS_OF, tz="UTC"),
            tickers=["AAA"],
            unit_of_work=object(),
            valid_until_for=lambda available: available + timedelta(days=1),
        )


def test_weekly_run_passes_its_represented_as_of_to_publication(monkeypatch):
    import themes.dynamic_theme.pipeline as pipeline

    represented_as_of = pd.Timestamp("2026-07-30T23:30:00-04:00")
    registry = pd.DataFrame({"cluster_id": [7], "theme_name": ["alpha_theme"]})
    captured: dict[str, object] = {}

    monkeypatch.setattr(pipeline, "ensure_outputs", lambda: None)
    monkeypatch.setattr(pipeline, "build_ticker_documents", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(pipeline, "generate_embeddings", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(
        pipeline,
        "cluster_tickers",
        lambda *a, **k: pd.DataFrame({"ticker": ["AAA"], "cluster_id": [7]}),
    )
    monkeypatch.setattr(
        pipeline,
        "build_cluster_summaries",
        lambda *a, **k: [{"cluster_id": 7, "tickers": ["AAA"]}],
    )
    monkeypatch.setattr(pipeline, "label_clusters", lambda *a, **k: registry)
    monkeypatch.setattr(pipeline, "discover_new_themes", lambda *a, **k: (0, registry))
    monkeypatch.setattr(pipeline, "compute_active_theme_centroids", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(pipeline, "find_duplicate_groups", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "build_canonical_map", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "build_relationship_graph", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "compute_memberships", lambda *a, **k: _membership_frame())
    monkeypatch.setattr(pipeline, "build_meta_features", lambda *a, **k: _features_frame())
    monkeypatch.setattr(
        pipeline,
        "_publish_completed_theme_outputs",
        lambda *a, **kwargs: captured.update(kwargs),
    )

    pipeline.weekly_run(
        as_of=represented_as_of,
        tickers=["AAA"],
        unit_of_work=object(),
        valid_until_for=lambda available: available + timedelta(days=1),
    )

    assert captured["represented_as_of"] == represented_as_of
