from __future__ import annotations

from datetime import datetime, timezone
import json
from uuid import uuid4

import pytest
from sqlalchemy import select

from core.nervous_system.contracts.enums import StateType
from core.nervous_system.contracts.quality import LineageRef
from core.nervous_system.persistence.models import (
    ImportItem,
    ImportQuarantine,
    ImportRun,
    LineageEdge,
    SourceArtifact,
    StateRecord,
)
from core.nervous_system.persistence.uow import UnitOfWork
from signals.catalysts.nervous_system_adapter import (
    _registry_source_id,
    _revision_hash,
    publish_catalyst_states,
)


UTC = timezone.utc
def _time(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 7, 30, hour, minute, tzinfo=UTC)


def _rows(token: str, *, revised: bool = False) -> tuple[dict[str, object], ...]:
    rows = [
        {
            "source": "wire",
            "source_record_id": "postgres-abc",
            "ticker": "ABC",
            "event_type": "company_news",
            "event_time": _time(13),
            "published_at": _time(13, 1),
            "observed_at": _time(13, 3),
            "available_at": _time(13, 3),
            "headline": "ABC revised" if revised else "ABC original",
            "score": 1.0,
        },
        {
            "source": "wire",
            "source_record_id": "postgres-msft",
            "ticker": "MSFT",
            "event_type": "company_news",
            "event_time": _time(13),
            "published_at": _time(13, 1),
            "observed_at": _time(13, 3),
            "available_at": _time(13, 3),
            "headline": "MSFT original",
            "score": 3.0,
        },
        {
            "source": "calendar",
            "source_record_id": "postgres-market",
            "event_type": "fomc_decision",
            "event_time": _time(20),
            "observed_at": _time(13, 3),
            "available_at": _time(13, 3),
            "headline": "FOMC scheduled",
            "score": -0.2,
        },
        {
            "source": "wire",
            "source_record_id": "postgres-quarantined",
            "ticker": "ABC",
            "event_type": "company_news",
            "event_time": _time(13),
            "observed_at": "2026-07-30 13:03:00",
            "available_at": _time(13, 3),
            "headline": "Naive observation",
        },
    ]
    for row in rows:
        row["record_locator"] = f"task11:postgres:{token}:row:{row['source_record_id']}"
    return tuple(rows + [dict(rows[-1])])


@pytest.mark.postgres
def test_task11_postgres_publication_is_atomic_idempotent_lineaged_and_revision_safe(
    session_factory,
):
    token = uuid4().hex
    source = LineageRef(
        source_id=f"task11-postgres-catalyst-feed-{token}",
        content_hash=token * 2,
        record_locator=f"task11:postgres:{token}:batch",
    )
    config_version = f"task11-postgres-{token}"
    run_ids = []

    def publish(rows, source_artifact=source):
        with UnitOfWork(session_factory) as uow:
            calls: list[str] = []
            original_commit = uow.commit
            original_rollback = uow.rollback

            def tracked_commit() -> None:
                calls.append("commit")
                original_commit()

            def tracked_rollback() -> None:
                calls.append("rollback")
                original_rollback()

            original_save_import_run = uow.registry.save_import_run

            def tracked_save_import_run(run):
                saved = original_save_import_run(run)
                run_ids.append(saved.import_run_id)
                return saved

            uow.registry.save_import_run = tracked_save_import_run

            uow.commit = tracked_commit
            uow.rollback = tracked_rollback
            results = publish_catalyst_states(
                rows,
                unit_of_work=uow,
                source_artifact=source_artifact,
                decision_time=_time(14),
                valid_until=_time(16),
                config_version=config_version,
            )
            assert calls == []
            uow.commit()
            assert calls == ["commit"]
            return results

    first = publish(_rows(token), source.model_copy(update={"source_id": f"  {source.source_id.upper()}  ", "content_hash": source.content_hash.upper(), "record_locator": f" {source.record_locator} "}))
    second = publish(_rows(token))
    revised = publish(_rows(token, revised=True)[:1])

    assert len(first) == len(second) == 5
    assert [result.quarantine_code for result in first] == [None, None, None, "NAIVE_OBSERVED_AT", "NAIVE_OBSERVED_AT"]
    assert [result.quarantine_code for result in second] == [None, None, None, "NAIVE_OBSERVED_AT", "NAIVE_OBSERVED_AT"]
    assert [result.quarantine_code for result in revised] == [None]
    assert revised[0].event is not None

    with session_factory() as session:
        source_rows = session.scalars(
            select(SourceArtifact).where(SourceArtifact.uri == source.source_id, SourceArtifact.sha256 == source.content_hash)
        ).all()
        assert len(source_rows) == 1
        source_id = _registry_source_id(source)
        assert (source_rows[0].source_id, source_rows[0].uri, source_rows[0].sha256) == (
            source_id,
            source.source_id,
            source.content_hash,
        )

        import_items = session.scalars(
            select(ImportItem).where(ImportItem.source_id == source_rows[0].source_id)
        ).all()
        assert len(import_items) == 5
        assert sum(item.status == "QUARANTINED" for item in import_items) == 1
        assert {item.source_id for item in import_items} == {source_id}

        assert len(run_ids) == 3
        runs_by_id = {run.import_run_id: run for run in session.scalars(select(ImportRun).where(ImportRun.import_run_id.in_(run_ids))).all()}
        runs = [runs_by_id[run_id] for run_id in run_ids]
        assert len(runs) == 3
        assert [run.status for run in runs] == ["COMPLETED"] * 3
        assert all(run.started_at is not None and run.finished_at is not None for run in runs)
        assert all(run.started_at.tzinfo is not None and run.finished_at.tzinfo is not None for run in runs)
        assert all(run.started_at.utcoffset() is not None and run.finished_at.utcoffset() is not None for run in runs)
        assert [run.counts for run in runs] == [
            {"attempted": 5, "parsed": 5, "imported": 3, "quarantined": 1, "duplicates": 1},
            {"attempted": 5, "parsed": 5, "imported": 0, "quarantined": 0, "duplicates": 5},
            {"attempted": 1, "parsed": 1, "imported": 1, "quarantined": 0, "duplicates": 0},
        ]

        quarantines = session.scalars(
            select(ImportQuarantine).where(ImportQuarantine.source_id == source_rows[0].source_id)
        ).all()
        assert len(quarantines) == 1
        assert quarantines[0].error_code == "NAIVE_OBSERVED_AT"
        assert quarantines[0].raw_payload["source_record_id"] == "postgres-quarantined"
        assert quarantines[0].record_locator == f"task11:postgres:{token}:row:postgres-quarantined"
        assert quarantines[0].raw_text
        assert quarantines[0].source_id == source_rows[0].source_id

        event_ids = {str(result.event.state_id) for result in first + revised if result.event is not None}
        events = session.scalars(select(StateRecord).where(StateRecord.state_id.in_(event_ids))).all()
        pressures = session.scalars(select(StateRecord).where(StateRecord.producer == "signals.catalysts", StateRecord.state_type == StateType.CATALYST_PRESSURE.value, StateRecord.config_version == config_version)).all()
        assert len(events) == 4
        assert len({state.state_id for state in events}) == 4
        assert len(pressures) == 4
        assert {state.payload["scope_id"] for state in pressures} == {"ABC", "MSFT", "MARKET"}
        assert all(not state.payload.get("transition_probabilities") for state in pressures)
        headlines = {state.payload["headline"] for state in events}
        assert headlines == {"ABC original", "ABC revised", "MSFT original", "FOMC scheduled"}
        event_by_headline = {state.payload["headline"]: state for state in events}
        assert event_by_headline["ABC original"].state_id != event_by_headline["ABC revised"].state_id
        edges = session.scalars(
            select(LineageEdge).where(
                LineageEdge.source_id == source_rows[0].source_id,
                LineageEdge.target_type == "CATALYST_EVENT",
            )
        ).all()
        assert len(edges) == 4
        assert {
            (edge.source_id, edge.target_type, edge.target_id, edge.relationship)
            for edge in edges
        } == {
            (source_id, "CATALYST_EVENT", str(state.state_id), "IMPORTED_AS")
            for state in events
        }
        for state in events:
            assert len(state.payload["lineage_ids"]) == 1
            lineage = json.loads(state.payload["lineage_ids"][0])
            assert "source_record_id" not in state.payload
            assert lineage["source_id"] == source.source_id
            assert lineage["content_hash"] == source.content_hash.lower()
            assert lineage["record_locator"] == {
                "ABC original": f"task11:postgres:{token}:row:postgres-abc",
                "ABC revised": f"task11:postgres:{token}:row:postgres-abc",
                "MSFT original": f"task11:postgres:{token}:row:postgres-msft",
                "FOMC scheduled": f"task11:postgres:{token}:row:postgres-market",
            }[state.payload["headline"]]

        expected_items = {
            (
                source_id,
                f"task11:postgres:{token}:row:postgres-abc",
                _revision_hash(_rows(token)[0]),
                "CATALYST_EVENT",
                str(event_by_headline["ABC original"].state_id),
                "IMPORTED",
            ),
            (
                source_id,
                f"task11:postgres:{token}:row:postgres-abc",
                _revision_hash(_rows(token, revised=True)[0]),
                "CATALYST_EVENT",
                str(event_by_headline["ABC revised"].state_id),
                "IMPORTED",
            ),
            (
                source_id,
                f"task11:postgres:{token}:row:postgres-msft",
                _revision_hash(_rows(token)[1]),
                "CATALYST_EVENT",
                str(event_by_headline["MSFT original"].state_id),
                "IMPORTED",
            ),
            (
                source_id,
                f"task11:postgres:{token}:row:postgres-market",
                _revision_hash(_rows(token)[2]),
                "CATALYST_EVENT",
                str(event_by_headline["FOMC scheduled"].state_id),
                "IMPORTED",
            ),
            (
                source_id,
                f"task11:postgres:{token}:row:postgres-quarantined",
                _revision_hash(_rows(token)[3]),
                "CATALYST_EVENT_QUARANTINE",
                None,
                "QUARANTINED",
            ),
        }
        actual_items = {
            (
                item.source_id,
                item.record_locator,
                item.normalized_hash,
                item.target_type,
                item.target_id,
                item.status,
            )
            for item in import_items
        }
        assert actual_items == expected_items
        assert _revision_hash(_rows(token)[0]) != _revision_hash(_rows(token, revised=True)[0])
        assert event_by_headline["ABC original"].state_id != event_by_headline["ABC revised"].state_id
        assert len([item for item in import_items if item.record_locator == f"task11:postgres:{token}:row:postgres-quarantined"]) == 1
        for item in import_items:
            assert item.warnings["source_lineage"] == {
                "source_id": source.source_id,
                "content_hash": source.content_hash.lower(),
                "record_locator": item.record_locator,
            }
