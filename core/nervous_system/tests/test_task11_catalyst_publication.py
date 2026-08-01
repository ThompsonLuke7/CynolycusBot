from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from core.nervous_system.contracts.enums import StateType
from core.nervous_system.contracts.quality import LineageRef
from core.nervous_system.persistence.models import (
    ImportItem,
    ImportQuarantine,
    SourceArtifact,
    StateRecord,
)
from core.nervous_system.persistence.uow import UnitOfWork
from signals.catalysts.nervous_system_adapter import publish_catalyst_states


UTC = timezone.utc
SOURCE = LineageRef(
    source_id="task11-postgres-catalyst-feed",
    content_hash="c" * 64,
    record_locator="task11:postgres:batch",
)


def _time(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 7, 30, hour, minute, tzinfo=UTC)


def _rows(*, revised: bool = False) -> tuple[dict[str, object], ...]:
    return (
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
    )


@pytest.mark.postgres
def test_task11_postgres_publication_is_atomic_idempotent_lineaged_and_revision_safe(
    session_factory,
):
    def publish(rows):
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

            uow.commit = tracked_commit
            uow.rollback = tracked_rollback
            results = publish_catalyst_states(
                rows,
                unit_of_work=uow,
                source_artifact=SOURCE,
                decision_time=_time(14),
                valid_until=_time(16),
            )
            assert calls == []
            uow.commit()
            assert calls == ["commit"]
            return results

    first = publish(_rows())
    second = publish(_rows())
    revised = publish(_rows(revised=True)[:1])

    assert [result.quarantine_code for result in first] == [None, None, None, "NAIVE_OBSERVED_AT"]
    assert all(result.quarantine_code is None for result in second[:3])
    assert revised[0].event is not None

    with session_factory() as session:
        source_rows = session.scalars(
            select(SourceArtifact).where(SourceArtifact.uri == SOURCE.source_id)
        ).all()
        assert len(source_rows) == 1
        assert source_rows[0].sha256 == SOURCE.content_hash

        import_items = session.scalars(
            select(ImportItem).where(ImportItem.source_id == source_rows[0].source_id)
        ).all()
        assert len(import_items) == 5
        assert sum(item.status == "QUARANTINED" for item in import_items) == 1

        quarantines = session.scalars(
            select(ImportQuarantine).where(ImportQuarantine.source_id == source_rows[0].source_id)
        ).all()
        assert len(quarantines) == 1
        assert quarantines[0].error_code == "NAIVE_OBSERVED_AT"
        assert quarantines[0].raw_payload["source_record_id"] == "postgres-quarantined"
        assert quarantines[0].record_locator == SOURCE.record_locator

        states = session.scalars(
            select(StateRecord).where(StateRecord.producer == "signals.catalysts")
        ).all()
        events = [state for state in states if state.state_type == StateType.CATALYST_EVENT.value]
        pressures = [
            state for state in states if state.state_type == StateType.CATALYST_PRESSURE.value
        ]
        assert len(events) == 4
        assert len(pressures) == 4
        assert {state.payload["scope_id"] for state in pressures} >= {"ABC", "MSFT", "MARKET"}
        assert any(
            SOURCE.source_id in state.payload["lineage_ids"][0]
            and SOURCE.content_hash in state.payload["lineage_ids"][0]
            and SOURCE.record_locator in state.payload["lineage_ids"][0]
            for state in events
        )
