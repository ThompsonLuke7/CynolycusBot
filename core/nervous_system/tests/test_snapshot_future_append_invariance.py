from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from core.nervous_system.config.freshness import META_4H_1420_PROFILE
from core.nervous_system.context.snapshot_builder import SnapshotBuilder
from core.nervous_system.contracts.base import canonical_json
from core.nervous_system.persistence.repositories.state import StateRepository

from core.nervous_system.tests.test_snapshot_builder import (
    DECISION_1420,
    DECISION_BAR,
    TICKER,
    _market,
    _required_states,
    _save,
    _ticker,
)


UTC = timezone.utc


@pytest.mark.postgres
def test_future_appends_do_not_change_original_canonical_snapshot(pg_session):
    repo = StateRepository(pg_session)
    _save(repo, *_required_states())
    builder = SnapshotBuilder(repo, sector_entity_ids=("technology",))

    first = builder.build(
        strategy_id="meta-future-invariant",
        entity_id=TICKER,
        decision_time=DECISION_1420,
        decision_bar=DECISION_BAR,
        profile=META_4H_1420_PROFILE,
    )
    before = canonical_json(first)

    future_available = DECISION_1420 + timedelta(hours=1)
    _save(
        repo,
        _ticker(
            UUID("00000000-0000-0000-0000-000000002001"),
            available_at=future_available,
            as_of=DECISION_1420,
            generated_at=future_available,
        ),
        _market(
            UUID("00000000-0000-0000-0000-000000002002"),
            available_at=future_available,
            as_of=DECISION_1420,
            generated_at=future_available,
        ),
    )

    second = builder.build(
        strategy_id="meta-future-invariant",
        entity_id=TICKER,
        decision_time=DECISION_1420,
        decision_bar=DECISION_BAR,
        profile=META_4H_1420_PROFILE,
    )

    assert canonical_json(second) == before
    assert second.content_hash == first.content_hash
    assert second.snapshot_id == first.snapshot_id
