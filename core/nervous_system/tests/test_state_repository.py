from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from uuid import UUID, uuid4

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy import event, select

from core.nervous_system.contracts.context import ContextSnapshot, StateRequest
from core.nervous_system.contracts.enums import MarketRegime, StateType, ThemeRegime
from core.nervous_system.contracts.quality import DataQualitySummary
from core.nervous_system.contracts.states import CatalystEvent, MarketState, ThemeState
from core.nervous_system.persistence.models import (
    ContextSnapshot as ContextSnapshotRow,
    StateRecord,
)
from core.nervous_system.persistence.repositories.state import StateRepository


UTC = timezone.utc
DECISION_TIME = datetime(2026, 7, 30, 18, 20, tzinfo=UTC)


def _market(
    *,
    state_id: UUID | None = None,
    as_of: datetime = datetime(2026, 7, 29, 20, 0, tzinfo=UTC),
    available_at: datetime = datetime(2026, 7, 29, 20, 30, tzinfo=UTC),
    valid_until: datetime = datetime(2026, 7, 30, 20, 30, tzinfo=UTC),
    metrics: dict[str, float] | None = None,
) -> MarketState:
    return MarketState(
        state_id=state_id or uuid4(),
        state_type=StateType.MARKET,
        entity_id="US",
        as_of=as_of,
        available_at=available_at,
        generated_at=available_at,
        valid_until=valid_until,
        source_window_start=as_of - timedelta(hours=1),
        source_window_end=as_of,
        schema_version=1,
        producer="test",
        model_version="test@1",
        feature_version="test@1",
        config_version="test@1",
        lineage_ids=(),
        data_quality=DataQualitySummary(),
        regime=MarketRegime.NEUTRAL,
        metrics=metrics or {},
    )


class _EmptyResult:
    def scalars(self):
        return self

    def all(self):
        return []

    def one_or_none(self):
        return None

    def scalar_one_or_none(self):
        return None


class _RecordingSession:
    def __init__(self):
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)
        return _EmptyResult()


def test_valid_state_statement_contains_causal_window_and_exact_tie_break():
    session = _RecordingSession()
    assert StateRepository(session).get_latest_valid_state(
        StateType.MARKET, "US", DECISION_TIME
    ) is None

    sql = str(
        session.statements[0].compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": False},
        )
    )
    assert "state_records.available_at <=" in sql
    assert "state_records.valid_until >" in sql
    order_sql = sql.split("ORDER BY", 1)[1]
    positions = [
        order_sql.index("state_records.available_at DESC"),
        order_sql.index("state_records.generated_at DESC"),
        order_sql.index("state_records.as_of DESC"),
        order_sql.index("state_records.state_id DESC"),
    ]
    assert positions == sorted(positions)


def test_snapshot_selection_is_one_bounded_query_and_preserves_request_order_offline():
    session = _RecordingSession()
    result = StateRepository(session).get_states_for_snapshot(
        (
            StateRequest(state_type=StateType.MARKET, entity_id="EU", required=True),
            StateRequest(state_type=StateType.MARKET, entity_id="US", required=True),
        ),
        DECISION_TIME,
    )
    assert result == ()
    assert len(session.statements) == 1
    sql = str(session.statements[0].compile(dialect=postgresql.dialect()))
    assert "state_records.available_at <=" in sql
    assert "state_records.valid_until >" in sql


@pytest.mark.postgres
def test_insert_states_idempotently_converges_on_state_identity_and_preserves_revision(pg_session):
    repo = StateRepository(pg_session)
    stable_id = UUID("00000000-0000-0000-0000-000000000101")
    first = _market(state_id=stable_id, metrics={"version": 1}).model_copy(
        update={"generated_at": datetime(2026, 7, 29, 21, 0, tzinfo=UTC)}
    )
    rerun = first.model_copy(
        update={"generated_at": datetime(2026, 7, 30, 21, 0, tzinfo=UTC)}
    )
    revised = first.model_copy(
        update={
            "state_id": UUID("00000000-0000-0000-0000-000000000102"),
            "lineage_ids": ("revised-artifact:row:1",),
        }
    )

    repo.insert_states_idempotently((first,))
    repo.insert_states_idempotently((rerun,))
    repo.insert_states_idempotently((revised,))
    pg_session.flush()

    rows = pg_session.execute(
        select(StateRecord).where(StateRecord.entity_id == "US")
    ).scalars().all()
    assert {row.state_id for row in rows} >= {first.state_id, revised.state_id}
    assert sum(row.state_id == first.state_id for row in rows) == 1
    stored = pg_session.get(StateRecord, stable_id)
    assert stored is not None
    assert stored.generated_at == first.generated_at
    assert len(rows) == 2


@pytest.mark.postgres
def test_catalyst_event_raw_score_round_trips_through_state_repository(pg_session):
    event_id = UUID("00000000-0000-0000-0000-000000000301")
    available_at = datetime(2026, 7, 30, 13, 3, tzinfo=UTC)
    event = CatalystEvent(
        state_id=event_id,
        entity_id=str(event_id),
        as_of=datetime(2026, 7, 30, 13, 0, tzinfo=UTC),
        available_at=available_at,
        generated_at=available_at,
        valid_until=datetime(2026, 7, 31, 13, 3, tzinfo=UTC),
        source_window_start=datetime(2026, 7, 30, 13, 0, tzinfo=UTC),
        source_window_end=datetime(2026, 7, 30, 13, 0, tzinfo=UTC),
        schema_version=1,
        producer="test.catalyst",
        model_version="test@1",
        feature_version="test@1",
        config_version="test@1",
        lineage_ids=("wire:sha:row-1",),
        data_quality=DataQualitySummary(),
        event_id=event_id,
        ticker="ABC",
        event_type="company_news",
        event_time=datetime(2026, 7, 30, 13, 0, tzinfo=UTC),
        published_at=datetime(2026, 7, 30, 13, 1, tzinfo=UTC),
        observed_at=available_at,
        source="wire",
        headline="typed score",
        channel="WIRE",
        raw_score=4.5,
    )

    repository = StateRepository(pg_session)
    repository.insert_states_idempotently((event,))
    restored = repository.get_state_as_of(
        StateType.CATALYST_EVENT,
        str(event_id),
        datetime(2026, 7, 30, 14, 0, tzinfo=UTC),
    )

    assert isinstance(restored, CatalystEvent)
    assert restored.raw_score == 4.5


@pytest.mark.postgres
def test_latest_valid_state_obeys_available_at_and_exclusive_valid_until(pg_session):
    repo = StateRepository(pg_session)
    early = _market(
        state_id=UUID("00000000-0000-0000-0000-000000000001"),
        metrics={"version": 1},
    )
    future = _market(
        state_id=UUID("00000000-0000-0000-0000-000000000002"),
        as_of=datetime(2026, 7, 30, 20, 0, tzinfo=UTC),
        available_at=early.valid_until,
        valid_until=datetime(2026, 7, 31, 20, 30, tzinfo=UTC),
        metrics={"version": 2},
    )
    repo.save_state(early)
    repo.save_state(future)

    assert repo.get_latest_valid_state(StateType.MARKET, "US", DECISION_TIME).state_id == early.state_id
    assert repo.get_latest_valid_state(StateType.MARKET, "US", early.valid_until).state_id == future.state_id


@pytest.mark.postgres
def test_valid_state_tie_break_is_available_generated_asof_then_state_id(pg_session):
    repo = StateRepository(pg_session)
    first = _market(
        state_id=UUID("00000000-0000-0000-0000-000000000010"),
        metrics={"version": 10},
    )
    second = _market(
        state_id=UUID("00000000-0000-0000-0000-000000000011"),
        metrics={"version": 11},
    )
    repo.save_state(first)
    repo.save_state(second)
    assert repo.get_latest_valid_state(StateType.MARKET, "US", DECISION_TIME).state_id == second.state_id


@pytest.mark.postgres
def test_get_state_as_of_is_historical_and_does_not_substitute_valid_selection(pg_session):
    repo = StateRepository(pg_session)
    expired = _market(
        as_of=datetime(2026, 7, 29, 12, 0, tzinfo=UTC),
        available_at=datetime(2026, 7, 29, 12, 30, tzinfo=UTC),
        valid_until=datetime(2026, 7, 30, 12, 30, tzinfo=UTC),
    )
    repo.save_state(expired)
    assert repo.get_latest_valid_state(StateType.MARKET, "US", DECISION_TIME) is None
    assert repo.get_state_as_of(StateType.MARKET, "US", DECISION_TIME).state_id == expired.state_id


@pytest.mark.postgres
def test_get_states_for_snapshot_runs_one_query_and_returns_request_order(pg_session):
    repo = StateRepository(pg_session)
    us = _market(state_id=uuid4(), metrics={"entity": 1})
    eu = _market(state_id=uuid4(), metrics={"entity": 2})
    eu = eu.model_copy(update={"entity_id": "EU"})
    repo.save_state(us)
    repo.save_state(eu)

    statements: list[str] = []

    def record_sql(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(pg_session.bind, "before_cursor_execute", record_sql)
    try:
        selected = repo.get_states_for_snapshot(
            (
                StateRequest(state_type=StateType.MARKET, entity_id="EU", required=True),
                StateRequest(state_type=StateType.MARKET, entity_id="US", required=True),
            ),
            DECISION_TIME,
        )
    finally:
        event.remove(pg_session.bind, "before_cursor_execute", record_sql)

    assert [state.entity_id for state in selected] == ["EU", "US"]
    assert len(statements) == 1


@pytest.mark.postgres
def test_save_context_snapshot_persists_validated_payload(pg_session):
    state = _market()
    snapshot = ContextSnapshot.from_states(
        snapshot_id=uuid4(),
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker="AMD",
        states=(state,),
        freshness_profile="qa",
    )
    saved = StateRepository(pg_session).save_context_snapshot(snapshot)
    assert saved == snapshot
    assert pg_session.get(ContextSnapshotRow, snapshot.snapshot_id).content_hash == snapshot.content_hash


def _theme(*, state_id: UUID, theme_id: str) -> ThemeState:
    return ThemeState(
        state_id=state_id,
        entity_id=theme_id,
        as_of=datetime(2026, 7, 30, 20, 0, tzinfo=UTC),
        available_at=datetime(2026, 7, 30, 20, 5, tzinfo=UTC),
        generated_at=datetime(2026, 7, 30, 20, 4, tzinfo=UTC),
        valid_until=datetime(2026, 7, 31, 20, 5, tzinfo=UTC),
        source_window_start=datetime(2026, 7, 30, 20, 0, tzinfo=UTC),
        source_window_end=datetime(2026, 7, 30, 20, 0, tzinfo=UTC),
        schema_version=1,
        producer="themes.dynamic_theme",
        model_version="theme-taxonomy@1",
        feature_version="theme-features@1",
        config_version="theme-state@1:taxonomy-v1",
        lineage_ids=("generic:row:0",),
        data_quality=DataQualitySummary(),
        theme_id=theme_id,
        theme_regime=ThemeRegime.UNKNOWN,
        membership_scores={"AAA": 0.82},
    )


def _legacy_theme_membership_row(
    *,
    state_id: UUID = UUID("00000000-0000-0000-0000-000000000201"),
    entity_id: str = "alpha_theme",
    payload_updates: dict[str, object] | None = None,
    remove_keys: tuple[str, ...] = (),
) -> StateRecord:
    payload = {
        "state_id": str(state_id),
        "state_type": "THEME",
        "entity_id": entity_id,
        "as_of": "2026-07-30T20:00:00Z",
        "available_at": "2026-07-30T20:05:00Z",
        "generated_at": "2026-07-30T20:04:00Z",
        "valid_until": "2026-07-31T20:05:00Z",
        "source_window_start": "2026-07-30T20:00:00Z",
        "source_window_end": "2026-07-30T20:00:00Z",
        "schema_version": 1,
        "producer": "themes.dynamic_theme",
        "model_version": "theme-taxonomy@1",
        "feature_version": "theme-features@1",
        "config_version": "theme-state@1:taxonomy-v1",
        "lineage_ids": ["legacy:row:0"],
        "data_quality": {"issues": []},
        "ticker": "AAA",
        "theme": entity_id,
        "membership_score": 0.82,
    }
    payload.update(payload_updates or {})
    for key in remove_keys:
        payload.pop(key, None)
    digest_payload = {key: value for key, value in payload.items() if key != "state_id"}
    state_hash = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    as_of = datetime(2026, 7, 30, 20, 0, tzinfo=UTC)
    available_at = datetime(2026, 7, 30, 20, 5, tzinfo=UTC)
    return StateRecord(
        state_id=state_id,
        state_type="THEME",
        entity_id=entity_id,
        as_of=as_of,
        available_at=available_at,
        generated_at=datetime(2026, 7, 30, 20, 4, tzinfo=UTC),
        valid_until=datetime(2026, 7, 31, 20, 5, tzinfo=UTC),
        schema_version=1,
        producer="themes.dynamic_theme",
        model_version="theme-taxonomy@1",
        feature_version="theme-features@1",
        config_version="theme-state@1:taxonomy-v1",
        quality_severity="INFO",
        content_hash=state_hash,
        payload=payload,
        created_at=available_at,
    )


@pytest.mark.postgres
def test_theme_membership_queries_round_trip_legacy_theme_payload(pg_session):
    row = _legacy_theme_membership_row()
    pg_session.add(row)
    pg_session.flush()
    repo = StateRepository(pg_session)
    decision_time = datetime(2026, 7, 30, 21, 0, tzinfo=UTC)

    latest = repo.get_latest_valid_state(
        StateType.THEME_MEMBERSHIP, "alpha_theme", decision_time
    )
    historical = repo.get_state_as_of(
        StateType.THEME_MEMBERSHIP, "alpha_theme", decision_time
    )
    snapshot = repo.get_states_for_snapshot(
        (StateRequest(state_type=StateType.THEME_MEMBERSHIP, entity_id="alpha_theme", required=True),),
        decision_time,
    )

    assert latest is not None
    assert latest.state_type is StateType.THEME_MEMBERSHIP
    assert latest.state_id == row.state_id
    assert latest.entity_id == "alpha_theme"
    assert latest.ticker == "AAA"
    assert latest.weight == 0.82
    assert historical == latest
    assert snapshot == (latest,)
    assert repo.get_latest_valid_state(StateType.THEME, "alpha_theme", decision_time) is None

    row.payload = {**row.payload, "membership_score": 0.81}
    pg_session.flush()
    with pytest.raises(ValueError, match="content_hash"):
        repo.get_state_as_of(StateType.THEME_MEMBERSHIP, "alpha_theme", decision_time)


@pytest.mark.postgres
def test_legacy_membership_queries_require_absent_aggregate_membership_key(pg_session):
    legacy = _legacy_theme_membership_row(
        state_id=UUID("00000000-0000-0000-0000-000000000211"),
        entity_id="legacy_theme",
    )
    explicit_null = _legacy_theme_membership_row(
        state_id=UUID("00000000-0000-0000-0000-000000000212"),
        entity_id="explicit_null_theme",
        payload_updates={"membership_scores": None},
    )
    malformed = _legacy_theme_membership_row(
        state_id=UUID("00000000-0000-0000-0000-000000000213"),
        entity_id="malformed_theme",
        remove_keys=("membership_score",),
    )
    generic = _theme(
        state_id=UUID("00000000-0000-0000-0000-000000000214"),
        theme_id="generic_theme",
    )
    pg_session.add_all((legacy, explicit_null, malformed))
    repo = StateRepository(pg_session)
    repo.save_state(generic)
    pg_session.flush()
    decision_time = datetime(2026, 7, 30, 21, 0, tzinfo=UTC)

    selected_legacy = repo.get_latest_valid_state(
        StateType.THEME_MEMBERSHIP,
        "legacy_theme",
        decision_time,
    )
    assert selected_legacy is not None
    assert selected_legacy.state_type is StateType.THEME_MEMBERSHIP
    assert repo.get_latest_valid_state(
        StateType.THEME_MEMBERSHIP,
        "explicit_null_theme",
        decision_time,
    ) is None
    assert repo.get_state_as_of(
        StateType.THEME_MEMBERSHIP,
        "malformed_theme",
        decision_time,
    ) is None
    assert repo.get_latest_valid_state(
        StateType.THEME_MEMBERSHIP,
        "generic_theme",
        decision_time,
    ) is None

    membership_snapshot = repo.get_states_for_snapshot(
        (
            StateRequest(
                state_type=StateType.THEME_MEMBERSHIP,
                entity_id="legacy_theme",
                required=True,
            ),
            StateRequest(
                state_type=StateType.THEME_MEMBERSHIP,
                entity_id="explicit_null_theme",
                required=True,
            ),
            StateRequest(
                state_type=StateType.THEME_MEMBERSHIP,
                entity_id="malformed_theme",
                required=True,
            ),
            StateRequest(
                state_type=StateType.THEME_MEMBERSHIP,
                entity_id="generic_theme",
                required=True,
            ),
        ),
        decision_time,
    )
    generic_snapshot = repo.get_states_for_snapshot(
        (
            StateRequest(
                state_type=StateType.THEME,
                entity_id="generic_theme",
                required=True,
            ),
        ),
        decision_time,
    )

    assert membership_snapshot == (selected_legacy,)
    assert generic_snapshot == (generic,)
    assert repo.get_latest_valid_state(
        StateType.THEME,
        "generic_theme",
        decision_time,
    ) == generic
