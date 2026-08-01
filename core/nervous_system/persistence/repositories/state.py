"""Typed causal state and context-snapshot persistence operations."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.orm import Session

from core.nervous_system.contracts.base import content_hash
from core.nervous_system.contracts.context import ContextSnapshot, StateRequest
from core.nervous_system.contracts.enums import DataQualitySeverity, StateType
from core.nervous_system.contracts.quality import DataQualitySummary
from core.nervous_system.contracts.states import (
    CatalystEvent,
    CatalystPressure,
    DealerState,
    MarketState,
    PortfolioState,
    ReadinessState,
    SectorState,
    StateEnvelope,
    ThemeMembership,
    ThemeState,
    TickerState,
)
from core.nervous_system.persistence.models import ContextSnapshot as ContextSnapshotRow
from core.nervous_system.persistence.models import StateRecord


_STATE_TYPES: dict[StateType, tuple[type[StateEnvelope], ...]] = {
    StateType.MARKET: (MarketState,),
    StateType.SECTOR: (SectorState,),
    StateType.THEME: (ThemeState,),
    StateType.THEME_MEMBERSHIP: (ThemeMembership,),
    StateType.TICKER: (TickerState,),
    StateType.CATALYST_EVENT: (CatalystEvent,),
    StateType.CATALYST_PRESSURE: (CatalystPressure,),
    StateType.DEALER: (DealerState,),
    StateType.PORTFOLIO: (PortfolioState,),
    StateType.READINESS: (ReadinessState,),
}


def _quality_severity(summary: DataQualitySummary) -> str:
    severities = {issue.severity for issue in summary.issues}
    for severity in (
        DataQualitySeverity.CRITICAL,
        DataQualitySeverity.ERROR,
        DataQualitySeverity.WARNING,
        DataQualitySeverity.INFO,
    ):
        if severity in severities:
            return severity.value
    return DataQualitySeverity.INFO.value


def _validate_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _contract_from_payload(row: StateRecord) -> StateEnvelope:
    errors: list[str] = []
    for contract_type in _STATE_TYPES[StateType(row.state_type)]:
        try:
            state = contract_type.model_validate(row.payload)
        except Exception as exc:  # Pydantic provides the useful field detail.
            errors.append(f"{contract_type.__name__}: {exc}")
            continue
        if state.state_id != row.state_id:
            raise ValueError("state payload state_id does not match relational state_id")
        if state.state_type.value != row.state_type or state.entity_id != row.entity_id:
            raise ValueError("state payload identity does not match relational columns")
        if (
            state.as_of != row.as_of
            or state.available_at != row.available_at
            or state.generated_at != row.generated_at
            or state.valid_until != row.valid_until
        ):
            raise ValueError("state payload timestamps do not match relational columns")
        if state.schema_version != row.schema_version:
            raise ValueError("state payload schema_version does not match relational column")
        if state.producer != row.producer:
            raise ValueError("state payload producer does not match relational column")
        if state.model_version != row.model_version:
            raise ValueError("state payload model_version does not match relational column")
        if state.feature_version != row.feature_version:
            raise ValueError("state payload feature_version does not match relational column")
        if state.config_version != row.config_version:
            raise ValueError("state payload config_version does not match relational column")
        expected_hash = content_hash(state, exclude={"state_id"})
        if row.content_hash != expected_hash:
            raise ValueError("state content_hash does not match payload")
        if row.quality_severity != _quality_severity(state.data_quality):
            raise ValueError("state quality_severity does not match payload")
        return state
    raise ValueError("unable to reconstruct state payload: " + " | ".join(errors))


def _one_or_none(result: Any) -> Any:
    scalars = result.scalars()
    if hasattr(scalars, "one_or_none"):
        return scalars.one_or_none()
    return scalars.first()


def _is_postgresql(session: Session) -> bool:
    bind = getattr(session, "bind", None)
    dialect = getattr(bind, "dialect", None)
    return getattr(dialect, "name", None) == "postgresql"


def _state_values(state: StateEnvelope) -> dict[str, Any]:
    return {
        "state_id": state.state_id,
        "state_type": state.state_type.value,
        "entity_id": state.entity_id,
        "as_of": state.as_of,
        "available_at": state.available_at,
        "generated_at": state.generated_at,
        "valid_until": state.valid_until,
        "schema_version": state.schema_version,
        "producer": state.producer,
        "model_version": state.model_version,
        "feature_version": state.feature_version,
        "config_version": state.config_version,
        "quality_severity": _quality_severity(state.data_quality),
        "content_hash": content_hash(state, exclude={"state_id"}),
        "payload": state.model_dump(mode="json"),
        "created_at": state.generated_at,
    }


class StateRepository:
    def __init__(self, session: Session):
        self._session = session

    def save_state(self, state: StateEnvelope) -> StateEnvelope:
        row = StateRecord(**_state_values(state))
        self._session.add(row)
        self._session.flush()
        return state

    def insert_states_if_absent(
        self, states: Sequence[StateEnvelope]
    ) -> dict[str, Any]:
        """Bulk insert immutable states and resolve every content hash to its ID."""

        if not states:
            return {}
        states_by_hash: dict[str, StateEnvelope] = {}
        for state in states:
            state_hash = content_hash(state, exclude={"state_id"})
            states_by_hash.setdefault(state_hash, state)
        values = [_state_values(state) for state in states_by_hash.values()]
        if not _is_postgresql(self._session):
            resolved: dict[str, Any] = {}
            for state_hash, state in states_by_hash.items():
                existing = self.get_state_by_content_hash(state_hash)
                if existing is None:
                    self.save_state(state)
                    resolved[state_hash] = state.state_id
                else:
                    resolved[state_hash] = existing.state_id
            return resolved

        inserted = self._session.execute(
            postgres_insert(StateRecord)
            .values(values)
            .on_conflict_do_nothing(index_elements=[StateRecord.content_hash])
            .returning(StateRecord.content_hash, StateRecord.state_id)
        ).all()
        resolved = {row[0]: row[1] for row in inserted}
        missing_hashes = set(states_by_hash) - set(resolved)
        if missing_hashes:
            existing = self._session.execute(
                select(StateRecord.content_hash, StateRecord.state_id).where(
                    StateRecord.content_hash.in_(missing_hashes)
                )
            ).all()
            resolved.update({row[0]: row[1] for row in existing})
        if set(resolved) != set(states_by_hash):
            raise RuntimeError("state conflict did not produce a readable row")
        return resolved

    def insert_states_idempotently(
        self, states: Sequence[StateEnvelope]
    ) -> dict[UUID, UUID]:
        """Atomically insert states by their stable identity.

        The state payload retains its producer generation timestamp, including
        the reviewed call-time fallback.  Publication identity is instead the
        deterministic ``state_id`` supplied by the adapter.  PostgreSQL's
        conflict-free insert handles reruns without turning an expected race
        into an integrity-error control flow; a changed source lineage gets a
        different identity and remains new evidence.
        """

        if not states:
            return {}
        states_by_id: dict[UUID, StateEnvelope] = {}
        for state in states:
            existing = states_by_id.get(state.state_id)
            if existing is not None and content_hash(existing, exclude={"state_id"}) != content_hash(
                state, exclude={"state_id"}
            ):
                raise ValueError("same state_id was supplied for different state content")
            states_by_id.setdefault(state.state_id, state)

        values = [_state_values(state) for state in states_by_id.values()]
        if not _is_postgresql(self._session):
            resolved: dict[UUID, UUID] = {}
            for state_id, state in states_by_id.items():
                existing_row = self._session.get(StateRecord, state_id)
                if existing_row is not None:
                    _contract_from_payload(existing_row)
                    resolved[state_id] = existing_row.state_id
                    continue
                existing = self.get_state_by_content_hash(
                    content_hash(state, exclude={"state_id"})
                )
                if existing is not None:
                    resolved[state_id] = existing.state_id
                    continue
                self.save_state(state)
                resolved[state_id] = state_id
            return resolved

        inserted = self._session.execute(
            postgres_insert(StateRecord)
            .values(values)
            .on_conflict_do_nothing()
            .returning(StateRecord.state_id)
        ).all()
        resolved = {row[0]: row[0] for row in inserted}
        missing_ids = set(states_by_id) - set(resolved)
        if missing_ids:
            existing_rows = self._session.execute(
                select(StateRecord).where(StateRecord.state_id.in_(missing_ids))
            ).scalars().all()
            for row in existing_rows:
                _contract_from_payload(row)
                resolved[row.state_id] = row.state_id

        # A content-hash conflict can only occur when an equivalent immutable
        # state was already published under another identity.  Resolve it
        # without catching an integrity error; revised lineage/content remains
        # a new row because its hash and stable identity differ.
        unresolved_ids = set(states_by_id) - set(resolved)
        if unresolved_ids:
            content_hashes = {
                content_hash(states_by_id[state_id], exclude={"state_id"})
                for state_id in unresolved_ids
            }
            existing_rows = self._session.execute(
                select(StateRecord).where(StateRecord.content_hash.in_(content_hashes))
            ).scalars().all()
            by_hash = {row.content_hash: row for row in existing_rows}
            for state_id in unresolved_ids:
                state_hash = content_hash(states_by_id[state_id], exclude={"state_id"})
                row = by_hash.get(state_hash)
                if row is not None:
                    _contract_from_payload(row)
                    resolved[state_id] = row.state_id
        if set(resolved) != set(states_by_id):
            raise RuntimeError("state identity conflict did not produce a readable row")
        return resolved

    def get_state_by_content_hash(self, state_hash: str) -> StateEnvelope | None:
        """Load an existing immutable state so a revised source can add lineage."""

        if len(state_hash) != 64:
            raise ValueError("state_hash must be a 64-character SHA-256 hex string")
        row = _one_or_none(
            self._session.execute(
                select(StateRecord).where(StateRecord.content_hash == state_hash)
            )
        )
        return None if row is None else _contract_from_payload(row)

    def get_latest_valid_state(
        self,
        state_type: StateType,
        entity_id: str,
        decision_time: datetime,
    ) -> StateEnvelope | None:
        _validate_aware(decision_time, "decision_time")
        stmt = (
            select(StateRecord)
            .where(
                StateRecord.state_type == state_type.value,
                StateRecord.entity_id == entity_id,
                StateRecord.available_at <= decision_time,
                StateRecord.valid_until > decision_time,
            )
            .order_by(
                StateRecord.available_at.desc(),
                StateRecord.generated_at.desc(),
                StateRecord.as_of.desc(),
                StateRecord.state_id.desc(),
            )
            .limit(1)
        )
        row = _one_or_none(self._session.execute(stmt))
        return None if row is None else _contract_from_payload(row)

    def get_state_as_of(
        self,
        state_type: StateType,
        entity_id: str,
        decision_time: datetime,
    ) -> StateEnvelope | None:
        _validate_aware(decision_time, "decision_time")
        stmt = (
            select(StateRecord)
            .where(
                StateRecord.state_type == state_type.value,
                StateRecord.entity_id == entity_id,
                StateRecord.as_of <= decision_time,
                StateRecord.available_at <= decision_time,
            )
            .order_by(
                StateRecord.as_of.desc(),
                StateRecord.available_at.desc(),
                StateRecord.generated_at.desc(),
                StateRecord.state_id.desc(),
            )
            .limit(1)
        )
        row = _one_or_none(self._session.execute(stmt))
        return None if row is None else _contract_from_payload(row)

    def get_states_for_snapshot(
        self,
        requests: Sequence[StateRequest],
        decision_time: datetime,
    ) -> tuple[StateEnvelope, ...]:
        _validate_aware(decision_time, "decision_time")
        if not requests:
            return ()
        request_conditions = []
        for request in requests:
            condition = [
                StateRecord.state_type == request.state_type.value,
                StateRecord.entity_id == request.entity_id,
                StateRecord.available_at <= decision_time,
                StateRecord.valid_until > decision_time,
            ]
            if request.bar_bound is not None:
                condition.append(StateRecord.as_of <= request.bar_bound)
            request_conditions.append(and_(*condition))
        stmt = (
            select(StateRecord)
            .where(or_(*request_conditions))
            .order_by(
                StateRecord.state_type.asc(),
                StateRecord.entity_id.asc(),
                StateRecord.available_at.desc(),
                StateRecord.generated_at.desc(),
                StateRecord.as_of.desc(),
                StateRecord.state_id.desc(),
            )
        )
        rows = self._session.execute(stmt).scalars().all()
        by_key: dict[tuple[str, str], list[StateRecord]] = {}
        for row in rows:
            by_key.setdefault((row.state_type, row.entity_id), []).append(row)

        selected: list[StateEnvelope] = []
        for request in requests:
            candidates = by_key.get((request.state_type.value, request.entity_id), ())
            chosen = next(
                (
                    row
                    for row in candidates
                    if request.bar_bound is None or row.as_of <= request.bar_bound
                ),
                None,
            )
            if chosen is not None:
                selected.append(_contract_from_payload(chosen))
        return tuple(selected)

    def save_context_snapshot(self, snapshot: ContextSnapshot) -> ContextSnapshot:
        if snapshot.content_hash != snapshot.computed_content_hash():
            raise ValueError("snapshot content_hash does not match content")
        row = ContextSnapshotRow(
            snapshot_id=snapshot.snapshot_id,
            decision_time=snapshot.decision_time,
            strategy_id=snapshot.strategy_id,
            ticker=snapshot.ticker,
            freshness_profile=snapshot.freshness_profile,
            content_hash=snapshot.content_hash,
            payload=snapshot.model_dump(mode="json"),
            created_at=snapshot.decision_time,
        )
        self._session.add(row)
        self._session.flush()
        return snapshot
