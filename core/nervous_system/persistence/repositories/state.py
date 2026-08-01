"""Typed causal state and context-snapshot persistence operations."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
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


def _is_legacy_theme_membership_payload(payload: object) -> bool:
    return (
        isinstance(payload, dict)
        and all(field in payload for field in ("ticker", "theme", "membership_score"))
        and "membership_scores" not in payload
    )


def _payload_timestamp(value: object, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"legacy payload {field_name} is not a timestamp") from exc
    else:
        raise ValueError(f"legacy payload {field_name} is not a timestamp")
    _validate_aware(parsed, f"legacy payload {field_name}")
    return parsed.astimezone(timezone.utc)


def _legacy_payload_hash(payload: dict[str, Any]) -> str:
    material = {key: value for key, value in payload.items() if key != "state_id"}
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _legacy_quality_severity(payload: dict[str, Any]) -> str:
    quality = payload.get("data_quality", {})
    issues = quality.get("issues", []) if isinstance(quality, dict) else []
    severities = {
        str(issue.get("severity"))
        for issue in issues
        if isinstance(issue, dict) and issue.get("severity") is not None
    }
    for severity in (
        DataQualitySeverity.CRITICAL.value,
        DataQualitySeverity.ERROR.value,
        DataQualitySeverity.WARNING.value,
        DataQualitySeverity.INFO.value,
    ):
        if severity in severities:
            return severity
    return DataQualitySeverity.INFO.value


def _validate_legacy_theme_membership_row(row: StateRecord) -> dict[str, Any]:
    payload = row.payload
    if not isinstance(payload, dict):
        raise ValueError("legacy state payload must be an object")
    if str(payload.get("state_id")) != str(row.state_id):
        raise ValueError("legacy state payload state_id does not match relational state_id")
    if payload.get("state_type") != row.state_type:
        raise ValueError("legacy state payload state_type does not match relational state_type")
    if payload.get("entity_id") != row.entity_id:
        raise ValueError("legacy state payload entity_id does not match relational entity_id")
    for field_name in ("as_of", "available_at", "generated_at", "valid_until"):
        payload_value = _payload_timestamp(payload.get(field_name), field_name)
        row_value = getattr(row, field_name)
        _validate_aware(row_value, f"state row {field_name}")
        if payload_value != row_value.astimezone(timezone.utc):
            raise ValueError(f"legacy state payload {field_name} does not match relational column")
    for field_name in (
        "schema_version",
        "producer",
        "model_version",
        "feature_version",
        "config_version",
    ):
        if payload.get(field_name) != getattr(row, field_name):
            raise ValueError(f"legacy state payload {field_name} does not match relational column")
    if payload.get("theme") != payload.get("entity_id"):
        raise ValueError("legacy membership theme does not match entity_id")
    if row.quality_severity != _legacy_quality_severity(payload):
        raise ValueError("state quality_severity does not match legacy payload")
    if row.content_hash != _legacy_payload_hash(payload):
        raise ValueError("state content_hash does not match legacy payload")
    return payload


def _legacy_theme_membership_contract(row: StateRecord) -> ThemeMembership:
    payload = _validate_legacy_theme_membership_row(row)
    theme_id = str(payload["theme"])
    converted = {
        "state_id": str(row.state_id),
        "state_type": StateType.THEME_MEMBERSHIP.value,
        "entity_id": theme_id,
        "as_of": payload["as_of"],
        "available_at": payload["available_at"],
        "generated_at": payload["generated_at"],
        "valid_until": payload["valid_until"],
        "source_window_start": payload.get("source_window_start", payload["as_of"]),
        "source_window_end": payload.get("source_window_end", payload["as_of"]),
        "schema_version": payload["schema_version"],
        "producer": payload["producer"],
        "model_version": payload["model_version"],
        "feature_version": payload["feature_version"],
        "config_version": payload["config_version"],
        "lineage_ids": payload.get("lineage_ids", []),
        "data_quality": payload.get("data_quality", {"issues": []}),
        "ticker": str(payload["ticker"]),
        "theme_id": theme_id,
        "weight": payload["membership_score"],
        "membership_version": str(
            payload.get("membership_version")
            or payload.get("taxonomy_version")
            or row.config_version
        ),
        "effective_from": payload.get("effective_from", payload["as_of"]),
        "effective_until": payload.get("effective_until"),
    }
    return ThemeMembership.model_validate(converted)


def _legacy_membership_query_condition():
    payload = StateRecord.payload
    return and_(
        payload.op("?")("ticker"),
        payload.op("?")("theme"),
        payload.op("?")("membership_score"),
        ~payload.op("?")("membership_scores"),
    )


def _state_type_query_condition(state_type: StateType):
    if state_type is StateType.THEME:
        return and_(
            StateRecord.state_type == StateType.THEME.value,
            ~_legacy_membership_query_condition(),
        )
    if state_type is not StateType.THEME_MEMBERSHIP:
        return StateRecord.state_type == state_type.value
    return or_(
        StateRecord.state_type == StateType.THEME_MEMBERSHIP.value,
        and_(
            StateRecord.state_type == StateType.THEME.value,
            _legacy_membership_query_condition(),
        ),
    )


def _contract_from_payload(row: StateRecord) -> StateEnvelope:
    state_type = StateType(row.state_type)
    if state_type is StateType.THEME and _is_legacy_theme_membership_payload(row.payload):
        return _legacy_theme_membership_contract(row)
    errors: list[str] = []
    for contract_type in _STATE_TYPES[state_type]:
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

    def get_state_candidates_for_snapshot(
        self,
        state_types: Sequence[StateType],
        decision_time: datetime,
    ) -> tuple[StateEnvelope, ...]:
        """Load the causal candidate pool in one query.

        Availability is the only time predicate applied in SQL.  Expiry and
        bar-bound checks intentionally remain in the pure snapshot selector so
        the builder can retain deterministic rejection evidence.  Rows that
        were not yet available at the decision time are excluded here; that
        keeps a later append from changing an already-built snapshot payload.
        """

        _validate_aware(decision_time, "decision_time")
        normalized_types = tuple(dict.fromkeys(state_types))
        if not normalized_types:
            return ()
        if any(not isinstance(state_type, StateType) for state_type in normalized_types):
            raise ValueError("state_types must contain only StateType values")
        type_conditions = [_state_type_query_condition(state_type) for state_type in normalized_types]
        stmt = (
            select(StateRecord)
            .where(
                or_(*type_conditions),
                StateRecord.available_at <= decision_time,
            )
            .order_by(
                StateRecord.state_type.asc(),
                StateRecord.entity_id.asc(),
                StateRecord.available_at.desc(),
                StateRecord.generated_at.desc(),
                StateRecord.state_id.desc(),
            )
        )
        rows = self._session.execute(stmt).scalars().all()
        return tuple(_contract_from_payload(row) for row in rows)

    def load_state_candidates(
        self,
        state_types: Sequence[StateType],
        decision_time: datetime,
    ) -> tuple[StateEnvelope, ...]:
        """Compatibility alias for the one-query snapshot candidate loader."""

        return self.get_state_candidates_for_snapshot(state_types, decision_time)

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
                _state_type_query_condition(state_type),
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
                _state_type_query_condition(state_type),
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
                _state_type_query_condition(request.state_type),
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
            row_type = row.state_type
            if row_type == StateType.THEME.value and _is_legacy_theme_membership_payload(
                row.payload
            ):
                row_type = StateType.THEME_MEMBERSHIP.value
            by_key.setdefault((row_type, row.entity_id), []).append(row)

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

    def get_context_snapshot(self, snapshot_id: UUID) -> ContextSnapshot | None:
        """Load and validate one persisted snapshot without changing the transaction."""

        row = self._session.get(ContextSnapshotRow, snapshot_id)
        if row is None:
            return None
        snapshot = ContextSnapshot.model_validate(row.payload)
        if snapshot.snapshot_id != row.snapshot_id:
            raise ValueError("snapshot payload snapshot_id does not match relational snapshot_id")
        if snapshot.content_hash != row.content_hash:
            raise ValueError("snapshot content_hash does not match relational content_hash")
        if snapshot.decision_time != row.decision_time:
            raise ValueError("snapshot decision_time does not match relational column")
        if snapshot.strategy_id != row.strategy_id or snapshot.ticker != row.ticker:
            raise ValueError("snapshot identity does not match relational columns")
        if snapshot.freshness_profile != row.freshness_profile:
            raise ValueError("snapshot freshness_profile does not match relational column")
        if snapshot.content_hash != snapshot.computed_content_hash():
            raise ValueError("snapshot content_hash does not match content")
        return snapshot

    def save_context_snapshot_idempotently(self, snapshot: ContextSnapshot) -> ContextSnapshot:
        """Persist a snapshot or return the identical existing snapshot."""

        existing = self.get_context_snapshot(snapshot.snapshot_id)
        if existing is not None:
            if existing.content_hash != snapshot.content_hash:
                raise ValueError("snapshot identity already exists with different content")
            return existing
        return self.save_context_snapshot(snapshot)
