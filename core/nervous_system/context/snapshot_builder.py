"""Causal, versioned context-snapshot construction."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from uuid import NAMESPACE_URL, UUID, uuid5

from core.nervous_system.config.freshness import SnapshotProfile
from core.nervous_system.context.requirements import (
    SnapshotEntityScope,
    decision_session,
    evaluate_requirements,
)
from core.nervous_system.contracts.context import ContextSnapshot
from core.nervous_system.persistence.repositories.state import StateRepository


_SNAPSHOT_NAMESPACE = uuid5(NAMESPACE_URL, "cynolycus.nervous-system.context-snapshot.v1")


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _snapshot_id(
    *,
    strategy_id: str,
    entity_id: str,
    decision_time: datetime,
    decision_bar: datetime,
    profile_hash: str,
    ordered_state_hashes: tuple[str, ...],
) -> UUID:
    material = {
        "strategy_id": strategy_id,
        "entity_id": entity_id,
        "decision_time": decision_time.isoformat(),
        "decision_bar": decision_bar.isoformat(),
        "profile_hash": profile_hash,
        "selected_state_hashes": ordered_state_hashes,
    }
    return uuid5(
        _SNAPSHOT_NAMESPACE,
        json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False),
    )


class SnapshotBuilder:
    """Build and persist one immutable causal context snapshot."""

    def __init__(
        self,
        repository: StateRepository,
        *,
        market_entity_id: str = "US",
        portfolio_entity_id: str = "paper",
        readiness_entity_id: str = "nightly_data_readiness",
        sector_entity_ids: tuple[str, ...] = (),
    ) -> None:
        self._repository = repository
        self._scope = SnapshotEntityScope(
            market_entity_id=market_entity_id,
            portfolio_entity_id=portfolio_entity_id,
            readiness_entity_id=readiness_entity_id,
            sector_entity_ids=sector_entity_ids,
        )

    def build(
        self,
        *,
        strategy_id: str,
        entity_id: str,
        decision_time: datetime,
        decision_bar: datetime,
        profile: SnapshotProfile,
    ) -> ContextSnapshot:
        """Load candidates once, select causally, and persist idempotently."""

        if not isinstance(strategy_id, str) or not strategy_id.strip():
            raise ValueError("strategy_id must be a non-empty string")
        if not isinstance(entity_id, str) or not entity_id.strip():
            raise ValueError("entity_id must be a non-empty string")
        if not isinstance(profile, SnapshotProfile):
            raise TypeError("profile must be a SnapshotProfile")
        decision_time_utc = _aware(decision_time, "decision_time")
        decision_bar_utc = _aware(decision_bar, "decision_bar")
        if decision_bar_utc > decision_time_utc:
            raise ValueError("decision_bar must not be after decision_time")

        state_types = tuple(rule.state_type for rule in profile.rules)
        # This is intentionally one repository call.  It filters only future
        # availability, leaving validity, bar bounds, freshness, and profile
        # gates to the pure evaluator.
        candidates = self._repository.get_state_candidates_for_snapshot(
            state_types,
            decision_time_utc,
        )
        evaluation = evaluate_requirements(
            candidates,
            entity_id=entity_id,
            decision_time=decision_time_utc,
            decision_bar=decision_bar_utc,
            profile=profile,
            scope=self._scope,
        )

        session_label = decision_session(decision_time_utc)
        common = {
            "decision_time": decision_time_utc,
            "decision_bar": decision_bar_utc,
            "decision_session": session_label,
            "strategy_id": strategy_id,
            "ticker": entity_id,
            "states": evaluation.selected_states,
            "freshness_profile": profile.profile_id,
            "freshness_profile_hash": profile.profile_hash,
            "stale_inputs": evaluation.stale_inputs,
            "missing_inputs": evaluation.missing_inputs,
            "rejected_candidates": evaluation.rejected_candidates,
            "requirement_results": evaluation.requirement_results,
            "valid": evaluation.valid,
        }

        # ContextSnapshot canonicalizes the selected state ordering.  Build a
        # provisional contract solely to obtain those ordered state hashes for
        # the UUIDv5 idempotency key, then build the final contract with that ID.
        provisional = ContextSnapshot.from_states(
            snapshot_id=UUID(int=0),
            **common,
        )
        snapshot_id = _snapshot_id(
            strategy_id=strategy_id,
            entity_id=entity_id,
            decision_time=decision_time_utc,
            decision_bar=decision_bar_utc,
            profile_hash=profile.profile_hash,
            ordered_state_hashes=provisional.state_hashes,
        )
        snapshot = ContextSnapshot.from_states(
            snapshot_id=snapshot_id,
            **common,
        )
        return self._repository.save_context_snapshot_idempotently(snapshot)


__all__ = ["SnapshotBuilder"]
