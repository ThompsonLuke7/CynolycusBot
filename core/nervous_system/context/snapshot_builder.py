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
    select_peer_memberships,
)
from core.nervous_system.contracts.context import ContextSnapshot
from core.nervous_system.contracts.enums import StateType
from core.nervous_system.contracts.states import PortfolioState
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

    def held_underlyings(self, *, decision_time: datetime) -> tuple[str, ...]:
        """Distinct underlyings held at ``decision_time``, newest state wins.

        The peer set for a decision IS the set of things already owned: those
        are the positions a concentration limit has to count.  Resolved from
        the registry's own PORTFOLIO state rather than from the broker, so the
        same answer replays later.

        Returns ``()`` when no portfolio state is available.  That degrades to
        today's behaviour -- every holding buckets to UNALLOCATED -- rather
        than blocking a decision on a missing peer set.
        """

        decision_time_utc = _aware(decision_time, "decision_time")
        candidates = self._repository.get_state_candidates_for_snapshot(
            (StateType.PORTFOLIO,), decision_time_utc
        )
        portfolios = [
            candidate
            for candidate in candidates
            if isinstance(candidate, PortfolioState)
            and candidate.entity_id == self._scope.portfolio_entity_id
            and decision_time_utc < candidate.valid_until
        ]
        if not portfolios:
            return ()
        latest = max(portfolios, key=lambda state: (state.available_at, state.as_of))
        return tuple(
            sorted({position.underlying for position in latest.positions if position.underlying})
        )

    def build(
        self,
        *,
        strategy_id: str,
        entity_id: str,
        decision_time: datetime,
        decision_bar: datetime,
        profile: SnapshotProfile,
        peer_entity_ids: tuple[str, ...] = (),
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

        # Advisory and additive: with no peers requested this selects nothing
        # and the snapshot -- including its content hash -- is byte-identical to
        # what the same inputs produced before peers existed.
        peer_memberships = select_peer_memberships(
            candidates,
            peer_entity_ids=peer_entity_ids,
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
            "peer_theme_memberships": peer_memberships,
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
        persisted = self._repository.save_context_snapshot_idempotently(snapshot)
        # End the transaction here, before returning.
        #
        # This builder owns a single long-lived session (build_router creates it
        # once via `SnapshotBuilder(StateRepository(session_factory()))`), and
        # the insert above takes a lock on the new snapshot_id that is held
        # until the transaction ends. The DecisionCoordinator then persists the
        # decision chain through a SEPARATE UnitOfWork, and `save_chain` inserts
        # the SAME snapshot_id — `ON CONFLICT DO NOTHING` still has to wait on
        # an uncommitted conflicting key, so that second session blocked on
        # this one, while this one waited on the caller that was blocked on the
        # second. One process, two connections, deadlocked against itself, with
        # no lock_timeout to break it.
        #
        # That is not hypothetical: meta_ranker's 2026-08-26 09:35 pre-open
        # flush hung exactly this way and was still holding the lock 13 hours
        # later. See research/daily_live_reports/2026-08-26.md.
        #
        # Committing also stops a read-only build from pinning an idle
        # transaction open for the life of the process, which blocks vacuum.
        self._repository.commit()
        return persisted


__all__ = ["SnapshotBuilder"]
