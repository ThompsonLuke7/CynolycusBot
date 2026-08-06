"""Causal selection of replay evidence that is not a state envelope.

States are selected by ``core.nervous_system.context.requirements``; this
applies the same rules to bars, option quotes, broker fills, and source
manifests so there is one causal discipline rather than two.

Pure: no clock, no IO. Every boundary is passed in, because a selector that
reads a clock cannot be replayed.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from uuid import UUID

from core.nervous_system.contracts.replay import Observation


def _aware(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        # A timestamp with no zone has no defined instant, so it cannot be
        # ordered against a decision time at all.
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def observation_tie_key(
    observation: Observation,
) -> tuple[datetime, datetime, str, str]:
    """The shared deterministic key: availability, generation, version, identity."""

    return (
        observation.available_at,
        observation.generated_at,
        observation.producer_version(),
        str(observation.observation_id),
    )


def is_causally_available(
    observation: Observation,
    *,
    decision_time: datetime,
    decision_bar: datetime,
) -> bool:
    """Whether one observation was knowable at the decision point."""

    decided = _aware(decision_time, "decision_time")
    bar = _aware(decision_bar, "decision_bar")
    if observation.available_at > decided:
        return False
    # Exclusive upper bound: something that expires exactly when we decide has
    # already expired.
    if decided >= observation.valid_until:
        return False
    if observation.bar_bound and observation.as_of > bar:
        return False
    return True


def select_causal_observations(
    candidates: Sequence[Observation],
    *,
    decision_time: datetime,
    decision_bar: datetime,
) -> tuple[Observation, ...]:
    """Return every causally available observation, deterministically ordered.

    Ordering and de-duplication are both required for replay: the result must
    not depend on the order a provider happened to yield rows, and the same
    observation supplied twice is still one observation.
    """

    eligible: dict[UUID, Observation] = {}
    for candidate in candidates:
        if not isinstance(candidate, Observation):
            raise TypeError("candidates must be Observation contracts")
        if is_causally_available(
            candidate, decision_time=decision_time, decision_bar=decision_bar
        ):
            eligible[candidate.observation_id] = candidate
    return tuple(sorted(eligible.values(), key=observation_tie_key))


__all__ = [
    "is_causally_available",
    "observation_tie_key",
    "select_causal_observations",
]
