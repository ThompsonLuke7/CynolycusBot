"""Say why a context snapshot came out invalid, using evidence it already has.

Every snapshot already records `requirement_results` (per rule: required?
FRESH/STALE/MISSING) and `rejected_candidates` (per discarded candidate: the
exact reason code). Nothing read them. A blocked order surfaced only as

    POLICY_VETO (SNAPSHOT_INVALID, SNAPSHOT_REQUIRED_STATE_MISSING)

which is a category, true of five different states for four different reasons,
and says nothing about which one or why.

The cost of that was concrete. The Meta Ranker's pre-open flush was blocked from
2026-08-18 to 2026-08-24 and every failed snapshot in that window carried, in
memory, the complete answer:

    MARKET   MARKET_SESSION_MISMATCH x16
    MARKET   FUTURE_BAR              x2

sixteen rows from the wrong session and two stamped after the decision bar —
the whole diagnosis, computed and discarded on every single order.

This module only renders. It logs nothing and decides nothing, so the policy
layer stays free of side effects and the same rendering serves both the live
runner's log and the pre-open preflight.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from core.nervous_system.contracts.context import ContextSnapshot


@dataclass(frozen=True)
class RequirementDiagnosis:
    """One freshness rule's verdict, with the evidence behind it."""

    state_type: str
    required: bool
    status: str
    reason_code: str
    entity_id: str
    #  reason_code -> how many candidates it discarded, worst first.
    rejections: tuple[tuple[str, int], ...] = ()

    @property
    def blocking(self) -> bool:
        return self.required and self.status != "FRESH"

    def describe(self) -> str:
        head = f"{self.state_type:<18} {'required' if self.required else 'optional':<8} {self.status:<8}"
        if self.status == "FRESH":
            # A rule that resolved needs no evidence. It will still have
            # rejections — every superseded or aged-out row is one — and
            # printing them reads as a fault where there is none.
            return head.rstrip()
        if not self.rejections:
            # Nothing of this type was even a candidate for this entity, which
            # is a different problem from a candidate that existed and was
            # refused: it means the producer never published, not that the
            # selector said no.
            return f"{head} nothing published"
        return f"{head} " + ", ".join(f"{code} x{count}" for code, count in self.rejections)


@dataclass(frozen=True)
class SnapshotDiagnosis:
    strategy_id: str
    entity_id: str
    decision_bar: str
    decision_time: str
    profile: str
    valid: bool
    requirements: tuple[RequirementDiagnosis, ...]

    @property
    def blocking(self) -> tuple[RequirementDiagnosis, ...]:
        return tuple(item for item in self.requirements if item.blocking)

    def describe(self, *, include_optional: bool = False) -> str:
        """One header line plus a line per rule. Blocking rules always shown."""

        header = (
            f"snapshot {'VALID' if self.valid else 'INVALID'} "
            f"{self.strategy_id}/{self.entity_id} "
            f"bar={self.decision_bar} time={self.decision_time} profile={self.profile}"
        )
        shown = [
            item for item in self.requirements
            if include_optional or item.required or item.blocking
        ]
        return "\n".join([header] + [f"  {item.describe()}" for item in shown])


def diagnose_snapshot(snapshot: ContextSnapshot) -> SnapshotDiagnosis:
    """Join a snapshot's requirement results to the candidates they rejected."""

    by_type: dict[str, Counter] = {}
    for candidate in snapshot.rejected_candidates:
        key = getattr(candidate.state_type, "value", str(candidate.state_type))
        by_type.setdefault(key, Counter())[candidate.reason_code] += 1

    requirements = []
    for result in snapshot.requirement_results:
        key = getattr(result.state_type, "value", str(result.state_type))
        counts = by_type.get(key, Counter())
        requirements.append(
            RequirementDiagnosis(
                state_type=key,
                required=bool(result.required),
                status=str(result.status),
                reason_code=str(result.reason_code),
                entity_id=str(result.entity_id),
                # Sorted by count then code so the same failure renders
                # identically every time and is greppable across days.
                rejections=tuple(
                    sorted(counts.items(), key=lambda item: (-item[1], item[0]))
                ),
            )
        )

    return SnapshotDiagnosis(
        strategy_id=str(snapshot.strategy_id),
        entity_id=str(snapshot.ticker),
        decision_bar=_stamp(getattr(snapshot, "decision_bar", None)),
        decision_time=_stamp(getattr(snapshot, "decision_time", None)),
        profile=str(snapshot.freshness_profile),
        valid=bool(snapshot.valid),
        requirements=tuple(
            sorted(requirements, key=lambda item: (not item.blocking, item.state_type))
        ),
    )


def _stamp(value) -> str:
    if value is None:
        return "?"
    try:
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    except AttributeError:
        return str(value)


__all__ = ["RequirementDiagnosis", "SnapshotDiagnosis", "diagnose_snapshot"]
