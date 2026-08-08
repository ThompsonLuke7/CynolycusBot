"""The evidence boundary: providers enforce their own cutoff.

This is the anti-leakage mechanism. The causal rules in ``evidence.py`` are
advisory as long as strategy code can reach past them — a function you have to
remember to call is not a guarantee. A provider never hands out an unrestricted,
future-loaded collection; it holds the whole corpus privately and only ever
returns what was knowable at the decision point it was asked about.

The clock is the other half. It advances only through scheduled decision
points, so there is no way to ask for evidence "as of now" and quietly get a
different answer on a second run.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timezone

from core.nervous_system.contracts.replay import Observation, ObservationKind
from core.nervous_system.replay.evidence import select_causal_observations


class ReplayBoundaryError(RuntimeError):
    """Raised when a caller tries to reach outside the replay's cutoff."""


def _aware(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


class ReplayClock:
    """Advances only through scheduled decision points, never freely.

    A replay whose clock can be set to an arbitrary instant is a replay whose
    results depend on who set it. The schedule is fixed up front and the clock
    walks it in order.
    """

    def __init__(self, decision_points: Sequence[tuple[datetime, datetime]]) -> None:
        if not decision_points:
            raise ValueError("a replay needs at least one scheduled decision point")
        normalised: list[tuple[datetime, datetime]] = []
        for decision_time, decision_bar in decision_points:
            time_utc = _aware(decision_time, "decision_time")
            bar_utc = _aware(decision_bar, "decision_bar")
            if bar_utc > time_utc:
                raise ValueError("a decision cannot be scheduled before its bar")
            normalised.append((time_utc, bar_utc))
        if normalised != sorted(normalised):
            # Out-of-order points would let a later decision see evidence an
            # earlier one could not, which is leakage wearing a schedule.
            raise ValueError("decision points must be in chronological order")
        self._points = tuple(normalised)
        self._index = -1

    def __iter__(self):
        for index in range(len(self._points)):
            self._index = index
            yield self._points[index]

    @property
    def position(self) -> int:
        return self._index

    @property
    def current(self) -> tuple[datetime, datetime]:
        if self._index < 0:
            raise ReplayBoundaryError("the replay clock has not started")
        return self._points[self._index]

    def __len__(self) -> int:
        return len(self._points)


class ReplayEvidenceProvider:
    """Serves evidence for one decision point, and nothing beyond it.

    The corpus is private on purpose. There is deliberately no accessor that
    returns everything: handing a caller the full set is exactly how a replay
    ends up reading its own future.
    """

    def __init__(self, observations: Iterable[Observation]) -> None:
        corpus: list[Observation] = []
        for observation in observations:
            if not isinstance(observation, Observation):
                raise TypeError("evidence must be Observation contracts")
            corpus.append(observation)
        self._corpus = tuple(corpus)

    def _visible(
        self, *, decision_time: datetime, decision_bar: datetime
    ) -> tuple[Observation, ...]:
        return select_causal_observations(
            self._corpus, decision_time=decision_time, decision_bar=decision_bar
        )

    def _of_kind(
        self,
        kind: ObservationKind,
        *,
        decision_time: datetime,
        decision_bar: datetime,
        instrument: str | None = None,
    ) -> tuple[Observation, ...]:
        visible = self._visible(decision_time=decision_time, decision_bar=decision_bar)
        return tuple(
            observation
            for observation in visible
            if observation.kind is kind
            and (instrument is None or observation.instrument == instrument)
        )

    # Separate methods per evidence type, as the brief requires: a single
    # "give me everything" call is what lets a caller quietly widen its reach.

    def states(self, **kwargs) -> tuple[Observation, ...]:
        return self._of_kind(ObservationKind.STATE, **kwargs)

    def bars(self, **kwargs) -> tuple[Observation, ...]:
        return self._of_kind(ObservationKind.BAR, **kwargs)

    def option_quotes(self, **kwargs) -> tuple[Observation, ...]:
        return self._of_kind(ObservationKind.OPTION_QUOTE, **kwargs)

    def broker_fills(self, **kwargs) -> tuple[Observation, ...]:
        return self._of_kind(ObservationKind.BROKER_FILL, **kwargs)

    def source_manifest(self, **kwargs) -> tuple[Observation, ...]:
        return self._of_kind(ObservationKind.SOURCE_MANIFEST, **kwargs)

    def next_executable_bar(
        self, *, instrument: str, after: datetime
    ) -> Observation | None:
        """The first bar strictly after ``after`` for one instrument.

        Strictly after, never at: a closed-bar signal fills from the next
        observation, and filling at the signal bar itself is the classic
        same-bar leak.
        """

        boundary = _aware(after, "after")
        candidates = [
            observation
            for observation in self._corpus
            if observation.kind is ObservationKind.BAR
            and observation.instrument == instrument
            and observation.as_of > boundary
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda observation: (observation.as_of, observation.available_at))


__all__ = ["ReplayBoundaryError", "ReplayClock", "ReplayEvidenceProvider"]
