"""Causal selection of replay evidence (Task 24).

States already have a causal selector in context/requirements.py. This covers
the evidence that is *not* a state envelope — bars, option quotes, broker
fills, source manifests — under the same rules, because a replay that can see
one microsecond into the future is not a replay.

The distinction that does the most work here: `as_of` is business/event time
and never substitutes for availability. A bar stamped 16:00 that only landed in
our store at 16:07 was not knowable at 16:03, and a replay that treats the
event time as the availability time will quietly outperform reality.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, uuid5

import pytest

from core.nervous_system.contracts.replay import (
    MarkType,
    Observation,
    ObservationKind,
)
from core.nervous_system.replay.evidence import (
    observation_tie_key,
    select_causal_observations,
)


BAR = datetime(2026, 8, 3, 20, 0, tzinfo=timezone.utc)
DECIDE = datetime(2026, 8, 3, 20, 5, tzinfo=timezone.utc)


def _obs(name: str = "a", **updates: object) -> Observation:
    payload: dict[str, object] = {
        "observation_id": uuid5(NAMESPACE_URL, f"obs/{name}"),
        "kind": ObservationKind.BAR,
        "instrument": "AMD",
        "as_of": BAR,
        "available_at": BAR + timedelta(minutes=1),
        "valid_until": BAR + timedelta(hours=4),
        "generated_at": BAR + timedelta(minutes=1),
        "artifact_hash": "a" * 64,
        "record_locator": "Data/shared/bars/4h/AMD.parquet#2026-08-03T20:00:00Z",
        "mark_type": MarkType.QUOTE_BID_ASK,
        "provider": "alpaca",
        "feed": "sip",
        "tier": "verified",
        "schema_version": 1,
        "producer": "shared_bars",
        "bar_bound": True,
    }
    payload.update(updates)
    return Observation(**payload)  # type: ignore[arg-type]


def _select(*candidates: Observation, **updates: object):
    payload: dict[str, object] = {"decision_time": DECIDE, "decision_bar": BAR}
    payload.update(updates)
    return select_causal_observations(candidates, **payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The availability window
# ---------------------------------------------------------------------------


def test_an_observation_available_exactly_at_the_decision_time_is_eligible() -> None:
    """The lower bound is inclusive: something knowable at the instant we
    decided was knowable.
    """

    assert _select(_obs(available_at=DECIDE)) == (_obs(available_at=DECIDE),)


def test_an_observation_available_after_the_decision_is_excluded() -> None:
    """This is the whole point. Anything that landed later did not exist for
    the decision, however early its event time reads.
    """

    assert _select(_obs(available_at=DECIDE + timedelta(seconds=1))) == ()


def test_valid_until_is_exclusive() -> None:
    """An observation that expires exactly when we decide has already expired;
    an inclusive bound here would let a just-stale value be treated as live.
    """

    assert _select(_obs(valid_until=DECIDE)) == ()
    assert _select(_obs(valid_until=DECIDE + timedelta(seconds=1))) != ()


def test_a_late_revision_of_an_early_event_is_still_excluded() -> None:
    """A restated bar carries the original event time but a new availability
    time. Selecting on `as_of` would silently import the revision.
    """

    revision = _obs(
        "revision",
        as_of=BAR,
        available_at=DECIDE + timedelta(minutes=30),
        generated_at=DECIDE + timedelta(minutes=30),
    )

    assert _select(revision) == ()


# ---------------------------------------------------------------------------
# Event time is not availability
# ---------------------------------------------------------------------------


def test_a_bar_bound_observation_after_the_decision_bar_is_excluded() -> None:
    assert _select(_obs(as_of=BAR + timedelta(hours=4), bar_bound=True)) == ()


def test_a_bar_bound_observation_on_the_decision_bar_is_eligible() -> None:
    assert _select(_obs(as_of=BAR, bar_bound=True)) != ()


def test_an_intraday_observation_is_not_held_to_the_bar_boundary() -> None:
    """An option quote observed at 20:03 is legitimate evidence for a 20:05
    decision even though it post-dates the 20:00 bar. Only bar-bound evidence
    is clamped to the bar.
    """

    quote = _obs(
        "quote",
        kind=ObservationKind.OPTION_QUOTE,
        bar_bound=False,
        as_of=BAR + timedelta(minutes=3),
        available_at=BAR + timedelta(minutes=3),
    )

    assert _select(quote) == (quote,)


def test_a_future_event_time_never_rescues_an_unavailable_observation() -> None:
    assert (
        _select(
            _obs(
                as_of=BAR - timedelta(days=1),
                bar_bound=False,
                available_at=DECIDE + timedelta(minutes=1),
            )
        )
        == ()
    )


# ---------------------------------------------------------------------------
# Timestamp hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["as_of", "available_at", "valid_until", "generated_at"]
)
def test_a_naive_timestamp_is_refused(field: str) -> None:
    """A timestamp with no zone has no defined instant, so it cannot be ordered
    against a decision time at all.
    """

    with pytest.raises(ValueError):
        _obs(**{field: datetime(2026, 8, 3, 20, 0)})


@pytest.mark.parametrize("field", ["decision_time", "decision_bar"])
def test_a_naive_decision_boundary_is_refused(field: str) -> None:
    with pytest.raises(ValueError):
        _select(_obs(), **{field: datetime(2026, 8, 3, 20, 0)})


def test_an_offset_timestamp_is_normalised_not_rejected() -> None:
    """A real instant expressed in another zone is still a real instant."""

    eastern = datetime(2026, 8, 3, 16, 1, tzinfo=timezone(timedelta(hours=-4)))
    observation = _obs(available_at=eastern)

    assert observation.available_at == BAR + timedelta(minutes=1)
    assert _select(observation) != ()


def test_valid_until_must_follow_available_at() -> None:
    with pytest.raises(ValueError):
        _obs(available_at=BAR, valid_until=BAR - timedelta(seconds=1))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_selection_is_independent_of_input_order() -> None:
    """Replay must not depend on the order a provider happened to yield rows."""

    first = _obs("first", available_at=BAR + timedelta(minutes=1))
    second = _obs("second", available_at=BAR + timedelta(minutes=2))
    third = _obs("third", available_at=BAR + timedelta(minutes=3))

    assert _select(first, second, third) == _select(third, first, second)


def test_the_tie_break_is_the_shared_deterministic_key() -> None:
    """Same key as the state selector: availability, then generation, then
    producer version, then identity. Two providers agreeing on everything but
    identity must still resolve the same way every run.
    """

    same_time = {"available_at": BAR + timedelta(minutes=1), "generated_at": BAR}
    a = _obs("a", **same_time)
    b = _obs("b", **same_time)

    assert _select(a, b) == _select(b, a)
    assert observation_tie_key(a)[:3] == observation_tie_key(b)[:3]
    assert observation_tie_key(a)[3] != observation_tie_key(b)[3]


def test_results_are_ordered_by_availability() -> None:
    late = _obs("late", available_at=BAR + timedelta(minutes=3))
    early = _obs("early", available_at=BAR + timedelta(minutes=1))

    assert _select(late, early) == (early, late)


def test_appending_a_future_observation_does_not_change_the_past() -> None:
    """Future-append invariance: re-running an old decision after more data has
    arrived must produce exactly what it produced at the time.
    """

    known = _obs("known")
    future = _obs(
        "future",
        available_at=DECIDE + timedelta(days=1),
        valid_until=DECIDE + timedelta(days=2),
    )

    assert _select(known) == _select(known, future)


def test_a_duplicate_observation_is_not_double_counted() -> None:
    assert _select(_obs("a"), _obs("a")) == (_obs("a"),)


def test_an_empty_candidate_set_selects_nothing() -> None:
    assert _select() == ()
