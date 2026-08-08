"""The evidence boundary and the replay clock (Task 24).

The causal rules in evidence.py are advisory as long as strategy code can reach
past them. These pin the boundary that makes them binding: the corpus is
private, every accessor is scoped to one decision point, and the clock walks a
fixed schedule instead of being set to whatever instant a caller likes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.nervous_system.contracts.replay import (
    MarkType,
    Observation,
    ObservationKind,
)
from core.nervous_system.replay.provider import (
    ReplayBoundaryError,
    ReplayClock,
    ReplayEvidenceProvider,
)


BAR = datetime(2026, 8, 3, 20, 0, tzinfo=timezone.utc)
DECIDE = BAR + timedelta(minutes=5)


def _obs(name: str, **updates) -> Observation:
    payload = {
        "observation_id": __import__("uuid").uuid5(__import__("uuid").NAMESPACE_URL, name),
        "kind": ObservationKind.BAR,
        "instrument": "AMD",
        "as_of": BAR,
        "available_at": BAR + timedelta(minutes=1),
        "valid_until": BAR + timedelta(days=30),
        "generated_at": BAR + timedelta(minutes=1),
        "artifact_hash": "a" * 64,
        "record_locator": f"locator/{name}",
        "provider": "alpaca",
        "feed": "sip",
        "tier": "verified",
        "schema_version": 1,
        "producer": "shared_bars",
        "mark_type": MarkType.QUOTE_BID_ASK,
        "bar_bound": True,
    }
    payload.update(updates)
    return Observation(**payload)


def _provider(*observations) -> ReplayEvidenceProvider:
    return ReplayEvidenceProvider(observations or (_obs("a"),))


def _at(provider, method="bars", **updates):
    payload = {"decision_time": DECIDE, "decision_bar": BAR}
    payload.update(updates)
    return getattr(provider, method)(**payload)


# ---------------------------------------------------------------------------
# The corpus is not reachable
# ---------------------------------------------------------------------------


def test_the_provider_exposes_no_way_to_read_everything() -> None:
    """A "give me the whole set" accessor is exactly how a replay ends up
    reading its own future. There must not be one.
    """

    public = {name for name in dir(_provider()) if not name.startswith("_")}

    assert public == {
        "states", "bars", "option_quotes", "broker_fills",
        "source_manifest", "next_executable_bar",
    }


def test_every_accessor_requires_a_decision_point() -> None:
    provider = _provider()

    for method in ("states", "bars", "option_quotes", "broker_fills", "source_manifest"):
        with pytest.raises(TypeError):
            getattr(provider, method)()


def test_evidence_from_after_the_decision_is_not_served(tmp_path=None) -> None:
    future = _obs(
        "future",
        available_at=DECIDE + timedelta(hours=1),
        valid_until=DECIDE + timedelta(days=2),
    )

    assert _at(_provider(_obs("known"), future)) == (_obs("known"),)


def test_each_evidence_type_has_its_own_accessor() -> None:
    """Separate methods, so widening what a caller reads is a visible change
    rather than a different argument to one catch-all call.
    """

    provider = _provider(
        _obs("bar"),
        _obs("quote", kind=ObservationKind.OPTION_QUOTE, bar_bound=False),
        _obs("fill", kind=ObservationKind.BROKER_FILL, bar_bound=False),
    )

    assert len(_at(provider, "bars")) == 1
    assert len(_at(provider, "option_quotes")) == 1
    assert len(_at(provider, "broker_fills")) == 1
    assert _at(provider, "states") == ()


def test_an_accessor_can_be_narrowed_to_one_instrument() -> None:
    provider = _provider(_obs("amd"), _obs("msft", instrument="MSFT"))

    served = _at(provider, "bars", instrument="MSFT")

    assert [o.instrument for o in served] == ["MSFT"]


def test_the_provider_refuses_anything_that_is_not_an_observation() -> None:
    with pytest.raises(TypeError):
        ReplayEvidenceProvider([{"instrument": "AMD"}])


# ---------------------------------------------------------------------------
# Next executable bar
# ---------------------------------------------------------------------------


def test_the_next_executable_bar_is_strictly_after_the_signal() -> None:
    """Filling at the signal bar itself is the classic same-bar leak."""

    later = _obs("later", as_of=BAR + timedelta(hours=4))
    provider = _provider(_obs("signal"), later)

    assert provider.next_executable_bar(instrument="AMD", after=BAR) == later


def test_there_may_be_no_next_bar() -> None:
    """A signal at the end of the data has no fill, and saying so beats
    inventing one.
    """

    assert _provider().next_executable_bar(instrument="AMD", after=BAR) is None


def test_the_next_bar_is_the_earliest_one_after_the_signal() -> None:
    near = _obs("near", as_of=BAR + timedelta(hours=4))
    far = _obs("far", as_of=BAR + timedelta(days=1))
    provider = _provider(far, near)

    assert provider.next_executable_bar(instrument="AMD", after=BAR) == near


# ---------------------------------------------------------------------------
# The clock
# ---------------------------------------------------------------------------


def test_the_clock_walks_its_schedule_in_order() -> None:
    points = [(DECIDE, BAR), (DECIDE + timedelta(hours=4), BAR + timedelta(hours=4))]
    clock = ReplayClock(points)

    assert list(clock) == points
    assert len(clock) == 2


def test_an_out_of_order_schedule_is_refused() -> None:
    """A later decision seeing evidence an earlier one could not is leakage
    wearing a schedule.
    """

    with pytest.raises(ValueError, match="chronological"):
        ReplayClock([(DECIDE + timedelta(hours=4), BAR + timedelta(hours=4)), (DECIDE, BAR)])


def test_a_decision_scheduled_before_its_bar_is_refused() -> None:
    with pytest.raises(ValueError, match="before its bar"):
        ReplayClock([(BAR - timedelta(minutes=1), BAR)])


def test_an_empty_schedule_is_refused() -> None:
    with pytest.raises(ValueError):
        ReplayClock([])


def test_a_naive_decision_point_is_refused() -> None:
    with pytest.raises(ValueError):
        ReplayClock([(datetime(2026, 8, 3, 20, 5), BAR)])


def test_reading_the_clock_before_it_starts_is_an_error() -> None:
    """There is no "now" until the schedule has been entered."""

    with pytest.raises(ReplayBoundaryError):
        ReplayClock([(DECIDE, BAR)]).current


def test_the_clock_reports_where_it_is() -> None:
    clock = ReplayClock([(DECIDE, BAR), (DECIDE + timedelta(hours=4), BAR + timedelta(hours=4))])

    for index, point in enumerate(clock):
        assert clock.position == index
        assert clock.current == point
