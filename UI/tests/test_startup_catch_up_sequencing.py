"""The startup catch-ups run one at a time, readiness first.

2026-08-24. Both catch-ups launched as independent daemon threads and both
wanted the same heavy-job lock. Nightly won it by seven milliseconds:

    09:00:45,871 Nightly jobs: launching .../nightly_market_data.sh
    09:00:45,878 Data readiness: skipped (combined-server-data-readiness:
                 another heavy data job is already running)

The nightly held the lock until 14:24. Readiness — the job that decides whether
entries are authorized at all — was refused 35 consecutive times, ran at 15:07
and finished at 15:53, seven minutes before the close. The server had already
logged at 09:00:45 that the stamp "will NOT authorize entries", so the entire
session's entry decisions were made against a stamp known to be stale.

Ordering is the fix, and it is the right priority in both directions: readiness
gates entry orders and takes roughly 40 minutes, while nightly market data is
enrichment its own log calls "non-critical to entry readiness" and takes about
five hours.
"""
from __future__ import annotations

import UI.combined_server as cs


def test_the_steps_run_in_the_order_given_not_concurrently() -> None:
    order: list[str] = []

    cs._run_startup_catch_ups(
        (("readiness", lambda: order.append("readiness")),
         ("nightly", lambda: order.append("nightly"))),
    )

    assert order == ["readiness", "nightly"]


def test_a_slow_readiness_step_finishes_before_nightly_starts() -> None:
    """The whole point: nightly must not be running while readiness needs the
    lock. Sequencing is what guarantees it, so assert on the interleaving.
    """

    events: list[str] = []

    def _readiness() -> None:
        events.append("readiness:start")
        events.append("readiness:end")

    def _nightly() -> None:
        events.append("nightly:start")

    cs._run_startup_catch_ups((("readiness", _readiness), ("nightly", _nightly)))

    assert events.index("readiness:end") < events.index("nightly:start")


def test_a_failing_step_does_not_strand_the_one_behind_it() -> None:
    ran: list[str] = []

    def _boom() -> None:
        raise RuntimeError("readiness exploded")

    result = cs._run_startup_catch_ups(
        (("readiness", _boom), ("nightly", lambda: ran.append("nightly"))),
    )

    assert ran == ["nightly"], "nightly still ran"
    assert result == ["readiness", "nightly"]


def test_no_steps_is_a_no_op() -> None:
    assert cs._run_startup_catch_ups(()) == []


def test_readiness_runs_first_even_when_nightly_was_collected_first() -> None:
    """main() collects nightly (section 5a) before readiness (section 5a1), so
    relying on collection order would put the five-hour enrichment job ahead of
    the job that authorizes entries — exactly the 2026-08-24 failure.
    """

    ordered = cs.order_startup_catch_ups(
        [("nightly", lambda: None), ("readiness", lambda: None)],
    )

    assert [name for name, _ in ordered] == ["readiness", "nightly"]


def test_an_unlisted_step_sorts_after_readiness() -> None:
    """A step added later must not be able to displace readiness by accident."""

    ordered = cs.order_startup_catch_ups(
        [("something-new", lambda: None), ("readiness", lambda: None)],
    )

    assert [name for name, _ in ordered][0] == "readiness"
