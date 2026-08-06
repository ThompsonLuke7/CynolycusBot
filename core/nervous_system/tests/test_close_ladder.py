"""Exit pricing: walk the mid-to-bid ladder before ever sending a market order.

A market close always fills but always pays the spread. Working down from the
mid to the bid first captures whatever price improvement is available, and only
then does the market order act as the guarantee that the position actually
closes.

The rungs are a pure function of the observed market, so an exit is reproducible
from the audit record.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from core.nervous_system.execution.options.close_ladder import (
    TICK,
    close_limit_ladder,
)


D = Decimal


def test_the_ladder_starts_at_the_mid_and_ends_at_the_bid() -> None:
    rungs = close_limit_ladder(mid=D("4.20"), bid=D("4.00"), attempts=5)

    assert rungs[0] == D("4.20")
    assert rungs[-1] == D("4.00")


def test_the_ladder_only_ever_walks_down() -> None:
    """A rung above its predecessor would re-offer a price the market already
    declined.
    """

    rungs = close_limit_ladder(mid=D("4.20"), bid=D("4.00"), attempts=5)

    assert list(rungs) == sorted(rungs, reverse=True)
    assert len(set(rungs)) == len(rungs)


def test_every_rung_is_on_a_penny() -> None:
    rungs = close_limit_ladder(mid=D("4.207"), bid=D("4.001"), attempts=5)

    assert all(rung == rung.quantize(TICK) for rung in rungs)


def test_the_ladder_honours_the_attempt_budget() -> None:
    assert len(close_limit_ladder(mid=D("4.20"), bid=D("4.00"), attempts=3)) <= 3


def test_a_one_penny_spread_collapses_to_the_distinct_rungs() -> None:
    """Rounding must not manufacture duplicate rungs; each one has to be a real
    new offer.
    """

    rungs = close_limit_ladder(mid=D("4.005"), bid=D("4.00"), attempts=5)

    assert len(set(rungs)) == len(rungs)
    assert rungs[-1] == D("4.00")


def test_a_mid_equal_to_the_bid_yields_a_single_rung() -> None:
    assert close_limit_ladder(mid=D("4.00"), bid=D("4.00"), attempts=5) == (D("4.00"),)


def test_an_unknown_bid_still_walks_down_from_the_mid() -> None:
    """A degraded close may have a mark but no bid. It should still try to earn
    price improvement rather than jumping straight to a market order.
    """

    rungs = close_limit_ladder(mid=D("4.20"), bid=None, attempts=4)

    assert rungs[0] == D("4.20")
    assert len(rungs) == 4
    assert list(rungs) == sorted(rungs, reverse=True)
    assert rungs[-1] < D("4.20")


def test_no_rung_is_ever_worth_less_than_a_penny() -> None:
    rungs = close_limit_ladder(mid=D("0.02"), bid=D("0.01"), attempts=6)

    assert all(rung >= TICK for rung in rungs)


def test_a_worthless_mid_yields_the_minimum_offer() -> None:
    assert close_limit_ladder(mid=D("0.001"), bid=None, attempts=3) == (TICK,)


@pytest.mark.parametrize("attempts", [0, -1])
def test_a_non_positive_budget_still_yields_one_rung(attempts: int) -> None:
    """Never return an empty ladder: the caller would have nothing to submit
    and the exit would silently not happen.
    """

    assert len(close_limit_ladder(mid=D("4.20"), bid=D("4.00"), attempts=attempts)) == 1


def test_a_crossed_market_is_refused() -> None:
    with pytest.raises(ValueError, match="crossed"):
        close_limit_ladder(mid=D("4.00"), bid=D("4.50"), attempts=3)


def test_a_non_positive_mid_is_refused() -> None:
    with pytest.raises(ValueError):
        close_limit_ladder(mid=D("0"), bid=None, attempts=3)


def test_rounding_never_produces_a_repeated_rung() -> None:
    """A narrow spread across many attempts makes the interpolated rungs round
    onto each other. Re-offering a price the market already declined wastes an
    attempt from a budget that ends in a market order, so duplicates are
    dropped rather than submitted twice.
    """

    rungs = close_limit_ladder(mid=D("4.03"), bid=D("4.00"), attempts=6)

    assert len(set(rungs)) == len(rungs)
    assert list(rungs) == sorted(rungs, reverse=True)
    assert rungs[0] == D("4.03")
    assert rungs[-1] == D("4.00")
