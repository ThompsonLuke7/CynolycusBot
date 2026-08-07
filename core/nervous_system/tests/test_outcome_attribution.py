"""Outcome attribution (Task 24).

Splits a realized result into the parts a human can act on: how much came from
the underlying moving, how much was lost between the price the decision assumed
and the price we actually got, how much went to fees, and how much is left over
because we traded a derivative instead of the underlying.

Two rules shape everything here.

*Broker fills are authoritative.* Realized P&L is computed from confirmed
filled quantity only. An order that was placed but not filled contributed
nothing, and treating requested quantity as filled quantity would invent
performance.

*The target window is fixed.* No observation after it may enter the outcome. A
position still open when the window closes is PENDING — never zero, because a
zero reads as a real flat result and would silently drag any aggregate toward
the middle.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from core.nervous_system.contracts.enums import OrderSide
from core.nervous_system.contracts.replay import (
    AttributionStatus,
    FillFact,
)
from core.nervous_system.replay.attribution import attribute_outcome


ENTRY = datetime(2026, 8, 3, 14, 0, tzinfo=timezone.utc)
EXIT = datetime(2026, 8, 10, 14, 0, tzinfo=timezone.utc)
TARGET = datetime(2026, 8, 11, 20, 0, tzinfo=timezone.utc)
D = Decimal


def _fill(**updates: object) -> FillFact:
    payload: dict[str, object] = {
        "leg_symbol": "AMD",
        "side": OrderSide.BUY,
        "quantity": D("100"),
        "price": D("100.00"),
        "filled_at": ENTRY,
        "fees": D("1.00"),
        "contract_multiplier": 1,
    }
    payload.update(updates)
    return FillFact(**payload)  # type: ignore[arg-type]


def _attribute(**updates: object):
    payload: dict[str, object] = {
        "entry_fills": (_fill(),),
        "exit_fills": (
            _fill(side=OrderSide.SELL, price=D("110.00"), filled_at=EXIT),
        ),
        "entry_reference_price": D("100.00"),
        "exit_reference_price": D("110.00"),
        "underlying_entry": D("100.00"),
        "underlying_exit": D("110.00"),
        "underlying_exposure": D("100"),
        "target_time": TARGET,
    }
    payload.update(updates)
    return attribute_outcome(**payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The decomposition is exact
# ---------------------------------------------------------------------------


def test_the_components_sum_to_the_realized_result() -> None:
    """An attribution whose parts do not add up to the whole is worse than no
    attribution: it invites people to trust a number that is missing something.
    """

    result = _attribute()

    assert (
        result.underlying_movement
        + result.slippage
        + result.instrument_transformation
        + result.fees
        == result.realized_pnl
    )


def test_a_clean_equity_round_trip_attributes_everything_to_the_underlying() -> None:
    result = _attribute()

    assert result.realized_pnl == D("998.00")  # 1000 gross, 2.00 of fees
    assert result.underlying_movement == D("1000.00")
    assert result.slippage == D("0")
    assert result.instrument_transformation == D("0")
    assert result.fees == D("-2.00")


def test_fees_are_a_negative_contribution() -> None:
    """Every component is a signed contribution to the realized result, so a
    cost has to read as negative rather than needing a sign convention in the
    reader's head.
    """

    assert _attribute().fees < 0


def test_paying_up_on_entry_is_charged_to_slippage_not_to_the_underlying() -> None:
    """The underlying did what it did; the difference between the price the
    decision assumed and the price we got is an execution cost.
    """

    result = _attribute(entry_fills=(_fill(price=D("100.50")),))

    assert result.slippage == D("-50.00")
    assert result.underlying_movement == D("1000.00")
    assert (
        result.underlying_movement
        + result.slippage
        + result.instrument_transformation
        + result.fees
        == result.realized_pnl
    )


def test_selling_below_the_reference_is_also_slippage() -> None:
    result = _attribute(
        exit_fills=(_fill(side=OrderSide.SELL, price=D("109.50"), filled_at=EXIT),)
    )

    assert result.slippage == D("-50.00")


def test_price_improvement_is_positive_slippage() -> None:
    result = _attribute(entry_fills=(_fill(price=D("99.50")),))

    assert result.slippage == D("50.00")


def test_the_residual_lands_in_instrument_transformation() -> None:
    """A call does not move one-for-one with its underlying. Whatever the
    underlying-equivalent exposure does not explain is labelled as the cost of
    trading the derivative rather than being quietly folded into the others.
    """

    result = _attribute(
        entry_fills=(_fill(price=D("4.00"), quantity=D("10"), contract_multiplier=100),),
        exit_fills=(
            _fill(
                side=OrderSide.SELL,
                price=D("6.00"),
                quantity=D("10"),
                contract_multiplier=100,
                filled_at=EXIT,
            ),
        ),
        entry_reference_price=D("4.00"),
        exit_reference_price=D("6.00"),
        underlying_exposure=D("450"),  # 10 contracts x 100 x 0.45 delta
    )

    assert result.realized_pnl == D("1998.00")
    assert result.underlying_movement == D("4500.00")
    assert result.instrument_transformation == D("-2500.00")
    assert (
        result.underlying_movement
        + result.slippage
        + result.instrument_transformation
        + result.fees
        == result.realized_pnl
    )


# ---------------------------------------------------------------------------
# Only confirmed fills count
# ---------------------------------------------------------------------------


def test_a_partial_fill_only_counts_the_quantity_that_filled() -> None:
    """Treating requested quantity as filled quantity would invent
    performance that no broker ever confirmed.
    """

    result = _attribute(
        entry_fills=(_fill(quantity=D("40")),),
        exit_fills=(
            _fill(side=OrderSide.SELL, quantity=D("40"), price=D("110.00"), filled_at=EXIT),
        ),
        underlying_exposure=D("40"),
    )

    assert result.realized_pnl == D("398.00")
    assert result.filled_entry_quantity == D("40")


def test_a_multi_leg_result_is_summed_per_leg() -> None:
    result = _attribute(
        entry_fills=(
            _fill(leg_symbol="LONG", quantity=D("10"), price=D("5.00"), contract_multiplier=100),
            _fill(leg_symbol="SHORT", side=OrderSide.SELL, quantity=D("10"), price=D("2.00"), contract_multiplier=100),
        ),
        exit_fills=(
            _fill(leg_symbol="LONG", side=OrderSide.SELL, quantity=D("10"), price=D("7.00"), contract_multiplier=100, filled_at=EXIT),
            _fill(leg_symbol="SHORT", quantity=D("10"), price=D("3.00"), contract_multiplier=100, filled_at=EXIT),
        ),
        entry_reference_price=D("3.00"),
        exit_reference_price=D("4.00"),
        underlying_exposure=D("300"),
    )

    # Long leg +2.00, short leg -1.00, net +1.00 x 10 x 100 = 1000, less 4.00 fees.
    assert result.realized_pnl == D("996.00")


def test_no_fills_at_all_is_pending_not_zero() -> None:
    """A zero would read as a real flat result and drag every aggregate built
    over it toward the middle.
    """

    result = _attribute(entry_fills=(), exit_fills=())

    assert result.status is AttributionStatus.PENDING
    assert result.realized_pnl is None


# ---------------------------------------------------------------------------
# The target window is fixed
# ---------------------------------------------------------------------------


def test_a_fill_after_the_target_window_never_enters_the_outcome() -> None:
    """The window defines what the outcome measures. Letting a later fill in
    would silently measure a different holding period than the one claimed.
    """

    late = _fill(
        side=OrderSide.SELL, price=D("200.00"), filled_at=TARGET + timedelta(days=1)
    )
    result = _attribute(exit_fills=(late,))

    assert result.status is AttributionStatus.PENDING
    assert result.realized_pnl is None
    assert result.excluded_fill_count == 1


def test_a_position_still_open_at_the_window_close_is_pending() -> None:
    result = _attribute(exit_fills=())

    assert result.status is AttributionStatus.PENDING
    assert result.realized_pnl is None


def test_a_fill_exactly_at_the_target_time_is_inside_the_window() -> None:
    result = _attribute(
        exit_fills=(_fill(side=OrderSide.SELL, price=D("110.00"), filled_at=TARGET),)
    )

    assert result.status is AttributionStatus.FINAL
    assert result.excluded_fill_count == 0


def test_a_fully_closed_position_inside_the_window_is_final() -> None:
    assert _attribute().status is AttributionStatus.FINAL


def test_a_partially_closed_position_is_pending() -> None:
    """Half an exit is not an outcome; reporting it as final would book a
    result for a position we still hold.
    """

    result = _attribute(
        exit_fills=(
            _fill(side=OrderSide.SELL, quantity=D("40"), price=D("110.00"), filled_at=EXIT),
        )
    )

    assert result.status is AttributionStatus.PENDING


# ---------------------------------------------------------------------------
# Boundary hygiene
# ---------------------------------------------------------------------------


def test_a_naive_fill_timestamp_is_refused() -> None:
    with pytest.raises(ValueError):
        _fill(filled_at=datetime(2026, 8, 3, 14, 0))


def test_a_non_positive_fill_quantity_is_refused() -> None:
    with pytest.raises(ValueError):
        _fill(quantity=D("0"))


def test_a_negative_fill_price_is_refused() -> None:
    with pytest.raises(ValueError):
        _fill(price=D("-1"))


def test_an_exit_before_its_entry_is_refused() -> None:
    with pytest.raises(ValueError, match="before"):
        _attribute(
            exit_fills=(
                _fill(
                    side=OrderSide.SELL,
                    price=D("110.00"),
                    filled_at=ENTRY - timedelta(days=1),
                ),
            )
        )


def test_option_slippage_is_scaled_by_the_contract_multiplier() -> None:
    """Five cents of slippage on 10 contracts is $50, not $0.50. Without the
    multiplier the execution cost of every option trade reads a hundred times
    too small.
    """

    result = _attribute(
        entry_fills=(_fill(price=D("4.05"), quantity=D("10"), contract_multiplier=100),),
        exit_fills=(
            _fill(
                side=OrderSide.SELL,
                price=D("6.00"),
                quantity=D("10"),
                contract_multiplier=100,
                filled_at=EXIT,
            ),
        ),
        entry_reference_price=D("4.00"),
        exit_reference_price=D("6.00"),
        underlying_exposure=D("450"),
    )

    assert result.slippage == D("-50.00")
    assert (
        result.underlying_movement
        + result.slippage
        + result.instrument_transformation
        + result.fees
        == result.realized_pnl
    )
