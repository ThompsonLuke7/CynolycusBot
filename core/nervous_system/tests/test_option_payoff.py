"""Exact expiry payoff and finite loss bounds (Task 17).

Every assertion compares exact ``Decimal`` values.  Approximate float
comparison would hide precisely the sign and rounding errors these bounds
exist to prevent.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from core.nervous_system.contracts.enums import (
    OptionType,
    OrderSide,
    PositionIntent,
)
from core.nervous_system.contracts.orders import OptionLeg
from core.nervous_system.execution.options.payoff import (
    OptionRiskProfile,
    RiskReason,
    expiry_payoff,
    payoff_per_share,
    same_expiry_risk_profile,
    upper_tail_slope,
)


UTC = timezone.utc
QUOTE_AT = datetime(2026, 7, 30, 18, 0, tzinfo=UTC)
EXPIRY = date(2026, 12, 18)
FAR_EXPIRY = date(2027, 1, 15)
UNDERLYING = "AMD"
D = Decimal


def occ(strike: Decimal, option_type: OptionType, expiration: date = EXPIRY) -> str:
    right = "C" if option_type is OptionType.CALL else "P"
    thousandths = int(strike * 1000)
    return f"{UNDERLYING}{expiration:%y%m%d}{right}{thousandths:08d}"


def leg(
    strike: str,
    option_type: OptionType,
    side: OrderSide,
    *,
    ratio: int = 1,
    expiration: date = EXPIRY,
) -> OptionLeg:
    strike_value = D(strike)
    return OptionLeg(
        symbol=occ(strike_value, option_type, expiration),
        underlying=UNDERLYING,
        option_type=option_type,
        strike=strike_value,
        expiration=expiration,
        side=side,
        ratio=ratio,
        position_intent=(
            PositionIntent.BUY_TO_OPEN
            if side is OrderSide.BUY
            else PositionIntent.SELL_TO_OPEN
        ),
        quote_at=QUOTE_AT,
        bid=D("1.00"),
        ask=D("1.10"),
    )


def long_call(strike: str = "200", **kw) -> OptionLeg:
    return leg(strike, OptionType.CALL, OrderSide.BUY, **kw)


def short_call(strike: str = "210", **kw) -> OptionLeg:
    return leg(strike, OptionType.CALL, OrderSide.SELL, **kw)


def long_put(strike: str = "200", **kw) -> OptionLeg:
    return leg(strike, OptionType.PUT, OrderSide.BUY, **kw)


def short_put(strike: str = "190", **kw) -> OptionLeg:
    return leg(strike, OptionType.PUT, OrderSide.SELL, **kw)


# --------------------------------------------------------------------------
# Single options
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("underlying_price", "expected"),
    [
        ("0", "-800.00"),
        ("150", "-800.00"),
        ("200", "-800.00"),
        ("208", "0.00"),
        ("220", "1200.00"),
        ("300", "9200.00"),
    ],
)
def test_long_call_payoff_table(underlying_price: str, expected: str) -> None:
    payoff = expiry_payoff(
        (long_call("200"),),
        underlying_price=D(underlying_price),
        net_debit=D("8.00"),
    )
    assert payoff == D(expected)


@pytest.mark.parametrize(
    ("underlying_price", "expected"),
    [
        ("0", "19300.00"),
        ("100", "9300.00"),
        ("193", "0.00"),
        ("200", "-700.00"),
        ("250", "-700.00"),
    ],
)
def test_long_put_payoff_table(underlying_price: str, expected: str) -> None:
    payoff = expiry_payoff(
        (long_put("200"),),
        underlying_price=D(underlying_price),
        net_debit=D("7.00"),
    )
    assert payoff == D(expected)


def test_long_call_bounds_are_premium_and_unbounded_upside() -> None:
    max_profit, max_loss, breakevens, bounded = same_expiry_risk_profile(
        (long_call("200"),), net_debit=D("8.00")
    )
    assert bounded is True
    assert max_profit is None, "a long call has an unbounded profit tail"
    assert max_loss == D("800.00")
    assert breakevens == (D("208"),)


def test_long_put_max_profit_is_capped_at_a_zero_underlying() -> None:
    max_profit, max_loss, breakevens, bounded = same_expiry_risk_profile(
        (long_put("200"),), net_debit=D("7.00")
    )
    assert bounded is True
    assert max_profit == D("19300.00")
    assert max_loss == D("700.00")
    assert breakevens == (D("193"),)


# --------------------------------------------------------------------------
# Verticals
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("underlying_price", "expected"),
    [
        ("0", "-300.00"),
        ("200", "-300.00"),
        ("203", "0.00"),
        ("210", "700.00"),
        ("400", "700.00"),
    ],
)
def test_debit_call_vertical_payoff_table(underlying_price: str, expected: str) -> None:
    legs = (long_call("200"), short_call("210"))
    assert expiry_payoff(
        legs, underlying_price=D(underlying_price), net_debit=D("3.00")
    ) == D(expected)


def test_debit_call_vertical_bounds() -> None:
    max_profit, max_loss, breakevens, bounded = same_expiry_risk_profile(
        (long_call("200"), short_call("210")), net_debit=D("3.00")
    )
    assert bounded is True
    assert max_profit == D("700.00")
    assert max_loss == D("300.00")
    assert breakevens == (D("203"),)
    assert max_profit + max_loss == D("1000.00"), "profit plus loss equals the width"


@pytest.mark.parametrize(
    ("underlying_price", "expected"),
    [
        ("0", "-700.00"),
        ("190", "-700.00"),
        ("197", "0.00"),
        ("200", "300.00"),
        ("400", "300.00"),
    ],
)
def test_credit_put_vertical_payoff_table(underlying_price: str, expected: str) -> None:
    legs = (short_put("200"), long_put("190"))
    assert expiry_payoff(
        legs, underlying_price=D(underlying_price), net_debit=D("-3.00")
    ) == D(expected)


def test_credit_put_vertical_bounds() -> None:
    max_profit, max_loss, breakevens, bounded = same_expiry_risk_profile(
        (short_put("200"), long_put("190")), net_debit=D("-3.00")
    )
    assert bounded is True
    assert max_profit == D("300.00")
    assert max_loss == D("700.00")
    assert breakevens == (D("197"),)


# --------------------------------------------------------------------------
# Straddles and strangles
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("underlying_price", "expected"),
    [
        ("0", "18500.00"),
        ("185", "0.00"),
        ("200", "-1500.00"),
        ("215", "0.00"),
        ("300", "8500.00"),
    ],
)
def test_long_straddle_payoff_table(underlying_price: str, expected: str) -> None:
    legs = (long_put("200"), long_call("200"))
    assert expiry_payoff(
        legs, underlying_price=D(underlying_price), net_debit=D("15.00")
    ) == D(expected)


def test_long_straddle_has_two_breakevens_and_capped_loss() -> None:
    max_profit, max_loss, breakevens, bounded = same_expiry_risk_profile(
        (long_put("200"), long_call("200")), net_debit=D("15.00")
    )
    assert bounded is True
    assert max_profit is None
    assert max_loss == D("1500.00")
    assert breakevens == (D("185"), D("215"))


def test_long_strangle_payoff_and_bounds() -> None:
    legs = (long_put("190"), long_call("210"))
    assert expiry_payoff(
        legs, underlying_price=D("200"), net_debit=D("9.00")
    ) == D("-900.00")
    max_profit, max_loss, breakevens, bounded = same_expiry_risk_profile(
        legs, net_debit=D("9.00")
    )
    assert bounded is True
    assert max_profit is None
    assert max_loss == D("900.00")
    assert breakevens == (D("181"), D("219"))


def test_naked_short_straddle_is_reported_as_unbounded() -> None:
    legs = (
        leg("200", OptionType.PUT, OrderSide.SELL),
        leg("200", OptionType.CALL, OrderSide.SELL),
    )
    max_profit, max_loss, breakevens, bounded = same_expiry_risk_profile(
        legs, net_debit=D("-15.00")
    )
    assert bounded is False, "an uncovered short call tail is unbounded"
    assert upper_tail_slope(legs) == D("-1")


# --------------------------------------------------------------------------
# Butterflies and condors
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("underlying_price", "expected"),
    [
        ("0", "-200.00"),
        ("190", "-200.00"),
        ("192", "0.00"),
        ("200", "800.00"),
        ("208", "0.00"),
        ("210", "-200.00"),
        ("400", "-200.00"),
    ],
)
def test_call_butterfly_payoff_table(underlying_price: str, expected: str) -> None:
    legs = (
        long_call("190"),
        short_call("200", ratio=2),
        long_call("210"),
    )
    assert expiry_payoff(
        legs, underlying_price=D(underlying_price), net_debit=D("2.00")
    ) == D(expected)


def test_call_butterfly_bounds_are_finite_on_both_tails() -> None:
    legs = (long_call("190"), short_call("200", ratio=2), long_call("210"))
    max_profit, max_loss, breakevens, bounded = same_expiry_risk_profile(
        legs, net_debit=D("2.00")
    )
    assert bounded is True
    assert upper_tail_slope(legs) == D("0")
    assert max_profit == D("800.00")
    assert max_loss == D("200.00")
    assert breakevens == (D("192"), D("208"))


@pytest.mark.parametrize(
    ("underlying_price", "expected"),
    [
        ("0", "-600.00"),
        ("190", "-600.00"),
        ("196", "0.00"),
        ("200", "400.00"),
        ("204", "0.00"),
        ("210", "-600.00"),
    ],
)
def test_iron_butterfly_payoff_table(underlying_price: str, expected: str) -> None:
    legs = (
        long_put("190"),
        leg("200", OptionType.PUT, OrderSide.SELL),
        short_call("200"),
        long_call("210"),
    )
    assert expiry_payoff(
        legs, underlying_price=D(underlying_price), net_debit=D("-4.00")
    ) == D(expected)


def test_iron_condor_payoff_and_bounds() -> None:
    legs = (
        long_put("180"),
        leg("190", OptionType.PUT, OrderSide.SELL),
        short_call("210"),
        long_call("220"),
    )
    net_debit = D("-3.00")
    assert expiry_payoff(legs, underlying_price=D("200"), net_debit=net_debit) == D("300.00")
    assert expiry_payoff(legs, underlying_price=D("170"), net_debit=net_debit) == D("-700.00")
    assert expiry_payoff(legs, underlying_price=D("230"), net_debit=net_debit) == D("-700.00")

    max_profit, max_loss, breakevens, bounded = same_expiry_risk_profile(
        legs, net_debit=net_debit
    )
    assert bounded is True
    assert max_profit == D("300.00")
    assert max_loss == D("700.00")
    assert breakevens == (D("187"), D("213"))


def test_call_condor_bounds() -> None:
    legs = (
        long_call("180"),
        short_call("190"),
        leg("210", OptionType.CALL, OrderSide.SELL),
        long_call("220"),
    )
    max_profit, max_loss, breakevens, bounded = same_expiry_risk_profile(
        legs, net_debit=D("4.00")
    )
    assert bounded is True
    assert upper_tail_slope(legs) == D("0")
    assert max_profit == D("600.00")
    assert max_loss == D("400.00")


# --------------------------------------------------------------------------
# Quantity and multiplier scaling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("quantity", [1, 3, 10])
def test_bounds_scale_linearly_with_quantity(quantity: int) -> None:
    legs = (long_call("200"), short_call("210"))
    max_profit, max_loss, _, bounded = same_expiry_risk_profile(
        legs, net_debit=D("3.00"), quantity=quantity
    )
    assert bounded is True
    assert max_profit == D("700.00") * quantity
    assert max_loss == D("300.00") * quantity


def test_non_default_multiplier_scales_payoff() -> None:
    assert expiry_payoff(
        (long_call("200"),),
        underlying_price=D("220"),
        net_debit=D("8.00"),
        contract_multiplier=10,
    ) == D("120.00")


def test_breakevens_are_exact_with_fractional_premium() -> None:
    _, _, breakevens, _ = same_expiry_risk_profile(
        (long_call("200"),), net_debit=D("8.25")
    )
    assert breakevens == (D("208.25"),)


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_expiry_payoff_refuses_mixed_expirations() -> None:
    legs = (short_call("200"), long_call("200", expiration=FAR_EXPIRY))
    with pytest.raises(ValueError, match="single expiration"):
        expiry_payoff(legs, underlying_price=D("200"), net_debit=D("2.00"))


def test_expiry_payoff_requires_legs_and_positive_multiplier() -> None:
    with pytest.raises(ValueError, match="at least one leg"):
        expiry_payoff((), underlying_price=D("200"), net_debit=D("1.00"))
    with pytest.raises(ValueError, match="multiplier must be positive"):
        expiry_payoff(
            (long_call(),),
            underlying_price=D("200"),
            net_debit=D("1.00"),
            contract_multiplier=0,
        )


def test_payoff_per_share_is_the_unscaled_payoff() -> None:
    assert payoff_per_share((long_call("200"),), D("220"), D("8.00")) == D("12.00")


def test_risk_profile_contract_rejects_inconsistent_validity() -> None:
    with pytest.raises(ValueError, match="cannot carry rejection reasons"):
        OptionRiskProfile(
            valid=True,
            max_profit=None,
            max_loss=D("0"),
            breakevens=(),
            net_debit=D("0"),
            collateral=D("0"),
            assignment_exposure=D("0"),
            contract_multiplier=100,
            quantity=1,
            reason_codes=(RiskReason.NAKED_SHORT_CALL.value,),
        )
    with pytest.raises(ValueError, match="requires a reason code"):
        OptionRiskProfile(
            valid=False,
            max_profit=None,
            max_loss=D("0"),
            breakevens=(),
            net_debit=D("0"),
            collateral=D("0"),
            assignment_exposure=D("0"),
            contract_multiplier=100,
            quantity=1,
            reason_codes=(),
        )
