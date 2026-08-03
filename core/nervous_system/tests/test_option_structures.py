"""Quote validation, structure templates, and coverage rules (Task 17)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from core.nervous_system.contracts.enums import (
    AssetClass,
    InstrumentFamily,
    OptionType,
    OrderSide,
    PositionIntent,
)
from core.nervous_system.contracts.states import PortfolioPosition
from core.nervous_system.execution.options.payoff import RiskReason
from core.nervous_system.execution.options.quotes import (
    OptionQuote,
    QuoteError,
    parse_occ_symbol,
)
from core.nervous_system.execution.options.structures import (
    SHARE_BACKED_FAMILIES,
    LegRole,
    StructureError,
    build_structure,
    validate_structure,
)
from core.nervous_system.tests.test_option_payoff import (
    EXPIRY,
    FAR_EXPIRY,
    QUOTE_AT,
    UNDERLYING,
    long_call,
    long_put,
    occ,
    leg as make_leg,
    short_call,
    short_put,
)


D = Decimal
UTC = timezone.utc


def quote(
    strike: str,
    option_type: OptionType,
    *,
    expiration: date = EXPIRY,
    bid: str = "4.90",
    ask: str = "5.10",
    **overrides,
) -> OptionQuote:
    strike_value = D(strike)
    payload = {
        "symbol": occ(strike_value, option_type, expiration),
        "underlying": UNDERLYING,
        "option_type": option_type,
        "strike": strike_value,
        "expiration": expiration,
        "quote_at": QUOTE_AT,
        "bid": D(bid),
        "ask": D(ask),
    }
    payload.update(overrides)
    return OptionQuote(**payload)


def call(strike: str, **kw) -> OptionQuote:
    return quote(strike, OptionType.CALL, **kw)


def put(strike: str, **kw) -> OptionQuote:
    return quote(strike, OptionType.PUT, **kw)


def shares(quantity: float = 100.0, entry: float | None = 200.0) -> PortfolioPosition:
    return PortfolioPosition(
        broker_position_id="pos-AMD",
        symbol=UNDERLYING,
        underlying=UNDERLYING,
        asset_class=AssetClass.EQUITY,
        quantity=quantity,
        average_entry_price=entry,
        market_value=quantity * 205.0,
    )


# --------------------------------------------------------------------------
# Step 1: quote validation
# --------------------------------------------------------------------------


def test_occ_symbol_round_trips() -> None:
    identity = parse_occ_symbol("AMD261218C00200000")
    assert identity.root == "AMD"
    assert identity.expiration == date(2026, 12, 18)
    assert identity.option_type is OptionType.CALL
    assert identity.strike == D("200")


@pytest.mark.parametrize(
    "symbol",
    [
        "AMD261218X00200000",
        "AMD26128C00200000",
        "261218C00200000",
        "AMD261218C0020000",
        "AMD261340C00200000",
        "AMD261218C00000000",
        "",
    ],
)
def test_invalid_occ_symbols_are_rejected(symbol: str) -> None:
    with pytest.raises(QuoteError):
        parse_occ_symbol(symbol)


def test_crossed_market_is_rejected() -> None:
    with pytest.raises(ValueError, match="crossed market"):
        call("200", bid="5.20", ask="5.10")


def test_zero_ask_is_not_a_tradable_market() -> None:
    with pytest.raises(ValueError, match="non-positive ask"):
        call("200", bid="0.00", ask="0.00")


def test_negative_prices_are_rejected() -> None:
    with pytest.raises(ValueError):
        call("200", bid="-0.01", ask="5.10")


def test_nonpositive_multiplier_is_rejected() -> None:
    with pytest.raises(ValueError):
        call("200", contract_multiplier=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("option_type", OptionType.PUT),
        ("strike", D("210")),
        ("expiration", date(2027, 1, 15)),
    ],
)
def test_fields_conflicting_with_occ_identity_are_rejected(field, value) -> None:
    """The symbol always says AMD 2026-12-18 C 200; the field disagrees."""

    payload = {
        "symbol": "AMD261218C00200000",
        "underlying": UNDERLYING,
        "option_type": OptionType.CALL,
        "strike": D("200"),
        "expiration": EXPIRY,
        "quote_at": QUOTE_AT,
        "bid": D("4.90"),
        "ask": D("5.10"),
    }
    payload[field] = value
    with pytest.raises(ValueError, match="conflicts with the OCC symbol"):
        OptionQuote(**payload)


def test_expiration_before_quote_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="expiration must not precede"):
        quote("200", OptionType.CALL, expiration=date(2026, 7, 29))


def test_trade_prints_are_kept_apart_from_the_mark() -> None:
    contract = call(
        "200",
        bid="4.90",
        ask="5.10",
        last_trade_price=D("1.00"),
        last_trade_at=QUOTE_AT - timedelta(days=6),
    )
    # The mark is the two-sided midpoint; a stale print never contaminates it.
    assert contract.mid == D("5.00")
    assert contract.last_trade_price == D("1.00")
    assert contract.spread == D("0.20")


def test_trade_price_and_time_must_arrive_together() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        call("200", last_trade_price=D("5.00"))


def test_trade_print_cannot_postdate_the_quote() -> None:
    with pytest.raises(ValueError, match="must not be after quote_at"):
        call(
            "200",
            last_trade_price=D("5.00"),
            last_trade_at=QUOTE_AT + timedelta(seconds=1),
        )


def test_quote_freshness_is_evaluated_against_an_explicit_time() -> None:
    contract = call("200")
    contract.require_fresh(QUOTE_AT + timedelta(seconds=30), D("60"))
    with pytest.raises(QuoteError, match="over the"):
        contract.require_fresh(QUOTE_AT + timedelta(seconds=120), D("60"))
    with pytest.raises(QuoteError, match="future"):
        contract.require_fresh(QUOTE_AT - timedelta(seconds=1), D("60"))


def test_spread_fraction_uses_the_mid() -> None:
    assert call("200", bid="4.00", ask="6.00").spread_fraction == D("0.4")


# --------------------------------------------------------------------------
# Step 6: structure templates
# --------------------------------------------------------------------------


def test_single_option_builds_one_long_leg() -> None:
    legs = build_structure(
        InstrumentFamily.SINGLE_OPTION,
        selected_contracts={LegRole.LONG_LEG: call("200")},
        quantity=2,
    )
    assert len(legs) == 1
    assert legs[0].side is OrderSide.BUY
    assert legs[0].position_intent is PositionIntent.BUY_TO_OPEN
    # Per-structure ratio; the 2 is the parent quantity, not baked into ratio.
    assert legs[0].ratio == 1


def test_vertical_builds_long_and_short_legs_in_order() -> None:
    legs = build_structure(
        InstrumentFamily.VERTICAL,
        selected_contracts={
            LegRole.LONG_LEG: call("200"),
            LegRole.SHORT_LEG: call("210"),
        },
        quantity=1,
    )
    assert [leg.side for leg in legs] == [OrderSide.BUY, OrderSide.SELL]
    assert [leg.strike for leg in legs] == [D("200"), D("210")]


def test_butterfly_body_carries_ratio_two() -> None:
    legs = build_structure(
        InstrumentFamily.BUTTERFLY,
        selected_contracts={
            LegRole.LOWER_WING: call("190"),
            LegRole.BODY: call("200"),
            LegRole.UPPER_WING: call("210"),
        },
        quantity=3,
    )
    # Ratios stay per-structure regardless of quantity, matching the
    # OrderRequest parent_quantity x ratio convention.
    assert [leg.ratio for leg in legs] == [1, 2, 1]


def test_quantity_is_not_multiplied_into_leg_ratios() -> None:
    """Baking quantity into ratio would scale intrinsic value but not the

    net debit, overstating max_profit by the structure count.
    """

    contracts = {LegRole.LONG_LEG: call("200"), LegRole.SHORT_LEG: call("210")}
    single = build_structure(
        InstrumentFamily.VERTICAL, selected_contracts=contracts, quantity=1
    )
    triple = build_structure(
        InstrumentFamily.VERTICAL, selected_contracts=contracts, quantity=3
    )
    assert [leg.ratio for leg in single] == [leg.ratio for leg in triple] == [1, 1]

    profile = validate_structure(triple, net_price=D("3.00"), quantity=3)
    assert profile.max_profit == D("2100.00")
    assert profile.max_loss == D("900.00")


def test_iron_condor_legs_are_put_put_call_call() -> None:
    legs = build_structure(
        InstrumentFamily.IRON_CONDOR,
        selected_contracts={
            LegRole.LOWER_WING: put("180"),
            LegRole.LOWER_BODY: put("190"),
            LegRole.UPPER_BODY: call("210"),
            LegRole.UPPER_WING: call("220"),
        },
        quantity=1,
    )
    assert [leg.option_type for leg in legs] == [
        OptionType.PUT,
        OptionType.PUT,
        OptionType.CALL,
        OptionType.CALL,
    ]
    assert [leg.side for leg in legs] == [
        OrderSide.BUY,
        OrderSide.SELL,
        OrderSide.SELL,
        OrderSide.BUY,
    ]


def test_build_is_deterministic_regardless_of_mapping_order() -> None:
    forward = build_structure(
        InstrumentFamily.VERTICAL,
        selected_contracts={
            LegRole.LONG_LEG: call("200"),
            LegRole.SHORT_LEG: call("210"),
        },
        quantity=1,
    )
    reverse = build_structure(
        InstrumentFamily.VERTICAL,
        selected_contracts={
            LegRole.SHORT_LEG: call("210"),
            LegRole.LONG_LEG: call("200"),
        },
        quantity=1,
    )
    assert forward == reverse


def test_closing_intents_are_emitted_when_requested() -> None:
    legs = build_structure(
        InstrumentFamily.VERTICAL,
        selected_contracts={
            LegRole.LONG_LEG: call("200"),
            LegRole.SHORT_LEG: call("210"),
        },
        quantity=1,
        closing=True,
    )
    assert [leg.position_intent for leg in legs] == [
        PositionIntent.SELL_TO_CLOSE,
        PositionIntent.BUY_TO_CLOSE,
    ]


def test_missing_or_extra_roles_are_rejected() -> None:
    with pytest.raises(StructureError, match="missing="):
        build_structure(
            InstrumentFamily.VERTICAL,
            selected_contracts={LegRole.LONG_LEG: call("200")},
            quantity=1,
        )


def test_unapproved_family_is_refused() -> None:
    with pytest.raises(StructureError, match="not an approved option structure"):
        build_structure(
            InstrumentFamily.ROLL,
            selected_contracts={LegRole.LONG_LEG: call("200")},
            quantity=1,
        )


@pytest.mark.parametrize(
    ("family", "contracts", "match"),
    [
        (
            InstrumentFamily.VERTICAL,
            {LegRole.LONG_LEG: call("200"), LegRole.SHORT_LEG: put("210")},
            "one option right",
        ),
        (
            InstrumentFamily.STRANGLE,
            {LegRole.LONG_PUT: put("220"), LegRole.LONG_CALL: call("210")},
            "put strike below the call strike",
        ),
        (
            InstrumentFamily.BUTTERFLY,
            {
                LegRole.LOWER_WING: call("210"),
                LegRole.BODY: call("200"),
                LegRole.UPPER_WING: call("190"),
            },
            "must ascend",
        ),
        (
            InstrumentFamily.IRON_BUTTERFLY,
            {
                LegRole.LOWER_WING: put("180"),
                LegRole.LOWER_BODY: put("190"),
                LegRole.UPPER_BODY: call("210"),
                LegRole.UPPER_WING: call("220"),
            },
            "shares one body strike",
        ),
        (
            InstrumentFamily.CALENDAR,
            {LegRole.SHORT_LEG: call("200"), LegRole.LONG_LEG: call("210", expiration=FAR_EXPIRY)},
            "one shared strike",
        ),
        (
            InstrumentFamily.DIAGONAL,
            {LegRole.SHORT_LEG: call("200"), LegRole.LONG_LEG: call("200", expiration=FAR_EXPIRY)},
            "different strikes",
        ),
    ],
)
def test_invalid_shapes_are_rejected(family, contracts, match) -> None:
    with pytest.raises(StructureError, match=match):
        build_structure(family, selected_contracts=contracts, quantity=1)


def test_calendar_requires_the_long_leg_to_expire_later() -> None:
    with pytest.raises(StructureError, match="expire after"):
        build_structure(
            InstrumentFamily.CALENDAR,
            selected_contracts={
                LegRole.SHORT_LEG: call("200", expiration=FAR_EXPIRY),
                LegRole.LONG_LEG: call("200"),
            },
            quantity=1,
        )


def test_mixed_multiplier_is_rejected() -> None:
    with pytest.raises(StructureError, match="one contract multiplier"):
        build_structure(
            InstrumentFamily.VERTICAL,
            selected_contracts={
                LegRole.LONG_LEG: call("200"),
                LegRole.SHORT_LEG: call("210", contract_multiplier=10),
            },
            quantity=1,
        )


# --------------------------------------------------------------------------
# Step 3: coverage, ratio, calendar, and roll safety
# --------------------------------------------------------------------------


def test_naked_short_call_is_rejected() -> None:
    profile = validate_structure((short_call("210"),), net_price=D("-5.00"))

    assert profile.valid is False
    assert RiskReason.NAKED_SHORT_CALL.value in profile.reason_codes
    assert RiskReason.UNBOUNDED_MAX_LOSS.value in profile.reason_codes


def test_naked_short_put_is_rejected_without_cash() -> None:
    profile = validate_structure(
        (short_put("190"),), net_price=D("-4.00"), available_cash=D("0")
    )

    assert profile.valid is False
    assert RiskReason.NAKED_SHORT_PUT.value in profile.reason_codes


def test_cash_secured_put_is_approved_with_full_assignment_cash() -> None:
    profile = validate_structure(
        (short_put("190"),), net_price=D("-4.00"), available_cash=D("19000")
    )

    assert profile.valid is True
    assert profile.max_loss == D("18600.00")
    assert profile.collateral == D("19000")
    assert profile.assignment_exposure == D("19000")


def test_covered_call_is_bounded_by_the_shares() -> None:
    profile = validate_structure(
        (short_call("210"),),
        net_price=D("-5.00"),
        existing_holdings=(shares(100.0, entry=200.0),),
    )

    assert profile.valid is True
    # Stock to zero: (200 cost - 5 credit) x 100.
    assert profile.max_loss == D("19500.00")
    # Called away: (210 - 200 + 5) x 100.
    assert profile.max_profit == D("1500.00")
    assert profile.breakevens == (D("195"),)


def test_covered_call_without_shares_is_naked() -> None:
    profile = validate_structure((short_call("210"),), net_price=D("-5.00"))
    assert profile.valid is False
    assert RiskReason.NAKED_SHORT_CALL.value in profile.reason_codes


def test_covered_call_with_unknown_cost_basis_is_rejected() -> None:
    profile = validate_structure(
        (short_call("210"),),
        net_price=D("-5.00"),
        existing_holdings=(shares(100.0, entry=None),),
    )

    assert profile.valid is False
    assert RiskReason.SHARE_COST_BASIS_UNKNOWN.value in profile.reason_codes


def test_collar_is_bounded_on_both_sides() -> None:
    legs = (long_put("190"), short_call("210"))
    profile = validate_structure(
        legs,
        net_price=D("0.00"),
        existing_holdings=(shares(100.0, entry=200.0),),
    )

    assert profile.valid is True
    assert profile.max_loss == D("1000.00")
    assert profile.max_profit == D("1000.00")


def test_uncovered_ratio_spread_is_rejected() -> None:
    legs = (long_call("200"), short_call("210", ratio=2))
    profile = validate_structure(legs, net_price=D("-1.00"))

    assert profile.valid is False
    assert RiskReason.UNCOVERED_RATIO.value in profile.reason_codes


def test_one_to_one_spread_is_not_a_ratio() -> None:
    profile = validate_structure(
        (long_call("200"), short_call("210")), net_price=D("3.00")
    )
    assert profile.valid is True
    assert profile.max_loss == D("300.00")


def test_short_straddle_is_rejected_even_with_partial_coverage() -> None:
    legs = (
        make_leg("200", OptionType.PUT, OrderSide.SELL),
        make_leg("200", OptionType.CALL, OrderSide.SELL),
    )
    profile = validate_structure(
        legs,
        net_price=D("-15.00"),
        existing_holdings=(shares(100.0, entry=200.0),),
        available_cash=D("0"),
    )

    assert profile.valid is False
    assert RiskReason.NAKED_SHORT_PUT.value in profile.reason_codes


def test_fully_secured_short_strangle_is_approved() -> None:
    legs = (
        make_leg("190", OptionType.PUT, OrderSide.SELL),
        make_leg("210", OptionType.CALL, OrderSide.SELL),
    )
    profile = validate_structure(
        legs,
        net_price=D("-8.00"),
        existing_holdings=(shares(100.0, entry=200.0),),
        available_cash=D("19000"),
    )

    assert profile.valid is True
    assert profile.max_profit is not None


# --- calendars and diagonals ------------------------------------------------


def test_long_debit_calendar_is_bounded_by_the_debit() -> None:
    legs = (
        short_call("200"),
        long_call("200", expiration=FAR_EXPIRY),
    )
    profile = validate_structure(legs, net_price=D("2.50"))

    assert profile.valid is True
    # Same strike, so no assignment strike gap.
    assert profile.max_loss == D("250.00")
    assert profile.max_profit is None, "a cross-expiry profit is not finitely bounded"
    assert profile.breakevens == ()


def test_long_diagonal_adds_the_adverse_strike_gap() -> None:
    legs = (
        short_call("200"),
        long_call("210", expiration=FAR_EXPIRY),
    )
    profile = validate_structure(legs, net_price=D("2.00"))

    assert profile.valid is True
    # Debit 2.00 plus the 10-point adverse gap between the short and its cover.
    assert profile.max_loss == D("1200.00")


def test_favourable_diagonal_gap_does_not_reduce_the_bound() -> None:
    legs = (
        short_call("210"),
        long_call("200", expiration=FAR_EXPIRY),
    )
    profile = validate_structure(legs, net_price=D("3.00"))

    assert profile.valid is True
    assert profile.max_loss == D("300.00")


def test_short_calendar_is_rejected() -> None:
    legs = (
        long_call("200"),
        make_leg("200", OptionType.CALL, OrderSide.SELL, expiration=FAR_EXPIRY),
    )
    profile = validate_structure(legs, net_price=D("-2.00"))

    assert profile.valid is False
    assert RiskReason.SHORT_CALENDAR_REJECTED.value in profile.reason_codes


def test_short_calendar_is_rejected_on_structure_not_merely_on_price() -> None:
    """Selling the back month is refused even when it is quoted as a debit.

    A short calendar normally prices as a credit, so a price-sign check alone
    would let an unusually quoted one through while the back-month short is
    still uncovered after the front leg expires.
    """

    legs = (
        long_call("200"),
        make_leg("200", OptionType.CALL, OrderSide.SELL, expiration=FAR_EXPIRY),
    )
    profile = validate_structure(legs, net_price=D("1.00"))

    assert profile.valid is False
    assert RiskReason.SHORT_CALENDAR_REJECTED.value in profile.reason_codes


def test_calendar_with_a_credit_is_rejected() -> None:
    legs = (short_call("200"), long_call("200", expiration=FAR_EXPIRY))
    profile = validate_structure(legs, net_price=D("-1.00"))

    assert profile.valid is False
    assert RiskReason.SHORT_CALENDAR_REJECTED.value in profile.reason_codes


def test_calendar_short_quantity_must_be_fully_covered() -> None:
    legs = (
        short_call("200", ratio=2),
        long_call("200", expiration=FAR_EXPIRY),
    )
    profile = validate_structure(legs, net_price=D("2.00"))

    assert profile.valid is False
    assert RiskReason.CALENDAR_QUANTITY_MISMATCH.value in profile.reason_codes


def test_calendar_cover_must_share_the_option_right() -> None:
    legs = (
        short_call("200"),
        long_put("200", expiration=FAR_EXPIRY),
    )
    profile = validate_structure(legs, net_price=D("2.00"))

    assert profile.valid is False
    assert RiskReason.CALENDAR_RIGHT_MISMATCH.value in profile.reason_codes


def test_calendar_fees_are_added_to_the_bound() -> None:
    legs = (short_call("200"), long_call("200", expiration=FAR_EXPIRY))
    profile = validate_structure(legs, net_price=D("2.50"), fees=D("1.30"))

    assert profile.max_loss == D("251.30")


# --- rolls and structural guards --------------------------------------------


def test_a_roll_cannot_be_one_atomic_order() -> None:
    """Eight legs are not a request; a roll is close then a separate open."""

    legs = tuple(
        make_leg(str(strike), OptionType.CALL, OrderSide.BUY)
        for strike in range(200, 208)
    )
    profile = validate_structure(legs, net_price=D("1.00"))

    assert profile.valid is False
    assert RiskReason.TOO_MANY_LEGS.value in profile.reason_codes


def test_empty_and_mixed_underlying_structures_are_rejected() -> None:
    assert RiskReason.EMPTY_STRUCTURE.value in validate_structure(
        (), net_price=D("1.00")
    ).reason_codes

    foreign = long_call("200").model_copy(update={"underlying": "NVDA"})
    profile = validate_structure((long_call("200"), foreign), net_price=D("1.00"))
    assert RiskReason.MIXED_UNDERLYING.value in profile.reason_codes


def test_share_backed_families_are_declared() -> None:
    assert SHARE_BACKED_FAMILIES == {
        InstrumentFamily.COVERED_CALL,
        InstrumentFamily.PROTECTIVE_PUT,
        InstrumentFamily.COLLAR,
    }


def test_quantity_scales_collateral_and_assignment_exposure() -> None:
    profile = validate_structure(
        (short_put("190"),),
        net_price=D("-4.00"),
        available_cash=D("38000"),
        quantity=2,
    )

    assert profile.valid is True
    assert profile.assignment_exposure == D("38000")
    assert profile.max_loss == D("37200.00")
