"""Approved option structures and their coverage rules.

Every structure this module will build has a determinable, finite maximum
loss.  Naked shorts and uncovered ratios are refused in every environment, not
merely sized down: a structure whose worst case cannot be established is not
tradable at any size.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from core.nervous_system.contracts.enums import (
    InstrumentFamily,
    OptionType,
    OrderSide,
    PositionIntent,
)
from core.nervous_system.contracts.orders import OptionLeg
from core.nervous_system.contracts.states import PortfolioPosition

from .payoff import (
    DEFAULT_CONTRACT_MULTIPLIER,
    OptionRiskProfile,
    RiskReason,
    _share_coverage,
    same_expiry_risk_profile,
    share_cost_basis,
)
from .quotes import OptionQuote


_ZERO = Decimal("0")


class StructureError(ValueError):
    """Raised when a structure cannot be built as specified."""


class LegRole(str, Enum):
    """Semantic position of a leg inside its structure."""

    LONG_LEG = "LONG_LEG"
    SHORT_LEG = "SHORT_LEG"
    LONG_CALL = "LONG_CALL"
    SHORT_CALL = "SHORT_CALL"
    LONG_PUT = "LONG_PUT"
    SHORT_PUT = "SHORT_PUT"
    LOWER_WING = "LOWER_WING"
    LOWER_BODY = "LOWER_BODY"
    UPPER_BODY = "UPPER_BODY"
    UPPER_WING = "UPPER_WING"
    BODY = "BODY"


@dataclass(frozen=True)
class LegSpec:
    role: LegRole
    side: OrderSide
    ratio: int = 1


_BUY = OrderSide.BUY
_SELL = OrderSide.SELL

# Ordered leg templates.  Order is the emitted leg order, so it is stable and
# semantic rather than dependent on quote iteration.
STRUCTURE_TEMPLATES: Mapping[InstrumentFamily, tuple[LegSpec, ...]] = {
    InstrumentFamily.SINGLE_OPTION: (LegSpec(LegRole.LONG_LEG, _BUY),),
    InstrumentFamily.VERTICAL: (
        LegSpec(LegRole.LONG_LEG, _BUY),
        LegSpec(LegRole.SHORT_LEG, _SELL),
    ),
    InstrumentFamily.CALENDAR: (
        LegSpec(LegRole.SHORT_LEG, _SELL),
        LegSpec(LegRole.LONG_LEG, _BUY),
    ),
    InstrumentFamily.DIAGONAL: (
        LegSpec(LegRole.SHORT_LEG, _SELL),
        LegSpec(LegRole.LONG_LEG, _BUY),
    ),
    InstrumentFamily.STRADDLE: (
        LegSpec(LegRole.LONG_PUT, _BUY),
        LegSpec(LegRole.LONG_CALL, _BUY),
    ),
    InstrumentFamily.STRANGLE: (
        LegSpec(LegRole.LONG_PUT, _BUY),
        LegSpec(LegRole.LONG_CALL, _BUY),
    ),
    InstrumentFamily.BUTTERFLY: (
        LegSpec(LegRole.LOWER_WING, _BUY),
        LegSpec(LegRole.BODY, _SELL, ratio=2),
        LegSpec(LegRole.UPPER_WING, _BUY),
    ),
    InstrumentFamily.IRON_BUTTERFLY: (
        LegSpec(LegRole.LOWER_WING, _BUY),
        LegSpec(LegRole.LOWER_BODY, _SELL),
        LegSpec(LegRole.UPPER_BODY, _SELL),
        LegSpec(LegRole.UPPER_WING, _BUY),
    ),
    InstrumentFamily.CONDOR: (
        LegSpec(LegRole.LOWER_WING, _BUY),
        LegSpec(LegRole.LOWER_BODY, _SELL),
        LegSpec(LegRole.UPPER_BODY, _SELL),
        LegSpec(LegRole.UPPER_WING, _BUY),
    ),
    InstrumentFamily.IRON_CONDOR: (
        LegSpec(LegRole.LOWER_WING, _BUY),
        LegSpec(LegRole.LOWER_BODY, _SELL),
        LegSpec(LegRole.UPPER_BODY, _SELL),
        LegSpec(LegRole.UPPER_WING, _BUY),
    ),
    InstrumentFamily.COVERED_CALL: (LegSpec(LegRole.SHORT_CALL, _SELL),),
    InstrumentFamily.CASH_SECURED_PUT: (LegSpec(LegRole.SHORT_PUT, _SELL),),
    InstrumentFamily.PROTECTIVE_PUT: (LegSpec(LegRole.LONG_PUT, _BUY),),
    InstrumentFamily.COLLAR: (
        LegSpec(LegRole.LONG_PUT, _BUY),
        LegSpec(LegRole.SHORT_CALL, _SELL),
    ),
}

# Alpaca cannot combine equities and options in one order, so these require
# verified existing shares rather than a synthetic combined request.
SHARE_BACKED_FAMILIES = frozenset(
    {
        InstrumentFamily.COVERED_CALL,
        InstrumentFamily.PROTECTIVE_PUT,
        InstrumentFamily.COLLAR,
    }
)

_MULTI_EXPIRY_FAMILIES = frozenset(
    {InstrumentFamily.CALENDAR, InstrumentFamily.DIAGONAL}
)

_OPENING_INTENT = {
    OrderSide.BUY: PositionIntent.BUY_TO_OPEN,
    OrderSide.SELL: PositionIntent.SELL_TO_OPEN,
}
_CLOSING_INTENT = {
    OrderSide.BUY: PositionIntent.BUY_TO_CLOSE,
    OrderSide.SELL: PositionIntent.SELL_TO_CLOSE,
}
_OPPOSITE_SIDE = {OrderSide.BUY: OrderSide.SELL, OrderSide.SELL: OrderSide.BUY}


def build_structure(
    structure: InstrumentFamily,
    *,
    selected_contracts: Mapping[LegRole, OptionQuote],
    quantity: int,
    closing: bool = False,
) -> tuple[OptionLeg, ...]:
    """Build the deterministic leg tuple for one approved structure."""

    template = STRUCTURE_TEMPLATES.get(structure)
    if template is None:
        raise StructureError(f"{structure.value} is not an approved option structure")
    if quantity <= 0:
        raise StructureError("quantity must be positive")

    expected = {spec.role for spec in template}
    supplied = set(selected_contracts)
    if supplied != expected:
        missing = sorted(role.value for role in expected - supplied)
        extra = sorted(role.value for role in supplied - expected)
        raise StructureError(
            f"{structure.value} requires roles {sorted(role.value for role in expected)}; "
            f"missing={missing} unexpected={extra}"
        )

    legs: list[OptionLeg] = []
    for spec in template:
        quote = selected_contracts[spec.role]
        # The template describes the open position.  Closing it trades each leg
        # in the opposite direction, so the side inverts along with the intent.
        side = spec.side if not closing else _OPPOSITE_SIDE[spec.side]
        intent = (_CLOSING_INTENT if closing else _OPENING_INTENT)[side]
        legs.append(
            OptionLeg(
                symbol=quote.symbol,
                underlying=quote.underlying,
                option_type=quote.option_type,
                strike=quote.strike,
                expiration=quote.expiration,
                side=side,
                ratio=spec.ratio * quantity,
                position_intent=intent,
                quote_at=quote.quote_at,
                bid=quote.bid,
                ask=quote.ask,
            )
        )

    _validate_shape(structure, tuple(legs), selected_contracts)
    return tuple(legs)


def _validate_shape(
    structure: InstrumentFamily,
    legs: tuple[OptionLeg, ...],
    contracts: Mapping[LegRole, OptionQuote],
) -> None:
    underlyings = {leg.underlying for leg in legs}
    if len(underlyings) != 1:
        raise StructureError("every leg must share one underlying")
    multipliers = {quote.contract_multiplier for quote in contracts.values()}
    if len(multipliers) != 1:
        raise StructureError("every leg must share one contract multiplier")

    expirations = {leg.expiration for leg in legs}
    if structure in _MULTI_EXPIRY_FAMILIES:
        near = contracts[LegRole.SHORT_LEG]
        far = contracts[LegRole.LONG_LEG]
        if near.option_type is not far.option_type:
            raise StructureError("calendars and diagonals require one option right")
        if far.expiration <= near.expiration:
            raise StructureError("the long leg must expire after the short leg")
        if structure is InstrumentFamily.CALENDAR and far.strike != near.strike:
            raise StructureError("a calendar requires one shared strike")
        if structure is InstrumentFamily.DIAGONAL and far.strike == near.strike:
            raise StructureError("a diagonal requires different strikes")
        return

    if len(expirations) != 1:
        raise StructureError(f"{structure.value} requires one shared expiration")

    if structure is InstrumentFamily.VERTICAL:
        long_leg, short_leg = contracts[LegRole.LONG_LEG], contracts[LegRole.SHORT_LEG]
        if long_leg.option_type is not short_leg.option_type:
            raise StructureError("a vertical requires one option right")
        if long_leg.strike == short_leg.strike:
            raise StructureError("a vertical requires different strikes")
    elif structure is InstrumentFamily.STRADDLE:
        if contracts[LegRole.LONG_PUT].strike != contracts[LegRole.LONG_CALL].strike:
            raise StructureError("a straddle requires one shared strike")
    elif structure is InstrumentFamily.STRANGLE:
        if contracts[LegRole.LONG_PUT].strike >= contracts[LegRole.LONG_CALL].strike:
            raise StructureError("a strangle requires the put strike below the call strike")
    elif structure is InstrumentFamily.BUTTERFLY:
        lower = contracts[LegRole.LOWER_WING]
        body = contracts[LegRole.BODY]
        upper = contracts[LegRole.UPPER_WING]
        if len({lower.option_type, body.option_type, upper.option_type}) != 1:
            raise StructureError("a butterfly requires one option right")
        if not lower.strike < body.strike < upper.strike:
            raise StructureError("butterfly strikes must ascend wing, body, wing")
    elif structure is InstrumentFamily.CONDOR:
        strikes = _body_strikes(contracts)
        if len({quote.option_type for quote in contracts.values()}) != 1:
            raise StructureError("a condor requires one option right")
        if not strikes[0] < strikes[1] < strikes[2] < strikes[3]:
            raise StructureError("condor strikes must strictly ascend")
    elif structure in {InstrumentFamily.IRON_BUTTERFLY, InstrumentFamily.IRON_CONDOR}:
        _validate_iron(structure, contracts)
    elif structure is InstrumentFamily.COLLAR:
        if contracts[LegRole.LONG_PUT].strike >= contracts[LegRole.SHORT_CALL].strike:
            raise StructureError("a collar requires the put strike below the call strike")
        if contracts[LegRole.LONG_PUT].option_type is not OptionType.PUT:
            raise StructureError("a collar requires a long put")
        if contracts[LegRole.SHORT_CALL].option_type is not OptionType.CALL:
            raise StructureError("a collar requires a short call")


def _body_strikes(contracts: Mapping[LegRole, OptionQuote]) -> tuple[Decimal, ...]:
    return (
        contracts[LegRole.LOWER_WING].strike,
        contracts[LegRole.LOWER_BODY].strike,
        contracts[LegRole.UPPER_BODY].strike,
        contracts[LegRole.UPPER_WING].strike,
    )


def _validate_iron(
    structure: InstrumentFamily,
    contracts: Mapping[LegRole, OptionQuote],
) -> None:
    lower_wing = contracts[LegRole.LOWER_WING]
    lower_body = contracts[LegRole.LOWER_BODY]
    upper_body = contracts[LegRole.UPPER_BODY]
    upper_wing = contracts[LegRole.UPPER_WING]
    if (
        lower_wing.option_type is not OptionType.PUT
        or lower_body.option_type is not OptionType.PUT
    ):
        raise StructureError("the lower legs of an iron structure must be puts")
    if (
        upper_body.option_type is not OptionType.CALL
        or upper_wing.option_type is not OptionType.CALL
    ):
        raise StructureError("the upper legs of an iron structure must be calls")
    if not lower_wing.strike < lower_body.strike:
        raise StructureError("the put wing must sit below the short put")
    if not upper_body.strike < upper_wing.strike:
        raise StructureError("the call wing must sit above the short call")
    if structure is InstrumentFamily.IRON_BUTTERFLY:
        if lower_body.strike != upper_body.strike:
            raise StructureError("an iron butterfly shares one body strike")
    elif lower_body.strike >= upper_body.strike:
        raise StructureError("an iron condor requires the short put below the short call")


def validate_structure(
    legs: Sequence[OptionLeg],
    *,
    net_price: Decimal,
    existing_holdings: Sequence[PortfolioPosition] = (),
    available_cash: Decimal = _ZERO,
    contract_multiplier: int = DEFAULT_CONTRACT_MULTIPLIER,
    quantity: int = 1,
    fees: Decimal = _ZERO,
) -> OptionRiskProfile:
    """Establish a finite risk profile, or reject with stable reason codes.

    ``net_price`` is the per-share net debit: positive when premium is paid,
    negative when premium is received.
    """

    reasons: list[RiskReason] = []
    if not legs:
        return _rejected([RiskReason.EMPTY_STRUCTURE], net_price, contract_multiplier, quantity)
    if len(legs) > 4:
        return _rejected([RiskReason.TOO_MANY_LEGS], net_price, contract_multiplier, quantity)
    if len({leg.underlying for leg in legs}) != 1:
        return _rejected([RiskReason.MIXED_UNDERLYING], net_price, contract_multiplier, quantity)

    underlying = legs[0].underlying
    share_cover = _share_coverage(existing_holdings, underlying, contract_multiplier)

    reasons.extend(
        _coverage_reasons(
            legs,
            share_cover=share_cover,
            available_cash=available_cash,
            contract_multiplier=contract_multiplier,
            quantity=quantity,
        )
    )

    assignment_exposure = sum(
        (
            leg.strike * Decimal(leg.ratio) * Decimal(contract_multiplier)
            for leg in legs
            if leg.side is OrderSide.SELL
        ),
        _ZERO,
    )

    expirations = {leg.expiration for leg in legs}
    if len(expirations) > 1:
        return _multi_expiry_profile(
            legs,
            net_price=net_price,
            reasons=reasons,
            assignment_exposure=assignment_exposure,
            contract_multiplier=contract_multiplier,
            quantity=quantity,
            fees=fees,
        )

    # Shares only enter the payoff when they are actually doing the covering.
    # A long call held alongside unrelated shares is not a share-backed
    # structure, so its profile must not borrow their upside.
    covering = _covering_contracts(legs, share_cover)
    cost_basis = _ZERO
    if covering > _ZERO:
        basis = share_cost_basis(existing_holdings, underlying)
        if basis is None:
            reasons.append(RiskReason.SHARE_COST_BASIS_UNKNOWN)
        else:
            cost_basis = basis

    max_profit, max_loss, breakevens, bounded = same_expiry_risk_profile(
        legs,
        net_debit=net_price,
        contract_multiplier=contract_multiplier,
        quantity=quantity,
        stock_contracts=covering,
        stock_cost_basis=cost_basis,
    )
    if not bounded:
        reasons.append(RiskReason.UNBOUNDED_MAX_LOSS)
    if reasons:
        return _rejected(reasons, net_price, contract_multiplier, quantity)

    max_loss = max_loss + fees
    return OptionRiskProfile(
        valid=True,
        max_profit=max_profit,
        max_loss=max_loss,
        breakevens=breakevens,
        net_debit=net_price,
        collateral=_collateral(
            legs,
            max_loss=max_loss,
            share_cover=share_cover,
            contract_multiplier=contract_multiplier,
            quantity=quantity,
        ),
        assignment_exposure=assignment_exposure * Decimal(quantity),
        contract_multiplier=contract_multiplier,
        quantity=quantity,
        reason_codes=(),
    )


def _covering_contracts(
    legs: Sequence[OptionLeg],
    share_cover: Decimal,
) -> Decimal:
    """Share-equivalents actually consumed covering naked short calls."""

    short_calls = sum(
        (
            Decimal(leg.ratio)
            for leg in legs
            if leg.option_type is OptionType.CALL and leg.side is OrderSide.SELL
        ),
        _ZERO,
    )
    long_calls = sum(
        (
            Decimal(leg.ratio)
            for leg in legs
            if leg.option_type is OptionType.CALL and leg.side is OrderSide.BUY
        ),
        _ZERO,
    )
    needed = short_calls - long_calls
    if needed <= _ZERO:
        return _ZERO
    return min(share_cover, needed)


def _coverage_reasons(
    legs: Sequence[OptionLeg],
    *,
    share_cover: Decimal,
    available_cash: Decimal,
    contract_multiplier: int,
    quantity: int,
) -> list[RiskReason]:
    reasons: list[RiskReason] = []

    def ratio_of(option_type: OptionType, side: OrderSide) -> Decimal:
        return sum(
            (
                Decimal(leg.ratio)
                for leg in legs
                if leg.option_type is option_type and leg.side is side
            ),
            _ZERO,
        )

    short_calls = ratio_of(OptionType.CALL, OrderSide.SELL)
    long_calls = ratio_of(OptionType.CALL, OrderSide.BUY)
    if short_calls > long_calls + share_cover:
        # A partially covered short is an uncovered ratio; a wholly bare one is
        # a naked short.  Both are refused, but they read differently in audit.
        reasons.append(
            RiskReason.UNCOVERED_RATIO
            if long_calls > _ZERO or share_cover > _ZERO
            else RiskReason.NAKED_SHORT_CALL
        )
        if share_cover > _ZERO and long_calls == _ZERO:
            reasons.append(RiskReason.INSUFFICIENT_SHARE_COVERAGE)

    short_puts = ratio_of(OptionType.PUT, OrderSide.SELL)
    long_puts = ratio_of(OptionType.PUT, OrderSide.BUY)
    uncovered_puts = short_puts - long_puts
    if uncovered_puts > _ZERO:
        required_cash = sum(
            (
                leg.strike * Decimal(leg.ratio) * Decimal(contract_multiplier)
                for leg in legs
                if leg.option_type is OptionType.PUT and leg.side is OrderSide.SELL
            ),
            _ZERO,
        ) * Decimal(quantity)
        if available_cash < required_cash:
            reasons.append(
                RiskReason.UNCOVERED_RATIO
                if long_puts > _ZERO
                else RiskReason.NAKED_SHORT_PUT
            )
            if available_cash > _ZERO:
                reasons.append(RiskReason.INSUFFICIENT_CASH_COLLATERAL)
    return reasons


def _multi_expiry_profile(
    legs: Sequence[OptionLeg],
    *,
    net_price: Decimal,
    reasons: list[RiskReason],
    assignment_exposure: Decimal,
    contract_multiplier: int,
    quantity: int,
    fees: Decimal,
) -> OptionRiskProfile:
    """Bound a debit calendar or diagonal without claiming one expiry payoff."""

    near_expiry = min(leg.expiration for leg in legs)
    far_expiry = max(leg.expiration for leg in legs)
    near_legs = [leg for leg in legs if leg.expiration == near_expiry]
    far_legs = [leg for leg in legs if leg.expiration == far_expiry]

    if any(leg.side is OrderSide.SELL for leg in far_legs):
        # Selling the back month leaves an uncovered short once the front leg
        # expires, so a short calendar is never approved.
        reasons.append(RiskReason.SHORT_CALENDAR_REJECTED)
    if net_price <= _ZERO:
        reasons.append(RiskReason.SHORT_CALENDAR_REJECTED)

    for near in near_legs:
        if near.side is not OrderSide.SELL:
            continue
        covering = [
            leg
            for leg in far_legs
            if leg.side is OrderSide.BUY and leg.option_type is near.option_type
        ]
        if not covering:
            reasons.append(RiskReason.CALENDAR_RIGHT_MISMATCH)
            continue
        if sum((Decimal(leg.ratio) for leg in covering), _ZERO) < Decimal(near.ratio):
            reasons.append(RiskReason.CALENDAR_QUANTITY_MISMATCH)

    if reasons:
        return _rejected(reasons, net_price, contract_multiplier, quantity)

    # Conservative bound: the debit paid can be lost outright, and assignment
    # of the front short against the back long can cost the adverse strike gap.
    gap = _adverse_strike_gap(near_legs, far_legs)
    scale = Decimal(contract_multiplier) * Decimal(quantity)
    max_loss = net_price * scale + gap * scale + fees
    return OptionRiskProfile(
        valid=True,
        # Not None-as-unbounded: a cross-expiry profit is simply not bounded by
        # this model, so it is never presented as a finite maximum.
        max_profit=None,
        max_loss=max_loss,
        breakevens=(),
        net_debit=net_price,
        collateral=max_loss,
        assignment_exposure=assignment_exposure * Decimal(quantity),
        contract_multiplier=contract_multiplier,
        quantity=quantity,
        reason_codes=(),
    )


def _adverse_strike_gap(
    near_legs: Sequence[OptionLeg],
    far_legs: Sequence[OptionLeg],
) -> Decimal:
    """Worst assignment gap between a front short and its back-month cover."""

    worst = _ZERO
    for near in near_legs:
        if near.side is not OrderSide.SELL:
            continue
        for far in far_legs:
            if far.side is not OrderSide.BUY or far.option_type is not near.option_type:
                continue
            if near.option_type is OptionType.CALL:
                gap = far.strike - near.strike
            else:
                gap = near.strike - far.strike
            worst = max(worst, gap)
    return worst


def _collateral(
    legs: Sequence[OptionLeg],
    *,
    max_loss: Decimal,
    share_cover: Decimal,
    contract_multiplier: int,
    quantity: int,
) -> Decimal:
    """Cash that must be reserved to hold the structure."""

    long_puts = sum(
        (
            Decimal(leg.ratio)
            for leg in legs
            if leg.option_type is OptionType.PUT and leg.side is OrderSide.BUY
        ),
        _ZERO,
    )
    cash_secured = _ZERO
    for leg in legs:
        if leg.option_type is not OptionType.PUT or leg.side is not OrderSide.SELL:
            continue
        uncovered = max(Decimal(leg.ratio) - long_puts, _ZERO)
        cash_secured += (
            leg.strike * uncovered * Decimal(contract_multiplier) * Decimal(quantity)
        )
    return max(max_loss, cash_secured)


def _rejected(
    reasons: Sequence[RiskReason],
    net_price: Decimal,
    contract_multiplier: int,
    quantity: int,
) -> OptionRiskProfile:
    ordered = tuple(dict.fromkeys(reason.value for reason in reasons))
    return OptionRiskProfile(
        valid=False,
        max_profit=None,
        max_loss=_ZERO,
        breakevens=(),
        net_debit=net_price,
        collateral=_ZERO,
        assignment_exposure=_ZERO,
        contract_multiplier=contract_multiplier,
        quantity=max(quantity, 1),
        reason_codes=ordered,
    )


__all__ = [
    "SHARE_BACKED_FAMILIES",
    "STRUCTURE_TEMPLATES",
    "LegRole",
    "LegSpec",
    "StructureError",
    "build_structure",
    "validate_structure",
]
