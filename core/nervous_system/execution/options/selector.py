"""Deterministic instrument selection.

Given the same intent, policy, snapshot, chain, portfolio, and config, this
returns the same selection and the same content hash regardless of the order
the chain arrives in.

Nothing here is a probability.  Score components are transparent normalised
measures of the chain, and every rejected candidate is recorded with its
reason codes rather than silently dropped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import ROUND_DOWN, Decimal
from enum import Enum
from itertools import combinations
from uuid import UUID, uuid5

from pydantic import model_validator

from core.nervous_system.config.options import SCORE_COMPONENTS, OptionSelectionConfig
from core.nervous_system.contracts.base import (
    ContractModel,
    FiniteDecimal,
    NonNegativeDecimal,
    Sha256Hex,
    UtcDatetime,
    content_hash,
)
from core.nervous_system.contracts.context import ContextSnapshot
from core.nervous_system.contracts.enums import (
    Direction,
    InstrumentFamily,
    OptionType,
    OrderSide,
    PolicyAction,
)
from core.nervous_system.contracts.intent import TradeIntent
from core.nervous_system.contracts.orders import OptionLeg
from core.nervous_system.contracts.policy import PolicyDecision
from core.nervous_system.contracts.portfolio import ImmutableDecimalMap
from core.nervous_system.contracts.states import PortfolioState

from .payoff import OptionRiskProfile
from .quotes import OptionQuote
from .structures import LegRole, StructureError, build_structure, validate_structure


_ZERO = Decimal("0")
_ONE = Decimal("1")
_SELECTION_NAMESPACE = UUID("2f5f1f0d-6d8a-5b3e-9a2c-1c7b9d4e6f10")
_SELECTION_HASH_EXCLUDE = frozenset({"selection_id", "content_hash"})


class SelectionOutcome(str, Enum):
    SELECTED_OPTION = "SELECTED_OPTION"
    SELECTED_EQUITY_FALLBACK = "SELECTED_EQUITY_FALLBACK"
    NO_ELIGIBLE_INSTRUMENT = "NO_ELIGIBLE_INSTRUMENT"


class FitnessReason(str, Enum):
    """Why a quote or candidate was not usable."""

    QUOTE_NOT_AVAILABLE_AT_DECISION = "QUOTE_NOT_AVAILABLE_AT_DECISION"
    QUOTE_STALE = "QUOTE_STALE"
    QUOTE_SPREAD_TOO_WIDE = "QUOTE_SPREAD_TOO_WIDE"
    QUOTE_ZERO_BID = "QUOTE_ZERO_BID"
    QUOTE_OPEN_INTEREST_TOO_LOW = "QUOTE_OPEN_INTEREST_TOO_LOW"
    QUOTE_VOLUME_TOO_LOW = "QUOTE_VOLUME_TOO_LOW"
    QUOTE_DTE_OUT_OF_RANGE = "QUOTE_DTE_OUT_OF_RANGE"
    QUOTE_WRONG_UNDERLYING = "QUOTE_WRONG_UNDERLYING"
    QUOTE_MULTIPLIER_MISMATCH = "QUOTE_MULTIPLIER_MISMATCH"
    NO_ELIGIBLE_QUOTES = "NO_ELIGIBLE_QUOTES"
    STRUCTURE_NOT_PERMITTED = "STRUCTURE_NOT_PERMITTED"
    STRUCTURE_SHAPE_INVALID = "STRUCTURE_SHAPE_INVALID"
    RISK_BUDGET_EXCEEDED = "RISK_BUDGET_EXCEEDED"
    QUANTITY_BELOW_ONE = "QUANTITY_BELOW_ONE"
    EQUITY_FALLBACK_NOT_PERMITTED = "EQUITY_FALLBACK_NOT_PERMITTED"
    POLICY_NOT_EXECUTABLE = "POLICY_NOT_EXECUTABLE"


class CandidateScore(ContractModel):
    """Transparent normalised selection measures. Not a probability."""

    components: ImmutableDecimalMap
    total: FiniteDecimal

    @model_validator(mode="after")
    def validate_components(self) -> CandidateScore:
        missing = [name for name in SCORE_COMPONENTS if name not in self.components]
        if missing:
            raise ValueError(f"score is missing components: {missing}")
        for name, value in self.components.items():
            if not _ZERO <= value <= _ONE:
                raise ValueError(f"score component {name!r} must lie in [0, 1]")
        return self


class RejectedInstrument(ContractModel):
    """One candidate that was considered and refused, with its evidence."""

    structure: InstrumentFamily
    leg_symbols: tuple[str, ...]
    reason_codes: tuple[str, ...]
    max_loss: FiniteDecimal | None = None


class InstrumentSelection(ContractModel):
    selection_id: UUID
    intent_id: UUID
    policy_decision_id: UUID
    snapshot_id: UUID
    outcome: SelectionOutcome
    structure: InstrumentFamily | None = None
    legs: tuple[OptionLeg, ...] = ()
    equity_symbol: str | None = None
    equity_side: OrderSide | None = None
    quantity: int = 0
    quote_snapshot_at: UtcDatetime | None = None
    estimated_net_price: FiniteDecimal | None = None
    max_loss: NonNegativeDecimal = _ZERO
    collateral: NonNegativeDecimal = _ZERO
    score: CandidateScore | None = None
    rejected: tuple[RejectedInstrument, ...] = ()
    config_hash: Sha256Hex
    content_hash: Sha256Hex

    def computed_content_hash(self) -> str:
        return content_hash(self, exclude=set(_SELECTION_HASH_EXCLUDE))

    @classmethod
    def create(cls, *, selection_id: UUID, **fields: object) -> InstrumentSelection:
        probe = cls.model_construct(
            selection_id=selection_id, content_hash="0" * 64, **fields
        )
        return cls(
            selection_id=selection_id,
            content_hash=content_hash(probe, exclude=set(_SELECTION_HASH_EXCLUDE)),
            **fields,
        )

    @model_validator(mode="after")
    def validate_selection(self) -> InstrumentSelection:
        if self.outcome is SelectionOutcome.SELECTED_OPTION:
            if not self.legs:
                raise ValueError("an option selection requires legs")
            if self.structure is None:
                raise ValueError("an option selection requires a structure")
            if self.quantity < 1:
                raise ValueError("an option selection requires a positive quantity")
            if self.quote_snapshot_at is None:
                raise ValueError("an option selection requires its quote snapshot time")
        elif self.outcome is SelectionOutcome.SELECTED_EQUITY_FALLBACK:
            if self.legs:
                raise ValueError("an equity fallback carries no option legs")
            if self.equity_symbol is None or self.equity_side is None:
                raise ValueError("an equity fallback requires symbol and side")
            if self.quantity < 1:
                raise ValueError("an equity fallback requires a positive quantity")
        else:
            if self.legs or self.equity_symbol is not None:
                raise ValueError("a rejected selection carries no instrument")
            if self.quantity:
                raise ValueError("a rejected selection has no quantity")
        if self.content_hash != self.computed_content_hash():
            raise ValueError("content_hash does not match selection content")
        return self


def select_instrument(
    intent: TradeIntent,
    policy: PolicyDecision,
    snapshot: ContextSnapshot,
    chain: Sequence[OptionQuote],
    portfolio: PortfolioState,
    *,
    config: OptionSelectionConfig,
) -> InstrumentSelection:
    """Choose the best permitted instrument, or explicitly choose nothing."""

    rejected: list[RejectedInstrument] = []
    selection_id = uuid5(
        _SELECTION_NAMESPACE,
        "|".join(
            (
                str(intent.intent_id),
                str(policy.policy_decision_id),
                snapshot.content_hash,
                config.content_hash,
            )
        ),
    )

    def finish(**fields: object) -> InstrumentSelection:
        return InstrumentSelection.create(
            selection_id=selection_id,
            intent_id=intent.intent_id,
            policy_decision_id=policy.policy_decision_id,
            snapshot_id=snapshot.snapshot_id,
            config_hash=config.content_hash,
            rejected=tuple(rejected),
            **fields,
        )

    if policy.action not in {PolicyAction.APPROVE, PolicyAction.APPROVE_REDUCED}:
        rejected.append(
            RejectedInstrument(
                structure=InstrumentFamily.EQUITY,
                leg_symbols=(),
                reason_codes=(FitnessReason.POLICY_NOT_EXECUTABLE.value,),
            )
        )
        return finish(outcome=SelectionOutcome.NO_ELIGIBLE_INSTRUMENT)

    budget = policy.final_risk_budget
    preferences = tuple(intent.instrument_preferences) or config.default_preferences
    permitted = [
        family for family in preferences if family in policy.allowed_instruments
    ]
    for family in preferences:
        if family not in policy.allowed_instruments:
            rejected.append(
                RejectedInstrument(
                    structure=family,
                    leg_symbols=(),
                    reason_codes=(FitnessReason.STRUCTURE_NOT_PERMITTED.value,),
                )
            )

    eligible, quote_rejections = _eligible_quotes(
        chain, intent=intent, snapshot=snapshot, config=config
    )
    rejected.extend(quote_rejections)

    # The preference list is ordered, not a set: the first permitted family
    # that yields a valid candidate wins.  Searching every family and taking
    # the globally best score would silently escalate past a strategy's
    # conservative first choice, which for Meta is plain equity.
    for family in permitted:
        if family is InstrumentFamily.EQUITY:
            equity = _equity_fallback(
                intent=intent,
                snapshot=snapshot,
                permitted=permitted,
                budget=budget,
                config=config,
            )
            if isinstance(equity, RejectedInstrument):
                rejected.append(equity)
                continue
            if equity is None:
                continue
            symbol, side, quantity = equity
            return finish(
                outcome=SelectionOutcome.SELECTED_EQUITY_FALLBACK,
                equity_symbol=symbol,
                equity_side=side,
                quantity=quantity,
                max_loss=budget,
                collateral=budget,
            )

        candidates, family_rejections = _candidates_for(
            family,
            eligible=eligible,
            intent=intent,
            portfolio=portfolio,
            budget=budget,
            config=config,
        )
        rejected.extend(family_rejections)
        best: _Candidate | None = None
        for candidate in candidates:
            if best is None or _sort_key(candidate) < _sort_key(best):
                best = candidate
        if best is not None:
            return finish(
                outcome=SelectionOutcome.SELECTED_OPTION,
                structure=best.structure,
                legs=best.legs,
                quantity=best.quantity,
                quote_snapshot_at=best.quote_snapshot_at,
                estimated_net_price=best.net_price,
                max_loss=best.profile.max_loss,
                collateral=best.profile.collateral,
                score=best.score,
            )

    return finish(outcome=SelectionOutcome.NO_ELIGIBLE_INSTRUMENT)


class _Candidate:
    __slots__ = (
        "structure",
        "legs",
        "quantity",
        "net_price",
        "profile",
        "score",
        "spread_fraction",
        "open_interest",
        "volume",
        "quote_snapshot_at",
    )

    def __init__(self, **kwargs: object) -> None:
        for name in self.__slots__:
            setattr(self, name, kwargs[name])


def _sort_key(candidate: _Candidate) -> tuple:
    """The plan's fixed ordering, with symbols as the final deterministic tie-break."""

    return (
        -candidate.score.total,
        candidate.spread_fraction,
        -candidate.open_interest,
        -candidate.volume,
        tuple(leg.symbol for leg in candidate.legs),
    )


def _eligible_quotes(
    chain: Sequence[OptionQuote],
    *,
    intent: TradeIntent,
    snapshot: ContextSnapshot,
    config: OptionSelectionConfig,
) -> tuple[tuple[OptionQuote, ...], list[RejectedInstrument]]:
    """Filter the chain to tradable, causally available, liquid quotes."""

    decision_time = snapshot.decision_time
    decision_date = decision_time.date()
    eligible: list[OptionQuote] = []
    rejections: list[RejectedInstrument] = []

    for quote in sorted(chain, key=lambda item: item.symbol):
        reasons: list[str] = []
        if quote.underlying != intent.ticker:
            reasons.append(FitnessReason.QUOTE_WRONG_UNDERLYING.value)
        if quote.quote_at > decision_time:
            reasons.append(FitnessReason.QUOTE_NOT_AVAILABLE_AT_DECISION.value)
        elif quote.age_seconds(decision_time) > config.max_quote_age_seconds:
            # A recent trade print never rescues a stale two-sided market.
            reasons.append(FitnessReason.QUOTE_STALE.value)
        if quote.bid <= _ZERO:
            reasons.append(FitnessReason.QUOTE_ZERO_BID.value)
        elif (
            quote.spread > config.max_spread_absolute
            or quote.spread_fraction > config.max_spread_fraction
        ):
            reasons.append(FitnessReason.QUOTE_SPREAD_TOO_WIDE.value)
        if (
            config.min_open_interest
            and (quote.open_interest or 0) < config.min_open_interest
        ):
            reasons.append(FitnessReason.QUOTE_OPEN_INTEREST_TOO_LOW.value)
        if config.min_volume and (quote.volume or 0) < config.min_volume:
            reasons.append(FitnessReason.QUOTE_VOLUME_TOO_LOW.value)
        dte = (quote.expiration - decision_date).days
        if not config.min_dte <= dte <= config.max_dte:
            reasons.append(FitnessReason.QUOTE_DTE_OUT_OF_RANGE.value)

        if reasons:
            rejections.append(
                RejectedInstrument(
                    structure=InstrumentFamily.SINGLE_OPTION,
                    leg_symbols=(quote.symbol,),
                    reason_codes=tuple(reasons),
                )
            )
        else:
            eligible.append(quote)

    multipliers = {quote.contract_multiplier for quote in eligible}
    if len(multipliers) > 1:
        # Mixed multipliers cannot form one order, so keep the dominant set and
        # record the rest rather than silently building an invalid structure.
        # Dominant means most quotes, not smallest value: an odd-lot adjusted
        # contract must not evict the whole standard chain.
        counts: dict[int, int] = {}
        for quote in eligible:
            counts[quote.contract_multiplier] = counts.get(quote.contract_multiplier, 0) + 1
        dominant = max(sorted(counts), key=lambda value: counts[value])
        kept: list[OptionQuote] = []
        for quote in eligible:
            if quote.contract_multiplier == dominant:
                kept.append(quote)
            else:
                rejections.append(
                    RejectedInstrument(
                        structure=InstrumentFamily.SINGLE_OPTION,
                        leg_symbols=(quote.symbol,),
                        reason_codes=(FitnessReason.QUOTE_MULTIPLIER_MISMATCH.value,),
                    )
                )
        eligible = kept

    return tuple(eligible), rejections


def _required_right(intent: TradeIntent) -> OptionType:
    return OptionType.CALL if intent.direction is Direction.LONG else OptionType.PUT


def _role_sets(
    family: InstrumentFamily,
    eligible: Sequence[OptionQuote],
    intent: TradeIntent,
    config: OptionSelectionConfig,
) -> list[dict[LegRole, OptionQuote]]:
    """Enumerate deterministic role assignments for one family."""

    right = _required_right(intent)
    calls = [quote for quote in eligible if quote.option_type is OptionType.CALL]
    puts = [quote for quote in eligible if quote.option_type is OptionType.PUT]
    same_right = calls if right is OptionType.CALL else puts
    limit = config.max_candidates_per_structure
    sets: list[dict[LegRole, OptionQuote]] = []

    def by_strike(quotes: Sequence[OptionQuote]) -> list[OptionQuote]:
        return sorted(quotes, key=lambda item: (item.expiration, item.strike, item.symbol))

    if family is InstrumentFamily.SINGLE_OPTION:
        sets = [{LegRole.LONG_LEG: quote} for quote in by_strike(same_right)]
    elif family is InstrumentFamily.VERTICAL:
        for lower, upper in combinations(by_strike(same_right), 2):
            if lower.expiration != upper.expiration or lower.strike >= upper.strike:
                continue
            # A debit vertical buys the more valuable leg: the lower call, or
            # the higher put.
            if right is OptionType.CALL:
                sets.append({LegRole.LONG_LEG: lower, LegRole.SHORT_LEG: upper})
            else:
                sets.append({LegRole.LONG_LEG: upper, LegRole.SHORT_LEG: lower})
    elif family in {InstrumentFamily.STRADDLE, InstrumentFamily.STRANGLE}:
        for put_quote in by_strike(puts):
            for call_quote in by_strike(calls):
                if put_quote.expiration != call_quote.expiration:
                    continue
                same = put_quote.strike == call_quote.strike
                if family is InstrumentFamily.STRADDLE and not same:
                    continue
                if family is InstrumentFamily.STRANGLE and put_quote.strike >= call_quote.strike:
                    continue
                sets.append(
                    {LegRole.LONG_PUT: put_quote, LegRole.LONG_CALL: call_quote}
                )
    elif family is InstrumentFamily.BUTTERFLY:
        for lower, body, upper in combinations(by_strike(same_right), 3):
            if len({lower.expiration, body.expiration, upper.expiration}) != 1:
                continue
            if not lower.strike < body.strike < upper.strike:
                continue
            sets.append(
                {
                    LegRole.LOWER_WING: lower,
                    LegRole.BODY: body,
                    LegRole.UPPER_WING: upper,
                }
            )
    elif family is InstrumentFamily.CONDOR:
        for quad in combinations(by_strike(same_right), 4):
            if len({quote.expiration for quote in quad}) != 1:
                continue
            if not quad[0].strike < quad[1].strike < quad[2].strike < quad[3].strike:
                continue
            sets.append(
                {
                    LegRole.LOWER_WING: quad[0],
                    LegRole.LOWER_BODY: quad[1],
                    LegRole.UPPER_BODY: quad[2],
                    LegRole.UPPER_WING: quad[3],
                }
            )
    elif family in {InstrumentFamily.IRON_BUTTERFLY, InstrumentFamily.IRON_CONDOR}:
        for put_pair in combinations(by_strike(puts), 2):
            for call_pair in combinations(by_strike(calls), 2):
                quotes = (*put_pair, *call_pair)
                if len({quote.expiration for quote in quotes}) != 1:
                    continue
                if not put_pair[0].strike < put_pair[1].strike:
                    continue
                if not call_pair[0].strike < call_pair[1].strike:
                    continue
                if family is InstrumentFamily.IRON_BUTTERFLY:
                    if put_pair[1].strike != call_pair[0].strike:
                        continue
                elif put_pair[1].strike >= call_pair[0].strike:
                    continue
                sets.append(
                    {
                        LegRole.LOWER_WING: put_pair[0],
                        LegRole.LOWER_BODY: put_pair[1],
                        LegRole.UPPER_BODY: call_pair[0],
                        LegRole.UPPER_WING: call_pair[1],
                    }
                )
    elif family in {InstrumentFamily.CALENDAR, InstrumentFamily.DIAGONAL}:
        for near, far in combinations(by_strike(same_right), 2):
            if near.expiration >= far.expiration:
                continue
            same_strike = near.strike == far.strike
            if family is InstrumentFamily.CALENDAR and not same_strike:
                continue
            if family is InstrumentFamily.DIAGONAL and same_strike:
                continue
            sets.append({LegRole.SHORT_LEG: near, LegRole.LONG_LEG: far})
    elif family is InstrumentFamily.COVERED_CALL:
        sets = [{LegRole.SHORT_CALL: quote} for quote in by_strike(calls)]
    elif family is InstrumentFamily.CASH_SECURED_PUT:
        sets = [{LegRole.SHORT_PUT: quote} for quote in by_strike(puts)]
    elif family is InstrumentFamily.PROTECTIVE_PUT:
        sets = [{LegRole.LONG_PUT: quote} for quote in by_strike(puts)]
    elif family is InstrumentFamily.COLLAR:
        for put_quote in by_strike(puts):
            for call_quote in by_strike(calls):
                if put_quote.expiration != call_quote.expiration:
                    continue
                if put_quote.strike >= call_quote.strike:
                    continue
                sets.append(
                    {LegRole.LONG_PUT: put_quote, LegRole.SHORT_CALL: call_quote}
                )

    return sets[:limit]


def _net_price(legs: Sequence[OptionLeg]) -> Decimal:
    """Conservative net debit: pay the ask, receive the bid."""

    total = _ZERO
    for leg in legs:
        price = leg.ask if leg.side is OrderSide.BUY else leg.bid
        sign = _ONE if leg.side is OrderSide.BUY else -_ONE
        total += sign * price * Decimal(leg.ratio)
    return total


def _candidates_for(
    family: InstrumentFamily,
    *,
    eligible: Sequence[OptionQuote],
    intent: TradeIntent,
    portfolio: PortfolioState,
    budget: Decimal,
    config: OptionSelectionConfig,
) -> tuple[list[_Candidate], list[RejectedInstrument]]:
    candidates: list[_Candidate] = []
    rejections: list[RejectedInstrument] = []

    role_sets = _role_sets(family, eligible, intent, config)
    if not role_sets:
        rejections.append(
            RejectedInstrument(
                structure=family,
                leg_symbols=(),
                reason_codes=(FitnessReason.NO_ELIGIBLE_QUOTES.value,),
            )
        )
        return candidates, rejections

    for contracts in role_sets:
        symbols = tuple(sorted(quote.symbol for quote in contracts.values()))
        try:
            legs = build_structure(
                family, selected_contracts=contracts, quantity=1
            )
        except StructureError:
            rejections.append(
                RejectedInstrument(
                    structure=family,
                    leg_symbols=symbols,
                    reason_codes=(FitnessReason.STRUCTURE_SHAPE_INVALID.value,),
                )
            )
            continue

        net_price = _net_price(legs)
        multiplier = next(iter(contracts.values())).contract_multiplier
        profile = validate_structure(
            legs,
            net_price=net_price,
            existing_holdings=portfolio.positions,
            available_cash=Decimal(str(portfolio.cash)),
            contract_multiplier=multiplier,
            quantity=1,
        )
        if not profile.valid:
            rejections.append(
                RejectedInstrument(
                    structure=family,
                    leg_symbols=symbols,
                    reason_codes=profile.reason_codes,
                )
            )
            continue

        per_structure = profile.max_loss
        if per_structure <= _ZERO:
            quantity = 1
        else:
            quantity = int((budget / per_structure).to_integral_value(ROUND_DOWN))
        if quantity < 1:
            rejections.append(
                RejectedInstrument(
                    structure=family,
                    leg_symbols=symbols,
                    reason_codes=(FitnessReason.RISK_BUDGET_EXCEEDED.value,),
                    max_loss=per_structure,
                )
            )
            continue

        sized = validate_structure(
            legs,
            net_price=net_price,
            existing_holdings=portfolio.positions,
            available_cash=Decimal(str(portfolio.cash)),
            contract_multiplier=multiplier,
            quantity=quantity,
        )
        if not sized.valid:
            rejections.append(
                RejectedInstrument(
                    structure=family,
                    leg_symbols=symbols,
                    reason_codes=sized.reason_codes,
                )
            )
            continue

        candidates.append(
            _Candidate(
                structure=family,
                legs=legs,
                quantity=quantity,
                net_price=net_price,
                profile=sized,
                score=_score(contracts, sized, budget, intent, config),
                spread_fraction=max(
                    quote.spread_fraction for quote in contracts.values()
                ),
                open_interest=min(
                    (quote.open_interest or 0) for quote in contracts.values()
                ),
                volume=min((quote.volume or 0) for quote in contracts.values()),
                quote_snapshot_at=min(
                    quote.quote_at for quote in contracts.values()
                ),
            )
        )

    return candidates, rejections


def _score(
    contracts: Mapping[LegRole, OptionQuote],
    profile: OptionRiskProfile,
    budget: Decimal,
    intent: TradeIntent,
    config: OptionSelectionConfig,
) -> CandidateScore:
    quotes = tuple(contracts.values())

    worst_spread = max(quote.spread_fraction for quote in quotes)
    spread = _clamp(_ONE - worst_spread / config.max_spread_fraction)

    open_interest = min((quote.open_interest or 0) for quote in quotes)
    reference = max(config.min_open_interest * 10, 1)
    liquidity = _clamp(Decimal(open_interest) / Decimal(reference))

    midpoint = Decimal(config.min_dte + config.max_dte) / Decimal("2")
    span = max(Decimal(config.max_dte - config.min_dte), _ONE)
    # Nearest the middle of the permitted window scores highest.
    worst_dte = max(
        abs(Decimal((quote.expiration - intent.created_at.date()).days) - midpoint)
        for quote in quotes
    )
    dte = _clamp(_ONE - worst_dte / (span / Decimal("2")))

    deltas = [quote.delta for quote in quotes if quote.delta is not None]
    if deltas:
        distance = min(abs(abs(value) - config.target_delta) for value in deltas)
        delta = _clamp(_ONE - distance / config.target_delta)
    else:
        # Unknown delta scores zero rather than being assumed on target.
        delta = _ZERO

    if budget > _ZERO and profile.max_loss > _ZERO:
        used = profile.max_loss / budget
        budget_fit = _clamp(used if used <= _ONE else _ZERO)
    else:
        budget_fit = _ZERO

    components = {
        "spread": spread,
        "liquidity": liquidity,
        "dte": dte,
        "delta": delta,
        "budget": budget_fit,
    }
    total = sum(
        (config.score_weights[name] * value for name, value in components.items()),
        _ZERO,
    )
    return CandidateScore(components=components, total=total)


def _clamp(value: Decimal) -> Decimal:
    if value < _ZERO:
        return _ZERO
    return _ONE if value > _ONE else value


def _equity_fallback(
    *,
    intent: TradeIntent,
    snapshot: ContextSnapshot,
    permitted: Sequence[InstrumentFamily],
    budget: Decimal,
    config: OptionSelectionConfig,
) -> tuple[str, OrderSide, int] | RejectedInstrument | None:
    if InstrumentFamily.EQUITY not in permitted:
        return None
    if not config.equity_fallback_allowed:
        return RejectedInstrument(
            structure=InstrumentFamily.EQUITY,
            leg_symbols=(),
            reason_codes=(FitnessReason.EQUITY_FALLBACK_NOT_PERMITTED.value,),
        )
    ticker_state = snapshot.ticker_state
    if ticker_state is None:
        return RejectedInstrument(
            structure=InstrumentFamily.EQUITY,
            leg_symbols=(),
            reason_codes=(FitnessReason.NO_ELIGIBLE_QUOTES.value,),
        )
    price = Decimal(str(ticker_state.reference_price))
    if price <= _ZERO:
        return RejectedInstrument(
            structure=InstrumentFamily.EQUITY,
            leg_symbols=(),
            reason_codes=(FitnessReason.NO_ELIGIBLE_QUOTES.value,),
        )
    shares = int((budget / price).to_integral_value(ROUND_DOWN))
    if shares < 1:
        return RejectedInstrument(
            structure=InstrumentFamily.EQUITY,
            leg_symbols=(),
            reason_codes=(FitnessReason.QUANTITY_BELOW_ONE.value,),
        )
    side = OrderSide.BUY if intent.direction is Direction.LONG else OrderSide.SELL
    return intent.ticker, side, shares


__all__ = [
    "CandidateScore",
    "FitnessReason",
    "InstrumentSelection",
    "RejectedInstrument",
    "SelectionOutcome",
    "select_instrument",
]
