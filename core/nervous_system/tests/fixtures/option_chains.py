"""Deterministic synthetic option chains for selection tests (Task 18).

Prices are hand-set so every payoff and bound is an exact ``Decimal``.  The
chain is intentionally boring: a flat, liquid, well-behaved surface, with
unfit contracts added explicitly by the tests that need them.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from core.nervous_system.contracts.enums import OptionType
from core.nervous_system.execution.options.quotes import OptionQuote


UTC = timezone.utc
UNDERLYING = "AMD"
DECISION_TIME = datetime(2026, 7, 30, 18, 20, tzinfo=UTC)
QUOTE_AT = DECISION_TIME - timedelta(seconds=30)
NEAR_EXPIRY = date(2026, 9, 18)  # 50 DTE
FAR_EXPIRY = date(2026, 10, 16)  # 78 DTE
REFERENCE_PRICE = Decimal("200")

D = Decimal


def occ(strike: Decimal, option_type: OptionType, expiration: date) -> str:
    right = "C" if option_type is OptionType.CALL else "P"
    return f"{UNDERLYING}{expiration:%y%m%d}{right}{int(strike * 1000):08d}"


def quote(
    strike: str,
    option_type: OptionType,
    *,
    expiration: date = NEAR_EXPIRY,
    bid: str,
    ask: str,
    open_interest: int = 5_000,
    volume: int = 900,
    delta: str | None = None,
    quote_at: datetime = QUOTE_AT,
    **overrides,
) -> OptionQuote:
    strike_value = D(strike)
    payload = {
        "symbol": occ(strike_value, option_type, expiration),
        "underlying": UNDERLYING,
        "option_type": option_type,
        "strike": strike_value,
        "expiration": expiration,
        "quote_at": quote_at,
        "bid": D(bid),
        "ask": D(ask),
        "open_interest": open_interest,
        "volume": volume,
        "delta": None if delta is None else D(delta),
    }
    payload.update(overrides)
    return OptionQuote(**payload)


# A flat, liquid surface around a 200 underlying.  Intrinsic-plus-time values
# are chosen so verticals price at round debits.
_CALL_LADDER = (
    ("190", "15.90", "16.10", "0.75"),
    ("195", "12.90", "13.10", "0.65"),
    ("200", "9.90", "10.10", "0.55"),
    ("205", "7.40", "7.60", "0.45"),
    ("210", "5.40", "5.60", "0.35"),
    ("220", "2.90", "3.10", "0.20"),
)
_PUT_LADDER = (
    ("180", "2.90", "3.10", "-0.20"),
    ("190", "5.40", "5.60", "-0.35"),
    ("195", "7.40", "7.60", "-0.45"),
    ("200", "9.90", "10.10", "-0.55"),
    ("205", "12.90", "13.10", "-0.65"),
    ("210", "15.90", "16.10", "-0.75"),
)


def base_chain(expiration: date = NEAR_EXPIRY) -> tuple[OptionQuote, ...]:
    """A fully fit chain of calls and puts at one expiration."""

    calls = tuple(
        quote(strike, OptionType.CALL, expiration=expiration, bid=bid, ask=ask, delta=delta)
        for strike, bid, ask, delta in _CALL_LADDER
    )
    puts = tuple(
        quote(strike, OptionType.PUT, expiration=expiration, bid=bid, ask=ask, delta=delta)
        for strike, bid, ask, delta in _PUT_LADDER
    )
    return calls + puts


def two_expiry_chain() -> tuple[OptionQuote, ...]:
    """Both expirations, so calendars and diagonals are constructible."""

    return base_chain(NEAR_EXPIRY) + base_chain(FAR_EXPIRY)


# --- deliberately unfit contracts -------------------------------------------


def stale_quote() -> OptionQuote:
    """An old two-sided market that still carries a trade print.

    Fitness is decided by the mark, never by trade evidence.  A print newer
    than the observation cannot even be represented -- ``OptionQuote`` rejects
    a record that mixes two observation times -- so trade recency can never
    rescue a stale quote by any route.
    """

    stale_at = DECISION_TIME - timedelta(hours=6)
    return quote(
        "200",
        OptionType.CALL,
        bid="9.90",
        ask="10.10",
        quote_at=stale_at,
        last_trade_price=D("10.00"),
        last_trade_at=stale_at,
    )


def future_quote() -> OptionQuote:
    """Timestamped after the decision, so it was not knowable in time."""

    return quote(
        "205",
        OptionType.CALL,
        bid="7.40",
        ask="7.60",
        quote_at=DECISION_TIME + timedelta(seconds=1),
    )


def wide_spread_quote() -> OptionQuote:
    return quote("215", OptionType.CALL, bid="1.00", ask="4.00")


def zero_bid_quote() -> OptionQuote:
    return quote("230", OptionType.CALL, bid="0.00", ask="0.20")


def illiquid_quote() -> OptionQuote:
    return quote("225", OptionType.CALL, bid="1.90", ask="2.00", open_interest=3, volume=0)


def short_dated_quote() -> OptionQuote:
    return quote(
        "200",
        OptionType.CALL,
        expiration=DECISION_TIME.date() + timedelta(days=3),
        bid="4.90",
        ask="5.10",
    )


def foreign_underlying_quote() -> OptionQuote:
    strike = D("200")
    return OptionQuote(
        symbol=f"NVDA{NEAR_EXPIRY:%y%m%d}C{int(strike * 1000):08d}",
        underlying="NVDA",
        option_type=OptionType.CALL,
        strike=strike,
        expiration=NEAR_EXPIRY,
        quote_at=QUOTE_AT,
        bid=D("9.90"),
        ask=D("10.10"),
        open_interest=5_000,
        volume=900,
    )


def mismatched_multiplier_quote() -> OptionQuote:
    return quote("200", OptionType.PUT, bid="9.90", ask="10.10", contract_multiplier=10)


__all__ = [
    "DECISION_TIME",
    "FAR_EXPIRY",
    "NEAR_EXPIRY",
    "QUOTE_AT",
    "REFERENCE_PRICE",
    "UNDERLYING",
    "base_chain",
    "foreign_underlying_quote",
    "future_quote",
    "illiquid_quote",
    "mismatched_multiplier_quote",
    "occ",
    "quote",
    "short_dated_quote",
    "stale_quote",
    "two_expiry_chain",
    "wide_spread_quote",
    "zero_bid_quote",
]
