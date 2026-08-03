"""Option quote contracts and OCC identity.

Bid/ask marks and trade prints are deliberately separate fields.  A trade
print is whatever last happened to trade, which on an illiquid contract can be
days stale; it is never a mark.  Conflating the two produced a fully retracted
options study in 2026-07, so nothing here derives a price from a trade print.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
import re
from typing import Annotated

from pydantic import Field, model_validator

from core.nervous_system.contracts.base import (
    ContractModel,
    FiniteDecimal,
    NonNegativeDecimal,
    PositiveDecimal,
    UtcDatetime,
)
from core.nervous_system.contracts.enums import OptionType


class QuoteError(ValueError):
    """Raised when a quote cannot be trusted as a tradable mark."""


# OCC: root, YYMMDD, C|P, then an 8-digit strike in thousandths.
_OCC_TAIL = re.compile(r"(\d{6})([CP])(\d{8})")
_OCC_SYMBOL = re.compile(r"^(?P<root>[A-Z][A-Z0-9.]{0,5})(?P<tail>\d{6}[CP]\d{8})$")
_STRIKE_SCALE = Decimal("1000")

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class OccIdentity(ContractModel):
    """The contract identity encoded in an OCC symbol."""

    root: str
    expiration: date
    option_type: OptionType
    strike: PositiveDecimal


def parse_occ_symbol(symbol: str) -> OccIdentity:
    """Parse an OCC symbol, failing loudly on anything malformed."""

    if not isinstance(symbol, str):
        raise QuoteError("OCC symbol must be a string")
    match = _OCC_SYMBOL.match(symbol)
    if match is None:
        raise QuoteError(f"{symbol!r} is not a valid OCC symbol")
    tail = _OCC_TAIL.fullmatch(match.group("tail"))
    if tail is None:  # pragma: no cover - guarded by _OCC_SYMBOL
        raise QuoteError(f"{symbol!r} has an invalid OCC tail")
    yymmdd, right, strike_digits = tail.groups()
    try:
        expiration = date(
            2000 + int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
        )
    except ValueError as exc:
        raise QuoteError(f"{symbol!r} encodes an invalid expiration") from exc
    strike = Decimal(strike_digits) / _STRIKE_SCALE
    if strike <= 0:
        raise QuoteError(f"{symbol!r} encodes a non-positive strike")
    return OccIdentity(
        root=match.group("root"),
        expiration=expiration,
        option_type=OptionType.CALL if right == "C" else OptionType.PUT,
        strike=strike,
    )


class OptionQuote(ContractModel):
    """One tradable two-sided market for one option contract."""

    symbol: str
    underlying: str
    option_type: OptionType
    strike: PositiveDecimal
    expiration: date
    quote_at: UtcDatetime
    bid: NonNegativeDecimal
    ask: NonNegativeDecimal
    contract_multiplier: PositiveInt = 100
    bid_size: NonNegativeInt | None = None
    ask_size: NonNegativeInt | None = None
    open_interest: NonNegativeInt | None = None
    volume: NonNegativeInt | None = None
    # Supplied by the quote source when available.  Never imputed: selection
    # scores an absent delta as unknown rather than assuming one.
    delta: FiniteDecimal | None = None
    # Trade-print evidence, kept strictly apart from the bid/ask mark.
    last_trade_price: NonNegativeDecimal | None = None
    last_trade_at: UtcDatetime | None = None

    @model_validator(mode="after")
    def validate_quote(self) -> OptionQuote:
        if self.ask < self.bid:
            raise ValueError("crossed market: ask is below bid")
        if self.ask <= 0:
            raise ValueError("a non-positive ask is not a tradable market")
        identity = parse_occ_symbol(self.symbol)
        if identity.option_type is not self.option_type:
            raise ValueError("option_type conflicts with the OCC symbol")
        if identity.strike != self.strike:
            raise ValueError("strike conflicts with the OCC symbol")
        if identity.expiration != self.expiration:
            raise ValueError("expiration conflicts with the OCC symbol")
        if self.expiration < self.quote_at.date():
            raise ValueError("expiration must not precede quote_at")
        if (self.last_trade_price is None) != (self.last_trade_at is None):
            raise ValueError("trade price and trade time must be supplied together")
        if self.last_trade_at is not None and self.last_trade_at > self.quote_at:
            raise ValueError("last_trade_at must not be after quote_at")
        return self

    @property
    def mid(self) -> Decimal:
        """The mark: midpoint of the two-sided market, never a trade print."""

        return (self.bid + self.ask) / Decimal("2")

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid

    @property
    def spread_fraction(self) -> Decimal:
        """Spread as a fraction of the mid, used by liquidity filters."""

        mid = self.mid
        if mid <= 0:
            raise QuoteError("cannot express a spread fraction against a zero mid")
        return self.spread / mid

    def age_seconds(self, as_of: UtcDatetime) -> Decimal:
        """Quote age at an explicit evaluation time; never reads a clock."""

        return Decimal(str((as_of - self.quote_at).total_seconds()))

    def require_fresh(self, as_of: UtcDatetime, max_age_seconds: Decimal) -> None:
        age = self.age_seconds(as_of)
        if age < 0:
            raise QuoteError(f"quote for {self.symbol} is timestamped in the future")
        if age > max_age_seconds:
            raise QuoteError(
                f"quote for {self.symbol} is {age}s old, over the {max_age_seconds}s limit"
            )


__all__ = [
    "OccIdentity",
    "OptionQuote",
    "QuoteError",
    "parse_occ_symbol",
]
