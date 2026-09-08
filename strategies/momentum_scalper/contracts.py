"""Immutable strategy-domain records used by replay, shadow, and paper modes."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Mapping


class SetupKind(str, Enum):
    HOD_BREAKOUT = "HOD_BREAKOUT"
    FIRST_PULLBACK = "FIRST_PULLBACK"
    BULL_FLAG = "BULL_FLAG"
    FLAT_TOP = "FLAT_TOP"


class SetupState(str, Enum):
    DISCOVERED = "DISCOVERED"
    IMPULSE = "IMPULSE"
    PULLBACK = "PULLBACK"
    ARMED = "ARMED"
    TRIGGERED = "TRIGGERED"
    INVALIDATED = "INVALIDATED"
    HALTED = "HALTED"
    DATA_STALE = "DATA_STALE"
    EXPIRED = "EXPIRED"


class PositionState(str, Enum):
    ORDER_PENDING = "ORDER_PENDING"
    OPEN = "OPEN"
    EXIT_PENDING = "EXIT_PENDING"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class QuoteSnapshot:
    ticker: str
    event_at: datetime
    received_at: datetime
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    feed: str = "UNKNOWN"

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_pct(self) -> float:
        return ((self.ask - self.bid) / self.mid * 100.0) if self.mid > 0 else float("inf")


@dataclass(frozen=True)
class PatternSignal:
    ticker: str
    timestamp: datetime
    kind: SetupKind
    trigger_price: float
    invalidation_price: float
    reason_codes: tuple[str, ...]
    source_levels: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SetupTransition:
    ticker: str
    timestamp: datetime
    kind: SetupKind
    previous: SetupState
    current: SetupState
    reason: str


@dataclass(frozen=True)
class OrderIntent:
    ticker: str
    created_at: datetime
    expires_at: datetime
    quantity: int
    limit_price: float
    stop_price: float
    pattern: SetupKind
    reason_codes: tuple[str, ...]
    extended_hours: bool
    paper_only: bool = True

    def __post_init__(self) -> None:
        if not self.paper_only:
            raise ValueError("momentum scalper order intents must remain paper-only")
        if self.quantity <= 0 or self.limit_price <= 0 or self.stop_price <= 0:
            raise ValueError("order intent requires positive quantity, limit, and stop")
        if self.expires_at <= self.created_at:
            raise ValueError("order intent expiry must follow creation")


@dataclass(frozen=True)
class FillEvent:
    ticker: str
    timestamp: datetime
    quantity: int
    price: float
    order_id: str
    is_exit: bool = False


__all__ = [
    "FillEvent", "OrderIntent", "PatternSignal", "PositionState", "QuoteSnapshot",
    "SetupKind", "SetupState", "SetupTransition",
]
