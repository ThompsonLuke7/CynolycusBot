"""Fill-aware position and partial-exit state for the momentum scalper."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from strategies.momentum_scalper.configs.v1 import RiskConfigV1


@dataclass(frozen=True)
class ExitInstruction:
    ticker: str
    timestamp: datetime
    quantity: int
    limit_price: float
    reason: str


@dataclass
class MomentumPosition:
    ticker: str
    opened_at: datetime
    entry_price: float
    initial_stop: float
    quantity: int
    remaining_quantity: int
    target_price: float
    high_water: float
    partial_taken: bool = False
    realized_pnl: float = 0.0

    @classmethod
    def from_fill(
        cls,
        *,
        ticker: str,
        opened_at: datetime,
        entry_price: float,
        stop_price: float,
        quantity: int,
        config: RiskConfigV1,
    ) -> MomentumPosition:
        if quantity <= 0 or entry_price <= stop_price:
            raise ValueError("position requires positive quantity and an entry above stop")
        risk = entry_price - stop_price
        return cls(
            ticker=ticker,
            opened_at=opened_at,
            entry_price=entry_price,
            initial_stop=stop_price,
            quantity=quantity,
            remaining_quantity=quantity,
            target_price=entry_price + risk * config.partial_at_r,
            high_water=entry_price,
        )

    def process_bar(
        self,
        *,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        config: RiskConfigV1,
    ) -> list[ExitInstruction]:
        """Return conservative exits; stop wins any same-bar target ambiguity."""

        if self.remaining_quantity <= 0:
            return []
        exits: list[ExitInstruction] = []
        active_stop = self.entry_price if self.partial_taken else self.initial_stop
        # With OHLC bars the intra-bar sequence is unknown; choose the adverse stop.
        if low <= active_stop:
            exits.append(ExitInstruction(self.ticker, timestamp, self.remaining_quantity, active_stop, "structure_stop" if not self.partial_taken else "breakeven_stop"))
            return exits
        self.high_water = max(self.high_water, high)
        if not self.partial_taken and high >= self.target_price:
            partial = min(self.remaining_quantity, max(1, int(self.quantity * config.partial_fraction)))
            exits.append(ExitInstruction(self.ticker, timestamp, partial, self.target_price, "partial_at_target"))
            if partial >= self.remaining_quantity:
                return exits
        elapsed = timestamp - self.opened_at
        if self.partial_taken and close < open_price:
            exits.append(ExitInstruction(self.ticker, timestamp, self.remaining_quantity, close, "first_red_after_extension"))
        elif elapsed >= timedelta(minutes=config.max_hold_minutes):
            exits.append(ExitInstruction(self.ticker, timestamp, self.remaining_quantity, close, "time_exit"))
        return exits

    def apply_exit(self, *, quantity: int, price: float) -> None:
        if quantity <= 0 or quantity > self.remaining_quantity:
            raise ValueError("exit quantity must be within the remaining position")
        self.realized_pnl += (price - self.entry_price) * quantity
        self.remaining_quantity -= quantity
        if quantity < self.quantity and self.remaining_quantity > 0:
            self.partial_taken = True


__all__ = ["ExitInstruction", "MomentumPosition"]
