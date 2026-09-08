"""Portfolio fuses and size calculations for paper-only momentum entries."""
from __future__ import annotations

from dataclasses import dataclass, field

from strategies.momentum_scalper.configs.v1 import RiskConfigV1


@dataclass
class DailyRiskState:
    open_positions: int = 0
    entries_today: int = 0
    realized_r: float = 0.0
    consecutive_losses: int = 0
    killed: bool = False


def entry_rejection_reason(state: DailyRiskState, config: RiskConfigV1) -> str | None:
    if state.killed:
        return "manual_kill_switch"
    if state.open_positions >= config.max_concurrent_positions:
        return "max_concurrent_positions"
    if state.entries_today >= config.max_trades_per_day:
        return "max_trades_per_day"
    if state.realized_r <= -config.max_daily_loss_r:
        return "max_daily_loss"
    if state.consecutive_losses >= config.max_consecutive_losses:
        return "max_consecutive_losses"
    return None


def size_shares(
    *,
    entry_price: float,
    stop_price: float,
    risk_budget_dollars: float,
    ask_size: float | None,
    config: RiskConfigV1,
) -> int:
    """Risk-size a long entry and cap it by notional and displayed liquidity."""

    per_share_risk = entry_price - stop_price
    if entry_price <= 0 or stop_price <= 0 or per_share_risk <= 0 or risk_budget_dollars <= 0:
        return 0
    by_risk = int(risk_budget_dollars // per_share_risk)
    by_notional = int(config.max_position_notional // entry_price)
    caps = [by_risk, by_notional]
    if ask_size is not None and ask_size > 0:
        caps.append(max(1, int(ask_size * config.max_quote_participation)))
    return max(0, min(caps))


@dataclass
class MomentumRiskBook:
    config: RiskConfigV1
    state: DailyRiskState = field(default_factory=DailyRiskState)

    def allow_entry(self) -> tuple[bool, str]:
        reason = entry_rejection_reason(self.state, self.config)
        return reason is None, reason or "ok"

    def record_entry(self) -> None:
        self.state.entries_today += 1
        self.state.open_positions += 1

    def record_close(self, realized_r: float) -> None:
        self.state.open_positions = max(0, self.state.open_positions - 1)
        self.state.realized_r += realized_r
        self.state.consecutive_losses = self.state.consecutive_losses + 1 if realized_r < 0 else 0

    def kill(self) -> None:
        self.state.killed = True


__all__ = ["DailyRiskState", "MomentumRiskBook", "entry_rejection_reason", "size_shares"]
