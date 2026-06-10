"""Exit simulation policies."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategies.momentum_scalper.configs.settings import ExitConfig


@dataclass(frozen=True)
class ExitResult:
    exit_price: float
    exit_timestamp: pd.Timestamp
    reason: str
    pnl_pct: float
    mfe_pct: float
    mae_pct: float
    right_tail_capture: float
    average_giveback: float


def simulate_exit(forward_bars: pd.DataFrame, entry_price: float, entry_timestamp: pd.Timestamp, config: ExitConfig = ExitConfig()) -> ExitResult | None:
    if forward_bars.empty or entry_price <= 0:
        return None
    risk = entry_price * (config.stop_risk / 100.0)
    take_profit = entry_price + risk * config.reward_risk
    stop = entry_price - risk
    high_water = entry_price
    mfe = 0.0
    mae = 0.0
    exit_price = float(forward_bars.iloc[-1]["close"])
    exit_ts = forward_bars.iloc[-1]["timestamp"]
    reason = "time stop"
    for _, bar in forward_bars.iterrows():
        high_water = max(high_water, float(bar["high"]))
        mfe = max(mfe, (float(bar["high"]) / entry_price - 1.0) * 100.0)
        mae = min(mae, (float(bar["low"]) / entry_price - 1.0) * 100.0)
        trail = high_water * (1.0 - config.atr_trail_mult / 100.0)
        if float(bar["high"]) >= take_profit:
            exit_price, exit_ts, reason = take_profit, bar["timestamp"], "hard TP"
            break
        if float(bar["low"]) <= stop:
            exit_price, exit_ts, reason = stop, bar["timestamp"], "hard stop"
            break
        if high_water > take_profit and float(bar["close"]) < trail:
            exit_price, exit_ts, reason = float(bar["close"]), bar["timestamp"], "ATR trailing stop"
            break
        if (bar["timestamp"] - entry_timestamp).total_seconds() / 60.0 >= config.max_hold_minutes:
            exit_price, exit_ts, reason = float(bar["close"]), bar["timestamp"], "time stop"
            break
    pnl = (exit_price / entry_price - 1.0) * 100.0
    capture = pnl / mfe if mfe > 0 else 0.0
    giveback = max(mfe - pnl, 0.0)
    return ExitResult(exit_price, exit_ts, reason, pnl, mfe, mae, capture, giveback)
