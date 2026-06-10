"""Live trading orchestration shell.

This module intentionally does not place broker orders. It produces actionable
orders from scanner/features/model/entry/exit logic, leaving broker adapters and
account risk controls to be attached explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategies.momentum_scalper.execution.entry_policy import EntrySignal, evaluate_entry
from strategies.momentum_scalper.live.live_features import build_live_rankings
from strategies.momentum_scalper.live.live_scanner import scan_once


@dataclass(frozen=True)
class TradeIntent:
    timestamp: pd.Timestamp
    ticker: str
    side: str
    pattern: str
    score: float
    probability: float
    entry_signal: EntrySignal


def generate_trade_intents(provider, min_probability: float = 0.55, top_n: int = 10) -> list[TradeIntent]:
    bars = provider()
    snapshot = scan_once(lambda: bars, top_n=top_n)
    ranked = build_live_rankings(snapshot, bars)
    intents: list[TradeIntent] = []
    if ranked.empty:
        return intents
    for _, row in ranked.head(top_n).iterrows():
        probability = float(row.get("breakout_quality_probability", row.get("score", 0.0)))
        if probability < min_probability:
            continue
        signal = evaluate_entry(row)
        if signal.should_enter:
            intents.append(
                TradeIntent(
                    timestamp=row["timestamp"],
                    ticker=row["ticker"],
                    side="buy",
                    pattern=signal.pattern,
                    score=float(row.get("score", probability)),
                    probability=probability,
                    entry_signal=signal,
                )
            )
    return intents
