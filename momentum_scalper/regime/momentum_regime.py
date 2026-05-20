"""Momentum regime classifier."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RegimeState:
    regime: str
    aggression_multiplier: float
    score: float


def classify_momentum_regime(
    daily_halt_count: float = 0.0,
    average_runner_extension: float = 0.0,
    small_cap_breadth: float = 0.0,
    gapper_success_rate: float = 0.0,
    iwm_trend: float = 0.0,
    vix: float = 20.0,
) -> RegimeState:
    score = (
        0.15 * min(daily_halt_count / 10.0, 1.0)
        + 0.25 * min(max(average_runner_extension, 0.0) / 30.0, 1.0)
        + 0.20 * min(max(small_cap_breadth, 0.0), 1.0)
        + 0.25 * min(max(gapper_success_rate, 0.0), 1.0)
        + 0.10 * (1.0 if iwm_trend > 0 else 0.0)
        - 0.05 * (1.0 if vix > 30 else 0.0)
    )
    if score >= 0.65:
        return RegimeState("HOT", 1.25, float(score))
    if score <= 0.30:
        return RegimeState("DEAD", 0.50, float(score))
    return RegimeState("NORMAL", 1.0, float(score))


def classify_from_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        state = classify_momentum_regime(**{k: row.get(k, 0.0) for k in ["daily_halt_count", "average_runner_extension", "small_cap_breadth", "gapper_success_rate", "iwm_trend", "vix"]})
        rows.append({**row.to_dict(), "regime": state.regime, "aggression_multiplier": state.aggression_multiplier, "regime_score": state.score})
    return pd.DataFrame(rows)
