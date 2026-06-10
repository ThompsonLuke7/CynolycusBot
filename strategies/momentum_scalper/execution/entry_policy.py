"""Entry pattern detection and risk gates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategies.momentum_scalper.configs.settings import EntryConfig


@dataclass(frozen=True)
class EntrySignal:
    should_enter: bool
    pattern: str
    trigger_price: float
    reason: str = ""


def _passes_filters(row: pd.Series, config: EntryConfig) -> tuple[bool, str]:
    if float(row.get("spread_pct", 0.0) or 0.0) > config.max_spread_pct:
        return False, "spread"
    if float(row.get("liquidity_score", 0.0) or 0.0) < config.min_liquidity_score:
        return False, "liquidity"
    if float(row.get("halt_count", 0.0) or 0.0) > 0:
        return False, "halt_risk"
    if float(row.get("dist_to_hod", 0.0) or 0.0) < -config.max_chase_pct_above_trigger:
        return False, "anti_chase"
    return True, ""


def evaluate_entry(row: pd.Series, config: EntryConfig = EntryConfig()) -> EntrySignal:
    passed, reason = _passes_filters(row, config)
    price = float(row.get("close", np.nan) if "close" in row else np.nan)
    if not passed:
        return EntrySignal(False, "none", price, reason)
    patterns = [
        ("HOD breakout", row.get("premarket_high_break", 0)),
        ("first pullback", row.get("micro_pullback", 0)),
        ("bull flag breakout", (row.get("bull_flag_tightness", 1) < 0.015) and (row.get("breakout_velocity", 0) > 0)),
        ("flat-top breakout", row.get("flat_top_breakout", 0)),
        ("VWAP reclaim", row.get("dist_to_vwap", -1) > 0),
        ("opening range breakout", row.get("opening_range_break", 0)),
    ]
    for pattern, ok in patterns:
        if bool(ok):
            return EntrySignal(True, pattern, price, "ok")
    return EntrySignal(False, "none", price, "no_pattern")
