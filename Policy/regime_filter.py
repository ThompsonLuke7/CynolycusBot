from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


Regime = Literal["bullish", "bearish", "neutral"]


@dataclass(frozen=True)
class StickyRegimeConfig:
    atr_length: int = 14
    ema_spread_min_atr: float = 0.15
    slope_min_atr: float = 0.05
    enter_confirm_bars: int = 3
    flip_confirm_bars: int = 3
    neutral_confirm_bars: int = 4
    stay_rule_min_count: int = 2


def _to_float_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        raise KeyError(f"Missing required regime column: {col}")
    return pd.to_numeric(frame[col], errors="coerce")


def _atr_proxy(frame: pd.DataFrame, length: int) -> pd.Series:
    high = _to_float_series(frame, "high")
    low = _to_float_series(frame, "low")
    close = _to_float_series(frame, "close")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(max(1, int(length)), min_periods=3).mean()
    return atr.bfill().ffill().clip(lower=1e-9)


def add_sticky_trend_regime(
    frame: pd.DataFrame,
    *,
    config: StickyRegimeConfig | None = None,
    output_col: str = "trend_regime",
) -> pd.DataFrame:
    """
    Add raw rule checks plus a sticky bullish/bearish/neutral regime column.

    The raw checks answer "what is true on this bar?" The sticky regime answers
    "what market state should we actually treat this as after confirmation,
    hysteresis, and neutral persistence?"
    """
    cfg = config or StickyRegimeConfig()
    out = frame.copy()
    close = _to_float_series(out, "close")
    ema_fast = _to_float_series(out, "ema_fast")
    ema_slow = _to_float_series(out, "ema_slow")
    atr = _atr_proxy(out, cfg.atr_length)

    slope = ema_slow.diff(3)
    spread = (ema_fast - ema_slow).abs()
    out["regime_atr_proxy"] = atr
    out["ema_slow_slope_3"] = slope
    out["ema_spread_atr"] = spread / atr
    out["ema_slow_slope_3_atr"] = slope / atr

    out["fast_gt_slow"] = ema_fast > ema_slow
    out["close_ge_fast"] = close >= ema_fast
    out["slope_up"] = slope > (float(cfg.slope_min_atr) * atr)
    out["fast_lt_slow"] = ema_fast < ema_slow
    out["close_le_fast"] = close <= ema_fast
    out["slope_down"] = slope < (-float(cfg.slope_min_atr) * atr)
    out["ema_spread_ok"] = spread >= (float(cfg.ema_spread_min_atr) * atr)

    bull_rule_cols = ["fast_gt_slow", "close_ge_fast", "slope_up"]
    bear_rule_cols = ["fast_lt_slow", "close_le_fast", "slope_down"]
    out["bull_rule_count"] = out[bull_rule_cols].sum(axis=1)
    out["bear_rule_count"] = out[bear_rule_cols].sum(axis=1)
    out["raw_bullish_regime"] = out["ema_spread_ok"] & (out["bull_rule_count"] == 3)
    out["raw_bearish_regime"] = out["ema_spread_ok"] & (out["bear_rule_count"] == 3)
    out["raw_trend_regime"] = np.select(
        [out["raw_bullish_regime"], out["raw_bearish_regime"]],
        ["bullish", "bearish"],
        default="neutral",
    )

    regimes = _sticky_regime_from_rules(out, cfg)
    out[output_col] = regimes
    return out


def _sticky_regime_from_rules(frame: pd.DataFrame, cfg: StickyRegimeConfig) -> list[Regime]:
    current: Regime = "neutral"
    bull_confirm = 0
    bear_confirm = 0
    weak_confirm = 0
    regimes: list[Regime] = []

    enter_confirm = max(1, int(cfg.enter_confirm_bars))
    flip_confirm = max(1, int(cfg.flip_confirm_bars))
    neutral_confirm = max(1, int(cfg.neutral_confirm_bars))
    stay_min = max(0, int(cfg.stay_rule_min_count))

    for row in frame.itertuples(index=False):
        raw_bull = bool(getattr(row, "raw_bullish_regime"))
        raw_bear = bool(getattr(row, "raw_bearish_regime"))
        bull_count = int(getattr(row, "bull_rule_count"))
        bear_count = int(getattr(row, "bear_rule_count"))
        spread_ok = bool(getattr(row, "ema_spread_ok"))

        bull_confirm = bull_confirm + 1 if raw_bull else 0
        bear_confirm = bear_confirm + 1 if raw_bear else 0

        if current == "neutral":
            weak_confirm = 0
            if bull_confirm >= enter_confirm and bull_confirm >= bear_confirm:
                current = "bullish"
                bear_confirm = 0
            elif bear_confirm >= enter_confirm and bear_confirm > bull_confirm:
                current = "bearish"
                bull_confirm = 0
        elif current == "bullish":
            weak = (bull_count < stay_min) or (not spread_ok and bull_count < 3)
            weak_confirm = weak_confirm + 1 if weak else 0
            if bear_confirm >= flip_confirm:
                current = "bearish"
                bull_confirm = 0
                weak_confirm = 0
            elif weak_confirm >= neutral_confirm:
                current = "neutral"
                weak_confirm = 0
        else:
            weak = (bear_count < stay_min) or (not spread_ok and bear_count < 3)
            weak_confirm = weak_confirm + 1 if weak else 0
            if bull_confirm >= flip_confirm:
                current = "bullish"
                bear_confirm = 0
                weak_confirm = 0
            elif weak_confirm >= neutral_confirm:
                current = "neutral"
                weak_confirm = 0

        regimes.append(current)

    return regimes
