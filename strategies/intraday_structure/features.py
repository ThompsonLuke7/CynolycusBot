from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import time
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np

from strategies.intraday_structure.models import Bar


ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class FeatureSnapshot:
    timestamp: object
    values: dict[str, float]

    def get(self, key: str, default: float = 0.0) -> float:
        value = self.values.get(key, default)
        return float(value) if value is not None and math.isfinite(float(value)) else float(default)

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)


def compute_features(
    bars: Sequence[Bar],
    *,
    spy_bars: Sequence[Bar] = (),
    qqq_bars: Sequence[Bar] = (),
    sector_bars: Sequence[Bar] = (),
) -> FeatureSnapshot:
    """Compute compact bar-close features using only ``bars <= current timestamp``."""
    if not bars:
        raise ValueError("at least one bar is required")
    ordered = sorted(bars, key=lambda b: b.timestamp)
    if any(a.timestamp >= b.timestamp for a, b in zip(ordered, ordered[1:])):
        raise ValueError("bars must have unique increasing timestamps")
    current = ordered[-1]
    closes = np.asarray([b.close for b in ordered], dtype=float)
    highs = np.asarray([b.high for b in ordered], dtype=float)
    lows = np.asarray([b.low for b in ordered], dtype=float)
    opens = np.asarray([b.open for b in ordered], dtype=float)
    volumes = np.asarray([b.volume for b in ordered], dtype=float)

    prev_closes = np.r_[closes[0], closes[:-1]]
    true_ranges = np.maximum.reduce([highs - lows, np.abs(highs - prev_closes), np.abs(lows - prev_closes)])
    atr = float(np.mean(true_ranges[-14:])) if len(true_ranges) else 0.0
    atr = max(atr, current.close * 1e-6)
    # Range compression: the 14-bar ATR against a 60-bar baseline. Below 1.0 the
    # tape is coiling, which is the condition under which a continuation setup
    # most often chops instead of running.
    atr_baseline = float(np.mean(true_ranges[-60:])) if len(true_ranges) else 0.0
    atr_baseline = max(atr_baseline, current.close * 1e-6)
    ranges = np.maximum(highs - lows, current.close * 1e-9)
    avg_range = float(np.mean(ranges[-21:-1])) if len(ranges) > 1 else float(ranges[-1])
    volume_ref = float(np.mean(volumes[-51:-1])) if len(volumes) > 1 else max(float(volumes[-1]), 1.0)
    volume5_ref = float(np.mean([sum(volumes[max(0, i - 4):i + 1]) for i in range(max(4, len(volumes) - 25), len(volumes) - 1)])) if len(volumes) > 6 else max(float(sum(volumes[-5:])), 1.0)
    vol_std = float(np.std(volumes[-51:-1], ddof=1)) if len(volumes) > 3 else 0.0

    session = _current_session_bars(ordered)
    session_vwaps = _running_vwap(session)
    session_vwap = session_vwaps[-1] if session_vwaps else current.close
    prior_session_vwap = session_vwaps[-6] if len(session_vwaps) >= 6 else session_vwaps[0] if session_vwaps else session_vwap
    above_vwap_duration = _trailing_true([b.close >= v for b, v in zip(session, session_vwaps)])
    below_vwap_duration = _trailing_true([b.close <= v for b, v in zip(session, session_vwaps)])

    ema9 = _ema(closes, 9)
    ema20 = _ema(closes, 20)
    ret1 = _return(closes, 1)
    ret3 = _return(closes, 3)
    prior_ret1 = (closes[-2] / closes[-3] - 1.0) if len(closes) >= 3 else 0.0
    direction = 1.0 if ret3 >= 0 else -1.0
    trend_strength = direction * abs(float(ema9[-1] - ema20[-1])) / atr
    close_location = float((current.close - current.low) / max(current.high - current.low, 1e-12))
    lower_wick = float((min(current.open, current.close) - current.low) / max(current.high - current.low, 1e-12))
    upper_wick = float((current.high - max(current.open, current.close)) / max(current.high - current.low, 1e-12))
    opening_high, opening_low = _opening_range(session, minutes=30)
    prior_high, prior_low = _prior_day_range(ordered)

    values = {
        "atr": atr,
        "atr_pct": atr / current.close,
        "atr_contraction": atr / atr_baseline,
        "session_vwap": session_vwap,
        "distance_to_vwap_atr": (current.close - session_vwap) / atr,
        "vwap_slope": (session_vwap - prior_session_vwap) / atr / max(1, min(5, len(session_vwaps))),
        "above_vwap_duration": float(above_vwap_duration),
        "below_vwap_duration": float(below_vwap_duration),
        "relative_volume_1m": current.volume / max(volume_ref, 1.0),
        "relative_volume_5m": float(sum(volumes[-5:])) / max(volume5_ref, 1.0),
        "volume_zscore": (current.volume - volume_ref) / vol_std if vol_std > 0 else 0.0,
        "range_expansion": float(ranges[-1]) / max(avg_range, 1e-12),
        "wick_to_range_ratio": max(lower_wick, upper_wick),
        "lower_wick_ratio": lower_wick,
        "upper_wick_ratio": upper_wick,
        "close_location_value": close_location,
        "ret_1": ret1,
        "ret_3": ret3,
        "selloff_acceleration": min(0.0, ret1 - prior_ret1),
        "downside_momentum_deceleration": max(0.0, ret1 - prior_ret1),
        "rebound_velocity": max(0.0, ret1) / max(atr / current.close, 1e-9),
        "ema_9": float(ema9[-1]),
        "ema_20": float(ema20[-1]),
        "distance_to_ema9_atr": (current.close - float(ema9[-1])) / atr,
        "distance_to_ema20_atr": (current.close - float(ema20[-1])) / atr,
        "trend_strength": trend_strength,
        "micro_higher_low": float(len(lows) >= 3 and lows[-1] > lows[-2] > lows[-3]),
        "micro_lower_high": float(len(highs) >= 3 and highs[-1] < highs[-2] < highs[-3]),
        "micro_swing_high": float(np.max(highs[-6:-1])) if len(highs) >= 2 else current.high,
        "micro_swing_low": float(np.min(lows[-6:-1])) if len(lows) >= 2 else current.low,
        "opening_range_high": opening_high,
        "opening_range_low": opening_low,
        "opening_range_position": _range_position(current.close, opening_low, opening_high),
        "prior_day_high": prior_high,
        "prior_day_low": prior_low,
        "prior_day_range_position": _range_position(current.close, prior_low, prior_high),
        "relative_strength_vs_spy": ret3 - _aligned_return(spy_bars, current.timestamp, 3),
        "relative_strength_vs_qqq": ret3 - _aligned_return(qqq_bars, current.timestamp, 3),
        "relative_strength_vs_sector": ret3 - _aligned_return(sector_bars, current.timestamp, 3),
        "bar_dollar_volume": current.close * current.volume,
    }
    values["capitulation_volume_score"] = max(0.0, values["relative_volume_1m"] - 1.0) * max(0.0, values["range_expansion"] - 1.0)
    values["pivot_break_strength"] = 0.0
    values["retest_depth"] = 0.0
    values["momentum_divergence"] = float(ret1 < 0 < prior_ret1 or ret1 > 0 > prior_ret1)
    return FeatureSnapshot(timestamp=current.timestamp, values=values)


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(values), dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _return(values: np.ndarray, periods: int) -> float:
    if len(values) <= periods or values[-periods - 1] <= 0:
        return 0.0
    return float(values[-1] / values[-periods - 1] - 1.0)


def _current_session_bars(bars: Sequence[Bar]) -> list[Bar]:
    current_date = bars[-1].timestamp.astimezone(ET).date()
    rth = [b for b in bars if b.timestamp.astimezone(ET).date() == current_date and time(9, 30) <= b.timestamp.astimezone(ET).time() < time(16, 0)]
    if rth:
        return rth
    return [b for b in bars if b.timestamp.astimezone(ET).date() == current_date]


def _running_vwap(bars: Sequence[Bar]) -> list[float]:
    out: list[float] = []
    cumulative_pv = 0.0
    cumulative_volume = 0.0
    for bar in bars:
        typical = (bar.high + bar.low + bar.close) / 3.0
        cumulative_pv += typical * bar.volume
        cumulative_volume += bar.volume
        out.append(cumulative_pv / cumulative_volume if cumulative_volume > 0 else bar.close)
    return out


def _trailing_true(flags: Sequence[bool]) -> int:
    count = 0
    for flag in reversed(flags):
        if not flag:
            break
        count += 1
    return count


def _opening_range(session: Sequence[Bar], minutes: int) -> tuple[float, float]:
    count = max(1, minutes)
    subset = list(session[:count])
    if not subset:
        return math.nan, math.nan
    return max(x.high for x in subset), min(x.low for x in subset)


def _prior_day_range(bars: Sequence[Bar]) -> tuple[float, float]:
    current_date = bars[-1].timestamp.astimezone(ET).date()
    prior_dates = sorted({b.timestamp.astimezone(ET).date() for b in bars if b.timestamp.astimezone(ET).date() < current_date})
    if not prior_dates:
        return math.nan, math.nan
    prior = [b for b in bars if b.timestamp.astimezone(ET).date() == prior_dates[-1]]
    return max(b.high for b in prior), min(b.low for b in prior)


def _range_position(value: float, low: float, high: float) -> float:
    if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
        return 0.5
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _aligned_return(bars: Sequence[Bar], timestamp: object, periods: int) -> float:
    eligible = [b.close for b in bars if b.timestamp <= timestamp]
    if len(eligible) <= periods or eligible[-periods - 1] <= 0:
        return 0.0
    return float(eligible[-1] / eligible[-periods - 1] - 1.0)
