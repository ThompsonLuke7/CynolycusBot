from __future__ import annotations

import math
from collections import defaultdict
from datetime import time
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from signals.location_features import add_liquidity_zone_features
from strategies.intraday_structure.config import LevelPolicy
from strategies.intraday_structure.models import Bar, Candidate, OptionsContext, StructuralLevel


ET = ZoneInfo("America/New_York")


class StructuralLevelProvider:
    """Unifies causal technical, session, volume-profile, and options levels."""

    def __init__(self, policy: LevelPolicy) -> None:
        self.policy = policy

    def levels(
        self,
        *,
        bars: Sequence[Bar],
        candidate: Candidate,
        options: OptionsContext,
        features: dict[str, float],
    ) -> list[StructuralLevel]:
        if not bars:
            return []
        current = bars[-1]
        atr = max(float(features.get("atr", 0.0)), current.close * 1e-6)
        raw: list[StructuralLevel] = []
        raw.extend(self._session_levels(bars, candidate))
        raw.extend(self._liquidity_levels(bars))
        raw.extend(self._volume_profile_levels(bars))
        raw.extend(self._round_numbers(current.close, atr))
        raw.extend(options.levels)
        raw.extend(_candidate_levels(candidate))
        return cluster_levels(
            raw, spot=current.close, atr=atr,
            atr_threshold=self.policy.cluster_atr,
            pct_threshold=self.policy.cluster_pct,
        )

    def _session_levels(self, bars: Sequence[Bar], candidate: Candidate) -> list[StructuralLevel]:
        current = bars[-1]
        today = current.timestamp.astimezone(ET).date()
        today_bars = [b for b in bars if b.timestamp.astimezone(ET).date() == today]
        rth = [b for b in today_bars if time(9, 30) <= b.timestamp.astimezone(ET).time() < time(16, 0)]
        premarket = [b for b in today_bars if time(self.policy.premarket_start_hour_et) <= b.timestamp.astimezone(ET).time() < time(9, 30)]
        prior_dates = sorted({b.timestamp.astimezone(ET).date() for b in bars if b.timestamp.astimezone(ET).date() < today})
        out: list[StructuralLevel] = []
        if prior_dates:
            prior = [b for b in bars if b.timestamp.astimezone(ET).date() == prior_dates[-1]]
            out.extend([
                _level(max(x.high for x in prior), "prior_day_high", 0.82, "resistance"),
                _level(min(x.low for x in prior), "prior_day_low", 0.82, "support"),
                _level(prior[-1].close, "prior_day_close", 0.60, "both"),
            ])
        if premarket:
            out.extend([
                _level(max(x.high for x in premarket), "premarket_high", 0.72, "resistance"),
                _level(min(x.low for x in premarket), "premarket_low", 0.72, "support"),
            ])
        opening_bars = rth[: self.policy.opening_range_minutes]
        if opening_bars:
            out.extend([
                _level(max(x.high for x in opening_bars), "opening_range_high", 0.78, "resistance"),
                _level(min(x.low for x in opening_bars), "opening_range_low", 0.78, "support"),
            ])
        if rth:
            session_vwap = _vwap(rth)
            out.append(_level(session_vwap, "session_vwap", 0.68, "both"))
            anchored = [b for b in rth if b.timestamp >= candidate.timestamp]
            if anchored:
                out.append(_level(_vwap(anchored), "candidate_anchored_vwap", 0.62, "both"))
        lookback = list(bars[-self.policy.swing_lookback_bars:])
        if len(lookback) >= 3:
            out.extend([
                _level(max(x.high for x in lookback[:-1]), "intraday_swing_high", 0.66, "resistance"),
                _level(min(x.low for x in lookback[:-1]), "intraday_swing_low", 0.66, "support"),
            ])
        return out

    def _liquidity_levels(self, bars: Sequence[Bar]) -> list[StructuralLevel]:
        subset = list(bars[-max(80, self.policy.swing_lookback_bars):])
        if len(subset) < 5:
            return []
        frame = pd.DataFrame([b.to_dict() for b in subset]).set_index("timestamp")
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
        enriched, _, _ = add_liquidity_zone_features(
            frame, lookback=min(78, len(frame)), swing_window=min(20, max(2, len(frame) // 3))
        )
        row = enriched.iloc[-1]
        result: list[StructuralLevel] = []
        for side, direction in (("support", "support"), ("resistance", "resistance")):
            price = _finite(row.get(f"nearest_{side}_zone"))
            if price is None:
                continue
            strength_raw = _finite(row.get(f"{side}_zone_strength")) or 0.0
            touches = int(_finite(row.get(f"{side}_touch_count")) or 0)
            result.append(StructuralLevel(
                price=price, level_type=f"liquidity_{side}_zone",
                strength=float(np.clip(0.35 + 0.04 * strength_raw, 0.35, 0.90)),
                directionality=direction, touch_count=touches,
                rejection_count=int(_finite(row.get("failed_breakdown_count" if side == "support" else "failed_breakout_count")) or 0),
                metadata={"provider": "signals.location_features.add_liquidity_zone_features"},
            ))
        return result

    def _volume_profile_levels(self, bars: Sequence[Bar]) -> list[StructuralLevel]:
        subset = list(bars[-max(30, self.policy.swing_lookback_bars):])
        if len(subset) < 12 or sum(b.volume for b in subset) <= 0:
            return []
        prices = np.asarray([(b.high + b.low + b.close) / 3.0 for b in subset])
        weights = np.asarray([b.volume for b in subset], dtype=float)
        lo, hi = float(prices.min()), float(prices.max())
        if hi <= lo:
            return []
        hist, edges = np.histogram(prices, bins=self.policy.volume_profile_bins, range=(lo, hi), weights=weights)
        centers = (edges[:-1] + edges[1:]) / 2.0
        high_idx = int(np.argmax(hist))
        nonzero = np.flatnonzero(hist > 0)
        low_idx = int(nonzero[np.argmin(hist[nonzero])]) if len(nonzero) else high_idx
        return [
            _level(float(centers[high_idx]), "volume_profile_hvn", 0.64, "both", volume=float(hist[high_idx])),
            _level(float(centers[low_idx]), "volume_profile_lvn", 0.46, "both", volume=float(hist[low_idx])),
        ]

    def _round_numbers(self, spot: float, atr: float) -> list[StructuralLevel]:
        out: list[StructuralLevel] = []
        for step in self.policy.round_number_steps:
            if step <= 0:
                continue
            center = round(spot / step) * step
            for price in (center - step, center, center + step):
                if price > 0 and abs(price - spot) <= max(3.0 * atr, 2.0 * step):
                    out.append(_level(price, f"round_number_{step:g}", 0.34, "both"))
        return out


def cluster_levels(
    levels: Iterable[StructuralLevel], *, spot: float, atr: float,
    atr_threshold: float, pct_threshold: float,
) -> list[StructuralLevel]:
    valid = [x for x in levels if math.isfinite(x.price) and x.price > 0]
    if not valid:
        return []
    threshold = max(abs(atr) * atr_threshold, abs(spot) * pct_threshold, abs(spot) * 1e-6)
    groups: list[list[StructuralLevel]] = []
    for level in sorted(valid, key=lambda x: x.price):
        if not groups or abs(level.price - groups[-1][-1].price) > threshold:
            groups.append([level])
        else:
            groups[-1].append(level)
    out: list[StructuralLevel] = []
    for group in groups:
        weights = np.asarray([max(x.strength, 0.05) for x in group])
        price = float(np.average([x.price for x in group], weights=weights))
        types = sorted({x.level_type for x in group})
        directionality = _merge_directionality(x.directionality for x in group)
        out.append(StructuralLevel(
            price=price, level_type="+".join(types),
            strength=float(np.clip(1.0 - np.prod([1.0 - min(max(x.strength, 0.0), 0.99) for x in group]), 0.0, 1.0)),
            freshness_bars=min(x.freshness_bars for x in group), directionality=directionality,
            touch_count=sum(x.touch_count for x in group), rejection_count=sum(x.rejection_count for x in group),
            break_status=_strongest_status([x.break_status for x in group], ("broken", "testing", "unbroken")),
            hold_status=_strongest_status([x.hold_status for x in group], ("held", "testing", "untested")),
            distance_from_spot=(price - spot) / spot,
            metadata={"sources": types, "cluster_size": len(group)},
        ))
    return out


def nearest_level(levels: Sequence[StructuralLevel], spot: float, direction: str, *, beyond: float | None = None) -> StructuralLevel | None:
    anchor = spot if beyond is None else beyond
    if direction == "long":
        candidates = [x for x in levels if x.price > anchor and x.directionality != "support"]
        return min(candidates, key=lambda x: x.price) if candidates else None
    candidates = [x for x in levels if x.price < anchor and x.directionality != "resistance"]
    return max(candidates, key=lambda x: x.price) if candidates else None


def nearest_support(levels: Sequence[StructuralLevel], spot: float) -> StructuralLevel | None:
    candidates = [x for x in levels if x.price <= spot and x.directionality != "resistance"]
    return max(candidates, key=lambda x: x.price) if candidates else None


def nearest_resistance(levels: Sequence[StructuralLevel], spot: float) -> StructuralLevel | None:
    candidates = [x for x in levels if x.price >= spot and x.directionality != "support"]
    return min(candidates, key=lambda x: x.price) if candidates else None


def _candidate_levels(candidate: Candidate) -> list[StructuralLevel]:
    out: list[StructuralLevel] = []
    if candidate.pivot is not None:
        out.append(_level(candidate.pivot, "candidate_pivot", 0.88, "both"))
    for item in candidate.metadata.get("structural_levels", []):
        try:
            out.append(StructuralLevel.from_mapping(item))
        except (TypeError, ValueError):
            continue
    return out


def _level(price: float, kind: str, strength: float, directionality: str, **metadata) -> StructuralLevel:
    return StructuralLevel(price=float(price), level_type=kind, strength=strength, directionality=directionality, metadata=metadata)


def _vwap(bars: Sequence[Bar]) -> float:
    volume = sum(b.volume for b in bars)
    if volume <= 0:
        return bars[-1].close
    return sum(((b.high + b.low + b.close) / 3.0) * b.volume for b in bars) / volume


def _finite(value) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _merge_directionality(values: Iterable[str]) -> str:
    unique = set(values)
    return next(iter(unique)) if len(unique) == 1 else "both"


def _strongest_status(values: Sequence[str], ordering: Sequence[str]) -> str:
    for item in ordering:
        if item in values:
            return item
    return values[0] if values else "unknown"
