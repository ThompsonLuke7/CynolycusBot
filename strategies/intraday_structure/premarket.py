"""Pre-open trade plans: trigger, invalidation, and the full target ladder.

WHY. The engine already computes every ingredient of a plan -- fused levels, a
runway to the next destination, a reward:risk, a regime -- but only ever at the
moment a one-minute bar confirms something, and only one target rung at a time.
Nothing assembles them before the open, so there is no artifact that says "I
only care about NVDA above 226.50, and if it goes the next vacuum is 231.80".

Two things this deliberately does NOT do:

* **It does not reuse ``StructuralLevelProvider``.**  That provider's session
  logic (opening range = first N *bars*, premarket window by ET clock time) is
  correct for one-minute bars and quietly wrong for hourly ones -- an "opening
  range" of the first 30 hourly bars is six days.  The level families here are
  the ones daily and hourly bars honestly support.  The shared *fusion* is
  reused: ``cluster_levels`` and ``score_runway`` are the same code the live
  engine runs.
* **It does not invent an overnight session.**  ``Data/shared/bars/1h`` holds
  ET 10:00-16:00 only, so overnight inventory, gap, and premarket high/low are
  not computable from stored data.  Every plan carries an explicit warning
  saying so rather than presenting a prior close as if it were a pre-open read.

Causality: the newest input is the prior session's close and a dealer snapshot
captured at roughly 15:45 the prior day.  Both precede the open this plans for.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd

from dataclasses import replace as _replace

from strategies.intraday_structure.config import IntradayStructureConfig, RegimePolicy
from strategies.intraday_structure.levels import cluster_levels, nearest_level
from strategies.intraday_structure.models import Direction, OptionsContext, StructuralLevel
from strategies.intraday_structure.regime import classify_context
from strategies.intraday_structure.runway import score_runway
from strategies.intraday_structure.models import MarketContext


#: How many rungs to publish up front. The live engine discovers rung 2 only
#: after reaching rung 1; a plan has to name the whole path so reward:risk can
#: be judged before entering.
TARGET_LADDER_DEPTH = 3

#: Stored bars end at the prior close, so the plan cannot see the overnight
#: session. Stated on every plan rather than silently assumed away.
OVERNIGHT_GAP_WARNING = "no_overnight_or_premarket_bars_in_shared_cache"


def premarket_regime_policy(policy: RegimePolicy) -> RegimePolicy:
    """The intraday regime policy, minus the test that cannot mean anything here.

    ``trapped_between_levels`` asks whether price is boxed relative to how far
    it travels over the horizon being traded. Pre-open the levels ARE the daily
    range and the ATR IS the daily ATR, so prior-day high and low bracket spot
    at roughly one ATR by construction and the test fired on 9 of 12 names in
    the first real run -- a flag that is always on. The other two components
    (ATR14/ATR60 contraction, ATR-normalised trend) are pure ratios and carry
    over unchanged.
    """
    return _replace(policy, trapped_room_atr=None)


@dataclass(frozen=True)
class TradePlan:
    ticker: str
    direction: str
    sources: list[str]
    score: float
    reference_price: float
    reference_as_of: str
    atr: float
    trigger: float | None
    trigger_level_type: str | None
    trigger_level_sources: list[str]
    trigger_distance_atr: float | None
    invalidation: float | None
    targets: list[float]
    target_level_types: list[str]
    obstacles: list[dict[str, Any]]
    runway_score: float | None
    reward_risk: float | None
    context_regime: str
    regime_evidence: list[str]
    no_trade_reason: str | None
    warnings: list[str] = field(default_factory=list)

    @property
    def actionable(self) -> bool:
        return self.no_trade_reason is None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["actionable"] = self.actionable
        return out


@dataclass(frozen=True)
class PremarketPlan:
    schema_version: str
    generated_at: str
    session: str
    engine_version: str
    market: list[dict[str, Any]]
    setups: list[dict[str, Any]]
    avoid: list[dict[str, Any]]
    warnings: list[str]
    inputs: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PLAN_SCHEMA_VERSION = "intraday_structure_premarket_plan_v1"


# ---------------------------------------------------------------------------
# Levels that daily + hourly bars actually support
# ---------------------------------------------------------------------------

def build_levels(
    *,
    daily: pd.DataFrame,
    hourly: pd.DataFrame,
    spot: float,
    atr: float,
    options: OptionsContext | None = None,
    config: IntradayStructureConfig | None = None,
) -> list[StructuralLevel]:
    """Fuse the level families that stored daily/hourly bars can justify."""
    config = config or IntradayStructureConfig()
    raw: list[StructuralLevel] = []
    raw.extend(_prior_session_levels(daily))
    raw.extend(_multi_day_swing_levels(daily))
    raw.extend(_hourly_structure_levels(hourly))
    raw.extend(_volume_profile_levels(hourly, bins=config.levels.volume_profile_bins))
    raw.extend(_round_numbers(spot, atr, config.levels.round_number_steps))
    if options is not None:
        raw.extend(options.levels)
    return cluster_levels(
        raw, spot=spot, atr=atr,
        atr_threshold=config.levels.cluster_atr,
        pct_threshold=config.levels.cluster_pct,
    )


def _prior_session_levels(daily: pd.DataFrame) -> list[StructuralLevel]:
    if daily.empty:
        return []
    last = daily.iloc[-1]
    out = [
        _level(float(last["high"]), "prior_day_high", 0.82, "resistance"),
        _level(float(last["low"]), "prior_day_low", 0.82, "support"),
        _level(float(last["close"]), "prior_day_close", 0.60, "both"),
    ]
    week = daily.tail(5)
    if len(week) >= 2:
        out.append(_level(float(week["high"].max()), "prior_week_high", 0.74, "resistance"))
        out.append(_level(float(week["low"].min()), "prior_week_low", 0.74, "support"))
    return out


def _multi_day_swing_levels(daily: pd.DataFrame) -> list[StructuralLevel]:
    out: list[StructuralLevel] = []
    for window, strength in ((20, 0.70), (60, 0.66)):
        subset = daily.tail(window)
        if len(subset) < max(3, window // 4):
            continue
        out.append(_level(float(subset["high"].max()), f"swing_high_{window}d", strength, "resistance"))
        out.append(_level(float(subset["low"].min()), f"swing_low_{window}d", strength, "support"))
    return out


def _hourly_structure_levels(hourly: pd.DataFrame) -> list[StructuralLevel]:
    subset = hourly.tail(35)  # roughly the last five RTH sessions
    if len(subset) < 4:
        return []
    return [
        _level(float(subset["high"].max()), "hourly_swing_high", 0.64, "resistance"),
        _level(float(subset["low"].min()), "hourly_swing_low", 0.64, "support"),
        _level(float(subset.iloc[-1]["close"]), "last_hourly_close", 0.50, "both"),
    ]


def _volume_profile_levels(hourly: pd.DataFrame, *, bins: int) -> list[StructuralLevel]:
    subset = hourly.tail(70)
    if len(subset) < 12 or float(subset["volume"].sum()) <= 0:
        return []
    prices = ((subset["high"] + subset["low"] + subset["close"]) / 3.0).to_numpy(dtype=float)
    weights = subset["volume"].to_numpy(dtype=float)
    low, high = float(prices.min()), float(prices.max())
    if high <= low:
        return []
    hist, edges = np.histogram(prices, bins=bins, range=(low, high), weights=weights)
    centers = (edges[:-1] + edges[1:]) / 2.0
    nonzero = np.flatnonzero(hist > 0)
    if not len(nonzero):
        return []
    return [
        _level(float(centers[int(np.argmax(hist))]), "volume_profile_hvn", 0.64, "both"),
        _level(float(centers[int(nonzero[np.argmin(hist[nonzero])])]), "volume_profile_lvn", 0.46, "both"),
    ]


def _round_numbers(spot: float, atr: float, steps: Sequence[float]) -> list[StructuralLevel]:
    out: list[StructuralLevel] = []
    for step in steps:
        if step <= 0:
            continue
        center = round(spot / step) * step
        for price in (center - step, center, center + step):
            if price > 0 and abs(price - spot) <= max(3.0 * atr, 2.0 * step):
                out.append(_level(price, f"round_number_{step:g}", 0.34, "both"))
    return out


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def build_trade_plan(
    *,
    ticker: str,
    direction: Direction | str,
    sources: Sequence[str],
    score: float,
    daily: pd.DataFrame,
    hourly: pd.DataFrame,
    options: OptionsContext | None = None,
    config: IntradayStructureConfig | None = None,
    reference_as_of: str | None = None,
) -> TradePlan | None:
    """One conditional plan: what has to happen, where it dies, where it goes."""
    config = config or IntradayStructureConfig()
    side = Direction(direction).value if not isinstance(direction, str) else Direction(direction).value
    if daily.empty:
        return None
    spot = float(daily.iloc[-1]["close"])
    atr = _atr(daily)
    if not (math.isfinite(spot) and spot > 0 and math.isfinite(atr) and atr > 0):
        return None

    levels = build_levels(daily=daily, hourly=hourly, spot=spot, atr=atr, options=options, config=config)
    regime_policy = premarket_regime_policy(config.regime)
    warnings = [OVERNIGHT_GAP_WARNING]
    if options is None or options.source == "none":
        warnings.append("no_dealer_levels_available")
    elif options.warnings:
        warnings.extend(options.warnings)

    # The trigger is the first structure price has to clear in this direction.
    trigger_level = nearest_level(levels, spot, side)
    regime = classify_context(
        spot=spot, atr=atr,
        features={
            "atr_contraction": _atr_contraction(daily),
            "trend_strength": _trend_strength(daily, atr),
            "distance_to_vwap_atr": 0.0,  # no intraday VWAP before the open
        },
        levels=levels, policy=regime_policy,
    )

    def plan(no_trade: str | None, **kwargs) -> TradePlan:
        base = dict(
            ticker=ticker, direction=side, sources=sorted(set(sources)), score=float(score),
            reference_price=spot, reference_as_of=reference_as_of or "",
            atr=atr, trigger=None, trigger_level_type=None, trigger_level_sources=[],
            trigger_distance_atr=None, invalidation=None, targets=[], target_level_types=[],
            obstacles=[], runway_score=None, reward_risk=None,
            context_regime=regime.regime, regime_evidence=list(regime.evidence),
            no_trade_reason=no_trade, warnings=warnings,
        )
        base.update(kwargs)
        return TradePlan(**base)

    if trigger_level is None:
        return plan("no_trigger_level_beyond_reference_price")

    trigger = float(trigger_level.price)
    trigger_sources = list(trigger_level.metadata.get("sources") or (trigger_level.level_type,))

    # Invalidation sits behind the trigger, at the nearest opposing structure if
    # one is close enough, otherwise at a fixed ATR distance.
    opposing = nearest_level(levels, trigger, _opposite(side))
    fallback = trigger - atr if side == "long" else trigger + atr
    invalidation = float(opposing.price) if opposing is not None else fallback
    risk = (trigger - invalidation) if side == "long" else (invalidation - trigger)
    if risk <= 0 or risk > config.target.max_invalidation_atr * atr:
        invalidation = fallback
        risk = atr

    # The full ladder, walked outward from the trigger, published up front.
    targets: list[float] = []
    target_types: list[str] = []
    obstacles: list[dict[str, Any]] = []
    runway_score: float | None = None
    cursor = trigger
    for rung in range(TARGET_LADDER_DEPTH):
        result = score_runway(
            spot=trigger, direction=side, atr=atr, levels=levels,
            trend_strength=_trend_strength(daily, atr),
            market=MarketContext(datetime.now(timezone.utc), market_alignment_score=0.5),
            options=options or OptionsContext(),
            beyond=None if rung == 0 else cursor,
        )
        if result.next_target is None:
            break
        # A rung beyond the reachability cap is not a destination; publishing it
        # would put a 99-ATR number in a plan a human is meant to act on.
        if abs(float(result.next_target) - trigger) > config.target.max_target_distance_atr * atr:
            if rung == 0:
                runway_score = result.runway_score
                obstacles = [dict(item) for item in result.intermediate_obstacles]
            break
        if rung == 0:
            runway_score = result.runway_score
            obstacles = [dict(item) for item in result.intermediate_obstacles]
        targets.append(float(result.next_target))
        target_types.append(result.target_level_type or "unknown")
        cursor = float(result.next_target)

    if not targets:
        return plan("no_causal_target_beyond_trigger",
                    trigger=trigger, trigger_level_type=trigger_level.level_type,
                    trigger_level_sources=trigger_sources,
                    trigger_distance_atr=abs(trigger - spot) / atr,
                    invalidation=invalidation)

    reward = (targets[0] - trigger) if side == "long" else (trigger - targets[0])
    reward_risk = reward / risk if risk > 0 else 0.0

    no_trade: str | None = None
    if runway_score is not None and runway_score < config.target.min_runway_score:
        no_trade = "runway_below_threshold"
    elif reward_risk < config.target.min_reward_risk:
        no_trade = "reward_risk_below_threshold"

    return plan(
        no_trade, trigger=trigger, trigger_level_type=trigger_level.level_type,
        trigger_level_sources=trigger_sources,
        trigger_distance_atr=abs(trigger - spot) / atr,
        invalidation=invalidation, targets=targets, target_level_types=target_types,
        obstacles=obstacles, runway_score=runway_score, reward_risk=reward_risk,
    )


def _opposite(side: str) -> str:
    return "short" if side == "long" else "long"


def _atr(daily: pd.DataFrame, window: int = 14) -> float:
    subset = daily.tail(window + 1)
    if len(subset) < 2:
        return float(subset.iloc[-1]["high"] - subset.iloc[-1]["low"]) if len(subset) else 0.0
    high = subset["high"].to_numpy(dtype=float)
    low = subset["low"].to_numpy(dtype=float)
    prev_close = subset["close"].shift(1).to_numpy(dtype=float)[1:]
    true_range = np.maximum.reduce([
        high[1:] - low[1:], np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close),
    ])
    return float(np.mean(true_range)) if len(true_range) else 0.0


def _atr_contraction(daily: pd.DataFrame) -> float:
    short = _atr(daily, 14)
    long = _atr(daily, 60)
    return short / long if long > 0 else 1.0


def _trend_strength(daily: pd.DataFrame, atr: float) -> float:
    closes = daily["close"].tail(60)
    if len(closes) < 20 or atr <= 0:
        return 0.0
    fast = float(closes.ewm(span=9, adjust=False).mean().iloc[-1])
    slow = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
    return (fast - slow) / atr


def _level(price: float, kind: str, strength: float, directionality: str) -> StructuralLevel:
    return StructuralLevel(price=float(price), level_type=kind, strength=strength, directionality=directionality)
