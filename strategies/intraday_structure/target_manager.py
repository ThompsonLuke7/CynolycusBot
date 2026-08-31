from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo

from strategies.intraday_structure.detectors.base import DetectionContext, DetectionDecision, is_long
from strategies.intraday_structure.models import SetupRecord, SetupState
from strategies.intraday_structure.runway import RunwayResult, score_runway


ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class TargetPlan:
    invalidation: float
    targets: tuple[float, ...]
    runway: RunwayResult
    reward_risk: float


@dataclass(frozen=True)
class TargetPlanOutcome:
    """A plan, or the specific reason there is none.

    The two failures used to collapse into one ``risk_or_target_plan_unavailable``
    warning, which made 1,472 of 1,474 refusals in a baseline run indistinguishable
    -- "no plan" tells you nothing about whether the stop was too wide or there was
    simply nowhere to go.
    """

    plan: TargetPlan | None = None
    reason: str | None = None


#: Structure exists, but the stop it implies is wider than the risk budget.
INVALIDATION_TOO_WIDE = "invalidation_wider_than_max_atr"
#: No causal level lies beyond spot in the trade's direction; nothing to aim at.
NO_CAUSAL_TARGET = "no_causal_target_beyond_spot"


def build_target_plan(setup: SetupRecord, ctx: DetectionContext) -> TargetPlanOutcome:
    f = ctx.features
    atr = f.get("atr")
    invalidation = setup.invalidation
    if invalidation is None:
        invalidation = ctx.bar.close - atr if is_long(setup) else ctx.bar.close + atr
    risk = (ctx.bar.close - invalidation) if is_long(setup) else (invalidation - ctx.bar.close)
    if risk <= 0:
        invalidation = ctx.bar.close - atr if is_long(setup) else ctx.bar.close + atr
        risk = atr
    minimum_risk = 0.25 * atr
    if risk < minimum_risk:
        invalidation = ctx.bar.close - minimum_risk if is_long(setup) else ctx.bar.close + minimum_risk
        risk = minimum_risk
    if risk > ctx.config.target.max_invalidation_atr * atr:
        return TargetPlanOutcome(reason=INVALIDATION_TOO_WIDE)
    runway = score_runway(
        spot=ctx.bar.close, direction=setup.direction.value, atr=atr, levels=ctx.levels,
        trend_strength=f.get("trend_strength"), market=ctx.market, options=ctx.options,
    )
    if runway.next_target is None:
        return TargetPlanOutcome(reason=NO_CAUSAL_TARGET)
    reward = (runway.next_target - ctx.bar.close) if is_long(setup) else (ctx.bar.close - runway.next_target)
    rr = reward / risk if risk > 0 else 0.0
    return TargetPlanOutcome(TargetPlan(float(invalidation), (float(runway.next_target),), runway, float(rr)))


def manage_running_setup(setup: SetupRecord, ctx: DetectionContext) -> DetectionDecision:
    policy = ctx.config.target
    bar = ctx.bar
    if policy.close_at_session_end and bar.timestamp.astimezone(ET).time() >= time(15, 59):
        return DetectionDecision(SetupState.CLOSED, "END_OF_DAY", "configured end-of-day close", ("end_of_day",))
    if setup.entry_price is not None:
        favorable = (bar.high - setup.entry_price) if is_long(setup) else (setup.entry_price - bar.low)
        adverse = (setup.entry_price - bar.low) if is_long(setup) else (bar.high - setup.entry_price)
        setup.max_favorable_excursion = max(setup.max_favorable_excursion, favorable)
        setup.max_adverse_excursion = max(setup.max_adverse_excursion, adverse)
    if setup.invalidation is not None:
        invalid = bar.low <= setup.invalidation if is_long(setup) else bar.high >= setup.invalidation
        if invalid:
            return DetectionDecision(SetupState.INVALIDATED, "INVALIDATED", "invalidation touched", ("invalidation_touched",))
    target = setup.active_target
    if target is not None:
        reached = bar.high >= target if is_long(setup) else bar.low <= target
        if reached:
            return DetectionDecision(
                SetupState.TARGET_REACHED, "TARGET_REACHED", "active target reached",
                (f"target_{setup.active_target_index + 1}_reached",),
                metadata={"target_reached_at": bar.timestamp.isoformat()},
            )
    if setup.bars_alive >= policy.max_setup_bars or (setup.entry_time and setup.bars_in_state >= policy.time_exit_bars):
        return DetectionDecision(SetupState.CLOSED, "TIME_EXIT", "maximum setup duration reached", ("time_exit",))
    _tighten_structure_stop(setup, ctx)
    return DetectionDecision()


def evaluate_extension(setup: SetupRecord, ctx: DetectionContext) -> DetectionDecision:
    policy = ctx.config.target
    if setup.extensions >= policy.max_extensions:
        return DetectionDecision(SetupState.EXHAUSTED, "TARGET_COMPLETE", "maximum target extensions reached", ("extension_limit",))
    current_target = setup.active_target
    runway = score_runway(
        spot=ctx.bar.close, direction=setup.direction.value, atr=ctx.features.get("atr"),
        levels=ctx.levels, trend_strength=ctx.features.get("trend_strength"),
        market=ctx.market, options=ctx.options, beyond=current_target,
    )
    supportive_volume = ctx.features.get("relative_volume_5m") >= 0.85
    supportive_trend = ctx.features.get("trend_strength") > 0 if is_long(setup) else ctx.features.get("trend_strength") < 0
    if runway.next_target is not None and runway.runway_score >= policy.extension_min_runway and supportive_volume and supportive_trend:
        setup.targets.append(float(runway.next_target))
        setup.active_target_index = len(setup.targets) - 1
        setup.extensions += 1
        setup.runway_score = runway.runway_score
        if policy.move_stop_to_entry_after_target and setup.entry_price is not None:
            setup.invalidation = max(setup.invalidation or setup.entry_price, setup.entry_price) if is_long(setup) else min(setup.invalidation or setup.entry_price, setup.entry_price)
        return DetectionDecision(
            SetupState.EXTENDED, "TARGET_EXTENDED", "structure supports target extension",
            ("target_extended", "trend_intact", "runway_clear"),
            metadata={"runway_components": runway.components},
        )
    failures = int(setup.metadata.get("target_failure_count", 0)) + 1
    if failures >= ctx.config.detector.max_failed_breaks:
        return DetectionDecision(SetupState.EXHAUSTED, "EXTENSION_FAILED", "target extension conditions failed", ("failed_target_extension",), metadata={"target_failure_count": failures})
    return DetectionDecision(phase="TARGET_REACHED", reason="waiting for extension evidence", evidence=("target_reached",), metadata={"target_failure_count": failures})


def _tighten_structure_stop(setup: SetupRecord, ctx: DetectionContext) -> None:
    atr_pad = ctx.config.target.trailing_structure_atr * ctx.features.get("atr")
    if is_long(setup):
        proposed = ctx.features.get("micro_swing_low") - atr_pad
        if setup.invalidation is None or proposed > setup.invalidation:
            setup.invalidation = proposed
    else:
        proposed = ctx.features.get("micro_swing_high") + atr_pad
        if setup.invalidation is None or proposed < setup.invalidation:
            setup.invalidation = proposed
