"""Causal pattern detection and explicit setup-state transitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from strategies.momentum_scalper.configs.v1 import PatternConfigV1
from strategies.momentum_scalper.contracts import PatternSignal, SetupKind, SetupState, SetupTransition


def _history(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"pattern history missing columns: {missing}")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    return out.sort_values("timestamp").reset_index(drop=True)


def _volume_confirmed(history: pd.DataFrame, multiplier: float) -> bool:
    if len(history) < 2:
        return False
    baseline = float(pd.to_numeric(history.iloc[:-1]["volume"], errors="coerce").tail(5).median())
    return baseline > 0 and float(history.iloc[-1]["volume"]) >= baseline * multiplier


def hod_breakout(history: pd.DataFrame, ticker: str, config: PatternConfigV1) -> PatternSignal | None:
    """Break a high established strictly before the current observation."""

    bars = _history(history)
    if len(bars) < 2:
        return None
    previous = bars.iloc[:-1]
    current = bars.iloc[-1]
    trigger = float(previous["high"].max())
    if float(current["high"]) < trigger or float(current["close"]) < trigger:
        return None
    if not _volume_confirmed(bars, config.min_breakout_volume_ratio):
        return None
    support = float(previous.tail(3)["low"].min())
    return PatternSignal(
        ticker=ticker,
        timestamp=current["timestamp"].to_pydatetime(),
        kind=SetupKind.HOD_BREAKOUT,
        trigger_price=trigger,
        invalidation_price=support,
        reason_codes=("prior_high_break", "volume_confirmed"),
        source_levels={"prior_high": trigger, "support": support},
    )


def first_pullback(history: pd.DataFrame, ticker: str, config: PatternConfigV1) -> PatternSignal | None:
    """Detect the first causal pullback/reclaim after a prior impulse high."""

    bars = _history(history)
    if len(bars) < 5:
        return None
    current = bars.iloc[-1]
    pre_current = bars.iloc[:-1]
    # The peak must be before at least one completed pullback bar, never current.
    peak_idx = int(pre_current.iloc[:-1]["high"].astype(float).to_numpy().argmax())
    if peak_idx == 0 or peak_idx >= len(pre_current) - 1:
        return None
    impulse_start = float(pre_current.iloc[: peak_idx + 1]["low"].min())
    peak = float(pre_current.iloc[peak_idx]["high"])
    if peak <= impulse_start:
        return None
    pullback = pre_current.iloc[peak_idx + 1 :]
    low = float(pullback["low"].min())
    retracement = (peak - low) / (peak - impulse_start) * 100.0
    reclaim = float(pullback["high"].max())
    if retracement > config.max_pullback_retracement_pct:
        return None
    if not (float(current["close"]) >= reclaim and float(current["close"]) > float(current["open"])):
        return None
    if not _volume_confirmed(bars, config.min_breakout_volume_ratio):
        return None
    return PatternSignal(
        ticker=ticker,
        timestamp=current["timestamp"].to_pydatetime(),
        kind=SetupKind.FIRST_PULLBACK,
        trigger_price=reclaim,
        invalidation_price=low,
        reason_codes=("first_pullback_reclaim", "volume_confirmed"),
        source_levels={"impulse_peak": peak, "pullback_low": low, "reclaim": reclaim},
    )


def bull_flag(history: pd.DataFrame, ticker: str, config: PatternConfigV1) -> PatternSignal | None:
    """Detect a pole, controlled flag, and break above a prior-only flag high."""

    bars = _history(history)
    needed = config.min_flag_bars + 3
    if len(bars) < needed:
        return None
    current = bars.iloc[-1]
    flag = bars.iloc[-(config.min_flag_bars + 1) : -1]
    pole = bars.iloc[: -(config.min_flag_bars + 1)]
    pole_low = float(pole["low"].min())
    pole_high = float(pole["high"].max())
    if pole_high <= pole_low:
        return None
    flag_low = float(flag["low"].min())
    retracement = (pole_high - flag_low) / (pole_high - pole_low) * 100.0
    trigger = float(flag["high"].max())
    if retracement > config.max_flag_retracement_pct:
        return None
    if not (float(current["high"]) >= trigger and float(current["close"]) >= trigger):
        return None
    if not _volume_confirmed(bars, config.min_breakout_volume_ratio):
        return None
    return PatternSignal(
        ticker=ticker,
        timestamp=current["timestamp"].to_pydatetime(),
        kind=SetupKind.BULL_FLAG,
        trigger_price=trigger,
        invalidation_price=flag_low,
        reason_codes=("flag_break", "controlled_retracement", "volume_confirmed"),
        source_levels={"pole_high": pole_high, "flag_high": trigger, "flag_low": flag_low},
    )


def flat_top(history: pd.DataFrame, ticker: str, config: PatternConfigV1) -> PatternSignal | None:
    """Detect repeated prior resistance and a new breakout bar."""

    bars = _history(history)
    if len(bars) < 5:
        return None
    current = bars.iloc[-1]
    top = bars.iloc[-4:-1]
    highs = pd.to_numeric(top["high"], errors="coerce")
    level = float(highs.max())
    tolerance = level * config.flat_top_tolerance_pct / 100.0
    if level <= 0 or float(highs.max() - highs.min()) > tolerance:
        return None
    if not (float(current["high"]) >= level and float(current["close"]) >= level):
        return None
    if not _volume_confirmed(bars, config.min_breakout_volume_ratio):
        return None
    support = float(top["low"].min())
    return PatternSignal(
        ticker=ticker,
        timestamp=current["timestamp"].to_pydatetime(),
        kind=SetupKind.FLAT_TOP,
        trigger_price=level,
        invalidation_price=support,
        reason_codes=("prior_flat_top_break", "volume_confirmed"),
        source_levels={"flat_top": level, "support": support},
    )


DETECTORS = (hod_breakout, first_pullback, bull_flag, flat_top)


@dataclass
class PatternStateMachine:
    """Small explicit state machine retaining no future-derived levels."""

    ticker: str
    config: PatternConfigV1 = PatternConfigV1()
    states: dict[SetupKind, SetupState] = field(default_factory=lambda: {kind: SetupState.DISCOVERED for kind in SetupKind})

    def observe(
        self,
        history: pd.DataFrame,
        *,
        halted: bool = False,
        data_stale: bool = False,
        expired: bool = False,
    ) -> tuple[list[PatternSignal], list[SetupTransition]]:
        bars = _history(history)
        if bars.empty:
            return [], []
        timestamp = bars.iloc[-1]["timestamp"].to_pydatetime()
        forced = SetupState.HALTED if halted else SetupState.DATA_STALE if data_stale else SetupState.EXPIRED if expired else None
        transitions: list[SetupTransition] = []
        signals: list[PatternSignal] = []
        for detector in DETECTORS:
            kind = {hod_breakout: SetupKind.HOD_BREAKOUT, first_pullback: SetupKind.FIRST_PULLBACK, bull_flag: SetupKind.BULL_FLAG, flat_top: SetupKind.FLAT_TOP}[detector]
            previous = self.states[kind]
            if forced is not None:
                current = forced
                signal = None
                reason = forced.value.lower()
            else:
                signal = detector(bars, self.ticker, self.config)
                current = SetupState.TRIGGERED if signal is not None else (SetupState.ARMED if len(bars) >= 2 else SetupState.DISCOVERED)
                reason = "trigger" if signal is not None else "awaiting_causal_trigger"
            if current != previous:
                transitions.append(SetupTransition(self.ticker, timestamp, kind, previous, current, reason))
                self.states[kind] = current
            if signal is not None:
                signals.append(signal)
        return signals, transitions


__all__ = [
    "DETECTORS", "PatternStateMachine", "bull_flag", "first_pullback", "flat_top", "hod_breakout",
]
