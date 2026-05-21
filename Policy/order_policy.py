from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Callable

import math
import pandas as pd
import re
import time as time_mod
from zoneinfo import ZoneInfo

from API.Alpaca_API.options.options_api import AlpacaOptionsClient


PHASE4_SWING_SETUP_BODYCLOSE_BODYCLOSE_V1 = "phase4_swing_setup_bodyclose_bodyclose_v1"


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _parse_hhmm(hhmm: str) -> time:
    parts = (hhmm or "").strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM time: {hhmm}")
    return time(hour=int(parts[0]), minute=int(parts[1]))


def _next_business_day(d: date, *, n: int = 1) -> date:
    out = d
    for _ in range(max(0, int(n))):
        out = out + timedelta(days=1)
        while out.weekday() >= 5:
            out = out + timedelta(days=1)
    return out


@dataclass(frozen=True)
class OptionOrderPolicyConfig:
    underlying: str = "SPY"
    env_file: str = ".env"
    tz_name: str = "America/New_York"
    atr_length: int = 14
    atr_multiplier: float = 1.0
    dte_cutoff_hhmm: str = "13:00"
    qty: int = 1
    order_type: str = "market"
    time_in_force: str = "day"
    close_on_flat: bool = True
    close_on_flip: bool = True
    same_side_reentry_grace_bars: int = 1
    opposite_confirm_bars: int = 2
    opposite_min_abs_action: float = 0.10
    opposite_min_prob_edge: float = 0.05
    submit_orders: bool = True
    long_options_only: bool = True
    verify_submitted_orders: bool = True
    verify_timeout_sec: float = 2.0
    verify_final_attempt_grace_sec: float = 600.0
    verify_poll_sec: float = 0.5
    broker_reconcile_max_age_sec: float = 60.0
    resubmit_on_terminal_fail: bool = True
    max_resubmit_attempts: int = 4
    action_deadband: float = 0.05
    ema_alpha: float = 0.85
    rebalance_deadband: float = 0.10
    max_step_contracts: int = 2
    price_mode: str = "mid"  # ask|mid|bid|last|mark
    max_contracts_fallback: int = 1
    max_contracts_cap: int = 0  # <=0 disables hard cap
    meta_trailing_stop_enabled: bool = True
    meta_trail_activate_atr: float = 0.75
    meta_trail_atr: float = 0.8
    meta_trail_atr_after_tp: float = 0.5
    meta_trail_tp_atr: float = 1.0
    meta_use_tp_to_tighten_trail: bool = True
    meta_hard_stop_atr: float = 0.0
    meta_no_chase_atr: float = 1.5
    meta_same_side_reentry_cooldown_bars: int = 10
    meta_stale_no_progress_minutes: int = 20
    meta_stale_no_progress_atr: float = 0.35
    meta_stale_after_favorable_minutes: int = 30
    meta_stale_retrace_atr: float = 0.25
    meta_setup_failure_exit_enabled: bool = True
    meta_setup_failure_buffer_atr: float = 0.10
    meta_no_progress_exit_enabled: bool = False
    meta_no_progress_exit_minutes: int = 10
    meta_no_progress_exit_atr: float = 0.20
    meta_execute_on_interval_close: bool = False
    meta_intrabar_execution_enabled: bool = True
    meta_intrabar_breakout_entry_only: bool = False
    meta_intrabar_entry_policy: str = "legacy_breakout_touch"
    meta_intrabar_setup_max_bars: int = 3
    meta_intrabar_setup_bar_minutes: int = 10
    meta_intrabar_max_confirmation_age_minutes: int = 30
    meta_intrabar_ref_chase_atr: float = 0.50
    meta_intrabar_long_setup_threshold: float | None = None
    meta_intrabar_short_setup_threshold: float | None = None
    meta_intrabar_opposite_dominance_delta: float = 0.0
    meta_opening_pullback_enabled: bool = True
    meta_opening_pullback_setup_hhmm: str = "09:30"
    meta_opening_pullback_until_hhmm: str = "10:00"
    meta_opening_pullback_stretch_atr: float = 0.35
    meta_opening_pullback_ema_length: int = 8
    scalp_mode_enabled: bool = False
    scalp_long_setup_threshold: float = 0.30
    scalp_short_setup_threshold: float = 0.55
    scalp_setup_max_bars: int = 1
    scalp_min_signal_range_atr: float = 0.35
    scalp_require_reversal_close: bool = True
    meta_countertrend_veto_enabled: bool = False
    meta_countertrend_min_prob: float = 0.65
    meta_neutral_long_min_prob: float = 0.50
    meta_neutral_short_min_prob: float = 0.75
    meta_continuation_entry_enabled: bool = False
    meta_continuation_source_short_threshold: float = 0.15
    meta_continuation_source_long_threshold: float = 0.42
    meta_continuation_pullback_atr: float = 0.75
    meta_replay_compatible_mode: bool = True
    meta_min_hold_bars: int = 2
    meta_exit_entry_delta: float = 0.15
    meta_soft_exit_confirm_bars: int = 2
    meta_urgent_exit_prob: float = 0.85
    meta_urgent_exit_delta: float = 0.30
    meta_profit_protect_enabled: bool = False
    meta_profit_protect_arm_atr: float = 2.0
    meta_profit_protect_giveback_atr_long: float = 0.75
    meta_profit_protect_giveback_atr_short: float = 1.0
    option_exit_policy: str = "option_adaptive_trail_v1"
    option_exit_take_profit_pct: float = 0.0
    option_exit_stop_loss_pct: float = 1.0
    option_exit_profit_lock_arm_pct: float = 2.0
    option_exit_profit_lock_floor_pct: float = 0.25
    option_exit_trailing_arm_pct: float = 1.0
    option_exit_trailing_giveback_pct: float = 0.20
    option_exit_no_progress_minutes: int = 0
    option_exit_no_progress_mfe_pct: float = 0.0
    option_exit_time_decay_minutes: int = 60
    option_exit_time_decay_progress_pct: float = 0.5
    option_exit_opposite_prob: float = 0.40
    option_exit_opposite_prob_long: float | None = 0.70
    option_exit_opposite_prob_short: float | None = 0.75
    option_exit_opposite_profit_pct: float = 0.25
    option_exit_quote_mode: str = "bid"
    option_new_entry_cutoff_hhmm: str | None = "15:00"
    max_live_entry_lag_sec: float = 0.0
    use_wall_clock_entry_cutoff: bool = False
    expiring_position_exit_hhmm: str | None = "15:40"
    auto_flatten_underlying_shares: bool = True


@dataclass
class _MetaTrailState:
    active: bool = False
    entry_price: float = float("nan")
    entry_atr: float = float("nan")
    favorable_anchor: float = float("nan")
    tp_seen: bool = False
    entry_ts: datetime | None = None
    last_favorable_ts: datetime | None = None


class OptionOrderPolicy:
    """
    Maps agent direction actions in [-1, 0, 1] to option orders:
      - long  -> buy call
      - short -> buy put
      - flat  -> optional sell-to-close
    Direction-only execution:
      - convert action to signed direction via deadband
      - ignore action magnitude for order sizing
      - size by fixed qty (bounded by max-contract checks)
      - trade only signed delta with per-step cap
    """

    def __init__(self, config: OptionOrderPolicyConfig) -> None:
        self.cfg = config
        self._tz = ZoneInfo(config.tz_name)
        self._cutoff = _parse_hhmm(config.dte_cutoff_hhmm)
        self._expiring_position_exit_cutoff = (
            _parse_hhmm(config.expiring_position_exit_hhmm)
            if str(config.expiring_position_exit_hhmm or "").strip()
            else None
        )
        self._new_entry_cutoff = (
            _parse_hhmm(config.option_new_entry_cutoff_hhmm)
            if str(config.option_new_entry_cutoff_hhmm or "").strip()
            else None
        )
        self._client = AlpacaOptionsClient(env_file=config.env_file)

        self._long_contracts = 0
        self._short_contracts = 0
        self._long_symbol: str | None = None
        self._short_symbol: str | None = None
        self._long_avg_entry_price: float = float("nan")
        self._short_avg_entry_price: float = float("nan")
        self._option_exit_best_price: dict[str, float] = {"long": float("nan"), "short": float("nan")}
        self._option_exit_best_seen_at: dict[str, datetime | None] = {"long": None, "short": None}
        self._pos = 0
        self._signed_contracts = 0
        self._open_symbol: str | None = None
        self._bars_interval: list[tuple[float, float, float]] = []
        self._action_ema: float | None = None
        self._action_effective: float = 0.0
        self._pending_flat_side: int = 0
        self._pending_flat_bars: int = 0
        self._pending_opposite_side: int = 0
        self._pending_opposite_bars: int = 0
        self._meta_long_trail = _MetaTrailState()
        self._meta_short_trail = _MetaTrailState()
        self._last_1m_close: float = float("nan")
        self._prev_1m_close: float = float("nan")
        self._recent_1m_closes: deque[float] = deque(maxlen=30)
        self._latest_meta_side_snapshot: dict[str, float] | None = None
        self._meta_side_entry_armed: dict[str, bool] = {"long": True, "short": True}
        self._meta_side_enter_above_threshold: dict[str, bool] = {"long": False, "short": False}
        self._meta_side_enter_peak_prob: dict[str, float] = {"long": float("nan"), "short": float("nan")}
        self._meta_rearm_session_day: date | None = None
        self._meta_side_cooldown_until: dict[str, datetime | None] = {"long": None, "short": None}
        self._meta_side_bars_held: dict[str, int] = {"long": -1, "short": -1}
        self._meta_side_entry_signal_ts: dict[str, datetime | None] = {"long": None, "short": None}
        self._meta_side_soft_exit_count: dict[str, int] = {"long": 0, "short": 0}
        self._meta_intrabar_entry_intent_active: dict[str, bool] = {"long": False, "short": False}
        self._meta_intrabar_entry_ref: dict[str, float] = {"long": float("nan"), "short": float("nan")}
        self._meta_intrabar_entry_setup_ts: dict[str, datetime | None] = {"long": None, "short": None}
        self._meta_intrabar_entry_expires_at: dict[str, datetime | None] = {"long": None, "short": None}
        self._meta_intrabar_setup_snapshot: dict[str, dict[str, Any] | None] = {"long": None, "short": None}
        self._opening_pullback_required: dict[str, bool] = {"long": False, "short": False}
        self._opening_pullback_seen: dict[str, bool] = {"long": False, "short": False}
        self._meta_entry_structure: dict[str, dict[str, Any] | None] = {"long": None, "short": None}
        self._meta_side_reason: dict[str, str | None] = {"long": None, "short": None}
        self._pending_broker_reconcile: dict[str, Any] | None = None
        self._last_broker_reconcile_monotonic: float = 0.0
        self._entry_lockout_until: datetime | None = None
        self._entry_lockout_reason: str | None = None
        self._contract_price_provider: Callable[..., float] | None = None

    def set_contract_price_provider(self, provider: Callable[..., float] | None) -> None:
        """
        Install an external contract pricer for simulated/replay runs.

        Live trading keeps using Alpaca quotes. Replay can inject a deterministic
        proxy so simulated entries and software exits are priced from historical
        underlying bars instead of a hardcoded fill.
        """
        self._contract_price_provider = provider

    def _trail_state(self, side: str) -> _MetaTrailState:
        return self._meta_long_trail if side == "long" else self._meta_short_trail

    def _reset_trail_state(self, side: str) -> None:
        state = self._trail_state(side)
        state.active = False
        state.entry_price = float("nan")
        state.entry_atr = float("nan")
        state.favorable_anchor = float("nan")
        state.tp_seen = False
        state.entry_ts = None
        state.last_favorable_ts = None

    def _seed_trail_state(
        self,
        *,
        side: str,
        close: float,
        atr: float,
        high: float,
        low: float,
        local_ts: datetime | None = None,
    ) -> None:
        state = self._trail_state(side)
        state.active = True
        state.entry_price = float(close) if math.isfinite(close) else float("nan")
        state.entry_atr = float(atr) if math.isfinite(atr) and atr > 0.0 else float("nan")
        if side == "long":
            anchor = high if math.isfinite(high) else close
        else:
            anchor = low if math.isfinite(low) else close
        state.favorable_anchor = float(anchor) if math.isfinite(anchor) else float(close)
        state.tp_seen = False
        state.entry_ts = local_ts
        state.last_favorable_ts = local_ts

    def _update_trail_state(
        self,
        *,
        side: str,
        close: float,
        high: float,
        low: float,
        local_ts: datetime | None = None,
    ) -> None:
        state = self._trail_state(side)
        if not state.active:
            return
        if side == "long":
            if math.isfinite(high):
                prev_anchor = float(state.favorable_anchor)
                next_anchor = max(prev_anchor, float(high))
                if next_anchor > prev_anchor:
                    state.last_favorable_ts = local_ts
                state.favorable_anchor = next_anchor
            elif math.isfinite(close):
                prev_anchor = float(state.favorable_anchor)
                next_anchor = max(prev_anchor, float(close))
                if next_anchor > prev_anchor:
                    state.last_favorable_ts = local_ts
                state.favorable_anchor = next_anchor
            if (
                not state.tp_seen
                and math.isfinite(state.entry_price)
                and math.isfinite(state.entry_atr)
                and state.entry_atr > 0.0
                and float(self.cfg.meta_trail_tp_atr) > 0.0
                and math.isfinite(high)
                and high >= state.entry_price + float(self.cfg.meta_trail_tp_atr) * state.entry_atr
            ):
                state.tp_seen = True
            return

        if math.isfinite(low):
            prev_anchor = float(state.favorable_anchor)
            next_anchor = min(prev_anchor, float(low))
            if next_anchor < prev_anchor:
                state.last_favorable_ts = local_ts
            state.favorable_anchor = next_anchor
        elif math.isfinite(close):
            prev_anchor = float(state.favorable_anchor)
            next_anchor = min(prev_anchor, float(close))
            if next_anchor < prev_anchor:
                state.last_favorable_ts = local_ts
            state.favorable_anchor = next_anchor
        if (
            not state.tp_seen
            and math.isfinite(state.entry_price)
            and math.isfinite(state.entry_atr)
            and state.entry_atr > 0.0
            and float(self.cfg.meta_trail_tp_atr) > 0.0
            and math.isfinite(low)
            and low <= state.entry_price - float(self.cfg.meta_trail_tp_atr) * state.entry_atr
        ):
            state.tp_seen = True

    def _trail_stop_hit(
        self,
        *,
        side: str,
        close: float,
        high: float,
        low: float,
    ) -> bool:
        if not bool(self.cfg.meta_trailing_stop_enabled):
            return False
        state = self._trail_state(side)
        if (
            (not state.active)
            or (not math.isfinite(state.entry_price))
            or (not math.isfinite(state.entry_atr))
            or state.entry_atr <= 0.0
            or (not math.isfinite(state.favorable_anchor))
        ):
            return False

        activate_mult = max(0.0, float(self.cfg.meta_trail_activate_atr))
        trail_mult = float(self.cfg.meta_trail_atr_after_tp) if (
            bool(self.cfg.meta_use_tp_to_tighten_trail) and bool(state.tp_seen)
        ) else float(self.cfg.meta_trail_atr)
        if trail_mult <= 0.0:
            trail_mult = float(self.cfg.meta_trail_atr)
        if trail_mult <= 0.0:
            return False

        trail_active = False
        if side == "long":
            move = float(state.favorable_anchor) - float(state.entry_price)
            if move >= activate_mult * float(state.entry_atr):
                trail_active = True
            if bool(self.cfg.meta_use_tp_to_tighten_trail) and bool(state.tp_seen):
                trail_active = True
            if not trail_active:
                return False
            trail_level = float(state.favorable_anchor) - trail_mult * float(state.entry_atr)
            return (
                (math.isfinite(low) and low <= trail_level)
                or (math.isfinite(close) and close <= trail_level)
            )

        move = float(state.entry_price) - float(state.favorable_anchor)
        if move >= activate_mult * float(state.entry_atr):
            trail_active = True
        if bool(self.cfg.meta_use_tp_to_tighten_trail) and bool(state.tp_seen):
            trail_active = True
        if not trail_active:
            return False
        trail_level = float(state.favorable_anchor) + trail_mult * float(state.entry_atr)
        return (
            (math.isfinite(high) and high >= trail_level)
            or (math.isfinite(close) and close >= trail_level)
        )

    def _stale_trade_exit_hit(
        self,
        *,
        side: str,
        close: float,
        local_ts: datetime | None,
    ) -> bool:
        state = self._trail_state(side)
        if (
            local_ts is None
            or (not state.active)
            or state.entry_ts is None
            or state.last_favorable_ts is None
            or (not math.isfinite(state.entry_price))
            or (not math.isfinite(state.entry_atr))
            or state.entry_atr <= 0.0
            or (not math.isfinite(state.favorable_anchor))
            or (not math.isfinite(close))
        ):
            return False

        entry_atr = float(state.entry_atr)
        elapsed_min = max(0.0, (local_ts - state.entry_ts).total_seconds() / 60.0)
        stale_min = max(0.0, (local_ts - state.last_favorable_ts).total_seconds() / 60.0)
        if side == "long":
            favorable_move = float(state.favorable_anchor) - float(state.entry_price)
            retrace_move = float(state.favorable_anchor) - float(close)
        else:
            favorable_move = float(state.entry_price) - float(state.favorable_anchor)
            retrace_move = float(close) - float(state.favorable_anchor)

        no_progress_hit = (
            elapsed_min >= max(0, int(self.cfg.meta_stale_no_progress_minutes))
            and favorable_move < float(self.cfg.meta_stale_no_progress_atr) * entry_atr
        )
        stale_retrace_hit = (
            stale_min >= max(0, int(self.cfg.meta_stale_after_favorable_minutes))
            and retrace_move >= float(self.cfg.meta_stale_retrace_atr) * entry_atr
        )
        return bool(no_progress_hit or stale_retrace_hit)

    def _setup_failure_exit_hit(
        self,
        *,
        side: str,
        close: float,
        high: float,
        low: float,
    ) -> bool:
        if not bool(self.cfg.meta_setup_failure_exit_enabled):
            return False
        side_key = str(side).strip().lower()
        structure = self._meta_entry_structure.get(side_key)
        if not isinstance(structure, dict):
            return False
        signal_atr = _as_float(structure.get("signal_atr"))
        trail_state = self._trail_state(side_key)
        ref_atr = signal_atr if math.isfinite(signal_atr) and signal_atr > 0.0 else _as_float(trail_state.entry_atr)
        if not (math.isfinite(ref_atr) and ref_atr > 0.0):
            return False
        buffer_atr = max(0.0, float(self.cfg.meta_setup_failure_buffer_atr))
        buffer = buffer_atr * ref_atr
        if side_key == "long":
            invalid_level = _as_float(structure.get("signal_low"))
            if not math.isfinite(invalid_level):
                return False
            invalid_level = float(invalid_level) - buffer
            probe = low if math.isfinite(low) else close
            return bool(math.isfinite(probe) and probe <= invalid_level)

        invalid_level = _as_float(structure.get("signal_high"))
        if not math.isfinite(invalid_level):
            return False
        invalid_level = float(invalid_level) + buffer
        probe = high if math.isfinite(high) else close
        return bool(math.isfinite(probe) and probe >= invalid_level)

    def _no_progress_exit_hit(
        self,
        *,
        side: str,
        close: float,
        local_ts: datetime | None,
    ) -> bool:
        if not bool(self.cfg.meta_no_progress_exit_enabled):
            return False
        state = self._trail_state(side)
        if (
            local_ts is None
            or (not state.active)
            or state.entry_ts is None
            or (not math.isfinite(state.entry_price))
            or (not math.isfinite(state.entry_atr))
            or state.entry_atr <= 0.0
            or (not math.isfinite(state.favorable_anchor))
            or (not math.isfinite(close))
        ):
            return False
        elapsed_min = max(0.0, (local_ts - state.entry_ts).total_seconds() / 60.0)
        if elapsed_min < max(0, int(self.cfg.meta_no_progress_exit_minutes)):
            return False
        threshold_atr = max(0.0, float(self.cfg.meta_no_progress_exit_atr))
        if side == "long":
            favorable_move = float(state.favorable_anchor) - float(state.entry_price)
            current_move = float(close) - float(state.entry_price)
        else:
            favorable_move = float(state.entry_price) - float(state.favorable_anchor)
            current_move = float(state.entry_price) - float(close)
        return bool(
            favorable_move < threshold_atr * float(state.entry_atr)
            and current_move <= 0.0
        )

    def _hard_stop_hit(
        self,
        *,
        side: str,
        close: float,
        high: float,
        low: float,
    ) -> bool:
        state = self._trail_state(side)
        stop_mult = float(self.cfg.meta_hard_stop_atr)
        if (
            stop_mult <= 0.0
            or (not state.active)
            or (not math.isfinite(state.entry_price))
            or (not math.isfinite(state.entry_atr))
            or state.entry_atr <= 0.0
        ):
            return False

        stop_dist = stop_mult * float(state.entry_atr)
        if side == "long":
            stop_level = float(state.entry_price) - stop_dist
            probe = low if math.isfinite(low) else close
            return math.isfinite(probe) and probe <= stop_level

        stop_level = float(state.entry_price) + stop_dist
        probe = high if math.isfinite(high) else close
        return math.isfinite(probe) and probe >= stop_level

    def _ohlc_exit_bar_is_after_entry(self, *, side: str, local_ts: datetime | None) -> bool:
        state = self._trail_state(side)
        return local_ts is None or state.entry_ts is None or local_ts > state.entry_ts

    def _profit_protect_exit_hit(
        self,
        *,
        side: str,
        close: float,
    ) -> bool:
        if not bool(self.cfg.meta_profit_protect_enabled):
            return False
        state = self._trail_state(side)
        if (
            (not state.active)
            or (not math.isfinite(state.entry_price))
            or (not math.isfinite(state.entry_atr))
            or state.entry_atr <= 0.0
            or (not math.isfinite(state.favorable_anchor))
            or (not math.isfinite(close))
        ):
            return False

        arm_atr = float(self.cfg.meta_profit_protect_arm_atr)
        if arm_atr <= 0.0:
            return False
        giveback_atr = (
            float(self.cfg.meta_profit_protect_giveback_atr_long)
            if side == "long"
            else float(self.cfg.meta_profit_protect_giveback_atr_short)
        )
        if giveback_atr <= 0.0:
            return False

        entry = float(state.entry_price)
        atr = float(state.entry_atr)
        favorable = float(state.favorable_anchor)
        if side == "long":
            mfe_atr = (favorable - entry) / atr
            realized_run_atr = (float(close) - entry) / atr
        else:
            mfe_atr = (entry - favorable) / atr
            realized_run_atr = (entry - float(close)) / atr
        if not (math.isfinite(mfe_atr) and math.isfinite(realized_run_atr)):
            return False
        if mfe_atr < arm_atr:
            return False
        current_giveback_atr = mfe_atr - realized_run_atr
        return bool(math.isfinite(current_giveback_atr) and current_giveback_atr >= giveback_atr)

    def _option_value_exit_hit(
        self,
        *,
        side: str,
        logger: Callable[[str], None],
        closed_bar: dict[str, Any] | None = None,
        local_ts: datetime | None = None,
    ) -> bool:
        policy = str(self.cfg.option_exit_policy or "meta").strip().lower()
        if policy not in {"option_value_bracket_v1", "software_oco_v1", "option_adaptive_trail_v1"}:
            return False
        side_key = str(side).strip().lower()
        symbol = self._long_symbol if side_key == "long" else self._short_symbol
        entry_price = self._long_avg_entry_price if side_key == "long" else self._short_avg_entry_price
        if not symbol or not (math.isfinite(entry_price) and entry_price > 0.0):
            return False
        current_price = self._get_contract_price(
            symbol=symbol,
            logger=logger,
            mode=str(self.cfg.option_exit_quote_mode or "bid"),
        )
        if not (math.isfinite(current_price) and current_price >= 0.0):
            return False

        prior_best = _as_float(self._option_exit_best_price.get(side_key))
        best_price = max(prior_best if math.isfinite(prior_best) else entry_price, float(current_price))
        if best_price > (prior_best if math.isfinite(prior_best) else entry_price):
            self._option_exit_best_seen_at[side_key] = local_ts
        elif self._option_exit_best_seen_at.get(side_key) is None:
            self._option_exit_best_seen_at[side_key] = local_ts
        self._option_exit_best_price[side_key] = best_price

        if policy == "option_adaptive_trail_v1":
            best_profit_pct = (best_price - entry_price) / entry_price
            current_profit_pct = (float(current_price) - entry_price) / entry_price
            stop_loss_pct = max(0.0, float(self.cfg.option_exit_stop_loss_pct))
            if 0.0 < stop_loss_pct < 1.0 and current_profit_pct <= -stop_loss_pct:
                self._meta_side_reason[side_key] = "option_stop_loss"
                return True

            trail_arm = max(0.0, float(self.cfg.option_exit_trailing_arm_pct))
            trail_giveback = max(0.0, float(self.cfg.option_exit_trailing_giveback_pct))
            peak_profit = max(0.0, best_price - entry_price)
            trail_floor_price = entry_price + peak_profit * (1.0 - min(1.0, trail_giveback))
            if trail_arm > 0.0 and best_profit_pct >= trail_arm and current_price <= trail_floor_price:
                self._meta_side_reason[side_key] = "option_adaptive_trail"
                return True

            no_progress_minutes = max(0, int(self.cfg.option_exit_no_progress_minutes))
            no_progress_mfe = max(0.0, float(self.cfg.option_exit_no_progress_mfe_pct))
            entry_ts = self._option_entry_local_ts(side_key)
            if (
                no_progress_minutes > 0
                and no_progress_mfe > 0.0
                and isinstance(local_ts, datetime)
                and isinstance(entry_ts, datetime)
            ):
                elapsed_minutes = (local_ts - entry_ts).total_seconds() / 60.0
                if elapsed_minutes >= float(no_progress_minutes) and best_profit_pct < no_progress_mfe:
                    self._meta_side_reason[side_key] = "option_no_progress"
                    return True

            stale_minutes = max(0, int(self.cfg.option_exit_time_decay_minutes))
            best_seen_at = self._option_exit_best_seen_at.get(side_key)
            stale_for_minutes = float("nan")
            if stale_minutes > 0 and isinstance(best_seen_at, datetime) and isinstance(local_ts, datetime):
                stale_for_minutes = (local_ts - best_seen_at).total_seconds() / 60.0
            time_decay_progress = max(0.0, float(self.cfg.option_exit_time_decay_progress_pct))
            if (
                math.isfinite(stale_for_minutes)
                and stale_for_minutes >= float(stale_minutes)
                and best_profit_pct < time_decay_progress
                and current_profit_pct <= 0.0
            ):
                self._meta_side_reason[side_key] = "option_time_decay"
                return True

            opp_thr = self._option_exit_opposite_threshold(side_key)
            opp_profit_pct = max(0.0, float(self.cfg.option_exit_opposite_profit_pct))
            bar = closed_bar or {}
            opp_key = "p_enter_short" if side_key == "long" else "p_enter_long"
            opp_prob = _as_float(bar.get(opp_key))
            if (
                opp_thr > 0.0
                and math.isfinite(opp_prob)
                and opp_prob >= opp_thr
                and best_profit_pct >= opp_profit_pct
            ):
                self._meta_side_reason[side_key] = "option_opposite_signal"
                return True
            return False

        tp_mult = 1.0 + max(0.0, float(self.cfg.option_exit_take_profit_pct))
        stop_mult = max(0.0, 1.0 - max(0.0, float(self.cfg.option_exit_stop_loss_pct)))
        arm_mult = 1.0 + max(0.0, float(self.cfg.option_exit_profit_lock_arm_pct))
        floor_mult = 1.0 + max(0.0, float(self.cfg.option_exit_profit_lock_floor_pct))

        if tp_mult > 1.0 and current_price >= entry_price * tp_mult:
            self._meta_side_reason[side_key] = "option_take_profit"
            return True
        if stop_mult > 0.0 and current_price <= entry_price * stop_mult:
            self._meta_side_reason[side_key] = "option_stop_loss"
            return True
        if arm_mult > 1.0 and floor_mult > 1.0 and best_price >= entry_price * arm_mult and current_price <= entry_price * floor_mult:
            self._meta_side_reason[side_key] = "option_profit_lock"
            return True
        return False

    def _option_exit_opposite_threshold(self, side_key: str) -> float:
        side = str(side_key).strip().lower()
        if side == "long" and self.cfg.option_exit_opposite_prob_long is not None:
            return max(0.0, float(self.cfg.option_exit_opposite_prob_long))
        if side == "short" and self.cfg.option_exit_opposite_prob_short is not None:
            return max(0.0, float(self.cfg.option_exit_opposite_prob_short))
        return max(0.0, float(self.cfg.option_exit_opposite_prob))

    def _option_entry_local_ts(self, side_key: str) -> datetime | None:
        structure = self._meta_entry_structure.get(side_key)
        if isinstance(structure, dict) and isinstance(structure.get("entry_ts"), datetime):
            return structure["entry_ts"]
        trail_ts = self._trail_state(side_key).entry_ts
        if isinstance(trail_ts, datetime):
            return trail_ts
        return self._option_exit_best_seen_at.get(side_key)

    def _option_value_exit_policy_enabled(self) -> bool:
        policy = str(self.cfg.option_exit_policy or "meta").strip().lower()
        return policy in {"option_value_bracket_v1", "software_oco_v1", "option_adaptive_trail_v1"}

    def _refresh_legacy_state(self) -> None:
        self._signed_contracts = int(self._long_contracts) - int(self._short_contracts)
        if self._long_contracts > 0 and self._short_contracts <= 0:
            self._pos = 1
            self._open_symbol = self._long_symbol
        elif self._short_contracts > 0 and self._long_contracts <= 0:
            self._pos = -1
            self._open_symbol = self._short_symbol
        elif self._long_contracts <= 0 and self._short_contracts <= 0:
            self._pos = 0
            self._open_symbol = None
        else:
            self._pos = 0
            self._open_symbol = None

    def _to_local_ts(self, ts: Any) -> datetime:
        dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(self._tz)

    def _update_bar_state(self, closed_bar: dict[str, Any]) -> float:
        h = _as_float(closed_bar.get("high"))
        l = _as_float(closed_bar.get("low"))
        c = _as_float(closed_bar.get("close"))
        if not (math.isfinite(h) and math.isfinite(l) and math.isfinite(c)):
            return float("nan")
        self._bars_interval.append((h, l, c))
        max_keep = max(250, self.cfg.atr_length * 6)
        if len(self._bars_interval) > max_keep:
            self._bars_interval = self._bars_interval[-max_keep:]
        return self._compute_atr()

    def on_interval_bar(self, *, closed_bar: dict[str, Any]) -> float:
        """
        Update internal interval bar state (ATR warmup) regardless of whether
        an action is produced on this bar.
        """
        return self._update_bar_state(closed_bar)

    def prefill_1m_bar(self, *, bar: dict[str, Any]) -> None:
        """
        Warm recent 1m close state without generating any decisions.
        """
        close = _as_float(bar.get("close"))
        if not math.isfinite(close):
            return
        if math.isfinite(self._last_1m_close):
            self._prev_1m_close = float(self._last_1m_close)
        self._last_1m_close = float(close)
        self._recent_1m_closes.append(float(close))

    def on_15m_bar(self, *, closed_bar: dict[str, Any]) -> float:
        """
        Backward-compatible alias for older callers. Uses the configured replay/live interval.
        """
        return self.on_interval_bar(closed_bar=closed_bar)

    def _compute_atr(self) -> float:
        n = int(self.cfg.atr_length)
        if n < 1 or len(self._bars_interval) < n + 1:
            return float("nan")

        trs: list[float] = []
        prev_close = self._bars_interval[0][2]
        for high, low, close in self._bars_interval:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(float(tr))
            prev_close = close

        seed = trs[:n]
        if not seed:
            return float("nan")
        atr = float(sum(seed) / n)
        for tr in trs[n:]:
            atr = ((n - 1) * atr + tr) / n
        return atr

    @staticmethod
    def _directional_prob_edge(*, closed_bar: dict[str, Any], desired_pos: int) -> float:
        """
        Compute directional confidence edge from available interval probability columns.
        Positive edge favors the desired side.
        """
        meta_long = _as_float(closed_bar.get("p_enter_long"))
        meta_short = _as_float(closed_bar.get("p_enter_short"))
        if math.isfinite(meta_long) and math.isfinite(meta_short):
            edge = float(meta_long - meta_short)
            return edge if int(desired_pos) > 0 else -edge

        p_long_vals = [
            _as_float(closed_bar.get("p_pivot_long")),
            _as_float(closed_bar.get("p_tb_long")),
        ]
        p_short_vals = [
            _as_float(closed_bar.get("p_pivot_short")),
            _as_float(closed_bar.get("p_tb_short")),
        ]
        p_long = max((v for v in p_long_vals if math.isfinite(v)), default=float("nan"))
        p_short = max((v for v in p_short_vals if math.isfinite(v)), default=float("nan"))
        if not (math.isfinite(p_long) and math.isfinite(p_short)):
            return float("nan")
        edge = float(p_long - p_short)
        return edge if int(desired_pos) > 0 else -edge

    def _resolve_expiration(self, local_ts: datetime) -> date:
        dte = 0 if local_ts.time() < self._cutoff else 1
        session_day = local_ts.date()
        return session_day if dte == 0 else _next_business_day(session_day, n=1)

    def _sim_contract_symbol(
        self,
        *,
        option_type: str,
        expiration: date,
        strike: float,
    ) -> str:
        cp = "C" if str(option_type).strip().lower() == "call" else "P"
        strike_int = int(max(0, round(float(strike))))
        return f".SIM_{self.cfg.underlying}_{expiration.strftime('%y%m%d')}_{cp}_{strike_int}"

    @staticmethod
    def _extract_positions(resp: Any) -> list[dict[str, Any]]:
        if isinstance(resp, list):
            return [x for x in resp if isinstance(x, dict)]
        if isinstance(resp, dict):
            for key in ("positions", "data"):
                value = resp.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]
        return []

    @staticmethod
    def _option_cp(symbol: str) -> str | None:
        """
        Extract option type (C/P) from OCC symbol, e.g. SPY251031P00680000.
        """
        text = symbol.strip().upper()
        # Live Alpaca symbols use OCC format. Replay simulation symbols use
        # .SIM_SPY_YYMMDD_C_STRIKE / .SIM_SPY_YYMMDD_P_STRIKE.
        sim = re.search(r"(?:^|_)([CP])(?:_|$)", text)
        if sim:
            return sim.group(1)
        m = re.match(r"^[A-Z]+(\d{6})([CP])(\d{8})$", text)
        if not m:
            return None
        return m.group(2)

    @staticmethod
    def _option_expiration(symbol: str) -> date | None:
        m = re.match(r"^[A-Z]+(\d{6})([CP])(\d{8})$", symbol.strip().upper())
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1), "%y%m%d").date()
        except ValueError:
            return None

    def _entry_lockout_active(self, *, local_ts: datetime | None) -> bool:
        if self._entry_lockout_until is None or local_ts is None:
            return False
        if local_ts >= self._entry_lockout_until:
            self._entry_lockout_until = None
            self._entry_lockout_reason = None
            return False
        return True

    def _new_entries_allowed(self, *, local_ts: datetime | None) -> bool:
        if local_ts is None:
            return True
        max_lag_sec = max(0.0, float(self.cfg.max_live_entry_lag_sec))
        if max_lag_sec > 0.0:
            now = datetime.now(self._tz)
            if (now - local_ts).total_seconds() > max_lag_sec:
                return False
        cutoff = self._new_entry_cutoff
        if cutoff is None:
            return True
        if bool(self.cfg.use_wall_clock_entry_cutoff) and datetime.now(self._tz).time() >= cutoff:
            return False
        return local_ts.time() < cutoff

    def _set_entry_lockout(self, *, local_ts: datetime, reason: str) -> None:
        end_of_day = datetime.combine(local_ts.date(), time(23, 59), tzinfo=self._tz)
        self._entry_lockout_until = end_of_day
        self._entry_lockout_reason = str(reason)

    def _symbol_expires_today(self, symbol: str | None, *, local_ts: datetime | None) -> bool:
        if not symbol or local_ts is None:
            return False
        expiration = self._option_expiration(symbol)
        return bool(expiration is not None and expiration == local_ts.date())

    def _latest_meta_snapshot_local_ts(self) -> datetime | None:
        snapshot = self._latest_meta_side_snapshot or {}
        raw_ts = snapshot.get("timestamp")
        if raw_ts in (None, ""):
            return None
        try:
            return self._to_local_ts(raw_ts)
        except Exception:
            return None

    def _stale_prior_session_entry_signal(self, *, local_ts: datetime | None) -> bool:
        """
        New entries must be confirmed by a fresh same-day 10m snapshot.
        Cached prior-session snapshots may still inform management of already-open
        positions, but they cannot open a brand-new position on the next session.
        """
        if local_ts is None:
            return False
        snapshot_ts = self._latest_meta_snapshot_local_ts()
        if snapshot_ts is None:
            return False
        return snapshot_ts.date() != local_ts.date()

    def _exit_waiting_for_next_snapshot(self, *, side: str) -> bool:
        side_key = str(side).strip().lower()
        if side_key not in {"long", "short"}:
            return False
        current_qty = self._long_contracts if side_key == "long" else self._short_contracts
        if current_qty <= 0:
            return False
        entry_signal_ts = self._meta_side_entry_signal_ts.get(side_key)
        latest_snapshot_ts = self._latest_meta_snapshot_local_ts()
        if entry_signal_ts is None or latest_snapshot_ts is None:
            return False
        return latest_snapshot_ts <= entry_signal_ts

    def _read_broker_position_state(self) -> dict[str, Any]:
        if not self.cfg.submit_orders:
            return {
                "underlying": self.cfg.underlying.strip().upper(),
                "long_contracts": int(self._long_contracts),
                "short_contracts": int(self._short_contracts),
                "long_symbol": self._long_symbol,
                "short_symbol": self._short_symbol,
                "avg_entry_price_long": (
                    float(_as_float(self._long_avg_entry_price))
                    if math.isfinite(_as_float(self._long_avg_entry_price))
                    else None
                ),
                "avg_entry_price_short": (
                    float(_as_float(self._short_avg_entry_price))
                    if math.isfinite(_as_float(self._short_avg_entry_price))
                    else None
                ),
                "qty_long": float(self._long_contracts),
                "qty_short": float(self._short_contracts),
                "multiple_positions": False,
                "ignored_short_positions": 0,
                "simulated": True,
            }
        resp = self._client.get_positions()
        positions = self._extract_positions(resp)
        under = self.cfg.underlying.strip().upper()

        long_candidates: list[tuple[float, str, float]] = []
        short_candidates: list[tuple[float, str, float]] = []
        underlying_equity_qty = 0.0
        underlying_equity_side: str | None = None
        ignored_short_count = 0
        for p in positions:
            symbol = str(p.get("symbol", "")).strip().upper()
            if symbol == under:
                qty_val = _as_float(p.get("qty"))
                if math.isfinite(qty_val) and abs(qty_val) >= 100.0:
                    side_raw = str(p.get("side", "")).strip().lower()
                    underlying_equity_qty = qty_val
                    underlying_equity_side = side_raw or ("long" if qty_val > 0 else "short")
                continue
            if not symbol.startswith(under):
                continue
            cp = self._option_cp(symbol)
            if cp is None:
                continue

            qty_val = _as_float(p.get("qty"))
            side_raw = str(p.get("side", "")).strip().lower()
            if self.cfg.long_options_only and side_raw == "short":
                ignored_short_count += 1
                continue
            side_mult = 1
            if side_raw == "short":
                side_mult = -1
            elif side_raw == "long":
                side_mult = 1
            elif math.isfinite(qty_val) and qty_val < 0:
                side_mult = -1

            qty_abs = abs(qty_val) if math.isfinite(qty_val) else 0.0
            avg_entry = _as_float(p.get("avg_entry_price"))
            if qty_abs <= 0.0:
                continue

            if cp == "C" and side_mult > 0:
                long_candidates.append((qty_abs, symbol, avg_entry))
            elif cp == "P" and side_mult > 0:
                short_candidates.append((qty_abs, symbol, avg_entry))

        long_candidates.sort(key=lambda x: x[0], reverse=True)
        short_candidates.sort(key=lambda x: x[0], reverse=True)
        long_qty, long_symbol, long_avg_entry = long_candidates[0] if long_candidates else (0.0, None, float("nan"))
        short_qty, short_symbol, short_avg_entry = short_candidates[0] if short_candidates else (0.0, None, float("nan"))
        return {
            "underlying": under,
            "long_contracts": int(round(long_qty)) if long_symbol else 0,
            "short_contracts": int(round(short_qty)) if short_symbol else 0,
            "long_symbol": long_symbol,
            "short_symbol": short_symbol,
            "avg_entry_price_long": float(long_avg_entry) if math.isfinite(long_avg_entry) else None,
            "avg_entry_price_short": float(short_avg_entry) if math.isfinite(short_avg_entry) else None,
            "qty_long": long_qty if long_symbol else 0.0,
            "qty_short": short_qty if short_symbol else 0.0,
            "multiple_positions": (len(long_candidates) > 1 or len(short_candidates) > 1),
            "ignored_short_positions": ignored_short_count,
            "underlying_equity_qty": underlying_equity_qty,
            "underlying_equity_side": underlying_equity_side,
        }

    def _maybe_flatten_underlying_equity(
        self,
        *,
        broker_state: dict[str, Any],
        logger: Callable[[str], None],
    ) -> dict[str, Any] | None:
        qty_val = _as_float(broker_state.get("underlying_equity_qty"))
        if not math.isfinite(qty_val) or abs(qty_val) < 100.0:
            return None
        symbol = str(broker_state.get("underlying") or self.cfg.underlying).strip().upper()
        side = "sell" if qty_val > 0 else "buy"
        qty = int(round(abs(qty_val)))
        try:
            resp = self._client.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                order_type="market",
                time_in_force="day",
            )
            logger(
                "[order_policy] UNDERLYING EQUITY FLATTEN SUBMITTED "
                f"symbol={symbol} side={side} qty={qty}"
            )
            return {
                "submitted": True,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "response": resp,
            }
        except Exception as exc:
            logger(
                "[order_policy] UNDERLYING EQUITY FLATTEN FAILED "
                f"symbol={symbol} side={side} qty={qty}: {exc}"
            )
            return {
                "submitted": False,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "error": str(exc),
            }

    def _apply_broker_position_state(
        self,
        *,
        broker_state: dict[str, Any],
        preserve_bars_held: bool,
        local_ts: datetime | None,
    ) -> None:
        prev_long_contracts = int(self._long_contracts)
        prev_short_contracts = int(self._short_contracts)
        prev_long_symbol = self._long_symbol
        prev_short_symbol = self._short_symbol

        next_long_contracts = int(broker_state.get("long_contracts", 0) or 0)
        next_short_contracts = int(broker_state.get("short_contracts", 0) or 0)
        next_long_symbol = broker_state.get("long_symbol")
        next_short_symbol = broker_state.get("short_symbol")

        self._long_contracts = next_long_contracts
        self._short_contracts = next_short_contracts
        self._long_symbol = next_long_symbol
        self._short_symbol = next_short_symbol
        self._long_avg_entry_price = _as_float(broker_state.get("avg_entry_price_long"))
        self._short_avg_entry_price = _as_float(broker_state.get("avg_entry_price_short"))

        for side, prev_qty, prev_symbol, next_qty, next_symbol in (
            ("long", prev_long_contracts, prev_long_symbol, next_long_contracts, next_long_symbol),
            ("short", prev_short_contracts, prev_short_symbol, next_short_contracts, next_short_symbol),
        ):
            if next_qty <= 0:
                self._meta_side_bars_held[side] = -1
                self._meta_side_entry_signal_ts[side] = None
                self._meta_side_soft_exit_count[side] = 0
                self._reset_trail_state(side)
                self._clear_entry_structure(side=side)
                self._option_exit_best_price[side] = float("nan")
                self._option_exit_best_seen_at[side] = None
                continue
            avg_entry = self._long_avg_entry_price if side == "long" else self._short_avg_entry_price
            if preserve_bars_held and prev_qty > 0 and prev_symbol == next_symbol:
                self._meta_side_bars_held[side] = max(0, int(self._meta_side_bars_held.get(side, -1)))
                prior_best = _as_float(self._option_exit_best_price.get(side))
                if not math.isfinite(prior_best) and math.isfinite(avg_entry):
                    self._option_exit_best_price[side] = float(avg_entry)
                if self._option_exit_best_seen_at.get(side) is None:
                    self._option_exit_best_seen_at[side] = local_ts
            else:
                self._meta_side_bars_held[side] = max(0, int(self.cfg.meta_min_hold_bars))
                self._meta_side_entry_signal_ts[side] = None
                self._meta_side_soft_exit_count[side] = 0
                self._reset_trail_state(side)
                self._clear_entry_structure(side=side)
                self._option_exit_best_price[side] = float(avg_entry) if math.isfinite(avg_entry) else float("nan")
                self._option_exit_best_seen_at[side] = local_ts
                if local_ts is not None and self._symbol_expires_today(next_symbol, local_ts=local_ts):
                    self._meta_side_entry_armed[side] = False

        self._refresh_legacy_state()
        self._pending_flat_side = 0
        self._pending_flat_bars = 0
        self._pending_opposite_side = 0
        self._pending_opposite_bars = 0

    def reconcile_with_broker(
        self,
        *,
        logger: Callable[[str], None] = print,
        local_ts: datetime | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        if not self.cfg.submit_orders:
            return {
                "checked": False,
                "skipped": True,
                "reason": "simulated_orders",
                "broker_state": self._read_broker_position_state(),
            }
        if local_ts is None:
            local_ts = datetime.now(tz=self._tz)
        now_mono = float(time_mod.monotonic())
        max_age_sec = max(1.0, float(self.cfg.broker_reconcile_max_age_sec))
        if not force and (now_mono - self._last_broker_reconcile_monotonic) < max_age_sec:
            return {"checked": False, "skipped": True, "reason": "fresh_enough"}

        prev_state = {
            "long_contracts": int(self._long_contracts),
            "short_contracts": int(self._short_contracts),
            "long_symbol": self._long_symbol,
            "short_symbol": self._short_symbol,
        }
        broker_state = self._read_broker_position_state()
        self._last_broker_reconcile_monotonic = now_mono
        equity_flatten = None
        if bool(self.cfg.auto_flatten_underlying_shares):
            equity_flatten = self._maybe_flatten_underlying_equity(
                broker_state=broker_state,
                logger=logger,
            )

        changed = any(
            prev_state.get(key) != broker_state.get(key)
            for key in ("long_contracts", "short_contracts", "long_symbol", "short_symbol")
        )
        if not changed:
            return {
                "checked": True,
                "changed": False,
                "broker_state": broker_state,
                "underlying_equity_flatten": equity_flatten,
            }

        expiring_long_was_open = (
            prev_state["long_contracts"] > 0
            and self._symbol_expires_today(prev_state["long_symbol"], local_ts=local_ts)
        )
        expiring_short_was_open = (
            prev_state["short_contracts"] > 0
            and self._symbol_expires_today(prev_state["short_symbol"], local_ts=local_ts)
        )
        self._apply_broker_position_state(
            broker_state=broker_state,
            preserve_bars_held=True,
            local_ts=local_ts,
        )
        if self._expiring_position_exit_cutoff and local_ts.time() >= self._expiring_position_exit_cutoff:
            if (
                expiring_long_was_open
                and int(broker_state.get("long_contracts", 0) or 0) <= 0
            ) or (
                expiring_short_was_open
                and int(broker_state.get("short_contracts", 0) or 0) <= 0
            ):
                self._set_entry_lockout(local_ts=local_ts, reason="broker_expiration_flattened")
        logger(
            "[order_policy] BROKER RECONCILE "
            f"prev_long={prev_state['long_contracts']} prev_short={prev_state['short_contracts']} "
            f"next_long={broker_state.get('long_contracts', 0)} next_short={broker_state.get('short_contracts', 0)} "
            f"next_long_symbol={broker_state.get('long_symbol')} next_short_symbol={broker_state.get('short_symbol')}"
        )
        return {
            "checked": True,
            "changed": True,
            "previous_state": prev_state,
            "broker_state": broker_state,
            "underlying_equity_flatten": equity_flatten,
            "position": int(self._pos),
        }

    @staticmethod
    def _extract_buying_power(resp: Any) -> float:
        if not isinstance(resp, dict):
            return float("nan")
        for key in ("buying_power", "non_marginable_buying_power", "regt_buying_power"):
            val = _as_float(resp.get(key))
            if math.isfinite(val) and val > 0.0:
                return val
        return float("nan")

    @staticmethod
    def _extract_quotes(resp: Any, *, symbol: str) -> list[dict[str, Any]]:
        sym = str(symbol).strip().upper()
        if isinstance(resp, dict):
            # Common format: {"quotes": {"SYMBOL": [{...}]}}
            quotes_obj = resp.get("quotes")
            if isinstance(quotes_obj, dict):
                for key, value in quotes_obj.items():
                    if str(key).strip().upper() != sym:
                        continue
                    if isinstance(value, list):
                        return [q for q in value if isinstance(q, dict)]
                    if isinstance(value, dict):
                        return [value]
            if isinstance(quotes_obj, list):
                return [
                    q for q in quotes_obj
                    if isinstance(q, dict) and str(q.get("symbol", "")).strip().upper() == sym
                ]
            # Alternate wrappers.
            for key in ("data",):
                value = resp.get(key)
                if isinstance(value, list):
                    return [
                        q for q in value
                        if isinstance(q, dict) and str(q.get("symbol", "")).strip().upper() == sym
                    ]
                if isinstance(value, dict):
                    maybe = value.get(sym) or value.get(sym.lower()) or value.get(sym.upper())
                    if isinstance(maybe, list):
                        return [q for q in maybe if isinstance(q, dict)]
                    if isinstance(maybe, dict):
                        return [maybe]
        if isinstance(resp, list):
            return [
                q for q in resp
                if isinstance(q, dict) and str(q.get("symbol", "")).strip().upper() == sym
            ]
        return []

    @staticmethod
    def _quote_price(quote: dict[str, Any], *, mode: str) -> float:
        mode_key = str(mode or "ask").strip().lower()
        ask = _as_float(
            quote.get("ask_price", quote.get("ap", quote.get("ask")))
        )
        bid = _as_float(
            quote.get("bid_price", quote.get("bp", quote.get("bid")))
        )
        last = _as_float(quote.get("last_price", quote.get("lp")))
        mark = _as_float(quote.get("mark_price", quote.get("mark")))
        if mode_key == "bid":
            return bid
        if mode_key == "last":
            return last
        if mode_key == "mark":
            return mark if math.isfinite(mark) else (0.5 * (bid + ask) if math.isfinite(bid) and math.isfinite(ask) else float("nan"))
        if mode_key == "mid":
            if math.isfinite(bid) and math.isfinite(ask):
                return 0.5 * (bid + ask)
            if math.isfinite(mark):
                return mark
            if math.isfinite(last):
                return last
            return ask if math.isfinite(ask) else bid
        # conservative default: ask
        if math.isfinite(ask):
            return ask
        if math.isfinite(mark):
            return mark
        if math.isfinite(last):
            return last
        if math.isfinite(bid):
            return bid
        return float("nan")

    def _get_buying_power(self) -> float:
        try:
            resp = self._client.get_account()
        except Exception:
            return float("nan")
        return self._extract_buying_power(resp)

    def _get_contract_price(
        self,
        *,
        symbol: str,
        logger: Callable[[str], None],
        mode: str | None = None,
    ) -> float:
        provider = self._contract_price_provider
        if provider is not None:
            try:
                price = provider(symbol=symbol, mode=mode or self.cfg.price_mode)
            except TypeError:
                price = provider(symbol)
            except Exception as exc:
                logger(f"[order_policy] simulated quote provider failed symbol={symbol}: {exc}")
                price = float("nan")
            price_f = _as_float(price)
            if math.isfinite(price_f) and price_f > 0.0:
                return price_f

        if not self.cfg.submit_orders:
            return float("nan")

        try:
            resp = self._client.get_option_quotes(symbols=symbol, limit=1)
            quotes = self._extract_quotes(resp, symbol=symbol)
            if not quotes:
                return float("nan")
            # Use the most recent quote if timestamps are present.
            quote = quotes[-1]
            price_mode = str(mode or self.cfg.price_mode)
            return self._quote_price(quote, mode=price_mode)
        except Exception as exc:
            logger(f"[order_policy] quote fetch failed symbol={symbol}: {exc}")
            return float("nan")

    @staticmethod
    def _extract_filled_avg_price(order_like: Any) -> float:
        if not isinstance(order_like, dict):
            return float("nan")
        for key in ("filled_avg_price", "avg_execution_price", "average_fill_price"):
            val = _as_float(order_like.get(key))
            if math.isfinite(val) and val > 0.0:
                return val
        return float("nan")

    def _contracts_max_for_symbol(self, *, symbol: str, logger: Callable[[str], None]) -> tuple[int, float, float]:
        if not self.cfg.submit_orders:
            price = self._get_contract_price(symbol=symbol, logger=logger)
            max_ct = max(0, int(self.cfg.max_contracts_fallback))
            if self.cfg.max_contracts_cap and int(self.cfg.max_contracts_cap) > 0:
                max_ct = min(max_ct, int(self.cfg.max_contracts_cap))
            return max_ct, float("nan"), price
        price = self._get_contract_price(symbol=symbol, logger=logger)
        bp = self._get_buying_power()
        max_ct = 0
        if math.isfinite(bp) and bp > 0.0 and math.isfinite(price) and price > 0.0:
            max_ct = int(math.floor(bp / (price * 100.0)))
        if max_ct <= 0:
            max_ct = max(0, int(self.cfg.max_contracts_fallback))
        if self.cfg.max_contracts_cap and int(self.cfg.max_contracts_cap) > 0:
            max_ct = min(max_ct, int(self.cfg.max_contracts_cap))
        return max_ct, bp, price

    @staticmethod
    def _meta_threshold_value(closed_bar: dict[str, Any], *keys: str) -> float:
        for key in keys:
            value = _as_float(closed_bar.get(key))
            if math.isfinite(value):
                return value
        return float("nan")

    def _intrabar_setup_threshold_override(self, *, side: str) -> float:
        if side == "long":
            return _as_float(self.cfg.meta_intrabar_long_setup_threshold)
        if side == "short":
            return _as_float(self.cfg.meta_intrabar_short_setup_threshold)
        return float("nan")

    def _effective_intrabar_setup_threshold(self, *, closed_bar: dict[str, Any], side: str) -> float:
        dynamic = self._meta_threshold_value(
            closed_bar,
            f"regime_thr_enter_{side}",
            f"dynamic_thr_enter_{side}",
        )
        if math.isfinite(dynamic):
            return dynamic
        override = self._intrabar_setup_threshold_override(side=side)
        if math.isfinite(override):
            return override
        return self._meta_threshold_value(closed_bar, f"thr_enter_{side}", f"enter_{side}_threshold")

    def _intrabar_entry_policy_name(self) -> str:
        return str(self.cfg.meta_intrabar_entry_policy or "legacy_breakout_touch").strip().lower()

    def _phase4_intrabar_entry_enabled(self) -> bool:
        return self._intrabar_entry_policy_name() in {
            PHASE4_SWING_SETUP_BODYCLOSE_BODYCLOSE_V1,
            "phase4_bodyclose_bodyclose_l42_s15_max4",
            "break_prev_stop_1m_body_and_close",
        }

    def _intrabar_intent_expires_at(
        self,
        *,
        setup_ts: datetime | None,
        max_bars_override: int | None = None,
    ) -> datetime | None:
        if setup_ts is None:
            return None
        max_bars = max(
            1,
            int(self.cfg.meta_intrabar_setup_max_bars if max_bars_override is None else max_bars_override),
        )
        bar_minutes = max(1, int(self.cfg.meta_intrabar_setup_bar_minutes))
        # Project 10m bars are timestamped by bar start. A setup from bar i is
        # actionable on the next max_bars bars, so expiration is i+(max_bars+1).
        return setup_ts + timedelta(minutes=bar_minutes * (max_bars + 1))

    def _intrabar_entry_intent_expired(self, *, side: str, local_ts: datetime | None) -> bool:
        expires_at = self._meta_intrabar_entry_expires_at.get(side)
        return bool(local_ts is not None and expires_at is not None and local_ts >= expires_at)

    def _clear_intrabar_entry_intent(self, *, side: str, reason: str | None = None) -> None:
        self._meta_intrabar_entry_intent_active[side] = False
        self._meta_intrabar_entry_ref[side] = float("nan")
        self._meta_intrabar_entry_setup_ts[side] = None
        self._meta_intrabar_entry_expires_at[side] = None
        self._meta_intrabar_setup_snapshot[side] = None
        self._opening_pullback_required[side] = False
        self._opening_pullback_seen[side] = False
        if reason:
            self._meta_side_reason[side] = reason

    def _recent_ema(self, *, length: int) -> float:
        values = [float(x) for x in self._recent_1m_closes if math.isfinite(_as_float(x))]
        span = max(1, int(length))
        if len(values) < max(2, min(span, 4)):
            return float("nan")
        alpha = 2.0 / float(span + 1)
        ema = values[0]
        for value in values[1:]:
            ema = alpha * value + (1.0 - alpha) * ema
        return float(ema)

    def _opening_pullback_applies(self, *, side: str, local_ts: datetime | None) -> bool:
        if not bool(self.cfg.meta_opening_pullback_enabled):
            return False
        side_key = str(side).strip().lower()
        if side_key != "long":
            return False
        if local_ts is None:
            return False
        setup_ts = self._meta_intrabar_entry_setup_ts.get(side_key)
        if not isinstance(setup_ts, datetime):
            return False
        setup_time = _parse_hhmm(str(self.cfg.meta_opening_pullback_setup_hhmm or "09:30"))
        until_time = _parse_hhmm(str(self.cfg.meta_opening_pullback_until_hhmm or "10:00"))
        if setup_ts.time().replace(second=0, microsecond=0) != setup_time:
            return False
        if local_ts.time().replace(second=0, microsecond=0) >= until_time:
            return False
        snap = self._meta_intrabar_setup_snapshot.get(side_key) or {}
        return str(snap.get("entry_kind") or "swing").strip().lower() == "swing"

    def _opening_pullback_triggered(
        self,
        *,
        side: str,
        close: float,
        open_: float,
        low: float,
        local_ts: datetime | None,
    ) -> bool:
        side_key = str(side).strip().lower()
        if not self._opening_pullback_applies(side=side_key, local_ts=local_ts):
            return True

        snap = self._meta_intrabar_setup_snapshot.get(side_key) or {}
        signal_close = _as_float(snap.get("signal_close"))
        signal_atr = _as_float(snap.get("signal_atr"))
        stretch_atr = max(0.0, float(self.cfg.meta_opening_pullback_stretch_atr))
        if not (math.isfinite(signal_close) and math.isfinite(signal_atr) and signal_atr > 0.0):
            return True
        stretched = bool(math.isfinite(close) and close > signal_close + stretch_atr * signal_atr)
        if not self._opening_pullback_required.get(side_key, False):
            if not stretched:
                return True
            self._opening_pullback_required[side_key] = True
            self._opening_pullback_seen[side_key] = False
            self._meta_side_reason[side_key] = "opening_pullback_waiting_for_pullback"
            return False

        target = self._recent_ema(length=max(1, int(self.cfg.meta_opening_pullback_ema_length)))
        if not math.isfinite(target):
            target = signal_close
        if math.isfinite(low) and low <= target:
            self._opening_pullback_seen[side_key] = True
        if not self._opening_pullback_seen.get(side_key, False):
            self._meta_side_reason[side_key] = "opening_pullback_waiting_for_pullback"
            return False
        if math.isfinite(close) and math.isfinite(open_) and close > target and close > open_:
            self._meta_side_reason[side_key] = "opening_pullback_reclaim"
            return True
        self._meta_side_reason[side_key] = "opening_pullback_waiting_for_reclaim"
        return False

    def _capture_intrabar_setup_snapshot(
        self,
        *,
        side: str,
        closed_bar: dict[str, Any],
        ref: float,
        setup_ts: datetime | None,
        expires_at: datetime | None,
        prob: float,
        threshold: float,
        entry_kind: str,
        source_side: str,
    ) -> None:
        self._meta_intrabar_setup_snapshot[side] = {
            "setup_ts": setup_ts.isoformat() if isinstance(setup_ts, datetime) else None,
            "expires_at": expires_at.isoformat() if isinstance(expires_at, datetime) else None,
            "signal_open": _as_float(closed_bar.get("open")),
            "signal_high": _as_float(closed_bar.get("high", closed_bar.get("signal_high"))),
            "signal_low": _as_float(closed_bar.get("low", closed_bar.get("signal_low"))),
            "signal_close": _as_float(closed_bar.get("close", closed_bar.get("signal_close"))),
            "signal_atr": _as_float(closed_bar.get("signal_atr")),
            "ref": float(ref) if math.isfinite(ref) else float("nan"),
            "prob": float(prob) if math.isfinite(prob) else float("nan"),
            "threshold": float(threshold) if math.isfinite(threshold) else float("nan"),
            "entry_kind": str(entry_kind),
            "source_side": str(source_side),
        }

    def _seed_entry_structure(
        self,
        *,
        side: str,
        local_ts: datetime | None,
        entry_price: float,
    ) -> None:
        snap = self._meta_intrabar_setup_snapshot.get(side)
        if not isinstance(snap, dict):
            self._meta_entry_structure[side] = None
            return
        setup_ts = snap.get("setup_ts")
        expires_at = snap.get("expires_at")
        parsed_setup_ts = pd.to_datetime(setup_ts, utc=True, errors="coerce") if setup_ts else None
        parsed_expires_at = pd.to_datetime(expires_at, utc=True, errors="coerce") if expires_at else None
        self._meta_entry_structure[side] = {
            "setup_ts": (
                parsed_setup_ts.to_pydatetime().astimezone(self._tz)
                if parsed_setup_ts is not None and not pd.isna(parsed_setup_ts)
                else None
            ),
            "expires_at": (
                parsed_expires_at.to_pydatetime().astimezone(self._tz)
                if parsed_expires_at is not None and not pd.isna(parsed_expires_at)
                else None
            ),
            "entry_ts": local_ts,
            "entry_price": float(entry_price) if math.isfinite(entry_price) else float("nan"),
            "signal_high": _as_float(snap.get("signal_high")),
            "signal_low": _as_float(snap.get("signal_low")),
            "signal_close": _as_float(snap.get("signal_close")),
            "signal_atr": _as_float(snap.get("signal_atr")),
            "ref": _as_float(snap.get("ref")),
            "prob": _as_float(snap.get("prob")),
            "threshold": _as_float(snap.get("threshold")),
            "entry_kind": str(snap.get("entry_kind") or ""),
            "source_side": str(snap.get("source_side") or ""),
        }

    def _clear_entry_structure(self, *, side: str) -> None:
        self._meta_entry_structure[side] = None

    @staticmethod
    def _blend_avg_entry_price(
        *,
        current_avg: float,
        current_qty: int,
        fill_price: float,
        fill_qty: int,
    ) -> float:
        if not (math.isfinite(fill_price) and int(fill_qty) > 0):
            return float("nan")
        if not (math.isfinite(current_avg) and int(current_qty) > 0):
            return float(fill_price)
        total_qty = int(current_qty) + int(fill_qty)
        if total_qty <= 0:
            return float(fill_price)
        return (
            float(current_avg) * int(current_qty)
            + float(fill_price) * int(fill_qty)
        ) / float(total_qty)

    def _intrabar_entry_triggered(
        self,
        *,
        side: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        local_ts: datetime | None,
    ) -> bool:
        if side == "long" and self._long_contracts > 0:
            self._clear_intrabar_entry_intent(side=side, reason="already_in_position")
            return False
        if side == "short" and self._short_contracts > 0:
            self._clear_intrabar_entry_intent(side=side, reason="already_in_position")
            return False
        if not bool(self._meta_intrabar_entry_intent_active.get(side)):
            return False
        if self._intrabar_entry_intent_expired(side=side, local_ts=local_ts):
            self._clear_intrabar_entry_intent(side=side, reason="intrabar_setup_expired")
            return False

        ref = _as_float(self._meta_intrabar_entry_ref.get(side))
        if not math.isfinite(ref):
            self._clear_intrabar_entry_intent(side=side, reason="intrabar_setup_missing_ref")
            return False

        if self._phase4_intrabar_entry_enabled():
            if side == "long":
                if (
                    self._opening_pullback_required.get(side, False)
                    and self._opening_pullback_applies(side=side, local_ts=local_ts)
                ):
                    return self._opening_pullback_triggered(
                        side=side,
                        close=close,
                        open_=open_,
                        low=low,
                        local_ts=local_ts,
                    )
                breakout_confirmed = bool(
                    math.isfinite(high)
                    and math.isfinite(open_)
                    and math.isfinite(close)
                    and high >= ref
                    and close > open_
                    and close > ref
                )
                if not breakout_confirmed:
                    return False
                return self._opening_pullback_triggered(
                    side=side,
                    close=close,
                    open_=open_,
                    low=low,
                    local_ts=local_ts,
                )
            return bool(
                math.isfinite(low)
                and math.isfinite(open_)
                and math.isfinite(close)
                and low <= ref
                and close < open_
                and close < ref
            )

        if side == "long":
            return bool(math.isfinite(high) and high >= ref)
        return bool(math.isfinite(low) and low <= ref)

    @staticmethod
    def _trend_gate_passed(
        *,
        side: str,
        transition: str,
        close: float,
        prev_close: float,
        recent_closes: list[float] | None = None,
    ) -> bool:
        side_key = str(side).strip().lower()
        transition_key = str(transition).strip().lower()
        if side_key not in {"long", "short"}:
            return True
        if transition_key not in {"enter", "exit"}:
            return True
        if not (math.isfinite(close) and math.isfinite(prev_close)):
            return True
        if transition_key == "enter":
            return close > prev_close if side_key == "long" else close < prev_close

        return close < prev_close if side_key == "long" else close > prev_close

    def _cache_meta_side_snapshot(self, *, closed_bar: dict[str, Any], atr: float) -> None:
        thr_enter_long = self._effective_intrabar_setup_threshold(closed_bar=closed_bar, side="long")
        thr_enter_short = self._effective_intrabar_setup_threshold(closed_bar=closed_bar, side="short")

        self._latest_meta_side_snapshot = {
            "timestamp": closed_bar.get("timestamp"),
            "signal_close": _as_float(closed_bar.get("close")),
            "signal_high": _as_float(closed_bar.get("high")),
            "signal_low": _as_float(closed_bar.get("low")),
            "signal_atr": float(atr) if math.isfinite(atr) else float("nan"),
            "p_enter_long": _as_float(closed_bar.get("p_enter_long")),
            "p_enter_short": _as_float(closed_bar.get("p_enter_short")),
            "p_exit_long": _as_float(closed_bar.get("p_exit_long")),
            "p_exit_short": _as_float(closed_bar.get("p_exit_short")),
            "ema_fast": _as_float(closed_bar.get("ema_fast")),
            "ema_slow": _as_float(closed_bar.get("ema_slow")),
            "thr_enter_long": thr_enter_long,
            "thr_enter_short": thr_enter_short,
            "regime_thr_enter_long": self._meta_threshold_value(closed_bar, "regime_thr_enter_long"),
            "regime_thr_enter_short": self._meta_threshold_value(closed_bar, "regime_thr_enter_short"),
            "trend_regime": closed_bar.get("trend_regime"),
            "p_enter_long_regime_pctile": self._meta_threshold_value(closed_bar, "p_enter_long_regime_pctile"),
            "p_enter_short_regime_pctile": self._meta_threshold_value(closed_bar, "p_enter_short_regime_pctile"),
            "thr_exit_long": self._meta_threshold_value(closed_bar, "thr_exit_long", "exit_long_threshold"),
            "thr_exit_short": self._meta_threshold_value(closed_bar, "thr_exit_short", "exit_short_threshold"),
        }
        self._update_meta_entry_rearm_state(closed_bar=self._latest_meta_side_snapshot)

    def _meta_side_validity_flags(self, *, closed_bar: dict[str, Any]) -> tuple[bool, bool]:
        p_enter_long = _as_float(closed_bar.get("p_enter_long"))
        p_enter_short = _as_float(closed_bar.get("p_enter_short"))
        thr_enter_long = self._effective_intrabar_setup_threshold(closed_bar=closed_bar, side="long")
        thr_enter_short = self._effective_intrabar_setup_threshold(closed_bar=closed_bar, side="short")
        long_ready = bool(
            math.isfinite(p_enter_long)
            and math.isfinite(thr_enter_long)
            and p_enter_long >= thr_enter_long
        )
        short_ready = bool(
            math.isfinite(p_enter_short)
            and math.isfinite(thr_enter_short)
            and p_enter_short >= thr_enter_short
        )
        long_margin = (p_enter_long - thr_enter_long) if long_ready else float("-inf")
        short_margin = (p_enter_short - thr_enter_short) if short_ready else float("-inf")
        dominance_delta = max(0.0, float(self.cfg.meta_intrabar_opposite_dominance_delta))
        long_invalidated = bool(short_ready and short_margin > long_margin + dominance_delta)
        short_invalidated = bool(long_ready and long_margin > short_margin + dominance_delta)
        long_valid = bool(long_ready and not long_invalidated)
        short_valid = bool(short_ready and not short_invalidated)
        if bool(self.cfg.meta_countertrend_veto_enabled):
            regime = str(closed_bar.get("trend_regime") or "").strip().lower()
            counter_min = max(0.0, float(self.cfg.meta_countertrend_min_prob))
            if regime == "bullish" and short_valid and p_enter_short < counter_min:
                short_valid = False
            elif regime == "bearish" and long_valid and p_enter_long < counter_min:
                long_valid = False
            elif regime == "neutral":
                if long_valid and p_enter_long < max(0.0, float(self.cfg.meta_neutral_long_min_prob)):
                    long_valid = False
                if short_valid and p_enter_short < max(0.0, float(self.cfg.meta_neutral_short_min_prob)):
                    short_valid = False
        return long_valid, short_valid

    def _continuation_entry_valid(self, *, closed_bar: dict[str, Any], side: str) -> bool:
        if not bool(self.cfg.meta_continuation_entry_enabled):
            return False
        side_key = str(side).strip().lower()
        regime = str(closed_bar.get("trend_regime") or "").strip().lower()
        close = _as_float(closed_bar.get("signal_close", closed_bar.get("close")))
        high = _as_float(closed_bar.get("signal_high", closed_bar.get("high")))
        low = _as_float(closed_bar.get("signal_low", closed_bar.get("low")))
        ema_fast = _as_float(closed_bar.get("ema_fast"))
        ema_slow = _as_float(closed_bar.get("ema_slow"))
        atr = _as_float(closed_bar.get("signal_atr"))
        if not math.isfinite(atr) or atr <= 0.0:
            atr = self._compute_atr()
        pullback_atr = max(0.0, float(self.cfg.meta_continuation_pullback_atr))

        if side_key == "long" and regime == "bullish":
            source_prob = _as_float(closed_bar.get("p_enter_short"))
            if source_prob < max(0.0, float(self.cfg.meta_continuation_source_short_threshold)):
                return False
            trend_ok = True
            shallow = True
            if math.isfinite(close) and math.isfinite(ema_fast):
                trend_ok = close >= ema_fast
            if trend_ok and math.isfinite(ema_fast) and math.isfinite(ema_slow):
                trend_ok = ema_fast >= ema_slow
            if math.isfinite(low) and math.isfinite(ema_fast) and math.isfinite(atr):
                shallow = low >= ema_fast - pullback_atr * atr
            return bool(trend_ok and shallow and math.isfinite(high))

        if side_key == "short" and regime == "bearish":
            source_prob = _as_float(closed_bar.get("p_enter_long"))
            if source_prob < max(0.0, float(self.cfg.meta_continuation_source_long_threshold)):
                return False
            trend_ok = True
            shallow = True
            if math.isfinite(close) and math.isfinite(ema_fast):
                trend_ok = close <= ema_fast
            if trend_ok and math.isfinite(ema_fast) and math.isfinite(ema_slow):
                trend_ok = ema_fast <= ema_slow
            if math.isfinite(high) and math.isfinite(ema_fast) and math.isfinite(atr):
                shallow = high <= ema_fast + pullback_atr * atr
            return bool(trend_ok and shallow and math.isfinite(low))
        return False

    def _scalp_entry_valid(self, *, closed_bar: dict[str, Any], side: str) -> bool:
        if not bool(self.cfg.scalp_mode_enabled):
            return False
        side_key = str(side).strip().lower()
        if side_key not in {"long", "short"}:
            return False

        prob = _as_float(closed_bar.get(f"p_enter_{side_key}"))
        threshold = (
            max(0.0, float(self.cfg.scalp_long_setup_threshold))
            if side_key == "long"
            else max(0.0, float(self.cfg.scalp_short_setup_threshold))
        )
        if not (math.isfinite(prob) and prob >= threshold):
            return False

        swing_threshold = self._effective_intrabar_setup_threshold(closed_bar=closed_bar, side=side_key)
        if math.isfinite(swing_threshold) and prob >= swing_threshold:
            return False

        high = _as_float(closed_bar.get("high", closed_bar.get("signal_high")))
        low = _as_float(closed_bar.get("low", closed_bar.get("signal_low")))
        open_ = _as_float(closed_bar.get("open", closed_bar.get("signal_open")))
        close = _as_float(closed_bar.get("close", closed_bar.get("signal_close")))
        atr = _as_float(closed_bar.get("signal_atr"))
        if not math.isfinite(atr) or atr <= 0.0:
            atr = self._compute_atr()
        min_range_atr = max(0.0, float(self.cfg.scalp_min_signal_range_atr))
        if min_range_atr > 0.0:
            if not (math.isfinite(high) and math.isfinite(low) and math.isfinite(atr) and atr > 0.0):
                return False
            if (high - low) < min_range_atr * atr:
                return False

        if bool(self.cfg.scalp_require_reversal_close):
            if not (math.isfinite(high) and math.isfinite(low) and math.isfinite(close)):
                return False
            mid = (high + low) / 2.0
            if side_key == "long" and close < mid:
                return False
            if side_key == "short" and close > mid:
                return False
            if math.isfinite(open_):
                if side_key == "long" and close <= open_:
                    return False
                if side_key == "short" and close >= open_:
                    return False
        return True

    def _refresh_intrabar_entry_intents(
        self,
        *,
        closed_bar: dict[str, Any],
        local_ts: datetime | None,
    ) -> None:
        long_valid, short_valid = self._meta_side_validity_flags(closed_bar=closed_bar)
        signal_high = _as_float(closed_bar.get("high"))
        if not math.isfinite(signal_high):
            signal_high = _as_float(closed_bar.get("signal_high"))
        signal_low = _as_float(closed_bar.get("low"))
        if not math.isfinite(signal_low):
            signal_low = _as_float(closed_bar.get("signal_low"))
        setup_ts = self._to_local_ts(closed_bar.get("timestamp"))
        continuation_valid = {
            "long": self._continuation_entry_valid(closed_bar=closed_bar, side="long"),
            "short": self._continuation_entry_valid(closed_bar=closed_bar, side="short"),
        }
        scalp_valid = {
            "long": self._scalp_entry_valid(closed_bar=closed_bar, side="long"),
            "short": self._scalp_entry_valid(closed_bar=closed_bar, side="short"),
        }
        for side_key, valid in (("long", long_valid), ("short", short_valid)):
            swing_valid = bool(valid)
            continuation_side_valid = bool(continuation_valid.get(side_key, False))
            scalp_side_valid = bool(
                scalp_valid.get(side_key, False)
                and not swing_valid
                and not continuation_side_valid
            )
            valid = bool(swing_valid or continuation_side_valid or scalp_side_valid)
            current_qty = self._long_contracts if side_key == "long" else self._short_contracts
            if current_qty > 0:
                self._clear_intrabar_entry_intent(side=side_key, reason="already_in_position")
                continue
            if (
                valid
                and bool(self._meta_side_entry_armed.get(side_key, True))
                and not self._side_reentry_cooldown_active(side=side_key, local_ts=local_ts)
            ):
                self._meta_intrabar_entry_intent_active[side_key] = True
                self._opening_pullback_required[side_key] = False
                self._opening_pullback_seen[side_key] = False
                self._meta_intrabar_entry_ref[side_key] = (
                    float(signal_high) if side_key == "long" else float(signal_low)
                )
                self._meta_intrabar_entry_setup_ts[side_key] = setup_ts
                expires_at = self._intrabar_intent_expires_at(
                    setup_ts=setup_ts,
                    max_bars_override=(
                        int(self.cfg.scalp_setup_max_bars)
                        if scalp_side_valid
                        else None
                    ),
                )
                self._meta_intrabar_entry_expires_at[side_key] = expires_at
                if scalp_side_valid:
                    entry_kind = "scalp"
                elif continuation_side_valid and not swing_valid:
                    entry_kind = "continuation"
                else:
                    entry_kind = "swing"
                prob = _as_float(closed_bar.get(f"p_enter_{side_key}"))
                if entry_kind == "scalp":
                    threshold = (
                        max(0.0, float(self.cfg.scalp_long_setup_threshold))
                        if side_key == "long"
                        else max(0.0, float(self.cfg.scalp_short_setup_threshold))
                    )
                else:
                    threshold = self._effective_intrabar_setup_threshold(closed_bar=closed_bar, side=side_key)
                source_side = side_key
                if entry_kind == "continuation":
                    source_side = "short_failed" if side_key == "long" else "long_failed"
                self._capture_intrabar_setup_snapshot(
                    side=side_key,
                    closed_bar=closed_bar,
                    ref=self._meta_intrabar_entry_ref[side_key],
                    setup_ts=setup_ts,
                    expires_at=expires_at,
                    prob=prob,
                    threshold=threshold,
                    entry_kind=entry_kind,
                    source_side=source_side,
                )
                self._meta_side_reason[side_key] = (
                    f"intrabar_{entry_kind}_setup_armed:{self._intrabar_entry_policy_name()}"
                )
            elif (
                bool(self._meta_intrabar_entry_intent_active.get(side_key, False))
                and not self._intrabar_entry_intent_expired(side=side_key, local_ts=local_ts)
            ):
                self._meta_side_reason[side_key] = f"intrabar_setup_pending:{self._intrabar_entry_policy_name()}"
            else:
                self._clear_intrabar_entry_intent(side=side_key, reason="intrabar_setup_not_armed")

    def _update_meta_entry_rearm_state(self, *, closed_bar: dict[str, Any]) -> None:
        snapshot_ts = self._to_local_ts(closed_bar.get("timestamp"))
        if snapshot_ts is not None:
            session_day = snapshot_ts.date()
            if self._meta_rearm_session_day is not None and session_day != self._meta_rearm_session_day:
                self._meta_side_entry_armed = {"long": True, "short": True}
                self._meta_side_enter_above_threshold = {"long": False, "short": False}
                self._meta_side_enter_peak_prob = {"long": float("nan"), "short": float("nan")}
            self._meta_rearm_session_day = session_day
        for side_key in ("long", "short"):
            enter_prob = _as_float(closed_bar.get(f"p_enter_{side_key}"))
            enter_thr = self._effective_intrabar_setup_threshold(
                closed_bar=closed_bar,
                side=side_key,
            )
            is_above = bool(
                math.isfinite(enter_prob)
                and math.isfinite(enter_thr)
                and enter_prob >= enter_thr
            )
            if not is_above:
                self._meta_side_entry_armed[side_key] = True
                self._meta_side_enter_above_threshold[side_key] = False
                self._meta_side_enter_peak_prob[side_key] = float("nan")
                continue
            self._meta_side_entry_armed[side_key] = True
            peak_prob = _as_float(self._meta_side_enter_peak_prob.get(side_key))
            if math.isfinite(enter_prob) and (not math.isfinite(peak_prob) or enter_prob > peak_prob):
                self._meta_side_enter_peak_prob[side_key] = enter_prob
            self._meta_side_enter_above_threshold[side_key] = True

    def _entry_no_chase_blocked(
        self,
        *,
        side: str,
        close: float,
        atr: float,
        local_ts: datetime | None,
    ) -> bool:
        side_key = str(side).strip().lower()
        if side_key not in {"long", "short"} or not math.isfinite(close):
            return False

        setup_ts = self._meta_intrabar_entry_setup_ts.get(side_key)
        max_age_min = max(0, int(self.cfg.meta_intrabar_max_confirmation_age_minutes))
        if (
            max_age_min > 0
            and local_ts is not None
            and isinstance(setup_ts, datetime)
            and local_ts >= setup_ts + timedelta(minutes=max_age_min)
        ):
            self._clear_intrabar_entry_intent(side=side_key, reason="entry_confirmation_too_late")
            return True

        setup_snapshot = self._meta_intrabar_setup_snapshot.get(side_key) or {}
        signal_close = _as_float(setup_snapshot.get("signal_close"))
        signal_atr = _as_float(setup_snapshot.get("signal_atr"))
        ref_atr = signal_atr if math.isfinite(signal_atr) and signal_atr > 0.0 else atr
        if math.isfinite(ref_atr) and ref_atr > 0.0:
            ref = _as_float(self._meta_intrabar_entry_ref.get(side_key))
            ref_chase_atr = max(0.0, float(self.cfg.meta_intrabar_ref_chase_atr))
            if ref_chase_atr > 0.0 and math.isfinite(ref):
                max_ref_chase = ref_chase_atr * ref_atr
                if side_key == "long" and close > ref + max_ref_chase:
                    self._meta_side_reason[side_key] = "entry_no_chase_ref_extension"
                    return True
                if side_key == "short" and close < ref - max_ref_chase:
                    self._meta_side_reason[side_key] = "entry_no_chase_ref_extension"
                    return True

        chase_atr = float(self.cfg.meta_no_chase_atr)
        if chase_atr <= 0.0:
            return False
        if not (math.isfinite(signal_close) and math.isfinite(ref_atr) and ref_atr > 0.0):
            return False
        max_chase = chase_atr * ref_atr
        if side_key == "long" and close > signal_close + max_chase:
            self._meta_side_reason[side_key] = "entry_no_chase_signal_extension"
            return True
        if side_key == "short" and close < signal_close - max_chase:
            self._meta_side_reason[side_key] = "entry_no_chase_signal_extension"
            return True
        return False

    def _side_reentry_cooldown_active(self, *, side: str, local_ts: datetime | None) -> bool:
        if local_ts is None:
            return False
        until_ts = self._meta_side_cooldown_until.get(str(side).strip().lower())
        return until_ts is not None and local_ts < until_ts

    @classmethod
    def _has_meta_side_thresholds(cls, closed_bar: dict[str, Any]) -> bool:
        return (
            math.isfinite(cls._meta_threshold_value(closed_bar, "thr_enter_long", "enter_long_threshold"))
            and math.isfinite(cls._meta_threshold_value(closed_bar, "thr_enter_short", "enter_short_threshold"))
            and math.isfinite(cls._meta_threshold_value(closed_bar, "thr_exit_long", "exit_long_threshold"))
            and math.isfinite(cls._meta_threshold_value(closed_bar, "thr_exit_short", "exit_short_threshold"))
        )

    def _has_meta_side_entry_thresholds(self, closed_bar: dict[str, Any]) -> bool:
        long_thr = self._meta_threshold_value(closed_bar, "thr_enter_long", "enter_long_threshold")
        short_thr = self._meta_threshold_value(closed_bar, "thr_enter_short", "enter_short_threshold")
        intrabar_long_thr = self._intrabar_setup_threshold_override(side="long")
        intrabar_short_thr = self._intrabar_setup_threshold_override(side="short")
        if math.isfinite(intrabar_long_thr):
            long_thr = intrabar_long_thr
        if math.isfinite(intrabar_short_thr):
            short_thr = intrabar_short_thr
        return (
            math.isfinite(long_thr)
            and math.isfinite(short_thr)
        )

    def _missed_trade_diagnostics(self) -> dict[str, Any]:
        snapshot = self._latest_meta_side_snapshot or {}
        out: dict[str, Any] = {}
        for side_key, opposite_key in (("long", "short"), ("short", "long")):
            prob = _as_float(snapshot.get(f"p_enter_{side_key}"))
            opp_prob = _as_float(snapshot.get(f"p_enter_{opposite_key}"))
            threshold = _as_float(snapshot.get(f"thr_enter_{side_key}"))
            close = _as_float(snapshot.get("signal_close"))
            high = _as_float(snapshot.get("signal_high"))
            low = _as_float(snapshot.get("signal_low"))
            ref = _as_float(self._meta_intrabar_entry_ref.get(side_key))
            gap = threshold - prob if math.isfinite(prob) and math.isfinite(threshold) else float("nan")
            near_miss = bool(math.isfinite(gap) and 0.0 < gap <= 0.10)
            below_ref = bool(math.isfinite(close) and math.isfinite(ref) and close < ref)
            above_ref = bool(math.isfinite(close) and math.isfinite(ref) and close > ref)
            if side_key == "long":
                ref_state = "above_ref" if above_ref else ("below_ref" if below_ref else "no_ref")
                last_range_ok = bool(math.isfinite(high) and math.isfinite(ref) and high >= ref)
            else:
                ref_state = "below_ref" if below_ref else ("above_ref" if above_ref else "no_ref")
                last_range_ok = bool(math.isfinite(low) and math.isfinite(ref) and low <= ref)
            out[side_key] = {
                "prob": float(prob) if math.isfinite(prob) else None,
                "threshold": float(threshold) if math.isfinite(threshold) else None,
                "gap_to_threshold": float(gap) if math.isfinite(gap) else None,
                "near_threshold": near_miss,
                "opposite_prob": float(opp_prob) if math.isfinite(opp_prob) else None,
                "opposite_suppressed_015": bool(math.isfinite(opp_prob) and opp_prob <= 0.15),
                "intent_active": bool(self._meta_intrabar_entry_intent_active.get(side_key, False)),
                "entry_ref": float(ref) if math.isfinite(ref) else None,
                "last_bar_touched_ref": last_range_ok,
                "close_vs_ref": ref_state,
                "reason": self._meta_side_reason.get(side_key),
                "expires_at": (
                    self._meta_intrabar_entry_expires_at[side_key].isoformat()
                    if isinstance(self._meta_intrabar_entry_expires_at.get(side_key), datetime)
                    else None
                ),
            }
        return out

    def _smooth_action(self, action: float) -> float:
        alpha = min(max(float(self.cfg.ema_alpha), 0.0), 0.9999)
        if self._action_ema is None or not math.isfinite(self._action_ema):
            self._action_ema = float(action)
        else:
            self._action_ema = alpha * float(self._action_ema) + (1.0 - alpha) * float(action)
        return float(self._action_ema)

    def sync_from_broker(self, *, logger: Callable[[str], None] = print) -> dict[str, Any]:
        """
        Seed in-memory position state from Alpaca open positions for this underlying.
        """
        try:
            broker_state = self._read_broker_position_state()
            self._apply_broker_position_state(
                broker_state=broker_state,
                preserve_bars_held=False,
                local_ts=datetime.now(tz=self._tz),
            )
            self._last_broker_reconcile_monotonic = float(time_mod.monotonic())
            if (
                int(broker_state.get("long_contracts", 0) or 0) <= 0
                and int(broker_state.get("short_contracts", 0) or 0) <= 0
            ):
                logger(
                    f"[order_policy] Startup sync: no open {broker_state.get('underlying')} long option positions found."
                )
            elif bool(broker_state.get("multiple_positions")):
                logger(
                    f"[order_policy] Startup sync warning: multiple open {broker_state.get('underlying')} option positions "
                    "found; using largest qty symbol per side."
                )
            logger(
                f"[order_policy] Startup sync: restored long={self._long_contracts} symbol={self._long_symbol} "
                f"short={self._short_contracts} symbol={self._short_symbol} "
                f"signed_contracts={self._signed_contracts}"
            )
            return {
                "synced": True,
                "position": self._pos,
                "signed_contracts": self._signed_contracts,
                "long_contracts": self._long_contracts,
                "short_contracts": self._short_contracts,
                "long_symbol": self._long_symbol,
                "short_symbol": self._short_symbol,
                "qty_long": broker_state.get("qty_long", 0.0),
                "qty_short": broker_state.get("qty_short", 0.0),
                "avg_entry_price_long": broker_state.get("avg_entry_price_long"),
                "avg_entry_price_short": broker_state.get("avg_entry_price_short"),
                "multiple_positions": bool(broker_state.get("multiple_positions")),
                "ignored_short_positions": int(broker_state.get("ignored_short_positions", 0) or 0),
            }
        except Exception as exc:
            logger(f"[order_policy] Startup sync failed: {exc}")
            return {"synced": False, "error": str(exc)}

    @staticmethod
    def _extract_contracts(resp: Any) -> list[dict[str, Any]]:
        if isinstance(resp, dict):
            for key in ("option_contracts", "contracts", "data"):
                value = resp.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]
        if isinstance(resp, list):
            return [x for x in resp if isinstance(x, dict)]
        return []

    @staticmethod
    def _status_key(status: Any) -> str:
        return str(status or "").strip().lower()

    @staticmethod
    def _order_is_success(status: Any) -> bool:
        return OptionOrderPolicy._status_key(status) in {"filled", "partially_filled"}

    @staticmethod
    def _order_is_terminal_fail(status: Any) -> bool:
        return OptionOrderPolicy._status_key(status) in {
            "canceled",
            "cancelled",
            "expired",
            "rejected",
            "failed",
            "suspended",
        }

    def _has_open_long_position(self, *, symbol: str) -> bool:
        if not self.cfg.submit_orders:
            target = str(symbol).strip().upper()
            return bool(
                (self._long_symbol and str(self._long_symbol).strip().upper() == target and self._long_contracts > 0)
                or (
                    self._short_symbol
                    and str(self._short_symbol).strip().upper() == target
                    and self._short_contracts > 0
                )
            )
        try:
            resp = self._client.get_positions()
            positions = self._extract_positions(resp)
        except Exception:
            return False

        target = str(symbol).strip().upper()
        for p in positions:
            sym = str(p.get("symbol", "")).strip().upper()
            if sym != target:
                continue
            side_raw = str(p.get("side", "")).strip().lower()
            qty_val = _as_float(p.get("qty"))
            if self.cfg.long_options_only:
                if side_raw == "short":
                    continue
                if side_raw == "long":
                    return True
                if math.isfinite(qty_val) and qty_val > 0:
                    return True
                continue
            if side_raw in {"long", "short"}:
                return True
            if math.isfinite(qty_val) and abs(qty_val) > 0.0:
                return True
        return False

    def _verify_order_submission(
        self,
        *,
        submitted_resp: dict[str, Any],
        symbol: str,
        intent: str,
        logger: Callable[[str], None],
    ) -> dict[str, Any]:
        order_id = str(submitted_resp.get("id", "")).strip()
        last = submitted_resp
        status = self._status_key(last.get("status"))
        timeout_sec = max(1.0, float(self.cfg.verify_timeout_sec))
        poll_sec = max(0.2, float(self.cfg.verify_poll_sec))
        deadline = time_mod.monotonic() + timeout_sec

        if self._order_is_success(status):
            return {
                "verified": True,
                "status": status,
                "order_id": order_id,
                "via": "submit_response",
                "retryable": False,
                "order": last,
            }
        if self._order_is_terminal_fail(status):
            return {
                "verified": False,
                "status": status,
                "order_id": order_id,
                "via": "submit_response",
                "retryable": True,
                "order": last,
            }

        while time_mod.monotonic() < deadline and order_id:
            time_mod.sleep(poll_sec)
            try:
                current = self._client.get_order(order_id)
                if isinstance(current, dict):
                    last = current
                    status = self._status_key(current.get("status"))
            except Exception as exc:
                logger(f"[order_policy] verify poll warning order_id={order_id}: {exc}")
                continue

            if self._order_is_success(status):
                return {
                    "verified": True,
                    "status": status,
                    "order_id": order_id,
                    "via": "order_poll",
                    "retryable": False,
                    "order": last,
                }
            if self._order_is_terminal_fail(status):
                return {
                    "verified": False,
                    "status": status,
                    "order_id": order_id,
                    "via": "order_poll",
                    "retryable": True,
                    "order": last,
                }

        # Timeout fallback: reconcile with actual positions to reduce false negatives.
        try:
            has_pos = self._has_open_long_position(symbol=symbol)
            if intent == "open" and has_pos:
                return {
                    "verified": True,
                    "status": status or "unknown",
                    "order_id": order_id,
                    "via": "positions_reconcile",
                    "retryable": False,
                    "order": last,
                }
            if intent == "close" and not has_pos:
                return {
                    "verified": True,
                    "status": status or "unknown",
                    "order_id": order_id,
                    "via": "positions_reconcile",
                    "retryable": False,
                    "order": last,
                }
        except Exception:
            pass

        return {
            "verified": False,
            "status": status or "unknown",
            "order_id": order_id,
            "via": "timeout",
            "retryable": bool(order_id),
            "cancel_required": bool(order_id),
            "order": last,
        }

    def _mark_pending_broker_reconcile(
        self,
        *,
        symbol: str,
        intent: str,
        side: str,
        qty: int,
        verify_result: dict[str, Any],
        logger: Callable[[str], None],
    ) -> None:
        grace_sec = max(1.0, float(self.cfg.verify_final_attempt_grace_sec))
        self._pending_broker_reconcile = {
            "symbol": str(symbol).strip().upper(),
            "intent": str(intent).strip().lower(),
            "side": str(side).strip().lower(),
            "qty": int(qty),
            "order_id": str(verify_result.get("order_id", "")).strip() or None,
            "status": str(verify_result.get("status", "")).strip().lower() or None,
            "via": str(verify_result.get("via", "")).strip().lower() or None,
            "created_monotonic": float(time_mod.monotonic()),
            "deadline_monotonic": float(time_mod.monotonic() + grace_sec),
        }
        logger(
            "[order_policy] ORDER PENDING RECONCILE "
            f"intent={intent} symbol={symbol} order_id={verify_result.get('order_id')} "
            f"status={verify_result.get('status')} via={verify_result.get('via')} "
            f"grace_sec={grace_sec:.0f}"
        )

    def has_pending_broker_reconcile(self) -> bool:
        return isinstance(self._pending_broker_reconcile, dict)

    def _pending_open_direction(self, pending: dict[str, Any]) -> str | None:
        if str(pending.get("intent", "")).strip().lower() != "open":
            return None
        cp = self._option_cp(str(pending.get("symbol", "")))
        if cp == "C":
            return "long"
        if cp == "P":
            return "short"
        return None

    def _pending_open_invalidated_by_setup(self, *, closed_bar: dict[str, Any]) -> tuple[bool, str]:
        pending = self._pending_broker_reconcile
        if not isinstance(pending, dict):
            return False, "no_pending_reconcile"
        pending_side = self._pending_open_direction(pending)
        if pending_side not in {"long", "short"}:
            return False, "not_pending_open"
        if not self._has_meta_side_entry_thresholds(closed_bar):
            return False, "no_setup_thresholds"
        long_valid, short_valid = self._meta_side_validity_flags(closed_bar=closed_bar)
        current_valid = long_valid if pending_side == "long" else short_valid
        opposite_valid = short_valid if pending_side == "long" else long_valid
        if current_valid:
            return False, f"pending_{pending_side}_still_valid"
        if opposite_valid:
            return True, f"opposite_setup_valid_pending_{pending_side}"
        return True, f"pending_{pending_side}_setup_no_longer_valid"

    def _cancel_pending_open_if_invalidated(
        self,
        *,
        closed_bar: dict[str, Any],
        logger: Callable[[str], None],
    ) -> dict[str, Any] | None:
        pending = self._pending_broker_reconcile
        if not isinstance(pending, dict):
            return None
        invalidated, reason = self._pending_open_invalidated_by_setup(closed_bar=closed_bar)
        if not invalidated:
            return None

        symbol = str(pending.get("symbol", "")).strip().upper()
        order_id = str(pending.get("order_id", "")).strip() or None
        pending_side = self._pending_open_direction(pending) or "unknown"
        if self.cfg.submit_orders and order_id:
            try:
                self._client.cancel_order(order_id)
                logger(
                    "[order_policy] STALE PENDING OPEN CANCELED "
                    f"side={pending_side} symbol={symbol} order_id={order_id} reason={reason}"
                )
            except Exception as exc:
                logger(
                    "[order_policy] stale pending open cancel warning "
                    f"side={pending_side} symbol={symbol} order_id={order_id}: {exc}"
                )
                return {
                    "event": "hold",
                    "reason": "stale_pending_open_cancel_failed",
                    "pending_side": pending_side,
                    "pending_symbol": symbol,
                    "pending_order_id": order_id,
                    "invalidate_reason": reason,
                    "error": str(exc),
                }
        else:
            logger(
                "[order_policy] STALE PENDING OPEN CLEARED "
                f"side={pending_side} symbol={symbol} order_id={order_id or 'n/a'} reason={reason}"
            )

        sync_result = self.sync_from_broker(logger=logger) if self.cfg.submit_orders else None
        self._pending_broker_reconcile = None
        return {
            "event": "stale_pending_open_canceled",
            "reason": reason,
            "pending_side": pending_side,
            "pending_symbol": symbol,
            "pending_order_id": order_id,
            "sync_result": sync_result,
        }

    def reconcile_pending_broker_order(
        self,
        *,
        logger: Callable[[str], None] = print,
    ) -> dict[str, Any] | None:
        pending = self._pending_broker_reconcile
        if not isinstance(pending, dict):
            return None
        if not self.cfg.submit_orders:
            self._pending_broker_reconcile = None
            return {"resolved": True, "timed_out": False, "simulated": True}

        symbol = str(pending.get("symbol", "")).strip().upper()
        intent = str(pending.get("intent", "")).strip().lower()
        order_id = str(pending.get("order_id", "")).strip() or None
        deadline = float(pending.get("deadline_monotonic", 0.0) or 0.0)

        sync_result = self.sync_from_broker(logger=logger)
        resolved = False
        try:
            has_pos = self._has_open_long_position(symbol=symbol)
            if intent == "open":
                resolved = bool(has_pos)
            elif intent == "close":
                resolved = not bool(has_pos)
        except Exception:
            resolved = False

        if resolved:
            logger(
                "[order_policy] ORDER RECONCILED "
                f"intent={intent} symbol={symbol} order_id={order_id or 'n/a'}"
            )
            self._pending_broker_reconcile = None
            return {"resolved": True, "timed_out": False, "sync_result": sync_result}

        now_mono = float(time_mod.monotonic())
        if now_mono >= deadline:
            if order_id:
                try:
                    self._client.cancel_order(order_id)
                    logger(
                        "[order_policy] ORDER CANCELED AFTER RECONCILE GRACE "
                        f"intent={intent} symbol={symbol} order_id={order_id}"
                    )
                except Exception as exc:
                    logger(
                        f"[order_policy] cancel-after-grace warning order_id={order_id}: {exc}"
                    )
            sync_result = self.sync_from_broker(logger=logger)
            self._pending_broker_reconcile = None
            return {"resolved": False, "timed_out": True, "sync_result": sync_result}

        return {"resolved": False, "timed_out": False, "sync_result": sync_result}

    def _maybe_force_close_expiring_positions(
        self,
        *,
        local_ts: datetime,
        logger: Callable[[str], None] = print,
    ) -> dict[str, Any] | None:
        cutoff = self._expiring_position_exit_cutoff
        if cutoff is None or local_ts.time() < cutoff:
            return None

        orders: list[dict[str, Any]] = []
        for side, qty, symbol in (
            ("long", int(self._long_contracts), self._long_symbol),
            ("short", int(self._short_contracts), self._short_symbol),
        ):
            if qty <= 0 or not symbol or not self._symbol_expires_today(symbol, local_ts=local_ts):
                continue
            close_resp = self._submit_order(
                symbol=symbol,
                side="sell",
                intent="close",
                qty=qty,
                logger=logger,
            )
            orders.append(
                {
                    "type": f"forced_close_{side}",
                    "side_key": side,
                    "symbol": symbol,
                    "qty": qty,
                    "response": close_resp,
                }
            )
            if side == "long":
                self._long_contracts = 0
                self._long_symbol = None
                self._long_avg_entry_price = float("nan")
            else:
                self._short_contracts = 0
                self._short_symbol = None
                self._short_avg_entry_price = float("nan")
            self._meta_side_bars_held[side] = -1
            self._meta_side_soft_exit_count[side] = 0
            self._meta_side_entry_armed[side] = False
            self._reset_trail_state(side)

        if not orders:
            return None

        self._set_entry_lockout(local_ts=local_ts, reason="expiring_position_exit_cutoff")
        self._refresh_legacy_state()
        self._pending_flat_side = 0
        self._pending_flat_bars = 0
        self._pending_opposite_side = 0
        self._pending_opposite_bars = 0
        logger(
            "[order_policy] EXPIRING POSITION CUT-OFF FLATTEN "
            f"cutoff={cutoff.strftime('%H:%M')} positions_closed={len(orders)}"
        )
        return {
            "event": "expiration_cutoff_flatten",
            "position": int(self._pos),
            "signed_contracts": int(self._signed_contracts),
            "long_contracts": int(self._long_contracts),
            "short_contracts": int(self._short_contracts),
            "open_long_symbol": self._long_symbol,
            "open_short_symbol": self._short_symbol,
            "entry_lockout_until": self._entry_lockout_until.isoformat() if self._entry_lockout_until else None,
            "orders": orders,
        }

    def _cancel_order_if_needed(
        self,
        *,
        verify_result: dict[str, Any] | None,
        logger: Callable[[str], None],
    ) -> None:
        if not self.cfg.submit_orders:
            return
        if not isinstance(verify_result, dict):
            return
        if not bool(verify_result.get("cancel_required")):
            return
        order_id = str(verify_result.get("order_id", "")).strip()
        if not order_id:
            return
        try:
            self._client.cancel_order(order_id)
            logger(f"[order_policy] ORDER CANCELED order_id={order_id} before retry")
        except Exception as exc:
            logger(f"[order_policy] cancel warning order_id={order_id}: {exc}")

    def _select_contract(
        self,
        *,
        option_type: str,
        expiration: date,
        target_strike: float,
        atr: float,
    ) -> tuple[str, float]:
        window = max(0.50, abs(float(atr)) * 2.0)
        lower = max(0.01, target_strike - window)
        upper = target_strike + window
        exp_str = expiration.isoformat()

        resp = self._client.get_option_contracts(
            underlying_symbol=self.cfg.underlying,
            expiration_date=exp_str,
            type=option_type,
            strike_price_gte=lower,
            strike_price_lte=upper,
            limit=200,
        )
        contracts = self._extract_contracts(resp)
        if not contracts:
            resp = self._client.get_option_contracts(
                underlying_symbol=self.cfg.underlying,
                expiration_date=exp_str,
                type=option_type,
                limit=1000,
            )
            contracts = self._extract_contracts(resp)
        if not contracts:
            raise RuntimeError(
                f"No option contracts found for {self.cfg.underlying} {option_type} exp={exp_str}"
            )

        candidates: list[tuple[float, str, float]] = []
        for contract in contracts:
            symbol = str(contract.get("symbol", "")).strip()
            strike = _as_float(contract.get("strike_price"))
            if not symbol or not math.isfinite(strike):
                continue
            # Prefer nearest strike to target.
            score = abs(strike - target_strike)
            candidates.append((score, symbol, strike))
        if not candidates:
            raise RuntimeError(
                f"Contracts returned but none had usable symbol/strike for exp={exp_str}"
            )

        candidates.sort(key=lambda x: x[0])
        _score, symbol, strike = candidates[0]
        return symbol, strike

    def _submit_order(
        self,
        *,
        symbol: str,
        side: str,
        intent: str = "open",
        qty: int | None = None,
        logger: Callable[[str], None] = print,
    ) -> dict[str, Any]:
        intent_key = str(intent).strip().lower()
        if intent_key not in {"open", "close"}:
            raise ValueError(f"Unknown order intent: {intent}")
        side_key = str(side).strip().lower()
        if self.cfg.long_options_only:
            if intent_key == "open" and side_key != "buy":
                raise ValueError("long_options_only=True requires buy-to-open orders.")
            if intent_key == "close" and side_key != "sell":
                raise ValueError("long_options_only=True requires sell-to-close orders.")

        order_qty = int(qty if qty is not None else self.cfg.qty)
        if not self.cfg.submit_orders:
            sim_mode = "ask" if intent_key == "open" else "bid"
            sim_price = self._get_contract_price(symbol=symbol, logger=logger, mode=sim_mode)
            if not math.isfinite(sim_price) or sim_price <= 0.0:
                sim_price = 1.0
                sim_mode = "fallback"
            sim_price = max(0.01, float(sim_price))
            payload = {
                "symbol": symbol,
                "qty": order_qty,
                "side": side_key,
                "intent": intent_key,
                "type": "limit",
                "time_in_force": self.cfg.time_in_force,
                "limit_price": round(sim_price, 4),
                "price_mode": sim_mode,
            }
            logger(f"[order_policy] SIMULATED ORDER {payload}")
            return {
                "simulated": True,
                "payload": payload,
                "response": {
                    "status": "filled",
                    "filled_qty": str(order_qty),
                    "filled_avg_price": str(round(sim_price, 4)),
                },
            }
        # Price ladder policy:
        # - opens use the configured entry quote mode and chase slowly
        # - closes use the configured exit quote mode, then move quickly toward/through bid
        valid_modes = {"ask", "mid", "bid", "last", "mark"}
        configured_mode = self.cfg.price_mode if intent_key == "open" else self.cfg.option_exit_quote_mode
        base_mode = str(configured_mode or "mid").strip().lower()
        if base_mode not in valid_modes:
            base_mode = "mid"
        base_limit = self._get_contract_price(symbol=symbol, logger=logger, mode=base_mode)
        if not math.isfinite(base_limit) or base_limit <= 0.0:
            fallback_mode = "ask" if intent_key == "open" else "bid"
            base_limit = self._get_contract_price(symbol=symbol, logger=logger, mode=fallback_mode)
            base_mode = fallback_mode
        if not self.cfg.submit_orders and (not math.isfinite(base_limit) or base_limit <= 0.0):
            base_limit = 1.0
            base_mode = "sim"
        if not math.isfinite(base_limit) or base_limit <= 0.0:
            raise RuntimeError(
                f"no_quote_for_limit_pricing intent={intent_key} symbol={symbol}"
            )
        logger(
            "[order_policy] ORDER PRICING "
            f"intent={intent_key} symbol={symbol} source={base_mode} base_limit={base_limit:.2f}"
        )
        tick = 0.01
        close_bid = float("nan")
        if intent_key == "close":
            close_bid = self._get_contract_price(symbol=symbol, logger=logger, mode="bid")

        max_attempts = max(0, int(self.cfg.max_resubmit_attempts))
        attempts = max_attempts + 1
        last_verify: dict[str, Any] | None = None
        for attempt in range(1, attempts + 1):
            offset = (attempt - 1) * tick
            if intent_key == "open":
                limit_price = base_limit + offset
            elif math.isfinite(close_bid) and close_bid > 0.0:
                limit_price = base_limit if attempt == 1 else close_bid - (attempt - 2) * tick
            else:
                limit_price = base_limit - (attempt - 1) * tick * 2.0
            limit_price = round(float(limit_price), 2)
            limit_price = max(tick, limit_price)

            resp = self._client.submit_option_order(
                symbol=symbol,
                qty=order_qty,
                side=side_key,
                order_type="limit",
                time_in_force=self.cfg.time_in_force,
                limit_price=limit_price,
            )
            status = self._status_key(resp.get("status") if isinstance(resp, dict) else None)
            oid = str(resp.get("id", "")).strip() if isinstance(resp, dict) else ""
            logger(
                "[order_policy] ORDER SUBMITTED "
                f"intent={intent_key} side={side_key} qty={order_qty} symbol={symbol} "
                f"order_id={oid or 'n/a'} status={status or 'n/a'} "
                f"limit_price={limit_price:.2f} attempt={attempt}/{attempts}"
            )

            verify_result: dict[str, Any] | None = None
            if self.cfg.verify_submitted_orders:
                verify_result = self._verify_order_submission(
                    submitted_resp=resp if isinstance(resp, dict) else {},
                    symbol=symbol,
                    intent=intent_key,
                    logger=logger,
                )
                last_verify = verify_result
                if verify_result.get("verified"):
                    logger(
                        "[order_policy] ORDER VERIFIED "
                        f"intent={intent_key} symbol={symbol} "
                        f"status={verify_result.get('status')} via={verify_result.get('via')}"
                    )
                    return {
                        "simulated": False,
                        "intent": intent_key,
                        "response": resp,
                        "verification": verify_result,
                        "side": side_key,
                        "qty": order_qty,
                        "symbol": symbol,
                    }

                retryable = bool(verify_result.get("retryable"))
                can_retry = (
                    self.cfg.resubmit_on_terminal_fail
                    and retryable
                    and attempt < attempts
                )
                logger(
                    "[order_policy] ORDER NOT VERIFIED "
                    f"intent={intent_key} symbol={symbol} "
                    f"status={verify_result.get('status')} via={verify_result.get('via')} "
                    f"retrying={can_retry}"
                )
                if can_retry:
                    if intent_key == "open":
                        next_limit = round(float(limit_price + tick), 2)
                    elif math.isfinite(close_bid) and close_bid > 0.0:
                        next_limit = round(float(max(tick, close_bid - max(0, attempt - 1) * tick)), 2)
                    else:
                        next_limit = round(float(max(tick, limit_price - tick * 2.0)), 2)
                    logger(
                        "[order_policy] ORDER RETRY "
                        f"intent={intent_key} symbol={symbol} reason={verify_result.get('via')} "
                        f"status={verify_result.get('status')} last_limit={limit_price:.2f} "
                        f"next_limit={next_limit:.2f}"
                    )
                    self._cancel_order_if_needed(verify_result=verify_result, logger=logger)
                    continue

                if (
                    str(verify_result.get("via", "")).strip().lower() == "timeout"
                    and bool(verify_result.get("order_id"))
                ):
                    self._mark_pending_broker_reconcile(
                        symbol=symbol,
                        intent=intent_key,
                        side=side_key,
                        qty=order_qty,
                        verify_result=verify_result,
                        logger=logger,
                    )
                    return {
                        "simulated": False,
                        "intent": intent_key,
                        "response": resp,
                        "verification": verify_result,
                        "side": side_key,
                        "qty": order_qty,
                        "symbol": symbol,
                        "pending_broker_reconcile": True,
                    }

                raise RuntimeError(
                    "order_not_verified:"
                    f" intent={intent_key} symbol={symbol} status={verify_result.get('status')} "
                    f"via={verify_result.get('via')} order_id={verify_result.get('order_id')}"
                )

            return {
                "simulated": False,
                "intent": intent_key,
                "response": resp,
                "side": side_key,
                "qty": order_qty,
                "symbol": symbol,
            }

        raise RuntimeError(f"order_submit_failed intent={intent_key} symbol={symbol} verify={last_verify}")

    def snapshot_state(self) -> dict[str, Any]:
        """
        Return a lightweight policy state snapshot for monitoring UIs.
        """
        atr = self._compute_atr()
        return {
            "underlying": self.cfg.underlying,
            "position": int(self._pos),
            "signed_contracts": int(self._signed_contracts),
            "open_symbol": self._open_symbol,
            "long_contracts": int(self._long_contracts),
            "short_contracts": int(self._short_contracts),
            "long_bars_held": int(self._meta_side_bars_held.get("long", -1)),
            "short_bars_held": int(self._meta_side_bars_held.get("short", -1)),
            "long_entry_signal_time": (
                self._meta_side_entry_signal_ts.get("long").isoformat()
                if isinstance(self._meta_side_entry_signal_ts.get("long"), datetime)
                else None
            ),
            "short_entry_signal_time": (
                self._meta_side_entry_signal_ts.get("short").isoformat()
                if isinstance(self._meta_side_entry_signal_ts.get("short"), datetime)
                else None
            ),
            "open_long_symbol": self._long_symbol,
            "open_short_symbol": self._short_symbol,
            "avg_entry_price_long": float(self._long_avg_entry_price) if math.isfinite(self._long_avg_entry_price) else None,
            "avg_entry_price_short": float(self._short_avg_entry_price) if math.isfinite(self._short_avg_entry_price) else None,
            "option_exit_policy": str(self.cfg.option_exit_policy),
            "option_exit_best_price_long": (
                float(self._option_exit_best_price.get("long", float("nan")))
                if math.isfinite(_as_float(self._option_exit_best_price.get("long")))
                else None
            ),
            "option_exit_best_price_short": (
                float(self._option_exit_best_price.get("short", float("nan")))
                if math.isfinite(_as_float(self._option_exit_best_price.get("short")))
                else None
            ),
            "option_exit_trailing_arm_pct": float(self.cfg.option_exit_trailing_arm_pct),
            "option_exit_trailing_giveback_pct": float(self.cfg.option_exit_trailing_giveback_pct),
            "option_exit_no_progress_minutes": int(self.cfg.option_exit_no_progress_minutes),
            "option_exit_no_progress_mfe_pct": float(self.cfg.option_exit_no_progress_mfe_pct),
            "option_exit_time_decay_minutes": int(self.cfg.option_exit_time_decay_minutes),
            "option_exit_time_decay_progress_pct": float(self.cfg.option_exit_time_decay_progress_pct),
            "option_exit_opposite_prob": float(self.cfg.option_exit_opposite_prob),
            "option_exit_opposite_prob_long": (
                None
                if self.cfg.option_exit_opposite_prob_long is None
                else float(self.cfg.option_exit_opposite_prob_long)
            ),
            "option_exit_opposite_prob_short": (
                None
                if self.cfg.option_exit_opposite_prob_short is None
                else float(self.cfg.option_exit_opposite_prob_short)
            ),
            "option_exit_opposite_profit_pct": float(self.cfg.option_exit_opposite_profit_pct),
            "option_exit_quote_mode": str(self.cfg.option_exit_quote_mode),
            "max_live_entry_lag_sec": float(self.cfg.max_live_entry_lag_sec),
            "atr": float(atr) if math.isfinite(atr) else None,
            "bars_interval": int(len(self._bars_interval)),
            "bars_15m": int(len(self._bars_interval)),
            "submit_orders": bool(self.cfg.submit_orders),
            "qty": int(self.cfg.qty),
            "price_mode": str(self.cfg.price_mode),
            "action_ema": float(self._action_ema) if self._action_ema is not None and math.isfinite(self._action_ema) else None,
            "action_effective": float(self._action_effective),
            "pending_flat_side": int(self._pending_flat_side),
            "pending_flat_bars": int(self._pending_flat_bars),
            "same_side_reentry_grace_bars": int(self.cfg.same_side_reentry_grace_bars),
            "pending_opposite_side": int(self._pending_opposite_side),
            "pending_opposite_bars": int(self._pending_opposite_bars),
            "opposite_confirm_bars": int(self.cfg.opposite_confirm_bars),
            "opposite_min_abs_action": float(self.cfg.opposite_min_abs_action),
            "opposite_min_prob_edge": float(self.cfg.opposite_min_prob_edge),
            "long_soft_exit_count": int(self._meta_side_soft_exit_count.get("long", 0)),
            "short_soft_exit_count": int(self._meta_side_soft_exit_count.get("short", 0)),
            "meta_intrabar_entry_policy": self._intrabar_entry_policy_name(),
            "meta_intrabar_max_confirmation_age_minutes": int(self.cfg.meta_intrabar_max_confirmation_age_minutes),
            "meta_intrabar_ref_chase_atr": float(self.cfg.meta_intrabar_ref_chase_atr),
            "meta_intrabar_long_setup_threshold": (
                float(self.cfg.meta_intrabar_long_setup_threshold)
                if self.cfg.meta_intrabar_long_setup_threshold is not None
                else None
            ),
            "meta_intrabar_short_setup_threshold": (
                float(self.cfg.meta_intrabar_short_setup_threshold)
                if self.cfg.meta_intrabar_short_setup_threshold is not None
                else None
            ),
            "meta_intrabar_setup_max_bars": int(self.cfg.meta_intrabar_setup_max_bars),
            "scalp_mode_enabled": bool(self.cfg.scalp_mode_enabled),
            "scalp_long_setup_threshold": float(self.cfg.scalp_long_setup_threshold),
            "scalp_short_setup_threshold": float(self.cfg.scalp_short_setup_threshold),
            "scalp_setup_max_bars": int(self.cfg.scalp_setup_max_bars),
            "scalp_min_signal_range_atr": float(self.cfg.scalp_min_signal_range_atr),
            "scalp_require_reversal_close": bool(self.cfg.scalp_require_reversal_close),
            "long_intrabar_setup_peak_prob": (
                float(self._meta_side_enter_peak_prob.get("long", float("nan")))
                if math.isfinite(_as_float(self._meta_side_enter_peak_prob.get("long")))
                else None
            ),
            "short_intrabar_setup_peak_prob": (
                float(self._meta_side_enter_peak_prob.get("short", float("nan")))
                if math.isfinite(_as_float(self._meta_side_enter_peak_prob.get("short")))
                else None
            ),
            "long_intrabar_intent_active": bool(self._meta_intrabar_entry_intent_active.get("long", False)),
            "short_intrabar_intent_active": bool(self._meta_intrabar_entry_intent_active.get("short", False)),
            "long_intrabar_entry_ref": (
                float(self._meta_intrabar_entry_ref.get("long", float("nan")))
                if math.isfinite(_as_float(self._meta_intrabar_entry_ref.get("long")))
                else None
            ),
            "short_intrabar_entry_ref": (
                float(self._meta_intrabar_entry_ref.get("short", float("nan")))
                if math.isfinite(_as_float(self._meta_intrabar_entry_ref.get("short")))
                else None
            ),
            "long_intrabar_setup_expires_at": (
                self._meta_intrabar_entry_expires_at["long"].isoformat()
                if isinstance(self._meta_intrabar_entry_expires_at.get("long"), datetime)
                else None
            ),
            "short_intrabar_setup_expires_at": (
                self._meta_intrabar_entry_expires_at["short"].isoformat()
                if isinstance(self._meta_intrabar_entry_expires_at.get("short"), datetime)
                else None
            ),
            "opening_pullback_required": dict(self._opening_pullback_required),
            "opening_pullback_seen": dict(self._opening_pullback_seen),
            "long_decision_reason": self._meta_side_reason.get("long"),
            "short_decision_reason": self._meta_side_reason.get("short"),
            "meta_setup_failure_exit_enabled": bool(self.cfg.meta_setup_failure_exit_enabled),
            "meta_setup_failure_buffer_atr": float(self.cfg.meta_setup_failure_buffer_atr),
            "meta_no_progress_exit_enabled": bool(self.cfg.meta_no_progress_exit_enabled),
            "meta_no_progress_exit_minutes": int(self.cfg.meta_no_progress_exit_minutes),
            "meta_no_progress_exit_atr": float(self.cfg.meta_no_progress_exit_atr),
            "long_entry_invalidation_level": (
                _as_float((self._meta_entry_structure.get("long") or {}).get("signal_low"))
                if math.isfinite(_as_float((self._meta_entry_structure.get("long") or {}).get("signal_low")))
                else None
            ),
            "short_entry_invalidation_level": (
                _as_float((self._meta_entry_structure.get("short") or {}).get("signal_high"))
                if math.isfinite(_as_float((self._meta_entry_structure.get("short") or {}).get("signal_high")))
                else None
            ),
            "pending_broker_reconcile": bool(self._pending_broker_reconcile),
            "pending_broker_reconcile_order_id": (
                self._pending_broker_reconcile.get("order_id")
                if isinstance(self._pending_broker_reconcile, dict)
                else None
            ),
            "entry_lockout_until": self._entry_lockout_until.isoformat() if self._entry_lockout_until else None,
            "entry_lockout_reason": self._entry_lockout_reason,
            "missed_trade_diagnostics": self._missed_trade_diagnostics(),
        }

    def export_runtime_state(self) -> dict[str, Any]:
        return {
            "latest_meta_side_snapshot": self._latest_meta_side_snapshot,
            "meta_side_soft_exit_count": dict(self._meta_side_soft_exit_count),
            "meta_intrabar_entry_intent_active": dict(self._meta_intrabar_entry_intent_active),
            "meta_intrabar_entry_ref": dict(self._meta_intrabar_entry_ref),
            "meta_intrabar_entry_setup_ts": {
                key: (_ts.isoformat() if isinstance(_ts, datetime) else None)
                for key, _ts in self._meta_intrabar_entry_setup_ts.items()
            },
            "meta_intrabar_entry_expires_at": {
                key: (_ts.isoformat() if isinstance(_ts, datetime) else None)
                for key, _ts in self._meta_intrabar_entry_expires_at.items()
            },
            "meta_intrabar_setup_snapshot": dict(self._meta_intrabar_setup_snapshot),
            "opening_pullback_required": dict(self._opening_pullback_required),
            "opening_pullback_seen": dict(self._opening_pullback_seen),
            "meta_entry_structure": {
                key: (
                    {
                        sub_key: (
                            sub_value.isoformat()
                            if isinstance(sub_value, datetime)
                            else sub_value
                        )
                        for sub_key, sub_value in value.items()
                    }
                    if isinstance(value, dict)
                    else None
                )
                for key, value in self._meta_entry_structure.items()
            },
            "meta_side_reason": dict(self._meta_side_reason),
            "meta_side_entry_armed": dict(self._meta_side_entry_armed),
            "meta_side_enter_above_threshold": dict(self._meta_side_enter_above_threshold),
            "meta_side_enter_peak_prob": dict(self._meta_side_enter_peak_prob),
            "meta_side_bars_held": dict(self._meta_side_bars_held),
            "meta_side_entry_signal_ts": {
                key: (_ts.isoformat() if isinstance(_ts, datetime) else None)
                for key, _ts in self._meta_side_entry_signal_ts.items()
            },
            "meta_side_cooldown_until": {
                key: (_ts.isoformat() if isinstance(_ts, datetime) else None)
                for key, _ts in self._meta_side_cooldown_until.items()
            },
            "prev_1m_close": float(self._prev_1m_close) if math.isfinite(self._prev_1m_close) else None,
            "last_1m_close": float(self._last_1m_close) if math.isfinite(self._last_1m_close) else None,
            "recent_1m_closes": [
                float(x) for x in list(self._recent_1m_closes) if math.isfinite(_as_float(x))
            ],
            "option_exit_best_price": {
                key: (float(value) if math.isfinite(_as_float(value)) else None)
                for key, value in self._option_exit_best_price.items()
            },
            "option_exit_best_seen_at": {
                key: (_ts.isoformat() if isinstance(_ts, datetime) else None)
                for key, _ts in self._option_exit_best_seen_at.items()
            },
            "entry_lockout_until": self._entry_lockout_until.isoformat() if self._entry_lockout_until else None,
            "entry_lockout_reason": self._entry_lockout_reason,
        }

    def load_runtime_state(
        self,
        state: dict[str, Any] | None,
        *,
        logger: Callable[[str], None] = print,
    ) -> bool:
        if not isinstance(state, dict):
            return False
        try:
            snap = state.get("latest_meta_side_snapshot")
            if isinstance(snap, dict):
                self._latest_meta_side_snapshot = dict(snap)

            soft_counts = state.get("meta_side_soft_exit_count")
            if isinstance(soft_counts, dict):
                for side in ("long", "short"):
                    self._meta_side_soft_exit_count[side] = max(0, int(soft_counts.get(side, 0)))

            intent_active = state.get("meta_intrabar_entry_intent_active")
            if isinstance(intent_active, dict):
                for side in ("long", "short"):
                    self._meta_intrabar_entry_intent_active[side] = bool(intent_active.get(side, False))

            entry_ref = state.get("meta_intrabar_entry_ref")
            if isinstance(entry_ref, dict):
                for side in ("long", "short"):
                    ref_val = _as_float(entry_ref.get(side))
                    self._meta_intrabar_entry_ref[side] = ref_val if math.isfinite(ref_val) else float("nan")

            setup_snapshot = state.get("meta_intrabar_setup_snapshot")
            if isinstance(setup_snapshot, dict):
                for side in ("long", "short"):
                    value = setup_snapshot.get(side)
                    self._meta_intrabar_setup_snapshot[side] = dict(value) if isinstance(value, dict) else None

            pullback_required = state.get("opening_pullback_required")
            if isinstance(pullback_required, dict):
                for side in ("long", "short"):
                    self._opening_pullback_required[side] = bool(pullback_required.get(side, False))

            pullback_seen = state.get("opening_pullback_seen")
            if isinstance(pullback_seen, dict):
                for side in ("long", "short"):
                    self._opening_pullback_seen[side] = bool(pullback_seen.get(side, False))

            entry_structure = state.get("meta_entry_structure")
            if isinstance(entry_structure, dict):
                for side in ("long", "short"):
                    value = entry_structure.get(side)
                    if not isinstance(value, dict):
                        self._meta_entry_structure[side] = None
                        continue
                    parsed_value: dict[str, Any] = {}
                    for sub_key, sub_value in value.items():
                        if sub_key.endswith("_ts") or sub_key == "expires_at":
                            if sub_value in (None, ""):
                                parsed_value[sub_key] = None
                            else:
                                parsed = pd.to_datetime(sub_value, utc=True, errors="coerce")
                                parsed_value[sub_key] = (
                                    None if pd.isna(parsed) else parsed.to_pydatetime().astimezone(self._tz)
                                )
                        else:
                            parsed_value[sub_key] = sub_value
                    self._meta_entry_structure[side] = parsed_value

            enter_above = state.get("meta_side_enter_above_threshold")
            if isinstance(enter_above, dict):
                for side in ("long", "short"):
                    self._meta_side_enter_above_threshold[side] = bool(enter_above.get(side, False))

            enter_peak = state.get("meta_side_enter_peak_prob")
            if isinstance(enter_peak, dict):
                for side in ("long", "short"):
                    peak_val = _as_float(enter_peak.get(side))
                    self._meta_side_enter_peak_prob[side] = peak_val if math.isfinite(peak_val) else float("nan")

            for state_key, target in (
                ("meta_intrabar_entry_setup_ts", self._meta_intrabar_entry_setup_ts),
                ("meta_intrabar_entry_expires_at", self._meta_intrabar_entry_expires_at),
            ):
                saved_ts = state.get(state_key)
                if isinstance(saved_ts, dict):
                    for side in ("long", "short"):
                        raw = saved_ts.get(side)
                        if raw in (None, ""):
                            target[side] = None
                            continue
                        parsed = pd.to_datetime(raw, utc=True, errors="coerce")
                        target[side] = None if pd.isna(parsed) else parsed.to_pydatetime().astimezone(self._tz)

            reasons = state.get("meta_side_reason")
            if isinstance(reasons, dict):
                for side in ("long", "short"):
                    raw = reasons.get(side)
                    self._meta_side_reason[side] = str(raw) if raw not in (None, "") else None

            entry_armed = state.get("meta_side_entry_armed")
            if isinstance(entry_armed, dict):
                for side in ("long", "short"):
                    self._meta_side_entry_armed[side] = bool(entry_armed.get(side, self._meta_side_entry_armed.get(side, True)))

            bars_held = state.get("meta_side_bars_held")
            if isinstance(bars_held, dict):
                for side in ("long", "short"):
                    saved = int(bars_held.get(side, self._meta_side_bars_held.get(side, -1)))
                    self._meta_side_bars_held[side] = max(int(self._meta_side_bars_held.get(side, -1)), saved)

            entry_signal_ts = state.get("meta_side_entry_signal_ts")
            if isinstance(entry_signal_ts, dict):
                for side in ("long", "short"):
                    raw = entry_signal_ts.get(side)
                    if raw in (None, ""):
                        self._meta_side_entry_signal_ts[side] = None
                        continue
                    parsed = datetime.fromisoformat(str(raw))
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=self._tz)
                    self._meta_side_entry_signal_ts[side] = parsed.astimezone(self._tz)

            cooldown_until = state.get("meta_side_cooldown_until")
            if isinstance(cooldown_until, dict):
                for side in ("long", "short"):
                    raw = cooldown_until.get(side)
                    if raw in (None, ""):
                        self._meta_side_cooldown_until[side] = None
                        continue
                    parsed = pd.to_datetime(raw, utc=True, errors="coerce")
                    self._meta_side_cooldown_until[side] = None if pd.isna(parsed) else parsed.to_pydatetime().astimezone(self._tz)

            prev_close = _as_float(state.get("prev_1m_close"))
            self._prev_1m_close = prev_close if math.isfinite(prev_close) else float("nan")
            last_close = _as_float(state.get("last_1m_close"))
            self._last_1m_close = last_close if math.isfinite(last_close) else float("nan")
            
            recent_closes = state.get("recent_1m_closes")
            if isinstance(recent_closes, list):
                self._recent_1m_closes.clear()
                maxlen = self._recent_1m_closes.maxlen or len(recent_closes)
                for value in recent_closes[-maxlen:]:
                    v = _as_float(value)
                    if math.isfinite(v):
                        self._recent_1m_closes.append(float(v))
            option_best = state.get("option_exit_best_price")
            if isinstance(option_best, dict):
                for side in ("long", "short"):
                    best = _as_float(option_best.get(side))
                    self._option_exit_best_price[side] = best if math.isfinite(best) else float("nan")
            option_best_seen = state.get("option_exit_best_seen_at")
            if isinstance(option_best_seen, dict):
                for side in ("long", "short"):
                    raw = option_best_seen.get(side)
                    if raw in (None, ""):
                        self._option_exit_best_seen_at[side] = None
                        continue
                    parsed = pd.to_datetime(raw, utc=True, errors="coerce")
                    self._option_exit_best_seen_at[side] = None if pd.isna(parsed) else parsed.to_pydatetime().astimezone(self._tz)
            raw_lockout_until = state.get("entry_lockout_until")
            if raw_lockout_until in (None, ""):
                self._entry_lockout_until = None
            else:
                parsed = datetime.fromisoformat(str(raw_lockout_until))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=self._tz)
                self._entry_lockout_until = parsed.astimezone(self._tz)
            raw_lockout_reason = state.get("entry_lockout_reason")
            self._entry_lockout_reason = str(raw_lockout_reason) if raw_lockout_reason not in (None, "") else None
            logger("[order_policy] Loaded runtime policy state cache.")
            return True
        except Exception as exc:
            logger(f"[order_policy] Failed to load runtime policy state cache: {exc}")
            return False

    def snapshot_broker_state(self, *, orders_limit: int = 20) -> dict[str, Any]:
        """
        Pull current broker-side open positions and recent orders for this underlying.
        """
        under = self.cfg.underlying.strip().upper()
        out: dict[str, Any] = {
            "underlying": under,
            "positions": [],
            "recent_orders": [],
            "ok": True,
        }
        if not self.cfg.submit_orders:
            if self._long_symbol and self._long_contracts > 0:
                out["positions"].append(
                    {
                        "symbol": self._long_symbol,
                        "side": "long",
                        "qty": float(self._long_contracts),
                        "avg_entry_price": (
                            float(_as_float(self._long_avg_entry_price))
                            if math.isfinite(_as_float(self._long_avg_entry_price))
                            else None
                        ),
                        "market_value": None,
                        "unrealized_pl": None,
                    }
                )
            if self._short_symbol and self._short_contracts > 0:
                out["positions"].append(
                    {
                        "symbol": self._short_symbol,
                        "side": "long",
                        "qty": float(self._short_contracts),
                        "avg_entry_price": (
                            float(_as_float(self._short_avg_entry_price))
                            if math.isfinite(_as_float(self._short_avg_entry_price))
                            else None
                        ),
                        "market_value": None,
                        "unrealized_pl": None,
                    }
                )
            out["simulation"] = True
            return out
        try:
            pos_resp = self._client.get_positions()
            positions = self._extract_positions(pos_resp)
            filtered_positions: list[dict[str, Any]] = []
            for p in positions:
                symbol = str(p.get("symbol", "")).strip().upper()
                if not symbol.startswith(under):
                    continue
                qty_val = _as_float(p.get("qty"))
                avg_entry = _as_float(p.get("avg_entry_price"))
                market_value = _as_float(p.get("market_value"))
                unrealized = _as_float(p.get("unrealized_pl"))
                filtered_positions.append(
                    {
                        "symbol": symbol,
                        "side": str(p.get("side", "")).strip().lower() or None,
                        "qty": qty_val if math.isfinite(qty_val) else None,
                        "avg_entry_price": avg_entry if math.isfinite(avg_entry) else None,
                        "market_value": market_value if math.isfinite(market_value) else None,
                        "unrealized_pl": unrealized if math.isfinite(unrealized) else None,
                    }
                )
            out["positions"] = filtered_positions
        except Exception as exc:
            out["ok"] = False
            out["positions_error"] = str(exc)

        try:
            limit = max(1, int(orders_limit))
            ord_resp = self._client.get_orders(status="all", limit=limit, direction="desc")
            orders: list[dict[str, Any]] = []
            if isinstance(ord_resp, list):
                raw_orders = [x for x in ord_resp if isinstance(x, dict)]
            elif isinstance(ord_resp, dict):
                data = ord_resp.get("orders")
                raw_orders = [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []
            else:
                raw_orders = []
            for o in raw_orders:
                symbol = str(o.get("symbol", "")).strip().upper()
                if not symbol.startswith(under):
                    continue
                orders.append(
                    {
                        "id": str(o.get("id", "")).strip() or None,
                        "symbol": symbol,
                        "side": str(o.get("side", "")).strip().lower() or None,
                        "status": str(o.get("status", "")).strip().lower() or None,
                        "qty": str(o.get("qty", "")).strip() or None,
                        "filled_qty": str(o.get("filled_qty", "")).strip() or None,
                        "filled_avg_price": str(o.get("filled_avg_price", "")).strip() or None,
                        "submitted_at": str(o.get("submitted_at", "")).strip() or None,
                        "filled_at": str(o.get("filled_at", "")).strip() or None,
                    }
                )
            out["recent_orders"] = orders
        except Exception as exc:
            out["ok"] = False
            out["orders_error"] = str(exc)
        return out

    def _target_contracts_for_side(
        self,
        *,
        side: str,
        closed_bar: dict[str, Any],
        close: float,
        high: float,
        low: float,
        atr: float,
        prev_close: float = float("nan"),
        enforce_trend_gate: bool = False,
        local_ts: datetime | None = None,
        allow_new_entries: bool = True,
        allow_exits: bool = True,
        logger: Callable[[str], None] = print,
    ) -> int:
        side_key = str(side).strip().lower()
        if side_key not in {"long", "short"}:
            return 0
        self._meta_side_reason[side_key] = None
        current_qty = self._long_contracts if side_key == "long" else self._short_contracts
        enter_prob = _as_float(closed_bar.get(f"p_enter_{side_key}"))
        exit_prob = _as_float(closed_bar.get(f"p_exit_{side_key}"))
        enter_thr = self._effective_intrabar_setup_threshold(
            closed_bar=closed_bar,
            side=side_key,
        )
        exit_thr = self._meta_threshold_value(closed_bar, f"thr_exit_{side_key}", f"exit_{side_key}_threshold")
        desired_qty = max(0, int(self.cfg.qty))
        if bool(self.cfg.meta_replay_compatible_mode):
            bars_held = int(self._meta_side_bars_held.get(side_key, -1))
            if current_qty > 0:
                if self._ohlc_exit_bar_is_after_entry(side=side_key, local_ts=local_ts):
                    hard_stop_signal = self._hard_stop_hit(side=side_key, close=close, high=high, low=low)
                    if hard_stop_signal:
                        self._meta_side_soft_exit_count[side_key] = 0
                        self._meta_side_reason[side_key] = "hard_stop"
                        return 0
                    setup_failure_signal = self._setup_failure_exit_hit(
                        side=side_key,
                        close=close,
                        high=high,
                        low=low,
                    )
                    if setup_failure_signal:
                        self._meta_side_soft_exit_count[side_key] = 0
                        self._meta_side_reason[side_key] = "setup_failure_exit"
                        return 0
                    no_progress_signal = self._no_progress_exit_hit(
                        side=side_key,
                        close=close,
                        local_ts=local_ts,
                    )
                    if no_progress_signal:
                        self._meta_side_soft_exit_count[side_key] = 0
                        self._meta_side_reason[side_key] = "no_progress_exit"
                        return 0
                option_exit_signal = self._option_value_exit_hit(
                    side=side_key,
                    logger=logger,
                    closed_bar=closed_bar,
                    local_ts=local_ts,
                )
                if option_exit_signal:
                    self._meta_side_soft_exit_count[side_key] = 0
                    return 0
                if self._option_value_exit_policy_enabled():
                    self._meta_side_reason[side_key] = "hold_option_value_bracket"
                    self._meta_side_soft_exit_count[side_key] = 0
                    return desired_qty
                if not bool(allow_exits):
                    self._meta_side_reason[side_key] = "exit_waiting_for_1m_confirmation"
                    return desired_qty
                if self._exit_waiting_for_next_snapshot(side=side_key):
                    self._meta_side_reason[side_key] = "waiting_next_10m_exit_signal"
                    return desired_qty
                hold_ready = bool(bars_held >= max(0, int(self.cfg.meta_min_hold_bars)))
                soft_exit_condition = bool(
                    hold_ready
                    and math.isfinite(enter_prob)
                    and math.isfinite(enter_thr)
                    and enter_prob < enter_thr
                )
                if soft_exit_condition:
                    self._meta_side_soft_exit_count[side_key] = int(self._meta_side_soft_exit_count.get(side_key, 0)) + 1
                else:
                    self._meta_side_soft_exit_count[side_key] = 0
                urgent_exit_prob_hit = bool(
                    hold_ready
                    and math.isfinite(exit_prob)
                    and exit_prob >= float(self.cfg.meta_urgent_exit_prob)
                )
                urgent_exit_delta_hit = bool(
                    hold_ready
                    and math.isfinite(exit_prob)
                    and math.isfinite(enter_prob)
                    and (exit_prob - enter_prob) >= float(self.cfg.meta_urgent_exit_delta)
                )
                profit_protect_hit = bool(
                    self._ohlc_exit_bar_is_after_entry(side=side_key, local_ts=local_ts)
                    and self._profit_protect_exit_hit(side=side_key, close=close)
                )
                urgent_exit = bool(
                    hold_ready
                    and (
                        urgent_exit_prob_hit
                        or urgent_exit_delta_hit
                        or profit_protect_hit
                    )
                )
                soft_exit_confirmed = bool(
                    int(self._meta_side_soft_exit_count.get(side_key, 0))
                    >= max(1, int(self.cfg.meta_soft_exit_confirm_bars))
                )
                if urgent_exit_prob_hit:
                    self._meta_side_reason[side_key] = "urgent_exit_prob"
                elif urgent_exit_delta_hit:
                    self._meta_side_reason[side_key] = "urgent_exit_delta"
                elif profit_protect_hit:
                    self._meta_side_reason[side_key] = "profit_protect"
                elif soft_exit_confirmed:
                    self._meta_side_reason[side_key] = "soft_exit_confirmed"
                elif soft_exit_condition:
                    self._meta_side_reason[side_key] = (
                        f"soft_exit_pending_{int(self._meta_side_soft_exit_count.get(side_key, 0))}"
                    )
                elif not hold_ready:
                    self._meta_side_reason[side_key] = "min_hold_not_ready"
                else:
                    self._meta_side_reason[side_key] = "hold_same_side_enter_still_valid"
                return 0 if (urgent_exit or soft_exit_confirmed) else desired_qty
            self._meta_side_soft_exit_count[side_key] = 0
            if not bool(allow_new_entries):
                self._meta_side_reason[side_key] = "entry_waiting_for_1m_confirmation"
                return 0
            if not self._new_entries_allowed(local_ts=local_ts):
                cutoff = self._new_entry_cutoff.strftime("%H:%M") if self._new_entry_cutoff else "n/a"
                self._meta_side_reason[side_key] = f"new_entry_cutoff_{cutoff}"
                return 0
            if self._stale_prior_session_entry_signal(local_ts=local_ts):
                self._meta_side_reason[side_key] = "waiting_same_day_10m_confirmation"
                return 0
            if self._entry_lockout_active(local_ts=local_ts):
                self._meta_side_reason[side_key] = self._entry_lockout_reason or "entry_lockout_active"
                return 0
            if math.isfinite(enter_prob) and math.isfinite(enter_thr) and enter_prob >= enter_thr:
                self._meta_side_reason[side_key] = "enter_threshold_met"
                return desired_qty
            self._meta_side_reason[side_key] = "below_entry_threshold"
            return 0
        if current_qty > 0:
            state = self._trail_state(side_key)
            if not state.active:
                self._seed_trail_state(
                    side=side_key,
                    close=close,
                    atr=atr,
                    high=high,
                    low=low,
                    local_ts=local_ts,
                )
            else:
                self._update_trail_state(
                    side=side_key,
                    close=close,
                    high=high,
                    low=low,
                    local_ts=local_ts,
                )
        else:
            self._reset_trail_state(side_key)
        if current_qty > 0:
            if self._ohlc_exit_bar_is_after_entry(side=side_key, local_ts=local_ts):
                hard_stop_signal = self._hard_stop_hit(side=side_key, close=close, high=high, low=low)
                if hard_stop_signal:
                    self._meta_side_reason[side_key] = "hard_stop_exit"
                    return 0
                setup_failure_signal = self._setup_failure_exit_hit(
                    side=side_key,
                    close=close,
                    high=high,
                    low=low,
                )
                if setup_failure_signal:
                    self._meta_side_reason[side_key] = "setup_failure_exit"
                    return 0
                no_progress_signal = self._no_progress_exit_hit(
                    side=side_key,
                    close=close,
                    local_ts=local_ts,
                )
                if no_progress_signal:
                    self._meta_side_reason[side_key] = "no_progress_exit"
                    return 0
            if self._option_value_exit_hit(
                side=side_key,
                logger=logger,
                closed_bar=closed_bar,
                local_ts=local_ts,
            ):
                return 0
            if self._option_value_exit_policy_enabled():
                self._meta_side_reason[side_key] = "hold_option_value_bracket"
                return desired_qty
            if not bool(allow_exits):
                self._meta_side_reason[side_key] = "exit_waiting_for_1m_confirmation"
                return desired_qty
            if self._exit_waiting_for_next_snapshot(side=side_key):
                self._meta_side_reason[side_key] = "waiting_next_10m_exit_signal"
                return desired_qty
            has_meta_exit_prob = (
                math.isfinite(exit_prob)
                and math.isfinite(exit_thr)
            )
            exit_signal = bool(has_meta_exit_prob and exit_prob >= exit_thr)
            fallback_exit_signal = False
            if not has_meta_exit_prob:
                if self._ohlc_exit_bar_is_after_entry(side=side_key, local_ts=local_ts):
                    stale_signal = self._stale_trade_exit_hit(side=side_key, close=close, local_ts=local_ts)
                    trail_signal = self._trail_stop_hit(side=side_key, close=close, high=high, low=low)
                    fallback_exit_signal = bool(stale_signal or trail_signal)
            if not (exit_signal or fallback_exit_signal):
                return desired_qty
            if enforce_trend_gate and not self._trend_gate_passed(
                side=side_key,
                transition="exit",
                close=close,
                prev_close=prev_close,
                recent_closes=list(self._recent_1m_closes),
            ):
                return desired_qty
            return 0
        if math.isfinite(enter_prob) and math.isfinite(enter_thr) and enter_prob >= enter_thr:
            if not self._new_entries_allowed(local_ts=local_ts):
                cutoff = self._new_entry_cutoff.strftime("%H:%M") if self._new_entry_cutoff else "n/a"
                self._meta_side_reason[side_key] = f"new_entry_cutoff_{cutoff}"
                return 0
            if not bool(self._meta_side_entry_armed.get(side_key, True)):
                return 0
            if self._stale_prior_session_entry_signal(local_ts=local_ts):
                self._meta_side_reason[side_key] = "waiting_same_day_10m_confirmation"
                return 0
            if self._entry_lockout_active(local_ts=local_ts):
                self._meta_side_reason[side_key] = self._entry_lockout_reason or "entry_lockout_active"
                return 0
            if self._side_reentry_cooldown_active(side=side_key, local_ts=local_ts):
                return 0
            if self._entry_no_chase_blocked(side=side_key, close=close, atr=atr, local_ts=local_ts):
                return 0
            if enforce_trend_gate and not self._trend_gate_passed(
                side=side_key,
                transition="enter",
                close=close,
                prev_close=prev_close,
                recent_closes=list(self._recent_1m_closes),
            ):
                return 0
            return desired_qty
        return 0

    def _select_side_contract(
        self,
        *,
        side: str,
        close: float,
        atr: float,
        local_ts: datetime,
    ) -> tuple[str, str, date, float, float]:
        option_type = "call" if side == "long" else "put"
        strike_target = (
            close + self.cfg.atr_multiplier * atr
            if side == "long"
            else close - self.cfg.atr_multiplier * atr
        )
        expiration = self._resolve_expiration(local_ts)
        if not self.cfg.submit_orders:
            contract_symbol = self._sim_contract_symbol(
                option_type=option_type,
                expiration=expiration,
                strike=strike_target,
            )
            return contract_symbol, option_type, expiration, strike_target, strike_target
        try:
            contract_symbol, picked_strike = self._select_contract(
                option_type=option_type,
                expiration=expiration,
                target_strike=strike_target,
                atr=atr,
            )
        except Exception as exc:
            if self.cfg.submit_orders:
                raise
            contract_symbol = self._sim_contract_symbol(
                option_type=option_type,
                expiration=expiration,
                strike=strike_target,
            )
            picked_strike = strike_target
            raise RuntimeError(
                f"sim_fallback:{option_type}:{expiration.isoformat()}:{strike_target:.2f}:{exc}"
            ) from exc
        return contract_symbol, option_type, expiration, strike_target, picked_strike

    def _on_independent_meta_decision(
        self,
        *,
        closed_bar: dict[str, Any],
        logger: Callable[[str], None],
        close: float,
        atr: float,
        local_ts: datetime,
        prev_close: float = float("nan"),
        enforce_trend_gate: bool = False,
        allow_new_entries: bool = True,
        allow_exits: bool = True,
    ) -> dict[str, Any]:
        if not math.isfinite(close):
            return {"event": "error", "reason": "invalid_close"}
        high = _as_float(closed_bar.get("high"))
        low = _as_float(closed_bar.get("low"))

        target_long = self._target_contracts_for_side(
            side="long",
            closed_bar=closed_bar,
            close=close,
            high=high,
            low=low,
            atr=atr,
            prev_close=prev_close,
            enforce_trend_gate=enforce_trend_gate,
            local_ts=local_ts,
            allow_new_entries=allow_new_entries,
            allow_exits=allow_exits,
            logger=logger,
        )
        target_short = self._target_contracts_for_side(
            side="short",
            closed_bar=closed_bar,
            close=close,
            high=high,
            low=low,
            atr=atr,
            prev_close=prev_close,
            enforce_trend_gate=enforce_trend_gate,
            local_ts=local_ts,
            allow_new_entries=allow_new_entries,
            allow_exits=allow_exits,
            logger=logger,
        )
        current_long = int(self._long_contracts)
        current_short = int(self._short_contracts)
        side_state: dict[str, dict[str, Any]] = {
            "long": {
                "current": current_long,
                "target": target_long,
                "symbol": self._long_symbol,
                "option_type": "call",
            },
            "short": {
                "current": current_short,
                "target": target_short,
                "symbol": self._short_symbol,
                "option_type": "put",
            },
        }

        orders: list[dict[str, Any]] = []
        side_events: list[str] = []
        buying_power = self._get_buying_power()

        for side in ("long", "short"):
            current_qty = int(side_state[side]["current"])
            target_qty = int(side_state[side]["target"])
            symbol = side_state[side]["symbol"]
            if current_qty > target_qty:
                if not symbol:
                    return {"event": "error", "reason": f"missing_{side}_symbol_for_close"}
                close_qty = current_qty - target_qty
                close_resp = self._submit_order(
                    symbol=symbol,
                    side="sell",
                    intent="close",
                    qty=close_qty,
                    logger=logger,
                )
                orders.append(
                    {
                        "type": f"close_{side}",
                        "side_key": side,
                        "symbol": symbol,
                        "qty": close_qty,
                        "response": close_resp,
                    }
                )
                side_events.append(f"close_{side}")
                if side == "long":
                    self._long_contracts = target_qty
                    if target_qty == 0:
                        self._long_symbol = None
                        self._long_avg_entry_price = float("nan")
                        self._option_exit_best_price["long"] = float("nan")
                        self._option_exit_best_seen_at["long"] = None
                        self._meta_side_entry_signal_ts["long"] = None
                        self._reset_trail_state("long")
                        self._clear_entry_structure(side="long")
                        self._meta_side_soft_exit_count["long"] = 0
                        self._meta_side_entry_armed["long"] = False
                        cooldown_bars = max(0, int(self.cfg.meta_same_side_reentry_cooldown_bars))
                        self._meta_side_cooldown_until["long"] = (
                            local_ts + timedelta(minutes=cooldown_bars) if cooldown_bars > 0 else None
                        )
                else:
                    self._short_contracts = target_qty
                    if target_qty == 0:
                        self._short_symbol = None
                        self._short_avg_entry_price = float("nan")
                        self._option_exit_best_price["short"] = float("nan")
                        self._option_exit_best_seen_at["short"] = None
                        self._meta_side_entry_signal_ts["short"] = None
                        self._reset_trail_state("short")
                        self._clear_entry_structure(side="short")
                        self._meta_side_soft_exit_count["short"] = 0
                        self._meta_side_entry_armed["short"] = False
                        cooldown_bars = max(0, int(self.cfg.meta_same_side_reentry_cooldown_bars))
                        self._meta_side_cooldown_until["short"] = (
                            local_ts + timedelta(minutes=cooldown_bars) if cooldown_bars > 0 else None
                        )

        for side in ("long", "short"):
            current_qty = int(self._long_contracts if side == "long" else self._short_contracts)
            target_qty = int(side_state[side]["target"])
            if target_qty <= current_qty:
                continue
            symbol = self._long_symbol if side == "long" else self._short_symbol
            option_type = "call" if side == "long" else "put"
            picked_strike = None
            strike_target = None
            expiration = None
            if not symbol:
                if not math.isfinite(atr) or atr <= 0.0:
                    return {"event": "error", "reason": "atr_unavailable", "close": close, "atr": atr}
                try:
                    symbol, option_type, expiration, strike_target, picked_strike = self._select_side_contract(
                        side=side,
                        close=close,
                        atr=atr,
                        local_ts=local_ts,
                    )
                except RuntimeError as exc:
                    msg = str(exc)
                    if msg.startswith("sim_fallback:"):
                        _, option_type, exp_str, strike_str, reason = msg.split(":", 4)
                        expiration = date.fromisoformat(exp_str)
                        strike_target = float(strike_str)
                        picked_strike = strike_target
                        symbol = self._sim_contract_symbol(
                            option_type=option_type,
                            expiration=expiration,
                            strike=strike_target,
                        )
                        logger(
                            "[order_policy] SIM fallback contract "
                            f"type={option_type} exp={expiration.isoformat()} "
                            f"strike={strike_target:.2f} reason={reason}"
                        )
                    else:
                        raise
            logger(
                "[order_policy] CONTRACT SELECTED "
                f"side={side} type={option_type} "
                f"exp={expiration.isoformat() if isinstance(expiration, date) else 'n/a'} "
                f"target_strike={strike_target:.2f} picked_strike={picked_strike:.2f} symbol={symbol}"
            )
            contracts_max, _bp_side, contract_price = self._contracts_max_for_symbol(
                symbol=symbol,
                logger=logger,
            )
            desired_final = min(target_qty, contracts_max) if contracts_max > 0 else target_qty
            if desired_final <= current_qty:
                continue
            open_qty = desired_final - current_qty
            open_resp = self._submit_order(
                symbol=symbol,
                side="buy",
                intent="open",
                qty=open_qty,
                logger=logger,
            )
            orders.append(
                {
                    "type": f"open_{side}",
                    "side_key": side,
                    "symbol": symbol,
                    "qty": open_qty,
                    "response": open_resp,
                    "contract_price": contract_price if math.isfinite(contract_price) else None,
                    "selected_option_type": option_type,
                    "expiration": expiration.isoformat() if isinstance(expiration, date) else None,
                    "target_strike": strike_target if strike_target is not None and math.isfinite(strike_target) else None,
                    "picked_strike": picked_strike if picked_strike is not None and math.isfinite(picked_strike) else None,
                }
            )
            side_events.append(f"open_{side}")
            if side == "long":
                prior_avg_entry = self._long_avg_entry_price
                self._long_contracts = desired_final
                self._long_symbol = symbol
                if current_qty <= 0 and desired_final > 0:
                    self._meta_side_entry_signal_ts["long"] = self._latest_meta_snapshot_local_ts() or local_ts
                    self._seed_trail_state(side="long", close=close, atr=atr, high=high, low=low, local_ts=local_ts)
                    self._seed_entry_structure(side="long", local_ts=local_ts, entry_price=close)
                filled_avg = self._extract_filled_avg_price(open_resp.get("verification", {}).get("order") if isinstance(open_resp, dict) else None)
                if not math.isfinite(filled_avg):
                    filled_avg = self._extract_filled_avg_price(open_resp.get("response") if isinstance(open_resp, dict) else None)
                if not math.isfinite(filled_avg) and math.isfinite(contract_price):
                    filled_avg = float(contract_price)
                if math.isfinite(filled_avg):
                    blended_avg = self._blend_avg_entry_price(
                        current_avg=prior_avg_entry,
                        current_qty=current_qty,
                        fill_price=float(filled_avg),
                        fill_qty=open_qty,
                    )
                    self._long_avg_entry_price = blended_avg if math.isfinite(blended_avg) else float(filled_avg)
                    if current_qty <= 0:
                        self._option_exit_best_price["long"] = self._long_avg_entry_price
                        self._option_exit_best_seen_at["long"] = local_ts
                    elif not math.isfinite(_as_float(self._option_exit_best_price.get("long"))):
                        self._option_exit_best_price["long"] = self._long_avg_entry_price
                        self._option_exit_best_seen_at["long"] = local_ts
                self._clear_intrabar_entry_intent(side="long", reason="entry_filled")
            else:
                prior_avg_entry = self._short_avg_entry_price
                self._short_contracts = desired_final
                self._short_symbol = symbol
                if current_qty <= 0 and desired_final > 0:
                    self._meta_side_entry_signal_ts["short"] = self._latest_meta_snapshot_local_ts() or local_ts
                    self._seed_trail_state(side="short", close=close, atr=atr, high=high, low=low, local_ts=local_ts)
                    self._seed_entry_structure(side="short", local_ts=local_ts, entry_price=close)
                filled_avg = self._extract_filled_avg_price(open_resp.get("verification", {}).get("order") if isinstance(open_resp, dict) else None)
                if not math.isfinite(filled_avg):
                    filled_avg = self._extract_filled_avg_price(open_resp.get("response") if isinstance(open_resp, dict) else None)
                if not math.isfinite(filled_avg) and math.isfinite(contract_price):
                    filled_avg = float(contract_price)
                if math.isfinite(filled_avg):
                    blended_avg = self._blend_avg_entry_price(
                        current_avg=prior_avg_entry,
                        current_qty=current_qty,
                        fill_price=float(filled_avg),
                        fill_qty=open_qty,
                    )
                    self._short_avg_entry_price = blended_avg if math.isfinite(blended_avg) else float(filled_avg)
                    if current_qty <= 0:
                        self._option_exit_best_price["short"] = self._short_avg_entry_price
                        self._option_exit_best_seen_at["short"] = local_ts
                    elif not math.isfinite(_as_float(self._option_exit_best_price.get("short"))):
                        self._option_exit_best_price["short"] = self._short_avg_entry_price
                        self._option_exit_best_seen_at["short"] = local_ts
                self._clear_intrabar_entry_intent(side="short", reason="entry_filled")

        self._refresh_legacy_state()
        prev_counts = {"long": current_long, "short": current_short}
        for side in ("long", "short"):
            prev_qty = int(prev_counts[side])
            new_qty = int(self._long_contracts if side == "long" else self._short_contracts)
            if prev_qty <= 0 and new_qty > 0:
                self._meta_side_bars_held[side] = 0
                self._meta_side_soft_exit_count[side] = 0
            elif prev_qty > 0 and new_qty > 0:
                self._meta_side_bars_held[side] = max(0, int(self._meta_side_bars_held.get(side, -1)) + 1)
            else:
                self._meta_side_bars_held[side] = -1
                self._meta_side_entry_signal_ts[side] = None
                self._meta_side_soft_exit_count[side] = 0
                self._clear_entry_structure(side=side)
        if not orders:
            return {
                "event": "hold",
                "mode": "independent_meta",
                "position": int(self._pos),
                "signed_contracts": int(self._signed_contracts),
                "long_contracts": int(self._long_contracts),
                "short_contracts": int(self._short_contracts),
                "open_long_symbol": self._long_symbol,
                "open_short_symbol": self._short_symbol,
                "target_long_contracts": int(target_long),
                "target_short_contracts": int(target_short),
                "meta_trailing_stop_enabled": bool(self.cfg.meta_trailing_stop_enabled),
                "option_exit_policy": str(self.cfg.option_exit_policy),
                "buying_power": buying_power if math.isfinite(buying_power) else None,
                "close": close,
                "atr": atr,
            }

        event = "multi_update" if len(side_events) > 1 else side_events[0]
        return {
            "event": event,
            "mode": "independent_meta",
            "position": int(self._pos),
            "signed_contracts": int(self._signed_contracts),
            "long_contracts": int(self._long_contracts),
            "short_contracts": int(self._short_contracts),
            "long_bars_held": int(self._meta_side_bars_held.get("long", -1)),
            "short_bars_held": int(self._meta_side_bars_held.get("short", -1)),
            "long_soft_exit_count": int(self._meta_side_soft_exit_count.get("long", 0)),
            "short_soft_exit_count": int(self._meta_side_soft_exit_count.get("short", 0)),
            "open_long_symbol": self._long_symbol,
            "open_short_symbol": self._short_symbol,
            "target_long_contracts": int(target_long),
            "target_short_contracts": int(target_short),
            "meta_trailing_stop_enabled": bool(self.cfg.meta_trailing_stop_enabled),
            "buying_power": buying_power if math.isfinite(buying_power) else None,
            "orders": orders,
            "close": close,
            "atr": atr,
        }

    def on_1m_bar(
        self,
        *,
        bar: dict[str, Any],
        logger: Callable[[str], None] = print,
    ) -> dict[str, Any]:
        """
        Optional intrabar execution monitor for independent meta mode.
        Uses the latest interval meta probabilities/thresholds and 1m trend gating.
        """
        open_ = _as_float(bar.get("open"))
        close = _as_float(bar.get("close"))
        high = _as_float(bar.get("high"))
        low = _as_float(bar.get("low"))
        local_ts = self._to_local_ts(bar.get("timestamp"))
        self.reconcile_with_broker(logger=logger, local_ts=local_ts, force=False)
        if not math.isfinite(close):
            return {"event": "hold", "mode": "intrabar_meta", "reason": "invalid_close"}
        if self.has_pending_broker_reconcile():
            self._meta_side_reason["long"] = "pending_broker_reconcile"
            self._meta_side_reason["short"] = "pending_broker_reconcile"
            return {"event": "hold", "mode": "intrabar_meta", "reason": "pending_broker_reconcile"}
        forced_flat = self._maybe_force_close_expiring_positions(local_ts=local_ts, logger=logger)
        if forced_flat is not None:
            forced_flat.setdefault("mode", "intrabar_meta")
            return forced_flat

        self._prev_1m_close = self._last_1m_close
        self._last_1m_close = float(close)
        self._recent_1m_closes.append(float(close))

        if not bool(self.cfg.meta_intrabar_execution_enabled):
            return {"event": "hold", "mode": "intrabar_meta", "reason": "intrabar_disabled"}

        if not self._latest_meta_side_snapshot:
            return {"event": "hold", "mode": "intrabar_meta", "reason": "no_meta_snapshot"}

        atr = self._compute_atr()
        side_bar = dict(self._latest_meta_side_snapshot)
        side_bar.update(
            {
                "timestamp": bar.get("timestamp"),
                "open": open_,
                "close": close,
                "high": high,
                "low": low,
            }
        )
        allow_new_entries = True
        allow_exits = True
        if bool(self.cfg.meta_intrabar_breakout_entry_only):
            long_triggered = self._intrabar_entry_triggered(
                side="long",
                open_=open_,
                high=high,
                low=low,
                close=close,
                local_ts=local_ts,
            )
            short_triggered = self._intrabar_entry_triggered(
                side="short",
                open_=open_,
                high=high,
                low=low,
                close=close,
                local_ts=local_ts,
            )
            if long_triggered and short_triggered:
                long_triggered = False
                short_triggered = False
                self._meta_side_reason["long"] = "intrabar_setup_conflict_skip"
                self._meta_side_reason["short"] = "intrabar_setup_conflict_skip"
            if not long_triggered and self._long_contracts <= 0:
                side_bar["p_enter_long"] = float("nan")
            if not short_triggered and self._short_contracts <= 0:
                side_bar["p_enter_short"] = float("nan")
        result = self._on_independent_meta_decision(
            closed_bar=side_bar,
            logger=logger,
            close=close,
            atr=atr,
            local_ts=local_ts,
            prev_close=open_,
            enforce_trend_gate=True,
            allow_new_entries=allow_new_entries,
            allow_exits=allow_exits,
        )
        if isinstance(result, dict):
            result.setdefault("mode", "intrabar_meta")
        return result

    def on_decision(
        self,
        *,
        action: float,
        closed_bar: dict[str, Any],
        logger: Callable[[str], None] = print,
        update_bar_state: bool = True,
    ) -> dict[str, Any]:
        """
        Process one interval close + agent action.

        Returns a dict with an `event` key:
          hold | enter | rebalance | flip | flat | error
        """
        try:
            local_ts = self._to_local_ts(closed_bar.get("timestamp"))
            self.reconcile_with_broker(logger=logger, local_ts=local_ts, force=False)
            if self.has_pending_broker_reconcile():
                stale_pending_result = self._cancel_pending_open_if_invalidated(
                    closed_bar=closed_bar,
                    logger=logger,
                )
                if stale_pending_result is not None:
                    if str(stale_pending_result.get("event", "")).strip().lower() == "hold":
                        self._meta_side_reason["long"] = stale_pending_result.get("reason", "pending_broker_reconcile")
                        self._meta_side_reason["short"] = stale_pending_result.get("reason", "pending_broker_reconcile")
                        return stale_pending_result
                    # The stale entry was canceled/cleared. Continue processing
                    # the fresh 10m setup so a newly valid opposite window can arm.
                self._meta_side_reason["long"] = "pending_broker_reconcile"
                self._meta_side_reason["short"] = "pending_broker_reconcile"
                if self.has_pending_broker_reconcile():
                    return {"event": "hold", "reason": "pending_broker_reconcile"}
            raw_action = _as_float(action)
            if not math.isfinite(raw_action):
                return {"event": "error", "reason": f"invalid_action:{action}"}
            act = max(-1.0, min(1.0, float(raw_action)))
            deadband = max(0.0, float(self.cfg.action_deadband))
            desired_pos = 0 if abs(act) <= deadband else (1 if act > 0.0 else -1)
            # Direction-only mode: ignore action magnitude for execution.
            # Keep these fields for monitoring/debug compatibility.
            smooth_action = float(desired_pos)
            effective_action = float(desired_pos)
            prev_effective = float(self._action_effective)
            smoothed_change_applied = abs(effective_action - prev_effective) > 0.0
            self._action_ema = smooth_action
            self._action_effective = effective_action

            close = _as_float(closed_bar.get("close"))
            atr = self._update_bar_state(closed_bar) if update_bar_state else self._compute_atr()

            if not math.isfinite(close):
                return {"event": "error", "reason": "invalid_close"}
            forced_flat = self._maybe_force_close_expiring_positions(local_ts=local_ts, logger=logger)
            if forced_flat is not None:
                forced_flat.setdefault("mode", "interval_policy")
                return forced_flat
            if self._has_meta_side_thresholds(closed_bar):
                self._cache_meta_side_snapshot(closed_bar=closed_bar, atr=atr)
                if bool(self.cfg.meta_intrabar_execution_enabled) and bool(self.cfg.meta_intrabar_breakout_entry_only):
                    self._refresh_intrabar_entry_intents(closed_bar=self._latest_meta_side_snapshot or closed_bar, local_ts=local_ts)
                    result = self._on_independent_meta_decision(
                        closed_bar=closed_bar,
                        logger=logger,
                        close=close,
                        atr=atr,
                        local_ts=local_ts,
                        prev_close=self._prev_1m_close,
                        enforce_trend_gate=False,
                        allow_new_entries=False,
                        allow_exits=True,
                    )
                    if isinstance(result, dict):
                        result.setdefault("mode", "independent_meta")
                        if str(result.get("event", "")).strip().lower() in {"hold", "no_change"}:
                            result["event"] = "intent_update"
                    return result
                if bool(self.cfg.meta_execute_on_interval_close):
                    result = self._on_independent_meta_decision(
                        closed_bar=closed_bar,
                        logger=logger,
                        close=close,
                        atr=atr,
                        local_ts=local_ts,
                        prev_close=self._prev_1m_close,
                        enforce_trend_gate=False,
                    )
                    if isinstance(result, dict):
                        result.setdefault("mode", "independent_meta")
                    return result
                return {
                    "event": "intent_update",
                    "mode": "independent_meta",
                    "close": close,
                    "atr": atr if math.isfinite(atr) else None,
            "long_contracts": int(self._long_contracts),
            "short_contracts": int(self._short_contracts),
            "long_bars_held": int(self._meta_side_bars_held.get("long", -1)),
            "short_bars_held": int(self._meta_side_bars_held.get("short", -1)),
            "open_long_symbol": self._long_symbol,
            "open_short_symbol": self._short_symbol,
                }
            self._latest_meta_side_snapshot = None
            current_signed = int(self._signed_contracts)
            current_pos = 1 if current_signed > 0 else (-1 if current_signed < 0 else 0)
            if current_signed == 0 and desired_pos != 0 and not self._new_entries_allowed(local_ts=local_ts):
                desired_pos = 0
            if current_signed == 0 and desired_pos != 0 and self._entry_lockout_active(local_ts=local_ts):
                desired_pos = 0

            flat_blocked = False
            flip_blocked = False
            if desired_pos == 0 and current_signed != 0 and not self.cfg.close_on_flat:
                flat_blocked = True
                desired_pos = current_pos
            if desired_pos != 0 and current_signed != 0 and desired_pos != current_pos and not self.cfg.close_on_flip:
                flip_blocked = True
                desired_pos = current_pos
            redundant_roundtrip_hold = False
            same_side_reentry_grace_bars = max(0, int(self.cfg.same_side_reentry_grace_bars))
            if current_signed == 0:
                self._pending_flat_side = 0
                self._pending_flat_bars = 0
            elif desired_pos == current_pos:
                self._pending_flat_side = 0
                self._pending_flat_bars = 0
            elif (
                desired_pos == 0
                and current_pos != 0
                and same_side_reentry_grace_bars > 0
                and self.cfg.close_on_flat
            ):
                if self._pending_flat_side != current_pos:
                    self._pending_flat_side = int(current_pos)
                    self._pending_flat_bars = 1
                else:
                    self._pending_flat_bars += 1
                if self._pending_flat_bars <= same_side_reentry_grace_bars:
                    redundant_roundtrip_hold = True
                    desired_pos = current_pos
                else:
                    self._pending_flat_side = 0
                    self._pending_flat_bars = 0
            else:
                self._pending_flat_side = 0
                self._pending_flat_bars = 0

            opposite_quality_blocked = False
            opposite_confirmation_pending = False
            opposite_prob_edge = float("nan")
            opposite_confirm_bars = max(1, int(self.cfg.opposite_confirm_bars))
            opposite_min_abs_action = max(0.0, float(self.cfg.opposite_min_abs_action))
            opposite_min_prob_edge = max(0.0, float(self.cfg.opposite_min_prob_edge))
            if current_signed == 0 or desired_pos == 0 or desired_pos == current_pos:
                self._pending_opposite_side = 0
                self._pending_opposite_bars = 0
            else:
                opposite_prob_edge = self._directional_prob_edge(
                    closed_bar=closed_bar,
                    desired_pos=desired_pos,
                )
                abs_quality_ok = abs(act) >= opposite_min_abs_action
                prob_quality_ok = (
                    opposite_min_prob_edge <= 0.0
                    or (math.isfinite(opposite_prob_edge) and opposite_prob_edge >= opposite_min_prob_edge)
                )
                if not (abs_quality_ok and prob_quality_ok):
                    opposite_quality_blocked = True
                    desired_pos = current_pos
                    self._pending_opposite_side = 0
                    self._pending_opposite_bars = 0
                else:
                    if self._pending_opposite_side != desired_pos:
                        self._pending_opposite_side = int(desired_pos)
                        self._pending_opposite_bars = 1
                    else:
                        self._pending_opposite_bars += 1
                    if self._pending_opposite_bars < opposite_confirm_bars:
                        opposite_confirmation_pending = True
                        desired_pos = current_pos
                    else:
                        self._pending_opposite_side = 0
                        self._pending_opposite_bars = 0

            option_type: str | None = None
            strike_target: float | None = None
            expiration: date | None = None
            contract_symbol: str | None = None
            picked_strike: float | None = None
            contracts_max = 0
            buying_power = float("nan")
            contract_price = float("nan")
            target_signed = 0

            if desired_pos != 0:
                if desired_pos == current_pos and self._open_symbol:
                    contract_symbol = self._open_symbol
                else:
                    if not math.isfinite(atr) or atr <= 0.0:
                        return {"event": "error", "reason": "atr_unavailable", "close": close, "atr": atr}
                    option_type = "call" if desired_pos > 0 else "put"
                    strike_target = (
                        close + self.cfg.atr_multiplier * atr
                        if desired_pos > 0
                        else close - self.cfg.atr_multiplier * atr
                    )
                    expiration = self._resolve_expiration(local_ts)
                    try:
                        contract_symbol, picked_strike = self._select_contract(
                            option_type=option_type,
                            expiration=expiration,
                            target_strike=strike_target,
                            atr=atr,
                        )
                    except Exception as exc:
                        if self.cfg.submit_orders:
                            raise
                        contract_symbol = self._sim_contract_symbol(
                            option_type=option_type,
                            expiration=expiration,
                            strike=strike_target,
                        )
                        picked_strike = strike_target
                        logger(
                            "[order_policy] SIM fallback contract "
                            f"type={option_type} exp={expiration.isoformat()} "
                            f"strike={strike_target:.2f} reason={exc}"
                        )

                if not contract_symbol:
                    return {"event": "error", "reason": "missing_contract_symbol"}

                contracts_max, buying_power, contract_price = self._contracts_max_for_symbol(
                    symbol=contract_symbol,
                    logger=logger,
                )
                # Size by fixed contract qty, not action magnitude.
                target_abs = max(0, int(self.cfg.qty))
                if contracts_max > 0:
                    target_abs = min(target_abs, int(contracts_max))
                target_signed = int(desired_pos * target_abs)
            else:
                buying_power = self._get_buying_power()
                target_signed = 0

            delta = int(target_signed - current_signed)
            step_cap = int(self.cfg.max_step_contracts)
            if step_cap > 0:
                step_delta = max(-step_cap, min(step_cap, delta))
            else:
                step_delta = delta
            step_target = int(current_signed + step_delta)
            step_pos = 1 if step_target > 0 else (-1 if step_target < 0 else 0)

            if step_target == current_signed:
                return {
                    "event": "hold",
                    "pos": current_pos,
                    "signed_contracts": current_signed,
                    "target_signed_contracts": target_signed,
                    "step_target_signed_contracts": step_target,
                    "exposure_raw": act,
                    "exposure_smooth": smooth_action,
                    "exposure_effective": effective_action,
                    "smoothed_change_applied": smoothed_change_applied,
                    "contracts_max": contracts_max,
                    "buying_power": buying_power if math.isfinite(buying_power) else None,
                    "contract_price": contract_price if math.isfinite(contract_price) else None,
                    "contract": contract_symbol or self._open_symbol,
                    "close": close,
                    "atr": atr,
                    "flat_blocked": flat_blocked,
                    "flip_blocked": flip_blocked,
                    "redundant_roundtrip_hold": redundant_roundtrip_hold,
                    "pending_flat_side": int(self._pending_flat_side),
                    "pending_flat_bars": int(self._pending_flat_bars),
                    "same_side_reentry_grace_bars": int(same_side_reentry_grace_bars),
                    "opposite_quality_blocked": opposite_quality_blocked,
                    "opposite_confirmation_pending": opposite_confirmation_pending,
                    "opposite_prob_edge": (
                        float(opposite_prob_edge) if math.isfinite(opposite_prob_edge) else None
                    ),
                    "pending_opposite_side": int(self._pending_opposite_side),
                    "pending_opposite_bars": int(self._pending_opposite_bars),
                    "opposite_confirm_bars": int(opposite_confirm_bars),
                    "opposite_min_abs_action": float(opposite_min_abs_action),
                    "opposite_min_prob_edge": float(opposite_min_prob_edge),
                }

            orders: list[dict[str, Any]] = []
            old_symbol = self._open_symbol

            # Leg 1: close/reduce existing side.
            if current_signed != 0 and (step_pos != current_pos or abs(step_target) < abs(current_signed)):
                if not self._open_symbol:
                    return {"event": "error", "reason": "missing_open_symbol_for_close"}
                close_qty = (
                    abs(current_signed)
                    if step_pos != current_pos
                    else abs(current_signed) - abs(step_target)
                )
                if close_qty > 0:
                    close_resp = self._submit_order(
                        symbol=self._open_symbol,
                        side="sell",
                        intent="close",
                        qty=close_qty,
                        logger=logger,
                    )
                    orders.append(
                        {
                            "type": "close",
                            "symbol": self._open_symbol,
                            "qty": close_qty,
                            "response": close_resp,
                        }
                    )
                if step_pos != current_pos:
                    self._open_symbol = None

            # Leg 2: open/increase desired side.
            if step_target != 0 and (step_pos != current_pos or abs(step_target) > abs(current_signed)):
                if step_pos == current_pos and self._open_symbol:
                    open_symbol = self._open_symbol
                    open_qty = abs(step_target) - abs(current_signed)
                else:
                    if not contract_symbol:
                        if not math.isfinite(atr) or atr <= 0.0:
                            return {"event": "error", "reason": "atr_unavailable", "close": close, "atr": atr}
                        option_type = "call" if step_pos > 0 else "put"
                        strike_target = (
                            close + self.cfg.atr_multiplier * atr
                            if step_pos > 0
                            else close - self.cfg.atr_multiplier * atr
                        )
                        expiration = self._resolve_expiration(local_ts)
                        try:
                            contract_symbol, picked_strike = self._select_contract(
                                option_type=option_type,
                                expiration=expiration,
                                target_strike=strike_target,
                                atr=atr,
                            )
                        except Exception as exc:
                            if self.cfg.submit_orders:
                                raise
                            contract_symbol = self._sim_contract_symbol(
                                option_type=option_type,
                                expiration=expiration,
                                strike=strike_target,
                            )
                            picked_strike = strike_target
                            logger(
                                "[order_policy] SIM fallback contract "
                                f"type={option_type} exp={expiration.isoformat()} "
                                f"strike={strike_target:.2f} reason={exc}"
                            )
                    open_symbol = contract_symbol
                    open_qty = abs(step_target) if step_pos != current_pos else max(0, abs(step_target) - abs(current_signed))

                if not open_symbol:
                    return {"event": "error", "reason": "missing_open_symbol_for_open"}

                if open_qty > 0:
                    open_resp = self._submit_order(
                        symbol=open_symbol,
                        side="buy",
                        intent="open",
                        qty=open_qty,
                        logger=logger,
                    )
                    orders.append(
                        {
                            "type": "open",
                            "symbol": open_symbol,
                            "qty": open_qty,
                            "response": open_resp,
                        }
                    )
                self._open_symbol = open_symbol

            self._signed_contracts = int(step_target)
            self._pos = 1 if self._signed_contracts > 0 else (-1 if self._signed_contracts < 0 else 0)
            if self._signed_contracts == 0:
                self._open_symbol = None
                self._long_contracts = 0
                self._short_contracts = 0
                self._long_symbol = None
                self._short_symbol = None
            elif self._signed_contracts > 0:
                self._long_contracts = int(abs(self._signed_contracts))
                self._short_contracts = 0
                self._long_symbol = self._open_symbol
                self._short_symbol = None
            else:
                self._long_contracts = 0
                self._short_contracts = int(abs(self._signed_contracts))
                self._long_symbol = None
                self._short_symbol = self._open_symbol

            if current_signed == 0 and self._signed_contracts != 0:
                event = "enter"
            elif self._signed_contracts == 0:
                event = "flat"
            elif current_pos != 0 and self._pos != current_pos:
                event = "flip"
            else:
                event = "rebalance"

            return {
                "event": event,
                "pos": self._pos,
                "signed_contracts": self._signed_contracts,
                "prev_signed_contracts": current_signed,
                "target_signed_contracts": target_signed,
                "step_target_signed_contracts": step_target,
                "delta_signed_contracts": delta,
                "step_delta_signed_contracts": step_delta,
                "exposure_raw": act,
                "exposure_smooth": smooth_action,
                "exposure_effective": effective_action,
                "smoothed_change_applied": smoothed_change_applied,
                "contracts_max": int(contracts_max),
                "buying_power": buying_power if math.isfinite(buying_power) else None,
                "contract_price": contract_price if math.isfinite(contract_price) else None,
                "contract": self._open_symbol,
                "selected_contract": contract_symbol,
                "selected_option_type": option_type,
                "expiration": expiration.isoformat() if isinstance(expiration, date) else None,
                "target_strike": strike_target if strike_target is not None and math.isfinite(strike_target) else None,
                "picked_strike": picked_strike if picked_strike is not None and math.isfinite(picked_strike) else None,
                "orders": orders,
                "closed_symbol": old_symbol if self._open_symbol != old_symbol else None,
                "close": close,
                "atr": atr,
                "flat_blocked": flat_blocked,
                "flip_blocked": flip_blocked,
                "redundant_roundtrip_hold": redundant_roundtrip_hold,
                "pending_flat_side": int(self._pending_flat_side),
                "pending_flat_bars": int(self._pending_flat_bars),
                "same_side_reentry_grace_bars": int(same_side_reentry_grace_bars),
                "opposite_quality_blocked": opposite_quality_blocked,
                "opposite_confirmation_pending": opposite_confirmation_pending,
                "opposite_prob_edge": (
                    float(opposite_prob_edge) if math.isfinite(opposite_prob_edge) else None
                ),
                "pending_opposite_side": int(self._pending_opposite_side),
                "pending_opposite_bars": int(self._pending_opposite_bars),
                "opposite_confirm_bars": int(opposite_confirm_bars),
                "opposite_min_abs_action": float(opposite_min_abs_action),
                "opposite_min_prob_edge": float(opposite_min_prob_edge),
            }
        except Exception as exc:
            return {"event": "error", "reason": str(exc)}
