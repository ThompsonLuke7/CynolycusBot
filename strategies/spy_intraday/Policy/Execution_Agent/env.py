from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

import numpy as np
import pandas as pd


ACTION_WAIT = 0
ACTION_EXECUTE = 1
N_EXEC_ACTIONS = 2

# Backward-compat aliases kept for older imports/checkpoints.
ACTION_ENTER = ACTION_EXECUTE
ACTION_SCALE_IN = ACTION_EXECUTE
ACTION_SCALE_OUT = ACTION_EXECUTE
ACTION_EXIT = ACTION_EXECUTE

EVENT_NONE = 0
EVENT_ENTER = 1
EVENT_EXIT = -1

PENDING_NONE = 0
PENDING_ENTER = 1
PENDING_EXIT = 2
PENDING_SWITCH = 3


@dataclass(frozen=True)
class ExecutionEnvConfig:
    add_position_features: bool = True
    max_units: int = 3
    entry_units: int = 1
    baseline_units: int = 1
    force_flat_on_dir_flip: bool = False
    action_deadband: float = 1e-9
    entry_min_since_flip: int = 1
    entry_max_since_flip: int = 12

    spread_bps: float = 1.0
    slippage_bps: float = 1.0
    commission_per_unit_ret: float = 0.0
    trade_penalty_ret: float = 0.00002
    churn_penalty_ret: float = 0.00003
    mae_penalty_lambda: float = 0.25
    low_conf_threshold: float = 0.35
    low_conf_penalty_lambda: float = 0.00015

    episode_mode: str = "flip_window"  # flip_window|day
    flip_start_delay_min: int = 0
    flip_window_minutes: int = 20
    seed: int = 7


class DirectionGatedExecutionEnv:
    """
    1m execution env conditioned on frozen 15m intent.
    Event-timing reward (on execute only) relative to naive immediate execution:
        reward = (agent_event_net - baseline_event_net) - MAE_penalty.
    Non-execute steps carry zero reward.
    """

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        feature_cols: list[str],
        add_position_features: bool = True,
        max_units: int = 3,
        entry_units: int = 1,
        baseline_units: int = 1,
        force_flat_on_dir_flip: bool = False,
        action_deadband: float = 1e-9,
        entry_min_since_flip: int = 1,
        entry_max_since_flip: int = 12,
        spread_bps: float = 1.0,
        slippage_bps: float = 1.0,
        commission_per_unit_ret: float = 0.0,
        trade_penalty_ret: float = 0.00002,
        churn_penalty_ret: float = 0.00003,
        mae_penalty_lambda: float = 0.25,
        low_conf_threshold: float = 0.35,
        low_conf_penalty_lambda: float = 0.00015,
        episode_mode: str = "flip_window",
        flip_start_delay_min: int = 0,
        flip_window_minutes: int = 20,
        seed: int = 7,
    ):
        self.df = df.copy()
        if "timestamp" not in self.df.columns:
            raise ValueError("df must contain timestamp")
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], errors="coerce")
        self.df = self.df[self.df["timestamp"].notna()].sort_values("timestamp").reset_index(drop=True)
        required = ["open", "high", "low", "close", "day_id", "htf_dir", "htf_conf"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.feature_cols = list(feature_cols)
        for c in self.feature_cols:
            if c not in self.df.columns:
                raise ValueError(f"Missing feature column: {c}")
        self.add_position_features = bool(add_position_features)

        self.max_units = max(1, int(max_units))
        self.entry_units = max(1, int(entry_units))
        self.baseline_units = max(1, int(baseline_units))
        self.force_flat_on_dir_flip = bool(force_flat_on_dir_flip)
        self.action_deadband = float(action_deadband)
        self.entry_min_since_flip = max(0, int(entry_min_since_flip))
        self.entry_max_since_flip = max(self.entry_min_since_flip, int(entry_max_since_flip))

        self.spread_bps = float(spread_bps)
        self.slippage_bps = float(slippage_bps)
        self.commission_per_unit_ret = float(commission_per_unit_ret)
        self.trade_penalty_ret = float(trade_penalty_ret)
        self.churn_penalty_ret = float(churn_penalty_ret)
        self.mae_penalty_lambda = float(mae_penalty_lambda)
        self.low_conf_threshold = float(low_conf_threshold)
        self.low_conf_penalty_lambda = float(low_conf_penalty_lambda)

        self.episode_mode = str(episode_mode).strip().lower()
        self.flip_start_delay_min = max(0, int(flip_start_delay_min))
        self.flip_window_minutes = max(2, int(flip_window_minutes))
        self.rng = np.random.default_rng(int(seed))

        self.df["ret_next"] = self.df["close"].shift(-1) / self.df["close"] - 1.0
        self._open = self.df["open"].to_numpy(dtype=np.float64, copy=False)
        self._high = self.df["high"].to_numpy(dtype=np.float64, copy=False)
        self._low = self.df["low"].to_numpy(dtype=np.float64, copy=False)
        self._close = self.df["close"].to_numpy(dtype=np.float64, copy=False)
        self._ret_next = self.df["ret_next"].to_numpy(dtype=np.float64, copy=False)
        self._day_id = self.df["day_id"].to_numpy(dtype=np.int64, copy=False)
        self._htf_dir = pd.to_numeric(self.df["htf_dir"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        self._htf_conf = pd.to_numeric(self.df["htf_conf"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        self._time_since_flip = (
            pd.to_numeric(self.df.get("time_since_flip_min", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        )
        self._feature_matrix = self.df[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
        self._build_episodes()

        self._base_obs_dim = int(self._feature_matrix.shape[1])
        self.obs_dim = self._base_obs_dim + (6 if self.add_position_features else 0)
        self._ep_ptr = -1
        self._i = 0
        self._ep_end = 0
        self._reset_state()

    def _build_episodes(self) -> None:
        n = len(self.df)
        if n < 3:
            raise ValueError("Need at least 3 rows for execution env.")

        day_change = np.r_[True, self._day_id[1:] != self._day_id[:-1]]
        day_starts = np.where(day_change)[0]
        day_ends = np.r_[day_starts[1:] - 1, n - 1]
        day_end_by_idx = np.empty(n, dtype=np.int64)
        for s, e in zip(day_starts, day_ends, strict=False):
            day_end_by_idx[s : e + 1] = e

        starts: list[int] = []
        ends: list[int] = []
        if self.episode_mode == "flip_window":
            htf_sign = np.sign(self._htf_dir)
            flips = np.where(htf_sign != np.r_[htf_sign[0], htf_sign[:-1]])[0]
            for f in flips:
                s = int(f + self.flip_start_delay_min)
                if s >= n - 1:
                    continue
                e = min(s + self.flip_window_minutes - 1, int(day_end_by_idx[s]), n - 2)
                if e > s:
                    starts.append(s)
                    ends.append(e)

        if not starts:
            for s, e in zip(day_starts, day_ends, strict=False):
                e2 = min(int(e), n - 2)
                if e2 > int(s):
                    starts.append(int(s))
                    ends.append(e2)

        if not starts:
            raise ValueError("No valid execution episodes produced.")
        self.day_starts = starts
        self._episode_ends = ends

    def _reset_state(self) -> None:
        self.agent_pos_units = 0
        self.baseline_pos_units = 0
        self.agent_entry_price = np.nan
        self.unrealized = 0.0
        self.time_in_pos = 0
        self.realized_agent = 0.0
        self.realized_baseline = 0.0
        self.pending_order = PENDING_NONE
        self.pending_target_dir = 0
        self.current_event_type = EVENT_NONE
        self.event_started_at_idx = -1
        self.event_prev_dir = 0
        self.event_start_pos_units = 0
        self.event_start_price = np.nan
        self.event_target_dir = 0

    def _trade_cost(self, units_delta: int) -> float:
        if units_delta <= 0:
            return 0.0
        bps = (self.spread_bps + self.slippage_bps) / 10000.0
        return float(units_delta) * (bps + self.commission_per_unit_ret + self.trade_penalty_ret)

    def _event_name(self, event_type: int) -> str:
        if int(event_type) > 0:
            return "ENTER_WINDOW"
        if int(event_type) < 0:
            return "EXIT_WINDOW"
        return "NONE"

    def _pending_name(self, pending: int) -> str:
        if pending == PENDING_ENTER:
            return "ENTER"
        if pending == PENDING_EXIT:
            return "EXIT"
        if pending == PENDING_SWITCH:
            return "SWITCH"
        return "NONE"

    def _event_spec_at_idx(self, idx: int) -> tuple[int, int, int, int]:
        cur_dir = int(np.sign(self._htf_dir[idx]))
        prev_dir = cur_dir if idx <= 0 else int(np.sign(self._htf_dir[idx - 1]))
        if cur_dir == prev_dir:
            return EVENT_NONE, PENDING_NONE, cur_dir, prev_dir
        if prev_dir == 0 and cur_dir != 0:
            return EVENT_ENTER, PENDING_ENTER, cur_dir, prev_dir
        if prev_dir != 0 and cur_dir == 0:
            return EVENT_EXIT, PENDING_EXIT, 0, prev_dir
        return EVENT_EXIT, PENDING_SWITCH, cur_dir, prev_dir

    def _segment_return(self, units: int, direction: int, start_idx: int, end_idx: int) -> float:
        if units <= 0 or direction == 0 or end_idx <= start_idx:
            return 0.0
        p0 = float(self._close[start_idx])
        p1 = float(self._close[end_idx])
        if (not np.isfinite(p0)) or (not np.isfinite(p1)) or p0 <= 0.0:
            return 0.0
        return float(units) * float(direction) * (p1 / p0 - 1.0)

    def _segment_mae(self, units: int, direction: int, start_idx: int, end_idx: int) -> float:
        if units <= 0 or direction == 0 or end_idx <= start_idx:
            return 0.0
        p0 = float(self._close[start_idx])
        if (not np.isfinite(p0)) or p0 <= 0.0:
            return 0.0
        lo = start_idx + 1
        hi = min(end_idx + 1, len(self._low))
        if hi <= lo:
            return 0.0
        if direction > 0:
            adverse = np.maximum(0.0, (p0 - self._low[lo:hi]) / p0)
        else:
            adverse = np.maximum(0.0, (self._high[lo:hi] - p0) / p0)
        if adverse.size == 0:
            return 0.0
        mae = float(np.nanmax(adverse))
        if not np.isfinite(mae):
            return 0.0
        return mae * float(units)

    def _timing_reward_vs_immediate(
        self,
        *,
        event_start_idx: int,
        exec_idx: int,
        horizon_idx: int,
        pending: int,
        target_dir: int,
        prev_dir: int,
        prev_pos_units: int,
    ) -> tuple[float, float, float, float, float]:
        """
        Returns:
            agent_net, baseline_net, mae_penalty, agent_gross, baseline_gross
        """
        horizon_idx = int(max(exec_idx, min(horizon_idx, len(self._close) - 1)))
        units_new = int(min(self.entry_units, self.max_units))
        units_old = int(abs(prev_pos_units))
        if pending == PENDING_ENTER:
            agent_gross = self._segment_return(units_new, target_dir, exec_idx, horizon_idx)
            baseline_gross = self._segment_return(units_new, target_dir, event_start_idx, horizon_idx)
            agent_cost = self._trade_cost(units_new)
            baseline_cost = self._trade_cost(units_new)
            mae = self._segment_mae(units_new, target_dir, exec_idx, horizon_idx)
        elif pending == PENDING_EXIT:
            hold_dir = int(np.sign(prev_dir))
            agent_gross = self._segment_return(units_old, hold_dir, event_start_idx, exec_idx)
            baseline_gross = 0.0
            agent_cost = self._trade_cost(units_old)
            baseline_cost = self._trade_cost(units_old)
            mae = self._segment_mae(units_old, hold_dir, event_start_idx, exec_idx)
        elif pending == PENDING_SWITCH:
            hold_dir = int(np.sign(prev_dir))
            agent_gross = self._segment_return(units_old, hold_dir, event_start_idx, exec_idx)
            agent_gross += self._segment_return(units_new, target_dir, exec_idx, horizon_idx)
            baseline_gross = self._segment_return(units_new, target_dir, event_start_idx, horizon_idx)
            agent_cost = self._trade_cost(units_old + units_new)
            baseline_cost = self._trade_cost(units_old + units_new)
            mae = self._segment_mae(units_old, hold_dir, event_start_idx, exec_idx)
            mae += self._segment_mae(units_new, target_dir, exec_idx, horizon_idx)
        else:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        agent_net = agent_gross - agent_cost
        baseline_net = baseline_gross - baseline_cost
        mae_penalty = self.mae_penalty_lambda * mae
        return agent_net, baseline_net, mae_penalty, agent_gross, baseline_gross

    def _apply_pending_order(self, pending: int, target_dir: int) -> tuple[int, bool]:
        prev = int(self.agent_pos_units)
        pos = prev
        if pending == PENDING_ENTER:
            if prev == 0 and int(target_dir) != 0:
                pos = int(target_dir) * min(self.entry_units, self.max_units)
        elif pending == PENDING_EXIT:
            pos = 0
        elif pending == PENDING_SWITCH:
            if int(target_dir) == 0:
                pos = 0
            else:
                pos = int(target_dir) * min(self.entry_units, self.max_units)
        self.agent_pos_units = int(pos)
        traded_units = abs(int(self.agent_pos_units) - prev)
        return traded_units, self.agent_pos_units != prev

    def _update_agent_position_state(self, prev_agent: int, price: float) -> None:
        new_agent = int(self.agent_pos_units)
        if prev_agent == new_agent:
            return
        prev_abs = abs(int(prev_agent))
        new_abs = abs(int(new_agent))
        prev_sign = int(np.sign(prev_agent))
        new_sign = int(np.sign(new_agent))

        if prev_abs > 0 and np.isfinite(self.agent_entry_price) and float(self.agent_entry_price) > 0.0:
            entry = float(self.agent_entry_price)
            if new_abs == 0 or new_sign != prev_sign:
                self.realized_agent += prev_sign * (price / entry - 1.0) * prev_abs
            elif new_abs < prev_abs:
                closed = prev_abs - new_abs
                self.realized_agent += prev_sign * (price / entry - 1.0) * closed

        if new_abs == 0:
            self.agent_entry_price = np.nan
            self.time_in_pos = 0
            return

        if prev_abs == 0 or new_sign != prev_sign:
            self.agent_entry_price = price
            self.time_in_pos = 0
            return

        if new_abs > prev_abs and np.isfinite(self.agent_entry_price):
            add_units = new_abs - prev_abs
            self.agent_entry_price = (
                prev_abs * float(self.agent_entry_price) + add_units * price
            ) / max(1, prev_abs + add_units)

    def _get_obs(self) -> np.ndarray:
        obs = np.empty(self.obs_dim, dtype=np.float32)
        obs[: self._base_obs_dim] = self._feature_matrix[self._i]
        if self.add_position_features:
            cursor = self._base_obs_dim
            obs[cursor] = float(self.agent_pos_units / max(1, self.max_units))
            obs[cursor + 1] = float(self.baseline_pos_units / max(1, self.baseline_units))
            obs[cursor + 2] = float(np.tanh(self.unrealized * 100.0))
            obs[cursor + 3] = float(np.clip(self._time_since_flip[self._i] / 30.0, 0.0, 1.0))
            obs[cursor + 4] = float(self.current_event_type)
            obs[cursor + 5] = float(1.0 if self.pending_order != PENDING_NONE else 0.0)
        return obs

    def reset(self, day_ptr: int | None = None) -> np.ndarray:
        if day_ptr is None:
            self._ep_ptr = (self._ep_ptr + 1) % len(self.day_starts)
        else:
            self._ep_ptr = int(day_ptr) % len(self.day_starts)
        self._i = int(self.day_starts[self._ep_ptr])
        self._ep_end = int(self._episode_ends[self._ep_ptr])
        self._reset_state()
        return self._get_obs()

    def _apply_baseline(self, desired_dir: int) -> int:
        prev = int(self.baseline_pos_units)
        tgt = 0 if desired_dir == 0 else int(desired_dir * self.baseline_units)
        self.baseline_pos_units = tgt
        return abs(tgt - prev)

    def step(self, action: int | float) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        action_idx = int(np.clip(int(round(float(action))), 0, N_EXEC_ACTIONS - 1))
        price = float(self._close[self._i])
        ret_next = float(self._ret_next[self._i])
        if not np.isfinite(ret_next):
            ret_next = 0.0
        desired_dir = int(np.sign(self._htf_dir[self._i]))
        htf_conf = float(np.clip(self._htf_conf[self._i], 0.0, 1.0))
        time_since_flip = float(self._time_since_flip[self._i])

        prev_agent = int(self.agent_pos_units)
        prev_base = int(self.baseline_pos_units)
        event_started = False
        event_type_now, pending_now, target_dir_now, prev_dir_now = self._event_spec_at_idx(self._i)
        if pending_now != PENDING_NONE:
            self.pending_order = pending_now
            self.pending_target_dir = int(target_dir_now)
            self.current_event_type = int(event_type_now)
            self.event_started_at_idx = int(self._i)
            self.event_prev_dir = int(prev_dir_now)
            self.event_start_pos_units = int(prev_agent)
            self.event_start_price = float(price)
            self.event_target_dir = int(target_dir_now)
            event_started = True

        active_event_type = int(self.current_event_type)
        active_pending_name = self._pending_name(self.pending_order)
        active_pending_target_dir = int(self.pending_target_dir)
        in_window = (
            float(time_since_flip) >= float(self.entry_min_since_flip)
            and float(time_since_flip) <= float(self.entry_max_since_flip)
        )
        forced_execute = bool(
            self.pending_order != PENDING_NONE
            and float(time_since_flip) > float(self.entry_max_since_flip)
        )
        execute_intent = action_idx == ACTION_EXECUTE
        execute_now = bool(
            self.pending_order != PENDING_NONE
            and ((execute_intent and in_window) or forced_execute)
        )
        reward_improvement = 0.0
        agent_net = 0.0
        base_net = 0.0
        timing_mae_pen = 0.0
        timing_agent_gross = 0.0
        timing_base_gross = 0.0
        pending_for_exec = int(self.pending_order)
        target_for_exec = int(self.pending_target_dir)
        event_start_idx = int(self.event_started_at_idx)
        event_prev_dir = int(self.event_prev_dir)
        event_prev_pos = int(self.event_start_pos_units)
        if execute_now:
            (
                agent_net,
                base_net,
                timing_mae_pen,
                timing_agent_gross,
                timing_base_gross,
            ) = self._timing_reward_vs_immediate(
                event_start_idx=event_start_idx,
                exec_idx=int(self._i),
                horizon_idx=int(self._ep_end),
                pending=pending_for_exec,
                target_dir=target_for_exec,
                prev_dir=event_prev_dir,
                prev_pos_units=event_prev_pos,
            )
            reward_improvement = float(agent_net - base_net)
            agent_trade_units, did_trade = self._apply_pending_order(
                self.pending_order,
                self.pending_target_dir,
            )
            self.pending_order = PENDING_NONE
            self.pending_target_dir = 0
            self.current_event_type = EVENT_NONE
            self.event_started_at_idx = -1
            self.event_prev_dir = 0
            self.event_start_pos_units = int(self.agent_pos_units)
            self.event_start_price = np.nan
            self.event_target_dir = 0
        else:
            agent_trade_units, did_trade = 0, False

        if self.force_flat_on_dir_flip and desired_dir != 0 and self.agent_pos_units != 0:
            if int(np.sign(self.agent_pos_units)) != desired_dir:
                prev_force = int(self.agent_pos_units)
                self.agent_pos_units = 0
                agent_trade_units += abs(prev_force)
                did_trade = did_trade or (prev_force != 0)
                self.pending_order = PENDING_NONE
                self.pending_target_dir = 0
                self.current_event_type = EVENT_NONE
                self.event_started_at_idx = -1
                self.event_prev_dir = 0
                self.event_start_pos_units = int(self.agent_pos_units)
                self.event_start_price = np.nan
                self.event_target_dir = 0

        baseline_trade_units = self._apply_baseline(desired_dir)
        self._update_agent_position_state(prev_agent, price)
        mae_pen = float(timing_mae_pen if execute_now else 0.0)
        churn_pen = 0.0
        low_conf_pen = 0.0
        reward = float(reward_improvement - mae_pen) if execute_now else 0.0

        if self.agent_pos_units != 0 and np.isfinite(self.agent_entry_price) and self.agent_entry_price > 0:
            next_price = price * (1.0 + ret_next)
            self.unrealized = int(np.sign(self.agent_pos_units)) * (next_price / self.agent_entry_price - 1.0) * abs(self.agent_pos_units)
            self.time_in_pos += 1
        else:
            self.unrealized = 0.0
            self.time_in_pos = 0

        info = {
            "timestamp": self.df.loc[self._i, "timestamp"],
            "action_idx": action_idx,
            "action_name": "EXECUTE" if action_idx == ACTION_EXECUTE else "WAIT",
            "did_trade": bool(did_trade),
            "did_execute_event": bool(execute_now),
            "forced_execute_event": bool(forced_execute and execute_now),
            "agent_pos_units": int(self.agent_pos_units),
            "baseline_pos_units": int(self.baseline_pos_units),
            "htf_dir": desired_dir,
            "htf_conf": htf_conf,
            "time_since_flip_min": time_since_flip,
            "event_started": bool(event_started),
            "event_type_code": active_event_type,
            "event_type": self._event_name(active_event_type),
            "pending_order": active_pending_name,
            "pending_target_dir": active_pending_target_dir,
            "reward": float(reward),
            "reward_improvement": float(reward_improvement),
            "agent_net": float(agent_net),
            "baseline_net": float(base_net),
            "mae_penalty": float(mae_pen),
            "churn_penalty": float(churn_pen),
            "low_conf_penalty": float(low_conf_pen),
            "timing_agent_gross": float(timing_agent_gross),
            "timing_baseline_gross": float(timing_base_gross),
            "agent_trade_units": int(agent_trade_units),
            "baseline_trade_units": int(baseline_trade_units),
            "ret_next": float(ret_next),
            "close": price,
            "prev_agent_pos_units": int(prev_agent),
            "prev_baseline_pos_units": int(prev_base),
            "unrealized": float(self.unrealized),
            "time_in_pos": int(self.time_in_pos),
        }

        done = self._i >= self._ep_end or self._i >= len(self.df) - 2
        if done:
            return self._get_obs(), float(reward), True, info
        self._i += 1
        return self._get_obs(), float(reward), False, info


def make_execution_env(
    *,
    df: pd.DataFrame,
    feature_cols: list[str],
    config: ExecutionEnvConfig | None = None,
    **overrides: Any,
) -> DirectionGatedExecutionEnv:
    cfg = config or ExecutionEnvConfig()
    if overrides:
        valid = {f.name for f in fields(ExecutionEnvConfig)}
        filtered = {k: v for k, v in overrides.items() if k in valid}
        cfg = replace(cfg, **filtered)
    return DirectionGatedExecutionEnv(df=df, feature_cols=feature_cols, **asdict(cfg))
