from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

import numpy as np
import pandas as pd


ACTION_WAIT = 0
ACTION_ENTER = 1
ACTION_SCALE_IN = 2
ACTION_SCALE_OUT = 3
ACTION_EXIT = 4
N_EXEC_ACTIONS = 5


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
    mae_penalty_lambda: float = 1.0
    low_conf_threshold: float = 0.35
    low_conf_penalty_lambda: float = 0.00015

    episode_mode: str = "flip_window"  # flip_window|day
    flip_start_delay_min: int = 0
    flip_window_minutes: int = 20
    seed: int = 7


class DirectionGatedExecutionEnv:
    """
    1m execution env conditioned on frozen 15m intent.
    Reward is relative to a dumb baseline:
        reward = (agent_net - baseline_net) - MAE_pen - churn_pen - low_conf_pen.
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
        mae_penalty_lambda: float = 1.0,
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
        self.obs_dim = self._base_obs_dim + (4 if self.add_position_features else 0)
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
            flips = np.where((htf_sign != np.r_[htf_sign[0], htf_sign[:-1]]) & (htf_sign != 0))[0]
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

    def _trade_cost(self, units_delta: int) -> float:
        if units_delta <= 0:
            return 0.0
        bps = (self.spread_bps + self.slippage_bps) / 10000.0
        return float(units_delta) * (bps + self.commission_per_unit_ret + self.trade_penalty_ret)

    def _get_obs(self) -> np.ndarray:
        obs = np.empty(self.obs_dim, dtype=np.float32)
        obs[: self._base_obs_dim] = self._feature_matrix[self._i]
        if self.add_position_features:
            cursor = self._base_obs_dim
            obs[cursor] = float(self.agent_pos_units / max(1, self.max_units))
            obs[cursor + 1] = float(self.baseline_pos_units / max(1, self.baseline_units))
            obs[cursor + 2] = float(np.tanh(self.unrealized * 100.0))
            obs[cursor + 3] = float(np.clip(self._time_since_flip[self._i] / 30.0, 0.0, 1.0))
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

    def _apply_agent_action(
        self,
        action_idx: int,
        desired_dir: int,
        time_since_flip_min: float,
    ) -> tuple[int, bool]:
        prev = int(self.agent_pos_units)
        pos = prev
        if self.force_flat_on_dir_flip and desired_dir != 0 and pos != 0 and int(np.sign(pos)) != desired_dir:
            pos = 0
        abs_pos = abs(pos)
        entry_window_ok = (
            float(time_since_flip_min) >= float(self.entry_min_since_flip)
            and float(time_since_flip_min) <= float(self.entry_max_since_flip)
        )

        if action_idx == ACTION_WAIT:
            pass
        elif action_idx == ACTION_ENTER:
            if desired_dir != 0 and pos == 0 and entry_window_ok:
                pos = desired_dir * min(self.entry_units, self.max_units)
        elif action_idx == ACTION_SCALE_IN:
            if desired_dir != 0 and entry_window_ok and (pos == 0 or int(np.sign(pos)) == desired_dir):
                new_abs = min(self.max_units, abs_pos + 1)
                pos = desired_dir * new_abs
        elif action_idx == ACTION_SCALE_OUT:
            if pos != 0:
                new_abs = max(0, abs_pos - 1)
                pos = int(np.sign(pos)) * new_abs
        elif action_idx == ACTION_EXIT:
            pos = 0

        if desired_dir == 0 and action_idx in (ACTION_ENTER, ACTION_SCALE_IN):
            pos = prev
        if pos != 0 and desired_dir != 0 and int(np.sign(pos)) != desired_dir:
            pos = prev

        self.agent_pos_units = int(pos)
        traded_units = abs(int(self.agent_pos_units) - prev)
        return traded_units, self.agent_pos_units != prev

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

        agent_trade_units, did_trade = self._apply_agent_action(action_idx, desired_dir, time_since_flip)
        baseline_trade_units = self._apply_baseline(desired_dir)

        if prev_agent == 0 and self.agent_pos_units != 0:
            self.agent_entry_price = price
        elif prev_agent != 0 and self.agent_pos_units == 0:
            if np.isfinite(self.agent_entry_price) and self.agent_entry_price > 0:
                signed = int(np.sign(prev_agent))
                self.realized_agent += signed * (price / self.agent_entry_price - 1.0) * abs(prev_agent)
            self.agent_entry_price = np.nan
            self.time_in_pos = 0
        elif prev_agent != 0 and self.agent_pos_units != 0 and np.sign(prev_agent) == np.sign(self.agent_pos_units):
            if abs(self.agent_pos_units) > abs(prev_agent) and np.isfinite(self.agent_entry_price):
                add_units = abs(self.agent_pos_units) - abs(prev_agent)
                old_units = abs(prev_agent)
                self.agent_entry_price = (
                    old_units * float(self.agent_entry_price) + add_units * price
                ) / max(1, old_units + add_units)

        agent_cost = self._trade_cost(agent_trade_units)
        base_cost = self._trade_cost(baseline_trade_units)
        agent_gross = float(self.agent_pos_units) * ret_next
        base_gross = float(self.baseline_pos_units) * ret_next
        agent_net = agent_gross - agent_cost
        base_net = base_gross - base_cost
        improvement = agent_net - base_net

        mae_pen = 0.0
        if self.agent_pos_units != 0 and np.isfinite(self.agent_entry_price) and self.agent_entry_price > 0 and self._i + 1 < len(self.df):
            entry = float(self.agent_entry_price)
            high_n = float(self._high[self._i + 1])
            low_n = float(self._low[self._i + 1])
            if self.agent_pos_units > 0:
                adverse = max(0.0, (entry - low_n) / entry)
            else:
                adverse = max(0.0, (high_n - entry) / entry)
            mae_pen = self.mae_penalty_lambda * adverse * abs(self.agent_pos_units)

        churn_pen = self.churn_penalty_ret * float(agent_trade_units)
        low_conf_pen = 0.0
        if self.agent_pos_units != 0 and htf_conf < self.low_conf_threshold:
            low_conf_pen = self.low_conf_penalty_lambda * (self.low_conf_threshold - htf_conf) * abs(self.agent_pos_units)
        reward = improvement - mae_pen - churn_pen - low_conf_pen

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
            "did_trade": bool(did_trade),
            "agent_pos_units": int(self.agent_pos_units),
            "baseline_pos_units": int(self.baseline_pos_units),
            "htf_dir": desired_dir,
            "htf_conf": htf_conf,
            "time_since_flip_min": time_since_flip,
            "reward": float(reward),
            "reward_improvement": float(improvement),
            "agent_net": float(agent_net),
            "baseline_net": float(base_net),
            "mae_penalty": float(mae_pen),
            "churn_penalty": float(churn_pen),
            "low_conf_penalty": float(low_conf_pen),
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
