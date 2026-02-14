from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
import math
import numpy as np
import pandas as pd


def sincos_time_of_day(ts: pd.Timestamp) -> Tuple[float, float]:
    minutes = ts.hour * 60 + ts.minute
    angle = 2.0 * math.pi * (minutes / 1440.0)
    return math.sin(angle), math.cos(angle)


class TradingEnv:
    """
    Minimal Gym-like trading env:
      reset() -> obs
      step(action) -> obs, reward, done, info

    Actions: continuous signed exposure in [-1, +1]
      +1.0 = max long, 0.0 = flat, -1.0 = max short
    Episode: one trading day (requires df['day_id'])
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        feature_array_col: str = "features",
        add_time_features: bool = True,
        add_position_features: bool = True,
        commission_per_trade: float = 0.0,
        slippage_bps: float = 0.0,
        spread_bps: float = 0.0,
        flip_penalty_ret: float = 0.0,
        trade_penalty_ret: float = 0.0,
        hold_penalty_ret: float = 0.0,
        reward_on_exit: bool = False,
        reward_exit_bonus: bool = False,
        exit_pivot_bonus_ret: float = 0.0,
        pivot_long_col: str = "p_pivot_long",
        pivot_short_col: str = "p_pivot_short",
        force_flat_at_close: bool = True,
        carry_positions_across_days: bool = False,
        allow_direct_flip: bool = True,
        use_convex_reward: bool = False,
        convex_k1: float = 1.0,
        convex_k2: float = 0.5,
        convex_theta: float = 0.01,
        convex_pivot_k: float = 0.0,
        convex_risk_lambda: float = 0.0,
        convex_bonus_cap: float = 0.0,
        convex_bonus_scale: float = 1.0,
        saturation_threshold: float = 0.9,
        saturation_penalty_ret: float = 0.0,
        # Legacy args retained for checkpoint/env override compatibility.
        convex_directional_bonus_only: bool = True,
        convex_wrong_side_scale: float = 0.0,
        convex_mfe_thresholds: Tuple[float, ...] = (1.0, 2.0, 3.0),
        convex_mfe_bonuses: Tuple[float, ...] = (0.1, 0.2, 0.3),
        action_deadband: float = 1e-3,
        dir_switch_penalty_ret: float = 0.0,
        size_change_penalty_ret: float = 0.0,
        seed: int = 7,
    ):
        self.df = df.copy()
        if "timestamp" not in self.df.columns:
            raise ValueError("df must contain 'timestamp'")
        if "day_id" not in self.df.columns:
            raise ValueError("df must contain 'day_id' (identifies each trading day)")
        if "close" not in self.df.columns:
            raise ValueError("df must contain 'close'")

        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)

        self.feature_cols = feature_cols
        self.feature_array_col = feature_array_col
        self.add_time_features = add_time_features
        self.add_position_features = add_position_features

        self.commission_per_trade = float(commission_per_trade)
        self.slippage_bps = float(slippage_bps)
        self.spread_bps = float(spread_bps)
        # flip_penalty_ret is in return units (not dollars).
        self.flip_penalty_ret = float(flip_penalty_ret)
        # trade_penalty_ret is applied per entry/exit (return units).
        self.trade_penalty_ret = float(trade_penalty_ret)
        # hold_penalty_ret is applied per bar while in position (return units).
        self.hold_penalty_ret = float(hold_penalty_ret)
        # reward_on_exit pays realized return only when closing a trade.
        self.reward_on_exit = bool(reward_on_exit)
        # reward_exit_bonus adds realized return on exit even when dense rewards are enabled.
        self.reward_exit_bonus = bool(reward_exit_bonus)
        # exit_pivot_bonus_ret scales a bonus on exit using pivot probabilities.
        self.exit_pivot_bonus_ret = float(exit_pivot_bonus_ret)
        self.pivot_long_col = str(pivot_long_col)
        self.pivot_short_col = str(pivot_short_col)
        self.force_flat_at_close = bool(force_flat_at_close)
        self.carry_positions_across_days = bool(carry_positions_across_days)
        self.allow_direct_flip = bool(allow_direct_flip)
        self.use_convex_reward = bool(use_convex_reward)
        self.convex_k1 = float(convex_k1)
        self.convex_k2 = float(convex_k2)
        self.convex_theta = float(convex_theta)
        self.convex_pivot_k = float(convex_pivot_k)
        self.convex_risk_lambda = max(0.0, float(convex_risk_lambda))
        self.convex_bonus_cap = max(0.0, float(convex_bonus_cap))
        self.convex_bonus_scale = float(convex_bonus_scale)
        self.saturation_threshold = float(np.clip(saturation_threshold, 0.0, 1.0))
        self.saturation_penalty_ret = max(0.0, float(saturation_penalty_ret))
        self._legacy_convex_directional_bonus_only = bool(convex_directional_bonus_only)
        self._legacy_convex_wrong_side_scale = float(convex_wrong_side_scale)
        self.convex_mfe_thresholds = tuple(float(x) for x in convex_mfe_thresholds)
        self.action_deadband = max(0.0, float(action_deadband))
        self.dir_switch_penalty_ret = max(0.0, float(dir_switch_penalty_ret))
        self.size_change_penalty_ret = max(0.0, float(size_change_penalty_ret))
        if not self.convex_mfe_thresholds:
            self.convex_mfe_thresholds = ()
        raw_bonuses = list(float(x) for x in convex_mfe_bonuses)
        if not raw_bonuses and self.convex_mfe_thresholds:
            raw_bonuses = [0.0 for _ in self.convex_mfe_thresholds]
        if len(raw_bonuses) < len(self.convex_mfe_thresholds):
            raw_bonuses.extend([raw_bonuses[-1] if raw_bonuses else 0.0] * (len(self.convex_mfe_thresholds) - len(raw_bonuses)))
        self.convex_mfe_bonuses = tuple(raw_bonuses[: len(self.convex_mfe_thresholds)])

        self.rng = np.random.default_rng(seed)

        # next-bar return
        self.df["ret_next"] = self.df["close"].shift(-1) / self.df["close"] - 1.0
        self.df["is_last_of_day"] = self.df["day_id"].shift(-1) != self.df["day_id"]

        # Precompute dense arrays for fast per-step access.
        self._close = self.df["close"].to_numpy(dtype=np.float64, copy=False)
        self._ret_next = self.df["ret_next"].to_numpy(dtype=np.float64, copy=False)
        self._is_last_of_day = self.df["is_last_of_day"].to_numpy(dtype=bool, copy=False)
        self._high = self.df["high"].to_numpy(dtype=np.float64, copy=False) if "high" in self.df.columns else self._close
        self._low = self.df["low"].to_numpy(dtype=np.float64, copy=False) if "low" in self.df.columns else self._close
        self._atr = self.df["atr"].to_numpy(dtype=np.float64, copy=False) if "atr" in self.df.columns else None
        self._atr_pct = self.df["atr_pct"].to_numpy(dtype=np.float64, copy=False) if "atr_pct" in self.df.columns else None

        if self.feature_cols is not None:
            self._feature_matrix = self.df.loc[:, self.feature_cols].to_numpy(
                dtype=np.float32, copy=False
            )
        else:
            if self.feature_array_col not in self.df.columns:
                raise ValueError(f"Provide feature_cols or df['{self.feature_array_col}']")
            raw_arrays = self.df[self.feature_array_col].to_list()
            if not raw_arrays:
                raise ValueError("Empty feature array column.")
            first = np.asarray(raw_arrays[0], dtype=np.float32)
            if first.ndim != 1:
                raise ValueError("features must be 1D")
            base_dim = int(first.shape[0])
            rows: list[np.ndarray] = [first]
            for arr in raw_arrays[1:]:
                vec = np.asarray(arr, dtype=np.float32)
                if vec.ndim != 1 or int(vec.shape[0]) != base_dim:
                    raise ValueError("All feature vectors must be 1D and same length.")
                rows.append(vec)
            self._feature_matrix = np.stack(rows, axis=0)

        self._base_dim = int(self._feature_matrix.shape[1])

        if self.add_time_features:
            ts = self.df["timestamp"]
            minutes = (
                ts.dt.hour.to_numpy(dtype=np.float32, copy=False) * 60.0
                + ts.dt.minute.to_numpy(dtype=np.float32, copy=False)
            )
            angle = (2.0 * math.pi * minutes) / 1440.0
            self._time_sin = np.sin(angle).astype(np.float32, copy=False)
            self._time_cos = np.cos(angle).astype(np.float32, copy=False)
        else:
            self._time_sin = np.empty((0,), dtype=np.float32)
            self._time_cos = np.empty((0,), dtype=np.float32)

        if self.pivot_long_col in self.df.columns:
            self._pivot_long = self.df[self.pivot_long_col].to_numpy(
                dtype=np.float64, copy=False
            )
        else:
            self._pivot_long = np.zeros(len(self.df), dtype=np.float64)

        if self.pivot_short_col in self.df.columns:
            self._pivot_short = self.df[self.pivot_short_col].to_numpy(
                dtype=np.float64, copy=False
            )
        else:
            self._pivot_short = np.zeros(len(self.df), dtype=np.float64)

        self.day_starts = self.df.index[self.df["day_id"].shift(1) != self.df["day_id"]].to_list()
        if len(self.day_starts) == 0:
            raise ValueError("No day boundaries found. Ensure 'day_id' changes across days.")

        self._day_ptr = -1
        self._i = 0

        self.position = 0.0
        self.entry_price = np.nan
        self.time_in_pos = 0
        self.unrealized_pnl = 0.0
        self.realized_pnl_today = 0.0
        self._mfe_max_atr = 0.0
        self._mfe_awarded: set[int] = set()

        self.obs_dim = self._compute_obs_dim()

    def _compute_obs_dim(self) -> int:
        extra = 0
        if self.add_time_features:
            extra += 2
        if self.add_position_features:
            extra += 4
        return self._base_dim + extra

    def _get_base_features(self, idx: int) -> np.ndarray:
        return self._feature_matrix[idx]

    def _get_obs(self) -> np.ndarray:
        feats = self._get_base_features(self._i)
        obs = np.empty(self.obs_dim, dtype=np.float32)
        cursor = 0
        obs[cursor : cursor + self._base_dim] = feats
        cursor += self._base_dim

        if self.add_time_features:
            obs[cursor] = self._time_sin[self._i]
            obs[cursor + 1] = self._time_cos[self._i]
            cursor += 2

        if self.add_position_features:
            obs[cursor] = float(self.position)
            obs[cursor + 1] = float(self.time_in_pos)
            obs[cursor + 2] = float(self.unrealized_pnl)
            obs[cursor + 3] = float(self.realized_pnl_today)
            cursor += 4

        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(f"Obs dim mismatch: got {obs.shape[0]} expected {self.obs_dim}")
        return obs

    def reset(self, day_ptr: Optional[int] = None) -> np.ndarray:
        if day_ptr is None:
            self._day_ptr = (self._day_ptr + 1) % len(self.day_starts)
        else:
            self._day_ptr = int(day_ptr) % len(self.day_starts)

        self._i = int(self.day_starts[self._day_ptr])

        if not self.carry_positions_across_days:
            self.position = 0.0
            self.entry_price = np.nan
            self.time_in_pos = 0
            self.unrealized_pnl = 0.0
            self.realized_pnl_today = 0.0
            self._mfe_max_atr = 0.0
            self._mfe_awarded.clear()
        else:
            self.realized_pnl_today = 0.0
            if (not self._is_flat(self.position)) and np.isfinite(self.entry_price):
                price = float(self._close[self._i])
                self.unrealized_pnl = (price / self.entry_price - 1.0) * float(self.position)
            else:
                self.unrealized_pnl = 0.0
                self.time_in_pos = 0
                self._mfe_max_atr = 0.0
                self._mfe_awarded.clear()
        return self._get_obs()

    def _trade_cost_ret(self, price: float) -> float:
        # bps already represent fractional cost vs notional
        bps_cost = (self.slippage_bps + self.spread_bps) / 10000.0
        commission_ret = (self.commission_per_trade / price) if price else 0.0
        return commission_ret + bps_cost

    def _is_flat(self, pos: float) -> bool:
        return abs(float(pos)) <= self.action_deadband

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        try:
            action_f = float(action)
        except (TypeError, ValueError):
            action_f = 0.0
        if not np.isfinite(action_f):
            action_f = 0.0
        action_f = float(np.clip(action_f, -1.0, 1.0))

        prev_pos = self.position
        desired_pos = 0.0 if abs(action_f) <= self.action_deadband else action_f

        price = float(self._close[self._i])
        ret_next = float(self._ret_next[self._i])
        ret_next_missing = not np.isfinite(ret_next)
        if ret_next_missing:
            ret_next = 0.0
        is_last = bool(self._is_last_of_day[self._i])

        reward_costs = 0.0
        reward_pnl = 0.0
        reward_pivot_bonus = 0.0
        reward_convex = 0.0
        reward_convex_linear = 0.0
        reward_convex_bonus = 0.0
        reward_convex_risk_penalty = 0.0
        reward_saturation_penalty = 0.0
        convex_term = 0.0
        atr_scale = np.nan
        convex_vol_proxy = np.nan
        pivot_anchor = 0.0
        mfe_bonus = 0.0
        switch_penalty = 0.0
        size_penalty = 0.0
        mfe_atr = None
        flipped = (
            (not self._is_flat(self.position))
            and (not self._is_flat(desired_pos))
            and (self.position * desired_pos < 0.0)
        )
        blocked_flip = flipped and not self.allow_direct_flip
        if blocked_flip:
            desired_pos = 0.0
            flipped = False

        trade_units = abs(desired_pos - prev_pos)
        did_trade = trade_units > self.action_deadband
        prev_dir = 0 if self._is_flat(prev_pos) else int(np.sign(prev_pos))
        new_dir = 0 if self._is_flat(desired_pos) else int(np.sign(desired_pos))
        did_dir_switch = (prev_dir != 0) and (new_dir != 0) and (prev_dir != new_dir)
        size_delta = abs(abs(desired_pos) - abs(prev_pos))
        if did_dir_switch and self.dir_switch_penalty_ret > 0.0:
            switch_penalty = self.dir_switch_penalty_ret
        if self.size_change_penalty_ret > 0.0 and size_delta > self.action_deadband:
            size_penalty = self.size_change_penalty_ret * size_delta
        if did_trade:
            # Cost scales with position change size:
            # 0 = hold, 1 = full enter/exit, 2 = full direct flip.
            leg_cost = self._trade_cost_ret(price) + self.trade_penalty_ret
            reward_costs += float(trade_units) * leg_cost
            if flipped:
                reward_costs += self.flip_penalty_ret

            # Realize PnL on the reduced/closed fraction of existing exposure.
            if (not self._is_flat(prev_pos)) and np.isfinite(self.entry_price):
                same_sign = prev_pos * desired_pos > 0.0
                prev_abs = abs(prev_pos)
                desired_abs = abs(desired_pos)
                closed_units = max(0.0, prev_abs - desired_abs) if same_sign else prev_abs
                if closed_units > self.action_deadband:
                    move_per_unit = (price / self.entry_price - 1.0) * float(np.sign(prev_pos))
                    move = move_per_unit * closed_units
                    self.realized_pnl_today += move
                else:
                    move = 0.0
                if (self.reward_on_exit or self.reward_exit_bonus) and not self.use_convex_reward:
                    reward_pnl += move
                if self.exit_pivot_bonus_ret:
                    if prev_pos > 0:
                        pivot_val = float(self._pivot_short[self._i])
                    else:
                        pivot_val = float(self._pivot_long[self._i])
                    if np.isfinite(pivot_val):
                        scale = closed_units if closed_units > self.action_deadband else 0.0
                        reward_pivot_bonus = self.exit_pivot_bonus_ret * pivot_val * scale
                        if not self.use_convex_reward:
                            reward_pnl += reward_pivot_bonus

            if self._is_flat(desired_pos):
                self.entry_price = np.nan
                self.time_in_pos = 0
                self.position = 0.0
            else:
                if self._is_flat(prev_pos) or (prev_pos * desired_pos <= 0.0) or (not np.isfinite(self.entry_price)):
                    self.entry_price = price
                    self.time_in_pos = 0
                    self._mfe_max_atr = 0.0
                    self._mfe_awarded.clear()
                else:
                    prev_abs = abs(prev_pos)
                    desired_abs = abs(desired_pos)
                    if desired_abs > prev_abs + self.action_deadband:
                        add_units = desired_abs - prev_abs
                        self.entry_price = (
                            (prev_abs * float(self.entry_price)) + (add_units * price)
                        ) / max(desired_abs, 1e-12)
                self.position = desired_pos

        if self.use_convex_reward:
            r = ret_next
            delta_s = price * ret_next
            atr = np.nan
            if self._atr is not None:
                atr = float(self._atr[self._i])
            elif self._atr_pct is not None:
                atr = float(self._atr_pct[self._i]) * price
            if not np.isfinite(atr) or atr <= 0.0:
                atr = np.nan
            if np.isfinite(atr) and atr > 0.0 and price > 0.0:
                atr_scale = max(atr / price, 1e-6)
                convex_term = (abs(r) * abs(r)) / atr_scale
            vol_proxy = float(atr_scale) if np.isfinite(atr_scale) else abs(r)
            if (not np.isfinite(vol_proxy)) or vol_proxy <= 0.0:
                vol_proxy = max(abs(r), 1e-6)
            convex_vol_proxy = vol_proxy
            pos_ret = float(self.position) * r
            reward_convex_linear = pos_ret * self.convex_k1
            bonus_raw = (
                float(self.position) * self.convex_k2 * (r * abs(r)) / max(vol_proxy, 1e-6)
            )
            if self.convex_bonus_cap > 0.0:
                reward_convex_bonus = self.convex_bonus_cap * math.tanh(
                    bonus_raw / self.convex_bonus_cap
                )
            else:
                reward_convex_bonus = bonus_raw
            reward_convex_bonus *= self.convex_bonus_scale
            reward_convex = reward_convex_linear + reward_convex_bonus
            # Pivot-anchor term rewards directional alignment with pivot edge.
            pivot_edge = float(self._pivot_long[self._i]) - float(self._pivot_short[self._i])
            if np.isfinite(pivot_edge):
                pivot_anchor = self.convex_pivot_k * float(self.position) * pivot_edge
                reward_convex += pivot_anchor
            reward_convex -= self.convex_theta * abs(self.position)
            if self.convex_risk_lambda > 0.0:
                reward_convex_risk_penalty = (
                    self.convex_risk_lambda * (abs(self.position) ** 2) * (vol_proxy ** 2)
                )
                reward_convex -= reward_convex_risk_penalty
            pos_abs = abs(self.position)
            if self.saturation_penalty_ret > 0.0 and pos_abs > self.saturation_threshold:
                reward_saturation_penalty = self.saturation_penalty_ret * (
                    pos_abs - self.saturation_threshold
                )
                reward_convex -= reward_saturation_penalty
            reward_pnl = reward_convex
            if (not self._is_flat(self.position)) and np.isfinite(atr) and np.isfinite(self.entry_price):
                high = float(self._high[self._i])
                low = float(self._low[self._i])
                if np.isfinite(high) and np.isfinite(low):
                    if self.position > 0:
                        mfe_atr = (high - self.entry_price) / atr
                    else:
                        mfe_atr = (self.entry_price - low) / atr
                    if np.isfinite(mfe_atr):
                        if mfe_atr > self._mfe_max_atr:
                            self._mfe_max_atr = float(mfe_atr)
                        for idx, th in enumerate(self.convex_mfe_thresholds):
                            if idx not in self._mfe_awarded and self._mfe_max_atr >= th:
                                bonus = self.convex_mfe_bonuses[idx]
                                mfe_bonus += bonus
                                self._mfe_awarded.add(idx)
            reward_pnl += mfe_bonus
        # pnl reward for holding through next bar (only in mark-to-market mode)
        elif not self.reward_on_exit:
            reward_pnl = float(self.position) * ret_next

        if (not self._is_flat(self.position)) and np.isfinite(self.entry_price):
            next_price = price * (1.0 + ret_next)
            self.unrealized_pnl = (next_price / self.entry_price - 1.0) * float(self.position)
            self.time_in_pos += 1
        else:
            self.unrealized_pnl = 0.0
            self.time_in_pos = 0

        hold_penalty = self.hold_penalty_ret * abs(self.position)
        reward = reward_pnl - reward_costs - hold_penalty - switch_penalty - size_penalty

        info: Dict[str, Any] = {
            "price": price,
            "action": action_f,
            "ret_next": ret_next,
            "ret_next_missing": ret_next_missing,
            "pos": self.position,
            "prev_pos": prev_pos,
            "did_trade": did_trade,
            "did_flip": flipped,
            "reward_pnl": reward_pnl,
            "reward_costs": reward_costs,
            "reward_pivot_bonus": reward_pivot_bonus,
            "reward_convex": reward_convex if self.use_convex_reward else None,
            "reward_convex_linear": reward_convex_linear if self.use_convex_reward else None,
            "reward_convex_bonus": reward_convex_bonus if self.use_convex_reward else None,
            "reward_convex_risk_penalty": reward_convex_risk_penalty if self.use_convex_reward else None,
            "convex_term": convex_term if self.use_convex_reward else None,
            "convex_atr_scale": float(atr_scale) if self.use_convex_reward and np.isfinite(atr_scale) else None,
            "convex_vol_proxy": float(convex_vol_proxy) if self.use_convex_reward and np.isfinite(convex_vol_proxy) else None,
            "reward_pivot_anchor": pivot_anchor if self.use_convex_reward else None,
            "reward_saturation_penalty": reward_saturation_penalty,
            "mfe_atr": float(mfe_atr) if mfe_atr is not None and np.isfinite(mfe_atr) else None,
            "mfe_bonus": mfe_bonus if self.use_convex_reward else None,
            "reward_switch_penalty": switch_penalty,
            "reward_size_penalty": size_penalty,
            "did_dir_switch": did_dir_switch,
            "size_delta": size_delta,
            "realized_pnl_today": self.realized_pnl_today,
            "unrealized_pnl": self.unrealized_pnl,
            "flip_blocked": blocked_flip,
        }

        if is_last or (self._i >= len(self.df) - 2):
            if self.force_flat_at_close and (not self._is_flat(self.position)):
                close_units = abs(self.position)
                exit_cost = self._trade_cost_ret(price) * close_units
                reward_costs += exit_cost
                if self.trade_penalty_ret:
                    reward_costs += self.trade_penalty_ret * close_units
                info["forced_flat_cost"] = exit_cost

                if np.isfinite(self.entry_price):
                    move = (price / self.entry_price - 1.0) * float(self.position)
                    self.realized_pnl_today += move
                    if (self.reward_on_exit or self.reward_exit_bonus) and not self.use_convex_reward:
                        reward_pnl += move
                    if self.exit_pivot_bonus_ret:
                        if self.position > 0:
                            pivot_val = float(self._pivot_short[self._i])
                        else:
                            pivot_val = float(self._pivot_long[self._i])
                        if np.isfinite(pivot_val):
                            reward_pivot_bonus = self.exit_pivot_bonus_ret * pivot_val * close_units
                            if not self.use_convex_reward:
                                reward_pnl += reward_pivot_bonus

                self.position = 0.0
                self.entry_price = np.nan
                self.unrealized_pnl = 0.0
                self.time_in_pos = 0
                reward = reward_pnl - reward_costs - hold_penalty - switch_penalty - size_penalty
                info["reward_pnl"] = reward_pnl
                info["reward_costs"] = reward_costs
                info["reward_pivot_bonus"] = reward_pivot_bonus
                info["reward_convex"] = reward_convex if self.use_convex_reward else None
                info["reward_convex_linear"] = reward_convex_linear if self.use_convex_reward else None
                info["reward_convex_bonus"] = reward_convex_bonus if self.use_convex_reward else None
                info["reward_convex_risk_penalty"] = (
                    reward_convex_risk_penalty if self.use_convex_reward else None
                )
                info["convex_term"] = convex_term if self.use_convex_reward else None
                info["convex_atr_scale"] = float(atr_scale) if self.use_convex_reward and np.isfinite(atr_scale) else None
                info["convex_vol_proxy"] = (
                    float(convex_vol_proxy)
                    if self.use_convex_reward and np.isfinite(convex_vol_proxy)
                    else None
                )
                info["reward_pivot_anchor"] = pivot_anchor if self.use_convex_reward else None
                info["reward_saturation_penalty"] = reward_saturation_penalty
                info["mfe_atr"] = float(mfe_atr) if mfe_atr is not None and np.isfinite(mfe_atr) else None
                info["mfe_bonus"] = mfe_bonus if self.use_convex_reward else None
                info["pos"] = self.position
                info["realized_pnl_today"] = self.realized_pnl_today
                info["unrealized_pnl"] = self.unrealized_pnl

            done = True
            return self._get_obs(), float(reward), done, info

        self._i += 1
        done = False
        return self._get_obs(), float(reward), done, info


class VecTradingEnv:
    """
    Simple vectorized wrapper around multiple TradingEnv instances.

    This runs envs in a synchronous loop (no multiprocessing) and auto-resets
    environments that finish an episode.
    """

    def __init__(
        self,
        envs: list[TradingEnv],
        *,
        auto_reset: bool = True,
        stagger_reset: bool = True,
    ):
        if not envs:
            raise ValueError("VecTradingEnv requires at least one env.")
        self.envs = envs
        self.n_envs = len(envs)
        self.auto_reset = bool(auto_reset)
        self.stagger_reset = bool(stagger_reset)
        self._did_initial_reset = False

        obs_dim = envs[0].obs_dim
        for env in envs[1:]:
            if env.obs_dim != obs_dim:
                raise ValueError("All envs must have the same obs_dim.")
        self.obs_dim = obs_dim

    def reset(self, day_ptrs: Optional[list[int]] = None) -> np.ndarray:
        if day_ptrs is not None and len(day_ptrs) != self.n_envs:
            raise ValueError("day_ptrs length must match number of envs.")

        if day_ptrs is None and self.stagger_reset and not self._did_initial_reset:
            base_days = len(self.envs[0].day_starts)
            day_ptrs = [i % max(1, base_days) for i in range(self.n_envs)]
            self._did_initial_reset = True

        obs = []
        for i, env in enumerate(self.envs):
            if day_ptrs is None:
                obs.append(env.reset())
            else:
                obs.append(env.reset(day_ptr=day_ptrs[i]))
        return np.stack(obs, axis=0)

    def step(
        self, actions: np.ndarray | list[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        if len(actions) != self.n_envs:
            raise ValueError("Actions length must match number of envs.")

        obs, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            if isinstance(action, (list, tuple, np.ndarray)):
                arr = np.asarray(action, dtype=np.float32).reshape(-1)
                action_value = float(arr[0]) if arr.size else 0.0
            else:
                action_value = float(action)
            o, r, d, info = env.step(action_value)
            if d and self.auto_reset:
                o = env.reset()
            obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        return (
            np.stack(obs, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            infos,
        )
