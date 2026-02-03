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

    Actions: 0=FLAT, 1=LONG, 2=SHORT
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
        allow_direct_flip: bool = True,
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
        self.allow_direct_flip = bool(allow_direct_flip)

        self.rng = np.random.default_rng(seed)

        # next-bar return
        self.df["ret_next"] = self.df["close"].shift(-1) / self.df["close"] - 1.0
        self.df["is_last_of_day"] = self.df["day_id"].shift(-1) != self.df["day_id"]

        self.day_starts = self.df.index[self.df["day_id"].shift(1) != self.df["day_id"]].to_list()
        if len(self.day_starts) == 0:
            raise ValueError("No day boundaries found. Ensure 'day_id' changes across days.")

        self._day_ptr = -1
        self._i = 0

        self.position = 0
        self.entry_price = np.nan
        self.time_in_pos = 0
        self.unrealized_pnl = 0.0
        self.realized_pnl_today = 0.0

        self.obs_dim = self._compute_obs_dim()

    def _compute_obs_dim(self) -> int:
        if self.feature_cols is not None:
            base_dim = len(self.feature_cols)
        else:
            if self.feature_array_col not in self.df.columns:
                raise ValueError(f"Provide feature_cols or df['{self.feature_array_col}']")
            first = self.df[self.feature_array_col].dropna().iloc[0]
            base_dim = int(np.asarray(first).shape[0])

        extra = 0
        if self.add_time_features:
            extra += 2
        if self.add_position_features:
            extra += 4
        return base_dim + extra

    def _get_base_features(self, idx: int) -> np.ndarray:
        if self.feature_cols is not None:
            return self.df.loc[idx, self.feature_cols].to_numpy(dtype=np.float32)
        arr = np.asarray(self.df.loc[idx, self.feature_array_col], dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("features must be 1D")
        return arr

    def _get_obs(self) -> np.ndarray:
        feats = self._get_base_features(self._i)
        parts = [feats]

        if self.add_time_features:
            s, c = sincos_time_of_day(self.df.loc[self._i, "timestamp"])
            parts.append(np.array([s, c], dtype=np.float32))

        if self.add_position_features:
            parts.append(
                np.array(
                    [
                        float(self.position),
                        float(self.time_in_pos),
                        float(self.unrealized_pnl),
                        float(self.realized_pnl_today),
                    ],
                    dtype=np.float32,
                )
            )

        obs = np.concatenate(parts, axis=0).astype(np.float32)
        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(f"Obs dim mismatch: got {obs.shape[0]} expected {self.obs_dim}")
        return obs

    def reset(self, day_ptr: Optional[int] = None) -> np.ndarray:
        if day_ptr is None:
            self._day_ptr = (self._day_ptr + 1) % len(self.day_starts)
        else:
            self._day_ptr = int(day_ptr) % len(self.day_starts)

        self._i = int(self.day_starts[self._day_ptr])

        self.position = 0
        self.entry_price = np.nan
        self.time_in_pos = 0
        self.unrealized_pnl = 0.0
        self.realized_pnl_today = 0.0
        return self._get_obs()

    def _trade_cost_ret(self, price: float) -> float:
        # bps already represent fractional cost vs notional
        bps_cost = (self.slippage_bps + self.spread_bps) / 10000.0
        commission_ret = (self.commission_per_trade / price) if price else 0.0
        return commission_ret + bps_cost


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action = int(action)
        if action not in (0, 1, 2):
            raise ValueError("action must be 0,1,2")

        prev_pos = self.position
        desired_pos = 0 if action == 0 else (1 if action == 1 else -1)

        price = float(self.df.loc[self._i, "close"])
        ret_next = float(self.df.loc[self._i, "ret_next"])
        ret_next_missing = not np.isfinite(ret_next)
        if ret_next_missing:
            ret_next = 0.0
        is_last = bool(self.df.loc[self._i, "is_last_of_day"])

        reward_costs = 0.0
        reward_pnl = 0.0
        reward_pivot_bonus = 0.0
        flipped = (self.position == 1 and desired_pos == -1) or (self.position == -1 and desired_pos == 1)
        blocked_flip = flipped and not self.allow_direct_flip
        if blocked_flip:
            desired_pos = 0
            flipped = False

        did_trade = desired_pos != prev_pos
        if desired_pos != self.position:
            reward_costs += self._trade_cost_ret(price)

            if flipped:
                reward_costs += self.flip_penalty_ret
                reward_costs += self.trade_penalty_ret * 2.0
            else:
                reward_costs += self.trade_penalty_ret

            # realize approximate pnl on exit
            if self.position != 0 and np.isfinite(self.entry_price):
                move = (price / self.entry_price - 1.0) * float(self.position)
                self.realized_pnl_today += move
                if self.reward_on_exit or self.reward_exit_bonus:
                    reward_pnl += move
                if self.exit_pivot_bonus_ret:
                    if self.position > 0:
                        pivot_val = (
                            float(self.df.loc[self._i, self.pivot_short_col])
                            if self.pivot_short_col in self.df.columns
                            else 0.0
                        )
                    else:
                        pivot_val = (
                            float(self.df.loc[self._i, self.pivot_long_col])
                            if self.pivot_long_col in self.df.columns
                            else 0.0
                        )
                    if np.isfinite(pivot_val):
                        reward_pivot_bonus = self.exit_pivot_bonus_ret * pivot_val
                        reward_pnl += reward_pivot_bonus

            if desired_pos != 0:
                self.entry_price = price
                self.time_in_pos = 0
            else:
                self.entry_price = np.nan
                self.time_in_pos = 0

            self.position = desired_pos

        # pnl reward for holding through next bar (only in mark-to-market mode)
        if not self.reward_on_exit:
            reward_pnl = float(self.position) * ret_next

        if self.position != 0 and np.isfinite(self.entry_price):
            next_price = price * (1.0 + ret_next)
            self.unrealized_pnl = (next_price / self.entry_price - 1.0) * float(self.position)
            self.time_in_pos += 1
        else:
            self.unrealized_pnl = 0.0
            self.time_in_pos = 0

        hold_penalty = self.hold_penalty_ret if self.position != 0 else 0.0
        reward = reward_pnl - reward_costs - hold_penalty

        info: Dict[str, Any] = {
            "price": price,
            "ret_next": ret_next,
            "ret_next_missing": ret_next_missing,
            "pos": self.position,
            "prev_pos": prev_pos,
            "did_trade": did_trade,
            "did_flip": flipped,
            "reward_pnl": reward_pnl,
            "reward_costs": reward_costs,
            "reward_pivot_bonus": reward_pivot_bonus,
            "realized_pnl_today": self.realized_pnl_today,
            "unrealized_pnl": self.unrealized_pnl,
            "flip_blocked": blocked_flip,
        }

        if is_last or (self._i >= len(self.df) - 2):
            if self.force_flat_at_close and self.position != 0:
                exit_cost = self._trade_cost_ret(price)
                reward_costs += exit_cost
                if self.trade_penalty_ret:
                    reward_costs += self.trade_penalty_ret
                info["forced_flat_cost"] = exit_cost

                if np.isfinite(self.entry_price):
                    move = (price / self.entry_price - 1.0) * float(self.position)
                    self.realized_pnl_today += move
                    if self.reward_on_exit or self.reward_exit_bonus:
                        reward_pnl += move
                    if self.exit_pivot_bonus_ret:
                        if self.position > 0:
                            pivot_val = (
                                float(self.df.loc[self._i, self.pivot_short_col])
                                if self.pivot_short_col in self.df.columns
                                else 0.0
                            )
                        else:
                            pivot_val = (
                                float(self.df.loc[self._i, self.pivot_long_col])
                                if self.pivot_long_col in self.df.columns
                                else 0.0
                            )
                        if np.isfinite(pivot_val):
                            reward_pivot_bonus = self.exit_pivot_bonus_ret * pivot_val
                            reward_pnl += reward_pivot_bonus

                self.position = 0
                self.entry_price = np.nan
                self.unrealized_pnl = 0.0
                self.time_in_pos = 0
                reward = reward_pnl - reward_costs - hold_penalty
                info["reward_pnl"] = reward_pnl
                info["reward_costs"] = reward_costs
                info["reward_pivot_bonus"] = reward_pivot_bonus
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
        self, actions: np.ndarray | list[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        if len(actions) != self.n_envs:
            raise ValueError("Actions length must match number of envs.")

        obs, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            o, r, d, info = env.step(int(action))
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
