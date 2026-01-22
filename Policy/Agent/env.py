from __future__ import annotations
from dataclasses import dataclass
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
        flip_penalty: float = 0.0,
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
        self.flip_penalty = float(flip_penalty)
        self.force_flat_at_close = bool(force_flat_at_close)
        self.allow_direct_flip = bool(allow_direct_flip)

        self.rng = np.random.default_rng(seed)

        # next-bar return
        self.df["ret_next"] = self.df["close"].shift(-1) / self.df["close"] - 1.0
        self.df["is_last_of_day"] = self.df["day_id"].shift(-1) != self.df["day_id"]

        self.day_starts = self.df.index[self.df["day_id"].shift(1) != self.df["day_id"]].to_list()
        if len(self.day_starts) == 0:
            raise ValueError("No day boundaries found. Ensure 'day_id' changes across days.")

        self._day_ptr = 0
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

    def _trade_cost(self, price: float) -> float:
        bps_cost = (self.slippage_bps + self.spread_bps) / 10000.0
        return self.commission_per_trade + abs(price) * bps_cost

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action = int(action)
        if action not in (0, 1, 2):
            raise ValueError("action must be 0,1,2")

        desired_pos = 0 if action == 0 else (1 if action == 1 else -1)

        price = float(self.df.loc[self._i, "close"])
        ret_next = float(self.df.loc[self._i, "ret_next"])
        is_last = bool(self.df.loc[self._i, "is_last_of_day"])

        reward_costs = 0.0
        flipped = (self.position == 1 and desired_pos == -1) or (self.position == -1 and desired_pos == 1)

        if desired_pos != self.position:
            if flipped and not self.allow_direct_flip:
                reward_costs += self._trade_cost(price)  # exit
                reward_costs += self._trade_cost(price)  # enter
            else:
                reward_costs += self._trade_cost(price)

            if flipped:
                reward_costs += self.flip_penalty

            # realize approximate pnl on exit
            if self.position != 0 and np.isfinite(self.entry_price):
                move = (price / self.entry_price - 1.0) * float(self.position)
                self.realized_pnl_today += move

            if desired_pos != 0:
                self.entry_price = price
                self.time_in_pos = 0
            else:
                self.entry_price = np.nan
                self.time_in_pos = 0

            self.position = desired_pos

        # pnl reward for holding through next bar
        reward_pnl = float(self.position) * ret_next

        if self.position != 0 and np.isfinite(self.entry_price):
            next_price = price * (1.0 + ret_next)
            self.unrealized_pnl = (next_price / self.entry_price - 1.0) * float(self.position)
            self.time_in_pos += 1
        else:
            self.unrealized_pnl = 0.0
            self.time_in_pos = 0

        reward = reward_pnl - reward_costs

        info: Dict[str, Any] = {
            "price": price,
            "ret_next": ret_next,
            "pos": self.position,
            "reward_pnl": reward_pnl,
            "reward_costs": reward_costs,
            "realized_pnl_today": self.realized_pnl_today,
            "unrealized_pnl": self.unrealized_pnl,
        }

        if is_last or (self._i >= len(self.df) - 2):
            if self.force_flat_at_close and self.position != 0:
                exit_cost = self._trade_cost(price)
                reward -= exit_cost
                info["forced_flat_cost"] = exit_cost

                if np.isfinite(self.entry_price):
                    move = (price / self.entry_price - 1.0) * float(self.position)
                    self.realized_pnl_today += move

                self.position = 0
                self.entry_price = np.nan
                self.unrealized_pnl = 0.0
                self.time_in_pos = 0

            done = True
            return self._get_obs(), float(reward), done, info

        self._i += 1
        done = False
        return self._get_obs(), float(reward), done, info
