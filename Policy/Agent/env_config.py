from __future__ import annotations

from dataclasses import dataclass, asdict, fields, replace
from typing import Any

from Agent.env import TradingEnv


@dataclass(frozen=True)
class TradingEnvConfig:
    add_time_features: bool = False
    add_position_features: bool = True
    commission_per_trade: float = 0.00005
    slippage_bps: float = 1.0
    spread_bps: float = 1.0
    flip_penalty_ret: float = 0.0006
    trade_penalty_ret: float = 0.0003
    hold_penalty_ret: float = 0.0
    reward_on_exit: bool = True
    reward_exit_bonus: bool = False
    exit_pivot_bonus_ret: float = 0.0
    force_flat_at_close: bool = False
    carry_positions_across_days: bool = True
    allow_direct_flip: bool = True
    use_convex_reward: bool = False
    convex_k1: float = 1.0
    convex_k2: float = 0.5
    convex_theta: float = 0.01
    convex_pivot_k: float = 0.0
    convex_directional_bonus_only: bool = True
    convex_wrong_side_scale: float = 0.0
    convex_mfe_thresholds: tuple[float, ...] = (1.0, 2.0, 3.0)
    convex_mfe_bonuses: tuple[float, ...] = (0.1, 0.2, 0.3)
    action_deadband: float = 1e-3
    dir_switch_penalty_ret: float = 0.0
    size_change_penalty_ret: float = 0.0
    seed: int = 7


def make_trading_env(
    *,
    df,
    feature_cols,
    config: TradingEnvConfig | None = None,
    **overrides: Any,
) -> TradingEnv:
    cfg = config or TradingEnvConfig()
    if overrides:
        valid = {f.name for f in fields(TradingEnvConfig)}
        filtered = {k: v for k, v in overrides.items() if k in valid}
        cfg = replace(cfg, **filtered)
    return TradingEnv(df=df, feature_cols=feature_cols, **asdict(cfg))
