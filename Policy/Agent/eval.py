from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from Agent.env import TradingEnv
from Agent.model import ActorCritic


def _resolve_device(device: str) -> torch.device:
    dev = (device or "auto").lower()
    if dev in ("auto", "gpu", "cuda"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _policy_action(model: ActorCritic, x: torch.Tensor, deterministic: bool) -> float:
    policy_out, _ = model(x)
    if getattr(model, "continuous_action", False):
        if deterministic:
            action_t = torch.tanh(policy_out)
        else:
            action_t, _logp, _entropy = model._sample_continuous(  # noqa: SLF001
                policy_out,
                deterministic=False,
            )
        return float(action_t.squeeze().item())
    if deterministic:
        act = int(torch.argmax(policy_out, dim=-1).item())
        return 0.0 if act == 0 else (1.0 if act == 1 else -1.0)
    dist = torch.distributions.Categorical(logits=policy_out)
    act = int(dist.sample().item())
    return 0.0 if act == 0 else (1.0 if act == 1 else -1.0)


@torch.no_grad()
def evaluate_policy(
    env: TradingEnv,
    model: ActorCritic,
    n_days: int = 20,
    device: str = "auto",
    deterministic: bool = True,
) -> pd.DataFrame:
    eps = 1e-3
    dev = _resolve_device(device)
    prev_training = model.training
    prev_device = next(model.parameters()).device
    moved = prev_device != dev
    if moved:
        model = model.to(dev)
    model.eval()
    rows = []
    total_steps = 0
    steps_in_pos = 0
    total_trades = 0
    hold_lengths = []
    current_hold = 0

    try:
        for d in range(min(n_days, len(env.day_starts))):
            obs = env.reset(day_ptr=d)
            done = False
            day_pnl = 0.0
            day_costs = 0.0
            trades = 0

            while not done:
                x = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
                action = _policy_action(model, x, deterministic)

                obs, r, done, info = env.step(action)
                day_pnl += float(info.get("reward_pnl", 0.0))
                day_costs += float(info.get("reward_costs", 0.0)) + float(info.get("forced_flat_cost", 0.0))
                if info.get("did_trade", False):
                    trades += 1
                    total_trades += 1

                total_steps += 1
                pos = float(info.get("pos", 0.0))
                prev_pos = float(info.get("prev_pos", 0.0))
                if abs(pos) > eps:
                    steps_in_pos += 1

                if abs(pos) > eps:
                    if abs(prev_pos) <= eps or (prev_pos * pos < 0.0):
                        if current_hold > 0:
                            hold_lengths.append(current_hold)
                        current_hold = 1
                    else:
                        current_hold += 1
                elif current_hold > 0:
                    hold_lengths.append(current_hold)
                    current_hold = 0

            rows.append({"day_ptr": d, "pnl_component": day_pnl, "costs_component": day_costs, "trades": trades})
    finally:
        if moved:
            model = model.to(prev_device)
        if prev_training:
            model.train()

    if current_hold > 0:
        hold_lengths.append(current_hold)

    time_in_pos = (steps_in_pos / total_steps) if total_steps else 0.0
    avg_hold = float(np.mean(hold_lengths)) if hold_lengths else 0.0
    print(
        "Eval summary:",
        f"total_trades={total_trades}",
        f"time_in_pos={time_in_pos:.2%}",
        f"avg_hold_len={avg_hold:.2f}",
    )

    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_policy_with_trace(
    env: TradingEnv,
    model: ActorCritic,
    device: str = "auto",
    deterministic: bool = True,
) -> pd.DataFrame:
    dev = _resolve_device(device)
    prev_training = model.training
    prev_device = next(model.parameters()).device
    moved = prev_device != dev
    if moved:
        model = model.to(dev)
    model.eval()

    has_ohlc = all(c in env.df.columns for c in ("open", "high", "low", "close"))
    rows = []
    try:
        for d in range(len(env.day_starts)):
            obs = env.reset(day_ptr=d)
            done = False
            while not done:
                idx = env._i
                ts = env.df.loc[idx, "timestamp"] if "timestamp" in env.df.columns else None
                close = float(env.df.loc[idx, "close"])
                if has_ohlc:
                    open_px = float(env.df.loc[idx, "open"])
                    high_px = float(env.df.loc[idx, "high"])
                    low_px = float(env.df.loc[idx, "low"])
                p_pivot_long = (
                    float(env.df.loc[idx, "p_pivot_long"])
                    if "p_pivot_long" in env.df.columns
                    else None
                )
                p_pivot_short = (
                    float(env.df.loc[idx, "p_pivot_short"])
                    if "p_pivot_short" in env.df.columns
                    else None
                )
                p_tb_long = (
                    float(env.df.loc[idx, "p_tb_long"])
                    if "p_tb_long" in env.df.columns
                    else None
                )
                p_tb_short = (
                    float(env.df.loc[idx, "p_tb_short"])
                    if "p_tb_short" in env.df.columns
                    else None
                )
                x = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
                action = _policy_action(model, x, deterministic)

                obs, reward, done, info = env.step(action)
                row = {
                    "day_ptr": d,
                    "timestamp": ts,
                    "close": close,
                    "action": action,
                    "position": float(info.get("pos", 0.0)),
                    "prev_pos": float(info.get("prev_pos", 0.0)),
                    "did_trade": bool(info.get("did_trade", False)),
                    "did_flip": bool(info.get("did_flip", False)),
                    "reward": float(reward),
                    "reward_pnl": float(info.get("reward_pnl", 0.0)),
                    "reward_costs": float(info.get("reward_costs", 0.0)),
                    "forced_flat_cost": float(info.get("forced_flat_cost", 0.0)),
                }
                if has_ohlc:
                    row["open"] = open_px
                    row["high"] = high_px
                    row["low"] = low_px
                if p_pivot_long is not None:
                    row["p_pivot_long"] = p_pivot_long
                if p_pivot_short is not None:
                    row["p_pivot_short"] = p_pivot_short
                if p_tb_long is not None:
                    row["p_tb_long"] = p_tb_long
                if p_tb_short is not None:
                    row["p_tb_short"] = p_tb_short
                rows.append(row)
    finally:
        if moved:
            model = model.to(prev_device)
        if prev_training:
            model.train()

    return pd.DataFrame(rows)
