from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from Agent.buffer import compute_gae
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


def _policy_decision(
    model: ActorCritic,
    x: torch.Tensor,
    deterministic: bool,
) -> tuple[float, int | None, float | None]:
    policy_out, _ = model(x)
    if getattr(model, "hybrid_action", False):
        if deterministic:
            dir_logits = policy_out[:, :3]
            mag_mean = policy_out[:, 3:4]
            dir_idx_t = torch.argmax(dir_logits, dim=-1)
            mag_t = torch.sigmoid(mag_mean)
            dir_sign = model._dir_idx_to_sign(dir_idx_t)  # noqa: SLF001
            exposure_t = dir_sign * mag_t.squeeze(-1)
            action_t = torch.stack(
                [exposure_t, dir_idx_t.to(dtype=torch.float32), mag_t.squeeze(-1)],
                dim=-1,
            )
        else:
            action_t, _logp, _entropy = model._sample_hybrid(  # noqa: SLF001
                policy_out,
                deterministic=False,
            )
        exposure = float(action_t[:, 0].squeeze().item())
        dir_idx = int(action_t[:, 1].round().clamp(min=0, max=2).squeeze().item())
        mag = float(action_t[:, 2].squeeze().item())
        return exposure, dir_idx, mag
    if getattr(model, "continuous_action", False):
        if deterministic:
            action_t = torch.tanh(policy_out)
        else:
            action_t, _logp, _entropy = model._sample_continuous(  # noqa: SLF001
                policy_out,
                deterministic=False,
            )
        return float(action_t.squeeze().item()), None, None
    if deterministic:
        act = int(torch.argmax(policy_out, dim=-1).item())
        return 0.0 if act == 0 else (1.0 if act == 1 else -1.0), act, None
    dist = torch.distributions.Categorical(logits=policy_out)
    act = int(dist.sample().item())
    return 0.0 if act == 0 else (1.0 if act == 1 else -1.0), act, None


def _policy_action(model: ActorCritic, x: torch.Tensor, deterministic: bool) -> float:
    action, _dir_idx, _mag = _policy_decision(model, x, deterministic)
    return action


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
                action, action_dir_idx, action_mag = _policy_decision(model, x, deterministic)

                obs, reward, done, info = env.step(action)
                row = {
                    "day_ptr": d,
                    "timestamp": ts,
                    "close": close,
                    "action": action,
                    "action_dir_idx": action_dir_idx if action_dir_idx is not None else np.nan,
                    "action_mag": action_mag if action_mag is not None else np.nan,
                    "position": float(info.get("pos", 0.0)),
                    "prev_pos": float(info.get("prev_pos", 0.0)),
                    "did_trade": bool(info.get("did_trade", False)),
                    "did_flip": bool(info.get("did_flip", False)),
                    "reward": float(reward),
                    "reward_pnl": float(info.get("reward_pnl", 0.0)),
                    "reward_costs": float(info.get("reward_costs", 0.0)),
                    "reward_pivot_bonus": float(info.get("reward_pivot_bonus", 0.0)),
                    "reward_convex": (
                        float(info.get("reward_convex"))
                        if info.get("reward_convex") is not None
                        else np.nan
                    ),
                    "reward_convex_linear": (
                        float(info.get("reward_convex_linear"))
                        if info.get("reward_convex_linear") is not None
                        else np.nan
                    ),
                    "reward_convex_bonus": (
                        float(info.get("reward_convex_bonus"))
                        if info.get("reward_convex_bonus") is not None
                        else np.nan
                    ),
                    "reward_pivot_anchor": (
                        float(info.get("reward_pivot_anchor"))
                        if info.get("reward_pivot_anchor") is not None
                        else np.nan
                    ),
                    "convex_term": (
                        float(info.get("convex_term"))
                        if info.get("convex_term") is not None
                        else np.nan
                    ),
                    "convex_atr_scale": (
                        float(info.get("convex_atr_scale"))
                        if info.get("convex_atr_scale") is not None
                        else np.nan
                    ),
                    "mfe_atr": (
                        float(info.get("mfe_atr"))
                        if info.get("mfe_atr") is not None
                        else np.nan
                    ),
                    "mfe_bonus": (
                        float(info.get("mfe_bonus"))
                        if info.get("mfe_bonus") is not None
                        else np.nan
                    ),
                    "reward_switch_penalty": float(info.get("reward_switch_penalty", 0.0)),
                    "reward_size_penalty": float(info.get("reward_size_penalty", 0.0)),
                    "did_dir_switch": bool(info.get("did_dir_switch", False)),
                    "size_delta": float(info.get("size_delta", 0.0)),
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


@torch.no_grad()
def evaluate_loss_metrics(
    env: TradingEnv,
    model: ActorCritic,
    *,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    device: str = "auto",
    deterministic: bool = True,
) -> dict[str, float]:
    dev = _resolve_device(device)
    prev_training = model.training
    prev_device = next(model.parameters()).device
    moved = prev_device != dev
    if moved:
        model = model.to(dev)
    model.eval()

    rewards: list[float] = []
    dones: list[float] = []
    values: list[float] = []
    next_values: list[float] = []
    logps: list[float] = []
    entropies: list[float] = []

    try:
        for d in range(len(env.day_starts)):
            obs = env.reset(day_ptr=d)
            done = False
            while not done:
                x = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
                policy_out, value_t = model(x)
                if getattr(model, "hybrid_action", False):
                    if deterministic:
                        dir_logits = policy_out[:, :3]
                        mag_mean = policy_out[:, 3:4]
                        dir_idx_t = torch.argmax(dir_logits, dim=-1)
                        mag_t = torch.sigmoid(mag_mean)
                        dir_sign = model._dir_idx_to_sign(dir_idx_t)  # noqa: SLF001
                        exposure_t = dir_sign * mag_t.squeeze(-1)
                        action_t = torch.stack(
                            [exposure_t, dir_idx_t.to(dtype=torch.float32), mag_t.squeeze(-1)],
                            dim=-1,
                        )
                    else:
                        action_t, _logp_tmp, _ent_tmp = model._sample_hybrid(  # noqa: SLF001
                            policy_out,
                            deterministic=False,
                        )
                    logp_t, entropy_t = model._hybrid_logp_entropy(  # noqa: SLF001
                        policy_out,
                        action_t,
                    )
                    action = float(action_t[:, 0].squeeze().item())
                elif getattr(model, "continuous_action", False):
                    if deterministic:
                        action_t = torch.tanh(policy_out)
                    else:
                        action_t, _logp_tmp, _ent_tmp = model._sample_continuous(  # noqa: SLF001
                            policy_out,
                            deterministic=False,
                        )
                    logp_t, entropy_t = model._continuous_logp_entropy(  # noqa: SLF001
                        policy_out,
                        action_t,
                    )
                    action = float(action_t.squeeze().item())
                else:
                    dist = torch.distributions.Categorical(logits=policy_out)
                    if deterministic:
                        action_idx = torch.argmax(policy_out, dim=-1)
                    else:
                        action_idx = dist.sample()
                    logp_t = dist.log_prob(action_idx)
                    entropy_t = dist.entropy()
                    a = int(action_idx.item())
                    action = 0.0 if a == 0 else (1.0 if a == 1 else -1.0)

                next_obs, reward, done, _info = env.step(action)
                x_next = torch.as_tensor(next_obs, dtype=torch.float32, device=dev).unsqueeze(0)
                _next_policy, next_value_t = model(x_next)

                rewards.append(float(reward))
                dones.append(float(done))
                values.append(float(value_t.squeeze().item()))
                next_values.append(float(next_value_t.squeeze().item()))
                logps.append(float(logp_t.squeeze().item()))
                entropies.append(float(entropy_t.squeeze().item()))
                obs = next_obs
    finally:
        if moved:
            model = model.to(prev_device)
        if prev_training:
            model.train()

    if not rewards:
        return {
            "eval_loss_actor": float("nan"),
            "eval_loss_value": float("nan"),
            "eval_entropy": float("nan"),
            "eval_avg_reward": float("nan"),
            "eval_avg_abs_reward": float("nan"),
        }

    rew_arr = np.asarray(rewards, dtype=np.float32)
    done_arr = np.asarray(dones, dtype=np.float32)
    val_arr = np.asarray(values, dtype=np.float32)
    next_val_arr = np.asarray(next_values, dtype=np.float32)
    adv, ret = compute_gae(
        rewards=rew_arr,
        dones=done_arr,
        values=val_arr,
        next_values=next_val_arr,
        gamma=float(gamma),
        lam=float(gae_lambda),
    )
    adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
    logp_arr = np.asarray(logps, dtype=np.float32)
    ent_arr = np.asarray(entropies, dtype=np.float32)

    actor_loss = -float(np.mean(logp_arr * adv_norm))
    value_loss = float(np.mean((val_arr - ret) ** 2))
    entropy = float(np.mean(ent_arr))
    avg_reward = float(np.mean(rew_arr))
    avg_abs_reward = float(np.mean(np.abs(rew_arr)))
    return {
        "eval_loss_actor": actor_loss,
        "eval_loss_value": value_loss,
        "eval_entropy": entropy,
        "eval_avg_reward": avg_reward,
        "eval_avg_abs_reward": avg_abs_reward,
    }
