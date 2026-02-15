from __future__ import annotations
from pathlib import Path
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from Agent.env import TradingEnv
from Agent.model import ActorCritic
from Agent.buffer import Rollout, compute_gae


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _resolve_device(device: str, verbose: bool) -> torch.device:
    dev = (device or "auto").lower()
    if dev in ("auto", "gpu", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if verbose:
            print("CUDA not available; falling back to CPU.")
        return torch.device("cpu")
    if dev == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        if verbose:
            print("MPS not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def train_ppo(
    env: TradingEnv,
    total_timesteps: int = 2_000_000,
    rollout_len: int = 2048,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    pi_lr: float = 3e-4,
    vf_lr: float = 1e-3,
    train_epochs: int = 10,
    minibatch_size: int = 256,
    entropy_coef: float = 0.004,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    action_type: str = "hybrid_dir_mag",
    device: str = "cuda",
    seed: int = 7,
    verbose: bool = True,
    return_history: bool = False,
    checkpoint_every_steps: int = 0,
    checkpoint_dir: str | None = None,
    checkpoint_prefix: str = "ppo_ckpt",
    checkpoint_payload: dict | None = None,
) -> ActorCritic | tuple[ActorCritic, list[dict[str, float]]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    dev = _resolve_device(device, verbose)

    model = ActorCritic(
        obs_dim=env.obs_dim,
        action_type=action_type,
        action_dim=1,
        hidden=128,
    ).to(dev)
    model.train()
    param_groups = [
        {"params": model.shared.parameters(), "lr": pi_lr},
        {"params": model.policy_mlp.parameters(), "lr": pi_lr},
        {"params": model.policy_head.parameters(), "lr": pi_lr},
        {"params": model.value_mlp.parameters(), "lr": vf_lr},
        {"params": model.value_head.parameters(), "lr": vf_lr},
    ]
    if model.policy_log_std is not None:
        param_groups.append({"params": [model.policy_log_std], "lr": pi_lr})
    optimizer = optim.Adam(param_groups)

    n_envs = int(getattr(env, "n_envs", 1))
    is_vectorized = n_envs > 1
    obs = env.reset(day_ptr=0) if not is_vectorized else env.reset()
    steps_done = 0
    next_checkpoint_step = (
        int(checkpoint_every_steps)
        if int(checkpoint_every_steps) > 0
        else None
    )
    train_start = time.perf_counter()
    history: list[dict[str, float]] = []

    while steps_done < total_timesteps:
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, done_buf, val_buf, next_val_buf = [], [], [], []
        loss_total_hist: list[float] = []
        loss_pi_hist: list[float] = []
        loss_v_hist: list[float] = []
        entropy_hist: list[float] = []
        approx_kl_hist: list[float] = []
        clipfrac_hist: list[float] = []

        for _ in range(rollout_len):
            if is_vectorized:
                actions_t, logp_t, val_t = model.act_batch(obs, dev)
                actions_raw = actions_t.detach().cpu().numpy().astype("float32")
                if model.hybrid_action:
                    if actions_raw.ndim != 2 or actions_raw.shape[-1] < 3:
                        raise RuntimeError(
                            f"Hybrid actions must be [n_envs,3], got {actions_raw.shape}"
                        )
                    env_actions = actions_raw[:, 0]
                    store_actions = actions_raw
                else:
                    env_actions = actions_raw
                    if actions_raw.ndim == 2 and actions_raw.shape[-1] == 1:
                        env_actions = actions_raw[:, 0]
                    store_actions = env_actions
                logp = logp_t.detach().cpu().numpy().astype("float32")
                val = val_t.detach().cpu().numpy().astype("float32")
                next_obs, reward, done, _info = env.step(env_actions)
                with torch.no_grad():
                    x_next = torch.as_tensor(next_obs, dtype=torch.float32, device=dev)
                    _, v_next = model(x_next)
                    v_next_f = v_next.detach().cpu().numpy().astype(np.float32)

                obs_buf.append(obs)
                act_buf.append(store_actions)
                logp_buf.append(logp)
                rew_buf.append(reward)
                done_buf.append(done)
                val_buf.append(val)
                next_val_buf.append(v_next_f)

                obs = next_obs
                steps_done += n_envs
            else:
                action, logp, val = model.act(obs, dev)
                store_action = action
                env_action = action
                if model.hybrid_action:
                    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
                    env_action = float(action_arr[0]) if action_arr.size else 0.0
                    store_action = action_arr
                elif isinstance(action, (list, tuple, np.ndarray)):
                    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
                    env_action = float(action_arr[0]) if action_arr.size else 0.0
                    store_action = env_action
                next_obs, reward, done, _info = env.step(env_action)

                with torch.no_grad():
                    x_next = torch.as_tensor(next_obs, dtype=torch.float32, device=dev).unsqueeze(0)
                    _, v_next = model(x_next)
                    v_next_f = float(v_next.item())

                obs_buf.append(obs)
                act_buf.append(store_action)
                logp_buf.append(logp)
                rew_buf.append(reward)
                done_buf.append(done)
                val_buf.append(val)
                next_val_buf.append(v_next_f)

                obs = next_obs
                steps_done += 1

                if done:
                    obs = env.reset()

            if steps_done >= total_timesteps:
                break

        if is_vectorized:
            obs_arr = np.asarray(obs_buf, dtype=np.float32)
            act_arr = np.asarray(act_buf, dtype=np.float32)
            logp_arr = np.asarray(logp_buf, dtype=np.float32)
            rew_arr = np.asarray(rew_buf, dtype=np.float32)
            done_arr = np.asarray(done_buf, dtype=np.float32)
            val_arr = np.asarray(val_buf, dtype=np.float32)
            next_val_arr = np.asarray(next_val_buf, dtype=np.float32)

            steps_collected = obs_arr.shape[0]
            adv_arr = np.zeros_like(rew_arr, dtype=np.float32)
            ret_arr = np.zeros_like(rew_arr, dtype=np.float32)
            for i in range(n_envs):
                adv_i, ret_i = compute_gae(
                    rewards=rew_arr[:, i],
                    dones=done_arr[:, i],
                    values=val_arr[:, i],
                    next_values=next_val_arr[:, i],
                    gamma=gamma,
                    lam=gae_lambda,
                )
                adv_arr[:, i] = adv_i
                ret_arr[:, i] = ret_i

            adv_flat = adv_arr.reshape(steps_collected * n_envs)
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
            ret_flat = ret_arr.reshape(steps_collected * n_envs)

            obs_arr = obs_arr.reshape(steps_collected * n_envs, obs_arr.shape[-1])
            if model.hybrid_action:
                act_arr = act_arr.reshape(steps_collected * n_envs, act_arr.shape[-1])
            else:
                act_arr = act_arr.reshape(steps_collected * n_envs)
            logp_arr = logp_arr.reshape(steps_collected * n_envs)
            rew_arr = rew_arr.reshape(steps_collected * n_envs)
            done_arr = done_arr.reshape(steps_collected * n_envs)
            val_arr = val_arr.reshape(steps_collected * n_envs)
            next_val_arr = next_val_arr.reshape(steps_collected * n_envs)

            adv = adv_flat.astype(np.float32)
            ret = ret_flat.astype(np.float32)
        else:
            obs_arr = np.asarray(obs_buf, dtype=np.float32)
            act_arr = np.asarray(act_buf, dtype=np.float32)
            logp_arr = np.asarray(logp_buf, dtype=np.float32)
            rew_arr = np.asarray(rew_buf, dtype=np.float32)
            done_arr = np.asarray(done_buf, dtype=np.float32)
            val_arr = np.asarray(val_buf, dtype=np.float32)
            next_val_arr = np.asarray(next_val_buf, dtype=np.float32)

            adv, ret = compute_gae(
                rewards=rew_arr,
                dones=done_arr,
                values=val_arr,
                next_values=next_val_arr,
                gamma=gamma,
                lam=gae_lambda,
            )
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        rollout = Rollout(
            obs=obs_arr,
            actions=act_arr,
            logp_old=logp_arr,
            rewards=rew_arr,
            dones=done_arr,
            values=val_arr,
            next_values=next_val_arr,
            advantages=adv.astype(np.float32),
            returns=ret.astype(np.float32),
        )

        n = rollout.obs.shape[0]
        idxs = np.arange(n)

        obs_t = torch.as_tensor(rollout.obs, dtype=torch.float32, device=dev)
        if model.continuous_action or model.hybrid_action:
            act_t = torch.as_tensor(rollout.actions, dtype=torch.float32, device=dev)
        else:
            act_t = torch.as_tensor(rollout.actions, dtype=torch.int64, device=dev)
        logp_old_t = torch.as_tensor(rollout.logp_old, dtype=torch.float32, device=dev)
        adv_t = torch.as_tensor(rollout.advantages, dtype=torch.float32, device=dev)
        ret_t = torch.as_tensor(rollout.returns, dtype=torch.float32, device=dev)

        for _epoch in range(train_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, minibatch_size):
                mb = idxs[start : start + minibatch_size]
                if mb.size == 0:
                    continue

                logp, entropy, values = model.evaluate_actions(obs_t[mb], act_t[mb])
                entropy = entropy.mean()

                ratio = torch.exp(logp - logp_old_t[mb])
                unclipped = ratio * adv_t[mb]
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_t[mb]
                pi_loss = -(torch.min(unclipped, clipped)).mean()

                v_loss = ((values - ret_t[mb]) ** 2).mean()

                loss = pi_loss + value_coef * v_loss - entropy_coef * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = (logp_old_t[mb] - logp).mean()
                    clipfrac = (torch.abs(ratio - 1.0) > clip_ratio).float().mean()
                loss_total_hist.append(float(loss.detach().item()))
                loss_pi_hist.append(float(pi_loss.detach().item()))
                loss_v_hist.append(float(v_loss.detach().item()))
                entropy_hist.append(float(entropy.detach().item()))
                approx_kl_hist.append(float(approx_kl.detach().item()))
                clipfrac_hist.append(float(clipfrac.detach().item()))

        if verbose:
            now = time.perf_counter()
            elapsed = max(now - train_start, 1e-9)
            speed = steps_done / elapsed
            remaining_steps = max(int(total_timesteps) - int(steps_done), 0)
            eta = (remaining_steps / speed) if speed > 0 else float("inf")
            progress_pct = 100.0 * min(steps_done, total_timesteps) / max(total_timesteps, 1)
            loss_total = float(np.mean(loss_total_hist)) if loss_total_hist else float("nan")
            loss_pi = float(np.mean(loss_pi_hist)) if loss_pi_hist else float("nan")
            loss_v = float(np.mean(loss_v_hist)) if loss_v_hist else float("nan")
            entropy_avg = float(np.mean(entropy_hist)) if entropy_hist else float("nan")
            approx_kl_avg = float(np.mean(approx_kl_hist)) if approx_kl_hist else float("nan")
            clipfrac_avg = float(np.mean(clipfrac_hist)) if clipfrac_hist else float("nan")
            rollout_avg_reward = float(np.mean(rollout.rewards))
            rollout_avg_abs_reward = float(np.mean(np.abs(rollout.rewards)))
            history.append(
                {
                    "steps": float(steps_done),
                    "progress_pct": float(progress_pct),
                    "rollout_avg_reward": rollout_avg_reward,
                    "rollout_avg_abs_reward": rollout_avg_abs_reward,
                    "loss_total": loss_total,
                    "loss_pi": loss_pi,
                    "loss_v": loss_v,
                    "entropy": entropy_avg,
                    "approx_kl": approx_kl_avg,
                    "clipfrac": clipfrac_avg,
                    "sps": float(speed),
                }
            )
            print(
                f"steps={steps_done:,} "
                f"({progress_pct:.2f}%) "
                f"rollout_avg_reward={rollout_avg_reward:.6f} "
                f"avg|r|={rollout_avg_abs_reward:.6f} "
                f"loss={loss_total:.6f} "
                f"pi={loss_pi:.6f} "
                f"v={loss_v:.6f} "
                f"ent={entropy_avg:.6f} "
                f"kl={approx_kl_avg:.6f} "
                f"clipfrac={clipfrac_avg:.3f} "
                f"sps={speed:,.1f} "
                f"elapsed={_format_duration(elapsed)} "
                f"eta={_format_duration(eta)}"
            )

        if next_checkpoint_step is not None and checkpoint_dir:
            while steps_done >= next_checkpoint_step:
                ckpt_dir = Path(checkpoint_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"{checkpoint_prefix}_{next_checkpoint_step:09d}.pt"
                payload = dict(checkpoint_payload or {})
                payload.setdefault("obs_dim", env.obs_dim)
                payload.setdefault("n_actions", 3)
                payload.setdefault("action_type", action_type)
                payload.setdefault("action_dim", int(getattr(model, "action_dim", 1)))
                payload.setdefault("action_low", -1.0)
                payload.setdefault("action_high", 1.0)
                payload["checkpoint_steps"] = int(next_checkpoint_step)
                if history:
                    payload["checkpoint_last_metrics"] = history[-1]
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        **payload,
                    },
                    ckpt_path,
                )
                if verbose:
                    print(f"Saved checkpoint: {ckpt_path}")
                next_checkpoint_step += int(checkpoint_every_steps)

    if return_history:
        return model, history
    return model
