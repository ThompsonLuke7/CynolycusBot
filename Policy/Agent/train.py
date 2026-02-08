from __future__ import annotations
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
    entropy_coef: float = 0.003,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    device: str = "cuda",
    seed: int = 7,
    verbose: bool = True,
) -> ActorCritic:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    dev = _resolve_device(device, verbose)

    model = ActorCritic(obs_dim=env.obs_dim, n_actions=3, hidden=128).to(dev)
    model.train()
    optimizer = optim.Adam(
        [
            {"params": model.shared.parameters(), "lr": pi_lr},
            {"params": model.policy_mlp.parameters(), "lr": pi_lr},
            {"params": model.policy_head.parameters(), "lr": pi_lr},
            {"params": model.value_mlp.parameters(), "lr": vf_lr},
            {"params": model.value_head.parameters(), "lr": vf_lr},
        ]
    )

    n_envs = int(getattr(env, "n_envs", 1))
    is_vectorized = n_envs > 1
    obs = env.reset(day_ptr=0) if not is_vectorized else env.reset()
    steps_done = 0
    train_start = time.perf_counter()

    while steps_done < total_timesteps:
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, done_buf, val_buf, next_val_buf = [], [], [], []

        for _ in range(rollout_len):
            if is_vectorized:
                actions_t, logp_t, val_t = model.act_batch(obs, dev)
                actions = actions_t.detach().cpu().numpy().astype("int64")
                logp = logp_t.detach().cpu().numpy().astype("float32")
                val = val_t.detach().cpu().numpy().astype("float32")
                next_obs, reward, done, _info = env.step(actions)
                with torch.no_grad():
                    x_next = torch.as_tensor(next_obs, dtype=torch.float32, device=dev)
                    _, v_next = model(x_next)
                    v_next_f = v_next.detach().cpu().numpy().astype(np.float32)

                obs_buf.append(obs)
                act_buf.append(actions)
                logp_buf.append(logp)
                rew_buf.append(reward)
                done_buf.append(done)
                val_buf.append(val)
                next_val_buf.append(v_next_f)

                obs = next_obs
                steps_done += n_envs
            else:
                action, logp, val = model.act(obs, dev)
                next_obs, reward, done, _info = env.step(action)

                with torch.no_grad():
                    x_next = torch.as_tensor(next_obs, dtype=torch.float32, device=dev).unsqueeze(0)
                    _, v_next = model(x_next)
                    v_next_f = float(v_next.item())

                obs_buf.append(obs)
                act_buf.append(action)
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
            act_arr = np.asarray(act_buf, dtype=np.int64)
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
            act_arr = np.asarray(act_buf, dtype=np.int64)
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

        if verbose:
            now = time.perf_counter()
            elapsed = max(now - train_start, 1e-9)
            speed = steps_done / elapsed
            remaining_steps = max(int(total_timesteps) - int(steps_done), 0)
            eta = (remaining_steps / speed) if speed > 0 else float("inf")
            progress_pct = 100.0 * min(steps_done, total_timesteps) / max(total_timesteps, 1)
            print(
                f"steps={steps_done:,} "
                f"({progress_pct:.2f}%) "
                f"rollout_avg_reward={float(np.mean(rollout.rewards)):.6f} "
                f"avg|r|={float(np.mean(np.abs(rollout.rewards))):.6f} "
                f"sps={speed:,.1f} "
                f"elapsed={_format_duration(elapsed)} "
                f"eta={_format_duration(eta)}"
            )

    return model
