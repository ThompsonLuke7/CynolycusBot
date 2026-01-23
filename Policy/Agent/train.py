from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Agent.env import TradingEnv
from Agent.model import ActorCritic
from Agent.buffer import Rollout, compute_gae


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
    total_timesteps: int = 200_000,
    rollout_len: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    pi_lr: float = 3e-4,
    vf_lr: float = 1e-3,
    train_epochs: int = 10,
    minibatch_size: int = 256,
    entropy_coef: float = 0.01,
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
            {"params": model.policy_head.parameters(), "lr": pi_lr},
            {"params": model.value_head.parameters(), "lr": vf_lr},
        ]
    )

    obs = env.reset(day_ptr=0)
    steps_done = 0

    while steps_done < total_timesteps:
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, done_buf, val_buf, next_val_buf = [], [], [], []

        for _ in range(rollout_len):
            action, logp, val = model.act(obs, dev)
            next_obs, reward, done, info = env.step(action)

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

                logits, values = model(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act_t[mb])
                entropy = dist.entropy().mean()

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
            print(
                f"steps={steps_done:,} "
                f"rollout_avg_reward={float(np.mean(rollout.rewards)):.6f} "
                f"avg|r|={float(np.mean(np.abs(rollout.rewards))):.6f}"
            )

    return model
