from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from Agent.model import ActorCritic
from Execution_Agent.env import DirectionGatedExecutionEnv


def _resolve_device(device: str) -> torch.device:
    d = (device or "auto").lower()
    if d in {"auto", "cuda", "gpu"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _select_action(model: ActorCritic, x: torch.Tensor, deterministic: bool) -> int:
    policy_out, _ = model(x)
    if deterministic:
        return int(torch.argmax(policy_out, dim=-1).item())
    dist = torch.distributions.Categorical(logits=policy_out)
    return int(dist.sample().item())


@torch.no_grad()
def evaluate_policy_with_trace(
    env: DirectionGatedExecutionEnv,
    model: ActorCritic,
    *,
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
    rows: list[dict[str, Any]] = []
    try:
        for ep in range(len(env.day_starts)):
            obs = env.reset(day_ptr=ep)
            done = False
            while not done:
                x = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
                a = _select_action(model, x, deterministic)
                obs, reward, done, info = env.step(a)
                row = {
                    "episode": ep,
                    "timestamp": info.get("timestamp"),
                    "action_idx": int(info.get("action_idx", a)),
                    "reward": float(reward),
                    "reward_improvement": float(info.get("reward_improvement", 0.0)),
                    "agent_net": float(info.get("agent_net", 0.0)),
                    "baseline_net": float(info.get("baseline_net", 0.0)),
                    "mae_penalty": float(info.get("mae_penalty", 0.0)),
                    "churn_penalty": float(info.get("churn_penalty", 0.0)),
                    "low_conf_penalty": float(info.get("low_conf_penalty", 0.0)),
                    "agent_pos_units": int(info.get("agent_pos_units", 0)),
                    "baseline_pos_units": int(info.get("baseline_pos_units", 0)),
                    "htf_dir": int(info.get("htf_dir", 0)),
                    "htf_conf": float(info.get("htf_conf", 0.0)),
                    "ret_next": float(info.get("ret_next", 0.0)),
                    "close": float(info.get("close", np.nan)),
                    "did_trade": bool(info.get("did_trade", False)),
                }
                rows.append(row)
    finally:
        if moved:
            model = model.to(prev_device)
        if prev_training:
            model.train()
    return pd.DataFrame(rows)


def summarize_trace(trace: pd.DataFrame) -> dict[str, float]:
    if trace.empty:
        return {
            "steps": 0.0,
            "avg_reward": 0.0,
            "avg_improvement": 0.0,
            "agent_return": 0.0,
            "baseline_return": 0.0,
            "beats_baseline": 0.0,
            "trade_rate": 0.0,
        }
    agent_equity = float(np.prod(1.0 + pd.to_numeric(trace["agent_net"], errors="coerce").fillna(0.0).to_numpy()))
    base_equity = float(np.prod(1.0 + pd.to_numeric(trace["baseline_net"], errors="coerce").fillna(0.0).to_numpy()))
    out = {
        "steps": float(len(trace)),
        "avg_reward": float(pd.to_numeric(trace["reward"], errors="coerce").fillna(0.0).mean()),
        "avg_improvement": float(pd.to_numeric(trace["reward_improvement"], errors="coerce").fillna(0.0).mean()),
        "agent_return": agent_equity - 1.0,
        "baseline_return": base_equity - 1.0,
        "beats_baseline": float((agent_equity - base_equity) > 0.0),
        "trade_rate": float(pd.to_numeric(trace["did_trade"], errors="coerce").fillna(0.0).mean()),
    }
    return out

