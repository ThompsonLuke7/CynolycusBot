from __future__ import annotations
import pandas as pd
import torch
import numpy as np

from Agent.env import TradingEnv
from Agent.model import ActorCritic


@torch.no_grad()
def evaluate_policy(
    env: TradingEnv,
    model: ActorCritic,
    n_days: int = 20,
    device: str = "cpu",
) -> pd.DataFrame:
    dev = torch.device(device)
    rows = []

    for d in range(min(n_days, len(env.day_starts))):
        obs = env.reset(day_ptr=d)
        done = False
        day_pnl = 0.0
        day_costs = 0.0
        trades = 0

        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
            logits, _ = model(x)
            action = int(torch.argmax(logits, dim=-1).item())

            obs, r, done, info = env.step(action)
            day_pnl += float(info.get("reward_pnl", 0.0))
            day_costs += float(info.get("reward_costs", 0.0)) + float(info.get("forced_flat_cost", 0.0))
            if float(info.get("reward_costs", 0.0)) > 0:
                trades += 1

        rows.append({"day_ptr": d, "pnl_component": day_pnl, "costs_component": day_costs, "trades": trades})

    return pd.DataFrame(rows)
