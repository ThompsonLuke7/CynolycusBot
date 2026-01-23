from pathlib import Path
import sys
import torch

import numpy as np
import pandas as pd

_POLICY_ROOT = Path(__file__).resolve().parents[1]
if str(_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_ROOT))

from Agent.env import TradingEnv
from Agent.train import train_ppo
from Agent.eval import evaluate_policy

# TODO: Replace with your real 15m SPY df with:
# columns: timestamp, day_id, close, and your feature columns
# Your feature columns should include the inference outputs from your other models

def build_dummy_df():
    n_days = 5
    bars_per_day = 26
    timestamps, day_ids, closes = [], [], []
    rng = np.random.default_rng(0)

    price = 500.0
    for d in range(n_days):
        day = pd.Timestamp("2024-01-02") + pd.Timedelta(days=d)
        for b in range(bars_per_day):
            ts = day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=15 * b)
            timestamps.append(ts)
            day_ids.append(d)
            price *= (1.0 + rng.normal(0, 0.0008))
            closes.append(price)

    df = pd.DataFrame({"timestamp": timestamps, "day_id": day_ids, "close": closes})
    df["p_pivot_long"] = rng.uniform(0, 1, size=len(df))
    df["p_pivot_short"] = rng.uniform(0, 1, size=len(df))
    df["p_leg_up"] = rng.uniform(0, 1, size=len(df))
    df["p_leg_down"] = rng.uniform(0, 1, size=len(df))
    df["p_cont_up"] = rng.uniform(0, 1, size=len(df))
    df["p_cont_down"] = rng.uniform(0, 1, size=len(df))
    return df


def main():
    df = build_dummy_df()

    feature_cols = [
        "p_pivot_long", "p_pivot_short",
        "p_leg_up", "p_leg_down",
        "p_cont_up", "p_cont_down"
    ]

    env = TradingEnv(
        df=df,
        feature_cols=feature_cols,
        add_time_features=True,
        add_position_features=True,
        commission_per_trade=0.00,
        slippage_bps=0.5,
        spread_bps=0.5,
        flip_penalty=0.0,
        force_flat_at_close=True,
        allow_direct_flip=True,
        seed=7,
    )

    model = train_ppo(
        env,
        total_timesteps=20_000,
        rollout_len=1024,
        train_epochs=5,
        minibatch_size=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True,
    )

    report = evaluate_policy(env, model, n_days=5, device="cpu")
    print(report)
    print("Avg pnl component:", report["pnl_component"].mean(), "Avg costs:", report["costs_component"].mean())


if __name__ == "__main__":
    main()
