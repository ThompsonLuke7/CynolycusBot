from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch

_POLICY_ROOT = Path(__file__).resolve().parents[1]
if str(_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_ROOT))

from Agent.model import ActorCritic
from Execution_Agent.data import default_execution_feature_cols, ensure_numeric_non_nan
from Execution_Agent.env import ExecutionEnvConfig, make_execution_env
from Execution_Agent.eval import evaluate_policy_with_trace, summarize_trace


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate execution PPO model.")
    p.add_argument("--data-csv", default=None)
    p.add_argument("--data-parquet", default=None)
    p.add_argument("--model-path", default="Data/outputs/execution_agent/ppo_model.pt")
    p.add_argument("--trace-out", default="Data/outputs/execution_agent/execution_trace_eval.csv")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def _load_df(args: argparse.Namespace) -> pd.DataFrame:
    if args.data_parquet:
        return pd.read_parquet(args.data_parquet)
    if args.data_csv:
        return pd.read_csv(args.data_csv)
    raise SystemExit("Provide --data-csv or --data-parquet.")


def main() -> None:
    args = _parse_args()
    ckpt = torch.load(Path(args.model_path), map_location="cpu")
    df = _load_df(args)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    feature_cols = list(ckpt.get("feature_cols") or default_execution_feature_cols(df))
    df = ensure_numeric_non_nan(df, feature_cols=feature_cols)
    if df.empty:
        raise SystemExit("No rows left after feature NaN filtering.")

    env_cfg = ExecutionEnvConfig(**(ckpt.get("env_overrides") or {}))
    env = make_execution_env(df=df, feature_cols=feature_cols, config=env_cfg)
    model = ActorCritic(
        obs_dim=int(ckpt["obs_dim"]),
        n_actions=int(ckpt.get("n_actions", 5)),
        action_type=str(ckpt.get("action_type", "discrete")),
        action_dim=int(ckpt.get("action_dim", 1)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    trace = evaluate_policy_with_trace(env, model, device=device, deterministic=True)
    out_path = Path(args.trace_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace.to_csv(out_path, index=False)
    print(f"Saved trace: {out_path}")
    print("Eval summary:", summarize_trace(trace))


if __name__ == "__main__":
    main()

