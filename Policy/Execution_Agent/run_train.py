from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch

_POLICY_ROOT = Path(__file__).resolve().parents[1]
if str(_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_ROOT))

from Data.retrieve_data import normalize_ticker
from Agent.train import train_ppo
from Execution_Agent.data import (
    build_execution_frame,
    default_execution_feature_cols,
    ensure_numeric_non_nan,
)
from Execution_Agent.env import (
    ACTION_ENTER,
    ACTION_EXIT,
    ACTION_SCALE_IN,
    ACTION_SCALE_OUT,
    ACTION_WAIT,
    N_EXEC_ACTIONS,
    ExecutionEnvConfig,
    make_execution_env,
)
from Execution_Agent.eval import evaluate_policy_with_trace, summarize_trace
from Execution_Agent.oracle import OracleConfig, build_oracle_entry_labels, train_oracle_sniper


def _action_map() -> dict[int, str]:
    return {
        ACTION_WAIT: "WAIT",
        ACTION_ENTER: "ENTER",
        ACTION_SCALE_IN: "SCALE_IN",
        ACTION_SCALE_OUT: "SCALE_OUT",
        ACTION_EXIT: "EXIT",
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train direction-gated 1m execution PPO agent.")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--raw-1m-parquet", required=True)
    p.add_argument("--htf-intent-path", required=True, help="15m intent trace with timestamp + action_dir_idx/action_mag (or htf_dir/htf_conf).")
    p.add_argument("--output-dir", default="Data/outputs/execution_agent")
    p.add_argument("--tz", default="America/New_York")

    p.add_argument("--train-frac", type=float, default=0.85)
    p.add_argument("--drop-na", action="store_true", default=True)

    p.add_argument("--skip-oracle", action="store_true")
    p.add_argument("--oracle-max-wait-min", type=int, default=12)
    p.add_argument("--oracle-horizon-min", type=int, default=20)
    p.add_argument("--oracle-mae-weight", type=float, default=1.5)
    p.add_argument("--oracle-cost-ret", type=float, default=0.0002)

    p.add_argument("--total-timesteps", type=int, default=1_500_000)
    p.add_argument("--rollout-len", type=int, default=1024)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-ratio", type=float, default=0.2)
    p.add_argument("--pi-lr", type=float, default=3e-4)
    p.add_argument("--vf-lr", type=float, default=1e-3)
    p.add_argument("--train-epochs", type=int, default=5)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--entropy-coef", type=float, default=0.004)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto")

    p.add_argument("--checkpoint-every-steps", type=int, default=250_000)
    p.add_argument("--checkpoint-start-steps", type=int, default=1_500_000)
    p.add_argument("--checkpoint-dir", default="Data/outputs/execution_agent/checkpoints")

    p.add_argument("--max-units", type=int, default=3)
    p.add_argument("--entry-units", type=int, default=1)
    p.add_argument("--baseline-units", type=int, default=1)
    p.add_argument("--episode-mode", choices=["flip_window", "day"], default="flip_window")
    p.add_argument("--flip-window-minutes", type=int, default=20)
    p.add_argument("--flip-start-delay-min", type=int, default=0)
    p.add_argument("--force-flat-on-dir-flip", action="store_true", default=False)
    p.add_argument("--entry-min-since-flip", type=int, default=1)
    p.add_argument("--entry-max-since-flip", type=int, default=12)

    p.add_argument("--spread-bps", type=float, default=1.0)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--commission-per-unit-ret", type=float, default=0.0)
    p.add_argument("--trade-penalty-ret", type=float, default=0.00002)
    p.add_argument("--churn-penalty-ret", type=float, default=0.00003)
    p.add_argument("--mae-penalty-lambda", type=float, default=1.0)
    p.add_argument("--low-conf-threshold", type=float, default=0.35)
    p.add_argument("--low-conf-penalty-lambda", type=float, default=0.00015)
    return p.parse_args()


def _build_env_cfg(args: argparse.Namespace) -> ExecutionEnvConfig:
    return ExecutionEnvConfig(
        max_units=int(args.max_units),
        entry_units=int(args.entry_units),
        baseline_units=int(args.baseline_units),
        force_flat_on_dir_flip=bool(args.force_flat_on_dir_flip),
        entry_min_since_flip=int(args.entry_min_since_flip),
        entry_max_since_flip=int(args.entry_max_since_flip),
        spread_bps=float(args.spread_bps),
        slippage_bps=float(args.slippage_bps),
        commission_per_unit_ret=float(args.commission_per_unit_ret),
        trade_penalty_ret=float(args.trade_penalty_ret),
        churn_penalty_ret=float(args.churn_penalty_ret),
        mae_penalty_lambda=float(args.mae_penalty_lambda),
        low_conf_threshold=float(args.low_conf_threshold),
        low_conf_penalty_lambda=float(args.low_conf_penalty_lambda),
        episode_mode=str(args.episode_mode),
        flip_start_delay_min=int(args.flip_start_delay_min),
        flip_window_minutes=int(args.flip_window_minutes),
        seed=int(args.seed),
    )


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[execution_train] Building 1m execution matrix...")
    df = build_execution_frame(
        ticker=args.ticker,
        raw_1m_path=args.raw_1m_parquet,
        htf_intent_path=args.htf_intent_path,
        tz=args.tz,
    )

    if not args.skip_oracle:
        print("[execution_train] Building oracle entry labels...")
        oracle_cfg = OracleConfig(
            max_wait_min=int(args.oracle_max_wait_min),
            horizon_min=int(args.oracle_horizon_min),
            mae_weight=float(args.oracle_mae_weight),
            cost_per_trade_ret=float(args.oracle_cost_ret),
        )
        df = build_oracle_entry_labels(df, cfg=oracle_cfg)
        pre_oracle_features = default_execution_feature_cols(df)
        pre_oracle_features = [c for c in pre_oracle_features if c not in {"oracle_enter", "oracle_score"}]
        sniper_prob, metrics = train_oracle_sniper(
            df,
            feature_cols=pre_oracle_features,
            save_model_path=output_dir / "oracle_sniper_xgb.json",
        )
        df["sniper_enter_prob"] = sniper_prob
        print("[execution_train] Oracle metrics:", metrics)
    else:
        df["oracle_enter"] = 0
        df["oracle_score"] = 0.0
        df["sniper_enter_prob"] = 0.0

    feature_cols = default_execution_feature_cols(df)
    if args.drop_na:
        df = ensure_numeric_non_nan(df, feature_cols=feature_cols)
    if len(df) < 1000:
        raise ValueError("Execution dataset is too small after cleaning.")

    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = max(1, min(len(df) - 1, int(float(args.train_frac) * len(df))))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(
        "[execution_train] Dataset sizes:",
        f"train={len(train_df):,}",
        f"test={len(test_df):,}",
        f"features={len(feature_cols)}",
    )

    env_cfg = _build_env_cfg(args)
    train_env = make_execution_env(df=train_df, feature_cols=feature_cols, config=env_cfg)

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (Path.cwd() / checkpoint_dir).resolve()

    print("[execution_train] PPO fine-tune (baseline-relative reward)...")
    model, history = train_ppo(
        train_env,
        total_timesteps=int(args.total_timesteps),
        rollout_len=int(args.rollout_len),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_ratio=float(args.clip_ratio),
        pi_lr=float(args.pi_lr),
        vf_lr=float(args.vf_lr),
        train_epochs=int(args.train_epochs),
        minibatch_size=int(args.minibatch_size),
        entropy_coef=float(args.entropy_coef),
        value_coef=float(args.value_coef),
        max_grad_norm=float(args.max_grad_norm),
        action_type="discrete",
        n_actions=N_EXEC_ACTIONS,
        device=str(args.device),
        seed=int(args.seed),
        verbose=True,
        return_history=True,
        checkpoint_every_steps=int(args.checkpoint_every_steps),
        checkpoint_start_steps=int(args.checkpoint_start_steps),
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_prefix="execution_ppo",
        checkpoint_payload={
            "n_actions": N_EXEC_ACTIONS,
            "action_type": "discrete",
            "feature_cols": feature_cols,
            "action_map": _action_map(),
            "env_overrides": env_cfg.__dict__,
            "ticker": normalize_ticker(args.ticker),
        },
    )

    pd.DataFrame(history).to_csv(output_dir / "ppo_train_metrics.csv", index=False)
    model_path = output_dir / "ppo_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": train_env.obs_dim,
            "n_actions": N_EXEC_ACTIONS,
            "action_type": "discrete",
            "feature_cols": feature_cols,
            "action_map": _action_map(),
            "env_overrides": env_cfg.__dict__,
            "ticker": normalize_ticker(args.ticker),
        },
        model_path,
    )
    print(f"[execution_train] Saved model: {model_path}")

    test_env = make_execution_env(df=test_df, feature_cols=feature_cols, config=env_cfg)
    trace = evaluate_policy_with_trace(
        test_env,
        model,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
        deterministic=True,
    )
    trace_out = output_dir / "execution_trace.csv"
    trace.to_csv(trace_out, index=False)
    print(f"[execution_train] Saved trace: {trace_out}")
    print("[execution_train] Eval summary:", summarize_trace(trace))

    matrix_out = output_dir / "execution_matrix.parquet"
    df.to_parquet(matrix_out, index=False)
    print(f"[execution_train] Saved matrix: {matrix_out}")


if __name__ == "__main__":
    main()
