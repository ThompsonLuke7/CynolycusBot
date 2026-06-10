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
from training_logging import log_training_run
from Execution_Agent.data import (
    build_execution_frame,
    default_execution_feature_cols,
    ensure_numeric_non_nan,
)
from Execution_Agent.env import (
    ACTION_EXECUTE,
    ACTION_WAIT,
    N_EXEC_ACTIONS,
    ExecutionEnvConfig,
    make_execution_env,
)
from Execution_Agent.eval import evaluate_policy_with_trace, summarize_trace
from Execution_Agent.oracle import (
    OracleConfig,
    build_oracle_event_labels,
    train_oracle_sniper_walk_forward,
)


def _action_map() -> dict[int, str]:
    return {
        ACTION_WAIT: "WAIT",
        ACTION_EXECUTE: "EXECUTE",
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
    p.add_argument("--oracle-oof-folds", type=int, default=5)
    p.add_argument("--oracle-oof-initial-size", type=int, default=0)

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
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--target-kl", type=float, default=0.015)
    p.add_argument("--policy-hidden-size", type=int, default=128)
    p.add_argument("--no-policy-head-mlp", action="store_true")
    p.add_argument("--policy-layer-norm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--policy-dropout-p", type=float, default=0.05)
    p.add_argument("--eval-every-updates", type=int, default=5)
    p.add_argument("--eval-n-days", type=int, default=0)
    p.add_argument("--early-stop-patience-updates", type=int, default=8)
    p.add_argument(
        "--early-stop-metric",
        type=str,
        default="pnl_net_mean",
        choices=["pnl_net_mean", "pnl_net_sum", "pnl_mean", "pnl_sum", "costs_mean", "trades_mean"],
    )
    p.add_argument("--early-stop-min-delta", type=float, default=0.0)
    p.add_argument("--no-restore-best-on-early-stop", action="store_true")
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
    p.add_argument("--mae-penalty-lambda", type=float, default=0.25)
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


def _extract_best_eval_metrics(
    history_df: pd.DataFrame,
    metric_name: str,
) -> dict[str, float | str]:
    if history_df.empty:
        return {}
    out: dict[str, float | str] = {"metric_name": str(metric_name)}
    if "eval_metric" in history_df.columns:
        eval_series = pd.to_numeric(history_df["eval_metric"], errors="coerce").dropna()
        if not eval_series.empty:
            out["final_eval_metric"] = float(eval_series.iloc[-1])
    if "best_eval_metric" in history_df.columns:
        best_series = pd.to_numeric(history_df["best_eval_metric"], errors="coerce").dropna()
        if not best_series.empty:
            best_val = float(best_series.iloc[-1])
            out["best_eval_metric"] = best_val
            matches = history_df.loc[
                pd.to_numeric(history_df["best_eval_metric"], errors="coerce") == best_val
            ]
            if not matches.empty and "steps" in matches.columns:
                out["best_eval_metric_steps"] = float(
                    pd.to_numeric(matches["steps"], errors="coerce").iloc[0]
                )
    return out


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
        allow_exact_matches=False,
    )
    oracle_metrics: dict[str, float] = {}
    oracle_artifacts: dict[str, str] = {}

    if not args.skip_oracle:
        print("[execution_train] Building oracle event labels...")
        oracle_cfg = OracleConfig(
            max_wait_min=int(args.oracle_max_wait_min),
            horizon_min=int(args.oracle_horizon_min),
            mae_weight=float(args.oracle_mae_weight),
            cost_per_trade_ret=float(args.oracle_cost_ret),
        )
        df = build_oracle_event_labels(df, cfg=oracle_cfg)
        pre_oracle_features = default_execution_feature_cols(df)
        oof_init = int(args.oracle_oof_initial_size)
        enter_oof, enter_full, enter_metrics = train_oracle_sniper_walk_forward(
            df,
            feature_cols=pre_oracle_features,
            label_col="oracle_enter",
            n_folds=int(args.oracle_oof_folds),
            initial_train_size=(None if oof_init <= 0 else oof_init),
            random_seed=int(args.seed),
            save_full_model_path=output_dir / "oracle_enter_xgb.json",
            event_window_max_wait=int(args.oracle_max_wait_min),
        )
        exit_oof, exit_full, exit_metrics = train_oracle_sniper_walk_forward(
            df,
            feature_cols=pre_oracle_features,
            label_col="oracle_exit",
            n_folds=int(args.oracle_oof_folds),
            initial_train_size=(None if oof_init <= 0 else oof_init),
            random_seed=int(args.seed) + 101,
            save_full_model_path=output_dir / "oracle_exit_xgb.json",
            event_window_max_wait=int(args.oracle_max_wait_min),
        )

        # Use OOF predictions for RL features; warmup rows (no OOF) get causal priors.
        enter_causal_prior = (
            pd.to_numeric(df["oracle_enter"], errors="coerce").fillna(0.0).astype(float).expanding(min_periods=1).mean().shift(1).fillna(0.0)
        )
        exit_causal_prior = (
            pd.to_numeric(df["oracle_exit"], errors="coerce").fillna(0.0).astype(float).expanding(min_periods=1).mean().shift(1).fillna(0.0)
        )
        sniper_enter_prob = (
            pd.to_numeric(enter_oof, errors="coerce")
            .astype(float)
            .fillna(enter_causal_prior)
            .clip(0.0, 1.0)
        )
        sniper_exit_prob = (
            pd.to_numeric(exit_oof, errors="coerce")
            .astype(float)
            .fillna(exit_causal_prior)
            .clip(0.0, 1.0)
        )
        df["sniper_enter_prob"] = sniper_enter_prob
        df["sniper_exit_prob"] = sniper_exit_prob
        oracle_probs_out = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df["timestamp"], errors="coerce"),
                "oracle_enter": pd.to_numeric(df["oracle_enter"], errors="coerce").fillna(0.0),
                "oracle_exit": pd.to_numeric(df["oracle_exit"], errors="coerce").fillna(0.0),
                "oracle_score": pd.to_numeric(df["oracle_score"], errors="coerce"),
                "oracle_exit_score": pd.to_numeric(df["oracle_exit_score"], errors="coerce"),
                "sniper_enter_prob_oof": pd.to_numeric(enter_oof, errors="coerce"),
                "sniper_enter_prob_full": pd.to_numeric(enter_full, errors="coerce"),
                "sniper_exit_prob_oof": pd.to_numeric(exit_oof, errors="coerce"),
                "sniper_exit_prob_full": pd.to_numeric(exit_full, errors="coerce"),
                "sniper_enter_prob": sniper_enter_prob,
                "sniper_exit_prob": sniper_exit_prob,
            }
        )
        oracle_probs_out.to_parquet(output_dir / "oracle_sniper_probs.parquet", index=False)
        combined_metrics = {
            **{f"enter_{k}": v for k, v in enter_metrics.items()},
            **{f"exit_{k}": v for k, v in exit_metrics.items()},
        }
        oracle_metrics = combined_metrics
        oracle_artifacts = {
            "oracle_enter_model": str(output_dir / "oracle_enter_xgb.json"),
            "oracle_exit_model": str(output_dir / "oracle_exit_xgb.json"),
            "oracle_probs_parquet": str(output_dir / "oracle_sniper_probs.parquet"),
        }
        print("[execution_train] Oracle metrics:", combined_metrics)
    else:
        df["oracle_enter"] = 0
        df["oracle_exit"] = 0
        df["oracle_score"] = 0.0
        df["oracle_exit_score"] = 0.0
        df["sniper_enter_prob"] = 0.0
        df["sniper_exit_prob"] = 0.0
        oracle_metrics = {"oracle_skipped": 1.0}

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
    eval_env_for_es = None
    if int(args.eval_every_updates) > 0:
        if test_df.empty:
            print(
                "[execution_train] Early-stop eval requested but test split is empty; disabling eval monitor."
            )
        else:
            eval_env_for_es = make_execution_env(
                df=test_df,
                feature_cols=feature_cols,
                config=env_cfg,
            )

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
        hidden_size=int(args.policy_hidden_size),
        policy_head_mlp=not bool(args.no_policy_head_mlp),
        policy_layer_norm=bool(args.policy_layer_norm),
        policy_dropout_p=float(args.policy_dropout_p),
        weight_decay=float(args.weight_decay),
        target_kl=float(args.target_kl),
        eval_env=eval_env_for_es,
        eval_every_updates=int(args.eval_every_updates),
        eval_n_days=int(args.eval_n_days),
        early_stop_patience_updates=int(args.early_stop_patience_updates),
        early_stop_metric=str(args.early_stop_metric),
        early_stop_min_delta=float(args.early_stop_min_delta),
        early_stop_best_model_path=(
            str(output_dir / "ppo_model_best.pt")
            if int(args.eval_every_updates) > 0 and int(args.early_stop_patience_updates) > 0
            else None
        ),
        restore_best_on_early_stop=not bool(args.no_restore_best_on_early_stop),
    )

    history_df = pd.DataFrame(history)
    history_csv_path = output_dir / "ppo_train_metrics.csv"
    history_df.to_csv(history_csv_path, index=False)
    model_path = output_dir / "ppo_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": train_env.obs_dim,
            "n_actions": N_EXEC_ACTIONS,
            "action_type": "discrete",
            "action_dim": int(getattr(model, "action_dim", 1)),
            "action_low": -1.0,
            "action_high": 1.0,
            "policy_hidden_size": int(args.policy_hidden_size),
            "policy_head_mlp": not bool(args.no_policy_head_mlp),
            "policy_layer_norm": bool(args.policy_layer_norm),
            "policy_dropout_p": float(args.policy_dropout_p),
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
    trace_summary = summarize_trace(trace)
    print("[execution_train] Eval summary:", trace_summary)

    matrix_out = output_dir / "execution_matrix.parquet"
    df.to_parquet(matrix_out, index=False)
    print(f"[execution_train] Saved matrix: {matrix_out}")
    final_train_metrics = (
        history_df.iloc[-1].to_dict() if not history_df.empty else {}
    )
    best_validation_metrics = _extract_best_eval_metrics(
        history_df,
        metric_name=str(args.early_stop_metric),
    )
    log_paths = log_training_run(
        run_name="execution_agent_run_train",
        output_dir=output_dir,
        hyperparameters=vars(args),
        train_metrics=final_train_metrics,
        validation_metrics=trace_summary,
        best_validation_metrics=best_validation_metrics,
        artifacts={
            "model_path": str(model_path),
            "train_metrics_csv": str(history_csv_path),
            "trace_csv": str(trace_out),
            "execution_matrix_parquet": str(matrix_out),
            **oracle_artifacts,
        },
        extra={
            "ticker": normalize_ticker(args.ticker),
            "feature_count": len(feature_cols),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            **oracle_metrics,
        },
    )
    print(f"[execution_train] Saved training run summary: {log_paths['latest_path']}")


if __name__ == "__main__":
    main()
