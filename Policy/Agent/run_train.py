from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import torch

_POLICY_ROOT = Path(__file__).resolve().parents[1]
if str(_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_ROOT))

from training_pipeline import (
    PipelineConfig,
    build_agent_training_matrix,
    filter_splits_for_non_nan,
    load_tree_split_indices,
    split_agent_matrix,
)
from Data.retrieve_data import normalize_ticker
from Agent.env import VecTradingEnv
from Agent.env_config import make_trading_env
from Agent.train import train_ppo
from Agent.eval import evaluate_loss_metrics, evaluate_policy, evaluate_policy_with_trace
from training_logging import log_training_run


def _daily_first_last(df):
    ordered = df.sort_values("timestamp")
    grouped = ordered.groupby("day_id", sort=True)
    daily = grouped.agg(
        first_ts=("timestamp", "first"),
        last_ts=("timestamp", "last"),
        first_close=("close", "first"),
        last_close=("close", "last"),
    )
    return daily.reset_index(drop=True)


def _buy_and_hold(daily_summary, initial_cash, trade_cost_ret):
    start = float(daily_summary["first_close"].iloc[0])
    end = float(daily_summary["last_close"].iloc[-1])
    total_ret = (end / start - 1.0) - trade_cost_ret(start) - trade_cost_ret(end)
    final_value = initial_cash * (1.0 + total_ret)
    return final_value, final_value - initial_cash, total_ret


def _intraday_long(daily_summary, initial_cash, trade_cost_ret):
    equity = float(initial_cash)
    for _, row in daily_summary.iterrows():
        start = float(row["first_close"])
        end = float(row["last_close"])
        day_ret = (end / start - 1.0) - trade_cost_ret(start) - trade_cost_ret(end)
        equity *= (1.0 + day_ret)
    total_ret = equity / initial_cash - 1.0
    return equity, equity - initial_cash, total_ret


def _dca_down_days(daily_summary, initial_cash):
    closes = daily_summary["last_close"].astype(float).to_list()
    tranche = initial_cash / max(1, len(closes))
    cash = float(initial_cash)
    shares = 0.0
    prev = None
    for close in closes:
        if prev is not None and close < prev and cash > 0:
            invest = min(tranche, cash)
            shares += invest / close
            cash -= invest
        prev = close
    final_value = cash + shares * closes[-1]
    return final_value, final_value - initial_cash, tranche


def _agent_equity_from_trace(trace, initial_cash):
    equity = float(initial_cash)
    equity_series = []
    for _, row in trace.iterrows():
        costs = float(row.get("reward_costs", 0.0)) + float(row.get("forced_flat_cost", 0.0))
        net_ret = float(row.get("reward_pnl", 0.0)) - float(costs)
        equity *= (1.0 + net_ret)
        equity_series.append(equity)
    trace = trace.copy()
    trace["equity"] = equity_series
    return trace


def _is_vix_feature(col: str) -> bool:
    return col.startswith("vix_") or col.endswith("_x_vix")


def _fit_feature_zscore_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, object]:
    stats: dict[str, dict[str, float]] = {}
    skipped: list[str] = []
    for col in feature_cols:
        if col not in df.columns:
            skipped.append(col)
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        mean_val = float(values.mean(skipna=True))
        std_val = float(values.std(skipna=True, ddof=0))
        if (not np.isfinite(mean_val)) or (not np.isfinite(std_val)) or std_val <= 1e-12:
            skipped.append(col)
            continue
        stats[col] = {"mean": mean_val, "std": std_val}
    return {
        "enabled": True,
        "method": "zscore",
        "cols": list(stats.keys()),
        "stats": stats,
        "skipped_cols": skipped,
    }


def _apply_feature_zscore(
    df: pd.DataFrame,
    feature_norm: dict[str, object] | None,
) -> pd.DataFrame:
    if not feature_norm or not bool(feature_norm.get("enabled", False)):
        return df
    stats = feature_norm.get("stats")
    if not isinstance(stats, dict) or not stats:
        return df
    out = df.copy()
    for col, cfg in stats.items():
        if col not in out.columns or not isinstance(cfg, dict):
            continue
        mean_val = float(cfg.get("mean", 0.0))
        std_val = float(cfg.get("std", 1.0))
        if (not np.isfinite(mean_val)) or (not np.isfinite(std_val)) or std_val <= 1e-12:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        out[col] = (values - mean_val) / std_val
    return out


def _build_train_env(
    *,
    df: pd.DataFrame,
    feature_cols: list[str],
    env_kwargs: dict[str, object],
    num_envs: int,
    seed_base: int,
):
    n_envs = max(1, int(num_envs))
    if n_envs > 1:
        envs = [
            make_trading_env(
                df=df,
                feature_cols=feature_cols,
                seed=int(seed_base) + i,
                **env_kwargs,
            )
            for i in range(n_envs)
        ]
        return VecTradingEnv(envs, auto_reset=True, stagger_reset=True)
    return make_trading_env(
        df=df,
        feature_cols=feature_cols,
        seed=int(seed_base),
        **env_kwargs,
    )


def _derive_htf_intent_from_trace(trace: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(trace.get("timestamp"), errors="coerce")
    if "action_dir_idx" in trace.columns:
        idx = pd.to_numeric(trace["action_dir_idx"], errors="coerce")
        d = pd.Series(0.0, index=trace.index)
        d[idx == 1] = 1.0
        d[idx == 2] = -1.0
        out["htf_dir"] = d
    else:
        out["htf_dir"] = np.sign(
            pd.to_numeric(trace.get("action", 0.0), errors="coerce").fillna(0.0)
        )
    if "action_mag" in trace.columns:
        out["htf_conf"] = (
            pd.to_numeric(trace["action_mag"], errors="coerce")
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
    else:
        out["htf_conf"] = (
            pd.to_numeric(trace.get("action", 0.0), errors="coerce")
            .abs()
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
    if "convex_atr_scale" in trace.columns:
        out["htf_atr_pct"] = pd.to_numeric(trace["convex_atr_scale"], errors="coerce")
    else:
        out["htf_atr_pct"] = np.nan
    out["htf_expected_edge"] = np.nan
    out = out[out["timestamp"].notna()].copy()
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    out["htf_dir"] = out["htf_dir"].fillna(0.0).clip(-1.0, 1.0)
    out["htf_conf"] = out["htf_conf"].fillna(0.0).clip(0.0, 1.0)
    flipped = out["htf_dir"].ne(out["htf_dir"].shift(1))
    last_flip_ts = out["timestamp"].where(flipped).ffill()
    minutes = (out["timestamp"] - last_flip_ts).dt.total_seconds() / 60.0
    out["time_since_flip_min"] = minutes.fillna(0.0).clip(lower=0.0)
    return out.reset_index(drop=True)


def _build_walkforward_day_folds(
    *,
    df: pd.DataFrame,
    n_folds: int,
    initial_train_days: int,
) -> list[dict[str, object]]:
    if "day_id" not in df.columns:
        raise ValueError("Walk-forward OOF requires a day_id column.")
    ordered = df
    if "timestamp" in ordered.columns:
        ordered = ordered.sort_values("timestamp")
    day_values = list(pd.unique(pd.Series(ordered["day_id"]).dropna()))
    n_days = len(day_values)
    if n_days < 3:
        raise ValueError("Need at least 3 distinct day_id values for walk-forward OOF.")
    if int(n_folds) < 1:
        raise ValueError("wf-n-folds must be >= 1.")

    init_days = int(initial_train_days)
    if init_days <= 0:
        init_days = max(1, n_days // (int(n_folds) + 1))
    if init_days >= n_days:
        raise ValueError(
            f"wf-initial-train-days={init_days} must be < number of days ({n_days})."
        )

    remaining = n_days - init_days
    fold_size = remaining // int(n_folds)
    if fold_size <= 0:
        raise ValueError(
            "Walk-forward fold size is 0. Reduce --wf-n-folds or lower --wf-initial-train-days."
        )

    folds: list[dict[str, object]] = []
    for fold_id in range(int(n_folds)):
        eval_start = init_days + fold_id * fold_size
        eval_end = init_days + (fold_id + 1) * fold_size
        if fold_id == int(n_folds) - 1:
            eval_end = n_days
        if eval_start >= eval_end:
            continue
        train_days = set(day_values[:eval_start])
        eval_days = set(day_values[eval_start:eval_end])
        folds.append(
            {
                "fold_id": fold_id,
                "train_days": train_days,
                "eval_days": eval_days,
                "train_day_count": len(train_days),
                "eval_day_count": len(eval_days),
                "eval_day_start": day_values[eval_start],
                "eval_day_end": day_values[eval_end - 1],
            }
        )
    if not folds:
        raise ValueError("No walk-forward folds were created.")
    return folds


def _run_walkforward_oof(
    *,
    source_df: pd.DataFrame,
    feature_cols: list[str],
    env_kwargs: dict[str, object],
    cfg: PipelineConfig,
    args: argparse.Namespace,
    action_deadband: float,
) -> dict[str, Path]:
    working = source_df.copy()
    if "timestamp" not in working.columns:
        raise ValueError("Walk-forward OOF requires a timestamp column.")
    if "timestamp" in working.columns:
        working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
        working = working[working["timestamp"].notna()].copy()
        working = working.sort_values("timestamp").reset_index(drop=True)
    folds = _build_walkforward_day_folds(
        df=working,
        n_folds=int(args.wf_n_folds),
        initial_train_days=int(args.wf_initial_train_days),
    )

    def _resolve_local_path(path_like: str) -> Path:
        p = Path(path_like)
        if p.is_absolute():
            return p
        return (Path.cwd() / p).resolve()

    trace_out_path = _resolve_local_path(args.wf_trace_out)
    intent_out_path = _resolve_local_path(args.wf_intent_out)
    manifest_out_path = _resolve_local_path(args.wf_manifest_out)
    model_dir = _resolve_local_path(args.wf_model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    trace_out_path.parent.mkdir(parents=True, exist_ok=True)
    intent_out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_out_path.parent.mkdir(parents=True, exist_ok=True)

    wf_total_timesteps = (
        int(args.wf_total_timesteps)
        if int(args.wf_total_timesteps) > 0
        else int(args.total_timesteps)
    )
    fold_traces: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    print(
        "[run_train] Walk-forward OOF:",
        f"folds={len(folds)}",
        f"timesteps_per_fold={wf_total_timesteps:,}",
        f"source_rows={len(working):,}",
    )
    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_fold_df = (
            working[working["day_id"].isin(fold["train_days"])]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        eval_fold_df = (
            working[working["day_id"].isin(fold["eval_days"])]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if train_fold_df.empty or eval_fold_df.empty:
            print(f"[run_train] Skipping empty fold {fold_id}.")
            continue

        print(
            f"[run_train] Fold {fold_id + 1}/{len(folds)}:",
            f"train_rows={len(train_fold_df):,}",
            f"eval_rows={len(eval_fold_df):,}",
            f"train_days={int(fold['train_day_count'])}",
            f"eval_days={int(fold['eval_day_count'])}",
            f"eval_day_range=[{fold['eval_day_start']}..{fold['eval_day_end']}]",
        )
        fold_feature_norm: dict[str, object] | None = None
        if bool(getattr(args, "feature_zscore", True)):
            fold_feature_norm = _fit_feature_zscore_stats(train_fold_df, feature_cols)
            train_fold_df = _apply_feature_zscore(train_fold_df, fold_feature_norm)
            eval_fold_df = _apply_feature_zscore(eval_fold_df, fold_feature_norm)
            print(
                f"[run_train] Fold {fold_id + 1} z-score:",
                f"scaled_cols={len(fold_feature_norm.get('cols', []))}",
                f"skipped_cols={len(fold_feature_norm.get('skipped_cols', []))}",
            )

        fold_seed = int(args.seed) + fold_id * 17
        train_env = _build_train_env(
            df=train_fold_df,
            feature_cols=feature_cols,
            env_kwargs=env_kwargs,
            num_envs=int(args.num_envs),
            seed_base=fold_seed,
        )
        model, history = train_ppo(
            train_env,
            total_timesteps=wf_total_timesteps,
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
            action_type=str(args.policy_action_type),
            device=str(args.device),
            seed=fold_seed,
            verbose=True,
            return_history=True,
            checkpoint_every_steps=0,
            checkpoint_start_steps=0,
            checkpoint_payload={
                "obs_dim": train_env.obs_dim,
                "n_actions": 3,
                "action_type": str(args.policy_action_type),
                "action_low": -1.0,
                "action_high": 1.0,
                "action_deadband": float(action_deadband),
                "feature_cols": feature_cols,
                "config": cfg.__dict__,
                "reward_mode": str(args.reward_mode),
                "env_overrides": env_kwargs,
                "feature_norm": fold_feature_norm or {},
            },
            hidden_size=int(args.policy_hidden_size),
            policy_head_mlp=not bool(args.no_policy_head_mlp),
            policy_layer_norm=bool(args.policy_layer_norm),
            policy_dropout_p=float(args.policy_dropout_p),
            weight_decay=float(args.weight_decay),
            target_kl=float(args.target_kl),
            eval_env=(
                make_trading_env(
                    df=eval_fold_df,
                    feature_cols=feature_cols,
                    seed=fold_seed + 10_000,
                    **env_kwargs,
                )
                if int(args.eval_every_updates) > 0
                else None
            ),
            eval_every_updates=int(args.eval_every_updates),
            eval_n_days=int(args.eval_n_days),
            early_stop_patience_updates=int(args.early_stop_patience_updates),
            early_stop_metric=str(args.early_stop_metric),
            early_stop_min_delta=float(args.early_stop_min_delta),
            early_stop_best_model_path=(
                str((model_dir / f"fold_{fold_id:02d}" / "ppo_model_best.pt"))
                if int(args.eval_every_updates) > 0 and int(args.early_stop_patience_updates) > 0
                else None
            ),
            restore_best_on_early_stop=not bool(args.no_restore_best_on_early_stop),
        )

        fold_dir = model_dir / f"fold_{fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        history_path = fold_dir / "ppo_train_metrics.csv"
        pd.DataFrame(history).to_csv(history_path, index=False)

        fold_model_path = fold_dir / "ppo_model.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "obs_dim": train_env.obs_dim,
                "n_actions": 3,
                "action_type": str(args.policy_action_type),
                "action_dim": int(getattr(model, "action_dim", 1)),
                "action_low": -1.0,
                "action_high": 1.0,
                "policy_hidden_size": int(args.policy_hidden_size),
                "policy_head_mlp": not bool(args.no_policy_head_mlp),
                "policy_layer_norm": bool(args.policy_layer_norm),
                "policy_dropout_p": float(args.policy_dropout_p),
                "action_deadband": float(action_deadband),
                "feature_cols": feature_cols,
                "config": cfg.__dict__,
                "reward_mode": str(args.reward_mode),
                "env_overrides": env_kwargs,
                "feature_norm": fold_feature_norm or {},
                "fold_id": fold_id,
                "fold_train_days": int(fold["train_day_count"]),
                "fold_eval_days": int(fold["eval_day_count"]),
            },
            fold_model_path,
        )

        eval_env = make_trading_env(
            df=eval_fold_df,
            feature_cols=feature_cols,
            **env_kwargs,
        )
        fold_trace = evaluate_policy_with_trace(
            eval_env,
            model,
            device=str(args.device),
            deterministic=True,
        )
        fold_trace["fold_id"] = fold_id
        fold_trace["is_oof"] = True
        fold_trace_path = fold_dir / "agent_trace.csv"
        fold_trace.to_csv(fold_trace_path, index=False)
        fold_traces.append(fold_trace)

        eval_start_ts = pd.to_datetime(eval_fold_df["timestamp"], errors="coerce").min()
        eval_end_ts = pd.to_datetime(eval_fold_df["timestamp"], errors="coerce").max()
        eval_end_excl = (
            eval_end_ts + pd.Timedelta(microseconds=1)
            if pd.notna(eval_end_ts)
            else pd.NaT
        )
        manifest_rows.append(
            {
                "fold_id": fold_id,
                "start_ts": eval_start_ts,
                "end_ts": eval_end_excl,
                "model_path": str(fold_model_path),
                "trace_path": str(fold_trace_path),
                "train_rows": int(len(train_fold_df)),
                "eval_rows": int(len(eval_fold_df)),
                "train_days": int(fold["train_day_count"]),
                "eval_days": int(fold["eval_day_count"]),
                "eval_day_start": fold["eval_day_start"],
                "eval_day_end": fold["eval_day_end"],
            }
        )

    if not fold_traces:
        raise RuntimeError("Walk-forward OOF produced no fold traces.")

    oof_trace = pd.concat(fold_traces, axis=0, ignore_index=True)
    if "timestamp" in oof_trace.columns:
        oof_trace["timestamp"] = pd.to_datetime(oof_trace["timestamp"], errors="coerce")
        oof_trace = oof_trace[oof_trace["timestamp"].notna()].copy()
        oof_trace = oof_trace.sort_values("timestamp")
        oof_trace = oof_trace.drop_duplicates(subset=["timestamp"], keep="last")
    oof_trace.to_csv(trace_out_path, index=False)

    intent = _derive_htf_intent_from_trace(oof_trace)
    if intent_out_path.suffix.lower() == ".csv":
        intent.to_csv(intent_out_path, index=False)
    else:
        intent.to_parquet(intent_out_path, index=False)

    manifest = pd.DataFrame(manifest_rows)
    manifest = manifest.sort_values("start_ts").reset_index(drop=True)
    manifest.to_csv(manifest_out_path, index=False)

    print(
        "[run_train] Walk-forward artifacts:",
        f"trace={trace_out_path}",
        f"intent={intent_out_path}",
        f"manifest={manifest_out_path}",
        f"rows={len(intent):,}",
    )
    return {
        "trace_path": trace_out_path,
        "intent_path": intent_out_path,
        "manifest_path": manifest_out_path,
    }


def _plot_actions(trace, output_path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot skipped: {exc}")
        return

    plot_df = trace.dropna(subset=["timestamp", "close"]).tail(100)
    if plot_df.empty:
        print("Plot skipped: no data.")
        return

    ts = pd.to_datetime(plot_df["timestamp"])
    pos = np.arange(len(plot_df))
    has_ohlc = all(c in plot_df.columns for c in ("open", "high", "low", "close"))
    if has_ohlc:
        ohlc_vals = plot_df[["open", "high", "low", "close"]].to_numpy(dtype=float)
        has_ohlc = np.isfinite(ohlc_vals).any()
    close = plot_df["close"].astype(float)

    prob_cols = []
    if "p_pivot_long" in plot_df.columns:
        prob_cols.append(("p_pivot_long", "#1565C0", "p_pivot_long"))
    if "p_pivot_short" in plot_df.columns:
        prob_cols.append(("p_pivot_short", "#EF6C00", "p_pivot_short"))
    if "p_tb_long" in plot_df.columns:
        prob_cols.append(("p_tb_long", "#2E7D32", "p_tb_long"))
    if "p_tb_short" in plot_df.columns:
        prob_cols.append(("p_tb_short", "#C62828", "p_tb_short"))
    has_probs = False
    if prob_cols:
        prob_vals = plot_df[[c[0] for c in prob_cols]].to_numpy(dtype=float)
        has_probs = np.isfinite(prob_vals).any()

    eps = 1e-3
    pos_now = plot_df["position"].to_numpy(dtype=float) if "position" in plot_df.columns else None
    prev_pos = plot_df["prev_pos"].to_numpy(dtype=float) if "prev_pos" in plot_df.columns else None
    if pos_now is not None and prev_pos is not None:
        longs = (pos_now > eps) & (prev_pos <= eps)
        shorts = (pos_now < -eps) & (prev_pos >= -eps)
    else:
        actions = plot_df["action"].to_numpy(dtype=float)
        prev_actions = np.roll(actions, 1)
        prev_actions[0] = 0.0
        if np.nanmax(np.abs(actions)) > 1.5:
            longs = (actions == 1.0) & (prev_actions != 1.0)
            shorts = (actions == 2.0) & (prev_actions != 2.0)
        else:
            longs = (actions > eps) & (prev_actions <= eps)
            shorts = (actions < -eps) & (prev_actions >= -eps)

    has_heads = False
    if ("action_dir_idx" in plot_df.columns) or ("action_mag" in plot_df.columns):
        head_cols = [c for c in ("action_dir_idx", "action_mag") if c in plot_df.columns]
        if head_cols:
            head_vals = plot_df[head_cols].to_numpy(dtype=float)
            has_heads = np.isfinite(head_vals).any()

    if has_probs and has_heads:
        fig, (ax, ax_prob, ax_heads) = plt.subplots(
            3,
            1,
            figsize=(12, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1, 0.9]},
        )
    elif has_probs:
        fig, (ax, ax_prob) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1]},
        )
        ax_heads = None
    elif has_heads:
        fig, (ax, ax_heads) = plt.subplots(
            2,
            1,
            figsize=(12, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 0.9]},
        )
        ax_prob = None
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax_prob = None
        ax_heads = None
    if has_ohlc:
        open_y = plot_df["open"].to_numpy(dtype=float)
        high_y = plot_df["high"].to_numpy(dtype=float)
        low_y = plot_df["low"].to_numpy(dtype=float)
        close_y = close.to_numpy(dtype=float)
        valid = np.isfinite(open_y) & np.isfinite(high_y) & np.isfinite(low_y) & np.isfinite(close_y)
        up = close_y >= open_y
        wick_color = "#4a4a4a"
        up_color = "#1976D2"
        down_color = "#E53935"
        ax.vlines(pos[valid], low_y[valid], high_y[valid], color=wick_color, linewidth=1.0, zorder=1)
        ax.bar(pos[valid & up], close_y[valid & up] - open_y[valid & up], width=0.8,
               bottom=open_y[valid & up], color=up_color, edgecolor="none", zorder=1.2)
        ax.bar(pos[valid & ~up], close_y[valid & ~up] - open_y[valid & ~up], width=0.8,
               bottom=open_y[valid & ~up], color=down_color, edgecolor="none", zorder=1.2)
        ax.scatter(pos[longs], close_y[longs], c="green", s=22, label="Long", marker="^", zorder=2.2)
        ax.scatter(pos[shorts], close_y[shorts], c="red", s=22, label="Short", marker="v", zorder=2.2)
        day_start = ts.dt.normalize().ne(ts.dt.normalize().shift())
        tick_positions = pos[day_start.to_numpy()]
        tick_labels = ts[day_start].dt.strftime("%Y-%m-%d").to_list()
        if len(tick_positions) > 25:
            step = int(np.ceil(len(tick_positions) / 25))
            tick_positions = tick_positions[::step]
            tick_labels = tick_labels[::step]
        bottom_ax = ax_heads if ax_heads is not None else ax_prob
        if bottom_ax is not None:
            ax.tick_params(labelbottom=False)
            bottom_ax.set_xticks(tick_positions)
            bottom_ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
            bottom_ax.set_xlabel("Date")
        else:
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
            ax.set_xlabel("Date")
    else:
        ax.plot(ts, close, label="SPY close", linewidth=1.5)
        ax.scatter(ts[longs], close[longs], c="green", s=18, label="Long", marker="^")
        ax.scatter(ts[shorts], close[shorts], c="red", s=18, label="Short", marker="v")
        if ax_prob is not None:
            ax_prob.set_xlabel("Time")
        else:
            ax.set_xlabel("Time")

    ax.set_title("SPY Candles with Agent Actions (Test Set)")
    ax.set_ylabel("Price")
    ax.legend()

    if ax_prob is not None and has_probs:
        for col, color, label in prob_cols:
            if col in plot_df.columns:
                ax_prob.plot(
                    pos if has_ohlc else ts,
                    plot_df[col].to_numpy(dtype=float),
                    color=color,
                    linewidth=1.2,
                    label=label,
                )
        ax_prob.set_ylim(0, 1.02)
        ax_prob.set_ylabel("Prob")
        ax_prob.legend(loc="upper left")

    if ax_heads is not None and has_heads:
        if "action_dir_idx" in plot_df.columns:
            dir_idx = pd.to_numeric(plot_df["action_dir_idx"], errors="coerce").to_numpy(dtype=float)
            dir_sign = np.full_like(dir_idx, np.nan, dtype=float)
            dir_sign = np.where(np.isfinite(dir_idx) & (dir_idx == 1.0), 1.0, dir_sign)
            dir_sign = np.where(np.isfinite(dir_idx) & (dir_idx == 2.0), -1.0, dir_sign)
            dir_sign = np.where(np.isfinite(dir_idx) & (dir_idx == 0.0), 0.0, dir_sign)
        else:
            actions = pd.to_numeric(plot_df["action"], errors="coerce").to_numpy(dtype=float)
            dir_sign = np.where(np.abs(actions) <= eps, 0.0, np.sign(actions))
        if "action_mag" in plot_df.columns:
            mag = pd.to_numeric(plot_df["action_mag"], errors="coerce").to_numpy(dtype=float)
        else:
            mag = np.clip(np.abs(pd.to_numeric(plot_df["action"], errors="coerce").to_numpy(dtype=float)), 0.0, 1.0)

        x_axis = pos if has_ohlc else ts
        ax_heads.step(x_axis, dir_sign, where="post", color="#8E24AA", linewidth=1.2, label="dir_sign (-1/0/+1)")
        ax_heads.plot(x_axis, mag, color="#00897B", linewidth=1.2, label="magnitude (0..1)")
        ax_heads.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.7)
        ax_heads.set_ylim(-1.05, 1.05)
        ax_heads.set_yticks([-1.0, 0.0, 1.0])
        ax_heads.set_ylabel("Head")
        ax_heads.legend(loc="upper left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved action plot to {output_path}")


def _plot_training_history(history_df: pd.DataFrame, output_path: Path) -> None:
    if history_df.empty:
        print("Training metrics plot skipped: no history rows.")
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Training metrics plot skipped: {exc}")
        return

    x = history_df["steps"].to_numpy(dtype=float)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for col, label, color in (
        ("loss_total", "loss_total", "#1f77b4"),
        ("loss_pi", "loss_pi", "#2ca02c"),
        ("loss_v", "loss_v", "#d62728"),
    ):
        if col in history_df.columns:
            y = pd.to_numeric(history_df[col], errors="coerce").to_numpy(dtype=float)
            ax0.plot(x, y, label=label, linewidth=1.2, color=color)
    ax0.set_ylabel("Loss")
    ax0.set_title("PPO Training Losses by Update")
    ax0.legend(loc="best")

    for col, label, color in (
        ("approx_kl", "approx_kl", "#9467bd"),
        ("clipfrac", "clipfrac", "#ff7f0e"),
        ("entropy", "entropy", "#8c564b"),
    ):
        if col in history_df.columns:
            y = pd.to_numeric(history_df[col], errors="coerce").to_numpy(dtype=float)
            ax1.plot(x, y, label=label, linewidth=1.2, color=color)
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Metric")
    ax1.legend(loc="best")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved training metrics plot to {output_path}")


def _summarize_eval_report(report: pd.DataFrame) -> dict[str, float]:
    if report is None or report.empty:
        return {}
    out: dict[str, float] = {"eval_rows": float(len(report))}
    for col in ("pnl_component", "costs_component", "trades"):
        if col in report.columns:
            vals = pd.to_numeric(report[col], errors="coerce")
            out[f"{col}_mean"] = float(vals.mean())
            out[f"{col}_sum"] = float(vals.sum())
    if "pnl_component" in report.columns and "costs_component" in report.columns:
        pnl = pd.to_numeric(report["pnl_component"], errors="coerce")
        costs = pd.to_numeric(report["costs_component"], errors="coerce")
        out["pnl_net_mean"] = float((pnl - costs).mean())
        out["pnl_net_sum"] = float((pnl - costs).sum())
    return out


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


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent.")
    parser.add_argument(
        "--train-full",
        dest="train_full",
        action="store_true",
        default=True,
        help="Train on the full dataset without train/val/test splits (default).",
    )
    parser.add_argument(
        "--use-splits",
        dest="train_full",
        action="store_false",
        help="Use train/val/test splits and run holdout evaluation.",
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        choices=["exit", "mtm", "convex"],
        default="convex",
        help="Reward mode: 'exit' (realized PnL on close), 'mtm' (mark-to-market each bar), or 'convex'.",
    )
    parser.add_argument("--total-timesteps", type=int, default=2_500_000)
    parser.add_argument("--rollout-len", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--pi-lr", type=float, default=3e-4)
    parser.add_argument("--vf-lr", type=float, default=1e-3)
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--entropy-coef", type=float, default=0.0015)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-6,
        help="L2 regularization (Adam weight_decay) for PPO network weights.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.015,
        help="If >0, stop minibatch epochs for an update when approx KL exceeds this value.",
    )
    parser.add_argument(
        "--policy-hidden-size",
        type=int,
        default=256,
        help="Hidden layer width for ActorCritic MLP.",
    )
    parser.add_argument(
        "--no-policy-head-mlp",
        action="store_true",
        help="Disable separate policy/value head MLP blocks (shared trunk only).",
    )
    parser.add_argument(
        "--policy-layer-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LayerNorm in PPO MLP blocks.",
    )
    parser.add_argument(
        "--policy-dropout-p",
        type=float,
        default=0.03,
        help="Dropout probability in PPO MLP blocks (recommend <=0.05).",
    )
    parser.add_argument(
        "--feature-zscore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Z-score normalize numeric PPO input features using train-split stats.",
    )
    parser.add_argument(
        "--eval-every-updates",
        type=int,
        default=5,
        help="Run holdout eval every N PPO updates for early-stop monitoring (<=0 disables).",
    )
    parser.add_argument(
        "--eval-n-days",
        type=int,
        default=0,
        help="Days to use per early-stop eval run (<=0 uses all available eval days).",
    )
    parser.add_argument(
        "--early-stop-patience-updates",
        type=int,
        default=8,
        help="Stop training after this many eval checks without improvement (<=0 disables).",
    )
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        default="pnl_net_mean",
        choices=["pnl_net_mean", "pnl_net_sum", "pnl_mean", "pnl_sum", "costs_mean", "trades_mean"],
        help="Metric used for patience-based early stopping.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum absolute improvement required to reset patience.",
    )
    parser.add_argument(
        "--no-restore-best-on-early-stop",
        action="store_true",
        help="When early stop triggers, keep last weights instead of restoring best monitored weights.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto/cuda/cpu/mps.",
    )
    parser.add_argument("--num-envs", type=int, default=12)
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=250_000,
        help="Autosave model checkpoints every N environment steps (<=0 disables).",
    )
    parser.add_argument(
        "--checkpoint-start-steps",
        type=int,
        default=1_500_000,
        help="Do not save checkpoints before this many env steps.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="Data/outputs/agent/checkpoints",
        help="Directory for autosaved PPO checkpoints.",
    )
    parser.add_argument(
        "--policy-action-type",
        type=str,
        choices=["hybrid_dir_mag", "continuous_tanh"],
        default="hybrid_dir_mag",
        help="Policy action head: hybrid direction+magnitude (recommended) or single continuous exposure.",
    )
    parser.add_argument("--convex-k1", type=float, default=1.0)
    parser.add_argument("--convex-k2", type=float, default=0.15)
    parser.add_argument("--convex-theta", type=float, default=0.0002)
    parser.add_argument(
        "--convex-risk-lambda",
        type=float,
        default=12.0,
        help="Risk term weight for -lambda * pos^2 * vol^2 (vol proxy from ATR scale).",
    )
    parser.add_argument(
        "--convex-bonus-cap",
        type=float,
        default=0.0,
        help="Soft cap (tanh) for convex bonus contribution per bar. Set <=0 to disable.",
    )
    parser.add_argument(
        "--convex-bonus-scale",
        type=float,
        default=1.0,
        help="Global multiplier for convex bonus after capping.",
    )
    parser.add_argument(
        "--convex-pivot-k",
        type=float,
        default=0.02,
        help="Directional anchor toward pivot edge (p_pivot_long - p_pivot_short).",
    )
    parser.add_argument(
        "--convex-hold-penalty",
        type=float,
        default=0.00002,
        help="Per-bar exposure penalty used in convex mode to reduce over-holding.",
    )
    parser.add_argument(
        "--dir-switch-penalty-ret",
        type=float,
        default=0.00012,
        help="Penalty when direction flips directly long<->short.",
    )
    parser.add_argument(
        "--size-change-penalty-ret",
        type=float,
        default=0.0001,
        help="Penalty scaled by |abs(pos_t)-abs(pos_t-1)|.",
    )
    parser.add_argument(
        "--saturation-threshold",
        type=float,
        default=0.75,
        help="Start penalizing exposure magnitude when |pos| exceeds this threshold.",
    )
    parser.add_argument(
        "--saturation-penalty-ret",
        type=float,
        default=0.0002,
        help="Penalty slope on max(0, |pos|-saturation_threshold).",
    )
    parser.add_argument(
        "--commission-per-trade",
        type=float,
        default=0.0,
        help="Per-trade commission in return units via price normalization.",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.1,
        help="Slippage basis points per trade leg.",
    )
    parser.add_argument(
        "--spread-bps",
        type=float,
        default=0.1,
        help="Spread basis points per trade leg.",
    )
    parser.add_argument(
        "--trade-penalty-ret",
        type=float,
        default=0.00002,
        help="Additional per-trade penalty in return units.",
    )
    parser.add_argument(
        "--flip-penalty-ret",
        type=float,
        default=0.00005,
        help="Extra penalty when flipping direction directly.",
    )
    parser.add_argument(
        "--action-deadband",
        type=float,
        default=1e-3,
        help="Base action deadband (kept at legacy default for non-convex modes).",
    )
    parser.add_argument(
        "--convex-action-deadband",
        type=float,
        default=0.001,
        help="Action deadband used in convex mode to reduce churn.",
    )
    parser.add_argument(
        "--convex-mfe-thresholds",
        type=str,
        default="1,2,3",
        help="Comma-separated MFE ATR thresholds for bonus (e.g. '1,2,3').",
    )
    parser.add_argument(
        "--convex-mfe-bonuses",
        type=str,
        default="0.001,0.002,0.003",
        help="Comma-separated bonuses aligned to thresholds (e.g. '0.1,0.2,0.3').",
    )
    parser.add_argument(
        "--walkforward-oof",
        action="store_true",
        help="Run expanding-window walk-forward OOF training and save intent outputs for the 1m execution agent.",
    )
    parser.add_argument(
        "--walkforward-only",
        action="store_true",
        help="When --walkforward-oof is enabled, skip the single final model train/eval and only emit OOF artifacts.",
    )
    parser.add_argument(
        "--wf-n-folds",
        type=int,
        default=5,
        help="Number of walk-forward OOF folds.",
    )
    parser.add_argument(
        "--wf-initial-train-days",
        type=int,
        default=0,
        help="Initial number of day_id groups for the first OOF train window (<=0 uses auto sizing).",
    )
    parser.add_argument(
        "--wf-total-timesteps",
        type=int,
        default=0,
        help="PPO timesteps per OOF fold (<=0 reuses --total-timesteps).",
    )
    parser.add_argument(
        "--wf-trace-out",
        type=str,
        default="Data/outputs/agent/agent_trace_oof.csv",
        help="Combined walk-forward OOF trace output path.",
    )
    parser.add_argument(
        "--wf-intent-out",
        type=str,
        default="Data/outputs/agent/htf_intent_oof.parquet",
        help="Derived HTF intent output path used by 1m execution training.",
    )
    parser.add_argument(
        "--wf-manifest-out",
        type=str,
        default="Data/outputs/agent/walkforward_manifest.csv",
        help="Walk-forward fold manifest (start/end/model path) output path.",
    )
    parser.add_argument(
        "--wf-model-dir",
        type=str,
        default="Data/outputs/agent/walkforward_models",
        help="Directory to save per-fold OOF models and fold traces.",
    )
    args = parser.parse_args()
    if args.walkforward_only and not args.walkforward_oof:
        raise ValueError("--walkforward-only requires --walkforward-oof.")
    reward_on_exit = args.reward_mode == "exit"
    use_convex_reward = args.reward_mode == "convex"
    if use_convex_reward:
        reward_on_exit = False

    def _parse_floats(raw: str) -> list[float]:
        return [float(x.strip()) for x in str(raw).split(",") if x.strip()]

    convex_thresholds = _parse_floats(args.convex_mfe_thresholds)
    convex_bonuses = _parse_floats(args.convex_mfe_bonuses)
    action_deadband = (
        float(args.convex_action_deadband)
        if use_convex_reward
        else float(args.action_deadband)
    )

    cfg = PipelineConfig(drop_na=True)
    df = build_agent_training_matrix(cfg, save_parquet=True)
    vix_cols = [
        c
        for c in (
            "vix_close",
            "vix_ret_1",
            "vix_ret_4",
            "vix_ret_16",
            "vix_range_pct",
            "vix_atr_pct",
            "vix_trend_ema_8_21",
            "vix_z_20",
            "vix_vol_of_vol_20",
            "ret_1_x_vix",
            "atr_pct_x_vix",
        )
        if c in df.columns
    ]
    if vix_cols and int(df[vix_cols].notna().sum().sum()) == 0:
        print(
            "[run_train] Warning: all VIX feature values are NaN. "
            "Check VIX data availability/alignment."
        )

    drop_base = {"timestamp", "day_id", "open", "high", "low", "close", "volume"}
    feature_cols = [c for c in df.columns if c not in drop_base]
    all_nan_cols = [c for c in feature_cols if df[c].isna().all()]
    if all_nan_cols:
        print(f"Dropping all-NaN feature columns: {all_nan_cols}")
        feature_cols = [c for c in feature_cols if c not in all_nan_cols]
    print(f"Training features ({len(feature_cols)}): {feature_cols}")
    if args.train_full:
        train_df = df
        if cfg.drop_na and not train_df.empty:
            mask = train_df[feature_cols].notna().all(axis=1)
            complete_rows = int(mask.sum())
            if complete_rows == 0:
                vix_features = [c for c in feature_cols if _is_vix_feature(c)]
                non_vix_features = [c for c in feature_cols if not _is_vix_feature(c)]
                if vix_features and non_vix_features:
                    # First try dropping only severely sparse VIX features before dropping all VIX.
                    vix_nan_share = train_df[vix_features].isna().mean().sort_values(ascending=False)
                    sparse_vix = [c for c in vix_nan_share.index if float(vix_nan_share[c]) > 0.90]
                    if sparse_vix:
                        candidate_features = [c for c in feature_cols if c not in sparse_vix]
                        candidate_complete = int(train_df[candidate_features].notna().all(axis=1).sum())
                        if candidate_complete > 0:
                            print(
                                "[run_train] Warning: 0 complete rows with full VIX set. "
                                "Dropping sparse VIX features only."
                            )
                            print(
                                f"[run_train] Dropped sparse VIX features (>90% NaN): {sparse_vix}"
                            )
                            print(
                                f"[run_train] Complete rows after sparse-VIX drop: {candidate_complete:,}"
                            )
                            feature_cols = candidate_features
                            mask = train_df[feature_cols].notna().all(axis=1)
                            complete_rows = int(mask.sum())
                            print(f"Training features ({len(feature_cols)}): {feature_cols}")
                    non_vix_complete = int(train_df[non_vix_features].notna().all(axis=1).sum())
                    if complete_rows == 0 and non_vix_complete > 0:
                        print(
                            "[run_train] Warning: 0 complete rows when combining base+VIX features. "
                            "Proceeding without VIX features for this run."
                        )
                        print(
                            f"[run_train] Complete rows with non-VIX features only: {non_vix_complete:,}"
                        )
                        feature_cols = non_vix_features
                        mask = train_df[feature_cols].notna().all(axis=1)
                        complete_rows = int(mask.sum())
                        print(f"Training features ({len(feature_cols)}): {feature_cols}")
            train_df = train_df.loc[mask].copy()
            if train_df.empty:
                nan_share = (
                    df[feature_cols].isna().mean().sort_values(ascending=False).head(15)
                )
                raise ValueError(
                    "No rows remain after NaN filtering on training features. "
                    "This usually means non-overlapping feature coverage "
                    "(for example, probs and VIX from different date windows). "
                    f"Top NaN share by feature:\n{nan_share}"
                )
        val_df = df.iloc[0:0].copy()
        test_df = df.iloc[0:0].copy()
        print(f"[run_train] Training on full dataset: {len(train_df):,} rows")
    else:
        splits = load_tree_split_indices(
            ticker=cfg.ticker,
            dataset_name=cfg.dataset_name,
            x_filename=cfg.x_filename,
        )
        if cfg.drop_na:
            splits = filter_splits_for_non_nan(df, splits, feature_cols)
        train_df, val_df, test_df = split_agent_matrix(df, splits, verbose=True)
        if not val_df.empty:
            train_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
        if train_df.empty or test_df.empty:
            nan_counts = df[feature_cols].isna().sum().sort_values(ascending=False).head(10)
            raise ValueError(
                "Train/Test split is empty after NaN filtering. "
                f"Top NaN counts:\n{nan_counts}"
            )

    num_envs = max(1, int(args.num_envs))
    hold_penalty = float(args.convex_hold_penalty) if use_convex_reward else 0.0
    dir_switch_penalty = float(args.dir_switch_penalty_ret) if use_convex_reward else 0.0
    size_change_penalty = float(args.size_change_penalty_ret) if use_convex_reward else 0.0
    env_overrides = {
        "carry_positions_across_days": True,
        "reward_on_exit": reward_on_exit,
        "use_convex_reward": use_convex_reward,
        "commission_per_trade": float(args.commission_per_trade),
        "slippage_bps": float(args.slippage_bps),
        "spread_bps": float(args.spread_bps),
        "trade_penalty_ret": float(args.trade_penalty_ret),
        "flip_penalty_ret": float(args.flip_penalty_ret),
        "convex_k1": args.convex_k1,
        "convex_k2": args.convex_k2,
        "convex_theta": args.convex_theta,
        "convex_risk_lambda": args.convex_risk_lambda,
        "convex_bonus_cap": args.convex_bonus_cap,
        "convex_bonus_scale": args.convex_bonus_scale,
        "convex_pivot_k": args.convex_pivot_k,
        "dir_switch_penalty_ret": dir_switch_penalty,
        "size_change_penalty_ret": size_change_penalty,
        "saturation_threshold": args.saturation_threshold,
        "saturation_penalty_ret": args.saturation_penalty_ret,
        "hold_penalty_ret": hold_penalty,
        "action_deadband": action_deadband,
        "convex_mfe_thresholds": tuple(convex_thresholds),
        "convex_mfe_bonuses": tuple(convex_bonuses),
    }
    if use_convex_reward:
        print(
            "[run_train] Magnitude controls:",
            f"k2={env_overrides['convex_k2']}",
            f"theta={env_overrides['convex_theta']}",
            f"risk_lambda={env_overrides['convex_risk_lambda']}",
            f"bonus_cap={env_overrides['convex_bonus_cap']}",
            f"hold_penalty={env_overrides['hold_penalty_ret']}",
            f"switch_penalty={env_overrides['dir_switch_penalty_ret']}",
            f"size_penalty={env_overrides['size_change_penalty_ret']}",
            f"sat_threshold={env_overrides['saturation_threshold']}",
            f"sat_penalty={env_overrides['saturation_penalty_ret']}",
        )
    env_kwargs = dict(env_overrides)
    output_dir = Path("Data") / "outputs" / "agent"
    output_dir.mkdir(parents=True, exist_ok=True)
    walkforward_artifacts: dict[str, Path] = {}

    if args.walkforward_oof:
        walkforward_artifacts = _run_walkforward_oof(
            source_df=train_df,
            feature_cols=feature_cols,
            env_kwargs=env_kwargs,
            cfg=cfg,
            args=args,
            action_deadband=action_deadband,
        )
        if args.walkforward_only:
            log_paths = log_training_run(
                run_name="agent_run_train_walkforward_only",
                output_dir=output_dir,
                hyperparameters=vars(args),
                train_metrics={"walkforward_only": True},
                validation_metrics={},
                best_validation_metrics={},
                artifacts={
                    **{k: str(v) for k, v in walkforward_artifacts.items()},
                },
                extra={
                    "ticker": normalize_ticker(cfg.ticker),
                    "dataset_name": cfg.dataset_name,
                    "feature_count": len(feature_cols),
                    "train_rows": int(len(train_df)),
                },
            )
            print("[run_train] Completed walk-forward OOF only (--walkforward-only).")
            print(f"[run_train] Saved training run summary: {log_paths['latest_path']}")
            return

    feature_norm: dict[str, object] = {}
    if bool(args.feature_zscore):
        feature_norm = _fit_feature_zscore_stats(train_df, feature_cols)
        train_df = _apply_feature_zscore(train_df, feature_norm)
        if not val_df.empty:
            val_df = _apply_feature_zscore(val_df, feature_norm)
        if not test_df.empty:
            test_df = _apply_feature_zscore(test_df, feature_norm)
        print(
            "[run_train] Z-score normalization:",
            f"scaled_cols={len(feature_norm.get('cols', []))}",
            f"skipped_cols={len(feature_norm.get('skipped_cols', []))}",
        )

    train_env = _build_train_env(
        df=train_df,
        feature_cols=feature_cols,
        env_kwargs=env_kwargs,
        num_envs=num_envs,
        seed_base=int(args.seed),
    )
    eval_env_for_es = None
    if int(args.eval_every_updates) > 0:
        eval_source_df = test_df if not test_df.empty else val_df
        if eval_source_df.empty:
            print(
                "[run_train] Early-stop eval requested but no holdout rows are available; disabling eval monitor."
            )
        else:
            eval_env_for_es = make_trading_env(
                df=eval_source_df,
                feature_cols=feature_cols,
                seed=int(args.seed) + 10_000,
                **env_kwargs,
            )
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (Path.cwd() / checkpoint_dir).resolve()
    checkpoint_payload = {
        "obs_dim": train_env.obs_dim,
        "n_actions": 3,
        "action_type": str(args.policy_action_type),
        "action_low": -1.0,
        "action_high": 1.0,
        "action_deadband": float(action_deadband),
        "feature_cols": feature_cols,
        "config": cfg.__dict__,
        "reward_mode": str(args.reward_mode),
        "env_overrides": env_overrides,
        "feature_norm": feature_norm,
    }

    model, train_history = train_ppo(
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
        action_type=str(args.policy_action_type),
        device=str(args.device),
        seed=int(args.seed),
        verbose=True,
        return_history=True,
        checkpoint_every_steps=int(args.checkpoint_every_steps),
        checkpoint_start_steps=int(args.checkpoint_start_steps),
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_prefix="ppo_model",
        checkpoint_payload=checkpoint_payload,
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
    history_df = pd.DataFrame(train_history)
    history_csv_path = output_dir / "ppo_train_metrics.csv"
    history_df.to_csv(history_csv_path, index=False)
    print(f"Saved training metrics to {history_csv_path}")
    history_plot_path = output_dir / "ppo_train_metrics.png"
    _plot_training_history(history_df, history_plot_path)

    model_path = output_dir / "ppo_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": train_env.obs_dim,
            "n_actions": 3,
            "action_type": str(args.policy_action_type),
            "action_dim": int(getattr(model, "action_dim", 1)),
            "action_low": -1.0,
            "action_high": 1.0,
            "policy_hidden_size": int(args.policy_hidden_size),
            "policy_head_mlp": not bool(args.no_policy_head_mlp),
            "policy_layer_norm": bool(args.policy_layer_norm),
            "policy_dropout_p": float(args.policy_dropout_p),
            "action_deadband": float(action_deadband),
            "feature_cols": feature_cols,
            "config": cfg.__dict__,
            "reward_mode": str(args.reward_mode),
            "env_overrides": env_overrides,
            "feature_norm": feature_norm,
        },
        model_path,
    )
    print(f"Saved model to {model_path}")
    final_train_metrics = (
        history_df.iloc[-1].to_dict() if not history_df.empty else {}
    )
    best_validation_metrics = _extract_best_eval_metrics(
        history_df,
        metric_name=str(args.early_stop_metric),
    )
    base_artifacts = {
        "model_path": str(model_path),
        "train_metrics_csv": str(history_csv_path),
        "train_metrics_plot": str(history_plot_path),
        **{k: str(v) for k, v in walkforward_artifacts.items()},
    }
    base_extra = {
        "ticker": normalize_ticker(cfg.ticker),
        "dataset_name": cfg.dataset_name,
        "feature_count": len(feature_cols),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "reward_mode": str(args.reward_mode),
    }

    if args.train_full:
        print("[run_train] Skipping evaluation because --train-full is set.")
        log_paths = log_training_run(
            run_name="agent_run_train",
            output_dir=output_dir,
            hyperparameters=vars(args),
            train_metrics=final_train_metrics,
            validation_metrics={"skipped": "--train-full"},
            best_validation_metrics=best_validation_metrics,
            artifacts=base_artifacts,
            extra=base_extra,
        )
        print(f"[run_train] Saved training run summary: {log_paths['latest_path']}")
        return

    test_env = make_trading_env(
        df=test_df,
        feature_cols=feature_cols,
        **env_kwargs,
    )

    eval_device = "cuda" if torch.cuda.is_available() else "cpu"
    report = evaluate_policy(
        test_env,
        model,
        n_days=len(test_env.day_starts),
        device=eval_device,
        deterministic=True,
    )
    print(report)
    print("Avg pnl component:", report["pnl_component"].mean(), "Avg costs:", report["costs_component"].mean())
    eval_loss_env = make_trading_env(
        df=test_df,
        feature_cols=feature_cols,
        **env_kwargs,
    )
    eval_loss = evaluate_loss_metrics(
        eval_loss_env,
        model,
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        device=eval_device,
        deterministic=True,
    )
    print(
        "Eval loss metrics:",
        f"actor={eval_loss.get('eval_loss_actor', float('nan')):.6f}",
        f"value={eval_loss.get('eval_loss_value', float('nan')):.6f}",
        f"entropy={eval_loss.get('eval_entropy', float('nan')):.6f}",
        f"avg_reward={eval_loss.get('eval_avg_reward', float('nan')):.6f}",
        f"avg_abs_reward={eval_loss.get('eval_avg_abs_reward', float('nan')):.6f}",
    )

    initial_cash = 100_000.0
    baseline_mode = "intraday"  # "intraday", "buy_hold", or "none"
    include_dca = False

    # Use a fresh env for trace generation so carry/state from the summary pass
    # cannot leak into plotted/recorded behavior.
    trace_env = make_trading_env(
        df=test_df,
        feature_cols=feature_cols,
        **env_kwargs,
    )
    trace = evaluate_policy_with_trace(trace_env, model, device=eval_device, deterministic=True)
    trace = _agent_equity_from_trace(trace, initial_cash=initial_cash)
    trace_path = output_dir / "agent_trace.csv"
    trace.to_csv(trace_path, index=False)
    print(f"Saved trace to {trace_path}")

    daily_summary = _daily_first_last(test_df)
    trade_cost_ret = test_env._trade_cost_ret
    agent_final = float(trace["equity"].iloc[-1]) if not trace.empty else initial_cash
    agent_pnl = agent_final - initial_cash
    agent_return = agent_final / initial_cash - 1.0

    print(f"Agent total return: {agent_return:.2%} (equity: {agent_final:,.2f}, PnL: {agent_pnl:,.2f})")

    if baseline_mode == "intraday":
        base_final, base_pnl, base_ret = _intraday_long(daily_summary, initial_cash, trade_cost_ret)
        print(f"Intraday Long final equity: {base_final:,.2f} (PnL: {base_pnl:,.2f}, Return: {base_ret:.2%})")
    elif baseline_mode == "buy_hold":
        if test_env.force_flat_at_close:
            print("Warning: Buy & Hold baseline holds overnight while agent is forced flat at close.")
        base_final, base_pnl, base_ret = _buy_and_hold(daily_summary, initial_cash, trade_cost_ret)
        print(f"Buy & Hold final equity: {base_final:,.2f} (PnL: {base_pnl:,.2f}, Return: {base_ret:.2%})")
    elif baseline_mode != "none":
        raise ValueError(f"Unknown baseline_mode '{baseline_mode}'. Use 'intraday', 'buy_hold', or 'none'.")

    if include_dca:
        dca_final, dca_pnl, tranche = _dca_down_days(daily_summary, initial_cash)
        print(f"DCA down-days final equity: {dca_final:,.2f} (PnL: {dca_pnl:,.2f})")
        print(f"DCA tranche size per down day: {tranche:,.2f}")

    ticker_slug = normalize_ticker(cfg.ticker).lower()
    output_path = (
        Path("Data")
        / "models"
        / "agent"
        / cfg.dataset_name
        / ticker_slug
        / "agent_actions_vs_price.png"
    )
    _plot_actions(trace, output_path)
    validation_metrics = {
        **_summarize_eval_report(report),
        **{f"loss_{k}": v for k, v in eval_loss.items()},
        "agent_total_return": float(agent_return),
        "agent_final_equity": float(agent_final),
    }
    if baseline_mode == "intraday":
        validation_metrics["baseline_intraday_return"] = float(base_ret)
        validation_metrics["baseline_intraday_equity"] = float(base_final)
    if baseline_mode == "buy_hold":
        validation_metrics["baseline_buy_hold_return"] = float(base_ret)
        validation_metrics["baseline_buy_hold_equity"] = float(base_final)
    log_paths = log_training_run(
        run_name="agent_run_train",
        output_dir=output_dir,
        hyperparameters=vars(args),
        train_metrics=final_train_metrics,
        validation_metrics=validation_metrics,
        best_validation_metrics=best_validation_metrics,
        artifacts={
            **base_artifacts,
            "trace_csv": str(trace_path),
            "action_plot": str(output_path),
        },
        extra=base_extra,
    )
    print(f"[run_train] Saved training run summary: {log_paths['latest_path']}")


if __name__ == "__main__":
    main()
