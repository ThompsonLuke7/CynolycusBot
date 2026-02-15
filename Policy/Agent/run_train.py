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
    parser.add_argument("--entropy-coef", type=float, default=0.004)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
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
    parser.add_argument("--convex-theta", type=float, default=0.00005)
    parser.add_argument(
        "--convex-risk-lambda",
        type=float,
        default=0.0,
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
        default=0.01,
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
        default=0.00003,
        help="Penalty scaled by |abs(pos_t)-abs(pos_t-1)|.",
    )
    parser.add_argument(
        "--saturation-threshold",
        type=float,
        default=0.90,
        help="Start penalizing exposure magnitude when |pos| exceeds this threshold.",
    )
    parser.add_argument(
        "--saturation-penalty-ret",
        type=float,
        default=0.0,
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
    args = parser.parse_args()
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
                    non_vix_complete = int(train_df[non_vix_features].notna().all(axis=1).sum())
                    if non_vix_complete > 0:
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
    env_kwargs = dict(env_overrides)

    if num_envs > 1:
        train_envs = [
            make_trading_env(
                df=train_df,
                feature_cols=feature_cols,
                seed=7 + i,
                **env_kwargs,
            )
            for i in range(num_envs)
        ]
        train_env = VecTradingEnv(train_envs, auto_reset=True, stagger_reset=True)
    else:
        train_env = make_trading_env(
            df=train_df,
            feature_cols=feature_cols,
            **env_kwargs,
        )

    output_dir = Path("Data") / "outputs" / "agent"
    output_dir.mkdir(parents=True, exist_ok=True)
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
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_prefix="ppo_model",
        checkpoint_payload=checkpoint_payload,
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
            "action_deadband": float(action_deadband),
            "feature_cols": feature_cols,
            "config": cfg.__dict__,
            "reward_mode": str(args.reward_mode),
            "env_overrides": env_overrides,
        },
        model_path,
    )
    print(f"Saved model to {model_path}")

    if args.train_full:
        print("[run_train] Skipping evaluation because --train-full is set.")
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


if __name__ == "__main__":
    main()
