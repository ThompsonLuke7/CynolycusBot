from __future__ import annotations

import argparse
from pathlib import Path
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
    load_tree_split_indices,
    split_agent_matrix,
)
from Data.retrieve_data import normalize_ticker
from Agent.env_config import make_trading_env
from Agent.eval import evaluate_loss_metrics, evaluate_policy, evaluate_policy_with_trace
from Agent.model import ActorCritic


def _normalize_env_overrides(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, object] = {}
    allowed = {
        "add_time_features",
        "add_position_features",
        "commission_per_trade",
        "slippage_bps",
        "spread_bps",
        "flip_penalty_ret",
        "trade_penalty_ret",
        "hold_penalty_ret",
        "reward_on_exit",
        "reward_exit_bonus",
        "exit_pivot_bonus_ret",
        "force_flat_at_close",
        "carry_positions_across_days",
        "allow_direct_flip",
        "use_convex_reward",
        "convex_k1",
        "convex_k2",
        "convex_theta",
        "convex_pivot_k",
        "convex_mfe_thresholds",
        "convex_mfe_bonuses",
        "action_deadband",
        "seed",
    }
    for key in allowed:
        if key in raw:
            out[key] = raw[key]
    return out


def _is_continuous_action_type(action_type: str) -> bool:
    val = str(action_type or "").strip().lower()
    return val in {"continuous", "continuous_tanh", "gaussian_tanh"}


def _load_eval_frame(path_like: str | None) -> pd.DataFrame | None:
    if not path_like:
        return None
    data_path = Path(path_like)
    if not data_path.exists():
        raise SystemExit(f"Missing eval data file: {data_path}")
    suffix = data_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(data_path)
    elif suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise SystemExit(
            f"Unsupported eval data extension '{suffix}'. Use .csv or .parquet."
        )
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "day_id" not in df.columns and "timestamp" in df.columns:
        df["day_id"] = pd.Series(df["timestamp"].dt.normalize()).factorize()[0]
    return df


def _agent_equity_from_trace(trace: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    equity = float(initial_cash)
    equity_series = []
    for _, row in trace.iterrows():
        costs = float(row.get("reward_costs", 0.0)) + float(row.get("forced_flat_cost", 0.0))
        net_ret = float(row.get("reward_pnl", 0.0)) - costs
        equity *= (1.0 + net_ret)
        equity_series.append(equity)
    trace = trace.copy()
    trace["equity"] = equity_series
    return trace


def _daily_first_last(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values("timestamp")
    grouped = ordered.groupby("day_id", sort=True)
    daily = grouped.agg(
        first_ts=("timestamp", "first"),
        last_ts=("timestamp", "last"),
        first_close=("close", "first"),
        last_close=("close", "last"),
    )
    return daily.reset_index(drop=True)


def _buy_and_hold(daily_summary: pd.DataFrame, initial_cash: float, trade_cost_ret) -> tuple[float, float, float]:
    start = float(daily_summary["first_close"].iloc[0])
    end = float(daily_summary["last_close"].iloc[-1])
    total_ret = (end / start - 1.0) - trade_cost_ret(start) - trade_cost_ret(end)
    final_value = initial_cash * (1.0 + total_ret)
    return final_value, final_value - initial_cash, total_ret


def _intraday_long(daily_summary: pd.DataFrame, initial_cash: float, trade_cost_ret) -> tuple[float, float, float]:
    equity = float(initial_cash)
    for _, row in daily_summary.iterrows():
        start = float(row["first_close"])
        end = float(row["last_close"])
        day_ret = (end / start - 1.0) - trade_cost_ret(start) - trade_cost_ret(end)
        equity *= (1.0 + day_ret)
    total_ret = equity / initial_cash - 1.0
    return equity, equity - initial_cash, total_ret


def _plot_actions(
    trace: pd.DataFrame,
    output_path: Path,
    *,
    tail: int = 100,
    random_window: bool = False,
    seed: int | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot skipped: {exc}")
        return

    plot_df = trace.dropna(subset=["timestamp", "close"])
    if plot_df.empty:
        print("Plot skipped: no data.")
        return

    if tail < 1:
        tail = 1
    if random_window and len(plot_df) > tail:
        rng = np.random.default_rng(seed)
        start = int(rng.integers(0, len(plot_df) - tail + 1))
        plot_df = plot_df.iloc[start : start + tail]
    else:
        plot_df = plot_df.tail(tail)
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

    eps = 1e-3
    prev_pos = plot_df["prev_pos"].to_numpy(dtype=float) if "prev_pos" in plot_df.columns else None
    pos_now = plot_df["position"].to_numpy(dtype=float) if "position" in plot_df.columns else None
    if prev_pos is None or pos_now is None:
        actions = plot_df["action"].to_numpy(dtype=float)
        prev_actions = np.roll(actions, 1)
        prev_actions[0] = 0.0
        if np.nanmax(np.abs(actions)) > 1.5:
            entry_long = (actions == 1.0) & (prev_actions != 1.0)
            entry_short = (actions == 2.0) & (prev_actions != 2.0)
            exit_long = (prev_actions == 1.0) & (actions != 1.0)
            exit_short = (prev_actions == 2.0) & (actions != 2.0)
        else:
            entry_long = (actions > eps) & (prev_actions <= eps)
            entry_short = (actions < -eps) & (prev_actions >= -eps)
            exit_long = (prev_actions > eps) & (actions <= eps)
            exit_short = (prev_actions < -eps) & (actions >= -eps)
    else:
        # Transition-based markers handle direct flips correctly.
        entry_long = (pos_now > eps) & (prev_pos <= eps)
        entry_short = (pos_now < -eps) & (prev_pos >= -eps)
        exit_long = (prev_pos > eps) & (pos_now <= eps)
        exit_short = (prev_pos < -eps) & (pos_now >= -eps)

    prob_cols = []
    if "p_pivot_long" in plot_df.columns:
        prob_cols.append(("p_pivot_long", "#1565C0", "p_pivot_long"))
    if "p_pivot_short" in plot_df.columns:
        prob_cols.append(("p_pivot_short", "#EF6C00", "p_pivot_short"))
    if "p_tb_long" in plot_df.columns:
        prob_cols.append(("p_tb_long", "#2E7D32", "p_tb_long"))
    if "p_tb_short" in plot_df.columns:
        prob_cols.append(("p_tb_short", "#C62828", "p_tb_short"))
    has_probs = bool(prob_cols)
    if has_probs:
        prob_vals = plot_df[[c[0] for c in prob_cols]].to_numpy(dtype=float)
        has_probs = np.isfinite(prob_vals).any()

    if has_probs:
        fig, (ax_price, ax_prob) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1]},
        )
    else:
        fig, ax_price = plt.subplots(figsize=(12, 6))
        ax_prob = None

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
        ax_price.vlines(pos[valid], low_y[valid], high_y[valid], color=wick_color, linewidth=1.0, zorder=1)
        ax_price.bar(
            pos[valid & up],
            close_y[valid & up] - open_y[valid & up],
            width=0.8,
            bottom=open_y[valid & up],
            color=up_color,
            edgecolor="none",
            zorder=1.2,
        )
        ax_price.bar(
            pos[valid & ~up],
            close_y[valid & ~up] - open_y[valid & ~up],
            width=0.8,
            bottom=open_y[valid & ~up],
            color=down_color,
            edgecolor="none",
            zorder=1.2,
        )
        ax_price.scatter(pos[entry_long], close_y[entry_long], c="green", s=26, label="Long entry", marker="^", zorder=2.2)
        ax_price.scatter(pos[entry_short], close_y[entry_short], c="red", s=26, label="Short entry", marker="v", zorder=2.2)
        ax_price.scatter(pos[exit_long], close_y[exit_long], c="#1565C0", s=26, label="Long exit", marker="x", zorder=2.2)
        ax_price.scatter(pos[exit_short], close_y[exit_short], c="#EF6C00", s=26, label="Short exit", marker="x", zorder=2.2)
        day_start = ts.dt.normalize().ne(ts.dt.normalize().shift())
        tick_positions = pos[day_start.to_numpy()]
        tick_labels = ts[day_start].dt.strftime("%Y-%m-%d").to_list()
        if len(tick_positions) > 25:
            step = int(np.ceil(len(tick_positions) / 25))
            tick_positions = tick_positions[::step]
            tick_labels = tick_labels[::step]
        if ax_prob is not None:
            ax_price.tick_params(labelbottom=False)
            ax_prob.set_xticks(tick_positions)
            ax_prob.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
            ax_prob.set_xlabel("Date")
        else:
            ax_price.set_xticks(tick_positions)
            ax_price.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
            ax_price.set_xlabel("Date")
    else:
        ax_price.plot(ts, close, label="SPY close", linewidth=1.5)
        ax_price.scatter(ts[entry_long], close[entry_long], c="green", s=20, label="Long entry", marker="^")
        ax_price.scatter(ts[entry_short], close[entry_short], c="red", s=20, label="Short entry", marker="v")
        ax_price.scatter(ts[exit_long], close[exit_long], c="#1565C0", s=20, label="Long exit", marker="x")
        ax_price.scatter(ts[exit_short], close[exit_short], c="#EF6C00", s=20, label="Short exit", marker="x")
        if ax_prob is not None:
            ax_prob.set_xlabel("Time")
        else:
            ax_price.set_xlabel("Time")

    ax_price.set_title("SPY Candles with Agent Actions (Test Set)")
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left")

    if ax_prob is not None and has_probs:
        for col, color, label in prob_cols:
            if col in plot_df.columns:
                ax_prob.plot(
                    pos if has_ohlc else ts,
                    plot_df[col].to_numpy(dtype=float),
                    color=color,
                    linewidth=1.3,
                    label=label,
                )
        ax_prob.set_ylim(0, 1.02)
        ax_prob.set_ylabel("Prob")
        ax_prob.legend(loc="upper left")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved action plot to {output_path}")


def _trade_stats_from_trace(trace: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-3
    trades = []
    in_trade = False
    entry_pos = 0.0
    entry_price = 0.0
    entry_time = None
    acc_costs = 0.0
    bars = 0

    for _, row in trace.iterrows():
        pos = float(row.get("position", 0.0))
        prev_pos = float(row.get("prev_pos", 0.0))
        price = float(row.get("close", 0.0))
        ts = row.get("timestamp")
        cost = float(row.get("reward_costs", 0.0)) + float(row.get("forced_flat_cost", 0.0))

        if not in_trade:
            if abs(prev_pos) <= eps and abs(pos) > eps:
                in_trade = True
                entry_pos = pos
                entry_price = price
                entry_time = ts
                acc_costs = cost
                bars = 1
            continue

        # Flip: close old trade and open new on same bar.
        if abs(prev_pos) > eps and abs(pos) > eps and (prev_pos * pos < 0.0):
            exit_cost = cost * 0.5
            acc_costs += exit_cost
            trade_ret = entry_pos * (price / entry_price - 1.0)
            net_ret = trade_ret - acc_costs
            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "side": float(np.sign(entry_pos)),
                    "entry_exposure": float(entry_pos),
                    "bars": bars,
                    "gross_ret": trade_ret,
                    "net_ret": net_ret,
                    "costs": acc_costs,
                }
            )
            entry_pos = pos
            entry_price = price
            entry_time = ts
            acc_costs = cost - exit_cost
            bars = 1
            continue

        acc_costs += cost
        if abs(pos) > eps:
            bars += 1

        if abs(pos) <= eps and abs(prev_pos) > eps:
            trade_ret = entry_pos * (price / entry_price - 1.0)
            net_ret = trade_ret - acc_costs
            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "side": float(np.sign(entry_pos)),
                    "entry_exposure": float(entry_pos),
                    "bars": bars,
                    "gross_ret": trade_ret,
                    "net_ret": net_ret,
                    "costs": acc_costs,
                }
            )
            in_trade = False

    return pd.DataFrame(trades)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent and plot actions.")
    parser.add_argument("--model-path", default="Data/outputs/agent/ppo_model.pt")
    parser.add_argument("--baseline", choices=["intraday", "buy_hold", "none"], default="intraday")
    parser.add_argument("--plot-tail", type=int, default=100)
    parser.add_argument("--plot-random-window", action="store_true")
    parser.add_argument("--plot-seed", type=int, default=None)
    parser.add_argument("--trace-out", default="Data/outputs/agent/agent_trace.csv")
    parser.add_argument("--trace-in", default="Data/outputs/agent/agent_trace.csv")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--data-csv",
        default=None,
        help="Optional CSV path for evaluation data (uses full file as test set).",
    )
    parser.add_argument(
        "--data-parquet",
        default=None,
        help="Optional parquet path for evaluation data (uses full file as test set).",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional evaluation data path (.csv or .parquet). Overrides --data-csv/--data-parquet.",
    )
    parser.add_argument("--plot-out", default=None)
    parser.add_argument("--device", default=None, help="cuda/cpu (defaults to auto)")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of deterministic mean action.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = PipelineConfig(drop_na=True)
    explicit_data_path = args.data_path or args.data_parquet or args.data_csv
    loaded_df = _load_eval_frame(explicit_data_path)
    if loaded_df is not None:
        df = loaded_df
        test_df = loaded_df
    else:
        df = build_agent_training_matrix(cfg, save_parquet=False)
        splits = load_tree_split_indices(
            ticker=cfg.ticker,
            dataset_name=cfg.dataset_name,
            x_filename=cfg.x_filename,
        )
        _train_df, _val_df, test_df = split_agent_matrix(df, splits, verbose=True)

    deterministic = not args.stochastic
    if args.plot_only:
        trace_path = Path(args.trace_in)
        if not trace_path.exists():
            raise SystemExit(f"Missing trace file: {trace_path}")
        trace = pd.read_csv(trace_path)
    else:
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise SystemExit(f"Missing model file: {model_path}")

        ckpt = torch.load(model_path, map_location="cpu")
        drop_base = {"timestamp", "day_id", "open", "high", "low", "close", "volume"}
        ckpt_feature_cols = ckpt.get("feature_cols")
        if ckpt_feature_cols:
            feature_cols = list(ckpt_feature_cols)
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    "Missing required feature columns from data: "
                    + ", ".join(missing)
                )
        else:
            feature_cols = [c for c in df.columns if c not in drop_base]
        if cfg.drop_na and not test_df.empty:
            test_df = test_df.dropna(subset=feature_cols)

        state_dict = ckpt["state_dict"]
        has_head_mlps = any(
            k.startswith("policy_mlp.") or k.startswith("value_mlp.")
            for k in state_dict
        )
        action_type = str(ckpt.get("action_type", "discrete"))
        reward_mode = str(ckpt.get("reward_mode", "unknown"))
        env_overrides = _normalize_env_overrides(ckpt.get("env_overrides"))
        if not env_overrides and reward_mode in {"exit", "mtm", "convex"}:
            env_overrides = {
                "reward_on_exit": reward_mode == "exit",
                "use_convex_reward": reward_mode == "convex",
            }
        model = ActorCritic(
            obs_dim=ckpt["obs_dim"],
            n_actions=ckpt.get("n_actions", 3),
            action_type=action_type,
            action_dim=int(ckpt.get("action_dim", 1)),
            head_mlp=has_head_mlps,
        )
        model.load_state_dict(state_dict)
        model.eval()

        test_env = make_trading_env(
            df=test_df,
            feature_cols=feature_cols,
            **env_overrides,
        )
        print(
            "[run_eval] Eval environment:",
            f"reward_mode={reward_mode}",
            f"use_convex={bool(getattr(test_env, 'use_convex_reward', False))}",
            f"continuous_action={_is_continuous_action_type(action_type)}",
        )

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        report = evaluate_policy(
            test_env,
            model,
            n_days=len(test_env.day_starts),
            device=device,
            deterministic=deterministic,
        )
        print(report)
        print("Avg pnl component:", report["pnl_component"].mean(), "Avg costs:", report["costs_component"].mean())
        eval_loss_env = make_trading_env(
            df=test_df,
            feature_cols=feature_cols,
            **env_overrides,
        )
        eval_loss = evaluate_loss_metrics(
            eval_loss_env,
            model,
            device=device,
            deterministic=deterministic,
        )
        print(
            "Eval loss metrics:",
            f"actor={eval_loss.get('eval_loss_actor', float('nan')):.6f}",
            f"value={eval_loss.get('eval_loss_value', float('nan')):.6f}",
            f"entropy={eval_loss.get('eval_entropy', float('nan')):.6f}",
            f"avg_reward={eval_loss.get('eval_avg_reward', float('nan')):.6f}",
            f"avg_abs_reward={eval_loss.get('eval_avg_abs_reward', float('nan')):.6f}",
        )

        # Use a fresh env for trace generation so carry/state from the summary pass
        # cannot leak into plotted/recorded behavior.
        trace_env = make_trading_env(
            df=test_df,
            feature_cols=feature_cols,
            **env_overrides,
        )
        trace = evaluate_policy_with_trace(trace_env, model, device=device, deterministic=deterministic)
        trace = _agent_equity_from_trace(trace, initial_cash=100_000.0)
        trace_path = Path(args.trace_out)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace.to_csv(trace_path, index=False)
        print(f"Saved trace to {trace_path}")

    if "timestamp" in trace.columns:
        trace["timestamp"] = pd.to_datetime(trace["timestamp"], errors="coerce")
    if ("p_tb_long" not in trace.columns or "p_tb_short" not in trace.columns) and (
        "p_tb_long" in test_df.columns or "p_tb_short" in test_df.columns
    ):
        tb_cols = [c for c in ("p_tb_long", "p_tb_short") if c in test_df.columns]
        tb_source = test_df[["timestamp", *tb_cols]].copy()
        tb_source["timestamp"] = pd.to_datetime(tb_source["timestamp"], errors="coerce")
        trace = trace.merge(tb_source, on="timestamp", how="left", suffixes=("", "_tb"))
        for col in tb_cols:
            merge_col = f"{col}_tb"
            if merge_col in trace.columns:
                trace[col] = trace[col].fillna(trace[merge_col])
                trace = trace.drop(columns=[merge_col])

    use_convex_reward = bool(getattr(test_env, "use_convex_reward", False)) if not args.plot_only else False
    continuous_action = _is_continuous_action_type(action_type) if not args.plot_only else False

    if not args.plot_only and continuous_action and "position" in trace.columns:
        pos_series = pd.to_numeric(trace["position"], errors="coerce").fillna(0.0)
        dpos = pos_series.diff().abs().fillna(0.0)
        eps = 1e-3
        print(
            "Exposure stats:",
            f"mean_abs={float(pos_series.abs().mean()):.4f}",
            f"p95_abs={float(pos_series.abs().quantile(0.95)):.4f}",
            f"flat_rate={(pos_series.abs() <= eps).mean():.2%}",
            f"mean_turnover={float(dpos.mean()):.4f}",
            f"p95_turnover={float(dpos.quantile(0.95)):.4f}",
        )

    if not args.plot_only and use_convex_reward:
        convex = (
            pd.to_numeric(trace["reward_convex"], errors="coerce")
            if "reward_convex" in trace.columns
            else pd.Series(dtype=float)
        )
        term = (
            pd.to_numeric(trace["convex_term"], errors="coerce")
            if "convex_term" in trace.columns
            else pd.Series(dtype=float)
        )
        mfe_atr = (
            pd.to_numeric(trace["mfe_atr"], errors="coerce")
            if "mfe_atr" in trace.columns
            else pd.Series(dtype=float)
        )
        mfe_bonus = (
            pd.to_numeric(trace["mfe_bonus"], errors="coerce")
            if "mfe_bonus" in trace.columns
            else pd.Series(dtype=float)
        )
        pivot_anchor = (
            pd.to_numeric(trace["reward_pivot_anchor"], errors="coerce")
            if "reward_pivot_anchor" in trace.columns
            else pd.Series(dtype=float)
        )
        reward_total = (
            pd.to_numeric(trace["reward"], errors="coerce")
            if "reward" in trace.columns
            else pd.Series(dtype=float)
        )
        reward_pnl = (
            pd.to_numeric(trace["reward_pnl"], errors="coerce")
            if "reward_pnl" in trace.columns
            else pd.Series(dtype=float)
        )
        reward_costs = (
            pd.to_numeric(trace["reward_costs"], errors="coerce")
            if "reward_costs" in trace.columns
            else pd.Series(dtype=float)
        )
        bonus_hits = int((mfe_bonus.fillna(0.0) > 0).sum()) if not mfe_bonus.empty else 0
        pos = (
            pd.to_numeric(trace["position"], errors="coerce").fillna(0.0)
            if "position" in trace.columns
            else pd.Series(dtype=float)
        )
        up_capture = 0.0
        down_capture = 0.0
        if "ret_next" in trace.columns and not pos.empty:
            retn = pd.to_numeric(trace["ret_next"], errors="coerce").fillna(0.0)
            pnl_dir = (pos * retn).dropna()
            if not pnl_dir.empty:
                up_capture = float(pnl_dir[pnl_dir > 0].mean()) if (pnl_dir > 0).any() else 0.0
                down_capture = float(pnl_dir[pnl_dir < 0].mean()) if (pnl_dir < 0).any() else 0.0
        print(
            "Convex stats:",
            f"mean_reward={float(reward_total.dropna().mean()) if reward_total.notna().any() else 0.0:.6f}",
            f"mean_reward_pnl={float(reward_pnl.dropna().mean()) if reward_pnl.notna().any() else 0.0:.6f}",
            f"mean_costs={float(reward_costs.dropna().mean()) if reward_costs.notna().any() else 0.0:.6f}",
            f"mean_reward_convex={float(convex.dropna().mean()) if convex.notna().any() else 0.0:.6f}",
            f"mean_convex_term={float(term.dropna().mean()) if term.notna().any() else 0.0:.6f}",
            f"mean_pivot_anchor={float(pivot_anchor.dropna().mean()) if pivot_anchor.notna().any() else 0.0:.6f}",
            f"mean_mfe_atr={float(mfe_atr.dropna().mean()) if mfe_atr.notna().any() else 0.0:.4f}",
            f"total_mfe_bonus={float(mfe_bonus.dropna().sum()) if mfe_bonus.notna().any() else 0.0:.6f}",
            f"mfe_bonus_hits={bonus_hits}",
            f"mean_up_capture={up_capture:.6f}",
            f"mean_down_capture={down_capture:.6f}",
        )

    if not args.plot_only and (not use_convex_reward) and "prev_pos" in trace.columns:
        prev = pd.to_numeric(trace["prev_pos"], errors="coerce").fillna(0.0)
        prev_bucket = np.where(prev > 1e-3, 1, np.where(prev < -1e-3, -1, 0))
        by_prev = trace.assign(prev_side=prev_bucket).groupby("prev_side")["reward_pnl"].mean()
        print("Mean reward_pnl by prev_pos:")
        print(by_prev)

    if not args.plot_only:
        trades = _trade_stats_from_trace(trace)
        if not trades.empty:
            win_mask = trades["net_ret"] > 0
            win_rate = float(win_mask.mean())
            avg_win = float(trades.loc[win_mask, "net_ret"].mean()) if win_mask.any() else 0.0
            avg_loss = float(trades.loc[~win_mask, "net_ret"].mean()) if (~win_mask).any() else 0.0
            median_ret = float(trades["net_ret"].median())
            trades_per_day = float(len(trades) / max(1, trace["day_ptr"].nunique()))
            print(
                "Trade stats:",
                f"count={len(trades)}",
                f"win_rate={win_rate:.2%}",
                f"median_ret={median_ret:.4%}",
                f"avg_win={avg_win:.4%}",
                f"avg_loss={avg_loss:.4%}",
                f"trades_per_day={trades_per_day:.2f}",
            )

    if not args.plot_only:
        daily_summary = _daily_first_last(test_df)
        trade_cost_ret = test_env._trade_cost_ret
        agent_final = float(trace["equity"].iloc[-1]) if not trace.empty else 100_000.0
        agent_return = agent_final / 100_000.0 - 1.0
        print(f"Agent total return: {agent_return:.2%} (equity: {agent_final:,.2f})")

        if args.baseline == "intraday":
            base_final, _base_pnl, base_ret = _intraday_long(daily_summary, 100_000.0, trade_cost_ret)
            print(f"Intraday Long final equity: {base_final:,.2f} (Return: {base_ret:.2%})")
        elif args.baseline == "buy_hold":
            if test_env.force_flat_at_close:
                print("Warning: Buy & Hold baseline holds overnight while agent is forced flat at close.")
            base_final, _base_pnl, base_ret = _buy_and_hold(daily_summary, 100_000.0, trade_cost_ret)
            print(f"Buy & Hold final equity: {base_final:,.2f} (Return: {base_ret:.2%})")

    if args.plot_out is None:
        ticker_slug = normalize_ticker(cfg.ticker).lower()
        plot_path = (
            Path("Data")
            / "models"
            / "agent"
            / cfg.dataset_name
            / ticker_slug
            / "agent_actions_vs_price.png"
        )
    else:
        plot_path = Path(args.plot_out)
    _plot_actions(
        trace,
        plot_path,
        tail=args.plot_tail,
        random_window=args.plot_random_window,
        seed=args.plot_seed,
    )


if __name__ == "__main__":
    main()
