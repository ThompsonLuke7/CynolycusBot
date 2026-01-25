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
from Agent.env import TradingEnv
from Agent.eval import evaluate_policy, evaluate_policy_with_trace
from Agent.model import ActorCritic


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


def _plot_actions(trace: pd.DataFrame, output_path: Path, tail: int = 100) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot skipped: {exc}")
        return

    plot_df = trace.dropna(subset=["timestamp", "close"]).tail(tail)
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

    if "did_trade" in plot_df.columns:
        trade_mask = plot_df["did_trade"].astype(bool).to_numpy()
    else:
        trade_mask = plot_df["action"].ne(plot_df["action"].shift(1)).fillna(False).to_numpy()
    longs = trade_mask & (plot_df["action"].to_numpy() == 1)
    shorts = trade_mask & (plot_df["action"].to_numpy() == 2)

    fig, ax = plt.subplots(figsize=(12, 6))
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
        ax.bar(
            pos[valid & up],
            close_y[valid & up] - open_y[valid & up],
            width=0.8,
            bottom=open_y[valid & up],
            color=up_color,
            edgecolor="none",
            zorder=1.2,
        )
        ax.bar(
            pos[valid & ~up],
            close_y[valid & ~up] - open_y[valid & ~up],
            width=0.8,
            bottom=open_y[valid & ~up],
            color=down_color,
            edgecolor="none",
            zorder=1.2,
        )
        ax.scatter(pos[longs], close_y[longs], c="green", s=22, label="Long", marker="^", zorder=2.2)
        ax.scatter(pos[shorts], close_y[shorts], c="red", s=22, label="Short", marker="v", zorder=2.2)
        day_start = ts.dt.normalize().ne(ts.dt.normalize().shift())
        tick_positions = pos[day_start.to_numpy()]
        tick_labels = ts[day_start].dt.strftime("%Y-%m-%d").to_list()
        if len(tick_positions) > 25:
            step = int(np.ceil(len(tick_positions) / 25))
            tick_positions = tick_positions[::step]
            tick_labels = tick_labels[::step]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("Date")
    else:
        ax.plot(ts, close, label="SPY close", linewidth=1.5)
        ax.scatter(ts[longs], close[longs], c="green", s=18, label="Long", marker="^")
        ax.scatter(ts[shorts], close[shorts], c="red", s=18, label="Short", marker="v")
        ax.set_xlabel("Time")

    ax.set_title("SPY Candles with Agent Actions (Test Set)")
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved action plot to {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent and plot actions.")
    parser.add_argument("--model-path", default="Data/outputs/agent/ppo_model.pt")
    parser.add_argument("--baseline", choices=["intraday", "buy_hold", "none"], default="intraday")
    parser.add_argument("--plot-tail", type=int, default=100)
    parser.add_argument("--trace-out", default="Data/outputs/agent/agent_trace.csv")
    parser.add_argument("--plot-out", default="Data/plots/agent_actions_vs_price.png")
    parser.add_argument("--device", default=None, help="cuda/cpu (defaults to auto)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"Missing model file: {model_path}")

    cfg = PipelineConfig(drop_na=True)
    df = build_agent_training_matrix(cfg, save_parquet=False)
    splits = load_tree_split_indices(
        ticker=cfg.ticker,
        dataset_name=cfg.dataset_name,
        x_filename=cfg.x_filename,
    )
    _train_df, _val_df, test_df = split_agent_matrix(df, splits, verbose=True)

    drop_base = {"timestamp", "day_id", "open", "high", "low", "close", "volume"}
    feature_cols = [c for c in df.columns if c not in drop_base]

    ckpt = torch.load(model_path, map_location="cpu")
    model = ActorCritic(obs_dim=ckpt["obs_dim"], n_actions=ckpt.get("n_actions", 3))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_env = TradingEnv(
        df=test_df,
        feature_cols=feature_cols,
        add_time_features=False,
        add_position_features=True,
        commission_per_trade=0.00,
        slippage_bps=0.5,
        spread_bps=0.5,
        flip_penalty_ret=0.0002,
        trade_penalty_ret=0.00005,
        hold_penalty_ret=0.0,
        reward_on_exit=False,
        reward_exit_bonus=True,
        exit_pivot_bonus_ret=0.0001,
        force_flat_at_close=True,
        allow_direct_flip=False,
        seed=7,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    report = evaluate_policy(
        test_env,
        model,
        n_days=len(test_env.day_starts),
        device=device,
        deterministic=True,
    )
    print(report)
    print("Avg pnl component:", report["pnl_component"].mean(), "Avg costs:", report["costs_component"].mean())

    trace = evaluate_policy_with_trace(test_env, model, device=device, deterministic=True)
    trace = _agent_equity_from_trace(trace, initial_cash=100_000.0)
    trace_path = Path(args.trace_out)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace.to_csv(trace_path, index=False)
    print(f"Saved trace to {trace_path}")

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

    plot_path = Path(args.plot_out)
    _plot_actions(trace, plot_path, tail=args.plot_tail)


if __name__ == "__main__":
    main()
