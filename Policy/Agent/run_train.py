from pathlib import Path
import sys
from regex import F
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
from Agent.env import TradingEnv
from Agent.train import train_ppo
from Agent.eval import evaluate_policy, evaluate_policy_with_trace


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


def main():
    cfg = PipelineConfig(drop_na=True)
    df = build_agent_training_matrix(cfg, save_parquet=True)
    splits = load_tree_split_indices(
        ticker=cfg.ticker,
        dataset_name=cfg.dataset_name,
        x_filename=cfg.x_filename,
    )

    drop_base = {"timestamp", "day_id", "open", "high", "low", "close", "volume"}
    feature_cols = [c for c in df.columns if c not in drop_base]
    all_nan_cols = [c for c in feature_cols if df[c].isna().all()]
    if all_nan_cols:
        print(f"Dropping all-NaN feature columns: {all_nan_cols}")
        feature_cols = [c for c in feature_cols if c not in all_nan_cols]
    if cfg.drop_na:
        splits = filter_splits_for_non_nan(df, splits, feature_cols)
    train_df, _val_df, test_df = split_agent_matrix(df, splits, verbose=True)
    if train_df.empty or test_df.empty:
        nan_counts = df[feature_cols].isna().sum().sort_values(ascending=False).head(10)
        raise ValueError(
            "Train/Test split is empty after NaN filtering. "
            f"Top NaN counts:\n{nan_counts}"
        )

    train_env = TradingEnv(
        df=train_df,
        feature_cols=feature_cols,
        add_time_features=False,
        add_position_features=True,
        commission_per_trade=0.00,
        slippage_bps=0.5,
        spread_bps=0.5,
        flip_penalty_ret=0.0002,
        force_flat_at_close=True,
        allow_direct_flip=False,
        seed=7,
    )

    model = train_ppo(
        train_env,
        total_timesteps=200_000,
        rollout_len=1024,
        train_epochs=5,
        minibatch_size=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True,
    )
    output_dir = Path("Data") / "outputs" / "agent"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "ppo_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": train_env.obs_dim,
            "n_actions": 3,
            "feature_cols": feature_cols,
            "config": cfg.__dict__,
        },
        model_path,
    )
    print(f"Saved model to {model_path}")

    test_env = TradingEnv(
        df=test_df,
        feature_cols=feature_cols,
        add_time_features=False,
        add_position_features=True,
        commission_per_trade=0.00,
        slippage_bps=0.5,
        spread_bps=0.5,
        flip_penalty_ret=0.0002,
        force_flat_at_close=True,
        allow_direct_flip=False,
        seed=7,
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

    initial_cash = 100_000.0
    baseline_mode = "intraday"  # "intraday", "buy_hold", or "none"
    include_dca = False

    trace = evaluate_policy_with_trace(test_env, model, device=eval_device, deterministic=True)
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

    output_path = Path("Data") / "plots" / "agent_actions_vs_price.png"
    _plot_actions(trace, output_path)


if __name__ == "__main__":
    main()
