from pathlib import Path
import sys
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
from Agent.train import train_ppo
from Agent.eval import evaluate_policy, evaluate_policy_with_trace


def _daily_closes(df):
    ordered = df.sort_values("timestamp")
    last_rows = ordered.groupby("day_id", sort=True).tail(1)
    return last_rows[["timestamp", "close"]].reset_index(drop=True)


def _buy_and_hold(daily_close, initial_cash):
    start = float(daily_close["close"].iloc[0])
    end = float(daily_close["close"].iloc[-1])
    shares = initial_cash / start
    final_value = shares * end
    return final_value, final_value - initial_cash


def _dca_down_days(daily_close, initial_cash):
    closes = daily_close["close"].astype(float).to_list()
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
        price = float(row["close"])
        costs = float(row.get("reward_costs", 0.0)) + float(row.get("forced_flat_cost", 0.0))
        net_ret = float(row.get("reward_pnl", 0.0)) - (costs / price if price else 0.0)
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

    plot_df = trace.dropna(subset=["timestamp", "close"])
    if plot_df.empty:
        print("Plot skipped: no data.")
        return

    ts = plot_df["timestamp"]
    close = plot_df["close"].astype(float)
    longs = plot_df["action"] == 1
    shorts = plot_df["action"] == 2

    plt.figure(figsize=(12, 6))
    plt.plot(ts, close, label="SPY close", linewidth=1.5)
    plt.scatter(ts[longs], close[longs], c="green", s=18, label="Long", marker="^")
    plt.scatter(ts[shorts], close[shorts], c="red", s=18, label="Short", marker="v")
    plt.title("SPY Close with Agent Actions (Test Set)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved action plot to {output_path}")


def main():
    cfg = PipelineConfig()
    df = build_agent_training_matrix(cfg)
    splits = load_tree_split_indices(
        ticker=cfg.ticker,
        dataset_name=cfg.dataset_name,
        x_filename=cfg.x_filename,
    )
    train_df, _val_df, test_df = split_agent_matrix(df, splits)

    feature_cols = [c for c in df.columns if c not in ("timestamp", "day_id", "close")]

    train_env = TradingEnv(
        df=train_df,
        feature_cols=feature_cols,
        add_time_features=False,
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
        train_env,
        total_timesteps=20_000,
        rollout_len=1024,
        train_epochs=5,
        minibatch_size=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True,
    )

    test_env = TradingEnv(
        df=test_df,
        feature_cols=feature_cols,
        add_time_features=False,
        add_position_features=True,
        commission_per_trade=0.00,
        slippage_bps=0.5,
        spread_bps=0.5,
        flip_penalty=0.0,
        force_flat_at_close=True,
        allow_direct_flip=True,
        seed=7,
    )

    report = evaluate_policy(test_env, model, n_days=len(test_env.day_starts), device="cpu")
    print(report)
    print("Avg pnl component:", report["pnl_component"].mean(), "Avg costs:", report["costs_component"].mean())

    trace = evaluate_policy_with_trace(test_env, model, device="cpu")
    trace = _agent_equity_from_trace(trace, initial_cash=100_000)

    daily_close = _daily_closes(test_df)
    bh_final, bh_pnl = _buy_and_hold(daily_close, 100_000)
    dca_final, dca_pnl, tranche = _dca_down_days(daily_close, 100_000)
    agent_final = float(trace["equity"].iloc[-1]) if not trace.empty else 100_000.0
    agent_pnl = agent_final - 100_000.0

    print(f"Agent final equity: {agent_final:,.2f} (PnL: {agent_pnl:,.2f})")
    print(f"Buy & Hold final equity: {bh_final:,.2f} (PnL: {bh_pnl:,.2f})")
    print(f"DCA down-days final equity: {dca_final:,.2f} (PnL: {dca_pnl:,.2f})")
    print(f"DCA tranche size per down day: {tranche:,.2f}")

    output_path = Path("Data") / "plots" / "agent_actions_vs_price.png"
    _plot_actions(trace, output_path)


if __name__ == "__main__":
    main()
