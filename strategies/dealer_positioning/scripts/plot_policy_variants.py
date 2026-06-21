from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

from core.shared_plotting import (
    apply_mpl_defaults,
    compute_marker_offset,
    plot_candles_from_frame,
    save_figure,
    style_figure,
    time_to_position,
)


def _load_bars(path: Path) -> pd.DataFrame:
    bars = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    return bars


def _load_trades(path: Path, policies: list[str]) -> pd.DataFrame:
    trades = pd.read_csv(path, parse_dates=["timestamp", "exit_time"])
    return trades[trades["policy"].isin(policies)].copy()


def _plot_policy(ax, bars: pd.DataFrame, trades: pd.DataFrame, policy: str, theme) -> None:
    plot_candles_from_frame(ax, bars, time_col="timestamp", compressed=True, theme=theme, label=False)
    ax.plot(range(len(bars)), bars["close"], color=theme.text, lw=0.9, alpha=0.72, label="SPY close")
    if trades.empty:
        ax.set_title(f"{policy} - no trades")
        return
    bar_index = pd.DatetimeIndex(bars["timestamp"])
    marker_offset = compute_marker_offset(bars, bars["high"].to_numpy(dtype=float), bars["low"].to_numpy(dtype=float))
    entries_x = time_to_position(bar_index, trades["timestamp"])
    exits_x = time_to_position(bar_index, trades["exit_time"])
    longs = trades["direction"].eq("long")
    shorts = trades["direction"].eq("short")
    wins = trades["pnl_points"] > 0

    ax.scatter(
        entries_x[longs],
        trades.loc[longs, "entry"] + marker_offset * 0.55,
        marker="^",
        s=80,
        color="#22c55e",
        edgecolor="#0f172a",
        linewidth=0.6,
        label="long entry",
        zorder=4,
    )
    ax.scatter(
        entries_x[shorts],
        trades.loc[shorts, "entry"] - marker_offset * 0.55,
        marker="v",
        s=80,
        color="#f59e0b",
        edgecolor="#0f172a",
        linewidth=0.6,
        label="short entry",
        zorder=4,
    )
    ax.scatter(
        exits_x[wins],
        trades.loc[wins, "exit_price"],
        marker="x",
        s=60,
        color="#86efac",
        linewidth=1.6,
        label="win exit",
        zorder=4.2,
    )
    ax.scatter(
        exits_x[~wins],
        trades.loc[~wins, "exit_price"],
        marker="x",
        s=60,
        color="#fb7185",
        linewidth=1.6,
        label="loss exit",
        zorder=4.2,
    )
    for _, trade in trades.iterrows():
        x0 = float(time_to_position(bar_index, pd.Series([trade["timestamp"]])).iloc[0])
        x1 = float(time_to_position(bar_index, pd.Series([trade["exit_time"]])).iloc[0])
        color = "#86efac" if float(trade["pnl_points"]) > 0 else "#fb7185"
        ax.plot([x0, x1], [trade["entry"], trade["exit_price"]], color=color, lw=1.1, alpha=0.8, zorder=3.5)

    pnl = trades["pnl_points"].sum()
    wr = (trades["pnl_points"] > 0).mean()
    ax.set_title(f"{policy} | trades={len(trades)} win={wr:.0%} pnl={pnl:.2f} pts")


def plot_variants(*, bars_path: Path, trades_path: Path, output_path: Path, policies: list[str]) -> Path:
    bars = _load_bars(bars_path)
    trades = _load_trades(trades_path, policies)
    theme = apply_mpl_defaults()
    fig, axes = plt.subplots(len(policies), 1, figsize=(16, 3.8 * len(policies)), sharex=True)
    if len(policies) == 1:
        axes = [axes]
    style_figure(fig, axes, theme)
    for ax, policy in zip(axes, policies):
        _plot_policy(ax, bars, trades[trades["policy"].eq(policy)].copy(), policy, theme)
        ax.set_ylabel("SPY")
    tick_rows = bars.iloc[::30]
    axes[-1].set_xticks(tick_rows.index.to_numpy(dtype=float))
    axes[-1].set_xticklabels(
        pd.to_datetime(tick_rows["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%H:%M"),
        rotation=45,
        ha="right",
    )
    axes[-1].set_xlabel("time ET")
    legend = axes[0].legend(loc="upper left", ncols=4, fontsize=9, framealpha=0.78)
    legend.get_frame().set_facecolor(theme.axes_bg)
    legend.get_frame().set_edgecolor(theme.spine)
    for text in legend.get_texts():
        text.set_color(theme.text)
    return save_figure(fig, output_path, close=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot selected dealer-positioning policy variants.")
    parser.add_argument("--bars", default="Data/raw/spy/spy_intraday_1min_2026_06_12.parquet")
    parser.add_argument("--trades", default="Data/dealer_positioning/reports/SPY_policy_experiment_trades_2026-06-12.csv")
    parser.add_argument("--out", default="Data/dealer_positioning/reports/SPY_policy_variant_comparison_2026-06-12.png")
    parser.add_argument(
        "--policies",
        default="long_only_t0.40_ch3,long_only_t0.60_ch3,put_bounce_only_tight,tight_channel_correct_side",
    )
    args = parser.parse_args()
    out = plot_variants(
        bars_path=Path(args.bars),
        trades_path=Path(args.trades),
        output_path=Path(args.out),
        policies=[x.strip() for x in args.policies.split(",") if x.strip()],
    )
    print(out)


if __name__ == "__main__":
    main()
