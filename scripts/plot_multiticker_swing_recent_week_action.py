from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plot_multiticker_swing_backtest_overview import (
    DEFAULT_RAW_30M_DIR,
    DEFAULT_SWEEP_DIR,
    TIER_COLORS,
    _load_best_trades,
    _local_naive,
    _metrics,
)
from shared_plotting import (
    DEFAULT_THEME,
    apply_mpl_defaults,
    apply_time_ticks,
    compute_time_ticks,
    plot_candles_from_frame,
    save_figure,
    style_figure,
    time_to_position,
)


DEFAULT_OUT_DIR = Path("UI/swing_audit/backtest_recent_week_20260606")
DEFAULT_OUT = DEFAULT_OUT_DIR / "multiticker_swing_30m_recent_week_action.png"
DEFAULT_METRICS_OUT = DEFAULT_OUT_DIR / "multiticker_swing_30m_recent_week_metrics.csv"
DEFAULT_SELECTED_TRADES_OUT = DEFAULT_OUT_DIR / "multiticker_swing_30m_recent_week_selected_trades.csv"


def _load_bars(ticker: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{ticker.upper()}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.columns = [str(c).lower() for c in df.columns]
    if "timestamp" not in df.columns and df.index.name and str(df.index.name).lower() == "timestamp":
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")


def _recent_window(trades: pd.DataFrame, days: int) -> tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    end = pd.to_datetime(trades["exit_time"], utc=True).max()
    start = end - pd.Timedelta(days=max(1, int(days)))
    recent = trades[trades["exit_time"].between(start, end)].copy()
    return start, end, recent


def _profit_factor(pnl: pd.Series) -> float:
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    return float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else math.inf


def _window_metrics(trades: pd.DataFrame, tiers: list[str]) -> pd.DataFrame:
    rows = [_metrics(trades[trades["tier"] == tier], tier) for tier in tiers if not trades[trades["tier"] == tier].empty]
    rows.append(_metrics(trades, "combined"))
    return pd.DataFrame(rows)


def _select_action_trades(
    trades: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    n_each: int = 3,
) -> pd.DataFrame:
    recent_entries = trades[trades["signal_time"].between(start, end)].copy()
    trade_pool = recent_entries[recent_entries["tier"].isin(["tier1", "tier2"])].copy()
    if len(trade_pool) < n_each * 2:
        trade_pool = trades[trades["tier"].isin(["tier1", "tier2"])].copy()
    if trade_pool.empty:
        trade_pool = trades.copy()

    winners = (
        trade_pool[trade_pool["pnl_pct"] > 0]
        .sort_values("pnl_pct", ascending=False)
        .drop_duplicates("ticker")
        .head(n_each)
    )
    losers = (
        trade_pool[trade_pool["pnl_pct"] <= 0]
        .sort_values("pnl_pct", ascending=True)
        .drop_duplicates("ticker")
        .head(n_each)
    )
    selected = pd.concat([winners, losers], ignore_index=True)
    selected["example_bucket"] = ["winner"] * len(winners) + ["loser"] * len(losers)
    return selected.head(n_each * 2)


def _plot_metrics_box(ax: plt.Axes, metrics: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    theme = DEFAULT_THEME
    ax.axis("off")
    combined = metrics[metrics["tier"] == "combined"].iloc[0]
    text = "\n".join(
        [
            f"Window: {_local_naive(pd.Series([start])).iloc[0]:%Y-%m-%d} to {_local_naive(pd.Series([end])).iloc[0]:%Y-%m-%d}",
            f"Trades: {int(combined['trades']):,}",
            f"Win rate: {combined['win_rate'] * 100:.1f}%",
            f"Profit factor: {combined['profit_factor']:.2f}",
            f"Sharpe: {combined['sharpe']:.2f}",
            f"Total PnL: {combined['total_pnl_pp']:+,.1f} pp",
            f"Avg trade: {combined['avg_trade_pp']:+.2f} pp",
            f"Max DD: {combined['max_dd_pp']:+,.1f} pp",
        ]
    )
    ax.text(
        0.02,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=theme.text,
        fontsize=11,
        linespacing=1.45,
        bbox={"facecolor": "#111827", "edgecolor": theme.spine, "boxstyle": "round,pad=0.5", "alpha": 0.95},
    )
    ax.set_title("Recent Week Summary", loc="left", fontsize=12, weight="bold")


def _plot_week_equity(ax: plt.Axes, trades: pd.DataFrame) -> None:
    theme = DEFAULT_THEME
    for label in ["tier1", "tier2", "tier3"]:
        t = trades[trades["tier"] == label].sort_values("exit_time").copy()
        if t.empty:
            continue
        t["cum_pnl_pp"] = t["pnl_pct"].cumsum() * 100.0
        ax.plot(_local_naive(t["exit_time"]), t["cum_pnl_pp"], color=TIER_COLORS[label], lw=1.4, alpha=0.82, label=label)

    combined = trades.sort_values("exit_time").copy()
    combined["cum_pnl_pp"] = combined["pnl_pct"].cumsum() * 100.0
    ax.plot(_local_naive(combined["exit_time"]), combined["cum_pnl_pp"], color=theme.text, lw=2.2, label="combined")
    ax.axhline(0, color=theme.neutral, lw=0.8, alpha=0.55)
    ax.set_title("Recent Week Backtest PnL by Exit Time", loc="left", fontsize=13, weight="bold")
    ax.set_ylabel("Cumulative PnL pp")
    ax.legend(loc="upper left", ncol=4, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))


def _plot_daily_bars(ax: plt.Axes, trades: pd.DataFrame) -> None:
    theme = DEFAULT_THEME
    work = trades.copy()
    work["day"] = _local_naive(work["exit_time"]).dt.strftime("%m-%d")
    daily = work.groupby(["day", "tier"])["pnl_pct"].sum().mul(100.0).unstack(fill_value=0.0).sort_index()
    x = np.arange(len(daily))
    bottom = np.zeros(len(daily))
    for tier in ["tier1", "tier2", "tier3"]:
        vals = daily[tier].to_numpy(dtype=float) if tier in daily.columns else np.zeros(len(daily))
        ax.bar(x, vals, bottom=bottom, color=TIER_COLORS[tier], alpha=0.85, width=0.7, label=tier)
        bottom += vals
    ax.axhline(0, color=theme.neutral, lw=0.8, alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(daily.index, rotation=30, ha="right")
    ax.set_title("Daily Realized PnL", loc="left", fontsize=12, weight="bold")
    ax.set_ylabel("PnL pp")
    ax.legend(loc="upper left", ncol=3, fontsize=8)


def _plot_trade_card(ax: plt.Axes, trade: pd.Series, raw_dir: Path) -> None:
    theme = DEFAULT_THEME
    ticker = str(trade["ticker"]).upper()
    bars = _load_bars(ticker, raw_dir)
    if bars.empty:
        ax.axis("off")
        ax.set_title(f"{ticker}: missing bars")
        return

    start = pd.Timestamp(trade["signal_time"]) - pd.Timedelta(days=2)
    end = pd.Timestamp(trade["exit_time"]) + pd.Timedelta(days=1)
    view = bars[bars["timestamp"].between(start, end)].copy()
    if view.empty:
        ax.axis("off")
        ax.set_title(f"{ticker}: no bars in view")
        return

    view = view.reset_index(drop=True)
    view_indexed = view.set_index("timestamp", drop=False)
    candle = plot_candles_from_frame(ax, view_indexed, compressed=True, theme=theme, width=0.68)
    x_entry = float(time_to_position(pd.DatetimeIndex(view_indexed.index), pd.Series([trade["signal_time"]])).iloc[0])
    x_exit = float(time_to_position(pd.DatetimeIndex(view_indexed.index), pd.Series([trade["exit_time"]])).iloc[0])
    direction = int(trade["direction"])
    pnl = float(trade["pnl_pct"])
    side = "LONG" if direction == 1 else "SHORT"
    color = theme.win if pnl > 0 else theme.loss
    marker = "^" if direction == 1 else "v"

    ax.axvspan(x_entry, x_exit, color=color, alpha=0.10, zorder=0.3)
    ax.plot([x_entry, x_exit], [trade["entry_price"], trade["exit_price"]], color=color, lw=1.2, ls="--", alpha=0.9, zorder=4)
    ax.scatter(x_entry, trade["entry_price"], marker=marker, s=72, color=theme.blue, edgecolor="#f8fafc", linewidth=0.55, zorder=5)
    ax.scatter(x_exit, trade["exit_price"], marker="X", s=72, color=color, edgecolor="#f8fafc", linewidth=0.55, zorder=5)
    ax.axhline(float(trade["entry_price"]), color=theme.blue, lw=0.7, ls=":", alpha=0.65)

    tick_pos, tick_labels = compute_time_ticks(pd.DatetimeIndex(view_indexed.index), candle.x, max_ticks=5, fmt="%m-%d")
    apply_time_ticks(ax, tick_pos, tick_labels, color=theme.muted_text, fontsize=7)
    ax.set_xlim(max(0, x_entry - 18), min(len(view) - 1, x_exit + 18))
    ax.set_title(
        f"{ticker} {side} {pnl * 100:+.1f}% | {trade['tier']} | {trade['exit_reason']}",
        loc="left",
        fontsize=9.5,
        weight="bold",
    )
    ax.tick_params(axis="y", labelsize=7)


def plot_recent_week(
    trades: pd.DataFrame,
    metrics: pd.DataFrame,
    selected: pd.DataFrame,
    raw_dir: Path,
    save_path: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Path:
    theme = DEFAULT_THEME
    apply_mpl_defaults(theme, font_size=10)
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = fig.add_gridspec(4, 3, height_ratios=[1.45, 0.95, 1.25, 1.25])
    ax_equity = fig.add_subplot(gs[0, :])
    ax_daily = fig.add_subplot(gs[1, :2])
    ax_metrics = fig.add_subplot(gs[1, 2])
    trade_axes = [fig.add_subplot(gs[2 + row, col]) for row in range(2) for col in range(3)]
    style_figure(fig, [ax_equity, ax_daily, ax_metrics, *trade_axes], theme)

    _plot_week_equity(ax_equity, trades)
    _plot_daily_bars(ax_daily, trades)
    _plot_metrics_box(ax_metrics, metrics, start, end)
    for ax, (_, trade) in zip(trade_axes, selected.iterrows(), strict=False):
        _plot_trade_card(ax, trade, raw_dir)
    for ax in trade_axes[len(selected):]:
        ax.axis("off")

    fig.suptitle(
        "Multi-Ticker Swing 30m Model Backtest in Action | Latest Week in Saved Test Window",
        x=0.01,
        ha="left",
        fontsize=16,
        weight="bold",
        color=theme.text,
    )
    save_figure(fig, save_path, dpi=170, tight=False, close=True)
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the latest-week multi-ticker swing backtest in action.")
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--raw-30m-dir", type=Path, default=DEFAULT_RAW_30M_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_OUT)
    parser.add_argument("--selected-trades-out", type=Path, default=DEFAULT_SELECTED_TRADES_OUT)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--tiers", nargs="+", default=["tier1", "tier2", "tier3"])
    args = parser.parse_args()

    trades = _load_best_trades(args.sweep_dir, args.raw_30m_dir, list(args.tiers))
    start, end, recent = _recent_window(trades, args.days)
    if recent.empty:
        raise SystemExit("No recent trades found in selected window.")

    metrics = _window_metrics(recent, list(args.tiers))
    selected = _select_action_trades(recent, start=start, end=end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics_out, index=False)
    selected.to_csv(args.selected_trades_out, index=False)
    out = plot_recent_week(recent, metrics, selected, args.raw_30m_dir, args.out, start=start, end=end)
    print(out)
    print(args.metrics_out)
    print(args.selected_trades_out)


if __name__ == "__main__":
    main()
