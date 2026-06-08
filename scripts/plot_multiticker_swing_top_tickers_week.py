from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from scripts.plot_multiticker_swing_backtest_overview import (
    DEFAULT_RAW_30M_DIR,
    DEFAULT_SWEEP_DIR,
    _load_best_trades,
    _local_naive,
)
from scripts.plot_multiticker_swing_recent_week_action import _load_bars
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


DEFAULT_OUT_DIR = Path("UI/swing_audit/backtest_top_tickers_week_20260606")
DEFAULT_OUT = DEFAULT_OUT_DIR / "multiticker_swing_30m_top5_week_trades.png"
DEFAULT_SUMMARY_OUT = DEFAULT_OUT_DIR / "multiticker_swing_30m_top5_week_summary.csv"
DEFAULT_TRADES_OUT = DEFAULT_OUT_DIR / "multiticker_swing_30m_top5_week_trades.csv"


def _window(trades: pd.DataFrame, days: int) -> tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    end = pd.to_datetime(trades["exit_time"], utc=True).max()
    start = end - pd.Timedelta(days=max(1, int(days)))
    week = trades[trades["exit_time"].between(start, end) & trades["signal_time"].between(start, end)].copy()
    return start, end, week


def _rank_tickers(week: pd.DataFrame, *, top_n: int, min_trades: int) -> pd.DataFrame:
    rows = []
    for (ticker, tier), group in week.groupby(["ticker", "tier"], sort=False):
        pnl = pd.to_numeric(group["pnl_pct"], errors="coerce")
        rows.append(
            {
                "ticker": ticker,
                "tier": tier,
                "trades": int(len(group)),
                "total_pnl_pct": float(pnl.sum()),
                "avg_pnl_pct": float(pnl.mean()),
                "win_rate": float((pnl > 0).mean()),
                "long_trades": int((group["direction"] == 1).sum()),
                "short_trades": int((group["direction"] == -1).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    ranked = summary[summary["trades"] >= int(min_trades)].sort_values(
        ["total_pnl_pct", "win_rate", "trades"],
        ascending=[False, False, False],
    )
    if len(ranked) < top_n:
        ranked = summary.sort_values(["total_pnl_pct", "win_rate", "trades"], ascending=[False, False, False])
    return ranked.head(top_n).reset_index(drop=True)


def _plot_ticker_week(ax: plt.Axes, ticker: str, bars: pd.DataFrame, trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    theme = DEFAULT_THEME
    local_start = start - pd.Timedelta(days=1)
    local_end = end + pd.Timedelta(hours=8)
    view = bars[bars["timestamp"].between(local_start, local_end)].copy()
    if view.empty:
        ax.axis("off")
        ax.set_title(f"{ticker}: no 30m bars found")
        return

    view = view.reset_index(drop=True)
    view_indexed = view.set_index("timestamp", drop=False)
    candle = plot_candles_from_frame(ax, view_indexed, compressed=True, theme=theme, width=0.68)
    index = pd.DatetimeIndex(view_indexed.index)

    for trade_num, (_, trade) in enumerate(trades.sort_values("signal_time").iterrows(), 1):
        x_entry = float(time_to_position(index, pd.Series([trade["signal_time"]])).iloc[0])
        x_exit = float(time_to_position(index, pd.Series([trade["exit_time"]])).iloc[0])
        pnl = float(trade["pnl_pct"])
        direction = int(trade["direction"])
        color = theme.win if pnl > 0 else theme.loss
        entry_marker = "^" if direction == 1 else "v"

        ax.axvspan(x_entry, x_exit, color=color, alpha=0.08, zorder=0.2)
        ax.plot(
            [x_entry, x_exit],
            [float(trade["entry_price"]), float(trade["exit_price"])],
            color=color,
            lw=1.15,
            ls="--",
            alpha=0.9,
            zorder=4,
        )
        ax.scatter(
            x_entry,
            float(trade["entry_price"]),
            marker=entry_marker,
            s=58,
            color=theme.blue,
            edgecolor="#f8fafc",
            linewidth=0.55,
            zorder=5,
        )
        ax.scatter(
            x_exit,
            float(trade["exit_price"]),
            marker="X",
            s=58,
            color=color,
            edgecolor="#f8fafc",
            linewidth=0.55,
            zorder=5,
        )
        ax.text(
            x_exit,
            float(trade["exit_price"]),
            f"{trade_num}:{pnl * 100:+.1f}%",
            color=theme.text,
            fontsize=7,
            ha="left",
            va="bottom" if pnl > 0 else "top",
            zorder=6,
        )

    local_week = view_indexed[index.to_series().between(start, end).to_numpy()]
    if not local_week.empty:
        x0 = float(time_to_position(index, pd.Series([start])).iloc[0])
        x1 = float(time_to_position(index, pd.Series([end])).iloc[0])
        ax.axvline(x0, color=theme.neutral, lw=0.8, ls=":", alpha=0.7)
        ax.axvline(x1, color=theme.neutral, lw=0.8, ls=":", alpha=0.7)

    tick_pos, tick_labels = compute_time_ticks(index, candle.x, max_ticks=8, fmt="%m-%d")
    apply_time_ticks(ax, tick_pos, tick_labels, color=theme.muted_text, fontsize=8)
    ax.tick_params(axis="y", labelsize=8)

    pnl_sum = trades["pnl_pct"].sum()
    win_rate = (trades["pnl_pct"] > 0).mean()
    tier = str(trades["tier"].iloc[0])
    ax.set_title(
        f"{ticker} | {tier} | {len(trades)} trades | {pnl_sum * 100:+.1f}% total | {win_rate * 100:.0f}% win",
        loc="left",
        fontsize=10,
        weight="bold",
    )


def plot_top_tickers_week(
    week: pd.DataFrame,
    ranked: pd.DataFrame,
    raw_dir: Path,
    save_path: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Path:
    theme = DEFAULT_THEME
    apply_mpl_defaults(theme, font_size=10)
    fig, axes = plt.subplots(len(ranked), 1, figsize=(18, max(10, len(ranked) * 2.6)), sharex=False)
    if len(ranked) == 1:
        axes = [axes]
    style_figure(fig, axes, theme)

    for ax, (_, row) in zip(axes, ranked.iterrows(), strict=False):
        ticker = str(row["ticker"]).upper()
        ticker_trades = week[week["ticker"].eq(ticker)].copy()
        bars = _load_bars(ticker, raw_dir)
        _plot_ticker_week(ax, ticker, bars, ticker_trades, start, end)

    start_label = _local_naive(pd.Series([start])).iloc[0].strftime("%Y-%m-%d")
    end_label = _local_naive(pd.Series([end])).iloc[0].strftime("%Y-%m-%d")
    fig.suptitle(
        f"Multi-Ticker Swing 30m Backtest | Top 5 Tickers by Latest-Week Trade PnL ({start_label} to {end_label})",
        x=0.01,
        ha="left",
        fontsize=15,
        weight="bold",
        color=theme.text,
    )
    save_figure(fig, save_path, dpi=175, tight=False, close=True)
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot top-performing tickers with all recent-week model trades.")
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--raw-30m-dir", type=Path, default=DEFAULT_RAW_30M_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--trades-out", type=Path, default=DEFAULT_TRADES_OUT)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--min-trades", type=int, default=2)
    parser.add_argument("--tiers", nargs="+", default=["tier1", "tier2", "tier3"])
    args = parser.parse_args()

    trades = _load_best_trades(args.sweep_dir, args.raw_30m_dir, list(args.tiers))
    start, end, week = _window(trades, args.days)
    if week.empty:
        raise SystemExit("No trades found in latest-week entry/exit window.")
    ranked = _rank_tickers(week, top_n=args.top_n, min_trades=args.min_trades)
    if ranked.empty:
        raise SystemExit("No ranked tickers found.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    picked = week[week["ticker"].isin(ranked["ticker"])].copy()
    ranked.to_csv(args.summary_out, index=False)
    picked.to_csv(args.trades_out, index=False)
    out = plot_top_tickers_week(picked, ranked, args.raw_30m_dir, args.out, start=start, end=end)
    print(out)
    print(args.summary_out)
    print(args.trades_out)
    print(ranked.to_string(index=False))


if __name__ == "__main__":
    main()
