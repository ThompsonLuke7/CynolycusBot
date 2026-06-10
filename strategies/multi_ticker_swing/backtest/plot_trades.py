"""
Plot candlestick charts with trade entry/exit overlays for the best sweep combo.

Usage:
  python multi_ticker_swing/backtest/plot_trades.py \
      --sweep-dir backtest/results/sweep_1500 \
      --proba     models/p_swing_probs.parquet \
      --best      IWM LABU AG NNE \
      --worst     SLB BE CVX FSLR
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from strategies.multi_ticker_swing.config.pipeline_config import RAW_30M_DIR, BACKTEST_RESULTS_DIR
from core.shared_plotting import (
    DEFAULT_THEME,
    apply_mpl_defaults,
    apply_time_ticks,
    compute_time_ticks,
    plot_candles,
    style_figure,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
TAIL_BARS = 315          # ~24 trading days ≈ 1 month

apply_mpl_defaults()


def load_raw(ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW_30M_DIR / f"{ticker}.parquet")
    if "timestamp" not in df.columns:
        df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def plot_ticker_trades(
    ticker: str,
    raw: pd.DataFrame,
    trades: pd.DataFrame,
    proba: pd.DataFrame,
    out_path: Path,
    tail_bars: int = TAIL_BARS,
) -> None:
    raw = raw.tail(tail_bars).reset_index(drop=True)
    ts_start = raw["timestamp"].iloc[0]
    ts_end = raw["timestamp"].iloc[-1]

    trades = trades[
        (trades["entry_time"] >= ts_start) & (trades["entry_time"] <= ts_end)
    ].copy()

    proba_t = proba[
        (proba["timestamp"] >= ts_start) & (proba["timestamp"] <= ts_end)
    ].copy()

    n = len(raw)
    pos = np.arange(n)
    ts_to_pos = {pd.Timestamp(ts, tz="UTC"): i for ts, i in zip(raw["timestamp"].values, pos)}

    fig, axes = plt.subplots(2, 1, figsize=(22, 10), height_ratios=[3, 1],
                              sharex=True, gridspec_kw={"hspace": 0.05})
    ax_candle = axes[0]
    ax_proba = axes[1]
    theme = DEFAULT_THEME
    style_figure(fig, axes, theme)

    # ── Candles ───────────────────────────────────────────────────────────────
    plot_candles(
        ax_candle, pos,
        raw["open"].values, raw["high"].values,
        raw["low"].values, raw["close"].values,
        width=0.6,
    )

    # ── Trade overlays ────────────────────────────────────────────────────────
    for _, t in trades.iterrows():
        entry_ts = pd.Timestamp(t["entry_time"])
        exit_ts = pd.Timestamp(t["exit_time"])

        entry_pos = ts_to_pos.get(entry_ts)
        exit_pos = ts_to_pos.get(exit_ts)
        if entry_pos is None or exit_pos is None:
            continue

        is_long = t["direction"] == 1
        is_win = t["pnl_pct"] > 0
        color = theme.win if is_win else theme.loss
        marker_entry = "^" if is_long else "v"

        ax_candle.plot(entry_pos, t["entry_price"], marker=marker_entry,
                       color=color, markersize=11, zorder=5,
                       markeredgecolor="white", markeredgewidth=0.8)

        reason_markers = {"tp": "D", "sl": "X", "time": "s", "trail": "P", "opp_signal": "o"}
        ax_candle.plot(exit_pos, t["exit_price"],
                       marker=reason_markers.get(t["exit_reason"], "o"),
                       color=color, markersize=9, zorder=5,
                       markeredgecolor="white", markeredgewidth=0.6)

        ax_candle.plot([entry_pos, exit_pos], [t["entry_price"], t["exit_price"]],
                       color=color, linewidth=1.2, alpha=0.6, linestyle="--", zorder=3)

    n_trades = len(trades)
    wins = (trades["pnl_pct"] > 0).sum() if n_trades > 0 else 0
    total_pnl = trades["pnl_pct"].sum() * 100 if n_trades > 0 else 0
    wr = wins / n_trades * 100 if n_trades > 0 else 0

    ax_candle.set_title(
        f"{ticker}  |  trail arm 2.5%, giveback 25%, SL 4×ATR, 5m confirm 6 bars  |  "
        f"Trades={n_trades}  WR={wr:.0f}%  totalPnL={total_pnl:+.1f}%",
        fontsize=12, fontweight="bold",
    )
    ax_candle.set_ylabel("Price", fontsize=11)

    # ── Probability panel — same style as SPY day trader ────────────────────
    if not proba_t.empty:
        # Compute directional-conditional probabilities — these are what actually
        # drive entries. p_long_dir = P(long) / (P(long) + P(short)).
        # Plotting raw P(long)/P(short) is misleading because neutral mass dilutes them.
        p_dir = (proba_t["p_long"] + proba_t["p_short"]).clip(lower=1e-8)
        proba_t = proba_t.copy()
        proba_t["p_long_dir"]  = proba_t["p_long"]  / p_dir
        proba_t["p_short_dir"] = proba_t["p_short"] / p_dir

        proba_positions, p_long_dir, p_short_dir = [], [], []
        for _, pr in proba_t.iterrows():
            p = ts_to_pos.get(pr["timestamp"])
            if p is not None:
                proba_positions.append(p)
                p_long_dir.append(pr["p_long_dir"])
                p_short_dir.append(pr["p_short_dir"])

        proba_positions = np.array(proba_positions)
        p_long_dir  = np.array(p_long_dir)
        p_short_dir = np.array(p_short_dir)

        ax_proba.plot(proba_positions, p_long_dir,
                      color=theme.long, linewidth=1.5, label="P(long|dir)")
        ax_proba.plot(proba_positions, p_short_dir,
                      color=theme.short, linewidth=1.5, label="P(short|dir)")

    # Entry threshold line — entries fire when directional prob crosses this
    ax_proba.axhline(0.5, color=theme.neutral, linewidth=0.8, linestyle="--", alpha=0.4, label="50%")

    ax_proba.set_ylabel("Directional probability", fontsize=11)
    ax_proba.set_ylim(0, 1.02)
    ax_proba.legend(loc="upper right", fontsize=9)

    # ── Time axis ─────────────────────────────────────────────────────────────
    date_idx = pd.DatetimeIndex(raw["timestamp"])
    tick_pos, tick_labels = compute_time_ticks(date_idx, pos, max_ticks=18)
    apply_time_ticks(ax_proba, tick_pos, tick_labels, color=theme.muted_text)

    # ── Trade legend ──────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color=theme.win, label="Win"),
        mpatches.Patch(color=theme.loss, label="Loss"),
        plt.Line2D([0], [0], marker="^", color="gray", markersize=8, linestyle="None", label="Long entry"),
        plt.Line2D([0], [0], marker="v", color="gray", markersize=8, linestyle="None", label="Short entry"),
        plt.Line2D([0], [0], marker="P", color="gray", markersize=8, linestyle="None", label="Trail exit"),
        plt.Line2D([0], [0], marker="X", color="gray", markersize=8, linestyle="None", label="SL exit"),
        plt.Line2D([0], [0], marker="s", color="gray", markersize=7, linestyle="None", label="Time exit"),
    ]
    ax_candle.legend(handles=legend_elements, loc="upper left", fontsize=9, ncol=4)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", required=True, help="Path to sweep result directory")
    p.add_argument("--proba",     required=True, help="Path to p_swing_probs.parquet")
    p.add_argument("--best",  nargs="+", default=[], help="Best tickers to plot")
    p.add_argument("--worst", nargs="+", default=[], help="Worst tickers to plot")
    p.add_argument("--tail-bars", type=int, default=TAIL_BARS)
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir)
    plots_dir = sweep_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _default_trades = "best_v2_trades.parquet"
    for _candidate in ("best_v4_trades.parquet", "best_v3_trades.parquet", "best_v2_trades.parquet"):
        if (sweep_dir / _candidate).exists():
            _default_trades = _candidate
            break
    trades_df = pd.read_parquet(sweep_dir / _default_trades)
    proba = pd.read_parquet(args.proba)
    proba = proba[proba["split"] == "test"].copy()
    proba["timestamp"] = pd.to_datetime(proba["timestamp"], utc=True)

    all_tickers = [(t, "best") for t in args.best] + [(t, "worst") for t in args.worst]
    for ticker, label in all_tickers:
        logger.info("Plotting %s (%s)...", ticker, label)
        try:
            raw = load_raw(ticker)
        except FileNotFoundError:
            logger.warning("No raw data for %s", ticker)
            continue

        ticker_trades = trades_df[trades_df["ticker"] == ticker].copy()
        if ticker_trades.empty:
            logger.warning("No trades for %s", ticker)
            continue

        ticker_trades = ticker_trades.copy()
        ticker_trades["entry_time"] = ticker_trades["signal_idx"].apply(
            lambda idx: raw.iloc[min(idx + 1, len(raw) - 1)]["timestamp"]
        )
        ticker_trades["exit_time"] = ticker_trades["exit_idx"].apply(
            lambda idx: raw.iloc[min(idx, len(raw) - 1)]["timestamp"]
        )
        ticker_trades = ticker_trades.dropna(subset=["entry_time", "exit_time"])

        proba_t = proba[proba["ticker"] == ticker]

        out_path = plots_dir / f"trades_{label}_{ticker}.png"
        plot_ticker_trades(ticker, raw, ticker_trades, proba_t, out_path,
                           tail_bars=args.tail_bars)

    logger.info("All plots saved → %s", plots_dir)


if __name__ == "__main__":
    main()
