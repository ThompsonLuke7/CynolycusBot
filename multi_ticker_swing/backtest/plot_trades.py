"""
Plot candlestick charts with trade entry/exit overlays for the best sweep combo.

Usage:
  python -m multi_ticker_swing.backtest.plot_trades
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from multi_ticker_swing.config.pipeline_config import RAW_30M_DIR, BACKTEST_RESULTS_DIR
from Data.plots.plots import _plot_candles, _compute_time_ticks, _apply_time_ticks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

SWEEP_DIR = BACKTEST_RESULTS_DIR / "sweep"
PLOTS_DIR = SWEEP_DIR / "plots"
PROBA_PATH = Path(__file__).resolve().parents[1] / "models" / "p_swing_probs.parquet"

BEST_TICKERS = ["SMCI", "MSTR", "VRT", "MU"]
WORST_TICKERS = ["NFLX", "SNDK", "UNH", "GDX"]
TAIL_BARS = 500


def load_raw(ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW_30M_DIR / f"{ticker}.parquet")
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

    fig, axes = plt.subplots(2, 1, figsize=(20, 10), height_ratios=[3, 1],
                              sharex=True, gridspec_kw={"hspace": 0.05})
    ax_candle = axes[0]
    ax_proba = axes[1]

    # Candles
    _plot_candles(
        ax_candle, pos,
        raw["open"].values, raw["high"].values,
        raw["low"].values, raw["close"].values,
        width=0.6,
    )

    # Trade overlays
    for _, t in trades.iterrows():
        entry_ts = pd.Timestamp(t["entry_time"])
        exit_ts = pd.Timestamp(t["exit_time"])

        entry_pos = ts_to_pos.get(entry_ts)
        exit_pos = ts_to_pos.get(exit_ts)
        if entry_pos is None or exit_pos is None:
            continue

        is_long = t["direction"] == 1
        is_win = t["pnl_pct"] > 0
        color = "#2E7D32" if is_win else "#C62828"
        marker_entry = "^" if is_long else "v"

        ax_candle.plot(entry_pos, t["entry_price"], marker=marker_entry,
                       color=color, markersize=10, zorder=5, markeredgecolor="black", markeredgewidth=0.5)

        reason_markers = {"tp": "D", "sl": "X", "time": "s", "trail": "P", "prob_exit": "o"}
        ax_candle.plot(exit_pos, t["exit_price"],
                       marker=reason_markers.get(t["exit_reason"], "o"),
                       color=color, markersize=8, zorder=5, markeredgecolor="black", markeredgewidth=0.5)

        ax_candle.plot([entry_pos, exit_pos], [t["entry_price"], t["exit_price"]],
                       color=color, linewidth=1.0, alpha=0.5, linestyle="--", zorder=3)

    n_trades = len(trades)
    wins = (trades["pnl_pct"] > 0).sum() if n_trades > 0 else 0
    total_pnl = trades["pnl_pct"].sum() * 100 if n_trades > 0 else 0
    wr = wins / n_trades * 100 if n_trades > 0 else 0

    ax_candle.set_title(
        f"{ticker}  |  Best Combo: entry≥0.75 + TP 4×ATR / SL 1×ATR  |  "
        f"Trades={n_trades}  WR={wr:.0f}%  totalPnL={total_pnl:+.1f}%",
        fontsize=13, fontweight="bold",
    )
    ax_candle.set_ylabel("Price", fontsize=11)

    # Probability panel
    if not proba_t.empty:
        proba_positions = []
        p_long_vals = []
        p_short_vals = []
        for _, pr in proba_t.iterrows():
            p = ts_to_pos.get(pr["timestamp"])
            if p is not None:
                proba_positions.append(p)
                p_dir = max(pr["p_long"] + pr["p_short"], 1e-8)
                p_long_vals.append(pr["p_long"] / p_dir)
                p_short_vals.append(pr["p_short"] / p_dir)

        proba_positions = np.array(proba_positions)
        p_long_vals = np.array(p_long_vals)
        p_short_vals = np.array(p_short_vals)

        ax_proba.fill_between(proba_positions, 0.5, p_long_vals,
                               where=p_long_vals > 0.5, color="#4CAF50", alpha=0.4, label="P(long|dir)")
        ax_proba.fill_between(proba_positions, 0.5, p_short_vals,
                               where=p_short_vals > 0.5, color="#F44336", alpha=0.4, label="P(short|dir)")
        ax_proba.plot(proba_positions, p_long_vals, color="#2E7D32", linewidth=0.8, alpha=0.7)
        ax_proba.plot(proba_positions, 1 - p_long_vals, color="#C62828", linewidth=0.8, alpha=0.7)

    ax_proba.axhline(0.75, color="gray", linewidth=0.8, linestyle=":", alpha=0.5, label="Threshold 0.75")
    ax_proba.axhline(0.5, color="gray", linewidth=0.5, linestyle="-", alpha=0.3)
    ax_proba.set_ylabel("P(dir|conditional)", fontsize=11)
    ax_proba.set_ylim(0, 1)
    ax_proba.legend(loc="upper left", fontsize=9)

    # Time axis
    date_idx = pd.DatetimeIndex(raw["timestamp"])
    tick_pos, tick_labels = _compute_time_ticks(date_idx, pos, max_ticks=20)
    _apply_time_ticks(ax_proba, tick_pos, tick_labels)

    # Legend for trade markers
    legend_elements = [
        mpatches.Patch(color="#2E7D32", label="Win"),
        mpatches.Patch(color="#C62828", label="Loss"),
        plt.Line2D([0], [0], marker="^", color="gray", markersize=8, linestyle="None", label="Long entry"),
        plt.Line2D([0], [0], marker="v", color="gray", markersize=8, linestyle="None", label="Short entry"),
        plt.Line2D([0], [0], marker="D", color="gray", markersize=7, linestyle="None", label="TP exit"),
        plt.Line2D([0], [0], marker="X", color="gray", markersize=8, linestyle="None", label="SL exit"),
    ]
    ax_candle.legend(handles=legend_elements, loc="upper left", fontsize=9, ncol=3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    trades_df = pd.read_parquet(SWEEP_DIR / "best_combo_trades.parquet")
    proba = pd.read_parquet(PROBA_PATH)
    proba = proba[proba["split"] == "test"].copy()
    proba["timestamp"] = pd.to_datetime(proba["timestamp"], utc=True)

    # Need to map entry_idx/exit_idx back to timestamps per ticker
    all_tickers = BEST_TICKERS + WORST_TICKERS
    for ticker in all_tickers:
        logger.info("Plotting %s...", ticker)
        try:
            raw = load_raw(ticker)
        except FileNotFoundError:
            logger.warning("No raw data for %s", ticker)
            continue

        ticker_trades = trades_df[trades_df["ticker"] == ticker].copy()
        if ticker_trades.empty:
            logger.warning("No trades for %s", ticker)
            continue

        # Map integer indices back to timestamps and prices
        ticker_trades["entry_time"] = ticker_trades["entry_idx"].map(
            lambda idx: raw.iloc[idx]["timestamp"] if idx < len(raw) else pd.NaT
        )
        ticker_trades["exit_time"] = ticker_trades["exit_idx"].map(
            lambda idx: raw.iloc[idx]["timestamp"] if idx < len(raw) else pd.NaT
        )
        ticker_trades = ticker_trades.dropna(subset=["entry_time", "exit_time"])

        proba_t = proba[proba["ticker"] == ticker]

        label = "best" if ticker in BEST_TICKERS else "worst"
        out_path = PLOTS_DIR / f"trades_{label}_{ticker}.png"
        plot_ticker_trades(ticker, raw, ticker_trades, proba_t, out_path)

    logger.info("All plots saved → %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
