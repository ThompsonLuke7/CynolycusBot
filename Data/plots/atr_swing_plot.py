import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from Data.retrieve_data import normalize_ticker


def _infer_bar_label(index: pd.DatetimeIndex) -> str:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return "Unknown bars"
    deltas = index.to_series().diff().dropna().dt.total_seconds()
    if deltas.empty:
        return "Unknown bars"
    seconds = float(deltas.median())
    if not np.isfinite(seconds) or seconds <= 0:
        return "Unknown bars"
    if seconds % 86400 == 0:
        days = int(seconds / 86400)
        return f"{days} day bars"
    if seconds % 3600 == 0:
        hours = int(seconds / 3600)
        return f"{hours} hour bars"
    if seconds % 60 == 0:
        minutes = int(seconds / 60)
        return f"{minutes} min bars"
    return f"{int(seconds)} sec bars"


def plot_atr_swing_signals(df: pd.DataFrame, save_path: str | None = None) -> None:
    """
    Plot OHLC candles with ATR swing labels from label generation.
    Uses compressed x positions to avoid gaps from non-trading days.
    """
    fig, ax = plt.subplots(figsize=(18, 6))

    date_index = df.index
    pos = np.arange(len(df))  # compressed positions
    open_y = df["open"].to_numpy()
    high_y = df["high"].to_numpy()
    low_y = df["low"].to_numpy()
    close_y = df["close"].to_numpy()
    if "atr" in df.columns:
        marker_offset = np.nanmedian(df["atr"].to_numpy())
    else:
        marker_offset = np.nanmedian(high_y - low_y)
    if not np.isfinite(marker_offset) or marker_offset <= 0:
        marker_offset = np.nanmax(high_y) * 0.001
    up_offset = marker_offset * 0.6
    down_offset = marker_offset * 0.6
    up = close_y >= open_y
    down = ~up
    wick_color = "#444444"
    up_color = "#1976D2"
    down_color = "#E53935"
    swing_long_color = "#2E7D32"
    swing_short_color = "#C62828"
    pivot_dn_color = "#00695C"
    pivot_up_color = "#6A1B9A"

    width = 0.8

    # Wick lines
    ax.vlines(pos, low_y, high_y, color=wick_color, linewidth=1.0, zorder=1)
    # Candle bodies
    ax.bar(
        pos[up],
        close_y[up] - open_y[up],
        width=width,
        bottom=open_y[up],
        color=up_color,
        edgecolor="none",
        label="Bull candle",
        zorder=1.2,
    )
    ax.bar(
        pos[down],
        close_y[down] - open_y[down],
        width=width,
        bottom=open_y[down],
        color=down_color,
        edgecolor="none",
        label="Bear candle",
        zorder=1.2,
    )

    swing_pos_y = low_y - down_offset
    swing_neg_y = high_y + up_offset
    if "atr_swing_label" in df.columns:
        mask_pos = (df["atr_swing_label"].fillna(0) == 1).to_numpy()
        mask_neg = (df["atr_swing_label"].fillna(0) == -1).to_numpy()
    else:
        mask_pos = (
            df["long_swing_label"].fillna(0).astype(int).to_numpy() == 1
            if "long_swing_label" in df.columns
            else np.zeros(len(df), dtype=bool)
        )
        mask_neg = (
            df["short_swing_label"].fillna(0).astype(int).to_numpy() == 1
            if "short_swing_label" in df.columns
            else np.zeros(len(df), dtype=bool)
        )

    if mask_pos.any():
        ax.scatter(
            pos[mask_pos],
            swing_pos_y[mask_pos],
            color=swing_long_color,
            marker="^",
            s=48,
            label="atr_swing_label = +1",
            alpha=0.95,
            zorder=2,
        )
    if mask_neg.any():
        ax.scatter(
            pos[mask_neg],
            swing_neg_y[mask_neg],
            color=swing_short_color,
            marker="v",
            s=48,
            label="atr_swing_label = -1",
            alpha=0.95,
            zorder=2,
        )

    if "pivot_down" in df.columns:
        mask_pivot_down = (df["pivot_down"].fillna(0).astype(int) == 1).to_numpy()
        if mask_pivot_down.any():
            ax.scatter(
                pos[mask_pivot_down],
                low_y[mask_pivot_down] - down_offset * 1.2,
                facecolors="none",
                edgecolors=pivot_dn_color,
                marker="v",
                s=56,
                label="pivot_down",
                alpha=0.95,
                zorder=2.1,
            )

    if "pivot_up" in df.columns:
        mask_pivot_up = (df["pivot_up"].fillna(0).astype(int) == 1).to_numpy()
        if mask_pivot_up.any():
            ax.scatter(
                pos[mask_pivot_up],
                high_y[mask_pivot_up] + up_offset * 1.2,
                facecolors="none",
                edgecolors=pivot_up_color,
                marker="^",
                s=56,
                label="pivot_up",
                alpha=0.95,
                zorder=2.1,
            )

    bar_label = _infer_bar_label(date_index)
    pos_count = int(mask_pos.sum())
    neg_count = int(mask_neg.sum())
    title = (
        f"{bar_label} | bars: {len(df)} | +1: {pos_count} | -1: {neg_count}"
    )
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Close Price")
    ax.legend(loc="upper left", fontsize=11, ncol=3)
    ax.set_xlabel("Date")

    dates = pd.Series(date_index)
    day_start = dates.dt.normalize().ne(dates.dt.normalize().shift())
    tick_positions = pos[day_start.to_numpy()]
    tick_labels = dates[day_start].dt.strftime("%Y-%m-%d").to_list()
    if len(tick_positions) > 25:
        step = int(np.ceil(len(tick_positions) / 25))
        tick_positions = tick_positions[::step]
        tick_labels = tick_labels[::step]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)

    # Light vertical lines for each new day to improve readability
    for x in tick_positions:
        ax.axvline(
            x, color="#d0d0d000", linestyle="--", linewidth=1, alpha=0.7, zorder=0.5
        )

    plt.tight_layout()
    plt.suptitle("Close Price with ATR Swing Labels", fontsize=17, y=1.02)
    plt.subplots_adjust(top=0.93)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved plot to {save_path}")
    plt.show()


def get_default_plot_path(ticker: str, data_dir: Path) -> Path:
    slug = normalize_ticker(ticker).lower()
    plots_dir = data_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    filename = "atr_swing_plot.png" if slug == "spy" else f"{slug}_atr_swing_plot.png"
    return plots_dir / filename
