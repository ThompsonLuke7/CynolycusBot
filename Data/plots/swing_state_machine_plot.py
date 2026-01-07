import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Data.retrieve_data import normalize_ticker


def plot_swing_state_machine_signals(
    df: pd.DataFrame,
    *,
    long_state_col: str = "p_long_state_gate",
    short_state_col: str = "p_short_state_gate",
    long_pending_col: str = "p_long_pending",
    short_pending_col: str = "p_short_pending",
    save_path: str | None = None,
) -> None:
    """
    Plot OHLC candles with swing state-machine labels.
    Uses compressed x positions to avoid gaps from non-trading days.
    """
    fig, ax = plt.subplots(figsize=(18, 6))

    date_index = df.index
    pos = np.arange(len(df))
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
    long_color = "#2E7D32"
    short_color = "#C62828"
    pending_long_color = "#43A047"
    pending_short_color = "#D32F2F"

    width = 0.8

    ax.vlines(pos, low_y, high_y, color=wick_color, linewidth=1.0, zorder=1)
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

    long_mask = None
    short_mask = None
    pending_long_mask = None
    pending_short_mask = None
    if long_state_col in df.columns:
        long_mask = df[long_state_col].fillna(0).astype(int).to_numpy() == 1
    if short_state_col in df.columns:
        short_mask = df[short_state_col].fillna(0).astype(int).to_numpy() == 1
    if long_pending_col in df.columns:
        pending_long_mask = (
            df[long_pending_col].fillna(0).astype(int).to_numpy() == 1
        )
    if short_pending_col in df.columns:
        pending_short_mask = (
            df[short_pending_col].fillna(0).astype(int).to_numpy() == 1
        )

    if long_mask is None and "p_long_state_gate" in df.columns:
        long_mask = df["p_long_state_gate"].fillna(0).astype(int).to_numpy() == 1
    if short_mask is None and "p_short_state_gate" in df.columns:
        short_mask = df["p_short_state_gate"].fillna(0).astype(int).to_numpy() == 1
    if pending_long_mask is None and "p_long_pending" in df.columns:
        pending_long_mask = df["p_long_pending"].fillna(0).astype(int).to_numpy() == 1
    if pending_short_mask is None and "p_short_pending" in df.columns:
        pending_short_mask = (
            df["p_short_pending"].fillna(0).astype(int).to_numpy() == 1
        )

    if long_mask is not None and long_mask.any():
        ax.scatter(
            pos[long_mask],
            close_y[long_mask] + up_offset,
            color=long_color,
            marker="o",
            s=36,
            label=long_state_col,
            alpha=0.9,
            zorder=2,
        )
    if short_mask is not None and short_mask.any():
        ax.scatter(
            pos[short_mask],
            close_y[short_mask] - down_offset,
            color=short_color,
            marker="o",
            s=36,
            label=short_state_col,
            alpha=0.9,
            zorder=2,
        )
    if pending_long_mask is not None and pending_long_mask.any():
        ax.scatter(
            pos[pending_long_mask],
            close_y[pending_long_mask] + up_offset * 0.6,
            facecolors="none",
            edgecolors=pending_long_color,
            marker="o",
            s=46,
            label=long_pending_col,
            alpha=0.95,
            zorder=2.1,
        )
    if pending_short_mask is not None and pending_short_mask.any():
        ax.scatter(
            pos[pending_short_mask],
            close_y[pending_short_mask] - down_offset * 0.6,
            facecolors="none",
            edgecolors=pending_short_color,
            marker="o",
            s=46,
            label=short_pending_col,
            alpha=0.95,
            zorder=2.1,
        )

    ax.set_title("Close with swing state-machine labels", fontsize=14)
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

    for x in tick_positions:
        ax.axvline(
            x, color="#d0d0d000", linestyle="--", linewidth=1, alpha=0.7, zorder=0.5
        )

    plt.tight_layout()
    plt.suptitle(
        "Close Price with Swing State Machine - Last Year",
        fontsize=17,
        y=1.02,
    )
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
    filename = (
        "swing_state_machine_plot.png"
        if slug == "spy"
        else f"{slug}_swing_state_machine_plot.png"
    )
    return plots_dir / filename
