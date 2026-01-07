import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Data.retrieve_data import normalize_ticker


def plot_leg_segmentation_signals(
    df: pd.DataFrame, save_path: str | None = None
) -> None:
    """
    Plot OHLC candles with ATR leg-state labels and pivot markers.
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

    leg_up_color = "#2E7D32"
    leg_down_color = "#C62828"
    leg_chop_color = "#757575"
    pivot_low_color = "#2E7D32"
    pivot_high_color = "#1E0D32"

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

    leg_up_mask = None
    leg_down_mask = None
    leg_chop_mask = None

    if "leg_up_label" in df.columns:
        leg_up_mask = (df["leg_up_label"].fillna(0).astype(int) == 1).to_numpy()
    if "leg_down_label" in df.columns:
        leg_down_mask = (df["leg_down_label"].fillna(0).astype(int) == 1).to_numpy()
    if leg_up_mask is None or leg_down_mask is None:
        if "atr_leg_label" in df.columns:
            leg_vals = df["atr_leg_label"].fillna(0).to_numpy()
            leg_up_mask = leg_vals == 1
            leg_down_mask = leg_vals == -1
            leg_chop_mask = leg_vals == 0
    if leg_chop_mask is None and leg_up_mask is not None and leg_down_mask is not None:
        leg_chop_mask = ~(leg_up_mask | leg_down_mask)

    if leg_up_mask is not None and leg_up_mask.any():
        ax.scatter(
            pos[leg_up_mask],
            close_y[leg_up_mask] + up_offset,
            color=leg_up_color,
            marker="o",
            s=36,
            label="leg_up_label",
            alpha=0.9,
            zorder=2,
        )
    if leg_down_mask is not None and leg_down_mask.any():
        ax.scatter(
            pos[leg_down_mask],
            close_y[leg_down_mask] - down_offset,
            color=leg_down_color,
            marker="o",
            s=36,
            label="leg_down_label",
            alpha=0.9,
            zorder=2,
        )
    if leg_chop_mask is not None and leg_chop_mask.any():
        ax.scatter(
            pos[leg_chop_mask],
            close_y[leg_chop_mask],
            facecolors="none",
            edgecolors=leg_chop_color,
            marker="o",
            s=28,
            label="leg_chop",
            alpha=0.85,
            zorder=2,
        )
    if "atr_leg_pivot_type" in df.columns:
        pivot_type = df["atr_leg_pivot_type"].fillna(0).to_numpy()
        pivot_price = (
            df["atr_leg_pivot_price"].to_numpy()
            if "atr_leg_pivot_price" in df.columns
            else np.full(len(df), np.nan)
        )
        mask_low = pivot_type == 1
        mask_high = pivot_type == -1

        pivot_low_y = np.where(
            np.isfinite(pivot_price), pivot_price - down_offset * 1.2, low_y
        )
        pivot_high_y = np.where(
            np.isfinite(pivot_price), pivot_price + up_offset * 1.2, high_y
        )

        if mask_low.any():
            ax.scatter(
                pos[mask_low],
                pivot_low_y[mask_low],
                color=pivot_low_color,
                marker="v",
                s=54,
                label="atr_leg_pivot_low",
                alpha=0.95,
                zorder=2.2,
            )
        if mask_high.any():
            ax.scatter(
                pos[mask_high],
                pivot_high_y[mask_high],
                color=pivot_high_color,
                marker="^",
                s=54,
                label="atr_leg_pivot_high",
                alpha=0.95,
                zorder=2.2,
            )

    ax.set_title("Close with ATR leg-state labels and pivots", fontsize=14)
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
        "Close Price with ATR Leg Segmentation - Last Year",
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
        "leg_segmentation_plot.png"
        if slug == "spy"
        else f"{slug}_leg_segmentation_plot.png"
    )
    return plots_dir / filename
