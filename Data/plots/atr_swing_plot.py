import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from Data.retrieve_data import normalize_ticker


def plot_atr_swing_signals(df: pd.DataFrame, save_path: str | None = None) -> None:
    """
    Plot OHLC candles with ATR swing labels, pivots, and flip markers for a quick visual check.
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
    cont_offset = marker_offset * 0.4

    up = close_y >= open_y
    down = ~up
    wick_color = "#444444"
    up_color = "#1976D2"
    down_color = "#E53935"
    cont_long_color = "#7CB342"
    cont_short_color = "#F9A825"

    width = 0.8

    # Marker positions
    pivot_up_y = high_y + up_offset * 1.2
    pivot_dn_y = low_y - down_offset * 1.2
    swing_pos_y = close_y + up_offset
    swing_neg_y = close_y - down_offset

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

    if "atr_swing_label" in df.columns:
        mask_pos = (df["atr_swing_label"].fillna(0) == 1).to_numpy()
        mask_neg = (df["atr_swing_label"].fillna(0) == -1).to_numpy()
        pos_idx = pos[mask_pos]
        neg_idx = pos[mask_neg]
        # Align with pivots when both occur
        if "pivot_down" in df.columns:
            mask_pivot_down = (df["pivot_down"].fillna(0).astype(int) == 1).to_numpy()
            coincide = mask_pos & mask_pivot_down
            swing_pos_y[coincide] = pivot_dn_y[coincide]
        if "pivot_up" in df.columns:
            mask_pivot_up = (df["pivot_up"].fillna(0).astype(int) == 1).to_numpy()
            coincide = mask_neg & mask_pivot_up
            swing_neg_y[coincide] = pivot_up_y[coincide]
        if len(pos_idx) > 0:
            ax.scatter(
                pos_idx,
                swing_pos_y[mask_pos],
                color="#1976D2",
                marker="^",
                s=42,
                label="atr_swing_label = +1",
                alpha=0.96,
                zorder=2,
            )
        if len(neg_idx) > 0:
            ax.scatter(
                neg_idx,
                swing_neg_y[mask_neg],
                color="#E53935",
                marker="v",
                s=42,
                label="atr_swing_label = -1",
                alpha=0.96,
                zorder=2,
            )

    if "pivot_down" in df.columns:
        mask_pivot_down = (df["pivot_down"].fillna(0).astype(int) == 1).to_numpy()
        pivot_idx = pos[mask_pivot_down]
        if len(pivot_idx) > 0:
            ax.scatter(
                pivot_idx,
                pivot_dn_y[mask_pivot_down],
                color="#2E7D32",
                marker="v",
                s=52,
                label="pivot_down",
                alpha=0.95,
                zorder=2.2,
            )

    if "pivot_up" in df.columns:
        mask_pivot_up = (df["pivot_up"].fillna(0).astype(int) == 1).to_numpy()
        pivot_idx = pos[mask_pivot_up]
        if len(pivot_idx) > 0:
            ax.scatter(
                pivot_idx,
                pivot_up_y[mask_pivot_up],
                color="#1E0D32",
                marker="^",
                s=52,
                label="pivot_up",
                alpha=0.95,
                zorder=2.2,
            )

    # Continuation labels (if present)
    if "long_cont_label" in df.columns or "short_cont_label" in df.columns:
        cont_long = (
            df["long_cont_label"].fillna(0).astype(int).to_numpy()
            if "long_cont_label" in df.columns
            else np.zeros(len(df), dtype=int)
        )
        cont_short = (
            df["short_cont_label"].fillna(0).astype(int).to_numpy()
            if "short_cont_label" in df.columns
            else np.zeros(len(df), dtype=int)
        )
        long_idx = pos[cont_long == 1]
        short_idx = pos[cont_short == 1]
        if len(long_idx) > 0:
            ax.scatter(
                long_idx,
                close_y[cont_long == 1] + cont_offset,
                color=cont_long_color,
                marker="o",
                s=38,
                label="long_cont_label",
                alpha=0.9,
                zorder=2.1,
            )
        if len(short_idx) > 0:
            ax.scatter(
                short_idx,
                close_y[cont_short == 1] - cont_offset,
                color=cont_short_color,
                marker="o",
                s=38,
                label="short_cont_label",
                alpha=0.9,
                zorder=2.1,
            )

    ax.set_title(
        "Close with atr_swing_label (+/-1 markers) & atr_swing_flip (vlines)",
        fontsize=14,
    )
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
    plt.suptitle(
        "Close Price with ATR Swing Label & Flip Points - Last Year",
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
    filename = "atr_swing_plot.png" if slug == "spy" else f"{slug}_atr_swing_plot.png"
    return plots_dir / filename
