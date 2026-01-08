import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Data.retrieve_data import normalize_ticker


def plot_continuation_signals(
    df: pd.DataFrame,
    *,
    long_cont_col: str = "long_cont_label",
    short_cont_col: str = "short_cont_label",
    cont_label_col: str = "atr_cont_label",
    save_path: str | None = None,
) -> None:
    """
    Plot OHLC candles with continuation labels only.
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
    cont_offset = marker_offset * 0.4

    up = close_y >= open_y
    down = ~up
    wick_color = "#444444"
    up_color = "#1976D2"
    down_color = "#E53935"
    cont_long_color = "#7CB342"
    cont_short_color = "#F9A825"

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

    cont_long_mask = None
    cont_short_mask = None
    if long_cont_col in df.columns:
        cont_long_mask = df[long_cont_col].fillna(0).astype(int).to_numpy() == 1
    if short_cont_col in df.columns:
        cont_short_mask = df[short_cont_col].fillna(0).astype(int).to_numpy() == 1
    if cont_long_mask is None or cont_short_mask is None:
        if cont_label_col in df.columns:
            cont_vals = df[cont_label_col].fillna(0).to_numpy()
            cont_long_mask = cont_vals == 1
            cont_short_mask = cont_vals == -1

    if cont_long_mask is not None and cont_long_mask.any():
        ax.scatter(
            pos[cont_long_mask],
            close_y[cont_long_mask] + cont_offset,
            color=cont_long_color,
            marker="^",
            s=40,
            label=long_cont_col,
            alpha=0.9,
            zorder=2,
        )
    if cont_short_mask is not None and cont_short_mask.any():
        ax.scatter(
            pos[cont_short_mask],
            close_y[cont_short_mask] - cont_offset,
            color=cont_short_color,
            marker="v",
            s=40,
            label=short_cont_col,
            alpha=0.9,
            zorder=2,
        )

    ax.set_title("Close with continuation labels", fontsize=14)
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
        "Close Price with Continuation Labels - Last Year",
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
        "continuation_plot.png" if slug == "spy" else f"{slug}_continuation_plot.png"
    )
    return plots_dir / filename
