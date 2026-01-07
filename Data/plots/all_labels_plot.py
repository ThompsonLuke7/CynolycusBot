import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Data.retrieve_data import normalize_ticker


def plot_all_labels(
    df: pd.DataFrame,
    *,
    long_state_col: str = "p_long_state_gate",
    short_state_col: str = "p_short_state_gate",
    long_pending_col: str = "p_long_pending",
    short_pending_col: str = "p_short_pending",
    pivot_down_col: str = "pivot_down",
    pivot_up_col: str = "pivot_up",
    long_cont_col: str = "long_cont_label",
    short_cont_col: str = "short_cont_label",
    leg_label_col: str = "atr_leg_label",
    leg_up_col: str = "leg_up_label",
    leg_down_col: str = "leg_down_label",
    save_path: str | None = None,
) -> None:
    """
    Plot candles with pivot, state machine, continuation labels, plus a leg-state subplot.
    Uses compressed x positions to avoid gaps from non-trading days.
    """
    fig, (ax, ax_leg) = plt.subplots(
        2, 1, figsize=(18, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

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
    cont_offset = marker_offset * 0.4

    up = close_y >= open_y
    down = ~up
    wick_color = "#444444"
    up_color = "#1976D2"
    down_color = "#E53935"

    pivot_dn_color = "#2E7D32"
    pivot_up_color = "#1E0D32"
    long_state_color = "#2E7D32"
    short_state_color = "#C62828"
    pending_long_color = "#43A047"
    pending_short_color = "#D32F2F"
    cont_long_color = "#7CB342"
    cont_short_color = "#F9A825"
    leg_up_color = "#2E7D32"
    leg_down_color = "#C62828"

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

    if pivot_down_col in df.columns:
        mask_pivot_down = (df[pivot_down_col].fillna(0).astype(int) == 1).to_numpy()
        if mask_pivot_down.any():
            ax.scatter(
                pos[mask_pivot_down],
                low_y[mask_pivot_down] - down_offset * 1.2,
                color=pivot_dn_color,
                marker="v",
                s=52,
                label=pivot_down_col,
                alpha=0.95,
                zorder=2.2,
            )

    if pivot_up_col in df.columns:
        mask_pivot_up = (df[pivot_up_col].fillna(0).astype(int) == 1).to_numpy()
        if mask_pivot_up.any():
            ax.scatter(
                pos[mask_pivot_up],
                high_y[mask_pivot_up] + up_offset * 1.2,
                color=pivot_up_color,
                marker="^",
                s=52,
                label=pivot_up_col,
                alpha=0.95,
                zorder=2.2,
            )

    if long_state_col in df.columns:
        mask_long = (df[long_state_col].fillna(0).astype(int) == 1).to_numpy()
        if mask_long.any():
            ax.scatter(
                pos[mask_long],
                close_y[mask_long] + up_offset,
                color=long_state_color,
                marker="o",
                s=36,
                label=long_state_col,
                alpha=0.9,
                zorder=2,
            )

    if short_state_col in df.columns:
        mask_short = (df[short_state_col].fillna(0).astype(int) == 1).to_numpy()
        if mask_short.any():
            ax.scatter(
                pos[mask_short],
                close_y[mask_short] - down_offset,
                color=short_state_color,
                marker="o",
                s=36,
                label=short_state_col,
                alpha=0.9,
                zorder=2,
            )

    if long_pending_col in df.columns:
        mask_pending_long = (
            df[long_pending_col].fillna(0).astype(int).to_numpy() == 1
        )
        if mask_pending_long.any():
            ax.scatter(
                pos[mask_pending_long],
                close_y[mask_pending_long] + up_offset * 0.6,
                facecolors="none",
                edgecolors=pending_long_color,
                marker="o",
                s=46,
                label=long_pending_col,
                alpha=0.95,
                zorder=2.1,
            )

    if short_pending_col in df.columns:
        mask_pending_short = (
            df[short_pending_col].fillna(0).astype(int).to_numpy() == 1
        )
        if mask_pending_short.any():
            ax.scatter(
                pos[mask_pending_short],
                close_y[mask_pending_short] - down_offset * 0.6,
                facecolors="none",
                edgecolors=pending_short_color,
                marker="o",
                s=46,
                label=short_pending_col,
                alpha=0.95,
                zorder=2.1,
            )

    if long_cont_col in df.columns:
        mask_cont_long = (df[long_cont_col].fillna(0).astype(int) == 1).to_numpy()
        if mask_cont_long.any():
            ax.scatter(
                pos[mask_cont_long],
                close_y[mask_cont_long] + cont_offset,
                color=cont_long_color,
                marker="^",
                s=40,
                label=long_cont_col,
                alpha=0.9,
                zorder=2.1,
            )

    if short_cont_col in df.columns:
        mask_cont_short = (df[short_cont_col].fillna(0).astype(int) == 1).to_numpy()
        if mask_cont_short.any():
            ax.scatter(
                pos[mask_cont_short],
                close_y[mask_cont_short] - cont_offset,
                color=cont_short_color,
                marker="v",
                s=40,
                label=short_cont_col,
                alpha=0.9,
                zorder=2.1,
            )

    ax.set_title("Close with pivots, state machine, and continuation labels", fontsize=14)
    ax.set_ylabel("Close Price")
    ax.legend(loc="upper left", fontsize=10, ncol=3)

    leg_vals = None
    if leg_label_col in df.columns:
        leg_vals = df[leg_label_col].fillna(0).to_numpy(dtype=float)
    elif leg_up_col in df.columns or leg_down_col in df.columns:
        leg_vals = np.zeros(len(df), dtype=float)
        if leg_up_col in df.columns:
            leg_vals[df[leg_up_col].fillna(0).astype(int).to_numpy() == 1] = 1.0
        if leg_down_col in df.columns:
            leg_vals[df[leg_down_col].fillna(0).astype(int).to_numpy() == 1] = -1.0

    if leg_vals is not None:
        ax_leg.step(
            pos,
            leg_vals,
            where="post",
            color="#455A64",
            linewidth=1.4,
            label=leg_label_col if leg_label_col in df.columns else "leg_state",
        )
        up_leg_mask = leg_vals == 1
        down_leg_mask = leg_vals == -1
        if up_leg_mask.any():
            ax_leg.scatter(
                pos[up_leg_mask],
                leg_vals[up_leg_mask],
                color=leg_up_color,
                s=14,
                alpha=0.9,
                label="leg_up",
            )
        if down_leg_mask.any():
            ax_leg.scatter(
                pos[down_leg_mask],
                leg_vals[down_leg_mask],
                color=leg_down_color,
                s=14,
                alpha=0.9,
                label="leg_down",
            )
        ax_leg.set_yticks([-1, 0, 1])
        ax_leg.axhline(0, color="#999999", linewidth=0.8)
        ax_leg.set_ylabel("Leg")
        ax_leg.legend(loc="upper left", fontsize=10, ncol=3)

    ax_leg.set_xlabel("Date")

    dates = pd.Series(date_index)
    day_start = dates.dt.normalize().ne(dates.dt.normalize().shift())
    tick_positions = pos[day_start.to_numpy()]
    tick_labels = dates[day_start].dt.strftime("%Y-%m-%d").to_list()
    if len(tick_positions) > 25:
        step = int(np.ceil(len(tick_positions) / 25))
        tick_positions = tick_positions[::step]
        tick_labels = tick_labels[::step]
    ax_leg.set_xticks(tick_positions)
    ax_leg.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)

    for x in tick_positions:
        ax.axvline(
            x, color="#d0d0d000", linestyle="--", linewidth=1, alpha=0.7, zorder=0.5
        )
        ax_leg.axvline(
            x, color="#d0d0d000", linestyle="--", linewidth=1, alpha=0.7, zorder=0.5
        )

    plt.tight_layout()
    plt.suptitle("All Labels Overview - Last Year", fontsize=17, y=1.02)
    plt.subplots_adjust(top=0.92)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved plot to {save_path}")
    plt.show()


def get_default_plot_path(ticker: str, data_dir: Path) -> Path:
    slug = normalize_ticker(ticker).lower()
    plots_dir = data_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    filename = "all_labels_plot.png" if slug == "spy" else f"{slug}_all_labels_plot.png"
    return plots_dir / filename
