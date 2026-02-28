from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Data.retrieve_data import normalize_ticker

if TYPE_CHECKING:
    from xgboost import XGBClassifier


DEFAULT_LABEL_PLOT_TYPES = [
    "atr_swing",
    "leg_segmentation",
    "continuation",
    "swing_state_machine",
    "triple_barrier",
    "all_labels",
]

_PLOT_TYPE_ALIASES = {
    "atr": "atr_swing",
    "leg": "leg_segmentation",
    "state_machine": "swing_state_machine",
    "swing_state": "swing_state_machine",
    "mfe": "mfe_mae",
    "mae": "mfe_mae",
    "exhaustion": "bars_to_exhaustion",
    "tb": "triple_barrier",
    "triple": "triple_barrier",
    "trend": "trend_phase",
    "phase": "trend_phase",
    "meta": "meta_entry",
    "entry": "meta_entry",
    "meta_labels": "meta_entry",
    "meta_exit": "meta_exit",
    "hazard": "meta_exit",
    "hazard_exit": "meta_exit",
    "exit": "meta_exit",
}

_LABEL_PLOT_FILES = {
    "atr_swing": "atr_swing_plot.png",
    "leg_segmentation": "leg_segmentation_plot.png",
    "continuation": "continuation_plot.png",
    "swing_state_machine": "swing_state_machine_plot.png",
    "triple_barrier": "triple_barrier_plot.png",
    "all_labels": "all_labels_plot.png",
    "mfe_mae": "mfe_mae_plot.png",
    "bars_to_exhaustion": "bars_to_exhaustion_plot.png",
    "trend_phase": "trend_phase_plot.png",
    "meta_entry": "meta_entry_plot.png",
    "meta_exit": "meta_exit_plot.png",
}

_PLOT_TAIL_BARS = 200

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


def _normalize_plot_type(plot_type: str) -> str:
    key = plot_type.strip().lower()
    key = _PLOT_TYPE_ALIASES.get(key, key)
    if key not in _LABEL_PLOT_FILES:
        raise ValueError(
            f"Unknown plot_type '{plot_type}'. Expected one of: "
            f"{', '.join(sorted(_LABEL_PLOT_FILES))}."
        )
    return key


def _select_plot_window(
    df: pd.DataFrame,
    *,
    window: int | None = _PLOT_TAIL_BARS,
    random_window: bool = False,
    seed: int | None = None,
) -> pd.DataFrame:
    if window is None or window <= 0 or len(df) <= window:
        return df
    if not random_window:
        return df.tail(window)
    rng = np.random.default_rng(seed)
    max_start = len(df) - window
    start = int(rng.integers(0, max_start + 1))
    end = start + window
    return df.iloc[start:end]


def _extract_ohlc(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos = np.arange(len(df))
    open_y = df["open"].to_numpy()
    high_y = df["high"].to_numpy()
    low_y = df["low"].to_numpy()
    close_y = df["close"].to_numpy()
    return pos, open_y, high_y, low_y, close_y


def _compute_marker_offset(
    df: pd.DataFrame,
    high_y: np.ndarray,
    low_y: np.ndarray,
    *,
    atr_col: str = "atr",
    fallback_scale: float = 0.001,
) -> float:
    if atr_col in df.columns:
        marker_offset = np.nanmedian(df[atr_col].to_numpy())
    else:
        marker_offset = np.nanmedian(high_y - low_y)
    if not np.isfinite(marker_offset) or marker_offset <= 0:
        marker_offset = np.nanmax(high_y) * fallback_scale
    return marker_offset


def _plot_candles(
    ax: plt.Axes,
    pos: np.ndarray,
    open_y: np.ndarray,
    high_y: np.ndarray,
    low_y: np.ndarray,
    close_y: np.ndarray,
    *,
    wick_color: str = "#444444",
    up_color: str = "#1976D2",
    down_color: str = "#E53935",
    width: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    up = close_y >= open_y
    down = ~up
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
    return up, down


def _compute_time_ticks(
    date_index: pd.DatetimeIndex,
    pos: np.ndarray,
    *,
    max_ticks: int = 25,
) -> tuple[np.ndarray | None, list[str] | None]:
    if not isinstance(date_index, pd.DatetimeIndex):
        return None, None
    dates = pd.Series(date_index)
    day_start = dates.dt.normalize().ne(dates.dt.normalize().shift())
    tick_positions = pos[day_start.to_numpy()]
    tick_labels = dates[day_start].dt.strftime("%Y-%m-%d").to_list()
    if len(tick_positions) > max_ticks:
        step = int(np.ceil(len(tick_positions) / max_ticks))
        tick_positions = tick_positions[::step]
        tick_labels = tick_labels[::step]
    return tick_positions, tick_labels


def _apply_time_ticks(
    ax: plt.Axes,
    tick_positions: np.ndarray | None,
    tick_labels: list[str] | None,
) -> None:
    if tick_positions is None or tick_labels is None:
        return
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)


def _draw_day_lines(
    axes: Sequence[plt.Axes],
    tick_positions: np.ndarray | None,
    *,
    line_color: str = "#d0d0d000",
) -> None:
    if tick_positions is None:
        return
    for ax in axes:
        for x in tick_positions:
            ax.axvline(
                x, color=line_color, linestyle="--", linewidth=1, alpha=0.7, zorder=0.5
            )


def _finalize_plot(
    fig: plt.Figure,
    *,
    suptitle: str | None = None,
    suptitle_y: float = 1.02,
    top: float = 0.93,
    save_path: str | Path | None = None,
) -> None:
    plt.tight_layout()
    if suptitle:
        plt.suptitle(suptitle, fontsize=17, y=suptitle_y)
        plt.subplots_adjust(top=top)
    if save_path:
        save_path = str(save_path)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_atr_swing_signals(
    df: pd.DataFrame,
    *,
    tail: int | None = _PLOT_TAIL_BARS,
    random_window: bool = False,
    seed: int | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot OHLC candles with ATR swing labels from label generation.
    Uses compressed x positions to avoid gaps from non-trading days.
    """
    df = _select_plot_window(df, window=tail, random_window=random_window, seed=seed)
    fig, ax = plt.subplots(figsize=(18, 6))

    date_index = df.index
    pos, open_y, high_y, low_y, close_y = _extract_ohlc(df)
    marker_offset = _compute_marker_offset(df, high_y, low_y)
    up_offset = marker_offset * 0.6
    down_offset = marker_offset * 0.6

    swing_long_color = "#2E7D32"
    swing_short_color = "#C62828"
    pivot_dn_color = "#00695C"
    pivot_up_color = "#6A1B9A"

    _plot_candles(ax, pos, open_y, high_y, low_y, close_y)

    swing_pos_y = low_y - down_offset
    swing_neg_y = high_y + up_offset

    mask_pos = np.zeros(len(df), dtype=bool)
    mask_neg = np.zeros(len(df), dtype=bool)
    if "atr_swing_label" in df.columns:
        mask_pos = (df["atr_swing_label"].fillna(0) == 1).to_numpy()
        mask_neg = (df["atr_swing_label"].fillna(0) == -1).to_numpy()
    else:
        if "long_swing_label" in df.columns:
            mask_pos = df["long_swing_label"].fillna(0).astype(int).to_numpy() == 1
        if "short_swing_label" in df.columns:
            mask_neg = df["short_swing_label"].fillna(0).astype(int).to_numpy() == 1

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
    title = (
        f"{bar_label} | bars: {len(df)} | +1: {int(mask_pos.sum())} | -1: {int(mask_neg.sum())}"
    )
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Close Price")
    ax.legend(loc="upper left", fontsize=11, ncol=3)
    ax.set_xlabel("Date")

    tick_positions, tick_labels = _compute_time_ticks(date_index, pos)
    _apply_time_ticks(ax, tick_positions, tick_labels)
    _draw_day_lines([ax], tick_positions)

    _finalize_plot(
        fig,
        suptitle="Close Price with ATR Swing Labels",
        suptitle_y=1.02,
        top=0.93,
        save_path=save_path,
    )


def plot_continuation_signals(
    df: pd.DataFrame,
    *,
    long_cont_col: str = "long_cont_label",
    short_cont_col: str = "short_cont_label",
    cont_label_col: str = "atr_cont_label",
    show_pivots: bool = False,
    pivot_down_col: str = "pivot_down",
    pivot_up_col: str = "pivot_up",
    tail: int | None = _PLOT_TAIL_BARS,
    random_window: bool = False,
    seed: int | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot OHLC candles with continuation labels only.
    Uses compressed x positions to avoid gaps from non-trading days.
    """
    df = _select_plot_window(df, window=tail, random_window=random_window, seed=seed)
    fig, ax = plt.subplots(figsize=(18, 6))

    date_index = df.index
    pos, open_y, high_y, low_y, close_y = _extract_ohlc(df)

    marker_offset = _compute_marker_offset(df, high_y, low_y)
    cont_offset = marker_offset * 0.4

    cont_long_color = "#7CB342"
    cont_short_color = "#F9A825"
    pivot_dn_color = "#2E7D32"
    pivot_up_color = "#1E0D32"

    _plot_candles(ax, pos, open_y, high_y, low_y, close_y)

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

    if show_pivots:
        if pivot_down_col in df.columns:
            mask_pivot_down = (df[pivot_down_col].fillna(0).astype(int) == 1).to_numpy()
            if mask_pivot_down.any():
                ax.scatter(
                    pos[mask_pivot_down],
                    low_y[mask_pivot_down] - cont_offset * 1.6,
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
                    high_y[mask_pivot_up] + cont_offset * 1.6,
                    color=pivot_up_color,
                    marker="^",
                    s=52,
                    label=pivot_up_col,
                    alpha=0.95,
                    zorder=2.2,
                )

    ax.set_title("Close with continuation labels", fontsize=14)
    ax.set_ylabel("Close Price")
    ax.legend(loc="upper left", fontsize=11, ncol=3)
    ax.set_xlabel("Date")

    tick_positions, tick_labels = _compute_time_ticks(date_index, pos)
    _apply_time_ticks(ax, tick_positions, tick_labels)
    _draw_day_lines([ax], tick_positions)

    _finalize_plot(
        fig,
        suptitle="Close Price with Continuation Labels - Last Year",
        suptitle_y=1.02,
        top=0.93,
        save_path=save_path,
    )


def plot_leg_segmentation_signals(
    df: pd.DataFrame,
    *,
    tail: int | None = _PLOT_TAIL_BARS,
    random_window: bool = False,
    seed: int | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot OHLC candles with ATR leg-state labels and pivot markers.
    Uses compressed x positions to avoid gaps from non-trading days.
    """
    df = _select_plot_window(df, window=tail, random_window=random_window, seed=seed)
    fig, ax = plt.subplots(figsize=(18, 6))

    date_index = df.index
    pos, open_y, high_y, low_y, close_y = _extract_ohlc(df)

    marker_offset = _compute_marker_offset(df, high_y, low_y)
    up_offset = marker_offset * 0.6
    down_offset = marker_offset * 0.6

    leg_up_color = "#2E7D32"
    leg_down_color = "#C62828"
    leg_chop_color = "#757575"
    pivot_low_color = "#2E7D32"
    pivot_high_color = "#1E0D32"

    _plot_candles(ax, pos, open_y, high_y, low_y, close_y)

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

    tick_positions, tick_labels = _compute_time_ticks(date_index, pos)
    _apply_time_ticks(ax, tick_positions, tick_labels)
    _draw_day_lines([ax], tick_positions)

    _finalize_plot(
        fig,
        suptitle="Close Price with ATR Leg Segmentation - Last Year",
        suptitle_y=1.02,
        top=0.93,
        save_path=save_path,
    )


def plot_swing_state_machine_signals(
    df: pd.DataFrame,
    *,
    long_state_col: str = "p_long_state_gate",
    short_state_col: str = "p_short_state_gate",
    long_pending_col: str = "p_long_pending",
    short_pending_col: str = "p_short_pending",
    tail: int | None = _PLOT_TAIL_BARS,
    random_window: bool = False,
    seed: int | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot OHLC candles with swing state-machine labels.
    Uses compressed x positions to avoid gaps from non-trading days.
    """
    df = _select_plot_window(df, window=tail, random_window=random_window, seed=seed)
    fig, ax = plt.subplots(figsize=(18, 6))

    date_index = df.index
    pos, open_y, high_y, low_y, close_y = _extract_ohlc(df)

    marker_offset = _compute_marker_offset(df, high_y, low_y)
    up_offset = marker_offset * 0.6
    down_offset = marker_offset * 0.6

    long_color = "#2E7D32"
    short_color = "#C62828"
    chop_color = "#757575"
    pending_long_color = "#43A047"
    pending_short_color = "#D32F2F"

    _plot_candles(ax, pos, open_y, high_y, low_y, close_y)

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

    tick_positions, tick_labels = _compute_time_ticks(date_index, pos)
    _apply_time_ticks(ax, tick_positions, tick_labels)
    _draw_day_lines([ax], tick_positions)

    _finalize_plot(
        fig,
        suptitle="Close Price with Swing State Machine - Last Year",
        suptitle_y=1.02,
        top=0.93,
        save_path=save_path,
    )


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
    tail: int | None = _PLOT_TAIL_BARS,
    random_window: bool = False,
    seed: int | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot candles with pivot, state machine, continuation labels, plus a leg-state subplot.
    Uses compressed x positions to avoid gaps from non-trading days.
    """
    df = _select_plot_window(df, window=tail, random_window=random_window, seed=seed)
    fig, (ax, ax_leg) = plt.subplots(
        2, 1, figsize=(18, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    date_index = df.index
    pos, open_y, high_y, low_y, close_y = _extract_ohlc(df)

    marker_offset = _compute_marker_offset(df, high_y, low_y)
    up_offset = marker_offset * 0.6
    down_offset = marker_offset * 0.6
    cont_offset = marker_offset * 0.4

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

    _plot_candles(ax, pos, open_y, high_y, low_y, close_y)

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

    tick_positions, tick_labels = _compute_time_ticks(date_index, pos)
    _apply_time_ticks(ax_leg, tick_positions, tick_labels)
    _draw_day_lines([ax, ax_leg], tick_positions)

    _finalize_plot(
        fig,
        suptitle="All Labels Overview - Last Year",
        suptitle_y=1.02,
        top=0.92,
        save_path=save_path,
    )


def plot_mfe_mae_labels(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    mfe_col: str = "mfe_up_atr",
    mae_col: str = "mae_down_atr",
    tail: int | None = _PLOT_TAIL_BARS,
    random_window: bool = False,
    seed: int | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot close price with a subplot showing MFE (up) and MAE (down) bars.
    """
    df = _select_plot_window(df, window=tail, random_window=random_window, seed=seed)
    fig, (ax_price, ax_bar) = plt.subplots(
        2, 1, figsize=(18, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )

    date_index = df.index
    pos = np.arange(len(df))
    close_y = df[close_col].to_numpy()

    ax_price.plot(pos, close_y, color="#1f77b4", linewidth=1.6, label="Close")
    ax_price.set_ylabel("Close Price")
    ax_price.legend(loc="upper left")
    ax_price.set_title("Close with MFE/MAE (ATR units)")

    mfe = df[mfe_col].to_numpy(dtype=float) if mfe_col in df.columns else None
    if mae_col in df.columns:
        mae = df[mae_col].to_numpy(dtype=float)
    elif "mfe_down_atr" in df.columns:
        mae = df["mfe_down_atr"].to_numpy(dtype=float)
    else:
        mae = None

    if mfe is None or mae is None:
        raise KeyError(f"Missing required columns: {mfe_col} and/or {mae_col}")

    mfe_plot = np.where(np.isfinite(mfe), mfe, 0.0)
    mae_plot = np.where(np.isfinite(mae), -mae, 0.0)

    ax_bar.bar(
        pos,
        mfe_plot,
        color="#43A047",
        width=0.8,
        alpha=0.7,
        label="MFE (up, ATR)",
    )
    ax_bar.bar(
        pos,
        mae_plot,
        color="#E53935",
        width=0.8,
        alpha=0.6,
        label="MAE (down, ATR)",
    )
    ax_bar.axhline(0, color="#999999", linewidth=0.8)
    ax_bar.set_ylabel("ATR Units")
    ax_bar.legend(loc="upper left", ncol=2)

    tick_positions, tick_labels = _compute_time_ticks(date_index, pos)
    _apply_time_ticks(ax_bar, tick_positions, tick_labels)
    _draw_day_lines([ax_price, ax_bar], tick_positions)

    _finalize_plot(
        fig,
        suptitle="Close with MFE/MAE Bars - Last Year",
        suptitle_y=1.02,
        top=0.92,
        save_path=save_path,
    )


def plot_exhaustion_progress(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    progress_long_col: str = "exhaustion_progress_long",
    progress_short_col: str = "exhaustion_progress_short",
    tail: int | None = _PLOT_TAIL_BARS,
    random_window: bool = False,
    seed: int | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot close price with a subplot showing exhaustion progress (0..1).
    """
    df = _select_plot_window(df, window=tail, random_window=random_window, seed=seed)
    fig, (ax_price, ax_bar) = plt.subplots(
        2, 1, figsize=(18, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )

    date_index = df.index
    pos, open_y, high_y, low_y, close_y = _extract_ohlc(df)

    _plot_candles(ax_price, pos, open_y, high_y, low_y, close_y)
    ax_price.set_ylabel("Close Price")
    ax_price.set_title("Close with Exhaustion Progress")

    if progress_long_col not in df.columns or progress_short_col not in df.columns:
        missing = [c for c in (progress_long_col, progress_short_col) if c not in df.columns]
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")

    prog_long = df[progress_long_col].to_numpy(dtype=float)
    prog_short = df[progress_short_col].to_numpy(dtype=float)

    long_plot = np.where(np.isfinite(prog_long), prog_long, 0.0)
    short_plot = np.where(np.isfinite(prog_short), prog_short, 0.0)
    width = 0.4
    ax_bar.bar(
        pos - width / 2,
        long_plot,
        color="#2E7D32",
        width=width,
        alpha=0.7,
        label="exhaustion_progress_long",
    )
    ax_bar.bar(
        pos + width / 2,
        short_plot,
        color="#C62828",
        width=width,
        alpha=0.7,
        label="exhaustion_progress_short",
    )
    ax_bar.set_ylabel("Progress")
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.legend(loc="upper left", ncol=2)

    tick_positions, tick_labels = _compute_time_ticks(date_index, pos)
    _apply_time_ticks(ax_bar, tick_positions, tick_labels)
    _draw_day_lines([ax_price, ax_bar], tick_positions)

    _finalize_plot(
        fig,
        suptitle="Exhaustion Progress - Last Year",
        suptitle_y=1.02,
        top=0.92,
        save_path=save_path,
    )


def plot_continuation_strength(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    strength_long_col: str = "cont_strength_long",
    strength_short_col: str = "cont_strength_short",
    tail: int | None = _PLOT_TAIL_BARS,
    random_window: bool = False,
    seed: int | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot close price with a subplot showing continuation strength (0..1).
    """
    df = _select_plot_window(df, window=tail, random_window=random_window, seed=seed)
    fig, (ax_price, ax_bar) = plt.subplots(
        2, 1, figsize=(18, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )

    date_index = df.index
    pos, open_y, high_y, low_y, close_y = _extract_ohlc(df)

    _plot_candles(ax_price, pos, open_y, high_y, low_y, close_y)
    ax_price.set_ylabel("Close Price")
    ax_price.set_title("Close with Continuation Strength")

    if strength_long_col not in df.columns or strength_short_col not in df.columns:
        missing = [
            c for c in (strength_long_col, strength_short_col) if c not in df.columns
        ]
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")

    cont_long = df[strength_long_col].to_numpy(dtype=float)
    cont_short = df[strength_short_col].to_numpy(dtype=float)

    def _print_stats(label: str, arr: np.ndarray) -> None:
        total = arr.size
        finite = arr[np.isfinite(arr)]
        nan_pct = 100.0 * (1.0 - (finite.size / max(total, 1)))
        if finite.size == 0:
            print(f"[continuation] {label}: n={total}, nan%={nan_pct:.2f}, no finite values")
            return
        pct0 = float(np.mean(finite == 0.0) * 100.0)
        pct1 = float(np.mean(finite == 1.0) * 100.0)
        p25 = float(np.percentile(finite, 25))
        p50 = float(np.percentile(finite, 50))
        p75 = float(np.percentile(finite, 75))
        print(
            f"[continuation] {label}: n={total}, nan%={nan_pct:.2f}, "
            f"mean={float(np.mean(finite)):.4f}, std={float(np.std(finite)):.4f}, "
            f"min={float(np.min(finite)):.4f}, p25={p25:.4f}, "
            f"median={p50:.4f}, p75={p75:.4f}, max={float(np.max(finite)):.4f}, "
            f"pct0={pct0:.2f}, pct1={pct1:.2f}"
        )

    _print_stats(strength_long_col, cont_long)
    _print_stats(strength_short_col, cont_short)

    long_plot = np.where(np.isfinite(cont_long), cont_long, 0.0)
    short_plot = np.where(np.isfinite(cont_short), cont_short, 0.0)
    width = 0.4
    ax_bar.bar(
        pos - width / 2,
        long_plot,
        color="#7CB342",
        width=width,
        alpha=0.7,
        label=strength_long_col,
    )
    ax_bar.bar(
        pos + width / 2,
        short_plot,
        color="#F9A825",
        width=width,
        alpha=0.7,
        label=strength_short_col,
    )
    ax_bar.set_ylabel("Strength")
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.legend(loc="upper left", ncol=2)

    tick_positions, tick_labels = _compute_time_ticks(date_index, pos)
    _apply_time_ticks(ax_bar, tick_positions, tick_labels)
    _draw_day_lines([ax_price, ax_bar], tick_positions)

    _finalize_plot(
        fig,
        suptitle="Continuation Strength - Last Year",
        suptitle_y=1.02,
        top=0.92,
        save_path=save_path,
    )


def plot_triple_barrier_signals(
    df: pd.DataFrame,
    *,
    label_col: str = "tb_label",
    tail: int | None = _PLOT_TAIL_BARS,
    random_window: bool = False,
    seed: int | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot OHLC candles with triple barrier labels (+1/-1).
    """
    df = _select_plot_window(df, window=tail, random_window=random_window, seed=seed)
    fig, ax = plt.subplots(figsize=(18, 6))

    date_index = df.index
    pos, open_y, high_y, low_y, close_y = _extract_ohlc(df)

    marker_offset = _compute_marker_offset(df, high_y, low_y)
    up_offset = marker_offset * 0.6
    down_offset = marker_offset * 0.6

    long_color = "#2E7D32"
    short_color = "#C62828"

    _plot_candles(ax, pos, open_y, high_y, low_y, close_y)

    if label_col not in df.columns:
        raise KeyError(f"Missing required column: {label_col}")

    labels = df[label_col].fillna(0).to_numpy(dtype=float)
    mask_pos = labels >= 0.5
    mask_neg = labels <= -0.5
    mask_zero = ~(mask_pos | mask_neg)

    if mask_pos.any():
        ax.scatter(
            pos[mask_pos],
            close_y[mask_pos] + up_offset,
            color=long_color,
            marker="^",
            s=42,
            label=f"{label_col} = +1",
            alpha=0.9,
            zorder=2,
        )
    if mask_neg.any():
        ax.scatter(
            pos[mask_neg],
            close_y[mask_neg] - down_offset,
            color=short_color,
            marker="v",
            s=42,
            label=f"{label_col} = -1",
            alpha=0.9,
            zorder=2,
        )
    if mask_zero.any():
        ax.scatter(
            pos[mask_zero],
            close_y[mask_zero],
            color="#757575",
            marker="o",
            s=28,
            label=f"{label_col} = 0",
            alpha=0.7,
            zorder=1.9,
        )

    bar_label = _infer_bar_label(date_index)
    title = (
        f"{bar_label} | bars: {len(df)} | +1: {int(mask_pos.sum())} | -1: {int(mask_neg.sum())}"
    )
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Close Price")
    ax.legend(loc="upper left", fontsize=11, ncol=3)
    ax.set_xlabel("Date")

    tick_positions, tick_labels = _compute_time_ticks(date_index, pos)
    _apply_time_ticks(ax, tick_positions, tick_labels)
    _draw_day_lines([ax], tick_positions)

    _finalize_plot(
        fig,
        suptitle="Close Price with Triple Barrier Labels",
        suptitle_y=1.02,
        top=0.93,
        save_path=save_path,
    )


def plot_trend_phase_signals(
    df: pd.DataFrame,
    *,
    phase_col: str = "trend_phase_label",
    show_phase_labels: bool = False,
    exit_long_col: str = "trend_phase_exit_long",
    exit_short_col: str = "trend_phase_exit_short",
    enter_long_col: str = "y_enter_long",
    enter_short_col: str = "y_enter_short",
    overlay_entries: bool = True,
    show_position_timeline: bool = True,
    side: str = "both",
    tail: int | None = _PLOT_TAIL_BARS,
    random_window: bool = False,
    seed: int | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot momentum-decay labels on OHLC:
      - enter long / enter short
      - decay-exit long / decay-exit short
    Consecutive entry 1s are collapsed to first-in-run markers.
    Optionally draws a position timeline (long/short spans) below price.
    Optional phase overlays can be enabled via show_phase_labels=True.
    """
    side_key = str(side).strip().lower()
    if side_key not in {"both", "long", "short"}:
        raise ValueError("side must be one of: both, long, short")

    df = _select_plot_window(df, window=tail, random_window=random_window, seed=seed)
    if show_position_timeline:
        fig, (ax, ax_pos) = plt.subplots(
            2,
            1,
            figsize=(18, 7.3),
            sharex=True,
            gridspec_kw={"height_ratios": [5.0, 1.15], "hspace": 0.04},
        )
    else:
        fig, ax = plt.subplots(figsize=(18, 6))
        ax_pos = None

    date_index = df.index
    pos, open_y, high_y, low_y, close_y = _extract_ohlc(df)
    marker_offset = _compute_marker_offset(df, high_y, low_y)

    _plot_candles(ax, pos, open_y, high_y, low_y, close_y)

    counts: dict[int, int] = {}
    if show_phase_labels and phase_col in df.columns:
        phase = pd.Series(df[phase_col], index=df.index).fillna(0).astype(int).to_numpy()
        phase_defs = (
            (0, "dead/chop", "#757575", "o", 26, 0.65),
            (1, "ignition", "#1E88E5", "^", 40, 0.9),
            (2, "expansion", "#2E7D32", "s", 36, 0.9),
            (3, "saturation/decay", "#C62828", "v", 40, 0.9),
        )
        for code, label, color, marker, size, alpha in phase_defs:
            mask = phase == code
            count = int(mask.sum())
            counts[code] = count
            if not count:
                continue
            if code == 0:
                yvals = close_y[mask]
            elif code == 3:
                yvals = close_y[mask] - marker_offset * 0.45
            else:
                yvals = close_y[mask] + marker_offset * 0.45
            ax.scatter(
                pos[mask],
                yvals,
                color=color,
                marker=marker,
                s=size,
                label=f"{label} ({count})",
                alpha=alpha,
                zorder=2.0,
            )

    long_exit_hits = np.zeros(len(df), dtype=bool)
    short_exit_hits = np.zeros(len(df), dtype=bool)
    enter_long_hits = np.zeros(len(df), dtype=bool)
    enter_short_hits = np.zeros(len(df), dtype=bool)

    if exit_long_col in df.columns:
        long_exit_hits = (
            pd.Series(df[exit_long_col], index=df.index).fillna(0).astype(int).to_numpy() == 1
        )
    if exit_short_col in df.columns:
        short_exit_hits = (
            pd.Series(df[exit_short_col], index=df.index).fillna(0).astype(int).to_numpy() == 1
        )

    if side_key == "long":
        short_exit_hits[:] = False
    elif side_key == "short":
        long_exit_hits[:] = False

    if long_exit_hits.any():
        ax.scatter(
            pos[long_exit_hits],
            close_y[long_exit_hits] + marker_offset * 0.75,
            color="#D81B60",
            marker="*",
            s=95,
            label=f"decay-exit long ({int(long_exit_hits.sum())})",
            alpha=0.95,
            zorder=2.1,
        )
    if short_exit_hits.any():
        ax.scatter(
            pos[short_exit_hits],
            close_y[short_exit_hits] - marker_offset * 0.75,
            color="#6A1B9A",
            marker="*",
            s=95,
            label=f"decay-exit short ({int(short_exit_hits.sum())})",
            alpha=0.95,
            zorder=2.1,
        )

    if overlay_entries:
        if enter_long_col in df.columns:
            enter_long_hits = (
                pd.Series(df[enter_long_col], index=df.index).fillna(0).astype(int).to_numpy()
                == 1
            )
        if enter_short_col in df.columns:
            enter_short_hits = (
                pd.Series(df[enter_short_col], index=df.index).fillna(0).astype(int).to_numpy()
                == 1
            )

        if side_key == "long":
            enter_short_hits[:] = False
        elif side_key == "short":
            enter_long_hits[:] = False

        # Collapse runs of repeated 1s to a single visible entry marker.
        enter_long_first = enter_long_hits.copy()
        enter_short_first = enter_short_hits.copy()
        if enter_long_first.size:
            enter_long_first[1:] &= ~enter_long_hits[:-1]
        if enter_short_first.size:
            enter_short_first[1:] &= ~enter_short_hits[:-1]

        if enter_long_first.any():
            ax.scatter(
                pos[enter_long_first],
                close_y[enter_long_first] + marker_offset * 1.02,
                color="#2E7D32",
                marker="^",
                s=32,
                label=f"enter long ({int(enter_long_first.sum())})",
                alpha=0.65,
                zorder=1.9,
            )
        if enter_short_first.any():
            ax.scatter(
                pos[enter_short_first],
                close_y[enter_short_first] - marker_offset * 1.02,
                color="#C62828",
                marker="v",
                s=32,
                label=f"enter short ({int(enter_short_first.sum())})",
                alpha=0.65,
                zorder=1.9,
            )
    else:
        enter_long_first = enter_long_hits
        enter_short_first = enter_short_hits

    def _build_position_spans(entry_first: np.ndarray, exit_hits: np.ndarray) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        in_pos = False
        start = 0
        i = 0
        n = len(entry_first)
        while i < n:
            if not in_pos and entry_first[i]:
                in_pos = True
                start = i
                i += 1
                continue
            if in_pos and exit_hits[i]:
                # For hazard exits, close on the end of the current 1-run (point exit).
                j = i
                while (j + 1) < n and exit_hits[j + 1]:
                    j += 1
                spans.append((start, j))
                in_pos = False
                i = j + 1
                continue
            i += 1
        if in_pos and n:
            spans.append((start, len(entry_first) - 1))
        return spans

    long_spans = _build_position_spans(enter_long_first, long_exit_hits)
    short_spans = _build_position_spans(enter_short_first, short_exit_hits)

    if ax_pos is not None:
        long_label_drawn = False
        short_label_drawn = False
        for s, e in long_spans:
            x0 = pos[s] - 0.42
            x1 = pos[e] + 0.42
            ax_pos.hlines(
                y=1.0,
                xmin=x0,
                xmax=x1,
                color="#2E7D32",
                linewidth=8.5,
                alpha=0.9,
                label="long position" if not long_label_drawn else None,
            )
            long_label_drawn = True
        for s, e in short_spans:
            x0 = pos[s] - 0.42
            x1 = pos[e] + 0.42
            ax_pos.hlines(
                y=0.0,
                xmin=x0,
                xmax=x1,
                color="#C62828",
                linewidth=8.5,
                alpha=0.9,
                label="short position" if not short_label_drawn else None,
            )
            short_label_drawn = True
        ax_pos.set_yticks([0.0, 1.0])
        ax_pos.set_yticklabels(["Short", "Long"], fontsize=9)
        ax_pos.set_ylim(-0.65, 1.65)
        ax_pos.set_ylabel("Position", fontsize=10)
        ax_pos.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.7)
        if long_label_drawn or short_label_drawn:
            ax_pos.legend(loc="upper left", fontsize=9, ncol=2)
        ax_pos.set_xlabel("Date")
        ax.tick_params(axis="x", labelbottom=False)

    bar_label = _infer_bar_label(date_index)
    if show_phase_labels and phase_col in df.columns:
        title = (
            f"{bar_label} | bars: {len(df)} | "
            f"side={side_key} | "
            f"dead={counts.get(0,0)} | ignition={counts.get(1,0)} | "
            f"expansion={counts.get(2,0)} | sat/decay={counts.get(3,0)} | "
            f"enterL={int(enter_long_first.sum())} | enterS={int(enter_short_first.sum())} | "
            f"exitL={int(long_exit_hits.sum())} | exitS={int(short_exit_hits.sum())}"
        )
    else:
        title = (
            f"{bar_label} | bars: {len(df)} | "
            f"side={side_key} | "
            f"enterL={int(enter_long_first.sum())} | enterS={int(enter_short_first.sum())} | "
            f"exitL={int(long_exit_hits.sum())} | exitS={int(short_exit_hits.sum())}"
        )
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Close Price")
    if ax_pos is None:
        ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=10, ncol=3)

    tick_positions, tick_labels = _compute_time_ticks(date_index, pos)
    _apply_time_ticks(ax_pos if ax_pos is not None else ax, tick_positions, tick_labels)
    if ax_pos is not None:
        _draw_day_lines([ax, ax_pos], tick_positions)
    else:
        _draw_day_lines([ax], tick_positions)

    _finalize_plot(
        fig,
        suptitle="Close Price with Momentum-Decay Labels",
        suptitle_y=1.02,
        top=0.93,
        save_path=save_path,
    )


def plot_meta_entry_signals(
    df: pd.DataFrame,
    *,
    long_col: str = "y_enter_long",
    short_col: str = "y_enter_short",
    side: str = "both",
    tail: int | None = _PLOT_TAIL_BARS,
    random_window: bool = False,
    seed: int | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot OHLC candles with meta-entry labels:
      - y_enter_long=1: long entry label
      - y_enter_short=1: short entry label
    """
    side_key = str(side).strip().lower()
    if side_key not in {"both", "long", "short"}:
        raise ValueError("side must be one of: both, long, short")

    df = _select_plot_window(df, window=tail, random_window=random_window, seed=seed)
    fig, ax = plt.subplots(figsize=(18, 6))

    date_index = df.index
    pos, open_y, high_y, low_y, close_y = _extract_ohlc(df)
    marker_offset = _compute_marker_offset(df, high_y, low_y)

    _plot_candles(ax, pos, open_y, high_y, low_y, close_y)

    if long_col not in df.columns:
        raise KeyError(f"Missing required column: {long_col}")
    if short_col not in df.columns:
        raise KeyError(f"Missing required column: {short_col}")

    long_hits = df[long_col].fillna(0).astype(int).to_numpy() == 1
    short_hits = df[short_col].fillna(0).astype(int).to_numpy() == 1
    if side_key == "long":
        short_hits[:] = False
    elif side_key == "short":
        long_hits[:] = False
    both_hits = long_hits & short_hits

    if long_hits.any():
        ax.scatter(
            pos[long_hits],
            close_y[long_hits] + marker_offset * 0.45,
            color="#2E7D32",
            marker="^",
            s=44,
            alpha=0.9,
            zorder=2.0,
            label=f"{long_col}=1 ({int(long_hits.sum())})",
        )
    if short_hits.any():
        ax.scatter(
            pos[short_hits],
            close_y[short_hits] - marker_offset * 0.45,
            color="#C62828",
            marker="v",
            s=44,
            alpha=0.9,
            zorder=2.0,
            label=f"{short_col}=1 ({int(short_hits.sum())})",
        )
    if both_hits.any():
        ax.scatter(
            pos[both_hits],
            close_y[both_hits],
            color="#6A1B9A",
            marker="D",
            s=30,
            alpha=0.75,
            zorder=2.1,
            label=f"both=1 ({int(both_hits.sum())})",
        )

    bar_label = _infer_bar_label(date_index)
    ax.set_title(
        (
            f"{bar_label} | bars: {len(df)} | "
            f"side={side_key} | "
            f"long_entries={int(long_hits.sum())} | short_entries={int(short_hits.sum())}"
        ),
        fontsize=14,
    )
    ax.set_ylabel("Close Price")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=10, ncol=3)

    tick_positions, tick_labels = _compute_time_ticks(date_index, pos)
    _apply_time_ticks(ax, tick_positions, tick_labels)
    _draw_day_lines([ax], tick_positions)

    _finalize_plot(
        fig,
        suptitle="Close Price with Meta Entry Labels",
        suptitle_y=1.02,
        top=0.93,
        save_path=save_path,
    )


def plot_meta_exit_signals(
    df: pd.DataFrame,
    *,
    long_col: str = "y_exit_long",
    short_col: str = "y_exit_short",
    point_long_col: str = "y_exit_long_point",
    point_short_col: str = "y_exit_short_point",
    reason_long_col: str = "exit_reason_long",
    reason_short_col: str = "exit_reason_short",
    enter_long_col: str = "y_enter_long",
    enter_short_col: str = "y_enter_short",
    overlay_entries: bool = True,
    show_hybrid_reasons: bool = True,
    side: str = "both",
    tail: int | None = _PLOT_TAIL_BARS,
    random_window: bool = False,
    seed: int | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot OHLC candles with hybrid hazard-style meta-exit labels:
      - y_exit_long=1: long trade exits within K bars
      - y_exit_short=1: short trade exits within K bars
      - Optional point-exit overlay with reasons: DECAY/TRAIL/SL/EOD
    """
    side_key = str(side).strip().lower()
    if side_key not in {"both", "long", "short"}:
        raise ValueError("side must be one of: both, long, short")

    df = _select_plot_window(df, window=tail, random_window=random_window, seed=seed)
    fig, ax = plt.subplots(figsize=(18, 6))

    date_index = df.index
    pos, open_y, high_y, low_y, close_y = _extract_ohlc(df)
    marker_offset = _compute_marker_offset(df, high_y, low_y)

    _plot_candles(ax, pos, open_y, high_y, low_y, close_y)

    if long_col not in df.columns:
        raise KeyError(f"Missing required column: {long_col}")
    if short_col not in df.columns:
        raise KeyError(f"Missing required column: {short_col}")

    long_hits = df[long_col].fillna(0).astype(int).to_numpy() == 1
    short_hits = df[short_col].fillna(0).astype(int).to_numpy() == 1
    long_point_hits = np.zeros_like(long_hits, dtype=bool)
    short_point_hits = np.zeros_like(short_hits, dtype=bool)
    long_reasons = np.full(len(df), "NONE", dtype=object)
    short_reasons = np.full(len(df), "NONE", dtype=object)
    if point_long_col in df.columns:
        long_point_hits = df[point_long_col].fillna(0).astype(int).to_numpy() == 1
    if point_short_col in df.columns:
        short_point_hits = df[point_short_col].fillna(0).astype(int).to_numpy() == 1
    if reason_long_col in df.columns:
        long_reasons = np.array(
            df[reason_long_col].fillna("NONE").astype(str).str.upper().to_numpy(dtype=object),
            dtype=object,
            copy=True,
        )
    if reason_short_col in df.columns:
        short_reasons = np.array(
            df[reason_short_col].fillna("NONE").astype(str).str.upper().to_numpy(dtype=object),
            dtype=object,
            copy=True,
        )
    if side_key == "long":
        short_hits[:] = False
        short_point_hits[:] = False
        short_reasons[:] = "NONE"
    elif side_key == "short":
        long_hits[:] = False
        long_point_hits[:] = False
        long_reasons[:] = "NONE"
    both_hits = long_hits & short_hits
    enter_long_hits = np.zeros_like(long_hits, dtype=bool)
    enter_short_hits = np.zeros_like(short_hits, dtype=bool)
    if overlay_entries:
        if enter_long_col not in df.columns:
            raise KeyError(f"Missing required column for overlay: {enter_long_col}")
        if enter_short_col not in df.columns:
            raise KeyError(f"Missing required column for overlay: {enter_short_col}")
        enter_long_hits = df[enter_long_col].fillna(0).astype(int).to_numpy() == 1
        enter_short_hits = df[enter_short_col].fillna(0).astype(int).to_numpy() == 1
        if side_key == "long":
            enter_short_hits[:] = False
        elif side_key == "short":
            enter_long_hits[:] = False

    if long_hits.any():
        ax.scatter(
            pos[long_hits],
            close_y[long_hits] + marker_offset * 0.45,
            color="#1565C0",
            marker="s",
            s=40,
            alpha=0.9,
            zorder=2.0,
            label=f"{long_col}=1 ({int(long_hits.sum())})",
        )
    if short_hits.any():
        ax.scatter(
            pos[short_hits],
            close_y[short_hits] - marker_offset * 0.45,
            color="#EF6C00",
            marker="s",
            s=40,
            alpha=0.9,
            zorder=2.0,
            label=f"{short_col}=1 ({int(short_hits.sum())})",
        )
    if both_hits.any():
        ax.scatter(
            pos[both_hits],
            close_y[both_hits],
            color="#6A1B9A",
            marker="D",
            s=30,
            alpha=0.75,
            zorder=2.1,
            label=f"both=1 ({int(both_hits.sum())})",
        )
    if overlay_entries and enter_long_hits.any():
        ax.scatter(
            pos[enter_long_hits],
            close_y[enter_long_hits] + marker_offset * 0.9,
            color="#2E7D32",
            marker="^",
            s=30,
            alpha=0.65,
            zorder=1.8,
            label=f"{enter_long_col}=1 ({int(enter_long_hits.sum())})",
        )
    if overlay_entries and enter_short_hits.any():
        ax.scatter(
            pos[enter_short_hits],
            close_y[enter_short_hits] - marker_offset * 0.9,
            color="#C62828",
            marker="v",
            s=30,
            alpha=0.65,
            zorder=1.8,
            label=f"{enter_short_col}=1 ({int(enter_short_hits.sum())})",
        )

    reason_counts_long = {"DECAY": 0, "TRAIL": 0, "SL": 0, "EOD": 0}
    reason_counts_short = {"DECAY": 0, "TRAIL": 0, "SL": 0, "EOD": 0}
    if show_hybrid_reasons and (long_point_hits.any() or short_point_hits.any()):
        reason_style = {
            "DECAY": {"color": "#8E24AA", "marker": "*"},
            "TRAIL": {"color": "#00897B", "marker": "P"},
            "SL": {"color": "#FB8C00", "marker": "X"},
            "EOD": {"color": "#546E7A", "marker": "D"},
        }
        for reason, style in reason_style.items():
            mask_l = long_point_hits & (long_reasons == reason)
            mask_s = short_point_hits & (short_reasons == reason)
            reason_counts_long[reason] = int(mask_l.sum())
            reason_counts_short[reason] = int(mask_s.sum())
            if mask_l.any():
                ax.scatter(
                    pos[mask_l],
                    close_y[mask_l] + marker_offset * 1.25,
                    color=style["color"],
                    marker=style["marker"],
                    s=62 if reason != "DECAY" else 88,
                    alpha=0.95,
                    zorder=2.15,
                    label=f"long {reason.lower()} ({int(mask_l.sum())})",
                )
            if mask_s.any():
                ax.scatter(
                    pos[mask_s],
                    close_y[mask_s] - marker_offset * 1.25,
                    color=style["color"],
                    marker=style["marker"],
                    s=62 if reason != "DECAY" else 88,
                    alpha=0.95,
                    zorder=2.15,
                    label=f"short {reason.lower()} ({int(mask_s.sum())})",
                )

    bar_label = _infer_bar_label(date_index)
    point_long_count = int(long_point_hits.sum())
    point_short_count = int(short_point_hits.sum())
    if show_hybrid_reasons and (point_long_count or point_short_count):
        reason_text = (
            f"L(decay/trail/sl/eod)="
            f"{reason_counts_long['DECAY']}/{reason_counts_long['TRAIL']}/"
            f"{reason_counts_long['SL']}/{reason_counts_long['EOD']} | "
            f"S(decay/trail/sl/eod)="
            f"{reason_counts_short['DECAY']}/{reason_counts_short['TRAIL']}/"
            f"{reason_counts_short['SL']}/{reason_counts_short['EOD']}"
        )
    else:
        reason_text = "point exits unavailable"
    ax.set_title(
        (
            f"{bar_label} | bars: {len(df)} | "
            f"side={side_key} | "
            f"long_exit_soon={int(long_hits.sum())} | short_exit_soon={int(short_hits.sum())} | "
            f"pointL={point_long_count} | pointS={point_short_count} | {reason_text}"
        ),
        fontsize=14,
    )
    ax.set_ylabel("Close Price")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=10, ncol=3)

    tick_positions, tick_labels = _compute_time_ticks(date_index, pos)
    _apply_time_ticks(ax, tick_positions, tick_labels)
    _draw_day_lines([ax], tick_positions)

    _finalize_plot(
        fig,
        suptitle="Close Price with Hybrid Meta Exit Labels (Entry Overlay)",
        suptitle_y=1.02,
        top=0.93,
        save_path=save_path,
    )


_LABEL_PLOTTERS = {
    "atr_swing": plot_atr_swing_signals,
    "leg_segmentation": plot_leg_segmentation_signals,
    "continuation": plot_continuation_signals,
    "swing_state_machine": plot_swing_state_machine_signals,
    "triple_barrier": plot_triple_barrier_signals,
    "all_labels": plot_all_labels,
    "mfe_mae": plot_mfe_mae_labels,
    "bars_to_exhaustion": plot_exhaustion_progress,
    "continuation_strength": plot_continuation_strength,
    "trend_phase": plot_trend_phase_signals,
    "meta_entry": plot_meta_entry_signals,
    "meta_exit": plot_meta_exit_signals,
}


def plot_selected_label_plots(
    df: pd.DataFrame,
    *,
    plot_types: Iterable[str] | str | None = None,
    save_paths: Mapping[str, str | Path] | None = None,
    plot_kwargs: Mapping[str, Mapping[str, object]] | None = None,
    tail: int | None = _PLOT_TAIL_BARS,
    random_window: bool = False,
    seed: int | None = None,
    ticker: str | None = None,
    data_dir: Path | None = None,
) -> None:
    """
    Run one or more label plots, with optional save paths or plot-specific kwargs.
    """
    if plot_types is None:
        plot_list = list(DEFAULT_LABEL_PLOT_TYPES)
    elif isinstance(plot_types, str):
        plot_types = plot_types.strip()
        if plot_types.lower() == "all":
            plot_list = list(DEFAULT_LABEL_PLOT_TYPES)
        else:
            plot_list = [p.strip() for p in plot_types.split(",") if p.strip()]
    else:
        plot_list = list(plot_types)

    seen: set[str] = set()
    resolved = []
    for plot_type in plot_list:
        canonical = _normalize_plot_type(plot_type)
        if canonical not in seen:
            resolved.append(canonical)
            seen.add(canonical)

    normalized_save_paths: dict[str, str | Path] = {}
    if save_paths:
        for key, value in save_paths.items():
            normalized_save_paths[_normalize_plot_type(key)] = value

    normalized_plot_kwargs: dict[str, dict[str, object]] = {}
    if plot_kwargs:
        for key, value in plot_kwargs.items():
            normalized_plot_kwargs[_normalize_plot_type(key)] = dict(value)

    for plot_type in resolved:
        plotter = _LABEL_PLOTTERS[plot_type]
        kwargs = dict(normalized_plot_kwargs.get(plot_type, {}))
        if "tail" not in kwargs:
            kwargs["tail"] = tail
        if "random_window" not in kwargs:
            kwargs["random_window"] = random_window
        if "seed" not in kwargs:
            kwargs["seed"] = seed
        if "save_path" not in kwargs:
            if plot_type in normalized_save_paths:
                kwargs["save_path"] = normalized_save_paths[plot_type]
            elif ticker is not None and data_dir is not None:
                kwargs["save_path"] = get_default_plot_path(ticker, data_dir, plot_type)
        plotter(df, **kwargs)


def get_default_plot_path(ticker: str, data_dir: Path, plot_type: str) -> Path:
    plot_key = _normalize_plot_type(plot_type)
    slug = normalize_ticker(ticker).lower()
    plots_dir = data_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    base = _LABEL_PLOT_FILES[plot_key]
    filename = base if slug == "spy" else f"{slug}_{base}"
    return plots_dir / filename


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path.cwd()


def _load_feature_frame(
    ticker: str,
    dataset_name: str,
    *,
    x_filename: str | None = None,
) -> tuple[pd.DataFrame, list[str], Path]:
    from Data.load_data import get_ticker_processed_base_dir

    clean = normalize_ticker(ticker)
    processed_dir = get_ticker_processed_base_dir(clean)
    dataset_dir = processed_dir / "datasets" / dataset_name
    if x_filename:
        X_path = dataset_dir / x_filename
    else:
        X_path = dataset_dir / "X.parquet"
        if not X_path.exists():
            candidates = [
                dataset_dir / f"X_{dataset_name}_tree.parquet",
                dataset_dir / f"X_{dataset_name}_lstm.parquet",
            ]
            if X_path.exists():
                candidates.insert(0, X_path)
            X_path = next((p for p in candidates if p.exists()), None)
            if X_path is None:
                any_match = sorted(dataset_dir.glob("X_*.parquet"))
                X_path = any_match[0] if any_match else None
    if X_path is None or not X_path.exists():
        raise FileNotFoundError(f"Missing X parquet under {dataset_dir}")

    X_df = pd.read_parquet(X_path)
    features_path = dataset_dir / f"features_{Path(X_path).stem}.txt"
    if not features_path.exists():
        features_path = dataset_dir / "features.txt"
    if features_path.exists():
        feature_cols = [
            line.strip()
            for line in features_path.read_text().splitlines()
            if line.strip()
        ]
        missing = [c for c in feature_cols if c not in X_df.columns]
        if missing:
            raise KeyError(
                f"Missing feature columns in X.parquet: {', '.join(missing)}"
            )
        X_df = X_df[feature_cols]
    else:
        feature_cols = list(X_df.columns)

    return X_df, feature_cols, X_path


def _load_plot_frame(
    ticker: str,
    row_idx: np.ndarray | None,
    *,
    x_path: Path | None = None,
) -> pd.DataFrame | None:
    from Data.load_data import load_ticker_parquet

    plot_df = None
    if x_path is not None:
        dataset_dir = x_path.parent
        plot_path = dataset_dir / "plot_frame.parquet"
        if plot_path.exists():
            plot_df = pd.read_parquet(plot_path)

    if plot_df is None:
        try:
            plot_df = load_ticker_parquet(ticker)
        except Exception:
            return None

    if row_idx is None or len(row_idx) == 0:
        return plot_df
    max_idx = int(np.max(row_idx))
    if max_idx >= len(plot_df):
        return None
    return plot_df.iloc[row_idx]


def _select_side_target(
    side: str, y_long: np.ndarray, y_short: np.ndarray
) -> np.ndarray:
    side_key = side.strip().lower()
    if side_key in {"long", "up"}:
        return y_long
    if side_key in {"short", "down"}:
        return y_short
    raise ValueError(f"Unknown side: {side}")


def _candidate_label_tokens(label_mode: str) -> list[str]:
    mode = label_mode.strip().lower()
    if mode == "mfe_mae":
        return ["mfe_mae", "mfe", "mae"]
    if mode in {"mfe", "mae"}:
        return [mode, "mfe_mae"]
    return [mode]


def _load_bilstm_target_stats(
    model_dir: Path,
    slug: str,
    dataset_name: str,
    label_mode: str,
    side: str,
    seq_len: int,
) -> tuple[float | None, float | None]:
    label_tokens = _candidate_label_tokens(label_mode)
    for token in label_tokens:
        meta_path = model_dir / f"{slug}_{dataset_name}_{token}_{side}_seq{seq_len}_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            mean = meta.get("target_mean")
            std = meta.get("target_std")
            if mean is None or std is None:
                return None, None
            return float(mean), float(std)
    return None, None


def _quintile_means(
    preds: np.ndarray, actual: np.ndarray, bins: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    mask = np.isfinite(preds) & np.isfinite(actual)
    if mask.sum() < bins:
        return None
    preds = preds[mask]
    actual = actual[mask]
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(preds, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        return None
    bin_ids = np.digitize(preds, edges[1:-1], right=True)
    means = np.zeros(edges.size - 1, dtype=float)
    counts = np.zeros(edges.size - 1, dtype=int)
    for idx in range(edges.size - 1):
        bin_mask = bin_ids == idx
        counts[idx] = int(bin_mask.sum())
        means[idx] = float(np.nanmean(actual[bin_mask])) if counts[idx] else np.nan
    return edges, means, counts


def _load_bilstm_split_data(
    *,
    ticker: str,
    dataset_name: str,
    label_mode: str,
    x_filename: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from Data.load_data import get_ticker_processed_base_dir, get_ticker_processed_split_dir

    clean = normalize_ticker(ticker)
    processed_dir = get_ticker_processed_base_dir(clean)
    dataset_dir = processed_dir / "datasets" / dataset_name
    x_path = dataset_dir / x_filename
    y_path = dataset_dir / "y.parquet"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing {x_filename} or y.parquet in {dataset_dir}")

    X = pd.read_parquet(x_path).to_numpy(dtype=np.float32)
    y_df = pd.read_parquet(y_path)

    if label_mode == "swing":
        long_col, short_col = "long_swing_label", "short_swing_label"
    elif label_mode == "leg":
        long_col, short_col = "leg_up_label", "leg_down_label"
    elif label_mode == "mfe":
        long_col, short_col = "mfe_up_atr", "mfe_down_atr"
    elif label_mode == "mae":
        long_col, short_col = "mae_down_atr", "mae_up_atr"
    elif label_mode == "mfe_mae":
        long_col, short_col = "mfe_up_atr", "mfe_down_atr"
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    missing_cols = [c for c in (long_col, short_col) if c not in y_df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing label columns in {y_path.name}: {', '.join(missing_cols)}"
        )

    if label_mode in {"mfe", "mae", "mfe_mae"}:
        y_long = y_df[long_col].to_numpy(dtype=np.float32)
        y_short = y_df[short_col].to_numpy(dtype=np.float32)
        y_long = np.nan_to_num(y_long, nan=0.0, posinf=0.0, neginf=0.0)
        y_short = np.nan_to_num(y_short, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        y_long = y_df[long_col].to_numpy(dtype=np.int64)
        y_short = y_df[short_col].to_numpy(dtype=np.int64)

    split_dir = (
        get_ticker_processed_split_dir(clean)
        / dataset_name
        / Path(x_filename).stem
    )
    split_path = split_dir / f"{split}_idx.npy"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    idx = np.load(split_path)
    return X[idx], y_long[idx], y_short[idx]


def _load_model_and_mask(model_dir: Path) -> tuple["XGBClassifier", np.ndarray]:
    from xgboost import XGBClassifier

    model_path = model_dir / "xgb_model.json"
    mask_path = model_dir / "best_mask.npy"
    if not model_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"Missing model artifacts in {model_dir}")

    model = XGBClassifier()
    model.load_model(model_path)
    mask = np.load(mask_path).astype(bool)
    return model, mask


def _ga_label_dir_from_mode(label_mode: str) -> str | None:
    mode = (label_mode or "").strip().lower()
    if mode in {"pivot", "pivots"}:
        return "pivots"
    if mode == "tb":
        return "tb"
    if mode == "swing":
        return "swing"
    return None


def _resolve_ga_side_dir(model_root: Path, side: str, label_mode: str) -> Path | None:
    base_side = model_root / side
    candidates: list[Path] = []
    label_dir = _ga_label_dir_from_mode(label_mode)
    if label_dir:
        candidates.append(base_side / label_dir)
        # Backward compatibility with older nested layout.
        candidates.append(base_side / "probs" / label_dir)
    candidates.append(base_side)
    for candidate in candidates:
        if (candidate / "best_mask.npy").exists() and (candidate / "xgb_model.json").exists():
            return candidate
    return None


def plot_bilstm_inference_vs_actual(
    *,
    ticker: str = "$SPY",
    dataset_name: str = "15min",
    model_name: str = "mabilstm",
    label_mode: str = "mfe_mae",
    seq_len: int = 30,
    x_filename: str | None = None,
    split: str = "test",
    sides: Sequence[str] = ("long", "short"),
    batch_size: int = 256,
    tail: int | None = None,
    save_path: str | None = None,
    device: str | None = None,
    quintile_bins: int = 5,
    print_quintile_test: bool = True,
) -> None:
    """
    Plot BiLSTM inference vs actual targets on a dataset split.
    """
    import torch
    from torch.utils.data import DataLoader

    from Models.bilstm.mabilstm_dataset import SequenceRegressionDataset
    from Models.bilstm.mabilstm_model import MABiLSTM

    if x_filename is None:
        x_filename = f"X_{dataset_name}_lstm.parquet"

    X_split, y_long, y_short = _load_bilstm_split_data(
        ticker=ticker,
        dataset_name=dataset_name,
        label_mode=label_mode,
        x_filename=x_filename,
        split=split,
    )

    if tail is not None and tail > 0:
        X_split = X_split[-tail:]
        y_long = y_long[-tail:]
        y_short = y_short[-tail:]

    if len(X_split) < seq_len:
        raise ValueError("Split is too small for the requested seq_len.")

    if device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    repo_root = _resolve_repo_root()
    model_dir = repo_root / "Data" / "models" / model_name
    slug = normalize_ticker(ticker).lower()

    regression_mode = label_mode in {"mfe", "mae", "mfe_mae"}

    fig, axes = plt.subplots(
        len(sides),
        1,
        figsize=(18, max(5, 4 * len(sides))),
        sharex=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, side in zip(axes, sides, strict=False):
        target = _select_side_target(side, y_long, y_short).astype(np.float32)
        dataset = SequenceRegressionDataset(X_split, target, seq_len=seq_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model_path = None
        matched_token = None
        label_tokens = _candidate_label_tokens(label_mode)
        for token in label_tokens:
            candidate = (
                model_dir / f"{slug}_{dataset_name}_{token}_{side}_seq{seq_len}.pth"
            )
            if candidate.exists():
                model_path = candidate
                matched_token = token
                break
        if model_path is None:
            expected = ", ".join(
                f"{slug}_{dataset_name}_{token}_{side}_seq{seq_len}.pth"
                for token in label_tokens
            )
            raise FileNotFoundError(
                f"Missing model file under {model_dir}. Tried: {expected}"
            )
        target_mean, target_std = _load_bilstm_target_stats(
            model_dir,
            slug,
            dataset_name,
            matched_token or label_mode,
            side,
            seq_len,
        )

        model = MABiLSTM(input_dim=X_split.shape[1]).to(torch_device)
        state = torch.load(model_path, map_location=torch_device)
        model.load_state_dict(state)
        model.eval()

        preds_list: list[np.ndarray] = []
        target_list: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(torch_device)
                yb = yb.to(torch_device)
                preds, _ = model(xb)
                if not regression_mode:
                    preds = torch.sigmoid(preds)
                preds_list.append(preds.detach().cpu().numpy())
                target_list.append(yb.detach().cpu().numpy())

        preds = np.concatenate(preds_list, axis=0) if preds_list else np.array([])
        actual = np.concatenate(target_list, axis=0) if target_list else np.array([])
        if (
            regression_mode
            and target_mean is not None
            and target_std is not None
            and np.isfinite(target_mean)
            and np.isfinite(target_std)
            and target_std != 0
        ):
            preds = preds * target_std + target_mean

        pos = np.arange(len(actual))
        ax.plot(pos, actual, color="#1f77b4", linewidth=1.6, label="actual")
        ax.plot(pos, preds, color="#FB8C00", linewidth=1.4, alpha=0.85, label="pred")
        ax.set_ylabel("Target")
        ax.set_title(f"{side.upper()} | {label_mode} | split={split}")
        ax.legend(loc="upper left")

        if print_quintile_test and regression_mode:
            side_key = side.strip().lower()
            if side_key in {"long", "up"}:
                result = _quintile_means(preds, actual, bins=quintile_bins)
                if result is None:
                    print("Quintile test skipped: not enough valid data.")
                else:
                    edges, means, counts = result
                    print("\nQuintile test (predicted MFE_up -> realized MFE_up):")
                    for i, (mean, count) in enumerate(zip(means, counts), start=1):
                        lo = edges[i - 1]
                        hi = edges[i]
                        print(f"  Bin {i}: [{lo:.4f}, {hi:.4f}] n={count} mean={mean:.4f}")
                    if len(means) >= 2 and np.isfinite(means[0]) and np.isfinite(means[-1]):
                        delta = means[-1] - means[0]
                        print(f"  Bin{len(means)} - Bin1: {delta:.4f}")
        from scipy.stats import spearmanr
        mask = np.isfinite(preds) & np.isfinite(actual)
        rho, p = spearmanr(preds[mask], actual[mask])
        print("Spearman rho:", rho, "p:", p, "n:", mask.sum())

    axes[-1].set_xlabel("Sample")

    _finalize_plot(
        fig,
        suptitle=f"{normalize_ticker(ticker)} | BiLSTM inference vs actual",
        suptitle_y=1.02,
        top=0.92,
        save_path=save_path,
    )


def _select_features(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 1:
        raise ValueError("best_mask.npy must be a 1D array.")
    if X.shape[1] != mask.size:
        raise ValueError(
            f"Feature count mismatch: X has {X.shape[1]} cols, mask has {mask.size}."
        )
    return X[:, mask]


def plot_model_inference(
    X_df: pd.DataFrame,
    long_probs: np.ndarray | None,
    short_probs: np.ndarray | None,
    *,
    long_actual: np.ndarray | None = None,
    short_actual: np.ndarray | None = None,
    long_label_name: str | None = None,
    short_label_name: str | None = None,
    threshold: float = 0.8,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    plot_index = X_df.index if isinstance(X_df.index, pd.DatetimeIndex) else None
    has_ohlc = all(c in X_df.columns for c in ("open", "high", "low", "close"))
    close_y = X_df["close"].to_numpy() if "close" in X_df.columns else None
    pos = np.arange(len(X_df))

    if long_actual is not None:
        long_actual = np.asarray(long_actual).reshape(-1)
    if short_actual is not None:
        short_actual = np.asarray(short_actual).reshape(-1)

    if long_actual is not None and len(long_actual) != len(X_df):
        raise ValueError("long_actual length must match X_df for plotting.")
    if short_actual is not None and len(short_actual) != len(X_df):
        raise ValueError("short_actual length must match X_df for plotting.")
    long_thr = float(threshold if long_threshold is None else long_threshold)
    short_thr = float(threshold if short_threshold is None else short_threshold)

    fig, (ax_price, ax_prob) = plt.subplots(
        2,
        1,
        figsize=(18, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    if has_ohlc:
        open_y = X_df["open"].to_numpy()
        high_y = X_df["high"].to_numpy()
        low_y = X_df["low"].to_numpy()
        valid_mask = (
            np.isfinite(open_y)
            & np.isfinite(high_y)
            & np.isfinite(low_y)
            & np.isfinite(close_y)
        )
    elif close_y is not None:
        valid_mask = np.isfinite(close_y)
    else:
        raise ValueError(
            "X.parquet must include close (or open/high/low/close) to plot."
        )

    if not valid_mask.any():
        raise ValueError("No valid price bars to plot after filtering NaNs.")

    if has_ohlc:
        wick_color = "#4a4a4a"
        up_color = "#1976D2"
        down_color = "#E53935"
        up = close_y >= open_y
        up_mask = up & valid_mask
        down_mask = (~up) & valid_mask

        ax_price.vlines(
            pos[valid_mask],
            low_y[valid_mask],
            high_y[valid_mask],
            color=wick_color,
            linewidth=1.0,
            zorder=1,
        )
        ax_price.bar(
            pos[up_mask],
            close_y[up_mask] - open_y[up_mask],
            width=0.8,
            bottom=open_y[up_mask],
            color=up_color,
            edgecolor="none",
            zorder=1.2,
        )
        ax_price.bar(
            pos[down_mask],
            close_y[down_mask] - open_y[down_mask],
            width=0.8,
            bottom=open_y[down_mask],
            color=down_color,
            edgecolor="none",
            zorder=1.2,
        )
        spread = (high_y - low_y)[valid_mask]
        marker_offset = np.nanmedian(spread)
        if not np.isfinite(marker_offset) or marker_offset <= 0:
            marker_offset = np.nanmax(high_y[valid_mask]) * 0.002
        long_y = low_y - marker_offset * 0.6
        short_y = high_y + marker_offset * 0.6
        long_actual_y = low_y - marker_offset * 1.2
        short_actual_y = high_y + marker_offset * 1.2
    elif close_y is not None:
        ax_price.plot(pos, close_y, color="#1f77b4", linewidth=1.6, label="Close")
        clean_close = close_y[valid_mask]
        marker_offset = np.nanmedian(np.abs(np.diff(clean_close)))
        if not np.isfinite(marker_offset) or marker_offset <= 0:
            marker_offset = np.nanmax(clean_close) * 0.002
        long_y = close_y - marker_offset * 2
        short_y = close_y + marker_offset * 2
        long_actual_y = close_y - marker_offset * 3
        short_actual_y = close_y + marker_offset * 3

    if long_probs is not None:
        long_mask = (long_probs >= long_thr) & valid_mask
        if long_mask.any():
            ax_price.scatter(
                pos[long_mask],
                long_y[long_mask],
                color="#1565C0",
                marker="^",
                s=60,
                label=f"LONG prob >= {long_thr:.2f}",
                zorder=2,
            )
    if short_probs is not None:
        short_mask = (short_probs >= short_thr) & valid_mask
        if short_mask.any():
            ax_price.scatter(
                pos[short_mask],
                short_y[short_mask],
                color="#FB8C00",
                marker="v",
                s=60,
                label=f"SHORT prob >= {short_thr:.2f}",
                zorder=2,
            )
    if long_actual is not None:
        long_label = f"{long_label_name or 'LONG'} actual"
        long_actual_mask = (long_actual == 1) & valid_mask
        if long_actual_mask.any():
            ax_price.scatter(
                pos[long_actual_mask],
                long_actual_y[long_actual_mask],
                facecolors="none",
                edgecolors="#0D47A1",
                linewidths=1.4,
                marker="^",
                s=56,
                label=long_label,
                zorder=2.2,
            )
    if short_actual is not None:
        short_label = f"{short_label_name or 'SHORT'} actual"
        short_actual_mask = (short_actual == 1) & valid_mask
        if short_actual_mask.any():
            ax_price.scatter(
                pos[short_actual_mask],
                short_actual_y[short_actual_mask],
                facecolors="none",
                edgecolors="#EF6C00",
                linewidths=1.4,
                marker="v",
                s=56,
                label=short_label,
                zorder=2.2,
            )

    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left")
    ax_price.set_title(title or "Model Inference (Window)")

    if long_probs is not None:
        ax_prob.plot(
            pos, long_probs, label="LONG P(class=1)", color="#1565C0", linewidth=1.5
        )
    if short_probs is not None:
        ax_prob.plot(
            pos, short_probs, label="SHORT P(class=1)", color="#FB8C00", linewidth=1.5
        )
    if long_actual is not None:
        ax_prob.step(
            pos,
            long_actual,
            where="post",
            color="#0D47A1",
            linewidth=1.1,
            linestyle="--",
            alpha=0.7,
            label=f"{long_label_name or 'LONG'} actual",
        )
    if short_actual is not None:
        ax_prob.step(
            pos,
            short_actual,
            where="post",
            color="#EF6C00",
            linewidth=1.1,
            linestyle="--",
            alpha=0.7,
            label=f"{short_label_name or 'SHORT'} actual",
        )
    if np.isfinite(long_thr):
        ax_prob.axhline(
            long_thr,
            color="#1565C0",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
            label=f"LONG thr={long_thr:.2f}",
        )
    if np.isfinite(short_thr) and abs(short_thr - long_thr) > 1e-12:
        ax_prob.axhline(
            short_thr,
            color="#FB8C00",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
            label=f"SHORT thr={short_thr:.2f}",
        )
    ax_prob.set_ylim(0, 1.02)
    ax_prob.set_title("Model Probabilities (Window)")
    ax_prob.legend(loc="upper right")
    tick_positions = None
    tick_labels = None
    if isinstance(plot_index, pd.DatetimeIndex):
        dates = pd.Series(plot_index)
        day_start = dates.dt.normalize().ne(dates.dt.normalize().shift())
        tick_positions = pos[day_start.to_numpy()]
        tick_labels = dates[day_start].dt.strftime("%Y-%m-%d").to_list()
    elif "month" in X_df.columns and "day_of_month" in X_df.columns:
        month = pd.Series(X_df["month"].to_numpy()).astype(int)
        day = pd.Series(X_df["day_of_month"].to_numpy()).astype(int)
        day_key = month.astype(str).str.zfill(2) + "-" + day.astype(str).str.zfill(2)
        day_start = day_key.ne(day_key.shift())
        tick_positions = pos[day_start.to_numpy()]
        tick_labels = day_key[day_start].to_list()

    if tick_positions is not None and len(tick_positions) > 0:
        if len(tick_positions) > 25:
            step = int(np.ceil(len(tick_positions) / 25))
            tick_positions = tick_positions[::step]
            tick_labels = tick_labels[::step]
        ax_prob.set_xticks(tick_positions)
        ax_prob.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
        for x in tick_positions:
            ax_price.axvline(
                x, color="#cfd8dc", linestyle="--", linewidth=0.8, alpha=0.7, zorder=0.5
            )
            ax_prob.axvline(
                x, color="#cfd8dc", linestyle="--", linewidth=0.8, alpha=0.7, zorder=0.5
            )
        ax_prob.set_xlabel("Session")
    else:
        ax_prob.set_xlabel("Bar")

    plt.tight_layout()
    if save_path:
        save_path = str(save_path)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved plot to {save_path}")
    plt.show()


def get_default_model_inference_plot_path(ticker: str, model_name: str) -> Path:
    from Data.load_data import get_ticker_plots_dir

    slug = normalize_ticker(ticker).lower()
    plots_dir = get_ticker_plots_dir(slug)
    filename = f"{slug}_{model_name}_inference.png"
    return plots_dir / filename


def _extract_inference_labels(
    y_df: pd.DataFrame,
    *,
    label_mode: str = "auto",
) -> tuple[np.ndarray | None, np.ndarray | None, str | None, str | None]:
    mode = (label_mode or "auto").strip().lower()

    def from_pair(long_col: str, short_col: str):
        if long_col in y_df.columns and short_col in y_df.columns:
            long_vals = (
                y_df[long_col].fillna(0).astype(int).to_numpy() == 1
            ).astype(np.int64)
            short_vals = (
                y_df[short_col].fillna(0).astype(int).to_numpy() == 1
            ).astype(np.int64)
            return long_vals, short_vals, long_col, short_col
        return None

    def from_signed(col: str):
        if col in y_df.columns:
            values = y_df[col].fillna(0).astype(int).to_numpy()
            long_vals = (values == 1).astype(np.int64)
            short_vals = (values == -1).astype(np.int64)
            return long_vals, short_vals, f"{col}=+1", f"{col}=-1"
        return None

    if mode in {"auto", "swing"}:
        result = from_pair("long_swing_label", "short_swing_label")
        if result:
            return result
        result = from_signed("atr_swing_label")
        if result:
            return result

    if mode in {"auto", "leg"}:
        result = from_pair("leg_up_label", "leg_down_label")
        if result:
            return result
        result = from_signed("atr_leg_label")
        if result:
            return result

    return None, None, None, None


def model_inference_main() -> None:
    from Data.load_data import get_ticker_processed_split_dir, load_split_indices

    parser = argparse.ArgumentParser(
        description="Plot model inference signals over price bars."
    )
    parser.add_argument("--ticker", default="$SPY")
    parser.add_argument("--dataset", default="15min")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument(
        "--split", choices=["all", "train", "val", "test"], default="test"
    )
    parser.add_argument(
        "--label-mode",
        choices=["auto", "swing", "leg"],
        default="auto",
        help="Which labels to overlay for comparison.",
    )
    parser.add_argument("--tail", type=int, default=None)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    X_df, feature_cols, x_path = _load_feature_frame(args.ticker, args.dataset)
    row_idx = np.arange(len(X_df))

    if args.split != "all":
        clean = normalize_ticker(args.ticker)
        split_root = get_ticker_processed_split_dir(clean)
        x_stem = Path(x_path).stem
        split_path = split_root / args.dataset / x_stem / f"{args.split}_idx.npy"
        if split_path.exists():
            row_idx = np.load(split_path)
        else:
            splits = load_split_indices(args.ticker, args.dataset)
            row_idx = splits[args.split]

    if args.tail:
        row_idx = row_idx[-args.tail :]

    row_idx = np.asarray(row_idx, dtype=int)
    X_df = X_df.iloc[row_idx]

    plot_df = _load_plot_frame(args.ticker, row_idx, x_path=x_path)
    if plot_df is None:
        plot_df = X_df

    X = X_df.to_numpy(dtype=np.float32)

    repo_root = _resolve_repo_root()
    model_root = repo_root / "Data" / "models" / args.model_name

    long_probs = None
    short_probs = None
    long_actual = None
    short_actual = None
    long_label_name = None
    short_label_name = None

    y_path = x_path.parent / "y.parquet"
    if y_path.exists():
        y_df = pd.read_parquet(y_path)
        max_idx = int(np.max(row_idx)) if len(row_idx) else -1
        if max_idx < len(y_df):
            y_df = y_df.iloc[row_idx]
            (
                long_actual,
                short_actual,
                long_label_name,
                short_label_name,
            ) = _extract_inference_labels(y_df, label_mode=args.label_mode)
            if (
                args.label_mode != "auto"
                and long_actual is None
                and short_actual is None
            ):
                raise KeyError(
                    f"No labels found for label_mode='{args.label_mode}' in {y_path.name}."
                )

    long_dir = _resolve_ga_side_dir(model_root, "long", args.label_mode)
    if long_dir is not None:
        long_model, long_mask = _load_model_and_mask(long_dir)
        long_probs = long_model.predict_proba(_select_features(X, long_mask))[:, 1]

    short_dir = _resolve_ga_side_dir(model_root, "short", args.label_mode)
    if short_dir is not None:
        short_model, short_mask = _load_model_and_mask(short_dir)
        short_probs = short_model.predict_proba(_select_features(X, short_mask))[:, 1]

    if long_probs is None and short_probs is None:
        raise FileNotFoundError(f"No model artifacts found under {model_root}")

    title = f"{normalize_ticker(args.ticker)} | {args.model_name} | split={args.split}"
    save_path = args.save or str(
        get_default_model_inference_plot_path(args.ticker, args.model_name)
    )
    plot_model_inference(
        plot_df,
        long_probs,
        short_probs,
        long_actual=long_actual,
        short_actual=short_actual,
        long_label_name=long_label_name,
        short_label_name=short_label_name,
        threshold=args.threshold,
        title=title,
        save_path=save_path,
    )
