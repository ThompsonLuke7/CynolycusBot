from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .theme import PlotTheme, get_theme
from .time_axis import to_mpl_time


@dataclass(frozen=True)
class CandlePlot:
    x: np.ndarray
    up: np.ndarray
    down: np.ndarray


def extract_ohlc(
    df: pd.DataFrame,
    *,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos = np.arange(len(df))
    open_y = pd.to_numeric(df[open_col], errors="coerce").to_numpy(dtype=float)
    high_y = pd.to_numeric(df[high_col], errors="coerce").to_numpy(dtype=float)
    low_y = pd.to_numeric(df[low_col], errors="coerce").to_numpy(dtype=float)
    close_y = pd.to_numeric(df[close_col], errors="coerce").to_numpy(dtype=float)
    return pos, open_y, high_y, low_y, close_y


def compute_marker_offset(
    df: pd.DataFrame,
    high_y: np.ndarray,
    low_y: np.ndarray,
    *,
    atr_col: str = "atr",
    fallback_scale: float = 0.001,
) -> float:
    if atr_col in df.columns:
        marker_offset = np.nanmedian(pd.to_numeric(df[atr_col], errors="coerce").to_numpy(dtype=float))
    else:
        marker_offset = np.nanmedian(high_y - low_y)
    if not np.isfinite(marker_offset) or marker_offset <= 0:
        marker_offset = np.nanmax(high_y) * fallback_scale
    return float(marker_offset)


def _infer_width(x: np.ndarray, fallback: float) -> float:
    if len(x) < 2:
        return fallback
    diffs = np.diff(np.sort(np.asarray(x, dtype=float)))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return fallback
    return float(np.nanmedian(diffs) * 0.68)


def plot_candles(
    ax,
    x: np.ndarray,
    open_y: np.ndarray,
    high_y: np.ndarray,
    low_y: np.ndarray,
    close_y: np.ndarray,
    *,
    theme: PlotTheme | None = None,
    wick_color: str | None = None,
    up_color: str | None = None,
    down_color: str | None = None,
    width: float | None = None,
    zorder: float = 1.0,
    label: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    theme = get_theme(theme)
    x = np.asarray(x, dtype=float)
    open_y = np.asarray(open_y, dtype=float)
    high_y = np.asarray(high_y, dtype=float)
    low_y = np.asarray(low_y, dtype=float)
    close_y = np.asarray(close_y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(open_y) & np.isfinite(high_y) & np.isfinite(low_y) & np.isfinite(close_y)
    up = valid & (close_y >= open_y)
    down = valid & ~up
    width = _infer_width(x[valid], 0.72) if width is None else width
    wick_color = wick_color or theme.wick
    up_color = up_color or theme.bull
    down_color = down_color or theme.bear

    ax.vlines(x[valid], low_y[valid], high_y[valid], color=wick_color, linewidth=1.0, zorder=zorder)
    ax.bar(
        x[up],
        close_y[up] - open_y[up],
        width=width,
        bottom=open_y[up],
        color=up_color,
        edgecolor="none",
        label="Bull candle" if label else None,
        zorder=zorder + 0.2,
    )
    ax.bar(
        x[down],
        close_y[down] - open_y[down],
        width=width,
        bottom=open_y[down],
        color=down_color,
        edgecolor="none",
        label="Bear candle" if label else None,
        zorder=zorder + 0.2,
    )
    return up, down


def plot_candles_from_frame(
    ax,
    df: pd.DataFrame,
    *,
    time_col: str | None = None,
    compressed: bool = True,
    timezone: str = "America/New_York",
    width: float | None = None,
    theme: PlotTheme | None = None,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    **kwargs,
) -> CandlePlot:
    pos, open_y, high_y, low_y, close_y = extract_ohlc(
        df,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
    )
    if compressed:
        x = pos.astype(float)
    else:
        times = df[time_col] if time_col else df.index
        x = to_mpl_time(times, timezone=timezone)
    up, down = plot_candles(
        ax,
        x,
        open_y,
        high_y,
        low_y,
        close_y,
        width=width,
        theme=theme,
        **kwargs,
    )
    return CandlePlot(x=x, up=up, down=down)
