from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def infer_bar_label(index: pd.DatetimeIndex) -> str:
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


def compute_time_ticks(
    date_index: pd.DatetimeIndex,
    pos: np.ndarray,
    *,
    max_ticks: int = 25,
    fmt: str = "%Y-%m-%d",
) -> tuple[np.ndarray | None, list[str] | None]:
    if not isinstance(date_index, pd.DatetimeIndex):
        return None, None
    dates = pd.Series(date_index)
    day_start = dates.dt.normalize().ne(dates.dt.normalize().shift())
    tick_positions = pos[day_start.to_numpy()]
    tick_labels = dates[day_start].dt.strftime(fmt).to_list()
    if len(tick_positions) > max_ticks:
        step = int(np.ceil(len(tick_positions) / max_ticks))
        tick_positions = tick_positions[::step]
        tick_labels = tick_labels[::step]
    return tick_positions, tick_labels


def apply_time_ticks(
    ax,
    tick_positions: np.ndarray | None,
    tick_labels: list[str] | None,
    *,
    color: str | None = None,
    rotation: int = 45,
    fontsize: int = 9,
) -> None:
    if tick_positions is None or tick_labels is None:
        return
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=rotation, ha="right", fontsize=fontsize)
    if color is not None:
        for label in ax.get_xticklabels():
            label.set_color(color)


def draw_day_lines(
    axes: Sequence,
    tick_positions: np.ndarray | None,
    *,
    line_color: str = "#334155",
    alpha: float = 0.35,
) -> None:
    if tick_positions is None:
        return
    for ax in axes:
        for x in tick_positions:
            ax.axvline(x, color=line_color, linestyle="--", linewidth=0.8, alpha=alpha, zorder=0.5)


def to_mpl_time(values, *, timezone: str = "America/New_York") -> np.ndarray:
    import matplotlib.dates as mdates

    ts = pd.to_datetime(values, utc=True, errors="coerce")
    if isinstance(ts, pd.Series):
        local = ts.dt.tz_convert(timezone).dt.tz_localize(None)
    else:
        local = pd.DatetimeIndex(ts).tz_convert(timezone).tz_localize(None)
    return mdates.date2num(local)


def setup_datetime_axis(
    ax,
    *,
    date_format: str = "%m/%d %H:%M",
    timezone: str = "America/New_York",
) -> None:
    import matplotlib.dates as mdates

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))


def time_to_position(index: pd.DatetimeIndex, times) -> pd.Series:
    parsed = pd.to_datetime(times, utc=True, errors="coerce")
    if index.tz is not None:
        parsed = parsed.dt.tz_convert(index.tz) if isinstance(parsed, pd.Series) else parsed.tz_convert(index.tz)
    stamps = index.view("int64")
    out: list[float] = []
    for ts in parsed:
        if pd.isna(ts):
            out.append(float("nan"))
            continue
        value = pd.Timestamp(ts).value
        right = int(np.searchsorted(stamps, value, side="right"))
        left = right - 1
        if left < 0 or right >= len(stamps):
            nearest = min(max(right, 0), len(stamps) - 1)
            out.append(float(nearest))
            continue
        span = max(1, stamps[right] - stamps[left])
        out.append(float(left) + float(value - stamps[left]) / float(span))
    out_index = getattr(times, "index", None)
    if callable(out_index):
        out_index = None
    return pd.Series(out, index=out_index, dtype=float)
