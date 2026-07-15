"""Shared plotting helpers for research, audits, and dashboards.

New Python plots should import from this package instead of hand-rolling
candle colors, axes styling, time ticks, or save behavior.
"""
from __future__ import annotations

from .candles import (
    CandlePlot,
    compute_marker_offset,
    extract_ohlc,
    plot_candles,
    plot_candles_from_frame,
)
from .figures import make_price_probability_figure, plot_direction_probabilities, save_figure
from .theme import (
    DARK_THEME,
    DEFAULT_THEME,
    LIGHT_THEME,
    PlotTheme,
    apply_mpl_defaults,
    get_theme,
    style_axis,
    style_figure,
)
from .time_axis import (
    apply_time_ticks,
    compute_time_ticks,
    draw_day_lines,
    infer_bar_label,
    setup_datetime_axis,
    time_to_position,
    to_mpl_time,
)

__all__ = [
    "CandlePlot",
    "DARK_THEME",
    "DEFAULT_THEME",
    "LIGHT_THEME",
    "PlotTheme",
    "apply_mpl_defaults",
    "apply_time_ticks",
    "compute_marker_offset",
    "compute_time_ticks",
    "draw_day_lines",
    "extract_ohlc",
    "get_theme",
    "infer_bar_label",
    "make_price_probability_figure",
    "plot_candles",
    "plot_candles_from_frame",
    "plot_direction_probabilities",
    "save_figure",
    "setup_datetime_axis",
    "style_axis",
    "style_figure",
    "time_to_position",
    "to_mpl_time",
]
