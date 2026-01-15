from pathlib import Path

from Data.plots.plots import get_default_plot_path as _get_default_plot_path
from Data.plots.plots import plot_atr_swing_signals


def get_default_plot_path(ticker: str, data_dir: Path) -> Path:
    return _get_default_plot_path(ticker, data_dir, "atr_swing")
