from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .theme import PlotTheme, get_theme, style_figure


def save_figure(
    fig,
    save_path: str | Path,
    *,
    dpi: int = 160,
    tight: bool = True,
    close: bool = False,
) -> Path:
    import matplotlib.pyplot as plt

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return out


def make_price_probability_figure(
    *,
    figsize: tuple[float, float] = (14, 8),
    height_ratios: Sequence[float] = (3.0, 1.0),
    theme: PlotTheme | None = None,
):
    import matplotlib.pyplot as plt

    theme = get_theme(theme)
    fig, (ax_price, ax_prob) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": list(height_ratios)},
    )
    style_figure(fig, (ax_price, ax_prob), theme)
    return fig, ax_price, ax_prob


def plot_direction_probabilities(
    ax,
    x,
    p_long,
    p_short,
    *,
    theme: PlotTheme | None = None,
    long_label: str = "P(long | directional)",
    short_label: str = "P(short | directional)",
    thresholds: Sequence[float] = (0.5,),
) -> None:
    theme = get_theme(theme)
    long = pd.to_numeric(p_long, errors="coerce")
    short = pd.to_numeric(p_short, errors="coerce")
    ax.plot(x, long, color=theme.long, lw=1.4, label=long_label)
    ax.plot(x, short, color=theme.short, lw=1.4, label=short_label)
    for threshold in thresholds:
        ax.axhline(float(threshold), color=theme.neutral, lw=0.8, ls="--", alpha=0.65)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Direction prob")
