from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PlotTheme:
    name: str
    figure_bg: str
    axes_bg: str
    text: str
    muted_text: str
    grid: str
    spine: str
    wick: str
    bull: str
    bear: str
    long: str
    short: str
    win: str
    loss: str
    neutral: str
    blue: str
    amber: str
    purple: str
    warning: str


DARK_THEME = PlotTheme(
    name="cynolycus_dark",
    figure_bg="#0b1220",
    axes_bg="#111827",
    text="#e5e7eb",
    muted_text="#9ca3af",
    grid="#334155",
    spine="#334155",
    wick="#94a3b8",
    bull="#22c55e",
    bear="#ef4444",
    long="#38bdf8",
    short="#f59e0b",
    win="#22c55e",
    loss="#ef4444",
    neutral="#64748b",
    blue="#60a5fa",
    amber="#fbbf24",
    purple="#a78bfa",
    warning="#f97316",
)

DEFAULT_THEME = DARK_THEME


def get_theme(theme: PlotTheme | None = None) -> PlotTheme:
    return theme or DEFAULT_THEME


def apply_mpl_defaults(theme: PlotTheme | None = None, *, font_size: int = 10) -> PlotTheme:
    """Apply repo-wide matplotlib defaults for generated plots."""
    import matplotlib as mpl

    theme = get_theme(theme)
    mpl.rcParams.update(
        {
            "figure.facecolor": theme.figure_bg,
            "axes.facecolor": theme.axes_bg,
            "axes.edgecolor": theme.spine,
            "axes.labelcolor": theme.text,
            "axes.titlecolor": theme.text,
            "xtick.color": theme.muted_text,
            "ytick.color": theme.muted_text,
            "grid.color": theme.grid,
            "grid.alpha": 0.45,
            "font.size": font_size,
            "savefig.facecolor": theme.figure_bg,
            "savefig.edgecolor": theme.figure_bg,
        }
    )
    return theme


def _iter_axes(axes) -> Iterable:
    if axes is None:
        return ()
    if isinstance(axes, (list, tuple)):
        for item in axes:
            yield from _iter_axes(item)
        return
    try:
        import numpy as np

        if isinstance(axes, np.ndarray):
            for item in axes.ravel():
                yield item
            return
    except Exception:
        pass
    yield axes


def style_axis(
    ax,
    theme: PlotTheme | None = None,
    *,
    grid: bool = True,
    tick_label_size: int = 9,
) -> None:
    theme = get_theme(theme)
    ax.set_facecolor(theme.axes_bg)
    ax.tick_params(colors=theme.muted_text, labelsize=tick_label_size)
    ax.xaxis.label.set_color(theme.text)
    ax.yaxis.label.set_color(theme.text)
    ax.title.set_color(theme.text)
    for spine in ax.spines.values():
        spine.set_edgecolor(theme.spine)
    if grid:
        ax.grid(True, color=theme.grid, linewidth=0.6, alpha=0.45)


def style_figure(fig, axes=None, theme: PlotTheme | None = None, *, grid: bool = True) -> PlotTheme:
    theme = get_theme(theme)
    fig.patch.set_facecolor(theme.figure_bg)
    for ax in _iter_axes(axes if axes is not None else fig.axes):
        style_axis(ax, theme, grid=grid)
    return theme
