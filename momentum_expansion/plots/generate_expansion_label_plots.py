"""
Generate 4H momentum expansion label plots for a small ticker set.

Each plot shows compressed 4H candles, positive expansion_target bars, compact
positive-target regimes, and the expansion_score trace.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Data.plots.plots import (
    _apply_time_ticks,
    _compute_marker_offset,
    _compute_time_ticks,
    _draw_day_lines,
    _extract_ohlc,
    _plot_candles,
)
from momentum_expansion.config.momentum_config import (
    LABEL_CONFIG,
    LABELS_COMBINED,
    PLOTS_DIR,
)
from momentum_expansion.data.load_bars import load_4h

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA"]
DEFAULT_OUT_DIR = PLOTS_DIR / "expansion_label_plots"
PLOT_COLUMNS = [
    "fwd_max_return",
    "fwd_max_alpha",
    "fwd_atr_adj_return",
    "fwd_max_drawdown",
    "fwd_close_return",
    "trend_persistence",
    "expansion_score",
    "expansion_target",
]


def _load_labels(ticker: str, labels_path: Path) -> pd.DataFrame:
    labels = pd.read_parquet(labels_path)
    if not isinstance(labels.index, pd.MultiIndex) or "ticker" not in labels.index.names:
        raise ValueError(f"Expected MultiIndex labels with a ticker level at {labels_path}")
    try:
        ticker_labels = labels.xs(ticker, level="ticker").copy()
    except KeyError as exc:
        raise FileNotFoundError(f"No expansion labels found for {ticker}") from exc
    ticker_labels.index = pd.to_datetime(ticker_labels.index, utc=True, errors="coerce")
    ticker_labels = ticker_labels[[c for c in PLOT_COLUMNS if c in ticker_labels.columns]]
    return ticker_labels.sort_index()


def _plot_window(df: pd.DataFrame, *, window: int, anchor: str) -> pd.DataFrame:
    if len(df) <= window:
        return df

    valid_target = pd.to_numeric(df["expansion_target"], errors="coerce").fillna(0.0)
    positive_idx = np.flatnonzero(valid_target.to_numpy(dtype=float) >= 1.0)

    if anchor == "tail" or len(positive_idx) == 0:
        return df.tail(window)
    if anchor == "first-positive":
        center = int(positive_idx[0])
    else:
        center = int(positive_idx[-1])

    start = max(0, center - window // 2)
    end = min(len(df), start + window)
    start = max(0, end - window)
    return df.iloc[start:end]


def _target_runs(target_mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    idxs = np.flatnonzero(target_mask)
    if len(idxs) == 0:
        return runs

    start = int(idxs[0])
    prev = start
    for raw_idx in idxs[1:]:
        idx = int(raw_idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = idx
        prev = idx
    runs.append((start, prev))
    return runs


def _draw_target_regimes(
    ax: plt.Axes,
    *,
    pos: np.ndarray,
    target_mask: np.ndarray,
    max_regimes: int,
) -> None:
    runs = _target_runs(target_mask)
    if not runs:
        return
    for start, end in runs[-max_regimes:]:
        ax.axvspan(
            pos[start] - 0.45,
            pos[end] + 0.45,
            color="#FFD166",
            alpha=0.08,
            zorder=0.2,
        )


def plot_ticker(
    ticker: str,
    *,
    labels_path: Path = LABELS_COMBINED,
    out_dir: Path = DEFAULT_OUT_DIR,
    window: int = 220,
    anchor: str = "last-positive",
    max_target_regimes: int = 8,
    show_zero_markers: bool = False,
) -> Path:
    bars = load_4h(ticker)
    labels = _load_labels(ticker, labels_path)
    df = bars.join(labels, how="inner").dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        raise ValueError(f"No joined 4H bars + labels for {ticker}")
    if "expansion_target" not in df.columns or "expansion_score" not in df.columns:
        raise ValueError(f"Missing expansion target/score columns for {ticker}")

    df = _plot_window(df, window=window, anchor=anchor)
    pos, open_y, high_y, low_y, close_y = _extract_ohlc(df)
    marker_offset = _compute_marker_offset(df, high_y, low_y, atr_col="atr_14")
    target = pd.to_numeric(df["expansion_target"], errors="coerce")
    score = pd.to_numeric(df["expansion_score"], errors="coerce")
    fwd_ret = pd.to_numeric(df.get("fwd_max_return"), errors="coerce")
    fwd_alpha = pd.to_numeric(df.get("fwd_max_alpha"), errors="coerce")
    target_mask = target.fillna(0.0).to_numpy(dtype=float) >= 1.0
    labeled_mask = target.notna().to_numpy()

    fig, (ax_price, ax_score) = plt.subplots(
        2,
        1,
        figsize=(20, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
    )
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax_price, ax_score):
        ax.set_facecolor("#0d1117")
        ax.grid(True, color="#30363d", linewidth=0.6, alpha=0.55)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=9)

    _plot_candles(
        ax_price,
        pos,
        open_y,
        high_y,
        low_y,
        close_y,
        wick_color="#8b949e",
        up_color="#26a641",
        down_color="#f85149",
        width=0.72,
    )

    _draw_target_regimes(
        ax_price,
        pos=pos,
        target_mask=target_mask,
        max_regimes=max_target_regimes,
    )

    if target_mask.any():
        ax_price.scatter(
            pos[target_mask],
            low_y[target_mask] - marker_offset * 0.75,
            marker="^",
            s=72,
            color="#FFD166",
            edgecolors="#111827",
            linewidths=0.7,
            zorder=4,
            label=f"expansion_target = 1 ({int(target_mask.sum())})",
        )

    valid_not_positive = labeled_mask & ~target_mask
    if show_zero_markers and valid_not_positive.any():
        ax_price.scatter(
            pos[valid_not_positive],
            low_y[valid_not_positive] - marker_offset * 0.35,
            marker=".",
            s=18,
            color="#6b7280",
            alpha=0.45,
            zorder=3,
            label=f"target = 0 ({int(valid_not_positive.sum())})",
        )

    ax_score.plot(
        pos,
        score.to_numpy(dtype=float),
        color="#60a5fa",
        linewidth=1.4,
        label="expansion_score",
    )
    if target_mask.any():
        ax_score.scatter(
            pos[target_mask],
            score.to_numpy(dtype=float)[target_mask],
            color="#FFD166",
            s=36,
            zorder=3,
            label="positive target bars",
        )
    ax_score.axhline(0.5, color="#8b949e", linewidth=0.8, linestyle="--", alpha=0.55)
    ax_score.set_ylim(-0.03, 1.03)

    tick_positions, tick_labels = _compute_time_ticks(df.index, pos, max_ticks=18)
    _apply_time_ticks(ax_score, tick_positions, tick_labels)
    _draw_day_lines([ax_price, ax_score], tick_positions, line_color="#30363d")

    target_count = int(target_mask.sum())
    avg_fwd_ret = float(fwd_ret[target_mask].mean() * 100.0) if target_count else float("nan")
    avg_fwd_alpha = float(fwd_alpha[target_mask].mean() * 100.0) if target_count else float("nan")
    start = df.index.min().strftime("%Y-%m-%d")
    end = df.index.max().strftime("%Y-%m-%d")
    ax_price.set_title(
        f"{ticker} | 4H momentum expansion labels | {start} to {end} | "
        f"target=1 bars: {target_count} | avg fwd max return: {avg_fwd_ret:.1f}% | "
        f"avg fwd alpha: {avg_fwd_alpha:.1f}%",
        color="#c9d1d9",
        fontsize=12,
        pad=10,
    )
    ax_price.set_ylabel("Price", color="#c9d1d9")
    ax_score.set_ylabel("Score", color="#c9d1d9")
    ax_score.set_xlabel("Date", color="#c9d1d9")

    handles = [
        mpatches.Patch(color="#26a641", label="Bull candle"),
        mpatches.Patch(color="#f85149", label="Bear candle"),
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor="#FFD166", markersize=8, label="expansion_target = 1"),
        mpatches.Patch(color="#FFD166", alpha=0.18, label="positive-target regime"),
        plt.Line2D([0], [0], color="#60a5fa", linewidth=1.4, label="expansion_score"),
    ]
    ax_price.legend(handles=handles, loc="upper left", fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax_score.legend(loc="upper left", fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ticker / "expansion_label_4h.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot 4H momentum expansion labels on candles.")
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--labels-path", type=Path, default=LABELS_COMBINED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--window", type=int, default=220)
    parser.add_argument(
        "--anchor",
        choices=["last-positive", "first-positive", "tail"],
        default="last-positive",
        help="Which part of each ticker history to plot.",
    )
    parser.add_argument("--max-target-regimes", type=int, default=8)
    parser.add_argument("--show-zero-markers", action="store_true")
    parser.add_argument("--log", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper()), format="%(asctime)s %(levelname)s %(message)s")
    tickers = [str(t).upper().strip().lstrip("$") for t in args.tickers if str(t).strip()]
    outputs: list[Path] = []
    for ticker in tickers:
        try:
            out = plot_ticker(
                ticker,
                labels_path=args.labels_path,
                out_dir=args.out_dir,
                window=args.window,
                anchor=args.anchor,
                max_target_regimes=args.max_target_regimes,
                show_zero_markers=args.show_zero_markers,
            )
        except Exception as exc:
            logger.warning("[%s] skipped: %s", ticker, exc)
            continue
        outputs.append(out)
        logger.info("[%s] saved -> %s", ticker, out)

    if not outputs:
        raise SystemExit("No plots generated.")
    print("Generated plots:")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
