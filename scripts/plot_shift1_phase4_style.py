from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
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
    _select_plot_window,
)


DEFAULT_ANALYSIS_DIR = Path("Data/models/ga_xgboost/10min_shift1/analysis/phase4_1m_bodyclose_l42_s15")
DEFAULT_TRADES = (
    DEFAULT_ANALYSIS_DIR
    / "best_phase4_asym_long_break_prev_stop_1m_body_and_close_short_break_prev_stop_1m_body_and_close_cooldown_cluster_longmax4_shortmax4_test_trades.csv"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot shift-1 Phase 4 diagnostics in candle style.")
    parser.add_argument("--signal-frame", default=str(DEFAULT_ANALYSIS_DIR / "phase4_signal_frame.parquet"))
    parser.add_argument("--labels", default="Data/processed/spy/datasets/10min_shift1/y.parquet")
    parser.add_argument("--trades", default=str(DEFAULT_TRADES))
    parser.add_argument("--out-dir", default=str(DEFAULT_ANALYSIS_DIR))
    parser.add_argument("--tail", type=int, default=400)
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--long-threshold", type=float, default=0.42)
    parser.add_argument("--short-threshold", type=float, default=0.15)
    return parser.parse_args()


def _prob(frame: pd.DataFrame, side: str) -> pd.Series:
    test = pd.to_numeric(frame.get(f"p_{side}_test"), errors="coerce")
    oof = pd.to_numeric(frame.get(f"p_{side}_oof_train"), errors="coerce")
    return test.combine_first(oof)


def _setup(frame: pd.DataFrame, side: str) -> pd.Series:
    test = frame.get(f"{side}_setup_test")
    oof = frame.get(f"{side}_setup_oof")
    if test is None:
        test = pd.Series(False, index=frame.index)
    if oof is None:
        oof = pd.Series(False, index=frame.index)
    return test.fillna(False).astype(bool) | oof.fillna(False).astype(bool)


def _mark_trade_setup_bars(frame: pd.DataFrame, trades: pd.DataFrame, side: str) -> np.ndarray:
    marks = np.zeros(len(frame), dtype=bool)
    if trades.empty:
        return marks
    side_trades = trades[trades["side"].astype(str).str.lower().eq(side)].copy()
    if side_trades.empty:
        return marks
    times = pd.to_datetime(side_trades["setup_bar_time"], utc=True, errors="coerce").dt.tz_convert(frame.index.tz)
    locs = frame.index.get_indexer(pd.DatetimeIndex(times.dropna()), method="nearest", tolerance=pd.Timedelta(minutes=5))
    locs = locs[locs >= 0]
    marks[locs] = True
    return marks


def _load(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(args.signal_frame).sort_index()
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("America/New_York")

    labels = pd.read_parquet(args.labels).sort_index()
    if labels.index.tz is None:
        labels.index = labels.index.tz_localize(frame.index.tz)
    labels = labels.reindex(frame.index)

    frame["p_long"] = _prob(frame, "long")
    frame["p_short"] = _prob(frame, "short")
    frame["long_setup"] = _setup(frame, "long")
    frame["short_setup"] = _setup(frame, "short")
    frame["actual_long"] = pd.to_numeric(labels["long_swing_label"], errors="coerce").fillna(0).astype(bool)
    frame["actual_short"] = pd.to_numeric(labels["short_swing_label"], errors="coerce").fillna(0).astype(bool)

    trades = pd.read_csv(args.trades)
    frame["long_entry"] = _mark_trade_setup_bars(frame, trades, "long")
    frame["short_entry"] = _mark_trade_setup_bars(frame, trades, "short")
    frame["long_trigger"] = frame["long_entry"]
    frame["short_trigger"] = frame["short_entry"]
    return frame, trades


def _last_n_trading_days(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    sessions = pd.Index(pd.Series(frame.index.date).drop_duplicates())
    keep = set(sessions[-max(1, int(days)) :])
    return frame[pd.Series(frame.index.date, index=frame.index).isin(keep).to_numpy()].copy()


def _plot_diagnostics(
    work: pd.DataFrame,
    *,
    title: str,
    save_path: Path,
    long_threshold: float,
    short_threshold: float,
) -> None:
    if work.empty:
        raise SystemExit(f"No rows for {save_path}")

    pos, open_y, high_y, low_y, close_y = _extract_ohlc(work)
    marker_offset = _compute_marker_offset(work, high_y, low_y)
    tick_positions, tick_labels = _compute_time_ticks(work.index, pos, max_ticks=20)

    fig, (ax_price, ax_prob) = plt.subplots(
        2,
        1,
        figsize=(18, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    _plot_candles(ax_price, pos, open_y, high_y, low_y, close_y)

    long_setup = work["long_setup"].to_numpy(dtype=bool)
    short_setup = work["short_setup"].to_numpy(dtype=bool)
    long_entry = work["long_entry"].to_numpy(dtype=bool)
    short_entry = work["short_entry"].to_numpy(dtype=bool)
    long_trigger = work["long_trigger"].to_numpy(dtype=bool)
    short_trigger = work["short_trigger"].to_numpy(dtype=bool)
    actual_long = work["actual_long"].to_numpy(dtype=bool)
    actual_short = work["actual_short"].to_numpy(dtype=bool)

    long_setup_only = long_setup & ~long_entry
    short_setup_only = short_setup & ~short_entry
    long_trigger_only = long_trigger & ~long_entry
    short_trigger_only = short_trigger & ~short_entry

    if long_setup_only.any():
        ax_price.scatter(pos[long_setup_only], low_y[long_setup_only] - marker_offset * 0.45, marker=".", s=28, color="#90CAF9", alpha=0.55, label="LONG setup only")
    if short_setup_only.any():
        ax_price.scatter(pos[short_setup_only], high_y[short_setup_only] + marker_offset * 0.45, marker=".", s=28, color="#FFCC80", alpha=0.55, label="SHORT setup only")
    if long_trigger_only.any():
        ax_price.scatter(pos[long_trigger_only], low_y[long_trigger_only] - marker_offset * 0.72, marker="x", s=52, color="#42A5F5", alpha=0.75, label="LONG trigger candidate")
    if short_trigger_only.any():
        ax_price.scatter(pos[short_trigger_only], high_y[short_trigger_only] + marker_offset * 0.72, marker="x", s=52, color="#FFA726", alpha=0.75, label="SHORT trigger candidate")
    if long_entry.any():
        ax_price.scatter(pos[long_entry], low_y[long_entry] - marker_offset * 0.9, marker="^", s=90, color="#1565C0", label="triggered LONG")
    if short_entry.any():
        ax_price.scatter(pos[short_entry], high_y[short_entry] + marker_offset * 0.9, marker="v", s=90, color="#FB8C00", label="triggered SHORT")
    if actual_long.any():
        ax_price.scatter(pos[actual_long], low_y[actual_long] - marker_offset * 1.35, marker="^", s=58, facecolors="none", edgecolors="#0D47A1", linewidths=1.2, label="actual LONG")
    if actual_short.any():
        ax_price.scatter(pos[actual_short], high_y[actual_short] + marker_offset * 1.35, marker="v", s=58, facecolors="none", edgecolors="#EF6C00", linewidths=1.2, label="actual SHORT")

    ax_prob.plot(pos, work["p_long"], label="p_long setup", color="#1565C0", linewidth=1.5)
    ax_prob.plot(pos, work["p_short"], label="p_short setup", color="#FB8C00", linewidth=1.5)
    ax_prob.axhline(long_threshold, color="#1565C0", linestyle="--", linewidth=1.0, alpha=0.65, label=f"long setup thr={long_threshold:.2f}")
    ax_prob.axhline(short_threshold, color="#FB8C00", linestyle="--", linewidth=1.0, alpha=0.65, label=f"short setup thr={short_threshold:.2f}")
    ax_prob.set_ylim(0, 1.02)

    ax_price.set_title(title)
    ax_prob.set_title("Setup probabilities")
    ax_price.set_ylabel("Price")
    ax_prob.set_ylabel("Probability")
    ax_prob.set_xlabel("Session")
    ax_price.grid(True, alpha=0.25)
    ax_prob.grid(True, alpha=0.25)
    ax_price.legend(loc="best", fontsize=8, ncols=2)
    ax_prob.legend(loc="best", fontsize=8)
    _draw_day_lines((ax_price, ax_prob), tick_positions, line_color="#d0d0d0")
    _apply_time_ticks(ax_prob, tick_positions, tick_labels)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[shift1] saved plot {save_path}")


def main() -> None:
    args = _parse_args()
    frame, trades = _load(args)
    test = frame[frame["is_test"].fillna(False).astype(bool)].copy()
    out_dir = Path(args.out_dir)

    tail = _select_plot_window(test, window=args.tail, random_window=False)
    _plot_diagnostics(
        tail,
        title=f"SPY shift1 Phase 4 l42/s15 candle diagnostic EV=0.1540 ATR | tail {len(tail)} bars",
        save_path=out_dir / "shift1_phase4_style_tail_candles.png",
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
    )

    recent = _last_n_trading_days(test, args.days)
    _plot_diagnostics(
        recent,
        title=f"SPY shift1 Phase 4 l42/s15 candle diagnostic EV=0.1540 ATR | last {args.days} trading days",
        save_path=out_dir / "shift1_phase4_style_last45_candles.png",
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
    )

    print(f"test_rows={len(test)}")
    print(f"trades={len(trades)}")


if __name__ == "__main__":
    main()
