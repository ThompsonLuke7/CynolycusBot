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
import pandas as pd

from Data.plots.plots import _extract_ohlc, _plot_candles


DEFAULT_ANALYSIS_DIR = Path(
    "Data/models/ga_xgboost/10min/analysis/"
    "phase4_1m_oof_focused_trigger_sweep_l42_s15_full_1m_train"
)
DEFAULT_REGIME = "late_hybrid_sl1.00_arm2.0_gb0.25_stale8_prog1.00_opp0.60"
REASON_COLORS = {
    "adaptive_trail": "#16a34a",
    "time_decay": "#f59e0b",
    "opposite_signal": "#7c3aed",
    "timeout": "#64748b",
    "stop_loss": "#dc2626",
    "take_profit": "#0891b2",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay best phase4 trigger entries with selected adaptive exits on SPY candles."
    )
    parser.add_argument("--signal-frame", default=str(DEFAULT_ANALYSIS_DIR / "phase4_signal_frame.parquet"))
    parser.add_argument(
        "--events",
        default=str(DEFAULT_ANALYSIS_DIR / "phase4_black_scholes_adaptive_exit_time_decay_selected_events.csv"),
    )
    parser.add_argument("--regime", default=DEFAULT_REGIME)
    parser.add_argument(
        "--out",
        default=str(DEFAULT_ANALYSIS_DIR / "phase4_best_trigger_adaptive_exit_last_month_overlay.png"),
    )
    parser.add_argument("--days", type=int, default=30, help="Calendar days from the latest selected entry.")
    parser.add_argument("--max-sessions", type=int, default=22, help="Maximum daily panels to draw.")
    return parser.parse_args()


def _nearest_close(frame: pd.DataFrame, times: pd.Series) -> pd.Series:
    idx = pd.to_datetime(times, utc=True, errors="coerce").dt.tz_convert(frame.index.tz)
    close = frame["close"].sort_index()
    matched = close.reindex(pd.DatetimeIndex(idx), method="nearest", tolerance=pd.Timedelta(minutes=10))
    return pd.Series(matched.to_numpy(dtype=float), index=times.index)


def _load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(args.signal_frame).sort_index()
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("America/New_York")

    events = pd.read_csv(args.events)
    events = events[events["regime"].astype(str).eq(str(args.regime))].copy()
    if events.empty:
        raise SystemExit(f"No events found for regime={args.regime}")
    events["entry_time"] = pd.to_datetime(events["entry_time"], utc=True, errors="coerce").dt.tz_convert(frame.index.tz)
    events["exit_time"] = pd.to_datetime(events["exit_time"], utc=True, errors="coerce").dt.tz_convert(frame.index.tz)
    events["entry_close"] = _nearest_close(frame, events["entry_time"])
    events["exit_close"] = _nearest_close(frame, events["exit_time"])
    events["outcome_pct_points"] = pd.to_numeric(events["outcome_pct"], errors="coerce") * 100.0
    events = events.dropna(subset=["entry_time", "exit_time", "entry_close", "exit_close"])

    latest_entry = events["entry_time"].max()
    start = latest_entry - pd.Timedelta(days=max(1, int(args.days)))
    events = events[events["entry_time"].between(start, latest_entry)].copy()
    events["session"] = events["entry_time"].dt.date
    sessions = list(events["session"].drop_duplicates().sort_values())
    if len(sessions) > int(args.max_sessions):
        keep = set(sessions[-int(args.max_sessions) :])
        events = events[events["session"].isin(keep)].copy()
    return frame, events


def _plot_overlay(frame: pd.DataFrame, events: pd.DataFrame, out_path: str, title: str) -> None:
    sessions = list(events["session"].drop_duplicates().sort_values())
    if not sessions:
        raise SystemExit("No selected events remain after date filtering.")

    n = len(sessions)
    fig_height = max(8.0, min(46.0, n * 2.05))
    fig, axes = plt.subplots(n, 1, figsize=(18, fig_height), squeeze=False, constrained_layout=True)
    fig.suptitle(title, fontsize=15)

    for ax, session in zip(axes.ravel(), sessions):
        day_frame = frame[frame.index.date == session]
        day_events = events[events["session"].eq(session)]
        if day_frame.empty:
            ax.set_axis_off()
            continue

        pos, open_y, high_y, low_y, close_y = _extract_ohlc(day_frame)
        _plot_candles(
            ax,
            pos,
            open_y,
            high_y,
            low_y,
            close_y,
            up_color="#16a34a",
            down_color="#dc2626",
            wick_color="#374151",
            width=0.68,
        )
        time_to_pos = pd.Series(pos, index=day_frame.index)

        for _, row in day_events.sort_values("entry_time").iterrows():
            color = REASON_COLORS.get(str(row["exit_reason"]), "#334155")
            entry_pos = time_to_pos.reindex(
                pd.DatetimeIndex([row["entry_time"]]),
                method="nearest",
                tolerance=pd.Timedelta(minutes=10),
            ).iloc[0]
            exit_pos = time_to_pos.reindex(
                pd.DatetimeIndex([row["exit_time"]]),
                method="nearest",
                tolerance=pd.Timedelta(minutes=10),
            ).iloc[0]
            if pd.isna(entry_pos) or pd.isna(exit_pos):
                continue
            entry_marker = "^" if str(row["side"]) == "long" else "v"
            ax.plot(
                [entry_pos, exit_pos],
                [row["entry_close"], row["exit_close"]],
                color=color,
                linewidth=1.0,
                alpha=0.52,
                zorder=3,
            )
            ax.scatter(
                entry_pos,
                row["entry_close"],
                marker=entry_marker,
                s=58,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
            )
            ax.scatter(
                exit_pos,
                row["exit_close"],
                marker="X",
                s=58,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )

        reason_counts = day_events["exit_reason"].value_counts().to_dict()
        reason_text = ", ".join(f"{k}:{v}" for k, v in reason_counts.items())
        ax.set_title(
            f"{session} | entries={len(day_events)} | {reason_text}",
            fontsize=9,
            loc="left",
        )
        tick_step = max(1, len(day_frame) // 8)
        ax.set_xticks(pos[::tick_step])
        ax.set_xticklabels(day_frame.index[::tick_step].strftime("%H:%M").to_list(), rotation=15, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.16)
        ax.set_ylabel("SPY", fontsize=8)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, label=reason, markersize=7)
        for reason, color in REASON_COLORS.items()
        if reason in set(events["exit_reason"].astype(str))
    ]
    legend_handles.extend(
        [
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#111827", label="entry long", markersize=7),
            plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="#111827", label="entry short", markersize=7),
            plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#111827", label="exit", markersize=7),
        ]
    )
    fig.legend(handles=legend_handles, loc="lower center", ncols=min(8, len(legend_handles)), fontsize=8)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    frame, events = _load_data(args)
    title = (
        "Best trigger entries + selected adaptive exits on SPY candles | "
        f"{events['entry_time'].min().date()} to {events['entry_time'].max().date()} | "
        f"trades={len(events)}"
    )
    _plot_overlay(frame, events, args.out, title)
    print(f"events={len(events)}")
    print(f"start={events['entry_time'].min()} end={events['entry_time'].max()}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
