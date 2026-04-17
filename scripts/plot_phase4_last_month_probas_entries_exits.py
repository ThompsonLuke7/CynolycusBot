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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from Data.plots.plots import _extract_ohlc, _plot_candles


DEFAULT_ANALYSIS_DIR = Path(
    "Data/models/ga_xgboost/10min/analysis/"
    "phase4_1m_oof_focused_trigger_sweep_l42_s15_full_1m_train"
)
DEFAULT_REGIME = "late_hybrid_sl1.00_arm2.0_gb0.25_stale8_prog1.00_opp0.60"
DEFAULT_EVENTS = (
    DEFAULT_ANALYSIS_DIR
    / "phase4_black_scholes_adaptive_exit_1m_confirmed_selected_events.csv"
)
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
        description="Plot last-month 10m setup probabilities with entry and exit events."
    )
    parser.add_argument("--signal-frame", default=str(DEFAULT_ANALYSIS_DIR / "phase4_signal_frame.parquet"))
    parser.add_argument(
        "--events",
        default=str(DEFAULT_EVENTS),
    )
    parser.add_argument("--regime", default=DEFAULT_REGIME)
    parser.add_argument(
        "--out",
        default=str(DEFAULT_ANALYSIS_DIR / "phase4_last_month_10m_probas_entries_exits.png"),
    )
    parser.add_argument("--days", type=int, default=31)
    parser.add_argument("--long-threshold", type=float, default=0.35)
    parser.add_argument("--short-threshold", type=float, default=0.65)
    return parser.parse_args()


def _prob_series(frame: pd.DataFrame, side: str) -> pd.Series:
    candidates = (
        [f"p_enter_{side}", f"p_swing_setup_{side}", f"p_{side}_test", f"p_{side}_oof_train"]
        if side in {"long", "short"}
        else []
    )
    out: pd.Series | None = None
    for col in candidates:
        if col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce")
            out = values if out is None else out.combine_first(values)
    if out is None:
        out = pd.Series(index=frame.index, dtype=float)
    setup_col = f"{side}_setup_test"
    setup_oof_col = f"{side}_setup_oof"
    if out.isna().all() and setup_col in frame.columns:
        out = pd.to_numeric(frame[setup_col], errors="coerce")
    if out.isna().all() and setup_oof_col in frame.columns:
        out = pd.to_numeric(frame[setup_oof_col], errors="coerce")
    return out


def _nearest_price(frame: pd.DataFrame, times: pd.Series) -> pd.Series:
    idx = pd.to_datetime(times, utc=True, errors="coerce").dt.tz_convert(frame.index.tz)
    close = frame["close"].sort_index()
    matched = close.reindex(pd.DatetimeIndex(idx), method="nearest", tolerance=pd.Timedelta(minutes=10))
    return pd.Series(matched.to_numpy(dtype=float), index=times.index)


def _event_numeric(events: pd.DataFrame, primary: str, fallback: str | None = None) -> pd.Series:
    if primary in events.columns:
        values = pd.to_numeric(events[primary], errors="coerce")
        if not values.isna().all():
            return values
    if fallback is not None and fallback in events.columns:
        return pd.to_numeric(events[fallback], errors="coerce")
    return pd.Series(index=events.index, dtype=float)


def _load(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(args.signal_frame).sort_index()
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("America/New_York")
    frame["p_plot_long"] = _prob_series(frame, "long")
    frame["p_plot_short"] = _prob_series(frame, "short")

    events = pd.read_csv(args.events)
    events = events[events["regime"].astype(str).eq(str(args.regime))].copy()
    if events.empty:
        raise SystemExit(f"No events found for regime={args.regime}")
    events["entry_time"] = pd.to_datetime(events["entry_time"], utc=True, errors="coerce").dt.tz_convert(frame.index.tz)
    events["exit_time"] = pd.to_datetime(events["exit_time"], utc=True, errors="coerce").dt.tz_convert(frame.index.tz)
    events["entry_close"] = _event_numeric(events, "entry_price")
    events["entry_close"] = events["entry_close"].combine_first(_nearest_price(frame, events["entry_time"]))
    events["exit_close"] = _event_numeric(events, "exit_price")
    events["exit_close"] = events["exit_close"].combine_first(_nearest_price(frame, events["exit_time"]))
    events["entry_premium_proxy"] = _event_numeric(events, "entry_premium_proxy", "premium_proxy")
    events["peak_premium_proxy"] = _event_numeric(events, "peak_premium_proxy")
    events["exit_premium_proxy"] = _event_numeric(events, "exit_premium_proxy")
    events["outcome_pct_points"] = pd.to_numeric(events["outcome_pct"], errors="coerce") * 100.0
    events = events.dropna(subset=["entry_time", "exit_time", "entry_close", "exit_close"])

    latest = min(frame.index.max(), events["entry_time"].max())
    start = latest - pd.Timedelta(days=max(1, int(args.days)))
    frame = frame[(frame.index >= start) & (frame.index <= latest)].copy()
    events = events[events["entry_time"].between(start, latest)].copy()
    return frame, events


def _time_to_pos(index: pd.DatetimeIndex, times: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(times, utc=True, errors="coerce").dt.tz_convert(index.tz)
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
    return pd.Series(out, index=times.index, dtype=float)


def _draw_session_lines(ax, frame: pd.DataFrame) -> None:
    sessions = pd.Series(frame.index.date, index=frame.index)
    starts = sessions.ne(sessions.shift(1))
    for pos in [i for i, is_start in enumerate(starts.to_numpy()) if is_start and i > 0]:
        ax.axvline(pos - 0.5, color="#94a3b8", linewidth=0.6, alpha=0.32, zorder=0)


def _premium_label(row: pd.Series) -> str:
    outcome = row.get("outcome_pct_points")
    mfe = pd.to_numeric(pd.Series([row.get("mfe_pct")]), errors="coerce").iloc[0] * 100.0
    parts = [str(row.get("exit_reason", "exit"))]
    if pd.notna(outcome):
        parts.append(f"{outcome:+.0f}%")
    if pd.notna(mfe):
        parts.append(f"peak {mfe:+.0f}%")
    return " ".join(parts)


def plot(frame: pd.DataFrame, events: pd.DataFrame, args: argparse.Namespace) -> None:
    if frame.empty:
        raise SystemExit("No frame rows in selected date window.")
    if events.empty:
        raise SystemExit("No events in selected date window.")

    pos, open_y, high_y, low_y, close_y = _extract_ohlc(frame)
    fig, ax_price = plt.subplots(1, 1, figsize=(28, 10), constrained_layout=True)

    _plot_candles(
        ax_price,
        pos,
        open_y,
        high_y,
        low_y,
        close_y,
        up_color="#16a34a",
        down_color="#dc2626",
        wick_color="#475569",
        width=0.64,
    )
    _draw_session_lines(ax_price, frame)

    price_min = float(pd.Series(low_y).min())
    price_max = float(pd.Series(high_y).max())
    price_span = max(1.0, price_max - price_min)
    prob_base = price_min - price_span * 0.18
    prob_height = price_span * 0.13
    long_prob_y = prob_base + pd.to_numeric(frame["p_plot_long"], errors="coerce").clip(0.0, 1.0) * prob_height
    short_prob_y = prob_base + pd.to_numeric(frame["p_plot_short"], errors="coerce").clip(0.0, 1.0) * prob_height
    long_thr_y = prob_base + float(args.long_threshold) * prob_height
    short_thr_y = prob_base + float(args.short_threshold) * prob_height
    ax_price.fill_between(pos, prob_base, long_prob_y.to_numpy(dtype=float), color="#2563eb", alpha=0.14, linewidth=0.0)
    ax_price.fill_between(pos, prob_base, short_prob_y.to_numpy(dtype=float), color="#d97706", alpha=0.14, linewidth=0.0)
    ax_price.plot(pos, long_prob_y, color="#2563eb", linewidth=0.9, alpha=0.75, label="long setup probability band")
    ax_price.plot(pos, short_prob_y, color="#d97706", linewidth=0.9, alpha=0.75, label="short setup probability band")
    ax_price.axhline(long_thr_y, color="#2563eb", linestyle="--", linewidth=0.7, alpha=0.45)
    ax_price.axhline(short_thr_y, color="#d97706", linestyle="--", linewidth=0.7, alpha=0.45)
    ax_price.text(
        0.01,
        0.02,
        "Probability bands share this SPY price axis: blue=long, amber=short",
        transform=ax_price.transAxes,
        fontsize=9,
        color="#334155",
        ha="left",
        va="bottom",
    )
    ax_price.set_ylim(prob_base - price_span * 0.03, price_max + price_span * 0.05)

    entry_pos = _time_to_pos(frame.index, events["entry_time"])
    exit_pos = _time_to_pos(frame.index, events["exit_time"])
    plotted_labels: set[str] = set()
    for i, row in events.iterrows():
        ep = entry_pos.loc[i]
        xp = exit_pos.loc[i]
        if pd.isna(ep) or pd.isna(xp):
            continue
        reason = str(row.get("exit_reason", "exit"))
        color = REASON_COLORS.get(reason, "#334155")
        side = str(row.get("side", "long"))
        entry_marker = "^" if side == "long" else "v"
        entry_color = "#16a34a" if side == "long" else "#dc2626"
        exit_label = f"{reason} exit"
        ax_price.plot([ep, xp], [row["entry_close"], row["exit_close"]], color=color, alpha=0.45, linewidth=1.0, zorder=3)
        ax_price.scatter(
            ep,
            row["entry_close"],
            marker=entry_marker,
            color=entry_color,
            edgecolor="#111827",
            linewidth=0.45,
            s=74,
            zorder=5,
        )
        ax_price.scatter(
            xp,
            row["exit_close"],
            marker="X",
            color=color,
            edgecolor="white",
            linewidth=0.6,
            s=72,
            label=exit_label if exit_label not in plotted_labels else None,
            zorder=6,
        )
        ax_price.annotate(
            _premium_label(row),
            (xp, row["exit_close"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.5,
            color=color,
            alpha=0.85,
            zorder=7,
        )
        plotted_labels.add(exit_label)

    tick_count = 18
    tick_idx = [round(i * (len(frame) - 1) / max(1, tick_count - 1)) for i in range(tick_count)]
    tick_idx = sorted(set(tick_idx))
    ax_price.set_xticks([pos[i] for i in tick_idx])
    ax_price.set_xticklabels([frame.index[i].strftime("%m/%d %H:%M") for i in tick_idx], rotation=25, ha="right")

    counts = events["exit_reason"].value_counts().to_dict()
    count_text = ", ".join(f"{k}:{v}" for k, v in counts.items())
    ax_price.set_title(
        "Last-month Phase4 10m setup probabilities with 1m-confirmed entries and adaptive exits\n"
        f"{frame.index.min():%Y-%m-%d} to {frame.index.max():%Y-%m-%d} | trades={len(events)} | {count_text}",
        loc="left",
        fontsize=13,
    )
    ax_price.set_ylabel("SPY 10m candles")
    ax_price.grid(True, alpha=0.16)
    legend_handles = [
        Line2D([0], [0], color="#2563eb", lw=1.2, label="long setup probability band"),
        Line2D([0], [0], color="#d97706", lw=1.2, label="short setup probability band"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#16a34a", markeredgecolor="#111827", label="long entry", markersize=7),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="#dc2626", markeredgecolor="#111827", label="short entry", markersize=7),
    ]
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            marker="X",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            label=f"{reason} exit",
            markersize=7,
        )
        for reason, color in REASON_COLORS.items()
        if reason in set(events["exit_reason"].astype(str))
    )
    ax_price.legend(handles=legend_handles, loc="upper left", ncols=5, fontsize=8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    frame, events = _load(args)
    plot(frame, events, args)
    print(f"rows={len(frame)}")
    print(f"events={len(events)}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
