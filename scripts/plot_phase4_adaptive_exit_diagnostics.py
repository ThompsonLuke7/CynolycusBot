from __future__ import annotations

import argparse
import json
import os
import re
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot phase4 adaptive exit diagnostics with entry triggers and exit reasons."
    )
    parser.add_argument(
        "--signal-frame",
        default=str(DEFAULT_ANALYSIS_DIR / "phase4_signal_frame.parquet"),
        help="Phase4 signal frame parquet with SPY OHLC.",
    )
    parser.add_argument(
        "--events",
        default=str(DEFAULT_ANALYSIS_DIR / "phase4_black_scholes_adaptive_exit_time_decay_selected_events.csv"),
        help="Selected adaptive-exit events CSV.",
    )
    parser.add_argument("--regime", default=DEFAULT_REGIME)
    parser.add_argument(
        "--overview-out",
        default=str(DEFAULT_ANALYSIS_DIR / "phase4_black_scholes_adaptive_exit_diagnostics_overview.png"),
    )
    parser.add_argument(
        "--sessions-out",
        default=str(DEFAULT_ANALYSIS_DIR / "phase4_black_scholes_adaptive_exit_diagnostics_sessions.png"),
    )
    parser.add_argument("--session-count", type=int, default=5)
    parser.add_argument(
        "--live-run-dir",
        default=None,
        help="Optional live-run audit directory to plot separately from the phase4 experiment frame.",
    )
    parser.add_argument(
        "--live-out",
        default=None,
        help="Output path for the optional live-run diagnostic plot.",
    )
    parser.add_argument(
        "--max-display-return-pct",
        type=float,
        default=1000.0,
        help="Hide larger absolute return outliers from the scatter display only.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _nearest_close(frame: pd.DataFrame, times: pd.Series) -> pd.Series:
    idx = pd.to_datetime(times, utc=True, errors="coerce").dt.tz_convert(frame.index.tz)
    close = frame["close"].sort_index()
    matched = close.reindex(pd.DatetimeIndex(idx), method="nearest", tolerance=pd.Timedelta(minutes=10))
    return pd.Series(matched.to_numpy(dtype=float), index=times.index)


def _load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(args.signal_frame).sort_index()
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("America/New_York")
    events = pd.read_csv(args.events)
    events = events[events["regime"].astype(str).eq(str(args.regime))].copy()
    if events.empty:
        raise SystemExit(f"No rows found for regime={args.regime}")
    events["entry_time"] = pd.to_datetime(events["entry_time"], utc=True, errors="coerce").dt.tz_convert(frame.index.tz)
    events["exit_time"] = pd.to_datetime(events["exit_time"], utc=True, errors="coerce").dt.tz_convert(frame.index.tz)
    events["entry_close"] = _nearest_close(frame, events["entry_time"])
    events["exit_close"] = _nearest_close(frame, events["exit_time"])
    events["outcome_pct_points"] = pd.to_numeric(events["outcome_pct"], errors="coerce") * 100.0
    events["entry_hour"] = events["entry_time"].dt.hour + events["entry_time"].dt.minute / 60.0
    return frame, events


def _plot_overview(events: pd.DataFrame, out_path: str, *, max_display_return_pct: float) -> None:
    colors = {
        "adaptive_trail": "#2ca25f",
        "time_decay": "#f59e0b",
        "opposite_signal": "#7c3aed",
        "timeout": "#64748b",
        "stop_loss": "#dc2626",
        "take_profit": "#0891b2",
    }
    reasons = list(events["exit_reason"].value_counts().index)
    display_events = events[
        pd.to_numeric(events["outcome_pct_points"], errors="coerce").abs() <= float(max_display_return_pct)
    ].copy()
    hidden_outliers = int(len(events) - len(display_events))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(
        f"Adaptive exit diagnostics: display capped at +/-{max_display_return_pct:.0f}% "
        f"({hidden_outliers} proxy outliers hidden)",
        fontsize=14,
    )

    ax = axes[0, 0]
    counts = events["exit_reason"].value_counts()
    ax.bar(counts.index, counts.values, color=[colors.get(x, "#334155") for x in counts.index])
    ax.set_title("Exit reason counts")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[0, 1]
    box_data = [events.loc[events["exit_reason"].eq(reason), "outcome_pct_points"].dropna() for reason in reasons]
    ax.boxplot(box_data, tick_labels=reasons, showfliers=False)
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.set_title("Outcome by exit reason, fliers hidden")
    ax.set_ylabel("Option proxy return (%)")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[1, 0]
    for reason in reasons:
        sub = display_events[display_events["exit_reason"].eq(reason)]
        ax.scatter(
            sub["entry_time"],
            sub["outcome_pct_points"],
            s=10,
            alpha=0.55,
            label=reason,
            color=colors.get(reason, "#334155"),
        )
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.set_title("Each open trigger by eventual exit reason")
    ax.set_ylabel("Option proxy return (%)")
    ax.set_ylim(-130.0, float(max_display_return_pct) * 1.05)
    ax.legend(loc="upper left", fontsize=8, ncols=2)

    ax = axes[1, 1]
    for reason in reasons:
        sub = events[events["exit_reason"].eq(reason)]
        ax.hist(
            sub["entry_hour"].dropna(),
            bins=26,
            range=(9.5, 16.0),
            alpha=0.45,
            label=reason,
            color=colors.get(reason, "#334155"),
        )
    ax.set_title("Open trigger time of day")
    ax.set_xlabel("ET hour")
    ax.legend(loc="upper right", fontsize=8)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_sessions(frame: pd.DataFrame, events: pd.DataFrame, out_path: str, session_count: int) -> None:
    colors = {
        "adaptive_trail": "#2ca25f",
        "time_decay": "#f59e0b",
        "opposite_signal": "#7c3aed",
        "timeout": "#64748b",
        "stop_loss": "#dc2626",
        "take_profit": "#0891b2",
    }
    events = events.copy()
    events["session"] = events["entry_time"].dt.date
    sessions = list(events["session"].dropna().drop_duplicates().sort_values().tail(max(1, int(session_count))))
    n = len(sessions)
    fig, axes = plt.subplots(n, 1, figsize=(16, max(4.5, n * 2.8)), squeeze=False, constrained_layout=True)
    fig.suptitle("Last available week: SPY close, setup order triggers, exits, and reasons", fontsize=14)

    for ax, session in zip(axes.ravel(), sessions):
        day_frame = frame[frame.index.date == session]
        day_events = events[events["session"].eq(session)]
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
            reason = str(row["exit_reason"])
            color = colors.get(str(reason), "#334155")
            entry_marker = "^" if str(row["side"]) == "long" else "v"
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
            ax.plot(
                [entry_pos, exit_pos],
                [row["entry_close"], row["exit_close"]],
                color=color,
                linewidth=0.9,
                alpha=0.55,
            )
            ax.scatter(entry_pos, row["entry_close"], marker=entry_marker, s=74, color=color, edgecolor="white", linewidth=0.7)
            ax.scatter(exit_pos, row["exit_close"], marker="X", s=70, color=color, edgecolor="white", linewidth=0.7)
            label = f"{reason}\\n{float(row['outcome_pct_points']):+.0f}%"
            ax.annotate(
                label,
                xy=(exit_pos, row["exit_close"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                color=color,
                alpha=0.95,
            )
        long_count = int(day_events["side"].eq("long").sum())
        short_count = int(day_events["side"].eq("short").sum())
        ax.set_title(f"{session} | triggers={len(day_events)} long={long_count} short={short_count}")
        tick_step = max(1, len(day_frame) // 8)
        tick_pos = pos[::tick_step]
        tick_labels = day_frame.index[::tick_step].strftime("%H:%M").to_list()
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=15, ha="right", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, alpha=0.18)
        ax.set_ylabel("SPY")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, label=reason, markersize=8)
        for reason, color in colors.items()
        if reason in set(events["exit_reason"].astype(str))
    ]
    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center", ncols=min(5, len(legend_handles)), fontsize=8)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_live_decisions(run_dir: Path) -> pd.DataFrame:
    rows = _read_jsonl(run_dir / "decision-10m.jsonl")
    if not rows:
        rows = _read_jsonl(run_dir / "actions.jsonl")
    bars = []
    for row in rows:
        payload = row.get("payload") or {}
        bar = dict(payload.get("bar") or {})
        if not bar:
            continue
        bar["action"] = (payload.get("action") or {}).get("selected_action_class")
        state = payload.get("agent_state") or {}
        for key, value in (state.get("last_prob_sources") or {}).items():
            bar[key] = value
        bars.append(bar)
    if not bars:
        raise SystemExit(f"No live decision bars found under {run_dir}")
    frame = pd.DataFrame(bars)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    frame = frame.dropna(subset=["timestamp"]).drop_duplicates("timestamp").set_index("timestamp").sort_index()
    for col in [
        "open",
        "high",
        "low",
        "close",
        "p_enter_long",
        "p_enter_short",
        "p_swing_setup_long",
        "p_swing_setup_short",
        "thr_enter_long",
        "thr_enter_short",
    ]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _extract_live_orders(run_dir: Path, session_dates: set) -> pd.DataFrame:
    orders_by_id: dict[str, dict] = {}
    for row in _read_jsonl(run_dir / "order-policy.jsonl"):
        broker_state = ((row.get("payload") or {}).get("broker_state") or {})
        for order in broker_state.get("recent_orders") or []:
            order_id = str(order.get("id") or "")
            if order_id:
                orders_by_id[order_id] = dict(order)
    if not orders_by_id:
        return pd.DataFrame()

    orders = pd.DataFrame(orders_by_id.values())
    orders["fill_time"] = pd.to_datetime(orders.get("filled_at"), utc=True, errors="coerce").dt.tz_convert("America/New_York")
    orders["submit_time"] = pd.to_datetime(orders.get("submitted_at"), utc=True, errors="coerce").dt.tz_convert("America/New_York")
    orders["filled_avg_price"] = pd.to_numeric(orders.get("filled_avg_price"), errors="coerce")
    orders = orders[orders["fill_time"].dt.date.isin(session_dates)]
    orders = orders[orders["status"].astype(str).eq("filled")]
    orders = orders[orders["symbol"].astype(str).str.startswith("SPY")]
    return orders.sort_values("fill_time")


def _extract_live_close_reason(run_dir: Path) -> str | None:
    reasons: list[str] = []
    for row in _read_jsonl(run_dir / "order-policy.jsonl"):
        payload = row.get("payload") or {}
        result = payload.get("result") or {}
        event = str(result.get("event") or "")
        state = payload.get("policy_state") or {}
        if event.startswith("close"):
            for key in ("long_decision_reason", "short_decision_reason"):
                value = state.get(key)
                if value and value != "entry_waiting_for_1m_confirmation":
                    reasons.append(str(value))
    return reasons[-1] if reasons else None


def _live_policy_thresholds(run_dir: Path) -> tuple[float | None, float | None]:
    meta_path = run_dir / "session_meta.json"
    if not meta_path.exists():
        return None, None
    try:
        config = json.loads(meta_path.read_text()).get("config") or {}
    except Exception:
        return None, None
    long_thr = config.get("meta_intrabar_long_setup_threshold")
    short_thr = config.get("meta_intrabar_short_setup_threshold")
    return (
        float(long_thr) if long_thr is not None else None,
        float(short_thr) if short_thr is not None else None,
    )


def _extract_live_log_events(run_dir: Path) -> list[tuple[pd.Timestamp, str]]:
    events: list[tuple[pd.Timestamp, str]] = []
    pattern = re.compile(r"ORDER (SUBMITTED|VERIFIED).*?intent=(open|close).*?symbol=([A-Z0-9]+)")
    for row in _read_jsonl(run_dir / "logs.jsonl"):
        payload = row.get("payload") or {}
        message = str(payload.get("message") or "")
        match = pattern.search(message)
        if not match:
            continue
        ts = pd.to_datetime(row.get("recorded_at"), utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        status, intent, symbol = match.groups()
        events.append((ts.tz_convert("America/New_York"), f"{intent} {status.lower()} {symbol}"))
    return events


def _fractional_time_pos(index: pd.DatetimeIndex, ts: pd.Timestamp) -> float | None:
    if pd.isna(ts) or len(index) == 0:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(index.tz)
    else:
        ts = ts.tz_convert(index.tz)
    if ts <= index[0]:
        return 0.0
    if ts >= index[-1]:
        return float(len(index) - 1)
    right = int(index.searchsorted(ts, side="right"))
    left = max(0, right - 1)
    right = min(len(index) - 1, right)
    span = (index[right] - index[left]).total_seconds()
    if span <= 0:
        return float(left)
    return float(left) + (ts - index[left]).total_seconds() / span


def _plot_live_run(run_dir: Path, out_path: str) -> None:
    frame = _load_live_decisions(run_dir)
    orders = _extract_live_orders(run_dir, set(frame.index.date))
    close_reason = _extract_live_close_reason(run_dir)
    log_events = _extract_live_log_events(run_dir)
    policy_long_thr, policy_short_thr = _live_policy_thresholds(run_dir)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.25]},
        constrained_layout=True,
    )
    ax = axes[0]
    pos, open_y, high_y, low_y, close_y = _extract_ohlc(frame)
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
    long_thr = policy_long_thr if policy_long_thr is not None else frame.get("thr_enter_long", 0.5)
    short_thr = policy_short_thr if policy_short_thr is not None else frame.get("thr_enter_short", 0.5)
    long_hits = frame["p_enter_long"].ge(long_thr)
    short_hits = frame["p_enter_short"].ge(short_thr)
    ax.scatter(pos[long_hits], frame.loc[long_hits, "low"], marker="^", color="#2563eb", s=70, label="long setup >= threshold")
    ax.scatter(pos[short_hits], frame.loc[short_hits, "high"], marker="v", color="#f97316", s=70, label="short setup >= threshold")

    for _, order in orders.iterrows():
        order_pos = _fractional_time_pos(frame.index, order["fill_time"])
        if order_pos is None:
            continue
        close = frame["close"].iloc[int(round(order_pos))]
        is_buy = str(order["side"]).lower() == "buy"
        color = "#22c55e" if is_buy else "#ef4444"
        marker = "*" if is_buy else "X"
        label = "filled buy/open" if is_buy else "filled sell/close"
        ax.scatter(order_pos, close, marker=marker, color=color, edgecolor="white", linewidth=0.8, s=160, label=label, zorder=5)
        text = f"{str(order['side']).upper()} {order['symbol']}\nfill {float(order['filled_avg_price']):.2f}"
        if not is_buy and close_reason:
            text += f"\n{close_reason}"
        ax.annotate(
            text,
            xy=(order_pos, close),
            xytext=(10, -34 if is_buy else 12),
            textcoords="offset points",
            fontsize=8,
            color=color,
            bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.72, "boxstyle": "round,pad=0.22"},
        )

    for ts, label in log_events:
        event_pos = _fractional_time_pos(frame.index, ts)
        if event_pos is not None:
            ax.axvline(event_pos, color="#94a3b8", linestyle="--", linewidth=0.8, alpha=0.35)

    title_date = frame.index.min().strftime("%Y-%m-%d")
    ax.set_title(f"Live run {run_dir.name}: SPY candles, setup triggers, and filled option orders ({title_date})")
    ax.set_ylabel("SPY")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.18)

    prob_ax = axes[1]
    prob_ax.plot(pos, frame["p_enter_long"], color="#2563eb", linewidth=1.5, label="p_enter_long")
    prob_ax.plot(pos, frame["p_enter_short"], color="#f97316", linewidth=1.5, label="p_enter_short")
    if policy_long_thr is not None:
        prob_ax.axhline(float(policy_long_thr), color="#2563eb", linestyle="--", linewidth=0.9, alpha=0.7, label="long policy threshold")
    elif "thr_enter_long" in frame:
        prob_ax.plot(pos, frame["thr_enter_long"], color="#2563eb", linestyle="--", linewidth=0.9, alpha=0.7, label="long threshold")
    if policy_short_thr is not None:
        prob_ax.axhline(float(policy_short_thr), color="#f97316", linestyle="--", linewidth=0.9, alpha=0.7, label="short policy threshold")
    elif "thr_enter_short" in frame:
        prob_ax.plot(pos, frame["thr_enter_short"], color="#f97316", linestyle="--", linewidth=0.9, alpha=0.7, label="short threshold")
    prob_ax.set_ylim(-0.05, 1.05)
    prob_ax.set_ylabel("probability")
    prob_ax.legend(loc="upper left", ncols=4, fontsize=8)
    prob_ax.grid(True, alpha=0.18)

    tick_step = max(1, len(frame) // 12)
    tick_pos = pos[::tick_step]
    tick_labels = frame.index[::tick_step].strftime("%H:%M").to_list()
    prob_ax.set_xticks(tick_pos)
    prob_ax.set_xticklabels(tick_labels, rotation=15, ha="right", fontsize=8)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"live_run={run_dir}")
    print(f"live_bars={len(frame)} live_orders={len(orders)}")
    print(f"live_plot={out_path}")


def main() -> None:
    args = _parse_args()
    frame, events = _load_inputs(args)
    _plot_overview(events, args.overview_out, max_display_return_pct=float(args.max_display_return_pct))
    _plot_sessions(frame, events, args.sessions_out, args.session_count)
    print(f"events={len(events)} regime={args.regime}")
    print(f"overview={args.overview_out}")
    print(f"sessions={args.sessions_out}")
    if args.live_run_dir:
        live_out = args.live_out
        if live_out is None:
            live_out = str(Path(args.live_run_dir) / "live_candlestick_setup_orders.png")
        _plot_live_run(Path(args.live_run_dir), live_out)


if __name__ == "__main__":
    main()
