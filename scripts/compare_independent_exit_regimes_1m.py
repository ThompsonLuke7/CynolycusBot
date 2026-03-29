from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from API.Alpaca_API.inference.live_inference import LiveMetaXGBAgent
from scripts.plot_independent_1m_entry_trace_10m import _load_inputs as _load_plot_inputs
from scripts.plot_independent_1m_entry_trace_10m import _plot_sessions
from scripts.replay_meta_independent import _load_meta_matrix, _normalize_bounds, _score_exit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare 1m breakout-entry execution under two independent 10m exit regimes."
    )
    parser.add_argument(
        "--meta-matrix",
        default="Data/inference/spy/10min/debug_matrices_warmup/spy/live_meta_matrix_on_trace_ts_live_2026_03_24.parquet",
        help="Cached meta matrix parquet.",
    )
    parser.add_argument(
        "--one-min-data",
        default="Data/raw/spy/spy_intraday_1min_live_2026_03_24.parquet",
        help="Raw 1m parquet for execution timing.",
    )
    parser.add_argument(
        "--model-root",
        default="Data/models/meta_xgboost/10min",
        help="Meta model root directory.",
    )
    parser.add_argument("--symbol", default="SPY", help="Symbol label.")
    parser.add_argument("--start", default="2026-02-13T00:00:00Z", help="UTC start timestamp.")
    parser.add_argument("--end", default="2026-03-23T23:59:59Z", help="UTC end timestamp.")
    parser.add_argument("--tz", default="America/New_York", help="Display timezone.")
    parser.add_argument("--entry-threshold", type=float, default=None, help="Optional override for both entry thresholds.")
    parser.add_argument("--exit-threshold", type=float, default=None, help="Optional override for both exit thresholds.")
    parser.add_argument("--min-hold-bars", type=int, default=2, help="Minimum 10m bars to hold before honoring exit.")
    parser.add_argument("--exit-entry-delta", type=float, default=0.15, help="Current-rule exit-vs-entry dominance margin.")
    parser.add_argument("--confirm-bars", type=int, default=2, help="Consecutive bars for entry-below-threshold exit confirmation.")
    parser.add_argument(
        "--opposite-dominance-delta",
        type=float,
        default=0.0,
        help="Required opposite-side margin advantage to invalidate a side intent.",
    )
    parser.add_argument(
        "--current-trace-out",
        default="Data/inference/spy/10min/meta/meta_trace_independent_1m_entry_current_exit.csv",
        help="Current-regime 10m trace CSV.",
    )
    parser.add_argument(
        "--current-events-out",
        default="Data/inference/spy/10min/meta/meta_events_independent_1m_entry_current_exit.csv",
        help="Current-regime 1m events CSV.",
    )
    parser.add_argument(
        "--alt-trace-out",
        default="Data/inference/spy/10min/meta/meta_trace_independent_1m_entry_entry_falls_below_exit.csv",
        help="Alternative-regime 10m trace CSV.",
    )
    parser.add_argument(
        "--alt-events-out",
        default="Data/inference/spy/10min/meta/meta_events_independent_1m_entry_entry_falls_below_exit.csv",
        help="Alternative-regime 1m events CSV.",
    )
    parser.add_argument(
        "--summary-out",
        default="Data/inference/spy/10min/meta/exit_regime_comparison_1m_summary.csv",
        help="Comparison summary CSV.",
    )
    parser.add_argument(
        "--equity-out",
        default="Data/inference/spy/10min/plots/exit_regime_comparison_1m_equity.png",
        help="Comparison equity PNG.",
    )
    parser.add_argument(
        "--current-out-dir",
        default="Data/inference/spy/10min/plots/independent_1m_entry_current_exit_sessions",
        help="Directory for current-regime session PNGs.",
    )
    parser.add_argument(
        "--alt-out-dir",
        default="Data/inference/spy/10min/plots/independent_1m_entry_entry_falls_below_exit_sessions",
        help="Directory for alternative-regime session PNGs.",
    )
    parser.add_argument("--sessions-per-fig", type=int, default=4, help="Sessions per output PNG.")
    parser.add_argument(
        "--skip-session-plots",
        action="store_true",
        help="Write traces/events/summary/equity only, and skip the heavier session PNG generation.",
    )
    return parser.parse_args()


def _load_one_min(path: Path, *, symbol: str, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"1m data at {path} must contain a timestamp column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    if "symbol" in df.columns:
        df = df[df["symbol"].astype(str).str.upper() == str(symbol).upper()].copy()
    if start is not None:
        df = df[df["timestamp"] >= start]
    if end is not None:
        df = df[df["timestamp"] <= end]
    if df.empty:
        raise ValueError("1m data is empty after filtering.")
    return df.sort_values("timestamp").reset_index(drop=True)


def _validity_flags(
    *,
    p_enter_long: float,
    p_enter_short: float,
    thr_enter_long: float,
    thr_enter_short: float,
    opposite_dominance_delta: float,
) -> tuple[bool, bool]:
    long_ready = np.isfinite(p_enter_long) and p_enter_long >= thr_enter_long
    short_ready = np.isfinite(p_enter_short) and p_enter_short >= thr_enter_short
    long_margin = (p_enter_long - thr_enter_long) if long_ready else -np.inf
    short_margin = (p_enter_short - thr_enter_short) if short_ready else -np.inf
    long_invalidated = bool(short_ready and short_margin > long_margin + opposite_dominance_delta)
    short_invalidated = bool(long_ready and long_margin > short_margin + opposite_dominance_delta)
    return bool(long_ready and not long_invalidated), bool(short_ready and not short_invalidated)


def _run_regime(
    *,
    meta_df: pd.DataFrame,
    one_min: pd.DataFrame,
    model_root: Path,
    symbol: str,
    entry_threshold: float | None,
    exit_threshold: float | None,
    min_hold_bars: int,
    exit_entry_delta: float,
    confirm_bars: int,
    opposite_dominance_delta: float,
    exit_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_agent = LiveMetaXGBAgent(
        model_root=model_root,
        precomputed_base_frame=meta_df,
        entry_threshold_override=entry_threshold,
        exit_threshold_override=exit_threshold,
    )
    short_agent = LiveMetaXGBAgent(
        model_root=model_root,
        precomputed_base_frame=meta_df,
        entry_threshold_override=entry_threshold,
        exit_threshold_override=exit_threshold,
    )
    entry_long_probs = long_agent._entry_long.predict_frame(meta_df)
    entry_short_probs = long_agent._entry_short.predict_frame(meta_df)
    thresholds = long_agent.last_thresholds() or {
        "enter_long": np.nan,
        "enter_short": np.nan,
        "exit_long": np.nan,
        "exit_short": np.nan,
    }

    long_active = False
    short_active = False
    long_intent_active = False
    short_intent_active = False
    long_ref_high = np.nan
    short_ref_low = np.nan
    long_signal_row: pd.Series | None = None
    short_signal_row: pd.Series | None = None
    pending_long_exit = False
    pending_short_exit = False
    long_exit_confirm = 0
    short_exit_confirm = 0
    one_min_pos = 0

    trace_rows: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    meta_index = meta_df.index.to_list()

    for idx, (_, row) in enumerate(meta_df.iterrows()):
        ts = pd.Timestamp(row.name)
        next_ts = meta_index[idx + 1] if idx + 1 < len(meta_index) else (ts + pd.Timedelta(minutes=10))
        decision_ts = ts + pd.Timedelta(minutes=10)
        next_decision_ts = next_ts + pd.Timedelta(minutes=10)

        p_enter_long = float(entry_long_probs[idx]) if idx < entry_long_probs.size else float("nan")
        p_enter_short = float(entry_short_probs[idx]) if idx < entry_short_probs.size else float("nan")
        work_row = row.copy()
        work_row["p_enter_long_oof"] = p_enter_long
        work_row["p_enter_short_oof"] = p_enter_short

        long_valid_signal, short_valid_signal = _validity_flags(
            p_enter_long=p_enter_long,
            p_enter_short=p_enter_short,
            thr_enter_long=float(thresholds["enter_long"]),
            thr_enter_short=float(thresholds["enter_short"]),
            opposite_dominance_delta=float(opposite_dominance_delta),
        )

        if not long_active:
            if long_valid_signal:
                long_intent_active = True
                long_ref_high = float(row.get("high", np.nan))
                long_signal_row = row.copy()
            else:
                long_intent_active = False
                long_ref_high = np.nan
                long_signal_row = None
        if not short_active:
            if short_valid_signal:
                short_intent_active = True
                short_ref_low = float(row.get("low", np.nan))
                short_signal_row = row.copy()
            else:
                short_intent_active = False
                short_ref_low = np.nan
                short_signal_row = None

        p_exit_long = _score_exit(long_agent, work_row, side="long") if long_active else float("nan")
        p_exit_short = _score_exit(short_agent, work_row, side="short") if short_active else float("nan")

        long_hold_ready = bool(long_active and int(long_agent._state.bars_since_entry) >= int(min_hold_bars))
        short_hold_ready = bool(short_active and int(short_agent._state.bars_since_entry) >= int(min_hold_bars))

        long_entry_still_supports = bool(
            np.isfinite(p_enter_long)
            and p_enter_long >= float(thresholds["enter_long"])
            and (not np.isfinite(p_exit_long) or (p_exit_long - p_enter_long) < float(exit_entry_delta))
        )
        short_entry_still_supports = bool(
            np.isfinite(p_enter_short)
            and p_enter_short >= float(thresholds["enter_short"])
            and (not np.isfinite(p_exit_short) or (p_exit_short - p_enter_short) < float(exit_entry_delta))
        )

        if exit_mode == "current":
            long_exit_condition = bool(
                long_active
                and np.isfinite(p_exit_long)
                and p_exit_long >= float(thresholds["exit_long"])
                and long_hold_ready
                and not long_entry_still_supports
            )
            short_exit_condition = bool(
                short_active
                and np.isfinite(p_exit_short)
                and p_exit_short >= float(thresholds["exit_short"])
                and short_hold_ready
                and not short_entry_still_supports
            )
            do_exit_long = bool(long_exit_condition)
            do_exit_short = bool(short_exit_condition)
        elif exit_mode == "enter_falls_below":
            long_exit_condition = bool(
                long_active
                and long_hold_ready
                and np.isfinite(p_enter_long)
                and p_enter_long < float(thresholds["enter_long"])
            )
            short_exit_condition = bool(
                short_active
                and short_hold_ready
                and np.isfinite(p_enter_short)
                and p_enter_short < float(thresholds["enter_short"])
            )
            long_exit_confirm = long_exit_confirm + 1 if long_exit_condition else 0
            short_exit_confirm = short_exit_confirm + 1 if short_exit_condition else 0
            do_exit_long = bool(long_exit_confirm >= int(confirm_bars))
            do_exit_short = bool(short_exit_confirm >= int(confirm_bars))
        else:
            raise ValueError(f"Unknown exit mode: {exit_mode}")

        if long_active:
            long_agent._advance_state(action=0 if do_exit_long else 1, row=work_row)
            if do_exit_long:
                pending_long_exit = True
        if short_active:
            short_agent._advance_state(action=0 if do_exit_short else -1, row=work_row)
            if do_exit_short:
                pending_short_exit = True

        interval = one_min.iloc[one_min_pos:]
        if not interval.empty:
            interval = interval[(interval["timestamp"] >= decision_ts) & (interval["timestamp"] < next_decision_ts)]
        for _, bar in interval.iterrows():
            bar_ts = pd.Timestamp(bar["timestamp"])
            bar_open = float(bar.get("open", np.nan))
            bar_high = float(bar.get("high", np.nan))
            bar_low = float(bar.get("low", np.nan))

            if pending_long_exit and long_active and np.isfinite(bar_open):
                long_active = False
                pending_long_exit = False
                long_intent_active = False
                long_ref_high = np.nan
                long_signal_row = None
                long_exit_confirm = 0
                events.append({"timestamp": bar_ts, "symbol": symbol, "event": "exit_long", "price": bar_open})
            if pending_short_exit and short_active and np.isfinite(bar_open):
                short_active = False
                pending_short_exit = False
                short_intent_active = False
                short_ref_low = np.nan
                short_signal_row = None
                short_exit_confirm = 0
                events.append({"timestamp": bar_ts, "symbol": symbol, "event": "exit_short", "price": bar_open})

            if long_intent_active and (not long_active) and long_signal_row is not None and np.isfinite(long_ref_high):
                if np.isfinite(bar_high) and bar_high >= float(long_ref_high):
                    fill_price = max(float(long_ref_high), float(bar_open)) if np.isfinite(bar_open) else float(long_ref_high)
                    long_active = True
                    long_intent_active = False
                    long_agent._set_trade_entry(position=1, row=long_signal_row, entry_price=fill_price)
                    events.append({"timestamp": bar_ts, "symbol": symbol, "event": "enter_long", "price": fill_price})

            if short_intent_active and (not short_active) and short_signal_row is not None and np.isfinite(short_ref_low):
                if np.isfinite(bar_low) and bar_low <= float(short_ref_low):
                    fill_price = min(float(short_ref_low), float(bar_open)) if np.isfinite(bar_open) else float(short_ref_low)
                    short_active = True
                    short_intent_active = False
                    short_agent._set_trade_entry(position=-1, row=short_signal_row, entry_price=fill_price)
                    events.append({"timestamp": bar_ts, "symbol": symbol, "event": "enter_short", "price": fill_price})

            one_min_pos += 1

        trace_rows.append(
            {
                "symbol": symbol,
                "timestamp": ts,
                "open": float(row.get("open", np.nan)),
                "high": float(row.get("high", np.nan)),
                "low": float(row.get("low", np.nan)),
                "close": float(row.get("close", np.nan)),
                "volume": float(row.get("volume", np.nan)),
                "p_enter_long": p_enter_long,
                "p_enter_short": p_enter_short,
                "p_exit_long": p_exit_long,
                "p_exit_short": p_exit_short,
                "thr_enter_long": float(thresholds["enter_long"]),
                "thr_enter_short": float(thresholds["enter_short"]),
                "thr_exit_long": float(thresholds["exit_long"]),
                "thr_exit_short": float(thresholds["exit_short"]),
                "long_valid_signal": bool(long_valid_signal),
                "short_valid_signal": bool(short_valid_signal),
                "long_intent_active": bool(long_intent_active),
                "short_intent_active": bool(short_intent_active),
                "long_ref_high": float(long_ref_high) if np.isfinite(long_ref_high) else np.nan,
                "short_ref_low": float(short_ref_low) if np.isfinite(short_ref_low) else np.nan,
                "long_active": int(long_active),
                "short_active": int(short_active),
                "long_bars_held": int(long_agent._state.bars_since_entry) if long_active else 0,
                "short_bars_held": int(short_agent._state.bars_since_entry) if short_active else 0,
                "do_exit_long": bool(do_exit_long),
                "do_exit_short": bool(do_exit_short),
                "long_entry_still_supports": bool(long_entry_still_supports),
                "short_entry_still_supports": bool(short_entry_still_supports),
                "long_exit_confirm_count": int(long_exit_confirm),
                "short_exit_confirm_count": int(short_exit_confirm),
            }
        )

    return pd.DataFrame(trace_rows), pd.DataFrame(events)


def _event_metrics(events: pd.DataFrame) -> dict[str, float]:
    events = events.sort_values("timestamp").reset_index(drop=True).copy()
    long_entry_price: float | None = None
    short_entry_price: float | None = None
    long_eq = 1.0
    short_eq = 1.0
    long_trades = 0
    short_trades = 0
    long_wins = 0
    short_wins = 0

    for _, row in events.iterrows():
        event = str(row["event"])
        price = float(row["price"])
        if not np.isfinite(price) or price <= 0.0:
            continue
        if event == "enter_long":
            long_entry_price = price
        elif event == "exit_long" and long_entry_price is not None:
            factor = price / long_entry_price
            long_eq *= factor
            long_trades += 1
            if factor > 1.0:
                long_wins += 1
            long_entry_price = None
        elif event == "enter_short":
            short_entry_price = price
        elif event == "exit_short" and short_entry_price is not None:
            factor = 1.0 + (short_entry_price - price) / short_entry_price
            short_eq *= factor
            short_trades += 1
            if factor > 1.0:
                short_wins += 1
            short_entry_price = None

    return {
        "long_entries": float(int(events["event"].eq("enter_long").sum())),
        "long_exits": float(int(events["event"].eq("exit_long").sum())),
        "short_entries": float(int(events["event"].eq("enter_short").sum())),
        "short_exits": float(int(events["event"].eq("exit_short").sum())),
        "long_trades_closed": float(long_trades),
        "short_trades_closed": float(short_trades),
        "long_win_rate": float(long_wins / long_trades) if long_trades else np.nan,
        "short_win_rate": float(short_wins / short_trades) if short_trades else np.nan,
        "long_equity_end": float(long_eq),
        "short_equity_end": float(short_eq),
        "combined_full_gross_end": float(long_eq + short_eq - 1.0),
    }


def _equity_curve_from_events(events: pd.DataFrame, one_min: pd.DataFrame) -> pd.DataFrame:
    ev = events.sort_values("timestamp").reset_index(drop=True).copy()
    one = one_min.sort_values("timestamp").reset_index(drop=True).copy()
    if one.empty:
        raise ValueError("1m data is empty for equity curve.")

    long_on = False
    short_on = False
    long_eq = 1.0
    short_eq = 1.0
    net_1x = 1.0
    buy_hold = 1.0
    ev_idx = 0
    records: list[dict[str, object]] = []

    for i in range(len(one) - 1):
        ts = pd.Timestamp(one.loc[i, "timestamp"])
        while ev_idx < len(ev) and pd.Timestamp(ev.loc[ev_idx, "timestamp"]) <= ts:
            event = str(ev.loc[ev_idx, "event"])
            if event == "enter_long":
                long_on = True
            elif event == "exit_long":
                long_on = False
            elif event == "enter_short":
                short_on = True
            elif event == "exit_short":
                short_on = False
            ev_idx += 1

        open_i = float(one.loc[i, "open"])
        open_n = float(one.loc[i + 1, "open"])
        if not (np.isfinite(open_i) and np.isfinite(open_n) and open_i > 0.0 and open_n > 0.0):
            continue
        ret = open_n / open_i - 1.0
        buy_hold *= 1.0 + ret
        if long_on:
            long_eq *= 1.0 + ret
        if short_on:
            short_eq *= 1.0 - ret
        if long_on and not short_on:
            net_1x *= 1.0 + ret
        elif short_on and not long_on:
            net_1x *= 1.0 - ret
        records.append(
            {
                "timestamp": pd.Timestamp(one.loc[i + 1, "timestamp"]),
                "buy_hold": buy_hold,
                "long_only": long_eq,
                "short_only": short_eq,
                "combined_full_gross": long_eq + short_eq - 1.0,
                "net_1x_style": net_1x,
            }
        )

    return pd.DataFrame(records)


def _save_equity_plot(*, current_eq: pd.DataFrame, alt_eq: pd.DataFrame, save_path: Path, symbol: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    for eq, col, label, color, width in (
        (current_eq, "buy_hold", "buy_hold", "#444444", 1.5),
        (current_eq, "net_1x_style", "current_1m_exit_net_1x", "#1565C0", 2.0),
        (alt_eq, "net_1x_style", "entry_falls_below_1m_exit_net_1x", "#2E7D32", 2.0),
        (current_eq, "combined_full_gross", "current_1m_exit_full_gross", "#8E24AA", 1.5),
        (alt_eq, "combined_full_gross", "entry_falls_below_1m_exit_full_gross", "#C62828", 1.5),
    ):
        ax.plot(
            pd.to_datetime(eq["timestamp"], utc=True).dt.tz_convert("America/New_York"),
            eq[col],
            label=label,
            color=color,
            linewidth=width,
        )
    ax.set_title(f"{symbol} | 1m breakout entry exit regime comparison")
    ax.set_ylabel("Equity")
    ax.set_xlabel("Session Time (America/New_York)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    start, end = _normalize_bounds(args.start, args.end)
    meta_df = _load_meta_matrix(Path(args.meta_matrix), start=start, end=end, tz=args.tz)
    one_min = _load_one_min(Path(args.one_min_data), symbol=args.symbol, start=start, end=end)

    current_trace, current_events = _run_regime(
        meta_df=meta_df,
        one_min=one_min,
        model_root=Path(args.model_root),
        symbol=args.symbol,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        min_hold_bars=max(0, int(args.min_hold_bars)),
        exit_entry_delta=float(args.exit_entry_delta),
        confirm_bars=max(1, int(args.confirm_bars)),
        opposite_dominance_delta=float(args.opposite_dominance_delta),
        exit_mode="current",
    )
    alt_trace, alt_events = _run_regime(
        meta_df=meta_df,
        one_min=one_min,
        model_root=Path(args.model_root),
        symbol=args.symbol,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        min_hold_bars=max(0, int(args.min_hold_bars)),
        exit_entry_delta=float(args.exit_entry_delta),
        confirm_bars=max(1, int(args.confirm_bars)),
        opposite_dominance_delta=float(args.opposite_dominance_delta),
        exit_mode="enter_falls_below",
    )

    current_trace_path = Path(args.current_trace_out)
    current_events_path = Path(args.current_events_out)
    alt_trace_path = Path(args.alt_trace_out)
    alt_events_path = Path(args.alt_events_out)
    for path in (current_trace_path, current_events_path, alt_trace_path, alt_events_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    current_trace.to_csv(current_trace_path, index=False)
    current_events.to_csv(current_events_path, index=False)
    alt_trace.to_csv(alt_trace_path, index=False)
    alt_events.to_csv(alt_events_path, index=False)

    current_eq = _equity_curve_from_events(current_events, one_min)
    alt_eq = _equity_curve_from_events(alt_events, one_min)
    _save_equity_plot(current_eq=current_eq, alt_eq=alt_eq, save_path=Path(args.equity_out), symbol=args.symbol)

    summary = pd.DataFrame(
        [
            {"regime": "current_1m_exit_logic", **_event_metrics(current_events)},
            {"regime": "entry_falls_below_threshold_1m_exit", **_event_metrics(alt_events)},
        ]
    )
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    current_plots: list[Path] = []
    alt_plots: list[Path] = []
    if not bool(args.skip_session_plots):
        current_trace_plot_df, current_events_plot_df = _load_plot_inputs(
            current_trace_path,
            current_events_path,
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            tz=args.tz,
        )
        alt_trace_plot_df, alt_events_plot_df = _load_plot_inputs(
            alt_trace_path,
            alt_events_path,
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            tz=args.tz,
        )
        current_plots = _plot_sessions(
            current_trace_plot_df,
            current_events_plot_df,
            out_dir=Path(args.current_out_dir),
            sessions_per_fig=max(1, int(args.sessions_per_fig)),
            tz=args.tz,
            symbol=args.symbol,
        )
        alt_plots = _plot_sessions(
            alt_trace_plot_df,
            alt_events_plot_df,
            out_dir=Path(args.alt_out_dir),
            sessions_per_fig=max(1, int(args.sessions_per_fig)),
            tz=args.tz,
            symbol=args.symbol,
        )

    print(summary.to_string(index=False))
    print(f"\nsummary_csv={summary_path}")
    print(f"equity_png={Path(args.equity_out)}")
    print(f"current_trace_csv={current_trace_path}")
    print(f"current_events_csv={current_events_path}")
    print(f"alt_trace_csv={alt_trace_path}")
    print(f"alt_events_csv={alt_events_path}")
    if current_plots or alt_plots:
        print("current_plots:")
        for path in current_plots:
            print(path)
        print("alt_plots:")
        for path in alt_plots:
            print(path)


if __name__ == "__main__":
    main()
