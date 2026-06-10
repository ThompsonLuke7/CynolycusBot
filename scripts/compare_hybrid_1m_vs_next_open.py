from __future__ import annotations

import argparse
import sys
from math import ceil
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.API.Alpaca_API.inference.live_inference import LiveMetaXGBAgent
from scripts.replay_meta_independent import _load_meta_matrix, _normalize_bounds, _score_exit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare 1m breakout entries vs next-10m-open entries using the same hybrid exit rule."
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
    parser.add_argument("--min-hold-bars", type=int, default=2, help="Minimum 10m bars to hold before soft exits.")
    parser.add_argument(
        "--soft-exit-confirm-bars",
        type=int,
        default=2,
        help="Consecutive 10m bars with p_enter below threshold before soft exit.",
    )
    parser.add_argument(
        "--urgent-exit-prob",
        type=float,
        default=0.85,
        help="Immediate exit if p_exit_side exceeds this value.",
    )
    parser.add_argument(
        "--urgent-exit-delta",
        type=float,
        default=0.30,
        help="Immediate exit if p_exit_side - p_enter_side exceeds this value.",
    )
    parser.add_argument(
        "--opposite-dominance-delta",
        type=float,
        default=0.0,
        help="Required opposite-side margin advantage to invalidate a side intent.",
    )
    parser.add_argument(
        "--trace-out",
        default="Data/inference/spy/10min/meta/meta_trace_hybrid_exit_compare.csv",
        help="Shared 10m trace CSV with probabilities.",
    )
    parser.add_argument(
        "--actual-events-out",
        default="Data/inference/spy/10min/meta/meta_events_hybrid_exit_actual_1m_breakout.csv",
        help="Actual 1m breakout events CSV.",
    )
    parser.add_argument(
        "--baseline-events-out",
        default="Data/inference/spy/10min/meta/meta_events_hybrid_exit_next_10m_open.csv",
        help="Baseline next-10m-open events CSV.",
    )
    parser.add_argument(
        "--summary-out",
        default="Data/inference/spy/10min/meta/hybrid_exit_1m_vs_next_open_summary.csv",
        help="Summary metrics CSV.",
    )
    parser.add_argument(
        "--equity-out",
        default="Data/inference/spy/10min/plots/hybrid_exit_1m_vs_next_open_equity.png",
        help="Equity comparison PNG.",
    )
    parser.add_argument(
        "--out-dir",
        default="Data/inference/spy/10min/plots/hybrid_exit_1m_vs_next_open_sessions",
        help="Directory for session comparison PNGs.",
    )
    parser.add_argument("--sessions-per-fig", type=int, default=2, help="Sessions per output PNG.")
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


def _run_hybrid(
    *,
    meta_df: pd.DataFrame,
    one_min: pd.DataFrame,
    model_root: Path,
    symbol: str,
    entry_threshold: float | None,
    exit_threshold: float | None,
    min_hold_bars: int,
    soft_exit_confirm_bars: int,
    urgent_exit_prob: float,
    urgent_exit_delta: float,
    opposite_dominance_delta: float,
    entry_mode: str,
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
    long_soft_confirm = 0
    short_soft_confirm = 0
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
        p_exit_long = _score_exit(long_agent, work_row, side="long") if long_active else float("nan")
        p_exit_short = _score_exit(short_agent, work_row, side="short") if short_active else float("nan")

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

        long_hold_ready = bool(long_active and int(long_agent._state.bars_since_entry) >= int(min_hold_bars))
        short_hold_ready = bool(short_active and int(short_agent._state.bars_since_entry) >= int(min_hold_bars))

        long_soft_exit_condition = bool(
            long_active and long_hold_ready and np.isfinite(p_enter_long) and p_enter_long < float(thresholds["enter_long"])
        )
        short_soft_exit_condition = bool(
            short_active and short_hold_ready and np.isfinite(p_enter_short) and p_enter_short < float(thresholds["enter_short"])
        )
        long_soft_confirm = long_soft_confirm + 1 if long_soft_exit_condition else 0
        short_soft_confirm = short_soft_confirm + 1 if short_soft_exit_condition else 0

        long_urgent_exit = bool(
            long_active
            and (
                (np.isfinite(p_exit_long) and p_exit_long >= float(urgent_exit_prob))
                or (
                    np.isfinite(p_exit_long)
                    and np.isfinite(p_enter_long)
                    and (p_exit_long - p_enter_long) >= float(urgent_exit_delta)
                )
            )
        )
        short_urgent_exit = bool(
            short_active
            and (
                (np.isfinite(p_exit_short) and p_exit_short >= float(urgent_exit_prob))
                or (
                    np.isfinite(p_exit_short)
                    and np.isfinite(p_enter_short)
                    and (p_exit_short - p_enter_short) >= float(urgent_exit_delta)
                )
            )
        )

        do_exit_long = bool(long_urgent_exit or long_soft_confirm >= int(soft_exit_confirm_bars))
        do_exit_short = bool(short_urgent_exit or short_soft_confirm >= int(soft_exit_confirm_bars))

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

        entered_long_this_interval = False
        entered_short_this_interval = False
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
                long_soft_confirm = 0
                events.append({"timestamp": bar_ts, "symbol": symbol, "event": "exit_long", "price": bar_open})
            if pending_short_exit and short_active and np.isfinite(bar_open):
                short_active = False
                pending_short_exit = False
                short_intent_active = False
                short_ref_low = np.nan
                short_signal_row = None
                short_soft_confirm = 0
                events.append({"timestamp": bar_ts, "symbol": symbol, "event": "exit_short", "price": bar_open})

            if entry_mode == "breakout":
                if (
                    long_intent_active
                    and (not long_active)
                    and (not entered_long_this_interval)
                    and long_signal_row is not None
                    and np.isfinite(long_ref_high)
                    and np.isfinite(bar_high)
                    and bar_high >= float(long_ref_high)
                ):
                    fill_price = max(float(long_ref_high), float(bar_open)) if np.isfinite(bar_open) else float(long_ref_high)
                    long_active = True
                    long_intent_active = False
                    entered_long_this_interval = True
                    long_agent._set_trade_entry(position=1, row=long_signal_row, entry_price=fill_price)
                    events.append({"timestamp": bar_ts, "symbol": symbol, "event": "enter_long", "price": fill_price})
                if (
                    short_intent_active
                    and (not short_active)
                    and (not entered_short_this_interval)
                    and short_signal_row is not None
                    and np.isfinite(short_ref_low)
                    and np.isfinite(bar_low)
                    and bar_low <= float(short_ref_low)
                ):
                    fill_price = min(float(short_ref_low), float(bar_open)) if np.isfinite(bar_open) else float(short_ref_low)
                    short_active = True
                    short_intent_active = False
                    entered_short_this_interval = True
                    short_agent._set_trade_entry(position=-1, row=short_signal_row, entry_price=fill_price)
                    events.append({"timestamp": bar_ts, "symbol": symbol, "event": "enter_short", "price": fill_price})
            elif entry_mode == "next_open":
                if (
                    long_intent_active
                    and (not long_active)
                    and (not entered_long_this_interval)
                    and long_signal_row is not None
                    and np.isfinite(bar_open)
                ):
                    fill_price = float(bar_open)
                    long_active = True
                    long_intent_active = False
                    entered_long_this_interval = True
                    long_agent._set_trade_entry(position=1, row=long_signal_row, entry_price=fill_price)
                    events.append({"timestamp": bar_ts, "symbol": symbol, "event": "enter_long", "price": fill_price})
                if (
                    short_intent_active
                    and (not short_active)
                    and (not entered_short_this_interval)
                    and short_signal_row is not None
                    and np.isfinite(bar_open)
                ):
                    fill_price = float(bar_open)
                    short_active = True
                    short_intent_active = False
                    entered_short_this_interval = True
                    short_agent._set_trade_entry(position=-1, row=short_signal_row, entry_price=fill_price)
                    events.append({"timestamp": bar_ts, "symbol": symbol, "event": "enter_short", "price": fill_price})
            else:
                raise ValueError(f"Unknown entry mode: {entry_mode}")

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
                "long_ref_high": float(long_ref_high) if np.isfinite(long_ref_high) else np.nan,
                "short_ref_low": float(short_ref_low) if np.isfinite(short_ref_low) else np.nan,
                "long_soft_exit_condition": bool(long_soft_exit_condition),
                "short_soft_exit_condition": bool(short_soft_exit_condition),
                "long_soft_confirm_count": int(long_soft_confirm),
                "short_soft_confirm_count": int(short_soft_confirm),
                "long_urgent_exit": bool(long_urgent_exit),
                "short_urgent_exit": bool(short_urgent_exit),
            }
        )

    trace_df = pd.DataFrame(trace_rows)
    events_df = pd.DataFrame(events)
    return trace_df, events_df


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
                "breakout_or_baseline": long_eq + short_eq - 1.0,
                "net_1x_style": net_1x,
            }
        )
    return pd.DataFrame(records)


def _save_equity_plot(*, breakout_eq: pd.DataFrame, baseline_eq: pd.DataFrame, save_path: Path, symbol: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    x_break = pd.to_datetime(breakout_eq["timestamp"], utc=True).dt.tz_convert("America/New_York")
    x_base = pd.to_datetime(baseline_eq["timestamp"], utc=True).dt.tz_convert("America/New_York")
    ax.plot(x_break, breakout_eq["buy_hold"], color="#444444", linewidth=1.5, label="buy_hold")
    ax.plot(x_break, breakout_eq["net_1x_style"], color="#1565C0", linewidth=2.0, label="1m_breakout_net_1x")
    ax.plot(x_base, baseline_eq["net_1x_style"], color="#2E7D32", linewidth=2.0, label="next_10m_open_net_1x")
    ax.plot(x_break, breakout_eq["breakout_or_baseline"], color="#8E24AA", linewidth=1.4, label="1m_breakout_full_gross")
    ax.plot(x_base, baseline_eq["breakout_or_baseline"], color="#C62828", linewidth=1.4, label="next_10m_open_full_gross")
    ax.set_title(f"{symbol} | hybrid exit | 1m breakout vs next 10m open")
    ax.set_ylabel("Equity")
    ax.set_xlabel("Session Time (America/New_York)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _plot_sessions(
    *,
    trace: pd.DataFrame,
    one_min: pd.DataFrame,
    actual_events: pd.DataFrame,
    baseline_events: pd.DataFrame,
    out_dir: Path,
    sessions_per_fig: int,
    tz: str,
    symbol: str,
    urgent_exit_prob: float,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    one = one_min.copy()
    one["ts_local"] = one["timestamp"].dt.tz_convert(tz)
    one["session_date"] = one["ts_local"].dt.normalize()
    trace = trace.copy()
    trace["ts_local"] = pd.to_datetime(trace["timestamp"], utc=True, errors="coerce").dt.tz_convert(tz)
    trace["session_date"] = trace["ts_local"].dt.normalize()
    actual = actual_events.copy()
    actual["ts_local"] = pd.to_datetime(actual["timestamp"], utc=True, errors="coerce").dt.tz_convert(tz)
    actual["session_date"] = actual["ts_local"].dt.normalize()
    baseline = baseline_events.copy()
    baseline["ts_local"] = pd.to_datetime(baseline["timestamp"], utc=True, errors="coerce").dt.tz_convert(tz)
    baseline["session_date"] = baseline["ts_local"].dt.normalize()

    session_dates = sorted(one["session_date"].dropna().unique().tolist())
    outputs: list[Path] = []
    chunks = int(ceil(len(session_dates) / max(1, sessions_per_fig)))
    candle_width = 0.65 / (24 * 60)

    filled_specs = {
        "enter_long": ("#2E7D32", "^", "actual enter long", True),
        "exit_long": ("#8c564b", "v", "actual exit long", True),
        "enter_short": ("#C62828", "v", "actual enter short", True),
        "exit_short": ("#9467bd", "^", "actual exit short", True),
    }
    hollow_specs = {
        "enter_long": ("#2E7D32", "^", "next open enter long", False),
        "exit_long": ("#8c564b", "v", "next open exit long", False),
        "enter_short": ("#C62828", "v", "next open enter short", False),
        "exit_short": ("#9467bd", "^", "next open exit short", False),
    }

    for chunk_idx in range(chunks):
        dates = session_dates[chunk_idx * sessions_per_fig:(chunk_idx + 1) * sessions_per_fig]
        n = len(dates)
        fig, axes = plt.subplots(
            n * 2,
            1,
            figsize=(18, max(7.0, 5.5 * n)),
            sharex=False,
            gridspec_kw={"height_ratios": [2.5, 1.0] * n},
        )
        if n == 1:
            axes = [axes[0], axes[1]]

        for idx, session_date in enumerate(dates):
            ax_price = axes[idx * 2]
            ax_prob = axes[idx * 2 + 1]

            one_s = one[one["session_date"] == session_date].copy().sort_values("ts_local")
            tr_s = trace[trace["session_date"] == session_date].copy().sort_values("ts_local")
            act_s = actual[actual["session_date"] == session_date].copy().sort_values("ts_local")
            base_s = baseline[baseline["session_date"] == session_date].copy().sort_values("ts_local")
            if one_s.empty or tr_s.empty:
                continue

            x = mdates.date2num(one_s["ts_local"].to_list())
            open_ = pd.to_numeric(one_s["open"], errors="coerce").to_numpy()
            high = pd.to_numeric(one_s["high"], errors="coerce").to_numpy()
            low = pd.to_numeric(one_s["low"], errors="coerce").to_numpy()
            close = pd.to_numeric(one_s["close"], errors="coerce").to_numpy()
            up = close >= open_
            down = ~up

            ax_price.vlines(x, low, high, color="#4a4a4a", linewidth=0.5, zorder=1)
            ax_price.bar(x[up], close[up] - open_[up], width=candle_width, bottom=open_[up], color="#1976D2", edgecolor="none", zorder=1.2, label="1m bull")
            ax_price.bar(x[down], close[down] - open_[down], width=candle_width, bottom=open_[down], color="#E53935", edgecolor="none", zorder=1.2, label="1m bear")

            if not act_s.empty:
                ex = mdates.date2num(act_s["ts_local"].to_list())
                ey = pd.to_numeric(act_s["price"], errors="coerce").to_numpy()
                for event_name, (color, marker, label, _) in filled_specs.items():
                    mask = act_s["event"].astype(str).eq(event_name).to_numpy()
                    if mask.any():
                        ax_price.scatter(ex[mask], ey[mask], color=color, marker=marker, s=42, label=label, zorder=2.5)

            if not base_s.empty:
                ex = mdates.date2num(base_s["ts_local"].to_list())
                ey = pd.to_numeric(base_s["price"], errors="coerce").to_numpy()
                for event_name, (color, marker, label, _) in hollow_specs.items():
                    mask = base_s["event"].astype(str).eq(event_name).to_numpy()
                    if mask.any():
                        ax_price.scatter(ex[mask], ey[mask], facecolors="none", edgecolors=color, marker=marker, s=52, linewidths=1.3, label=label, zorder=2.4)

            ax_price.set_title(f"{symbol} | {session_date.date()} | hybrid exit | 1m breakout vs next 10m open")
            ax_price.set_ylabel("Price")
            ax_price.grid(True, alpha=0.25)
            handles, labels = ax_price.get_legend_handles_labels()
            dedup = dict(zip(labels, handles))
            ax_price.legend(dedup.values(), dedup.keys(), loc="upper left", fontsize=8, ncol=4)

            mx = mdates.date2num(tr_s["ts_local"].to_list())
            for col, color, label in (
                ("p_enter_long", "#2ca02c", "p_enter_long"),
                ("p_enter_short", "#d62728", "p_enter_short"),
                ("p_exit_long", "#17becf", "p_exit_long"),
                ("p_exit_short", "#ff7f0e", "p_exit_short"),
            ):
                y = pd.to_numeric(tr_s[col], errors="coerce")
                if y.notna().any():
                    ax_prob.step(mx, y.to_numpy(), where="post", color=color, linewidth=1.2, label=label)
            for col, color, label in (
                ("thr_enter_long", "#2ca02c", "thr_enter_long"),
                ("thr_enter_short", "#d62728", "thr_enter_short"),
            ):
                y = pd.to_numeric(tr_s[col], errors="coerce").dropna()
                if not y.empty:
                    ax_prob.axhline(float(y.iloc[-1]), color=color, linestyle="--", linewidth=1.0, alpha=0.85, label=label)
            ax_prob.axhline(float(urgent_exit_prob), color="#17becf", linestyle=":", linewidth=1.0, alpha=0.8, label="urgent_exit_prob")

            ax_prob.set_ylim(-0.02, 1.02)
            ax_prob.set_ylabel("Probability")
            ax_prob.set_xlabel(f"Session Time ({tz})")
            ax_prob.grid(True, alpha=0.25)
            handles, labels = ax_prob.get_legend_handles_labels()
            dedup = dict(zip(labels, handles))
            ax_prob.legend(dedup.values(), dedup.keys(), loc="upper right", fontsize=8, ncol=4)

            session_tz = one_s["ts_local"].dt.tz
            locator = mdates.HourLocator(interval=1, tz=session_tz)
            formatter = mdates.DateFormatter("%H:%M", tz=session_tz)
            ax_price.xaxis.set_major_locator(locator)
            ax_price.xaxis.set_major_formatter(formatter)
            ax_prob.xaxis.set_major_locator(locator)
            ax_prob.xaxis.set_major_formatter(formatter)
            ax_price.tick_params(axis="x", labelbottom=False)
            ax_price.set_xlim(x.min() - candle_width * 6, x.max() + candle_width * 6)
            ax_prob.set_xlim(x.min() - candle_width * 6, x.max() + candle_width * 6)

        fig.tight_layout()
        out_path = out_dir / f"{symbol.lower()}_hybrid_1m_vs_next_open_part{chunk_idx + 1}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def main() -> None:
    args = _parse_args()
    start, end = _normalize_bounds(args.start, args.end)
    meta_df = _load_meta_matrix(Path(args.meta_matrix), start=start, end=end, tz=args.tz)
    one_min = _load_one_min(Path(args.one_min_data), symbol=args.symbol, start=start, end=end)

    trace_df, actual_events = _run_hybrid(
        meta_df=meta_df,
        one_min=one_min,
        model_root=Path(args.model_root),
        symbol=args.symbol,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        min_hold_bars=max(0, int(args.min_hold_bars)),
        soft_exit_confirm_bars=max(1, int(args.soft_exit_confirm_bars)),
        urgent_exit_prob=float(args.urgent_exit_prob),
        urgent_exit_delta=float(args.urgent_exit_delta),
        opposite_dominance_delta=float(args.opposite_dominance_delta),
        entry_mode="breakout",
    )
    _, baseline_events = _run_hybrid(
        meta_df=meta_df,
        one_min=one_min,
        model_root=Path(args.model_root),
        symbol=args.symbol,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        min_hold_bars=max(0, int(args.min_hold_bars)),
        soft_exit_confirm_bars=max(1, int(args.soft_exit_confirm_bars)),
        urgent_exit_prob=float(args.urgent_exit_prob),
        urgent_exit_delta=float(args.urgent_exit_delta),
        opposite_dominance_delta=float(args.opposite_dominance_delta),
        entry_mode="next_open",
    )

    trace_path = Path(args.trace_out)
    actual_events_path = Path(args.actual_events_out)
    baseline_events_path = Path(args.baseline_events_out)
    for path in (trace_path, actual_events_path, baseline_events_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    trace_df.to_csv(trace_path, index=False)
    actual_events.to_csv(actual_events_path, index=False)
    baseline_events.to_csv(baseline_events_path, index=False)

    breakout_eq = _equity_curve_from_events(actual_events, one_min)
    baseline_eq = _equity_curve_from_events(baseline_events, one_min)
    _save_equity_plot(breakout_eq=breakout_eq, baseline_eq=baseline_eq, save_path=Path(args.equity_out), symbol=args.symbol)

    summary = pd.DataFrame(
        [
            {"regime": "1m_breakout_entry_hybrid_exit", **_event_metrics(actual_events)},
            {"regime": "next_10m_open_entry_hybrid_exit", **_event_metrics(baseline_events)},
        ]
    )
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    plots = _plot_sessions(
        trace=trace_df,
        one_min=one_min,
        actual_events=actual_events,
        baseline_events=baseline_events,
        out_dir=Path(args.out_dir),
        sessions_per_fig=max(1, int(args.sessions_per_fig)),
        tz=args.tz,
        symbol=args.symbol,
        urgent_exit_prob=float(args.urgent_exit_prob),
    )

    print(summary.to_string(index=False))
    print(f"\ntrace_csv={trace_path}")
    print(f"actual_events_csv={actual_events_path}")
    print(f"baseline_events_csv={baseline_events_path}")
    print(f"summary_csv={summary_path}")
    print(f"equity_png={Path(args.equity_out)}")
    print("plots:")
    for path in plots:
        print(path)


if __name__ == "__main__":
    main()
