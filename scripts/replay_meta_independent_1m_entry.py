from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from API.Alpaca_API.inference.live_inference import LiveMetaXGBAgent
from scripts.replay_meta_independent import _load_meta_matrix, _normalize_bounds, _score_exit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay independent 10m meta state with 1m breakout entry confirmation and 10m exits."
    )
    parser.add_argument(
        "--meta-matrix",
        default="Data/inference/spy/10min/debug_matrices_warmup/spy/live_meta_matrix_on_trace_ts.parquet",
        help="Cached meta matrix parquet.",
    )
    parser.add_argument(
        "--one-min-data",
        default="Data/raw/spy/spy_intraday_1min.parquet",
        help="Raw 1m parquet for execution timing.",
    )
    parser.add_argument(
        "--model-root",
        default="Data/models/meta_xgboost/10min",
        help="Meta model root directory.",
    )
    parser.add_argument("--symbol", default="SPY", help="Symbol label.")
    parser.add_argument("--start", default="2026-02-13T00:00:00Z", help="UTC start timestamp.")
    parser.add_argument("--end", default="2026-03-13T23:59:59Z", help="UTC end timestamp.")
    parser.add_argument("--tz", default="America/New_York", help="Display timezone.")
    parser.add_argument("--entry-threshold", type=float, default=None, help="Optional override for both entry thresholds.")
    parser.add_argument("--exit-threshold", type=float, default=None, help="Optional override for both exit thresholds.")
    parser.add_argument("--min-hold-bars", type=int, default=2, help="Minimum 10m bars to hold before honoring exit.")
    parser.add_argument("--exit-entry-delta", type=float, default=0.15, help="Exit dominance margin over same-side entry.")
    parser.add_argument(
        "--opposite-dominance-delta",
        type=float,
        default=0.0,
        help="Required opposite-side margin advantage to invalidate a side intent.",
    )
    parser.add_argument(
        "--trace-out",
        default="Data/inference/spy/10min/meta/meta_trace_independent_1m_entry_last_month.csv",
        help="Output 10m trace CSV.",
    )
    parser.add_argument(
        "--events-out",
        default="Data/inference/spy/10min/meta/meta_events_independent_1m_entry_last_month.csv",
        help="Output execution events CSV.",
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


def main() -> None:
    args = _parse_args()
    start, end = _normalize_bounds(args.start, args.end)
    meta_df = _load_meta_matrix(Path(args.meta_matrix), start=start, end=end, tz=args.tz)
    one_min = _load_one_min(Path(args.one_min_data), symbol=args.symbol, start=start, end=end)

    long_agent = LiveMetaXGBAgent(
        model_root=Path(args.model_root),
        precomputed_base_frame=meta_df,
        entry_threshold_override=args.entry_threshold,
        exit_threshold_override=args.exit_threshold,
    )
    short_agent = LiveMetaXGBAgent(
        model_root=Path(args.model_root),
        precomputed_base_frame=meta_df,
        entry_threshold_override=args.entry_threshold,
        exit_threshold_override=args.exit_threshold,
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
            opposite_dominance_delta=float(args.opposite_dominance_delta),
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

        long_exit_threshold_hit = bool(long_active and np.isfinite(p_exit_long) and p_exit_long >= float(thresholds["exit_long"]))
        short_exit_threshold_hit = bool(short_active and np.isfinite(p_exit_short) and p_exit_short >= float(thresholds["exit_short"]))
        long_hold_ready = bool(long_active and int(long_agent._state.bars_since_entry) >= int(args.min_hold_bars))
        short_hold_ready = bool(short_active and int(short_agent._state.bars_since_entry) >= int(args.min_hold_bars))
        long_entry_still_supports = bool(
            np.isfinite(p_enter_long)
            and p_enter_long >= float(thresholds["enter_long"])
            and (not np.isfinite(p_exit_long) or (p_exit_long - p_enter_long) < float(args.exit_entry_delta))
        )
        short_entry_still_supports = bool(
            np.isfinite(p_enter_short)
            and p_enter_short >= float(thresholds["enter_short"])
            and (not np.isfinite(p_exit_short) or (p_exit_short - p_enter_short) < float(args.exit_entry_delta))
        )
        do_exit_long = bool(long_exit_threshold_hit and long_hold_ready and not long_entry_still_supports)
        do_exit_short = bool(short_exit_threshold_hit and short_hold_ready and not short_entry_still_supports)

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
                events.append({"timestamp": bar_ts, "symbol": args.symbol, "event": "exit_long", "price": bar_open})
            if pending_short_exit and short_active and np.isfinite(bar_open):
                short_active = False
                pending_short_exit = False
                short_intent_active = False
                short_ref_low = np.nan
                short_signal_row = None
                events.append({"timestamp": bar_ts, "symbol": args.symbol, "event": "exit_short", "price": bar_open})

            if long_intent_active and (not long_active) and long_signal_row is not None and np.isfinite(long_ref_high):
                if np.isfinite(bar_high) and bar_high >= float(long_ref_high):
                    fill_price = max(float(long_ref_high), float(bar_open)) if np.isfinite(bar_open) else float(long_ref_high)
                    long_active = True
                    long_intent_active = False
                    long_agent._set_trade_entry(position=1, row=long_signal_row, entry_price=fill_price)
                    events.append({"timestamp": bar_ts, "symbol": args.symbol, "event": "enter_long", "price": fill_price})

            if short_intent_active and (not short_active) and short_signal_row is not None and np.isfinite(short_ref_low):
                if np.isfinite(bar_low) and bar_low <= float(short_ref_low):
                    fill_price = min(float(short_ref_low), float(bar_open)) if np.isfinite(bar_open) else float(short_ref_low)
                    short_active = True
                    short_intent_active = False
                    short_agent._set_trade_entry(position=-1, row=short_signal_row, entry_price=fill_price)
                    events.append({"timestamp": bar_ts, "symbol": args.symbol, "event": "enter_short", "price": fill_price})

            one_min_pos += 1

        trace_rows.append(
            {
                "symbol": args.symbol,
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
            }
        )

    trace_df = pd.DataFrame(trace_rows)
    events_df = pd.DataFrame(events)
    trace_out = Path(args.trace_out)
    events_out = Path(args.events_out)
    trace_out.parent.mkdir(parents=True, exist_ok=True)
    events_out.parent.mkdir(parents=True, exist_ok=True)
    trace_df.to_csv(trace_out, index=False)
    events_df.to_csv(events_out, index=False)
    print(trace_out)
    print(events_out)


if __name__ == "__main__":
    main()
