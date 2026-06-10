from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.API.Alpaca_API.inference.live_inference import LiveMetaXGBAgent  # noqa: E402
from scripts.compare_baseline_vs_profit_protect_exit import (  # noqa: E402
    _equity_curve_from_events,
    _event_metrics,
    _load_one_min,
    _normalize_bounds,
    _save_equity_plot,
)
from scripts.compare_baseline_vs_profit_protect_exit import _plot_sessions as _plot_profit_sessions  # noqa: E402
from scripts.compare_baseline_vs_profit_protect_exit import _run_regime as _run_baseline_regime  # noqa: E402
from scripts.replay_meta_independent import _load_meta_matrix, _score_exit  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline next-10m-open execution vs directional 1m confirmation execution."
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
    parser.add_argument("--model-root", default="Data/models/meta_xgboost/10min", help="Meta model root.")
    parser.add_argument("--symbol", default="SPY", help="Symbol label.")
    parser.add_argument("--start", default="2026-02-13T00:00:00Z", help="UTC start timestamp.")
    parser.add_argument("--end", default="2026-03-23T23:59:59Z", help="UTC end timestamp.")
    parser.add_argument("--tz", default="America/New_York", help="Display timezone.")
    parser.add_argument("--entry-threshold", type=float, default=None, help="Optional override for both entry thresholds.")
    parser.add_argument("--exit-threshold", type=float, default=None, help="Optional override for both exit thresholds.")
    parser.add_argument("--min-hold-bars", type=int, default=2, help="Minimum 10m bars before soft exits.")
    parser.add_argument("--soft-exit-confirm-bars", type=int, default=2, help="Consecutive bars for soft exit confirmation.")
    parser.add_argument("--urgent-exit-prob", type=float, default=0.85, help="Immediate exit if p_exit_side exceeds this value.")
    parser.add_argument("--urgent-exit-delta", type=float, default=0.30, help="Immediate exit if p_exit_side - p_enter_side exceeds this value.")
    parser.add_argument("--opposite-dominance-delta", type=float, default=0.0, help="Opposite-side margin needed to invalidate a side intent.")
    parser.add_argument(
        "--trace-out",
        default="Data/inference/spy/10min/meta/meta_trace_baseline_vs_1m_directional_execution.csv",
        help="Shared 10m trace CSV with probabilities.",
    )
    parser.add_argument(
        "--baseline-events-out",
        default="Data/inference/spy/10min/meta/meta_events_baseline_exit_current.csv",
        help="Baseline event CSV.",
    )
    parser.add_argument(
        "--variant-events-out",
        default="Data/inference/spy/10min/meta/meta_events_1m_directional_execution.csv",
        help="Variant event CSV.",
    )
    parser.add_argument(
        "--summary-out",
        default="Data/inference/spy/10min/meta/baseline_vs_1m_directional_execution_summary.csv",
        help="Summary metrics CSV.",
    )
    parser.add_argument(
        "--equity-out",
        default="Data/inference/spy/10min/plots/baseline_vs_1m_directional_execution_equity.png",
        help="Equity comparison PNG.",
    )
    parser.add_argument(
        "--out-dir",
        default="Data/inference/spy/10min/plots/baseline_vs_1m_directional_execution_sessions",
        help="Directory for session comparison PNGs.",
    )
    parser.add_argument("--sessions-per-fig", type=int, default=2, help="Sessions per PNG.")
    return parser.parse_args()


def _validity_flags(
    *,
    p_enter_long: float,
    p_enter_short: float,
    thr_enter_long: float,
    thr_enter_short: float,
    opposite_dominance_delta: float,
) -> tuple[bool, bool]:
    long_ready = pd.notna(p_enter_long) and p_enter_long >= thr_enter_long
    short_ready = pd.notna(p_enter_short) and p_enter_short >= thr_enter_short
    long_margin = (p_enter_long - thr_enter_long) if long_ready else float("-inf")
    short_margin = (p_enter_short - thr_enter_short) if short_ready else float("-inf")
    long_invalidated = bool(short_ready and short_margin > long_margin + opposite_dominance_delta)
    short_invalidated = bool(long_ready and long_margin > short_margin + opposite_dominance_delta)
    return bool(long_ready and not long_invalidated), bool(short_ready and not short_invalidated)


def _run_directional_1m_regime(
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
        "enter_long": float("nan"),
        "enter_short": float("nan"),
        "exit_long": float("nan"),
        "exit_short": float("nan"),
    }

    meta_index = meta_df.index.to_list()
    one_min_pos = 0

    long_active = False
    short_active = False
    long_entry_pending = False
    short_entry_pending = False
    long_exit_pending = False
    short_exit_pending = False
    long_signal_row: pd.Series | None = None
    short_signal_row: pd.Series | None = None
    long_soft_confirm = 0
    short_soft_confirm = 0

    trace_rows: list[dict[str, object]] = []
    events: list[dict[str, object]] = []

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

        if not long_active and long_valid_signal and not long_entry_pending:
            long_entry_pending = True
            long_signal_row = row.copy()
        if not short_active and short_valid_signal and not short_entry_pending:
            short_entry_pending = True
            short_signal_row = row.copy()

        long_hold_ready = bool(long_active and int(long_agent._state.bars_since_entry) >= int(min_hold_bars))
        short_hold_ready = bool(short_active and int(short_agent._state.bars_since_entry) >= int(min_hold_bars))

        long_soft_exit_condition = bool(
            long_active and long_hold_ready and pd.notna(p_enter_long) and p_enter_long < float(thresholds["enter_long"])
        )
        short_soft_exit_condition = bool(
            short_active and short_hold_ready and pd.notna(p_enter_short) and p_enter_short < float(thresholds["enter_short"])
        )
        long_soft_confirm = long_soft_confirm + 1 if long_soft_exit_condition else 0
        short_soft_confirm = short_soft_confirm + 1 if short_soft_exit_condition else 0

        long_urgent_exit = bool(
            long_active
            and (
                (pd.notna(p_exit_long) and p_exit_long >= float(urgent_exit_prob))
                or (
                    pd.notna(p_exit_long)
                    and pd.notna(p_enter_long)
                    and (p_exit_long - p_enter_long) >= float(urgent_exit_delta)
                )
            )
        )
        short_urgent_exit = bool(
            short_active
            and (
                (pd.notna(p_exit_short) and p_exit_short >= float(urgent_exit_prob))
                or (
                    pd.notna(p_exit_short)
                    and pd.notna(p_enter_short)
                    and (p_exit_short - p_enter_short) >= float(urgent_exit_delta)
                )
            )
        )

        do_exit_long = bool(long_urgent_exit or long_soft_confirm >= int(soft_exit_confirm_bars))
        do_exit_short = bool(short_urgent_exit or short_soft_confirm >= int(soft_exit_confirm_bars))

        if long_active:
            long_agent._advance_state(action=0 if do_exit_long else 1, row=work_row)
            if do_exit_long:
                long_exit_pending = True
        if short_active:
            short_agent._advance_state(action=0 if do_exit_short else -1, row=work_row)
            if do_exit_short:
                short_exit_pending = True

        interval = one_min.iloc[one_min_pos:]
        if not interval.empty:
            interval = interval[(interval["timestamp"] >= decision_ts) & (interval["timestamp"] < next_decision_ts)]

        for _, bar in interval.iterrows():
            bar_ts = pd.Timestamp(bar["timestamp"])
            bar_open = float(bar.get("open", float("nan")))
            bar_close = float(bar.get("close", float("nan")))

            bullish = pd.notna(bar_open) and pd.notna(bar_close) and bar_close > bar_open
            bearish = pd.notna(bar_open) and pd.notna(bar_close) and bar_close < bar_open

            if long_exit_pending and long_active and bearish and pd.notna(bar_close):
                long_active = False
                long_exit_pending = False
                long_soft_confirm = 0
                long_agent._reset_trade_state()
                events.append({"timestamp": bar_ts, "symbol": symbol, "event": "exit_long", "price": bar_close, "reason": "1m_directional"})
            if short_exit_pending and short_active and bullish and pd.notna(bar_close):
                short_active = False
                short_exit_pending = False
                short_soft_confirm = 0
                short_agent._reset_trade_state()
                events.append({"timestamp": bar_ts, "symbol": symbol, "event": "exit_short", "price": bar_close, "reason": "1m_directional"})
            if long_entry_pending and (not long_active) and bullish and pd.notna(bar_close) and long_signal_row is not None:
                long_active = True
                long_entry_pending = False
                long_agent._set_trade_entry(position=1, row=long_signal_row, entry_price=float(bar_close))
                events.append({"timestamp": bar_ts, "symbol": symbol, "event": "enter_long", "price": bar_close, "reason": "1m_directional"})
            if short_entry_pending and (not short_active) and bearish and pd.notna(bar_close) and short_signal_row is not None:
                short_active = True
                short_entry_pending = False
                short_agent._set_trade_entry(position=-1, row=short_signal_row, entry_price=float(bar_close))
                events.append({"timestamp": bar_ts, "symbol": symbol, "event": "enter_short", "price": bar_close, "reason": "1m_directional"})

            one_min_pos += 1

        trace_rows.append(
            {
                "symbol": symbol,
                "timestamp": ts,
                "open": float(row.get("open", float("nan"))),
                "high": float(row.get("high", float("nan"))),
                "low": float(row.get("low", float("nan"))),
                "close": float(row.get("close", float("nan"))),
                "volume": float(row.get("volume", float("nan"))),
                "atr": float(row.get("atr", float("nan"))),
                "p_enter_long": p_enter_long,
                "p_enter_short": p_enter_short,
                "p_exit_long": p_exit_long,
                "p_exit_short": p_exit_short,
                "thr_enter_long": float(thresholds["enter_long"]),
                "thr_enter_short": float(thresholds["enter_short"]),
                "thr_exit_long": float(thresholds["exit_long"]),
                "thr_exit_short": float(thresholds["exit_short"]),
                "long_soft_confirm_count": int(long_soft_confirm),
                "short_soft_confirm_count": int(short_soft_confirm),
                "long_urgent_exit": bool(long_urgent_exit),
                "short_urgent_exit": bool(short_urgent_exit),
            }
        )

    return pd.DataFrame(trace_rows), pd.DataFrame(events)


def _save_variant_equity_plot(
    *,
    variant_eq: pd.DataFrame,
    baseline_eq: pd.DataFrame,
    save_path: Path,
    symbol: str,
) -> None:
    _save_equity_plot(
        variant_eq=variant_eq,
        baseline_eq=baseline_eq,
        save_path=save_path,
        symbol=symbol,
        profit_protect_arm_atr=0.0,
        profit_protect_giveback_long=0.0,
        profit_protect_giveback_short=0.0,
    )


def main() -> None:
    args = _parse_args()
    start, end = _normalize_bounds(args.start, args.end)
    meta_df = _load_meta_matrix(Path(args.meta_matrix), start=start, end=end, tz=args.tz)
    one_min = _load_one_min(Path(args.one_min_data), symbol=args.symbol, start=start, end=end)

    trace_df, baseline_events = _run_baseline_regime(
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
        profit_protect_arm_atr=None,
        profit_protect_giveback_long=None,
        profit_protect_giveback_short=None,
    )
    _, variant_events = _run_directional_1m_regime(
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
    )

    trace_path = Path(args.trace_out)
    baseline_path = Path(args.baseline_events_out)
    variant_path = Path(args.variant_events_out)
    for path in (trace_path, baseline_path, variant_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    trace_df.to_csv(trace_path, index=False)
    baseline_events.to_csv(baseline_path, index=False)
    variant_events.to_csv(variant_path, index=False)

    baseline_eq = _equity_curve_from_events(baseline_events, one_min)
    variant_eq = _equity_curve_from_events(variant_events, one_min)
    _save_variant_equity_plot(
        variant_eq=variant_eq,
        baseline_eq=baseline_eq,
        save_path=Path(args.equity_out),
        symbol=args.symbol,
    )

    summary = pd.DataFrame(
        [
            {"regime": "baseline_next_open_exit_baseline", **_event_metrics(baseline_events)},
            {"regime": "directional_1m_entry_exit_baseline", **_event_metrics(variant_events)},
        ]
    )
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    plots = _plot_profit_sessions(
        trace=trace_df,
        one_min=one_min,
        variant_events=variant_events,
        baseline_events=baseline_events,
        out_dir=Path(args.out_dir),
        sessions_per_fig=max(1, int(args.sessions_per_fig)),
        tz=args.tz,
        symbol=args.symbol,
        profit_protect_arm_atr=0.0,
        profit_protect_giveback_long=0.0,
        profit_protect_giveback_short=0.0,
    )

    print(summary.to_string(index=False))
    print(f"\ntrace_csv={trace_path}")
    print(f"baseline_events_csv={baseline_path}")
    print(f"variant_events_csv={variant_path}")
    print(f"summary_csv={summary_path}")
    print(f"equity_png={Path(args.equity_out)}")
    print("plots:")
    for path in plots:
        print(path)


if __name__ == "__main__":
    main()
