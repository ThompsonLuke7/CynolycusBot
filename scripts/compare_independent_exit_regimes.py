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

from core.API.Alpaca_API.inference.live_inference import LiveMetaXGBAgent
from scripts.replay_meta_independent import _load_meta_matrix, _normalize_bounds, _save_plot as _save_trace_plot, _score_exit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current independent exit logic against entry-falls-below-threshold exits."
    )
    parser.add_argument(
        "--meta-matrix",
        default="Data/inference/spy/10min/debug_matrices_warmup/spy/live_meta_matrix_on_trace_ts_live_2026_03_24.parquet",
        help="Cached meta matrix parquet.",
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
    parser.add_argument("--min-hold-bars", type=int, default=2, help="Minimum bars to hold before honoring exits.")
    parser.add_argument(
        "--exit-entry-delta",
        type=float,
        default=0.15,
        help="Current-rule exit-vs-entry dominance margin.",
    )
    parser.add_argument(
        "--confirm-bars",
        type=int,
        default=2,
        help="Consecutive bars required for the entry-below-threshold exit confirmation rule.",
    )
    parser.add_argument(
        "--summary-out",
        default="Data/inference/spy/10min/meta/exit_regime_comparison_summary.csv",
        help="CSV summary output path.",
    )
    parser.add_argument(
        "--equity-out",
        default="Data/inference/spy/10min/plots/exit_regime_comparison_equity.png",
        help="Equity comparison PNG path.",
    )
    parser.add_argument(
        "--alt-trace-out",
        default="Data/inference/spy/10min/meta/meta_trace_independent_entry_falls_below_exit.csv",
        help="Alternative exit-regime trace CSV path.",
    )
    parser.add_argument(
        "--alt-plot-out",
        default="Data/inference/spy/10min/plots/meta_entries_exits_probs_entry_falls_below_exit.png",
        help="Alternative exit-regime bars/probabilities PNG path.",
    )
    return parser.parse_args()


def _trade_metrics(trace: pd.DataFrame) -> dict[str, float]:
    total_long_entries = int(trace["entry_long"].sum())
    total_short_entries = int(trace["entry_short"].sum())
    total_long_exits = int(trace["exit_long"].sum())
    total_short_exits = int(trace["exit_short"].sum())
    return {
        "long_entries": float(total_long_entries),
        "long_exits": float(total_long_exits),
        "short_entries": float(total_short_entries),
        "short_exits": float(total_short_exits),
    }


def _equity_from_trace(trace: pd.DataFrame) -> pd.DataFrame:
    df = trace.sort_values("timestamp").reset_index(drop=True).copy()
    records: list[dict[str, float | pd.Timestamp]] = []
    long_on = False
    short_on = False
    long_equity = 1.0
    short_equity = 1.0
    net_1x = 1.0
    buy_hold = 1.0

    for i in range(len(df) - 1):
        open_i = float(df.loc[i, "open"])
        open_n = float(df.loc[i + 1, "open"])
        if not (np.isfinite(open_i) and np.isfinite(open_n) and open_i > 0.0 and open_n > 0.0):
            continue
        ret = open_n / open_i - 1.0
        buy_hold *= 1.0 + ret
        if long_on:
            long_equity *= 1.0 + ret
        if short_on:
            short_equity *= 1.0 - ret
        if long_on and not short_on:
            net_1x *= 1.0 + ret
        elif short_on and not long_on:
            net_1x *= 1.0 - ret
        records.append(
            {
                "timestamp": pd.Timestamp(df.loc[i + 1, "timestamp"]),
                "buy_hold": buy_hold,
                "long_only": long_equity,
                "short_only": short_equity,
                "combined_full_gross": long_equity + short_equity - 1.0,
                "net_1x_style": net_1x,
            }
        )
        if bool(df.loc[i, "exit_long"]) and long_on:
            long_on = False
        if bool(df.loc[i, "exit_short"]) and short_on:
            short_on = False
        if bool(df.loc[i, "entry_long"]) and not long_on:
            long_on = True
        if bool(df.loc[i, "entry_short"]) and not short_on:
            short_on = True
    return pd.DataFrame(records)


def _run_trace(
    *,
    meta_df: pd.DataFrame,
    model_root: Path,
    entry_threshold: float | None,
    exit_threshold: float | None,
    min_hold_bars: int,
    exit_entry_delta: float,
    confirm_bars: int,
    exit_mode: str,
) -> pd.DataFrame:
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

    rows: list[dict[str, object]] = []
    long_active = False
    short_active = False
    long_bars_held = -1
    short_bars_held = -1
    long_exit_confirm = 0
    short_exit_confirm = 0

    for idx, (_, row) in enumerate(meta_df.iterrows()):
        ts = pd.Timestamp(row.name)
        p_enter_long = float(entry_long_probs[idx]) if idx < entry_long_probs.size else float("nan")
        p_enter_short = float(entry_short_probs[idx]) if idx < entry_short_probs.size else float("nan")
        work_row = row.copy()
        work_row["p_enter_long_oof"] = p_enter_long
        work_row["p_enter_short_oof"] = p_enter_short
        p_exit_long = _score_exit(long_agent, work_row, side="long") if long_active else float("nan")
        p_exit_short = _score_exit(short_agent, work_row, side="short") if short_active else float("nan")

        long_hold_ready = bool(long_active and long_bars_held >= int(min_hold_bars))
        short_hold_ready = bool(short_active and short_bars_held >= int(min_hold_bars))

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
        else:
            raise ValueError(f"Unknown exit mode: {exit_mode}")

        long_exit_confirm = long_exit_confirm + 1 if long_exit_condition else 0
        short_exit_confirm = short_exit_confirm + 1 if short_exit_condition else 0
        do_exit_long = bool(long_exit_confirm >= int(confirm_bars))
        do_exit_short = bool(short_exit_confirm >= int(confirm_bars))

        do_entry_long = bool((not long_active) and np.isfinite(p_enter_long) and p_enter_long >= float(thresholds["enter_long"]))
        do_entry_short = bool((not short_active) and np.isfinite(p_enter_short) and p_enter_short >= float(thresholds["enter_short"]))

        next_long_active = bool((long_active and not do_exit_long) or do_entry_long)
        next_short_active = bool((short_active and not do_exit_short) or do_entry_short)

        long_action = 1 if next_long_active else 0
        short_action = -1 if next_short_active else 0
        long_agent._advance_state(action=long_action, row=work_row)
        short_agent._advance_state(action=short_action, row=work_row)
        long_active = next_long_active
        short_active = next_short_active

        if do_entry_long:
            long_bars_held = 0
        elif long_active:
            long_bars_held = max(0, long_bars_held + 1)
        else:
            long_bars_held = -1
            long_exit_confirm = 0
        if do_entry_short:
            short_bars_held = 0
        elif short_active:
            short_bars_held = max(0, short_bars_held + 1)
        else:
            short_bars_held = -1
            short_exit_confirm = 0

        rows.append(
            {
                "timestamp": ts,
                "open": float(row.get("open", np.nan)),
                "high": float(row.get("high", np.nan)),
                "low": float(row.get("low", np.nan)),
                "close": float(row.get("close", np.nan)),
                "p_enter_long": p_enter_long,
                "p_enter_short": p_enter_short,
                "p_exit_long": p_exit_long,
                "p_exit_short": p_exit_short,
                "thr_enter_long": float(thresholds["enter_long"]),
                "thr_enter_short": float(thresholds["enter_short"]),
                "thr_exit_long": float(thresholds["exit_long"]),
                "thr_exit_short": float(thresholds["exit_short"]),
                "entry_long": bool(do_entry_long),
                "entry_short": bool(do_entry_short),
                "exit_long": bool(do_exit_long),
                "exit_short": bool(do_exit_short),
                "ind_entry_long": bool(do_entry_long),
                "ind_entry_short": bool(do_entry_short),
                "ind_exit_long": bool(do_exit_long),
                "ind_exit_short": bool(do_exit_short),
                "long_active": int(long_active),
                "short_active": int(short_active),
                "ind_long_active": int(long_active),
                "ind_short_active": int(short_active),
                "long_bars_held": int(max(long_bars_held, 0)),
                "short_bars_held": int(max(short_bars_held, 0)),
                "long_exit_confirm_count": int(long_exit_confirm),
                "short_exit_confirm_count": int(short_exit_confirm),
            }
        )

    return pd.DataFrame(rows)


def _save_equity_plot(*, current_eq: pd.DataFrame, alt_eq: pd.DataFrame, save_path: Path, symbol: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    for eq, label, color, width in (
        (current_eq, "buy_hold", "#444444", 1.5),
        (current_eq, "current_exit_net_1x", "#1565C0", 2.0),
        (alt_eq, "entry_falls_below_exit_net_1x", "#2E7D32", 2.0),
        (current_eq, "current_exit_full_gross", "#8E24AA", 1.5),
        (alt_eq, "entry_falls_below_exit_full_gross", "#C62828", 1.5),
    ):
        col = {
            "buy_hold": "buy_hold",
            "current_exit_net_1x": "net_1x_style",
            "entry_falls_below_exit_net_1x": "net_1x_style",
            "current_exit_full_gross": "combined_full_gross",
            "entry_falls_below_exit_full_gross": "combined_full_gross",
        }[label]
        ax.plot(
            pd.to_datetime(eq["timestamp"], utc=True).dt.tz_convert("America/New_York"),
            eq[col],
            label=label,
            color=color,
            linewidth=width,
        )
    ax.set_title(f"{symbol} | exit regime comparison")
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

    current_trace = _run_trace(
        meta_df=meta_df,
        model_root=Path(args.model_root),
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        min_hold_bars=max(0, int(args.min_hold_bars)),
        exit_entry_delta=float(args.exit_entry_delta),
        confirm_bars=max(1, int(args.confirm_bars)),
        exit_mode="current",
    )
    alt_trace = _run_trace(
        meta_df=meta_df,
        model_root=Path(args.model_root),
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        min_hold_bars=max(0, int(args.min_hold_bars)),
        exit_entry_delta=float(args.exit_entry_delta),
        confirm_bars=max(1, int(args.confirm_bars)),
        exit_mode="enter_falls_below",
    )

    alt_trace["symbol"] = args.symbol
    alt_trace["volume"] = pd.to_numeric(meta_df["volume"], errors="coerce").to_numpy()[: len(alt_trace)]
    alt_trace["ts_local"] = pd.to_datetime(alt_trace["timestamp"], utc=True, errors="coerce").dt.tz_convert(args.tz)

    current_eq = _equity_from_trace(current_trace)
    alt_eq = _equity_from_trace(alt_trace)
    _save_equity_plot(current_eq=current_eq, alt_eq=alt_eq, save_path=Path(args.equity_out), symbol=args.symbol)

    alt_trace_path = Path(args.alt_trace_out)
    alt_trace_path.parent.mkdir(parents=True, exist_ok=True)
    alt_trace.to_csv(alt_trace_path, index=False)
    _save_trace_plot(alt_trace, save_path=Path(args.alt_plot_out), symbol=args.symbol)

    summary = pd.DataFrame(
        [
            {
                "regime": "current_exit_logic",
                **_trade_metrics(current_trace),
                "buy_hold_end": float(current_eq["buy_hold"].iloc[-1]),
                "net_1x_end": float(current_eq["net_1x_style"].iloc[-1]),
                "full_gross_end": float(current_eq["combined_full_gross"].iloc[-1]),
            },
            {
                "regime": "entry_falls_below_threshold_exit",
                **_trade_metrics(alt_trace),
                "buy_hold_end": float(alt_eq["buy_hold"].iloc[-1]),
                "net_1x_end": float(alt_eq["net_1x_style"].iloc[-1]),
                "full_gross_end": float(alt_eq["combined_full_gross"].iloc[-1]),
            },
        ]
    )
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nsummary_csv={summary_path}")
    print(f"equity_png={Path(args.equity_out)}")
    print(f"alt_trace_csv={alt_trace_path}")
    print(f"alt_plot_png={Path(args.alt_plot_out)}")


if __name__ == "__main__":
    main()
