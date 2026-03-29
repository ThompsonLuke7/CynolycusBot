from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_baseline_vs_profit_protect_exit import (  # noqa: E402
    _event_metrics,
    _load_one_min,
    _normalize_bounds,
    _run_regime,
)
from scripts.replay_meta_independent import _load_meta_matrix  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep profit-protect arm/giveback parameters against the baseline exit."
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
        "--arm-values",
        default="2.5,3.0",
        help="Comma-separated profit-protect arm ATR values.",
    )
    parser.add_argument(
        "--giveback-values",
        default="0.6,0.8,1.0,1.2",
        help="Comma-separated giveback ATR values applied symmetrically to long/short.",
    )
    parser.add_argument(
        "--summary-out",
        default="Data/inference/spy/10min/meta/profit_protect_grid_sweep_summary.csv",
        help="CSV summary path.",
    )
    return parser.parse_args()


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def main() -> None:
    args = _parse_args()
    start, end = _normalize_bounds(args.start, args.end)
    meta_df = _load_meta_matrix(Path(args.meta_matrix), start=start, end=end, tz=args.tz)
    one_min = _load_one_min(Path(args.one_min_data), symbol=args.symbol, start=start, end=end)

    common_kwargs = dict(
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

    _, baseline_events = _run_regime(
        **common_kwargs,
        profit_protect_arm_atr=None,
        profit_protect_giveback_long=None,
        profit_protect_giveback_short=None,
    )
    baseline_metrics = _event_metrics(baseline_events)
    rows: list[dict[str, float | str]] = [
        {
            "regime": "baseline",
            "arm_atr": float("nan"),
            "giveback_atr": float("nan"),
            **baseline_metrics,
        }
    ]

    arm_values = _parse_float_list(args.arm_values)
    giveback_values = _parse_float_list(args.giveback_values)

    for arm_atr in arm_values:
        for giveback_atr in giveback_values:
            _, variant_events = _run_regime(
                **common_kwargs,
                profit_protect_arm_atr=float(arm_atr),
                profit_protect_giveback_long=float(giveback_atr),
                profit_protect_giveback_short=float(giveback_atr),
            )
            metrics = _event_metrics(variant_events)
            rows.append(
                {
                    "regime": f"profit_protect_a{arm_atr:.2f}_gb{giveback_atr:.2f}",
                    "arm_atr": float(arm_atr),
                    "giveback_atr": float(giveback_atr),
                    **metrics,
                }
            )

    summary = pd.DataFrame(rows)
    baseline_combined = float(summary.loc[summary["regime"].eq("baseline"), "combined_full_gross_end"].iloc[0])
    summary["delta_vs_baseline"] = summary["combined_full_gross_end"] - baseline_combined
    summary["rank_combined"] = summary["combined_full_gross_end"].rank(method="min", ascending=False)
    summary = summary.sort_values(
        by=["combined_full_gross_end", "short_equity_end", "long_equity_end"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    display_cols = [
        "regime",
        "arm_atr",
        "giveback_atr",
        "combined_full_gross_end",
        "delta_vs_baseline",
        "long_equity_end",
        "short_equity_end",
        "long_win_rate",
        "short_win_rate",
        "profit_protect_exit_count",
        "rank_combined",
    ]
    print(summary[display_cols].to_string(index=False))
    print(f"\nsummary_csv={out_path}")


if __name__ == "__main__":
    main()
