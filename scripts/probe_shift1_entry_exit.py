from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.spy_intraday.Models.ga_xgboost.analyze_phase4_triggers import (  # noqa: E402
    _add_phase4_features,
    _evaluate_asymmetric_policy,
    _load_execution_1m,
    _load_phase4_inputs,
)


POLICIES: dict[str, tuple[str, str | None]] = {
    "next_open": ("next_open", None),
    "next_open_direction": ("next_open_direction", None),
    "break_touch": ("break_prev_stop", None),
    "break_1m_body": ("break_prev_stop_1m_body", None),
    "break_1m_body_and_close": ("break_prev_stop_1m_body_and_close", None),
    "micro_reversal_1m": ("micro_reversal_1m", None),
}


def _eval(
    feature_df: pd.DataFrame,
    p_long,
    p_short,
    eval_idx,
    execution_1m: pd.DataFrame,
    *,
    long_name: str,
    short_name: str,
    max_bars: int,
    cluster: bool,
    long_threshold: float,
    short_threshold: float,
    horizon_bars: int,
    tp_atr: float,
    sl_atr: float,
) -> dict[str, object]:
    long_policy, long_trigger = POLICIES[long_name]
    short_policy, short_trigger = POLICIES[short_name]
    row, *_ = _evaluate_asymmetric_policy(
        feature_df,
        p_long,
        p_short,
        split_name="test",
        eval_idx=eval_idx,
        long_setup_threshold=long_threshold,
        short_setup_threshold=short_threshold,
        cooldown_bars=5,
        one_per_setup_cluster=cluster,
        long_policy_name=long_name,
        long_policy=long_policy,
        long_trigger_col=long_trigger,
        short_policy_name=short_name,
        short_policy=short_policy,
        short_trigger_col=short_trigger,
        max_bars=max_bars,
        horizon_bars=horizon_bars,
        tp_atr=tp_atr,
        sl_atr=sl_atr,
        execution_1m=execution_1m,
    )
    row.update(
        {
            "long_policy_name": long_name,
            "short_policy_name": short_name,
            "max_bars": int(max_bars),
            "cluster": bool(cluster),
            "horizon_bars": int(horizon_bars),
            "tp_atr": float(tp_atr),
            "sl_atr": float(sl_atr),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Small targeted SPY shift1 entry/exit probe.")
    parser.add_argument("--long-thresholds", default="0.42")
    parser.add_argument("--short-thresholds", default="0.15")
    parser.add_argument("--output-dir", default="Data/models/ga_xgboost/10min_shift1/analysis/shift1_probe")
    args = parser.parse_args()

    load_args = argparse.Namespace(
        ticker="SPY",
        dataset_name="10min_shift1",
        x_filename="X_10min_shift1_tree.parquet",
        model_root="Data/models",
        single_label_dir="swing_support_single",
        split_root=None,
        use_1m_execution=True,
        execution_1m_path="Data/raw/spy/spy_intraday_1min.parquet",
    )
    plot_df, _y_df, _probs_df, loaded = _load_phase4_inputs(load_args)
    feature_df = _add_phase4_features(plot_df, derive_vwap=False)
    execution_1m = _load_execution_1m(load_args, feature_df.index)
    if execution_1m is None:
        raise RuntimeError("1m execution data required")

    p_long = loaded["p_long_test"]
    p_short = loaded["p_short_test"]
    eval_idx = loaded["test"]
    long_thresholds = [float(x) for x in args.long_thresholds.split(",") if x.strip()]
    short_thresholds = [float(x) for x in args.short_thresholds.split(",") if x.strip()]

    rows: list[dict[str, object]] = []
    policy_pairs = [
        ("next_open", "next_open"),
        ("break_touch", "break_touch"),
        ("break_1m_body_and_close", "break_1m_body_and_close"),
        ("micro_reversal_1m", "micro_reversal_1m"),
    ]
    exit_grid = [
        (6, 0.8, 0.6),
        (8, 1.0, 0.8),
        (12, 1.0, 0.8),
        (16, 1.5, 1.0),
    ]

    for long_threshold in long_thresholds:
        for short_threshold in short_thresholds:
            for long_name, short_name in policy_pairs:
                for max_bars in (1, 2, 4):
                    for horizon_bars, tp_atr, sl_atr in exit_grid:
                        rows.append(
                            _eval(
                                feature_df,
                                p_long,
                                p_short,
                                eval_idx,
                                execution_1m,
                                long_name=long_name,
                                short_name=short_name,
                                max_bars=max_bars,
                                cluster=True,
                                long_threshold=long_threshold,
                                short_threshold=short_threshold,
                                horizon_bars=horizon_bars,
                                tp_atr=tp_atr,
                                sl_atr=sl_atr,
                            )
                        )

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("total_ev_atr", ascending=False)
    csv_path = out_dir / "shift1_probe_sweep.csv"
    json_path = out_dir / "shift1_probe_summary.json"
    df.to_csv(csv_path, index=False)
    summary = {
        "rows": int(len(df)),
        "best": df.head(20).to_dict(orient="records"),
        "csv": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[probe] wrote {csv_path}")
    print(f"[probe] wrote {json_path}")
    print(df.head(20)[[
        "long_setup_threshold",
        "short_setup_threshold",
        "long_policy_name",
        "short_policy_name",
        "max_bars",
        "horizon_bars",
        "tp_atr",
        "sl_atr",
        "total_ev_atr",
        "long_ev_atr",
        "short_ev_atr",
        "total_trades",
        "long_trades",
        "short_trades",
        "total_trades_per_day",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
