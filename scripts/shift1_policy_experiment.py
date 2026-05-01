from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.ga_xgboost.analyze_phase4_triggers import (  # noqa: E402
    _add_phase4_features,
    _evaluate_asymmetric_policy,
    _load_execution_1m,
    _load_phase4_inputs,
)


def _policy_catalog() -> dict[str, tuple[str, str | None]]:
    return {
        "next_open": ("next_open", None),
        "next_open_direction": ("next_open_direction", None),
        "break_touch": ("break_prev_stop", None),
        "break_1m_confirm": ("break_prev_stop_1m_confirm", None),
        "break_1m_body": ("break_prev_stop_1m_body", None),
        "break_1m_momentum": ("break_prev_stop_1m_momentum", None),
        "break_1m_body_or_close": ("break_prev_stop_1m_body_or_close", None),
        "break_1m_body_and_close": ("break_prev_stop_1m_body_and_close", None),
        "break_1m_confirm_no_fresh": ("break_prev_stop_1m_confirm_no_fresh_low", None),
        "break_1m_body_and_close_no_fresh": ("break_prev_stop_1m_body_and_close_no_fresh_low", None),
        "micro_reversal_1m": ("micro_reversal_1m", None),
        "ema_break": ("trigger_close", "trigger_E"),
        "body_ema": ("trigger_close", "trigger_H"),
        "ema_slope": ("trigger_close", "trigger_G"),
        "break_or_ema": ("trigger_close", "trigger_I"),
    }


def _candidate_pairs() -> list[tuple[str, str]]:
    symmetric = [
        "next_open",
        "next_open_direction",
        "break_touch",
        "break_1m_confirm",
        "break_1m_body",
        "break_1m_momentum",
        "break_1m_body_or_close",
        "break_1m_body_and_close",
        "micro_reversal_1m",
        "ema_break",
        "body_ema",
        "ema_slope",
        "break_or_ema",
    ]
    long_focus = [
        "next_open",
        "break_touch",
        "break_1m_confirm",
        "break_1m_body_and_close",
        "break_1m_confirm_no_fresh",
        "break_1m_body_and_close_no_fresh",
        "micro_reversal_1m",
        "ema_break",
        "body_ema",
    ]
    short_focus = [
        "next_open",
        "break_touch",
        "break_1m_confirm",
        "break_1m_body_or_close",
        "break_1m_body_and_close",
        "micro_reversal_1m",
        "ema_break",
        "body_ema",
    ]
    pairs = {(name, name) for name in symmetric}
    pairs.update((long_name, short_name) for long_name in long_focus for short_name in short_focus)
    return sorted(pairs)


def _eval_pair(
    feature_df: pd.DataFrame,
    p_long,
    p_short,
    eval_idx,
    *,
    long_name: str,
    short_name: str,
    max_bars: int,
    one_per_cluster: bool,
    long_threshold: float,
    short_threshold: float,
    cooldown_bars: int,
    horizon_bars: int,
    tp_atr: float,
    sl_atr: float,
    execution_1m: pd.DataFrame,
) -> dict[str, object]:
    catalog = _policy_catalog()
    long_policy, long_trigger = catalog[long_name]
    short_policy, short_trigger = catalog[short_name]
    if long_trigger is not None:
        long_trigger = f"{long_trigger}_long"
    if short_trigger is not None:
        short_trigger = f"{short_trigger}_short"

    row, *_ = _evaluate_asymmetric_policy(
        feature_df,
        p_long,
        p_short,
        split_name="test",
        eval_idx=eval_idx,
        long_setup_threshold=long_threshold,
        short_setup_threshold=short_threshold,
        cooldown_bars=cooldown_bars,
        one_per_setup_cluster=one_per_cluster,
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
            "cluster_mode": "cooldown_cluster" if one_per_cluster else "cooldown_only",
            "horizon_bars": int(horizon_bars),
            "tp_atr": float(tp_atr),
            "sl_atr": float(sl_atr),
        }
    )
    return row


def _plot_entry_summary(entry_df: pd.DataFrame, path: Path, baseline_ev: float) -> None:
    top = entry_df.sort_values("total_ev_atr", ascending=False).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 8))
    labels = [
        f"L:{r.long_policy_name} | S:{r.short_policy_name} | max{int(r.max_bars)} | {r.cluster_mode.replace('cooldown_', '')}"
        for r in top.itertuples()
    ]
    colors = ["#16a34a" if i == len(top) - 1 else "#5b8ff9" for i in range(len(top))]
    ax.barh(labels, top["total_ev_atr"], color=colors)
    ax.axvline(baseline_ev, color="#dc2626", linestyle="--", linewidth=1.4, label=f"baseline {baseline_ev:.4f}")
    ax.set_xlabel("test total EV / ATR per trade")
    ax.set_title("SPY shift1 entry-policy sweep, top 20")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_exit_summary(exit_df: pd.DataFrame, path: Path, baseline_ev: float) -> None:
    top = exit_df.sort_values("total_ev_atr", ascending=False).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 8))
    labels = [
        f"{r.entry_rank}: h{int(r.horizon_bars)} tp{r.tp_atr:g}/sl{r.sl_atr:g} | L:{r.long_policy_name} S:{r.short_policy_name}"
        for r in top.itertuples()
    ]
    colors = ["#16a34a" if i == len(top) - 1 else "#7c3aed" for i in range(len(top))]
    ax.barh(labels, top["total_ev_atr"], color=colors)
    ax.axvline(baseline_ev, color="#dc2626", linestyle="--", linewidth=1.4, label=f"baseline {baseline_ev:.4f}")
    ax.set_xlabel("test total EV / ATR per trade")
    ax.set_title("SPY shift1 exit-policy sweep, top 20")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded entry/exit policy sweep for the shifted-1-bar SPY model.")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--dataset-name", default="10min_shift1")
    parser.add_argument("--x-filename", default="X_10min_shift1_tree.parquet")
    parser.add_argument("--model-root", default="Data/models")
    parser.add_argument("--single-label-dir", default="swing_support_single")
    parser.add_argument("--split-root", default=None)
    parser.add_argument("--execution-1m-path", default="Data/raw/spy/spy_intraday_1min.parquet")
    parser.add_argument("--long-threshold", type=float, default=0.42)
    parser.add_argument("--short-threshold", type=float, default=0.35)
    parser.add_argument("--cooldown-bars", type=int, default=5)
    parser.add_argument("--entry-max-bars", default="1,2,4,6")
    parser.add_argument("--top-entry-count", type=int, default=8)
    parser.add_argument("--output-dir", default="Data/models/ga_xgboost/10min_shift1/analysis/policy_experiment")
    args = parser.parse_args()

    load_args = argparse.Namespace(
        ticker=args.ticker,
        dataset_name=args.dataset_name,
        x_filename=args.x_filename,
        model_root=args.model_root,
        single_label_dir=args.single_label_dir,
        split_root=args.split_root,
        use_1m_execution=True,
        execution_1m_path=args.execution_1m_path,
    )
    plot_df, _y_df, _probs_df, loaded = _load_phase4_inputs(load_args)
    feature_df = _add_phase4_features(plot_df, derive_vwap=False)
    execution_1m = _load_execution_1m(load_args, feature_df.index)
    if execution_1m is None:
        raise RuntimeError("1m execution data is required for this experiment")

    p_long = loaded["p_long_test"]
    p_short = loaded["p_short_test"]
    eval_idx = loaded["test"]
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    max_bars_values = [int(x.strip()) for x in args.entry_max_bars.split(",") if x.strip()]
    entry_rows: list[dict[str, object]] = []
    for long_name, short_name in _candidate_pairs():
        for max_bars in max_bars_values:
            for one_per_cluster in (False, True):
                entry_rows.append(
                    _eval_pair(
                        feature_df,
                        p_long,
                        p_short,
                        eval_idx,
                        long_name=long_name,
                        short_name=short_name,
                        max_bars=max_bars,
                        one_per_cluster=one_per_cluster,
                        long_threshold=args.long_threshold,
                        short_threshold=args.short_threshold,
                        cooldown_bars=args.cooldown_bars,
                        horizon_bars=12,
                        tp_atr=1.0,
                        sl_atr=0.8,
                        execution_1m=execution_1m,
                    )
                )
    entry_df = pd.DataFrame(entry_rows).sort_values("total_ev_atr", ascending=False)
    entry_csv = out_dir / "shift1_entry_policy_sweep.csv"
    entry_df.to_csv(entry_csv, index=False)

    exit_rows: list[dict[str, object]] = []
    top_entries = entry_df.head(args.top_entry_count).copy()
    exit_grid = [
        (6, 0.8, 0.6),
        (6, 1.0, 0.8),
        (8, 0.8, 0.6),
        (8, 1.0, 0.8),
        (12, 1.0, 0.8),
        (12, 1.2, 0.8),
        (12, 1.5, 1.0),
        (16, 1.0, 0.8),
        (16, 1.2, 0.8),
        (16, 1.5, 1.0),
        (20, 1.2, 0.8),
        (20, 1.5, 1.0),
        (24, 1.5, 1.0),
        (24, 2.0, 1.0),
    ]
    for rank, entry in enumerate(top_entries.itertuples(), start=1):
        for horizon_bars, tp_atr, sl_atr in exit_grid:
            row = _eval_pair(
                feature_df,
                p_long,
                p_short,
                eval_idx,
                long_name=str(entry.long_policy_name),
                short_name=str(entry.short_policy_name),
                max_bars=int(entry.max_bars),
                one_per_cluster=str(entry.cluster_mode) == "cooldown_cluster",
                long_threshold=args.long_threshold,
                short_threshold=args.short_threshold,
                cooldown_bars=args.cooldown_bars,
                horizon_bars=horizon_bars,
                tp_atr=tp_atr,
                sl_atr=sl_atr,
                execution_1m=execution_1m,
            )
            row["entry_rank"] = rank
            row["entry_default_ev_atr"] = float(entry.total_ev_atr)
            exit_rows.append(row)
    exit_df = pd.DataFrame(exit_rows).sort_values("total_ev_atr", ascending=False)
    exit_csv = out_dir / "shift1_exit_policy_sweep.csv"
    exit_df.to_csv(exit_csv, index=False)

    baseline_ev = 0.2037438063752998
    entry_plot = out_dir / "shift1_entry_policy_sweep_top20.png"
    exit_plot = out_dir / "shift1_exit_policy_sweep_top20.png"
    _plot_entry_summary(entry_df, entry_plot, baseline_ev)
    _plot_exit_summary(exit_df, exit_plot, baseline_ev)

    summary = {
        "thresholds": {"long": args.long_threshold, "short": args.short_threshold},
        "baseline_best_total_ev_atr": baseline_ev,
        "best_entry_default_exit": entry_df.head(1).to_dict(orient="records")[0],
        "best_entry_exit_combo": exit_df.head(1).to_dict(orient="records")[0],
        "entry_csv": str(entry_csv),
        "exit_csv": str(exit_csv),
        "entry_plot": str(entry_plot),
        "exit_plot": str(exit_plot),
    }
    summary_path = out_dir / "shift1_policy_experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"[policy] wrote {entry_csv}")
    print(f"[policy] wrote {exit_csv}")
    print(f"[policy] wrote {entry_plot}")
    print(f"[policy] wrote {exit_plot}")
    print(f"[policy] wrote {summary_path}")
    print("[policy] best entry/default exit:")
    print(entry_df.head(10)[[
        "long_policy_name",
        "short_policy_name",
        "max_bars",
        "cluster_mode",
        "total_ev_atr",
        "long_ev_atr",
        "short_ev_atr",
        "total_trades",
        "long_trades",
        "short_trades",
        "total_trades_per_day",
    ]].to_string(index=False))
    print("[policy] best entry/exit:")
    print(exit_df.head(10)[[
        "entry_rank",
        "long_policy_name",
        "short_policy_name",
        "max_bars",
        "cluster_mode",
        "horizon_bars",
        "tp_atr",
        "sl_atr",
        "total_ev_atr",
        "long_ev_atr",
        "short_ev_atr",
        "total_trades",
        "long_trades",
        "short_trades",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
