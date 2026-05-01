from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.ga_xgboost.analyze_phase4_triggers import (  # noqa: E402
    _add_phase4_features,
    _apply_conflict_cooldown_cluster,
    _load_execution_1m,
    _load_phase4_inputs,
    _session_reset_mask,
    _trade_metrics_for_entries,
)
from scripts.fast_shift1_policy_search import _active_setup, _candidate_from_setups, _precompute_triggers


def _parse(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _metrics_row(
    feature_df: pd.DataFrame,
    eval_idx: np.ndarray,
    execution_1m: pd.DataFrame,
    long_entries: np.ndarray,
    short_entries: np.ndarray,
    long_prices: np.ndarray,
    short_prices: np.ndarray,
    long_times: np.ndarray,
    short_times: np.ndarray,
    *,
    horizon: int,
    tp: float,
    sl: float,
) -> tuple[dict[str, float], dict[str, float], float]:
    lm = _trade_metrics_for_entries(
        feature_df,
        long_entries,
        side="long",
        eval_idx=eval_idx,
        horizon_bars=horizon,
        tp_atr=tp,
        sl_atr=sl,
        entry_prices=long_prices,
        entry_times=long_times,
        execution_1m=execution_1m,
    )
    sm = _trade_metrics_for_entries(
        feature_df,
        short_entries,
        side="short",
        eval_idx=eval_idx,
        horizon_bars=horizon,
        tp_atr=tp,
        sl_atr=sl,
        entry_prices=short_prices,
        entry_times=short_times,
        execution_1m=execution_1m,
    )
    total = lm["trades"] + sm["trades"]
    total_ev = np.nan
    if total:
        total_ev = (lm["ev_atr"] * lm["trades"] + sm["ev_atr"] * sm["trades"]) / total
    return lm, sm, float(total_ev)


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused tradable shift1 body-close search with origin setup filters.")
    parser.add_argument("--output-dir", default="Data/models/ga_xgboost/10min_shift1/analysis/focused_bodyclose_origin")
    parser.add_argument("--long-thresholds", default="0.35,0.42,0.50")
    parser.add_argument("--short-thresholds", default="0.10,0.15,0.24,0.35")
    parser.add_argument("--min-edges", default="-999,-0.2,-0.1,0.0")
    parser.add_argument("--max-opps", default="999,0.35,0.25,0.15")
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
    raw_eval_idx = np.sort(loaded["test"])
    slice_start = max(0, int(raw_eval_idx.min()) - 20)
    slice_stop = min(len(feature_df), int(raw_eval_idx.max()) + 20)
    feature_df = feature_df.iloc[slice_start:slice_stop].copy()
    execution_1m = _load_execution_1m(load_args, feature_df.index)
    if execution_1m is None:
        raise RuntimeError("1m execution data required")

    p_long = loaded["p_long_test"][slice_start:slice_stop]
    p_short = loaded["p_short_test"][slice_start:slice_stop]
    eval_idx = raw_eval_idx[(raw_eval_idx >= slice_start) & (raw_eval_idx < slice_stop)] - slice_start
    eval_mask = np.zeros(len(feature_df), dtype=bool)
    eval_mask[eval_idx] = True
    finite = np.isfinite(p_long) & np.isfinite(p_short)

    long_trigger = _precompute_triggers(feature_df, execution_1m, "long", "break_body_close")
    short_trigger = _precompute_triggers(feature_df, execution_1m, "short", "break_body_close")
    rows: list[dict[str, object]] = []
    exit_grid = [(6, 0.8, 0.6), (8, 1.0, 0.8), (12, 1.0, 0.8), (16, 1.5, 1.0)]

    for long_threshold in _parse(args.long_thresholds):
        for short_threshold in _parse(args.short_thresholds):
            for min_edge in _parse(args.min_edges):
                for max_opp in _parse(args.max_opps):
                    long_raw = eval_mask & finite & (p_long >= long_threshold)
                    short_raw = eval_mask & finite & (p_short >= short_threshold)
                    if min_edge > -100:
                        long_raw &= (p_long - p_short) >= min_edge
                        short_raw &= (p_short - p_long) >= min_edge
                    if max_opp < 100:
                        long_raw &= p_short <= max_opp
                        short_raw &= p_long <= max_opp
                    for max_bars in (1, 2, 4, 6):
                        lc, lp, lt, _lo = _candidate_from_setups(feature_df, long_raw, long_trigger, side="long", max_bars=max_bars)
                        sc, sp, st, _so = _candidate_from_setups(feature_df, short_raw, short_trigger, side="short", max_bars=max_bars)
                        for max_intrabar_min in (2.0, 4.0, 6.0, 10.0):
                            lc2 = lc.copy()
                            sc2 = sc.copy()
                            for signal, times in ((lc2, lt), (sc2, st)):
                                idxs = np.flatnonzero(signal)
                                for i in idxs:
                                    if pd.isna(times[i]):
                                        signal[i] = False
                                        continue
                                    lag = (pd.Timestamp(times[i]) - pd.Timestamp(feature_df.index[i])).total_seconds() / 60.0
                                    if lag > max_intrabar_min:
                                        signal[i] = False
                            le, se, stats = _apply_conflict_cooldown_cluster(
                                lc2,
                                sc2,
                                _active_setup(feature_df, long_raw, max_bars),
                                _active_setup(feature_df, short_raw, max_bars),
                                cooldown_bars=5,
                                one_per_setup_cluster=True,
                                session_reset=_session_reset_mask(feature_df.index),
                            )
                            for horizon, tp, sl in exit_grid:
                                lm, sm, total_ev = _metrics_row(
                                    feature_df,
                                    eval_idx,
                                    execution_1m,
                                    le,
                                    se,
                                    lp,
                                    sp,
                                    lt,
                                    st,
                                    horizon=horizon,
                                    tp=tp,
                                    sl=sl,
                                )
                                total = lm["trades"] + sm["trades"]
                                if total < 150:
                                    continue
                                rows.append(
                                    {
                                        "long_threshold": long_threshold,
                                        "short_threshold": short_threshold,
                                        "min_edge": min_edge,
                                        "max_opp": max_opp,
                                        "max_bars": max_bars,
                                        "max_intrabar_min": max_intrabar_min,
                                        "horizon_bars": horizon,
                                        "tp_atr": tp,
                                        "sl_atr": sl,
                                        "total_ev_atr": total_ev,
                                        "total_trades": total,
                                        "long_ev_atr": lm["ev_atr"],
                                        "short_ev_atr": sm["ev_atr"],
                                        "long_trades": lm["trades"],
                                        "short_trades": sm["trades"],
                                        "long_win_rate": lm["win_rate"],
                                        "short_win_rate": sm["win_rate"],
                                        **stats,
                                    }
                                )

    out = pd.DataFrame(rows).sort_values("total_ev_atr", ascending=False)
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "focused_bodyclose_origin_search.csv"
    summary_path = out_dir / "focused_bodyclose_origin_summary.json"
    out.to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps({"rows": len(out), "best": out.head(30).to_dict(orient="records")}, indent=2))
    print(f"[focused] wrote {csv_path}")
    print(f"[focused] wrote {summary_path}")
    print(out.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
