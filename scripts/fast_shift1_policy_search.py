from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.spy_intraday.Models.ga_xgboost.analyze_phase4_triggers import (  # noqa: E402
    _add_phase4_features,
    _apply_conflict_cooldown_cluster,
    _load_execution_1m,
    _load_phase4_inputs,
    _session_reset_mask,
    _trade_metrics_for_entries,
)


@dataclass(frozen=True)
class TriggerCache:
    hit: np.ndarray
    price: np.ndarray
    time: np.ndarray


def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _bar_windows(index: pd.DatetimeIndex, execution_1m: pd.DataFrame) -> list[pd.DataFrame]:
    minute_index = execution_1m.index
    windows: list[pd.DataFrame] = []
    for i in range(len(index)):
        if i + 1 >= len(index) or index[i].normalize() != index[i + 1].normalize():
            windows.append(execution_1m.iloc[0:0])
            continue
        left = minute_index.searchsorted(index[i], side="right")
        right = minute_index.searchsorted(index[i + 1], side="right")
        windows.append(execution_1m.iloc[left:right])
    return windows


def _precompute_triggers(feature_df: pd.DataFrame, execution_1m: pd.DataFrame, side: str, policy: str) -> TriggerCache:
    n = len(feature_df)
    hit = np.zeros(n, dtype=bool)
    price = np.full(n, np.nan, dtype=float)
    time = np.full(n, pd.NaT, dtype=object)
    open_ = pd.to_numeric(feature_df["open"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(feature_df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(feature_df["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(feature_df["close"], errors="coerce").to_numpy(dtype=float)
    windows = _bar_windows(feature_df.index, execution_1m)

    for i in range(1, n - 1):
        if policy == "next_open":
            hit[i] = np.isfinite(open_[i])
            price[i] = open_[i]
            time[i] = feature_df.index[i]
            continue
        if policy == "setup_close":
            hit[i] = np.isfinite(close[i - 1])
            price[i] = close[i - 1]
            time[i] = feature_df.index[i]
            continue

        window = windows[i]
        if window.empty:
            continue
        w_open = pd.to_numeric(window["open"], errors="coerce")
        w_high = pd.to_numeric(window["high"], errors="coerce")
        w_low = pd.to_numeric(window["low"], errors="coerce")
        w_close = pd.to_numeric(window["close"], errors="coerce")

        if policy == "micro_reversal":
            cand = window[(w_close > w_open) & (w_close > w_high.shift(1))] if side == "long" else window[(w_close < w_open) & (w_close < w_low.shift(1))]
            if not cand.empty:
                hit[i] = True
                price[i] = float(pd.to_numeric(cand["close"], errors="coerce").iloc[0])
                time[i] = cand.index[0]
            continue

        stop = high[i - 1] if side == "long" else low[i - 1]
        if not np.isfinite(stop):
            continue
        cand = window[w_high >= stop] if side == "long" else window[w_low <= stop]
        if cand.empty:
            continue
        c_open = pd.to_numeric(cand["open"], errors="coerce")
        c_close = pd.to_numeric(cand["close"], errors="coerce")
        if policy == "break_body":
            cand = cand[c_close > c_open] if side == "long" else cand[c_close < c_open]
        elif policy == "break_body_close":
            cand = cand[(c_close > c_open) & (c_close > stop)] if side == "long" else cand[(c_close < c_open) & (c_close < stop)]
        elif policy != "break_touch":
            raise ValueError(f"unknown policy {policy}")
        if cand.empty:
            continue
        hit[i] = True
        price[i] = float(stop)
        time[i] = cand.index[0]
    return TriggerCache(hit=hit, price=price, time=time)


def _candidate_from_setups(
    feature_df: pd.DataFrame,
    raw_setup: np.ndarray,
    trigger: TriggerCache,
    *,
    side: str,
    max_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(feature_df)
    candidate = np.zeros(n, dtype=bool)
    prices = np.full(n, np.nan, dtype=float)
    times = np.full(n, pd.NaT, dtype=object)
    origins = np.full(n, -1, dtype=int)
    sessions = feature_df.index.normalize().to_numpy()

    for setup_idx in np.flatnonzero(raw_setup):
        end = min(n - 1, setup_idx + max(1, int(max_bars)))
        for i in range(setup_idx + 1, end + 1):
            if sessions[i] != sessions[setup_idx]:
                break
            if not trigger.hit[i] or not np.isfinite(trigger.price[i]):
                continue
            candidate[i] = True
            if not np.isfinite(prices[i]):
                prices[i] = trigger.price[i]
                times[i] = trigger.time[i]
                origins[i] = setup_idx
            else:
                better = trigger.price[i] < prices[i] if side == "long" else trigger.price[i] > prices[i]
                if better:
                    prices[i] = trigger.price[i]
                    times[i] = trigger.time[i]
                    origins[i] = setup_idx
            break
    return candidate, prices, times, origins


def _active_setup(feature_df: pd.DataFrame, raw_setup: np.ndarray, max_bars: int) -> np.ndarray:
    active = np.zeros(len(feature_df), dtype=bool)
    sessions = feature_df.index.normalize().to_numpy()
    for setup_idx in np.flatnonzero(raw_setup):
        end = min(len(feature_df) - 1, setup_idx + max(1, int(max_bars)))
        for i in range(setup_idx + 1, end + 1):
            if sessions[i] != sessions[setup_idx]:
                break
            active[i] = True
    return active


def _row(
    feature_df: pd.DataFrame,
    eval_idx: np.ndarray,
    long_entries: np.ndarray,
    short_entries: np.ndarray,
    long_prices: np.ndarray,
    short_prices: np.ndarray,
    long_times: np.ndarray,
    short_times: np.ndarray,
    execution_1m: pd.DataFrame,
    *,
    long_threshold: float,
    short_threshold: float,
    min_edge: float,
    max_opp: float,
    long_policy: str,
    short_policy: str,
    max_bars: int,
    horizon_bars: int,
    tp_atr: float,
    sl_atr: float,
    stats: dict[str, int],
) -> dict[str, object]:
    lm = _trade_metrics_for_entries(
        feature_df,
        long_entries,
        side="long",
        eval_idx=eval_idx,
        horizon_bars=horizon_bars,
        tp_atr=tp_atr,
        sl_atr=sl_atr,
        entry_prices=long_prices,
        entry_times=long_times,
        execution_1m=execution_1m,
    )
    sm = _trade_metrics_for_entries(
        feature_df,
        short_entries,
        side="short",
        eval_idx=eval_idx,
        horizon_bars=horizon_bars,
        tp_atr=tp_atr,
        sl_atr=sl_atr,
        entry_prices=short_prices,
        entry_times=short_times,
        execution_1m=execution_1m,
    )
    total = lm["trades"] + sm["trades"]
    ev = np.nan
    if total:
        ev = (lm["ev_atr"] * lm["trades"] + sm["ev_atr"] * sm["trades"]) / total
    return {
        "long_threshold": long_threshold,
        "short_threshold": short_threshold,
        "min_edge": min_edge,
        "max_opp": max_opp,
        "long_policy": long_policy,
        "short_policy": short_policy,
        "max_bars": max_bars,
        "horizon_bars": horizon_bars,
        "tp_atr": tp_atr,
        "sl_atr": sl_atr,
        "total_ev_atr": float(ev),
        "total_trades": float(total),
        "long_ev_atr": lm["ev_atr"],
        "short_ev_atr": sm["ev_atr"],
        "long_trades": lm["trades"],
        "short_trades": sm["trades"],
        "long_win_rate": lm["win_rate"],
        "short_win_rate": sm["win_rate"],
        "long_avg_mfe_atr": lm["avg_mfe_atr"],
        "long_avg_mae_atr": lm["avg_mae_atr"],
        "short_avg_mfe_atr": sm["avg_mfe_atr"],
        "short_avg_mae_atr": sm["avg_mae_atr"],
        **stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast cached shift1 entry/exit policy search.")
    parser.add_argument("--output-dir", default="Data/models/ga_xgboost/10min_shift1/analysis/fast_shift1_search")
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
    execution_1m = _load_execution_1m(load_args, feature_df.index)
    if execution_1m is None:
        raise RuntimeError("1m execution data required")
    p_long = loaded["p_long_test"]
    p_short = loaded["p_short_test"]
    eval_mask = np.zeros(len(feature_df), dtype=bool)
    eval_idx = loaded["test"]
    eval_mask[eval_idx] = True
    finite = np.isfinite(p_long) & np.isfinite(p_short)

    policies = ["next_open", "break_touch", "break_body", "break_body_close", "micro_reversal"]
    triggers = {
        (side, policy): _precompute_triggers(feature_df, execution_1m, side, policy)
        for side in ("long", "short")
        for policy in policies
    }
    pairs = [
        ("next_open", "next_open"),
        ("break_touch", "break_touch"),
        ("break_body", "break_body"),
        ("break_body_close", "break_body_close"),
        ("break_body_close", "break_touch"),
        ("break_body_close", "break_body"),
        ("micro_reversal", "micro_reversal"),
    ]
    exit_grid = [(6, 0.8, 0.6), (8, 1.0, 0.8), (12, 1.0, 0.8), (16, 1.5, 1.0)]
    rows: list[dict[str, object]] = []

    for long_threshold in _parse_floats(args.long_thresholds):
        for short_threshold in _parse_floats(args.short_thresholds):
            for min_edge in _parse_floats(args.min_edges):
                for max_opp in _parse_floats(args.max_opps):
                    long_raw = eval_mask & finite & (p_long >= long_threshold)
                    short_raw = eval_mask & finite & (p_short >= short_threshold)
                    if min_edge > -100:
                        long_raw &= (p_long - p_short) >= min_edge
                        short_raw &= (p_short - p_long) >= min_edge
                    if max_opp < 100:
                        long_raw &= p_short <= max_opp
                        short_raw &= p_long <= max_opp
                    for max_bars in (1, 2, 4, 6):
                        long_active = _active_setup(feature_df, long_raw, max_bars)
                        short_active = _active_setup(feature_df, short_raw, max_bars)
                        for long_policy, short_policy in pairs:
                            lc, lp, lt, _lo = _candidate_from_setups(
                                feature_df,
                                long_raw,
                                triggers[("long", long_policy)],
                                side="long",
                                max_bars=max_bars,
                            )
                            sc, sp, st, _so = _candidate_from_setups(
                                feature_df,
                                short_raw,
                                triggers[("short", short_policy)],
                                side="short",
                                max_bars=max_bars,
                            )
                            le, se, stats = _apply_conflict_cooldown_cluster(
                                lc,
                                sc,
                                long_active,
                                short_active,
                                cooldown_bars=5,
                                one_per_setup_cluster=True,
                                session_reset=_session_reset_mask(feature_df.index),
                            )
                            for horizon_bars, tp_atr, sl_atr in exit_grid:
                                rows.append(
                                    _row(
                                        feature_df,
                                        eval_idx,
                                        le,
                                        se,
                                        lp,
                                        sp,
                                        lt,
                                        st,
                                        execution_1m,
                                        long_threshold=long_threshold,
                                        short_threshold=short_threshold,
                                        min_edge=min_edge,
                                        max_opp=max_opp,
                                        long_policy=long_policy,
                                        short_policy=short_policy,
                                        max_bars=max_bars,
                                        horizon_bars=horizon_bars,
                                        tp_atr=tp_atr,
                                        sl_atr=sl_atr,
                                        stats=stats,
                                    )
                                )

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("total_ev_atr", ascending=False)
    csv_path = out_dir / "fast_shift1_policy_search.csv"
    summary_path = out_dir / "fast_shift1_policy_search_summary.json"
    df.to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps({"rows": len(df), "best": df.head(30).to_dict(orient="records"), "csv": str(csv_path)}, indent=2))
    print(f"[fast] wrote {csv_path}")
    print(f"[fast] wrote {summary_path}")
    print(df.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
