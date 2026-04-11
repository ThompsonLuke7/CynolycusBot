from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path.cwd()


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from Data.plots.plots import get_default_model_inference_plot_path, plot_model_inference
from Data.retrieve_data import normalize_ticker
from Models.ga_xgboost.rethreshold import _label_columns, _load_probs
from Models.ga_xgboost.train import (
    _find_best_fbeta_threshold,
    _load_split_indices,
    _normalize_ga_label_dir,
)


@dataclass(frozen=True)
class Signal:
    idx: int
    side: str
    prob: float


def _load_inputs(
    args: argparse.Namespace,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    clean = normalize_ticker(args.ticker)
    dataset_dir = REPO_ROOT / "Data" / "processed" / clean.lower() / "datasets" / args.dataset_name
    plot_df = pd.read_parquet(dataset_dir / "plot_frame.parquet")
    y_df = pd.read_parquet(dataset_dir / "y.parquet")
    label_dir = _normalize_ga_label_dir(args.label_mode)
    model_root = REPO_ROOT / args.model_root / "ga_xgboost" / args.dataset_name
    long_probs_path = model_root / "long" / label_dir / "p_long_probs.parquet"
    short_probs_path = model_root / "short" / label_dir / "p_short_probs.parquet"
    p_long_oof = _load_probs(long_probs_path, "p_long_oof_train")
    p_short_oof = _load_probs(short_probs_path, "p_short_oof_train")
    p_long_eval = _load_probs(long_probs_path, args.prob_column_long)
    p_short_eval = _load_probs(short_probs_path, args.prob_column_short)
    split_root = Path(args.split_root) if args.split_root else None
    splits = _load_split_indices(
        args.ticker,
        args.dataset_name,
        args.x_filename,
        split_root=split_root,
    )
    train_val_idx = np.sort(np.concatenate([np.sort(splits["train"]), np.sort(splits["val"])]))
    test_idx = np.sort(splits["test"])
    return plot_df, y_df, p_long_oof, p_short_oof, p_long_eval, p_short_eval, train_val_idx, test_idx


def _compute_atr(plot_df: pd.DataFrame, *, length: int = 14) -> np.ndarray:
    high = pd.to_numeric(plot_df["high"], errors="coerce")
    low = pd.to_numeric(plot_df["low"], errors="coerce")
    close = pd.to_numeric(plot_df["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / float(length), adjust=False, min_periods=length).mean().to_numpy(dtype=float)


def _thresholds(
    args: argparse.Namespace,
    y_df: pd.DataFrame,
    p_long_oof: np.ndarray,
    p_short_oof: np.ndarray,
    train_val_idx: np.ndarray,
) -> tuple[float, float]:
    if args.long_threshold is not None and args.short_threshold is not None:
        return float(args.long_threshold), float(args.short_threshold)
    long_col, short_col = _label_columns(args.label_mode)
    y_long = pd.to_numeric(y_df[long_col], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    y_short = pd.to_numeric(y_df[short_col], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    beta = float(args.threshold_beta)
    long_thr, long_score = _find_best_fbeta_threshold(
        y_long[train_val_idx],
        p_long_oof[train_val_idx],
        beta=beta,
    )
    short_thr, short_score = _find_best_fbeta_threshold(
        y_short[train_val_idx],
        p_short_oof[train_val_idx],
        beta=beta,
    )
    print(
        f"[filters] thresholds from OOF F{beta:g}: "
        f"long={long_thr:.4f} score={long_score:.4f}, "
        f"short={short_thr:.4f} score={short_score:.4f}"
    )
    return long_thr, short_thr


def _collapse_clusters(mask: np.ndarray, probs: np.ndarray) -> np.ndarray:
    keep = np.zeros(mask.shape[0], dtype=bool)
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    for start in starts:
        end = start
        while end + 1 < mask.shape[0] and mask[end + 1]:
            end += 1
        segment = np.arange(start, end + 1)
        best = int(segment[np.nanargmax(probs[segment])])
        keep[best] = True
    return keep


def _raw_signals(
    p_long: np.ndarray,
    p_short: np.ndarray,
    *,
    long_threshold: float,
    short_threshold: float,
    eval_idx: np.ndarray,
) -> list[Signal]:
    long_mask = np.zeros(p_long.shape[0], dtype=bool)
    short_mask = np.zeros(p_short.shape[0], dtype=bool)
    long_mask[eval_idx] = np.isfinite(p_long[eval_idx]) & (p_long[eval_idx] >= long_threshold)
    short_mask[eval_idx] = np.isfinite(p_short[eval_idx]) & (p_short[eval_idx] >= short_threshold)
    long_keep = _collapse_clusters(long_mask, p_long)
    short_keep = _collapse_clusters(short_mask, p_short)
    signals = [
        *(Signal(idx=int(i), side="long", prob=float(p_long[i])) for i in np.flatnonzero(long_keep)),
        *(Signal(idx=int(i), side="short", prob=float(p_short[i])) for i in np.flatnonzero(short_keep)),
    ]
    return sorted(signals, key=lambda s: s.idx)


def _filter_signals(
    signals: list[Signal],
    plot_df: pd.DataFrame,
    *,
    cooldown_bars: int,
    reset_atr_mult: float,
    atr_col: str,
) -> list[Signal]:
    close = pd.to_numeric(plot_df["close"], errors="coerce").to_numpy(dtype=float)
    atr = (
        pd.to_numeric(plot_df[atr_col], errors="coerce").to_numpy(dtype=float)
        if atr_col in plot_df.columns
        else _compute_atr(plot_df)
    )
    kept: list[Signal] = []
    last_idx: dict[str, int | None] = {"long": None, "short": None}
    last_entry: dict[str, float | None] = {"long": None, "short": None}
    since_opposite: dict[str, bool] = {"long": True, "short": True}

    for sig in signals:
        side = sig.side
        opp = "short" if side == "long" else "long"
        prev_idx = last_idx[side]
        if prev_idx is not None and sig.idx - prev_idx <= cooldown_bars:
            continue
        allow = since_opposite[side]
        if not allow and last_entry[side] is not None:
            atr_val = atr[sig.idx]
            if np.isfinite(atr_val) and atr_val > 0 and np.isfinite(close[sig.idx]):
                allow = abs(close[sig.idx] - float(last_entry[side])) >= reset_atr_mult * atr_val
        if not allow:
            continue
        kept.append(sig)
        last_idx[side] = sig.idx
        last_entry[side] = close[sig.idx] if np.isfinite(close[sig.idx]) else None
        since_opposite[side] = False
        since_opposite[opp] = True
    return kept


def _trade_stats(
    signals: list[Signal],
    plot_df: pd.DataFrame,
    y_df: pd.DataFrame,
    *,
    horizon_bars: int,
    atr_col: str,
    event_window_bars: int,
    slippage_bps: float,
    fee_bps: float,
) -> pd.DataFrame:
    high = pd.to_numeric(plot_df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(plot_df["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(plot_df["close"], errors="coerce").to_numpy(dtype=float)
    atr = (
        pd.to_numeric(plot_df[atr_col], errors="coerce").to_numpy(dtype=float)
        if atr_col in plot_df.columns
        else _compute_atr(plot_df)
    )
    long_actual = pd.to_numeric(y_df.get("long_swing_label", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    short_actual = pd.to_numeric(y_df.get("short_swing_label", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    rows: list[dict] = []
    cost = (float(slippage_bps) + float(fee_bps)) / 10000.0
    n = len(plot_df)
    for sig in signals:
        i = sig.idx
        end = min(n - 1, i + int(horizon_bars))
        if i >= end or not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        future_high = np.nanmax(high[i + 1 : end + 1])
        future_low = np.nanmin(low[i + 1 : end + 1])
        exit_close = close[end]
        if sig.side == "long":
            mfe_atr = (future_high - close[i]) / atr[i]
            mae_atr = (close[i] - future_low) / atr[i]
            ret = exit_close / close[i] - 1.0 - cost
            actual = long_actual
        else:
            mfe_atr = (close[i] - future_low) / atr[i]
            mae_atr = (future_high - close[i]) / atr[i]
            ret = close[i] / exit_close - 1.0 - cost
            actual = short_actual
        start = max(0, i - int(event_window_bars))
        stop = min(n, i + int(event_window_bars) + 1)
        hit = bool(np.any(actual[start:stop] == 1))
        rows.append(
            {
                "timestamp": plot_df.index[i],
                "bar_idx": int(i),
                "side": sig.side,
                "prob": float(sig.prob),
                "entry_close": float(close[i]),
                "exit_close": float(exit_close),
                "mfe_atr": float(mfe_atr),
                "mae_atr": float(mae_atr),
                "ret_net": float(ret),
                f"hit_within_{event_window_bars}_bars": hit,
            }
        )
    return pd.DataFrame(rows)


def _print_summary(df: pd.DataFrame, label: str) -> None:
    print(f"[filters] {label}: signals={len(df)}")
    if df.empty:
        return
    for side in ("long", "short", "all"):
        sub = df if side == "all" else df[df["side"] == side]
        if sub.empty:
            print(f"[filters] {label} {side}: signals=0")
            continue
        hit_col = [c for c in sub.columns if c.startswith("hit_within_")][0]
        print(
            f"[filters] {label} {side}: n={len(sub)}, "
            f"hit_rate={sub[hit_col].mean():.4f}, "
            f"avg_mfe={sub['mfe_atr'].mean():.4f}, "
            f"avg_mae={sub['mae_atr'].mean():.4f}, "
            f"expectancy_net={sub['ret_net'].mean():.5f}, "
            f"win_rate_net={(sub['ret_net'] > 0).mean():.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze GA-XGB swing predictions after signal filtering."
    )
    parser.add_argument("--ticker", type=str, default="SPY")
    parser.add_argument("--dataset-name", type=str, default="10min")
    parser.add_argument("--x-filename", type=str, default="X_10min_tree.parquet")
    parser.add_argument("--label-mode", type=str, default="swing")
    parser.add_argument("--model-root", type=str, default="Data/models")
    parser.add_argument("--split-root", type=str, default=None)
    parser.add_argument("--threshold-beta", type=float, default=0.5)
    parser.add_argument("--long-threshold", type=float, default=None)
    parser.add_argument("--short-threshold", type=float, default=None)
    parser.add_argument("--prob-column-long", type=str, default="p_long_test")
    parser.add_argument("--prob-column-short", type=str, default="p_short_test")
    parser.add_argument("--cooldown-bars", type=int, default=8)
    parser.add_argument("--reset-atr-mult", type=float, default=0.75)
    parser.add_argument("--horizon-bars", type=int, default=20)
    parser.add_argument("--event-window-bars", type=int, default=1)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--tail", type=int, default=200)
    args = parser.parse_args()

    (
        plot_df,
        y_df,
        p_long_oof,
        p_short_oof,
        p_long,
        p_short,
        train_val_idx,
        test_idx,
    ) = _load_inputs(args)
    long_thr, short_thr = _thresholds(args, y_df, p_long_oof, p_short_oof, train_val_idx)
    raw = _raw_signals(
        p_long,
        p_short,
        long_threshold=long_thr,
        short_threshold=short_thr,
        eval_idx=test_idx,
    )
    filtered = _filter_signals(
        raw,
        plot_df,
        cooldown_bars=int(args.cooldown_bars),
        reset_atr_mult=float(args.reset_atr_mult),
        atr_col="atr",
    )
    raw_df = _trade_stats(
        raw,
        plot_df,
        y_df,
        horizon_bars=int(args.horizon_bars),
        atr_col="atr",
        event_window_bars=int(args.event_window_bars),
        slippage_bps=float(args.slippage_bps),
        fee_bps=float(args.fee_bps),
    )
    filtered_df = _trade_stats(
        filtered,
        plot_df,
        y_df,
        horizon_bars=int(args.horizon_bars),
        atr_col="atr",
        event_window_bars=int(args.event_window_bars),
        slippage_bps=float(args.slippage_bps),
        fee_bps=float(args.fee_bps),
    )
    _print_summary(raw_df, "raw_clustered")
    _print_summary(filtered_df, "filtered")

    out_dir = REPO_ROOT / "Data" / "models" / "ga_xgboost" / args.dataset_name / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "swing_signal_filter_raw.csv"
    filtered_path = out_dir / "swing_signal_filter_filtered.csv"
    raw_df.to_csv(raw_path, index=False)
    filtered_df.to_csv(filtered_path, index=False)
    print(f"[filters] wrote {raw_path}")
    print(f"[filters] wrote {filtered_path}")

    long_filtered = np.zeros(len(plot_df), dtype=float)
    short_filtered = np.zeros(len(plot_df), dtype=float)
    for sig in filtered:
        if sig.side == "long":
            long_filtered[sig.idx] = sig.prob
        else:
            short_filtered[sig.idx] = sig.prob
    test_tail_idx = test_idx[-int(args.tail) :] if int(args.tail) > 0 else test_idx
    y_long = pd.to_numeric(y_df["long_swing_label"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    y_short = pd.to_numeric(y_df["short_swing_label"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    plot_path = get_default_model_inference_plot_path(
        args.ticker,
        "ga_xgb_swing_filtered_test",
    )
    plot_model_inference(
        plot_df.iloc[test_tail_idx],
        long_filtered[test_tail_idx],
        short_filtered[test_tail_idx],
        long_actual=y_long[test_tail_idx],
        short_actual=y_short[test_tail_idx],
        long_label_name="LONG",
        short_label_name="SHORT",
        long_threshold=1e-12,
        short_threshold=1e-12,
        title=(
            f"{normalize_ticker(args.ticker)} | GA-XGB swing filtered "
            f"(cooldown={args.cooldown_bars}, reset={args.reset_atr_mult:g} ATR)"
        ),
        save_path=str(plot_path),
    )


if __name__ == "__main__":
    main()
