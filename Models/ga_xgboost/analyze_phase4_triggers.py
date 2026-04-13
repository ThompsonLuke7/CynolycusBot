from __future__ import annotations

import argparse
import json
import shutil
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Data.plots.plots import (
    _apply_time_ticks,
    _compute_marker_offset,
    _compute_time_ticks,
    _draw_day_lines,
    _extract_ohlc,
    _plot_candles,
    _select_plot_window,
)
from Data.retrieve_data import normalize_ticker
from Models.ga_xgboost.swing_label_weights import (
    apply_swing_pivot_zone_weights,
    compute_wilder_atr,
    keep_first_same_side_event,
)
from Models.ga_xgboost.train import _load_split_indices


@dataclass(frozen=True)
class TriggerVariant:
    name: str
    long_col: str | None
    short_col: str | None
    available: bool = True
    reason: str = ""


def _load_phase4_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    clean = normalize_ticker(args.ticker)
    dataset_dir = REPO_ROOT / "Data" / "processed" / clean.lower() / "datasets" / args.dataset_name
    plot_path = dataset_dir / "plot_frame.parquet"
    y_path = dataset_dir / "y.parquet"
    if not plot_path.exists():
        raise FileNotFoundError(plot_path)
    if not y_path.exists():
        raise FileNotFoundError(y_path)

    plot_df = pd.read_parquet(plot_path)
    y_df = pd.read_parquet(y_path)
    if not plot_df.index.equals(y_df.index):
        common = plot_df.index.intersection(y_df.index)
        plot_df = plot_df.loc[common]
        y_df = y_df.loc[common]

    probs_path = (
        REPO_ROOT
        / args.model_root
        / "ga_xgboost"
        / args.dataset_name
        / "single"
        / args.single_label_dir
        / "p_swing_probs.parquet"
    )
    if not probs_path.exists():
        raise FileNotFoundError(probs_path)
    probs_df = pd.read_parquet(probs_path).reindex(plot_df.index)

    required = [
        "p_long_oof_train",
        "p_short_oof_train",
        "p_neutral_oof_train",
        "p_long_test",
        "p_short_test",
        "p_neutral_test",
    ]
    missing = [col for col in required if col not in probs_df.columns]
    if missing:
        raise KeyError(f"Missing probability columns in {probs_path}: {', '.join(missing)}")

    probs = {
        col: pd.to_numeric(probs_df[col], errors="coerce").to_numpy(dtype=np.float32)
        for col in required
    }
    splits = _load_split_indices(
        args.ticker,
        args.dataset_name,
        args.x_filename,
        split_root=Path(args.split_root) if args.split_root else None,
    )
    split_idx = {
        "oof": np.sort(np.concatenate([np.sort(splits["train"]), np.sort(splits["val"])])),
        "test": np.sort(splits["test"]),
    }
    return plot_df, y_df, probs_df, {**probs, **split_idx}


def _load_execution_1m(args: argparse.Namespace, feature_index: pd.Index) -> pd.DataFrame | None:
    if not bool(getattr(args, "use_1m_execution", False)):
        return None
    path = REPO_ROOT / str(args.execution_1m_path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise KeyError(f"1m execution file is missing timestamp column: {path}")
    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"1m execution file is missing columns: {', '.join(missing)}")

    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if isinstance(feature_index, pd.DatetimeIndex) and feature_index.tz is not None:
        ts = ts.dt.tz_convert(feature_index.tz)
    else:
        ts = ts.dt.tz_localize(None)

    out = df.copy()
    out.index = pd.DatetimeIndex(ts)
    out = out.loc[out.index.notna()].sort_index()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if isinstance(feature_index, pd.DatetimeIndex) and len(feature_index):
        start = feature_index.min() - pd.Timedelta(days=1)
        end = feature_index.max() + pd.Timedelta(days=1)
        out = out.loc[(out.index >= start) & (out.index <= end)]

    print(f"[phase4] loaded 1m execution bars: {path} rows={len(out)}")
    return out


def _support_labels(y_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    long_raw = pd.to_numeric(y_df["long_swing_label"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    short_raw = pd.to_numeric(y_df["short_swing_label"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    long_first, short_first, _, _ = keep_first_same_side_event(long_raw, short_raw)
    long_label, _, _ = apply_swing_pivot_zone_weights(
        long_first,
        positive_window_bars=1,
        ambiguous_window_bars=3,
        neighbor_weight=0.75,
        ambiguous_weight=0.0,
    )
    short_label, _, _ = apply_swing_pivot_zone_weights(
        short_first,
        positive_window_bars=1,
        ambiguous_window_bars=3,
        neighbor_weight=0.75,
        ambiguous_weight=0.0,
    )
    return long_label.astype(np.int64), short_label.astype(np.int64)


def _add_phase4_features(plot_df: pd.DataFrame, *, derive_vwap: bool = False) -> pd.DataFrame:
    out = plot_df.copy()
    for col in ("open", "high", "low", "close"):
        if col not in out.columns:
            raise KeyError(f"Missing OHLC column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["break_prev_high"] = out["close"] > out["high"].shift(1)
    out["break_prev_low"] = out["close"] < out["low"].shift(1)
    out["ret_1"] = out["close"].pct_change(1)
    out["mom_up"] = out["ret_1"] > 0
    out["mom_dn"] = out["ret_1"] < 0
    out["bull_body"] = out["close"] > out["open"]
    out["bear_body"] = out["close"] < out["open"]
    out["ema_fast"] = out["close"].ewm(span=3, adjust=False, min_periods=3).mean()
    out["ema_slow"] = out["close"].ewm(span=8, adjust=False, min_periods=8).mean()
    out["fast_above_slow"] = out["ema_fast"] > out["ema_slow"]
    out["fast_below_slow"] = out["ema_fast"] < out["ema_slow"]

    vwap_col = next((col for col in out.columns if col.lower() == "vwap"), None)
    if vwap_col is None and derive_vwap and "volume" in out.columns:
        typical = (out["high"] + out["low"] + out["close"]) / 3.0
        volume = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
        session = out.index.normalize() if isinstance(out.index, pd.DatetimeIndex) else pd.Series(np.zeros(len(out)), index=out.index)
        pv = typical * volume
        cum_pv = pv.groupby(session).cumsum()
        cum_vol = volume.groupby(session).cumsum()
        out["derived_vwap"] = cum_pv / cum_vol.replace(0, np.nan)
        vwap_col = "derived_vwap"

    if vwap_col is not None:
        out["above_vwap"] = out["close"] > pd.to_numeric(out[vwap_col], errors="coerce")
        out["below_vwap"] = out["close"] < pd.to_numeric(out[vwap_col], errors="coerce")
    else:
        out["above_vwap"] = False
        out["below_vwap"] = False

    out["trigger_A_long"] = out["break_prev_high"]
    out["trigger_A_short"] = out["break_prev_low"]
    out["trigger_B_long"] = out["break_prev_high"] & out["mom_up"]
    out["trigger_B_short"] = out["break_prev_low"] & out["mom_dn"]
    out["trigger_C_long"] = out["break_prev_high"] & out["bull_body"]
    out["trigger_C_short"] = out["break_prev_low"] & out["bear_body"]
    out["trigger_D_long"] = out["break_prev_high"] & out["mom_up"] & out["above_vwap"]
    out["trigger_D_short"] = out["break_prev_low"] & out["mom_dn"] & out["below_vwap"]
    out["trigger_E_long"] = out["break_prev_high"] & out["fast_above_slow"]
    out["trigger_E_short"] = out["break_prev_low"] & out["fast_below_slow"]
    out["trigger_F_long"] = out["bull_body"] & out["mom_up"]
    out["trigger_F_short"] = out["bear_body"] & out["mom_dn"]
    out["trigger_G_long"] = out["fast_above_slow"]
    out["trigger_G_short"] = out["fast_below_slow"]
    out["trigger_H_long"] = out["bull_body"] & out["fast_above_slow"]
    out["trigger_H_short"] = out["bear_body"] & out["fast_below_slow"]
    out["trigger_I_long"] = out["break_prev_high"] | out["fast_above_slow"]
    out["trigger_I_short"] = out["break_prev_low"] | out["fast_below_slow"]
    out["trigger_J_long"] = (out["close"] > out["ema_fast"]) & (out["close"].shift(1) <= out["ema_fast"].shift(1))
    out["trigger_J_short"] = (out["close"] < out["ema_fast"]) & (out["close"].shift(1) >= out["ema_fast"].shift(1))
    return out


def _variants(feature_df: pd.DataFrame, *, include_vwap: bool) -> list[TriggerVariant]:
    has_vwap = bool(include_vwap) and (
        ("vwap" in {col.lower() for col in feature_df.columns})
        or "derived_vwap" in feature_df.columns
    )
    return [
        TriggerVariant("raw_threshold", None, None),
        TriggerVariant("trigger_A_break", "trigger_A_long", "trigger_A_short"),
        TriggerVariant("trigger_B_break_momentum", "trigger_B_long", "trigger_B_short"),
        TriggerVariant("trigger_C_break_body", "trigger_C_long", "trigger_C_short"),
        TriggerVariant(
            "trigger_D_break_momentum_vwap",
            "trigger_D_long",
            "trigger_D_short",
            available=has_vwap,
            reason="no vwap column available" if not has_vwap else "",
        ),
        TriggerVariant("trigger_E_break_ema", "trigger_E_long", "trigger_E_short"),
        TriggerVariant("trigger_F_body_momentum", "trigger_F_long", "trigger_F_short"),
        TriggerVariant("trigger_G_ema_slope", "trigger_G_long", "trigger_G_short"),
        TriggerVariant("trigger_H_body_ema", "trigger_H_long", "trigger_H_short"),
        TriggerVariant("trigger_I_break_or_ema", "trigger_I_long", "trigger_I_short"),
        TriggerVariant("trigger_J_close_cross_fast_ema", "trigger_J_long", "trigger_J_short"),
    ]


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        values.append(max(0, int(text)))
    if not values:
        values.append(0)
    return sorted(set(values))


def _expand_recent_setup(setup: np.ndarray, hold_bars: int) -> np.ndarray:
    hold = max(0, int(hold_bars))
    if hold <= 0:
        return setup.astype(bool).copy()
    return (
        pd.Series(setup.astype(bool))
        .rolling(window=hold + 1, min_periods=1)
        .max()
        .fillna(False)
        .to_numpy(dtype=bool)
    )


def _get_vwap_values(feature_df: pd.DataFrame) -> np.ndarray:
    vwap_col = next((col for col in feature_df.columns if col.lower() == "vwap"), None)
    if vwap_col is None and "derived_vwap" in feature_df.columns:
        vwap_col = "derived_vwap"
    if vwap_col is None:
        return np.full(len(feature_df), np.nan, dtype=float)
    return pd.to_numeric(feature_df[vwap_col], errors="coerce").to_numpy(dtype=float)


def _expand_setup_with_invalidation(
    feature_df: pd.DataFrame,
    raw_setup: np.ndarray,
    *,
    side: str,
    hold_bars: int,
    invalidation: str,
    invalidation_atr: float,
) -> tuple[np.ndarray, int]:
    hold = max(0, int(hold_bars))
    if hold <= 0 or invalidation == "none":
        return _expand_recent_setup(raw_setup, hold), 0

    high = pd.to_numeric(feature_df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(feature_df["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(feature_df["close"], errors="coerce").to_numpy(dtype=float)
    atr = compute_wilder_atr(high, low, close, length=14)
    vwap = _get_vwap_values(feature_df)
    use_vwap = invalidation == "price_vwap"
    active = np.zeros(len(feature_df), dtype=bool)
    setup_idxs = np.flatnonzero(raw_setup.astype(bool))
    invalidated = 0

    for setup_idx in setup_idxs:
        end = min(len(feature_df) - 1, setup_idx + hold)
        if end < setup_idx:
            continue
        atr_i = atr[setup_idx]
        buffer = float(invalidation_atr) * atr_i if np.isfinite(atr_i) and atr_i > 0 else 0.0
        setup_high = high[setup_idx]
        setup_low = low[setup_idx]
        for i in range(setup_idx, end + 1):
            invalid = False
            if side == "long":
                if i > setup_idx and np.isfinite(setup_low) and np.isfinite(close[i]) and close[i] < setup_low - buffer:
                    invalid = True
                if (
                    use_vwap
                    and i > setup_idx
                    and np.isfinite(vwap[i])
                    and np.isfinite(close[i])
                    and close[i] < vwap[i] - buffer
                ):
                    invalid = True
            else:
                if i > setup_idx and np.isfinite(setup_high) and np.isfinite(close[i]) and close[i] > setup_high + buffer:
                    invalid = True
                if (
                    use_vwap
                    and i > setup_idx
                    and np.isfinite(vwap[i])
                    and np.isfinite(close[i])
                    and close[i] > vwap[i] + buffer
                ):
                    invalid = True
            if invalid:
                invalidated += 1
                break
            active[i] = True
    return active, invalidated


def _write_signal_frame(
    feature_df: pd.DataFrame,
    loaded: dict[str, np.ndarray],
    *,
    long_setup_threshold: float,
    short_setup_threshold: float,
    out_path: Path,
) -> None:
    signal_cols = [
        "break_prev_high",
        "break_prev_low",
        "ret_1",
        "mom_up",
        "mom_dn",
        "bull_body",
        "bear_body",
        "above_vwap",
        "below_vwap",
        "ema_fast",
        "ema_slow",
        "fast_above_slow",
        "fast_below_slow",
        "trigger_A_long",
        "trigger_A_short",
        "trigger_B_long",
        "trigger_B_short",
        "trigger_C_long",
        "trigger_C_short",
        "trigger_D_long",
        "trigger_D_short",
        "trigger_E_long",
        "trigger_E_short",
        "trigger_F_long",
        "trigger_F_short",
        "trigger_G_long",
        "trigger_G_short",
        "trigger_H_long",
        "trigger_H_short",
        "trigger_I_long",
        "trigger_I_short",
        "trigger_J_long",
        "trigger_J_short",
    ]
    keep_cols = [col for col in ["open", "high", "low", "close", "volume", "vwap", "derived_vwap", *signal_cols] if col in feature_df.columns]
    out = feature_df[keep_cols].copy()
    for col in (
        "p_long_oof_train",
        "p_short_oof_train",
        "p_neutral_oof_train",
        "p_long_test",
        "p_short_test",
        "p_neutral_test",
    ):
        out[col] = loaded[col]

    for split_name, idx in (("oof", loaded["oof"]), ("test", loaded["test"])):
        split_mask = np.zeros(len(feature_df), dtype=bool)
        split_mask[idx] = True
        p_suffix = "oof_train" if split_name == "oof" else "test"
        p_long = loaded[f"p_long_{p_suffix}"]
        p_short = loaded[f"p_short_{p_suffix}"]
        finite = np.isfinite(p_long) & np.isfinite(p_short)
        out[f"is_{split_name}"] = split_mask
        out[f"long_setup_{split_name}"] = split_mask & finite & (p_long >= float(long_setup_threshold))
        out[f"short_setup_{split_name}"] = split_mask & finite & (p_short >= float(short_setup_threshold))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    print(f"[phase4] wrote {out_path}")


def _apply_conflict_cooldown_cluster(
    long_signal: np.ndarray,
    short_signal: np.ndarray,
    long_setup: np.ndarray,
    short_setup: np.ndarray,
    *,
    cooldown_bars: int,
    one_per_setup_cluster: bool,
    session_reset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    long_candidate = long_signal.astype(bool).copy()
    short_candidate = short_signal.astype(bool).copy()
    conflict = long_candidate & short_candidate
    long_candidate[conflict] = False
    short_candidate[conflict] = False

    long_out = np.zeros(long_candidate.shape[0], dtype=bool)
    short_out = np.zeros(short_candidate.shape[0], dtype=bool)
    last_long = -10**9
    last_short = -10**9
    long_cluster_used = False
    short_cluster_used = False
    cooldown_removed_long = 0
    cooldown_removed_short = 0
    cluster_removed_long = 0
    cluster_removed_short = 0
    session_reset = (
        np.zeros(long_candidate.shape[0], dtype=bool)
        if session_reset is None
        else session_reset.astype(bool)
    )

    cooldown = max(0, int(cooldown_bars))
    for i in range(long_candidate.shape[0]):
        if bool(session_reset[i]):
            last_long = -10**9
            last_short = -10**9
            long_cluster_used = False
            short_cluster_used = False

        if not bool(long_setup[i]):
            long_cluster_used = False
        if not bool(short_setup[i]):
            short_cluster_used = False

        if long_candidate[i]:
            if i - last_long <= cooldown:
                cooldown_removed_long += 1
            elif one_per_setup_cluster and long_cluster_used:
                cluster_removed_long += 1
            else:
                long_out[i] = True
                last_long = i
                long_cluster_used = True

        if short_candidate[i]:
            if i - last_short <= cooldown:
                cooldown_removed_short += 1
            elif one_per_setup_cluster and short_cluster_used:
                cluster_removed_short += 1
            else:
                short_out[i] = True
                last_short = i
                short_cluster_used = True

    stats = {
        "conflicts": int(conflict.sum()),
        "cooldown_removed_long": int(cooldown_removed_long),
        "cooldown_removed_short": int(cooldown_removed_short),
        "cluster_removed_long": int(cluster_removed_long),
        "cluster_removed_short": int(cluster_removed_short),
    }
    return long_out, short_out, stats


def _session_reset_mask(index: pd.Index) -> np.ndarray:
    reset = np.zeros(len(index), dtype=bool)
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        sessions = index.normalize().to_numpy()
        reset[1:] = sessions[1:] != sessions[:-1]
    return reset


def _bar_interval_for_close_timestamp(index: pd.Index, i: int) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if not isinstance(index, pd.DatetimeIndex) or i < 0 or i + 1 >= len(index):
        return None, None
    # The intraday bars in this project are timestamped by bar start. A setup
    # scored on bar i is only actionable after that 10-minute bar completes, so
    # the 1-minute execution window for bar i is [index[i], index[i + 1]).
    start = index[i]
    end = index[i + 1]
    if start.normalize() != end.normalize() or end <= start:
        return None, None
    return start, end


def _find_1m_breakout_touch(
    execution_1m: pd.DataFrame | None,
    feature_index: pd.Index,
    i: int,
    *,
    side: str,
    stop_price: float,
    confirmation: str,
) -> tuple[int, float, pd.Timestamp | None] | None:
    if execution_1m is None or not np.isfinite(stop_price):
        return None
    start, end = _bar_interval_for_close_timestamp(feature_index, i)
    if start is None or end is None:
        return None
    minute_index = execution_1m.index
    left = minute_index.searchsorted(start, side="right")
    right = minute_index.searchsorted(end, side="right")
    window = execution_1m.iloc[left:right]
    if window.empty:
        return None

    if side == "long":
        hit = window[pd.to_numeric(window["high"], errors="coerce") >= stop_price]
    else:
        hit = window[pd.to_numeric(window["low"], errors="coerce") <= stop_price]
    if hit.empty:
        return None

    confirmation = str(confirmation).strip().lower()
    hit_open = pd.to_numeric(hit["open"], errors="coerce")
    hit_close = pd.to_numeric(hit["close"], errors="coerce")
    hit_prev_close = pd.to_numeric(window["close"], errors="coerce").shift(1).reindex(hit.index)
    if confirmation == "close_through":
        hit = hit[hit_close > stop_price] if side == "long" else hit[hit_close < stop_price]
    elif confirmation == "body":
        hit = hit[hit_close > hit_open] if side == "long" else hit[hit_close < hit_open]
    elif confirmation == "momentum":
        hit = hit[hit_close > hit_prev_close] if side == "long" else hit[hit_close < hit_prev_close]
    elif confirmation == "body_or_close_through":
        if side == "long":
            hit = hit[(hit_close > hit_open) | (hit_close > stop_price)]
        else:
            hit = hit[(hit_close < hit_open) | (hit_close < stop_price)]
    elif confirmation == "body_and_close_through":
        if side == "long":
            hit = hit[(hit_close > hit_open) & (hit_close > stop_price)]
        else:
            hit = hit[(hit_close < hit_open) & (hit_close < stop_price)]
    elif confirmation in {"touch", "none"}:
        pass
    else:
        raise ValueError(f"Unknown 1m breakout confirmation: {confirmation}")
    if hit.empty:
        return None
    return i, float(stop_price), hit.index[0]


def _find_1m_micro_reversal(
    execution_1m: pd.DataFrame | None,
    feature_index: pd.Index,
    i: int,
    *,
    side: str,
) -> tuple[int, float, pd.Timestamp | None] | None:
    if execution_1m is None:
        return None
    start, end = _bar_interval_for_close_timestamp(feature_index, i)
    if start is None or end is None:
        return None
    minute_index = execution_1m.index
    left = minute_index.searchsorted(start, side="right")
    right = minute_index.searchsorted(end, side="right")
    window = execution_1m.iloc[left:right].copy()
    if window.empty:
        return None

    open_ = pd.to_numeric(window["open"], errors="coerce")
    high = pd.to_numeric(window["high"], errors="coerce")
    low = pd.to_numeric(window["low"], errors="coerce")
    close = pd.to_numeric(window["close"], errors="coerce")
    if side == "long":
        hit = window[(close > open_) & (close > high.shift(1))]
    else:
        hit = window[(close < open_) & (close < low.shift(1))]
    if hit.empty:
        return None
    return i, float(pd.to_numeric(hit["close"], errors="coerce").iloc[0]), hit.index[0]


def _has_1m_adverse_break_before_entry(
    execution_1m: pd.DataFrame | None,
    feature_index: pd.Index,
    setup_idx: int,
    entry_time: pd.Timestamp | None,
    *,
    side: str,
    threshold: float,
) -> bool:
    if execution_1m is None or entry_time is None or pd.isna(entry_time) or not np.isfinite(threshold):
        return False
    if not isinstance(feature_index, pd.DatetimeIndex) or setup_idx + 1 >= len(feature_index):
        return False
    start = feature_index[setup_idx + 1]
    end = pd.Timestamp(entry_time)
    if end <= start:
        return False

    minute_index = execution_1m.index
    left = minute_index.searchsorted(start, side="left")
    right = minute_index.searchsorted(end, side="left")
    window = execution_1m.iloc[left:right]
    if window.empty:
        return False
    if side == "long":
        return bool((pd.to_numeric(window["low"], errors="coerce") < threshold).any())
    return bool((pd.to_numeric(window["high"], errors="coerce") > threshold).any())


def _trade_metrics_for_entries(
    feature_df: pd.DataFrame,
    entries: np.ndarray,
    *,
    side: str,
    eval_idx: np.ndarray,
    horizon_bars: int,
    tp_atr: float,
    sl_atr: float,
    entry_prices: np.ndarray | None = None,
    entry_times: np.ndarray | None = None,
    execution_1m: pd.DataFrame | None = None,
) -> dict[str, float]:
    open_ = pd.to_numeric(feature_df["open"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(feature_df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(feature_df["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(feature_df["close"], errors="coerce").to_numpy(dtype=float)
    atr = compute_wilder_atr(high, low, close, length=14)
    idxs = np.flatnonzero(entries)
    outcomes: list[float] = []
    mfes: list[float] = []
    maes: list[float] = []
    wins = losses = timeouts = 0
    horizon = max(1, int(horizon_bars))
    tp_atr = float(tp_atr)
    sl_atr = float(sl_atr)
    n = len(feature_df)

    for idx in idxs:
        if idx + 1 >= n:
            continue
        entry = entry_prices[idx] if entry_prices is not None else close[idx]
        if not np.isfinite(entry):
            entry = close[idx]
        atr_i = atr[idx]
        if not np.isfinite(entry) or not np.isfinite(atr_i) or atr_i <= 0:
            continue
        use_1m = execution_1m is not None and entry_times is not None and pd.notna(entry_times[idx])
        if use_1m:
            entry_ts = pd.Timestamp(entry_times[idx])
            end_ts = entry_ts + pd.Timedelta(minutes=10 * horizon)
            minute_index = execution_1m.index
            left = minute_index.searchsorted(entry_ts, side="right")
            right = minute_index.searchsorted(end_ts, side="right")
            path = execution_1m.iloc[left:right]
            if path.empty:
                continue
            path_high = pd.to_numeric(path["high"], errors="coerce").to_numpy(dtype=float)
            path_low = pd.to_numeric(path["low"], errors="coerce").to_numpy(dtype=float)
            path_close = pd.to_numeric(path["close"], errors="coerce").to_numpy(dtype=float)
        else:
            end = min(n - 1, idx + horizon)
            if end <= idx:
                continue
            path_high = high[idx + 1 : end + 1]
            path_low = low[idx + 1 : end + 1]
            path_close = close[idx + 1 : end + 1]
            if path_close.size == 0:
                continue

        win = loss = False
        if side == "long":
            tp_px = entry + tp_atr * atr_i
            sl_px = entry - sl_atr * atr_i
            mfe = (np.nanmax(path_high) - entry) / atr_i
            mae = (entry - np.nanmin(path_low)) / atr_i
            for hi, lo in zip(path_high, path_low):
                if lo <= sl_px:
                    loss = True
                    break
                if hi >= tp_px:
                    win = True
                    break
            timeout_ret = (path_close[-1] - entry) / atr_i
        else:
            tp_px = entry - tp_atr * atr_i
            sl_px = entry + sl_atr * atr_i
            mfe = (entry - np.nanmin(path_low)) / atr_i
            mae = (np.nanmax(path_high) - entry) / atr_i
            for hi, lo in zip(path_high, path_low):
                if hi >= sl_px:
                    loss = True
                    break
                if lo <= tp_px:
                    win = True
                    break
            timeout_ret = (entry - path_close[-1]) / atr_i

        if not np.isfinite(mfe) or not np.isfinite(mae):
            continue
        mfes.append(float(mfe))
        maes.append(float(mae))
        if win:
            wins += 1
            outcomes.append(tp_atr)
        elif loss:
            losses += 1
            outcomes.append(-sl_atr)
        else:
            timeouts += 1
            outcomes.append(float(timeout_ret))

    trades = len(outcomes)
    if isinstance(feature_df.index, pd.DatetimeIndex) and eval_idx.size:
        days = max(1, int(pd.Series(feature_df.index[eval_idx].normalize()).nunique()))
    else:
        days = max(1.0, float(eval_idx.size) / 39.0)
    return {
        "trades": float(trades),
        "trades_per_day": float(trades / days),
        "win_rate": float(wins / max(trades, 1)) if trades else float("nan"),
        "ev_atr": float(np.nanmean(outcomes)) if outcomes else float("nan"),
        "avg_mfe_atr": float(np.nanmean(mfes)) if mfes else float("nan"),
        "avg_mae_atr": float(np.nanmean(maes)) if maes else float("nan"),
        "wins": float(wins),
        "losses": float(losses),
        "timeouts": float(timeouts),
    }


def _trace_trades_for_entries(
    feature_df: pd.DataFrame,
    entries: np.ndarray,
    *,
    side: str,
    split_name: str,
    variant: str,
    mode: str,
    p_long: np.ndarray,
    p_short: np.ndarray,
    setup: np.ndarray,
    trigger: np.ndarray | None,
    horizon_bars: int,
    tp_atr: float,
    sl_atr: float,
    entry_prices: np.ndarray | None = None,
    entry_times: np.ndarray | None = None,
    execution_1m: pd.DataFrame | None = None,
) -> pd.DataFrame:
    open_ = pd.to_numeric(feature_df["open"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(feature_df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(feature_df["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(feature_df["close"], errors="coerce").to_numpy(dtype=float)
    atr = compute_wilder_atr(high, low, close, length=14)
    idxs = np.flatnonzero(entries)
    rows: list[dict[str, float | str | bool]] = []
    horizon = max(1, int(horizon_bars))
    tp_atr = float(tp_atr)
    sl_atr = float(sl_atr)
    n = len(feature_df)

    for trace_n, idx in enumerate(idxs, start=1):
        if idx + 1 >= n:
            continue
        entry = entry_prices[idx] if entry_prices is not None else close[idx]
        if not np.isfinite(entry):
            entry = close[idx]
        atr_i = atr[idx]
        if not np.isfinite(entry) or not np.isfinite(atr_i) or atr_i <= 0:
            continue
        setup_ts = feature_df.index[idx] if isinstance(feature_df.index, pd.DatetimeIndex) else idx
        entry_ts = entry_times[idx] if entry_times is not None else pd.NaT
        if pd.isna(entry_ts):
            entry_ts = setup_ts

        use_1m = execution_1m is not None and pd.notna(entry_ts)
        if use_1m:
            entry_ts = pd.Timestamp(entry_ts)
            end_ts = entry_ts + pd.Timedelta(minutes=10 * horizon)
            minute_index = execution_1m.index
            left = minute_index.searchsorted(entry_ts, side="right")
            right = minute_index.searchsorted(end_ts, side="right")
            path = execution_1m.iloc[left:right]
            if path.empty:
                continue
            path_high = pd.to_numeric(path["high"], errors="coerce").to_numpy(dtype=float)
            path_low = pd.to_numeric(path["low"], errors="coerce").to_numpy(dtype=float)
            path_close = pd.to_numeric(path["close"], errors="coerce").to_numpy(dtype=float)
            path_times = list(path.index)
        else:
            end = min(n - 1, idx + horizon)
            if end <= idx:
                continue
            path_high = high[idx + 1 : end + 1]
            path_low = low[idx + 1 : end + 1]
            path_close = close[idx + 1 : end + 1]
            if path_close.size == 0:
                continue
            if isinstance(feature_df.index, pd.DatetimeIndex):
                path_times = list(feature_df.index[idx + 1 : end + 1])
            else:
                path_times = list(range(idx + 1, end + 1))

        if side == "long":
            tp_px = entry + tp_atr * atr_i
            sl_px = entry - sl_atr * atr_i
            mfe = (np.nanmax(path_high) - entry) / atr_i
            mae = (entry - np.nanmin(path_low)) / atr_i
            timeout_ret = (path_close[-1] - entry) / atr_i
            hit_reason = "timeout"
            outcome_atr = float(timeout_ret)
            exit_time = path_times[-1]
            for t, hi, lo in zip(path_times, path_high, path_low):
                if lo <= sl_px:
                    hit_reason = "sl"
                    outcome_atr = -sl_atr
                    exit_time = t
                    break
                if hi >= tp_px:
                    hit_reason = "tp"
                    outcome_atr = tp_atr
                    exit_time = t
                    break
        else:
            tp_px = entry - tp_atr * atr_i
            sl_px = entry + sl_atr * atr_i
            mfe = (entry - np.nanmin(path_low)) / atr_i
            mae = (np.nanmax(path_high) - entry) / atr_i
            timeout_ret = (entry - path_close[-1]) / atr_i
            hit_reason = "timeout"
            outcome_atr = float(timeout_ret)
            exit_time = path_times[-1]
            for t, hi, lo in zip(path_times, path_high, path_low):
                if hi >= sl_px:
                    hit_reason = "sl"
                    outcome_atr = -sl_atr
                    exit_time = t
                    break
                if lo <= tp_px:
                    hit_reason = "tp"
                    outcome_atr = tp_atr
                    exit_time = t
                    break

        if not np.isfinite(mfe) or not np.isfinite(mae):
            continue
        rows.append(
            {
                "trace_id": f"{side.upper()}_{trace_n:04d}",
                "split": split_name,
                "variant": variant,
                "mode": mode,
                "side": side,
                "bar_index": float(idx),
                "setup_bar_time": str(setup_ts),
                "entry_time": str(entry_ts),
                "exit_time": str(exit_time),
                "entry_price": float(entry),
                "tp_price": float(tp_px),
                "sl_price": float(sl_px),
                "atr": float(atr_i),
                "outcome": hit_reason,
                "outcome_atr": float(outcome_atr),
                "mfe_atr": float(mfe),
                "mae_atr": float(mae),
                "p_long": float(p_long[idx]) if np.isfinite(p_long[idx]) else float("nan"),
                "p_short": float(p_short[idx]) if np.isfinite(p_short[idx]) else float("nan"),
                "setup_active": bool(setup[idx]),
                "trigger_candidate": bool(trigger[idx]) if trigger is not None else bool(entries[idx]),
                "bar_open": float(open_[idx]),
                "bar_high": float(high[idx]),
                "bar_low": float(low[idx]),
                "bar_close": float(close[idx]),
            }
        )
    return pd.DataFrame(rows)


def _merge_side_metrics(
    *,
    split_name: str,
    variant: str,
    mode: str,
    available: bool,
    reason: str,
    long_setup_threshold: float,
    short_setup_threshold: float,
    cooldown_bars: int,
    one_per_setup_cluster: bool,
    setup_invalidation: str,
    setup_invalidation_atr: float,
    long_setup_count: int,
    short_setup_count: int,
    long_raw_setup_count: int,
    short_raw_setup_count: int,
    long_trigger_count: int,
    short_trigger_count: int,
    long_entries_count: int,
    short_entries_count: int,
    removal_stats: dict[str, int],
    long_metrics: dict[str, float],
    short_metrics: dict[str, float],
    long_setup_hold_bars: int = 0,
    short_setup_hold_bars: int = 0,
    long_invalidated: int = 0,
    short_invalidated: int = 0,
    post_setup_max_bars: int = 0,
) -> dict[str, float | str | bool]:
    total_trades = long_metrics["trades"] + short_metrics["trades"]
    total_ev = np.nan
    if total_trades:
        total_ev = (
            long_metrics["ev_atr"] * long_metrics["trades"]
            + short_metrics["ev_atr"] * short_metrics["trades"]
        ) / total_trades

    row: dict[str, float | str | bool] = {
        "split": split_name,
        "variant": variant,
        "mode": mode,
        "available": available,
        "reason": reason,
        "long_setup_threshold": float(long_setup_threshold),
        "short_setup_threshold": float(short_setup_threshold),
        "cooldown_bars": float(cooldown_bars),
        "one_per_setup_cluster": bool(one_per_setup_cluster),
        "setup_hold_bars": float(long_setup_hold_bars) if long_setup_hold_bars == short_setup_hold_bars else float("nan"),
        "long_setup_hold_bars": float(long_setup_hold_bars),
        "short_setup_hold_bars": float(short_setup_hold_bars),
        "post_setup_max_bars": float(post_setup_max_bars),
        "setup_invalidation": setup_invalidation,
        "setup_invalidation_atr": float(setup_invalidation_atr),
        "long_setup_invalidated": float(long_invalidated),
        "short_setup_invalidated": float(short_invalidated),
        "long_raw_setup_bars": float(long_raw_setup_count),
        "short_raw_setup_bars": float(short_raw_setup_count),
        "long_setup_bars": float(long_setup_count),
        "short_setup_bars": float(short_setup_count),
        "long_triggered_candidates": float(long_trigger_count),
        "short_triggered_candidates": float(short_trigger_count),
        "long_trigger_conversion_rate": float(long_trigger_count / max(long_setup_count, 1)),
        "short_trigger_conversion_rate": float(short_trigger_count / max(short_setup_count, 1)),
        "long_entries": float(long_entries_count),
        "short_entries": float(short_entries_count),
        "conflicts": float(removal_stats["conflicts"]),
        "cooldown_removed_long": float(removal_stats["cooldown_removed_long"]),
        "cooldown_removed_short": float(removal_stats["cooldown_removed_short"]),
        "cluster_removed_long": float(removal_stats["cluster_removed_long"]),
        "cluster_removed_short": float(removal_stats["cluster_removed_short"]),
        "total_trades": float(total_trades),
        "total_ev_atr": float(total_ev),
        "total_trades_per_day": float(long_metrics["trades_per_day"] + short_metrics["trades_per_day"]),
    }
    for prefix, metrics in (("long", long_metrics), ("short", short_metrics)):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    return row


def _post_setup_side_candidates(
    feature_df: pd.DataFrame,
    raw_setup: np.ndarray,
    eval_mask: np.ndarray,
    *,
    side: str,
    policy: str,
    trigger_col: str | None,
    max_bars: int,
    execution_1m: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    open_ = pd.to_numeric(feature_df["open"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(feature_df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(feature_df["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(feature_df["close"], errors="coerce").to_numpy(dtype=float)
    trigger = (
        feature_df[trigger_col].fillna(False).to_numpy(dtype=bool)
        if trigger_col is not None
        else np.zeros(len(feature_df), dtype=bool)
    )
    entries = np.zeros(len(feature_df), dtype=bool)
    entry_prices = np.full(len(feature_df), np.nan, dtype=float)
    entry_times = np.full(len(feature_df), pd.NaT, dtype=object)
    setup_idxs = np.flatnonzero(raw_setup.astype(bool))
    triggered = 0
    max_bars = max(1, int(max_bars))
    n = len(feature_df)
    sessions = feature_df.index.normalize().to_numpy() if isinstance(feature_df.index, pd.DatetimeIndex) else None

    for setup_idx in setup_idxs:
        start = setup_idx + 1
        end = min(n - 1, setup_idx + max_bars)
        if sessions is not None:
            setup_session = sessions[setup_idx]
            while end >= start and sessions[end] != setup_session:
                end -= 1
        if start > end:
            continue
        entry_idx = -1
        entry_price = np.nan
        entry_time: pd.Timestamp | None = None
        if policy == "next_open":
            entry_idx = start
            entry_price = open_[entry_idx]
            entry_time = feature_df.index[entry_idx] if isinstance(feature_df.index, pd.DatetimeIndex) else None
        elif policy == "next_open_direction":
            if bool(eval_mask[start]) and np.isfinite(open_[start]) and np.isfinite(close[setup_idx]):
                if side == "long" and open_[start] > close[setup_idx]:
                    entry_idx = start
                    entry_price = open_[entry_idx]
                    entry_time = feature_df.index[entry_idx] if isinstance(feature_df.index, pd.DatetimeIndex) else None
                if side == "short" and open_[start] < close[setup_idx]:
                    entry_idx = start
                    entry_price = open_[entry_idx]
                    entry_time = feature_df.index[entry_idx] if isinstance(feature_df.index, pd.DatetimeIndex) else None
        elif policy in {
            "break_prev_stop",
            "break_prev_stop_1m_confirm",
            "break_prev_stop_1m_body",
            "break_prev_stop_1m_momentum",
            "break_prev_stop_1m_body_or_close",
            "break_prev_stop_1m_body_and_close",
            "break_prev_stop_1m_confirm_no_fresh_low",
            "break_prev_stop_1m_body_and_close_no_fresh_low",
        }:
            confirmation = {
                "break_prev_stop": "touch",
                "break_prev_stop_1m_confirm": "close_through",
                "break_prev_stop_1m_body": "body",
                "break_prev_stop_1m_momentum": "momentum",
                "break_prev_stop_1m_body_or_close": "body_or_close_through",
                "break_prev_stop_1m_body_and_close": "body_and_close_through",
                "break_prev_stop_1m_confirm_no_fresh_low": "close_through",
                "break_prev_stop_1m_body_and_close_no_fresh_low": "body_and_close_through",
            }[policy]
            no_fresh_adverse = policy.endswith("_no_fresh_low")
            for i in range(start, end + 1):
                if not bool(eval_mask[i]):
                    continue
                if side == "long" and np.isfinite(high[i - 1]):
                    touch = _find_1m_breakout_touch(
                        execution_1m,
                        feature_df.index,
                        i,
                        side=side,
                        stop_price=high[i - 1],
                        confirmation=confirmation,
                    )
                    if touch is not None:
                        entry_idx, entry_price, entry_time = touch
                        if no_fresh_adverse and _has_1m_adverse_break_before_entry(
                            execution_1m,
                            feature_df.index,
                            setup_idx,
                            entry_time,
                            side=side,
                            threshold=low[setup_idx],
                        ):
                            continue
                        break
                    if execution_1m is not None:
                        continue
                if side == "long" and np.isfinite(high[i - 1]) and high[i] >= high[i - 1]:
                    if confirmation == "close_through" and not (np.isfinite(close[i]) and close[i] > high[i - 1]):
                        continue
                    if confirmation == "body" and not (np.isfinite(close[i]) and np.isfinite(open_[i]) and close[i] > open_[i]):
                        continue
                    if confirmation == "body_and_close_through" and not (
                        np.isfinite(close[i]) and np.isfinite(open_[i]) and close[i] > open_[i] and close[i] > high[i - 1]
                    ):
                        continue
                    entry_idx = i
                    entry_price = high[i - 1]
                    if no_fresh_adverse and i > start and np.nanmin(low[start:i]) < low[setup_idx]:
                        entry_idx = -1
                        entry_price = np.nan
                        continue
                    break
                if side == "short" and np.isfinite(low[i - 1]):
                    touch = _find_1m_breakout_touch(
                        execution_1m,
                        feature_df.index,
                        i,
                        side=side,
                        stop_price=low[i - 1],
                        confirmation=confirmation,
                    )
                    if touch is not None:
                        entry_idx, entry_price, entry_time = touch
                        break
                    if execution_1m is not None:
                        continue
                if side == "short" and np.isfinite(low[i - 1]) and low[i] <= low[i - 1]:
                    if confirmation == "close_through" and not (np.isfinite(close[i]) and close[i] < low[i - 1]):
                        continue
                    if confirmation == "body" and not (np.isfinite(close[i]) and np.isfinite(open_[i]) and close[i] < open_[i]):
                        continue
                    if confirmation == "body_and_close_through" and not (
                        np.isfinite(close[i]) and np.isfinite(open_[i]) and close[i] < open_[i] and close[i] < low[i - 1]
                    ):
                        continue
                    entry_idx = i
                    entry_price = low[i - 1]
                    break
        elif policy == "micro_reversal_1m":
            for i in range(start, end + 1):
                if not bool(eval_mask[i]):
                    continue
                touch = _find_1m_micro_reversal(
                    execution_1m,
                    feature_df.index,
                    i,
                    side=side,
                )
                if touch is not None:
                    entry_idx, entry_price, entry_time = touch
                    break
        elif policy == "trigger_close":
            for i in range(start, end + 1):
                if bool(eval_mask[i]) and bool(trigger[i]):
                    entry_idx = i
                    entry_price = close[i]
                    entry_time = feature_df.index[i] if isinstance(feature_df.index, pd.DatetimeIndex) else None
                    break
        else:
            raise ValueError(f"Unknown post-setup policy: {policy}")

        if entry_idx < 0 or not bool(eval_mask[entry_idx]) or not np.isfinite(entry_price):
            continue
        if entry_time is None and isinstance(feature_df.index, pd.DatetimeIndex):
            entry_time = feature_df.index[entry_idx]
        entries[entry_idx] = True
        if not np.isfinite(entry_prices[entry_idx]):
            entry_prices[entry_idx] = entry_price
        else:
            entry_prices[entry_idx] = min(entry_prices[entry_idx], entry_price) if side == "long" else max(entry_prices[entry_idx], entry_price)
        if entry_time is not None:
            entry_times[entry_idx] = entry_time
        triggered += 1
    return entries, entry_prices, entry_times, int(len(setup_idxs)), int(triggered)


def _evaluate_post_setup_policy(
    feature_df: pd.DataFrame,
    p_long: np.ndarray,
    p_short: np.ndarray,
    *,
    split_name: str,
    eval_idx: np.ndarray,
    long_setup_threshold: float,
    short_setup_threshold: float,
    cooldown_bars: int,
    one_per_setup_cluster: bool,
    policy_name: str,
    long_trigger_col: str | None,
    short_trigger_col: str | None,
    max_bars: int,
    horizon_bars: int,
    tp_atr: float,
    sl_atr: float,
    execution_1m: pd.DataFrame | None = None,
) -> tuple[
    dict[str, float | str | bool],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    eval_mask = np.zeros(len(feature_df), dtype=bool)
    eval_mask[eval_idx] = True
    finite = np.isfinite(p_long) & np.isfinite(p_short)
    long_raw_setup = eval_mask & finite & (p_long >= float(long_setup_threshold))
    short_raw_setup = eval_mask & finite & (p_short >= float(short_setup_threshold))
    policy = (
        _post_setup_policy_to_internal(policy_name)
    )

    long_candidate, long_prices, long_times, long_setup_count, long_trigger_count = _post_setup_side_candidates(
        feature_df,
        long_raw_setup,
        eval_mask,
        side="long",
        policy=policy,
        trigger_col=long_trigger_col,
        max_bars=max_bars,
        execution_1m=execution_1m,
    )
    short_candidate, short_prices, short_times, short_setup_count, short_trigger_count = _post_setup_side_candidates(
        feature_df,
        short_raw_setup,
        eval_mask,
        side="short",
        policy=policy,
        trigger_col=short_trigger_col,
        max_bars=max_bars,
        execution_1m=execution_1m,
    )
    active_long = _expand_recent_setup(long_raw_setup, max_bars) & eval_mask
    active_short = _expand_recent_setup(short_raw_setup, max_bars) & eval_mask
    long_entries, short_entries, removal_stats = _apply_conflict_cooldown_cluster(
        long_candidate,
        short_candidate,
        active_long,
        active_short,
        cooldown_bars=cooldown_bars,
        one_per_setup_cluster=one_per_setup_cluster,
        session_reset=_session_reset_mask(feature_df.index),
    )
    long_metrics = _trade_metrics_for_entries(
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
    short_metrics = _trade_metrics_for_entries(
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
    mode = f"{'cooldown_cluster' if one_per_setup_cluster else 'cooldown_only'}_max{max_bars}"
    return (
        _merge_side_metrics(
            split_name=split_name,
            variant=policy_name,
            mode=mode,
            available=True,
            reason="",
            long_setup_threshold=long_setup_threshold,
            short_setup_threshold=short_setup_threshold,
            cooldown_bars=cooldown_bars,
            one_per_setup_cluster=one_per_setup_cluster,
            setup_invalidation="post_setup",
            setup_invalidation_atr=0.0,
            long_setup_count=long_setup_count,
            short_setup_count=short_setup_count,
            long_raw_setup_count=int(long_raw_setup.sum()),
            short_raw_setup_count=int(short_raw_setup.sum()),
            long_trigger_count=long_trigger_count,
            short_trigger_count=short_trigger_count,
            long_entries_count=int(long_entries.sum()),
            short_entries_count=int(short_entries.sum()),
            removal_stats=removal_stats,
            long_metrics=long_metrics,
            short_metrics=short_metrics,
            post_setup_max_bars=max_bars,
        ),
        long_entries,
        short_entries,
        long_candidate,
        short_candidate,
        long_prices,
        short_prices,
        long_times,
        short_times,
    )


def _post_setup_policy_to_internal(policy_name: str) -> str:
    return {
        "post_setup_next_open": "next_open",
        "post_setup_break_prev_stop": "break_prev_stop",
        "post_setup_break_prev_stop_1m_confirm": "break_prev_stop_1m_confirm",
        "post_setup_break_prev_stop_1m_body": "break_prev_stop_1m_body",
        "post_setup_break_prev_stop_1m_momentum": "break_prev_stop_1m_momentum",
        "post_setup_break_prev_stop_1m_body_or_close": "break_prev_stop_1m_body_or_close",
        "post_setup_break_prev_stop_1m_body_and_close": "break_prev_stop_1m_body_and_close",
        "post_setup_break_prev_stop_1m_confirm_no_fresh_low": "break_prev_stop_1m_confirm_no_fresh_low",
        "post_setup_break_prev_stop_1m_body_and_close_no_fresh_low": "break_prev_stop_1m_body_and_close_no_fresh_low",
        "post_setup_micro_reversal_1m": "micro_reversal_1m",
    }.get(policy_name, "trigger_close")


def _evaluate_asymmetric_policy(
    feature_df: pd.DataFrame,
    p_long: np.ndarray,
    p_short: np.ndarray,
    *,
    split_name: str,
    eval_idx: np.ndarray,
    long_setup_threshold: float,
    short_setup_threshold: float,
    cooldown_bars: int,
    one_per_setup_cluster: bool,
    long_policy_name: str,
    long_policy: str,
    long_trigger_col: str | None,
    short_policy_name: str,
    short_policy: str,
    short_trigger_col: str | None,
    max_bars: int,
    horizon_bars: int,
    tp_atr: float,
    sl_atr: float,
    execution_1m: pd.DataFrame | None = None,
) -> tuple[
    dict[str, float | str | bool],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    eval_mask = np.zeros(len(feature_df), dtype=bool)
    eval_mask[eval_idx] = True
    finite = np.isfinite(p_long) & np.isfinite(p_short)
    long_raw_setup = eval_mask & finite & (p_long >= float(long_setup_threshold))
    short_raw_setup = eval_mask & finite & (p_short >= float(short_setup_threshold))

    long_candidate, long_prices, long_times, long_setup_count, long_trigger_count = _post_setup_side_candidates(
        feature_df,
        long_raw_setup,
        eval_mask,
        side="long",
        policy=long_policy,
        trigger_col=long_trigger_col,
        max_bars=max_bars,
        execution_1m=execution_1m,
    )
    short_candidate, short_prices, short_times, short_setup_count, short_trigger_count = _post_setup_side_candidates(
        feature_df,
        short_raw_setup,
        eval_mask,
        side="short",
        policy=short_policy,
        trigger_col=short_trigger_col,
        max_bars=max_bars,
        execution_1m=execution_1m,
    )

    active_long = _expand_recent_setup(long_raw_setup, max_bars) & eval_mask
    active_short = _expand_recent_setup(short_raw_setup, max_bars) & eval_mask
    long_entries, short_entries, removal_stats = _apply_conflict_cooldown_cluster(
        long_candidate,
        short_candidate,
        active_long,
        active_short,
        cooldown_bars=cooldown_bars,
        one_per_setup_cluster=one_per_setup_cluster,
        session_reset=_session_reset_mask(feature_df.index),
    )
    long_metrics = _trade_metrics_for_entries(
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
    short_metrics = _trade_metrics_for_entries(
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
    mode = f"{'cooldown_cluster' if one_per_setup_cluster else 'cooldown_only'}_longmax{max_bars}_shortmax{max_bars}"
    return (
        _merge_side_metrics(
            split_name=split_name,
            variant=f"asym_long_{long_policy_name}_short_{short_policy_name}",
            mode=mode,
            available=True,
            reason="asymmetric post-setup confirmation",
            long_setup_threshold=long_setup_threshold,
            short_setup_threshold=short_setup_threshold,
            cooldown_bars=cooldown_bars,
            one_per_setup_cluster=one_per_setup_cluster,
            setup_invalidation="asym_post_setup",
            setup_invalidation_atr=0.0,
            long_setup_count=int(active_long.sum()),
            short_setup_count=short_setup_count,
            long_raw_setup_count=int(long_raw_setup.sum()),
            short_raw_setup_count=int(short_raw_setup.sum()),
            long_trigger_count=long_trigger_count,
            short_trigger_count=short_trigger_count,
            long_entries_count=int(long_entries.sum()),
            short_entries_count=int(short_entries.sum()),
            removal_stats=removal_stats,
            long_metrics=long_metrics,
            short_metrics=short_metrics,
            post_setup_max_bars=max_bars,
        ),
        long_entries,
        short_entries,
        long_candidate,
        short_candidate,
        long_prices,
        short_prices,
        long_times,
        short_times,
    )


def _evaluate_sr_breakout_policy(
    feature_df: pd.DataFrame,
    *,
    split_name: str,
    eval_idx: np.ndarray,
    lookback_bars: int,
    cooldown_bars: int,
    one_per_setup_cluster: bool,
    horizon_bars: int,
    tp_atr: float,
    sl_atr: float,
    execution_1m: pd.DataFrame | None = None,
) -> tuple[
    dict[str, float | str | bool],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    high = pd.to_numeric(feature_df["high"], errors="coerce")
    low = pd.to_numeric(feature_df["low"], errors="coerce")
    lookback = max(2, int(lookback_bars))
    resistance = high.shift(1).rolling(lookback, min_periods=lookback).max().to_numpy(dtype=float)
    support = low.shift(1).rolling(lookback, min_periods=lookback).min().to_numpy(dtype=float)

    eval_mask = np.zeros(len(feature_df), dtype=bool)
    eval_mask[eval_idx] = True
    long_candidate = np.zeros(len(feature_df), dtype=bool)
    short_candidate = np.zeros(len(feature_df), dtype=bool)
    long_prices = np.full(len(feature_df), np.nan, dtype=float)
    short_prices = np.full(len(feature_df), np.nan, dtype=float)
    long_times = np.full(len(feature_df), pd.NaT, dtype=object)
    short_times = np.full(len(feature_df), pd.NaT, dtype=object)

    for i in np.flatnonzero(eval_mask):
        if i <= 0:
            continue
        if np.isfinite(resistance[i]):
            touch = _find_1m_breakout_touch(
                execution_1m,
                feature_df.index,
                i,
                side="long",
                stop_price=resistance[i],
                confirmation="touch",
            )
            if touch is not None:
                entry_idx, entry_price, entry_time = touch
                long_candidate[entry_idx] = True
                long_prices[entry_idx] = entry_price
                long_times[entry_idx] = entry_time
            elif execution_1m is None and high.iloc[i] >= resistance[i]:
                long_candidate[i] = True
                long_prices[i] = resistance[i]
                long_times[i] = feature_df.index[i] if isinstance(feature_df.index, pd.DatetimeIndex) else pd.NaT
        if np.isfinite(support[i]):
            touch = _find_1m_breakout_touch(
                execution_1m,
                feature_df.index,
                i,
                side="short",
                stop_price=support[i],
                confirmation="touch",
            )
            if touch is not None:
                entry_idx, entry_price, entry_time = touch
                short_candidate[entry_idx] = True
                short_prices[entry_idx] = entry_price
                short_times[entry_idx] = entry_time
            elif execution_1m is None and low.iloc[i] <= support[i]:
                short_candidate[i] = True
                short_prices[i] = support[i]
                short_times[i] = feature_df.index[i] if isinstance(feature_df.index, pd.DatetimeIndex) else pd.NaT

    long_entries, short_entries, removal_stats = _apply_conflict_cooldown_cluster(
        long_candidate,
        short_candidate,
        long_candidate,
        short_candidate,
        cooldown_bars=cooldown_bars,
        one_per_setup_cluster=one_per_setup_cluster,
        session_reset=_session_reset_mask(feature_df.index),
    )
    long_metrics = _trade_metrics_for_entries(
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
    short_metrics = _trade_metrics_for_entries(
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
    mode = f"{'cooldown_cluster' if one_per_setup_cluster else 'cooldown_only'}_lookback{lookback}"
    return (
        _merge_side_metrics(
            split_name=split_name,
            variant=f"sr_breakout_{lookback}",
            mode=mode,
            available=True,
            reason="rolling support/resistance breakout baseline",
            long_setup_threshold=float("nan"),
            short_setup_threshold=float("nan"),
            cooldown_bars=cooldown_bars,
            one_per_setup_cluster=one_per_setup_cluster,
            setup_invalidation="sr_breakout",
            setup_invalidation_atr=0.0,
            long_setup_count=int(long_candidate.sum()),
            short_setup_count=int(short_candidate.sum()),
            long_raw_setup_count=int(long_candidate.sum()),
            short_raw_setup_count=int(short_candidate.sum()),
            long_trigger_count=int(long_candidate.sum()),
            short_trigger_count=int(short_candidate.sum()),
            long_entries_count=int(long_entries.sum()),
            short_entries_count=int(short_entries.sum()),
            removal_stats=removal_stats,
            long_metrics=long_metrics,
            short_metrics=short_metrics,
        ),
        long_entries,
        short_entries,
        long_candidate,
        short_candidate,
        long_prices,
        short_prices,
        long_times,
        short_times,
    )


def _evaluate_variant(
    feature_df: pd.DataFrame,
    p_long: np.ndarray,
    p_short: np.ndarray,
    *,
    variant: TriggerVariant,
    split_name: str,
    eval_idx: np.ndarray,
    long_setup_threshold: float,
    short_setup_threshold: float,
    cooldown_bars: int,
    one_per_setup_cluster: bool,
    long_setup_hold_bars: int,
    short_setup_hold_bars: int,
    setup_invalidation: str,
    setup_invalidation_atr: float,
    horizon_bars: int,
    tp_atr: float,
    sl_atr: float,
) -> tuple[
    dict[str, float | str | bool],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n = len(feature_df)
    eval_mask = np.zeros(n, dtype=bool)
    eval_mask[eval_idx] = True
    finite = np.isfinite(p_long) & np.isfinite(p_short)
    long_raw_setup = eval_mask & finite & (p_long >= long_setup_threshold)
    short_raw_setup = eval_mask & finite & (p_short >= short_setup_threshold)
    long_setup, long_invalidated = _expand_setup_with_invalidation(
        feature_df,
        long_raw_setup,
        side="long",
        hold_bars=long_setup_hold_bars,
        invalidation=setup_invalidation,
        invalidation_atr=setup_invalidation_atr,
    )
    short_setup, short_invalidated = _expand_setup_with_invalidation(
        feature_df,
        short_raw_setup,
        side="short",
        hold_bars=short_setup_hold_bars,
        invalidation=setup_invalidation,
        invalidation_atr=setup_invalidation_atr,
    )
    long_setup &= eval_mask
    short_setup &= eval_mask

    if variant.long_col is None:
        long_setup = long_raw_setup
        short_setup = short_raw_setup
        long_setup_hold_bars = 0
        short_setup_hold_bars = 0
        long_invalidated = 0
        short_invalidated = 0
        trigger_long = eval_mask.copy()
        trigger_short = eval_mask.copy()
    else:
        trigger_long = feature_df[variant.long_col].fillna(False).to_numpy(dtype=bool)
        trigger_short = feature_df[variant.short_col].fillna(False).to_numpy(dtype=bool)

    long_candidate = long_setup & trigger_long
    short_candidate = short_setup & trigger_short
    long_prices = np.full(n, np.nan, dtype=float)
    short_prices = np.full(n, np.nan, dtype=float)
    long_times = np.full(n, pd.NaT, dtype=object)
    short_times = np.full(n, pd.NaT, dtype=object)
    long_entries, short_entries, removal_stats = _apply_conflict_cooldown_cluster(
        long_candidate,
        short_candidate,
        long_setup,
        short_setup,
        cooldown_bars=cooldown_bars,
        one_per_setup_cluster=one_per_setup_cluster,
        session_reset=_session_reset_mask(feature_df.index),
    )
    long_metrics = _trade_metrics_for_entries(
        feature_df,
        long_entries,
        side="long",
        eval_idx=eval_idx,
        horizon_bars=horizon_bars,
        tp_atr=tp_atr,
        sl_atr=sl_atr,
    )
    short_metrics = _trade_metrics_for_entries(
        feature_df,
        short_entries,
        side="short",
        eval_idx=eval_idx,
        horizon_bars=horizon_bars,
        tp_atr=tp_atr,
        sl_atr=sl_atr,
    )

    long_setup_count = int(long_setup.sum())
    short_setup_count = int(short_setup.sum())
    long_trigger_count = int(long_candidate.sum())
    short_trigger_count = int(short_candidate.sum())
    long_entries_count = int(long_entries.sum())
    short_entries_count = int(short_entries.sum())
    total_trades = long_metrics["trades"] + short_metrics["trades"]
    total_ev = np.nan
    if total_trades:
        total_ev = (
            long_metrics["ev_atr"] * long_metrics["trades"]
            + short_metrics["ev_atr"] * short_metrics["trades"]
        ) / total_trades

    row: dict[str, float | str | bool] = {
        "split": split_name,
        "variant": variant.name,
        "mode": "cooldown_cluster" if one_per_setup_cluster else "cooldown_only",
        "available": variant.available,
        "reason": variant.reason,
        "long_setup_threshold": float(long_setup_threshold),
        "short_setup_threshold": float(short_setup_threshold),
        "cooldown_bars": float(cooldown_bars),
        "one_per_setup_cluster": bool(one_per_setup_cluster),
        "setup_hold_bars": float(long_setup_hold_bars) if long_setup_hold_bars == short_setup_hold_bars else float("nan"),
        "long_setup_hold_bars": float(long_setup_hold_bars),
        "short_setup_hold_bars": float(short_setup_hold_bars),
        "setup_invalidation": setup_invalidation,
        "setup_invalidation_atr": float(setup_invalidation_atr),
        "long_setup_invalidated": float(long_invalidated),
        "short_setup_invalidated": float(short_invalidated),
        "long_raw_setup_bars": float(long_raw_setup.sum()),
        "short_raw_setup_bars": float(short_raw_setup.sum()),
        "long_setup_bars": float(long_setup_count),
        "short_setup_bars": float(short_setup_count),
        "long_triggered_candidates": float(long_trigger_count),
        "short_triggered_candidates": float(short_trigger_count),
        "long_trigger_conversion_rate": float(long_trigger_count / max(long_setup_count, 1)),
        "short_trigger_conversion_rate": float(short_trigger_count / max(short_setup_count, 1)),
        "long_entries": float(long_entries_count),
        "short_entries": float(short_entries_count),
        "conflicts": float(removal_stats["conflicts"]),
        "cooldown_removed_long": float(removal_stats["cooldown_removed_long"]),
        "cooldown_removed_short": float(removal_stats["cooldown_removed_short"]),
        "cluster_removed_long": float(removal_stats["cluster_removed_long"]),
        "cluster_removed_short": float(removal_stats["cluster_removed_short"]),
        "total_trades": float(total_trades),
        "total_ev_atr": float(total_ev),
        "total_trades_per_day": float(long_metrics["trades_per_day"] + short_metrics["trades_per_day"]),
    }
    for prefix, metrics in (("long", long_metrics), ("short", short_metrics)):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    return (
        row,
        long_entries,
        short_entries,
        long_candidate,
        short_candidate,
        long_prices,
        short_prices,
        long_times,
        short_times,
    )


def _plot_diagnostics(
    feature_df: pd.DataFrame,
    p_long: np.ndarray,
    p_short: np.ndarray,
    long_setup: np.ndarray,
    short_setup: np.ndarray,
    long_entries: np.ndarray,
    short_entries: np.ndarray,
    long_trigger: np.ndarray | None,
    short_trigger: np.ndarray | None,
    y_long: np.ndarray,
    y_short: np.ndarray,
    *,
    title: str,
    save_path: Path,
    tail: int,
    long_setup_threshold: float | None = None,
    short_setup_threshold: float | None = None,
) -> None:
    work = feature_df.copy()
    work["p_long"] = p_long
    work["p_short"] = p_short
    work["long_setup"] = long_setup
    work["short_setup"] = short_setup
    work["long_entry"] = long_entries
    work["short_entry"] = short_entries
    work["long_trigger"] = np.zeros(len(feature_df), dtype=bool) if long_trigger is None else long_trigger
    work["short_trigger"] = np.zeros(len(feature_df), dtype=bool) if short_trigger is None else short_trigger
    work["actual_long"] = y_long
    work["actual_short"] = y_short
    work = _select_plot_window(work, window=tail, random_window=False)

    pos, open_y, high_y, low_y, close_y = _extract_ohlc(work)
    marker_offset = _compute_marker_offset(work, high_y, low_y)
    tick_positions, tick_labels = _compute_time_ticks(work.index, pos, max_ticks=20)

    fig, (ax_price, ax_prob) = plt.subplots(
        2,
        1,
        figsize=(18, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    _plot_candles(ax_price, pos, open_y, high_y, low_y, close_y)

    long_setup_only = work["long_setup"].to_numpy(dtype=bool) & ~work["long_entry"].to_numpy(dtype=bool)
    short_setup_only = work["short_setup"].to_numpy(dtype=bool) & ~work["short_entry"].to_numpy(dtype=bool)
    long_trigger_only = (
        work["long_trigger"].to_numpy(dtype=bool)
        & ~work["long_entry"].to_numpy(dtype=bool)
    )
    short_trigger_only = (
        work["short_trigger"].to_numpy(dtype=bool)
        & ~work["short_entry"].to_numpy(dtype=bool)
    )
    long_entry = work["long_entry"].to_numpy(dtype=bool)
    short_entry = work["short_entry"].to_numpy(dtype=bool)
    actual_long = work["actual_long"].to_numpy(dtype=bool)
    actual_short = work["actual_short"].to_numpy(dtype=bool)

    if long_setup_only.any():
        ax_price.scatter(pos[long_setup_only], low_y[long_setup_only] - marker_offset * 0.45, marker=".", s=28, color="#90CAF9", alpha=0.55, label="LONG setup only")
    if short_setup_only.any():
        ax_price.scatter(pos[short_setup_only], high_y[short_setup_only] + marker_offset * 0.45, marker=".", s=28, color="#FFCC80", alpha=0.55, label="SHORT setup only")
    if long_trigger_only.any():
        ax_price.scatter(pos[long_trigger_only], low_y[long_trigger_only] - marker_offset * 0.72, marker="x", s=52, color="#42A5F5", alpha=0.75, label="LONG trigger candidate")
    if short_trigger_only.any():
        ax_price.scatter(pos[short_trigger_only], high_y[short_trigger_only] + marker_offset * 0.72, marker="x", s=52, color="#FFA726", alpha=0.75, label="SHORT trigger candidate")
    if long_entry.any():
        ax_price.scatter(pos[long_entry], low_y[long_entry] - marker_offset * 0.9, marker="^", s=90, color="#1565C0", label="triggered LONG")
    if short_entry.any():
        ax_price.scatter(pos[short_entry], high_y[short_entry] + marker_offset * 0.9, marker="v", s=90, color="#FB8C00", label="triggered SHORT")
    if actual_long.any():
        ax_price.scatter(pos[actual_long], low_y[actual_long] - marker_offset * 1.35, marker="^", s=58, facecolors="none", edgecolors="#0D47A1", linewidths=1.2, label="actual LONG")
    if actual_short.any():
        ax_price.scatter(pos[actual_short], high_y[actual_short] + marker_offset * 1.35, marker="v", s=58, facecolors="none", edgecolors="#EF6C00", linewidths=1.2, label="actual SHORT")

    ax_prob.plot(pos, work["p_long"], label="p_long setup", color="#1565C0", linewidth=1.5)
    ax_prob.plot(pos, work["p_short"], label="p_short setup", color="#FB8C00", linewidth=1.5)
    if long_setup_threshold is not None:
        ax_prob.axhline(
            float(long_setup_threshold),
            color="#1565C0",
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
            label=f"long setup thr={float(long_setup_threshold):.2f}",
        )
    if short_setup_threshold is not None:
        ax_prob.axhline(
            float(short_setup_threshold),
            color="#FB8C00",
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
            label=f"short setup thr={float(short_setup_threshold):.2f}",
        )
    ax_prob.set_ylim(0, 1.02)
    ax_price.set_title(title)
    ax_prob.set_title("Setup probabilities")
    ax_price.set_ylabel("Price")
    ax_prob.set_ylabel("Probability")
    ax_prob.set_xlabel("Session")
    ax_price.grid(True, alpha=0.25)
    ax_prob.grid(True, alpha=0.25)
    ax_price.legend(loc="best", fontsize=8, ncols=2)
    ax_prob.legend(loc="best", fontsize=8)
    _draw_day_lines((ax_price, ax_prob), tick_positions, line_color="#d0d0d0")
    _apply_time_ticks(ax_prob, tick_positions, tick_labels)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    print(f"[phase4] saved plot {save_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 setup + trigger + cooldown analysis for single-head swing model.")
    parser.add_argument("--ticker", type=str, default="SPY")
    parser.add_argument("--dataset-name", type=str, default="10min")
    parser.add_argument("--x-filename", type=str, default="X_10min_tree.parquet")
    parser.add_argument("--model-root", type=str, default="Data/models")
    parser.add_argument("--single-label-dir", type=str, default="swing_support_single")
    parser.add_argument("--split-root", type=str, default=None)
    parser.add_argument("--long-setup-threshold", type=float, default=0.36)
    parser.add_argument("--short-setup-threshold", type=float, default=0.34)
    parser.add_argument(
        "--setup-hold-bars",
        type=str,
        default="0",
        help="Comma-separated setup latch lengths. A value of 6 means a setup can trigger up to 6 bars after the setup bar.",
    )
    parser.add_argument("--long-setup-hold-bars", type=str, default=None)
    parser.add_argument("--short-setup-hold-bars", type=str, default=None)
    parser.add_argument("--skip-default-variants", action="store_true")
    parser.add_argument("--include-post-setup", action="store_true")
    parser.add_argument(
        "--include-asym-post-setup",
        action="store_true",
        help="Evaluate asymmetric policies: long waits for post-setup break confirmation, short uses same-bar trigger confirmation.",
    )
    parser.add_argument(
        "--post-setup-max-bars",
        type=str,
        default="4",
        help="Comma-separated bars after setup to allow post-setup order triggers.",
    )
    parser.add_argument(
        "--post-setup-policy-filter",
        type=str,
        default=None,
        help="Optional comma-separated post-setup policy names to run, e.g. post_setup_break_prev_stop,post_setup_break_prev_stop_1m_confirm.",
    )
    parser.add_argument(
        "--asym-long-policy-filter",
        type=str,
        default=None,
        help="Optional comma-separated asymmetric long policy names to run, e.g. micro_reversal_1m,break_prev_stop_1m_body_and_close.",
    )
    parser.add_argument(
        "--asym-short-policy-filter",
        type=str,
        default=None,
        help="Optional comma-separated asymmetric short policy names to run, e.g. break_prev_stop_1m_confirm,break_prev_stop_1m_body_and_close.",
    )
    parser.add_argument(
        "--setup-invalidation",
        choices=("none", "price", "price_vwap"),
        default="none",
        help="For held setups, expire early on adverse price/VWAP invalidation.",
    )
    parser.add_argument("--setup-invalidation-atr", type=float, default=0.5)
    parser.add_argument("--cooldown-bars", type=int, default=5)
    parser.add_argument("--horizon-bars", type=int, default=12)
    parser.add_argument("--tp-atr", type=float, default=1.0)
    parser.add_argument("--sl-atr", type=float, default=0.8)
    parser.add_argument("--derive-vwap", action="store_true")
    parser.add_argument(
        "--use-1m-execution",
        action="store_true",
        help="Use 1-minute SPY bars for post-setup breakout touch detection and TP/SL path evaluation.",
    )
    parser.add_argument(
        "--execution-1m-path",
        type=str,
        default="Data/raw/spy/spy_intraday_1min.parquet",
        help="Parquet file with 1-minute OHLC bars and a timestamp column.",
    )
    parser.add_argument("--include-sr-baseline", action="store_true")
    parser.add_argument(
        "--sr-lookback-bars",
        type=str,
        default="12,24",
        help="Comma-separated rolling support/resistance lookbacks for the breakout baseline.",
    )
    parser.add_argument("--tail", type=int, default=300)
    parser.add_argument("--plot-top-n", type=int, default=3)
    parser.add_argument(
        "--splits",
        type=str,
        default="oof,test",
        help="Comma-separated splits to evaluate: oof,test. Use test for quick visual/debug passes.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    plot_df, y_df, _, loaded = _load_phase4_inputs(args)
    feature_df = _add_phase4_features(plot_df, derive_vwap=bool(args.derive_vwap))
    execution_1m = _load_execution_1m(args, feature_df.index)
    y_long, y_short = _support_labels(y_df)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "Data" / "models" / "ga_xgboost" / args.dataset_name / "analysis" / "phase4"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_signal_frame(
        feature_df,
        loaded,
        long_setup_threshold=float(args.long_setup_threshold),
        short_setup_threshold=float(args.short_setup_threshold),
        out_path=out_dir / "phase4_signal_frame.parquet",
    )

    include_vwap = bool(args.derive_vwap) or any(col.lower() == "vwap" for col in feature_df.columns)
    variants = _variants(feature_df, include_vwap=include_vwap)
    setup_hold_values = _parse_int_list(args.setup_hold_bars)
    long_setup_hold_values = (
        _parse_int_list(args.long_setup_hold_bars)
        if args.long_setup_hold_bars is not None
        else setup_hold_values
    )
    short_setup_hold_values = (
        _parse_int_list(args.short_setup_hold_bars)
        if args.short_setup_hold_bars is not None
        else setup_hold_values
    )
    post_setup_max_values = _parse_int_list(args.post_setup_max_bars)
    sr_lookback_values = _parse_int_list(args.sr_lookback_bars)
    post_setup_policies: list[tuple[str, str | None, str | None]] = [
        ("post_setup_next_open", None, None),
        ("post_setup_break_prev_stop", None, None),
        ("post_setup_break_prev_stop_1m_confirm", None, None),
        ("post_setup_break_prev_stop_1m_body", None, None),
        ("post_setup_break_prev_stop_1m_momentum", None, None),
        ("post_setup_break_prev_stop_1m_body_or_close", None, None),
        ("post_setup_break_prev_stop_1m_body_and_close", None, None),
        ("post_setup_micro_reversal_1m", None, None),
        ("post_setup_trigger_A_break", "trigger_A_long", "trigger_A_short"),
        ("post_setup_trigger_B_break_momentum", "trigger_B_long", "trigger_B_short"),
        ("post_setup_trigger_C_break_body", "trigger_C_long", "trigger_C_short"),
        ("post_setup_trigger_E_break_ema", "trigger_E_long", "trigger_E_short"),
        ("post_setup_trigger_F_body_momentum", "trigger_F_long", "trigger_F_short"),
        ("post_setup_trigger_G_ema_slope", "trigger_G_long", "trigger_G_short"),
        ("post_setup_trigger_H_body_ema", "trigger_H_long", "trigger_H_short"),
        ("post_setup_trigger_I_break_or_ema", "trigger_I_long", "trigger_I_short"),
        ("post_setup_trigger_J_close_cross_fast_ema", "trigger_J_long", "trigger_J_short"),
    ]
    if args.post_setup_policy_filter:
        wanted = {part.strip() for part in str(args.post_setup_policy_filter).split(",") if part.strip()}
        post_setup_policies = [policy for policy in post_setup_policies if policy[0] in wanted]
    asym_long_policies: list[tuple[str, str, str | None]] = [
        ("next_open", "next_open", None),
        ("next_open_direction", "next_open_direction", None),
        ("break_prev_stop", "break_prev_stop", None),
        ("break_prev_stop_1m_confirm", "break_prev_stop_1m_confirm", None),
        ("break_prev_stop_1m_body", "break_prev_stop_1m_body", None),
        ("break_prev_stop_1m_momentum", "break_prev_stop_1m_momentum", None),
        ("break_prev_stop_1m_body_or_close", "break_prev_stop_1m_body_or_close", None),
        ("break_prev_stop_1m_body_and_close", "break_prev_stop_1m_body_and_close", None),
        ("break_prev_stop_1m_confirm_no_fresh_low", "break_prev_stop_1m_confirm_no_fresh_low", None),
        (
            "break_prev_stop_1m_body_and_close_no_fresh_low",
            "break_prev_stop_1m_body_and_close_no_fresh_low",
            None,
        ),
        ("micro_reversal_1m", "micro_reversal_1m", None),
        ("H", "trigger_close", "trigger_H_long"),
        ("G", "trigger_close", "trigger_G_long"),
        ("E", "trigger_close", "trigger_E_long"),
        ("I", "trigger_close", "trigger_I_long"),
    ]
    asym_short_policies: list[tuple[str, str, str | None]] = [
        ("next_open", "next_open", None),
        ("next_open_direction", "next_open_direction", None),
        ("break_prev_stop", "break_prev_stop", None),
        ("break_prev_stop_1m_confirm", "break_prev_stop_1m_confirm", None),
        ("break_prev_stop_1m_body", "break_prev_stop_1m_body", None),
        ("break_prev_stop_1m_momentum", "break_prev_stop_1m_momentum", None),
        ("break_prev_stop_1m_body_or_close", "break_prev_stop_1m_body_or_close", None),
        ("break_prev_stop_1m_body_and_close", "break_prev_stop_1m_body_and_close", None),
        ("micro_reversal_1m", "micro_reversal_1m", None),
        ("H", "trigger_close", "trigger_H_short"),
        ("G", "trigger_close", "trigger_G_short"),
        ("E", "trigger_close", "trigger_E_short"),
        ("I", "trigger_close", "trigger_I_short"),
    ]
    if args.asym_long_policy_filter:
        wanted = {part.strip() for part in str(args.asym_long_policy_filter).split(",") if part.strip()}
        asym_long_policies = [policy for policy in asym_long_policies if policy[0] in wanted]
    if args.asym_short_policy_filter:
        wanted = {part.strip() for part in str(args.asym_short_policy_filter).split(",") if part.strip()}
        asym_short_policies = [policy for policy in asym_short_policies if policy[0] in wanted]
    rows: list[dict[str, float | str | bool]] = []
    entry_maps: dict[tuple[str, str, str], tuple[np.ndarray, ...]] = {}

    split_names = {part.strip().lower() for part in str(args.splits).split(",") if part.strip()}
    selected_splits = [(name, loaded[name]) for name in ("oof", "test") if name in split_names]
    for split_name, eval_idx in selected_splits:
        p_long = loaded[f"p_long_{'oof_train' if split_name == 'oof' else 'test'}"]
        p_short = loaded[f"p_short_{'oof_train' if split_name == 'oof' else 'test'}"]
        finite = np.isfinite(p_long) & np.isfinite(p_short)
        long_setup = finite & (p_long >= float(args.long_setup_threshold))
        short_setup = finite & (p_short >= float(args.short_setup_threshold))
        eval_mask = np.zeros(len(feature_df), dtype=bool)
        eval_mask[eval_idx] = True
        long_setup &= eval_mask
        short_setup &= eval_mask

        for variant in ([] if bool(args.skip_default_variants) else variants):
            if not variant.available:
                rows.append(
                    {
                        "split": split_name,
                        "variant": variant.name,
                        "mode": "unavailable",
                        "available": False,
                        "reason": variant.reason,
                    }
                )
                continue
            variant_hold_pairs = (
                [(0, 0)]
                if variant.name == "raw_threshold"
                else [(long_hold, short_hold) for long_hold in long_setup_hold_values for short_hold in short_setup_hold_values]
            )
            modes: list[tuple[str, int, bool]]
            if variant.name == "raw_threshold":
                modes = [
                    ("raw_no_cooldown", -1, False),
                    ("cooldown_only", int(args.cooldown_bars), False),
                    ("cooldown_cluster", int(args.cooldown_bars), True),
                ]
            else:
                modes = [
                    ("cooldown_only", int(args.cooldown_bars), False),
                    ("cooldown_cluster", int(args.cooldown_bars), True),
                ]
            for long_setup_hold_bars, short_setup_hold_bars in variant_hold_pairs:
                for mode_name, cooldown_bars, one_per_cluster in modes:
                    (
                        row,
                        long_entries,
                        short_entries,
                        long_trigger,
                        short_trigger,
                        long_prices,
                        short_prices,
                        long_times,
                        short_times,
                    ) = _evaluate_variant(
                        feature_df,
                        p_long,
                        p_short,
                        variant=variant,
                        split_name=split_name,
                        eval_idx=eval_idx,
                        long_setup_threshold=float(args.long_setup_threshold),
                        short_setup_threshold=float(args.short_setup_threshold),
                        cooldown_bars=cooldown_bars,
                        one_per_setup_cluster=one_per_cluster,
                        long_setup_hold_bars=int(long_setup_hold_bars),
                        short_setup_hold_bars=int(short_setup_hold_bars),
                        setup_invalidation=str(args.setup_invalidation),
                        setup_invalidation_atr=float(args.setup_invalidation_atr),
                        horizon_bars=int(args.horizon_bars),
                        tp_atr=float(args.tp_atr),
                        sl_atr=float(args.sl_atr),
                    )
                    if variant.name == "raw_threshold":
                        hold_suffix = ""
                    elif long_setup_hold_bars == short_setup_hold_bars:
                        hold_suffix = f"_hold{long_setup_hold_bars}"
                    else:
                        hold_suffix = f"_holdL{long_setup_hold_bars}_S{short_setup_hold_bars}"
                    row["mode"] = f"{mode_name}{hold_suffix}"
                    row["cooldown_bars"] = float(cooldown_bars)
                    rows.append(row)
                    entry_maps[(split_name, variant.name, str(row["mode"]))] = (
                        long_entries,
                        short_entries,
                        long_trigger,
                        short_trigger,
                        long_prices,
                        short_prices,
                        long_times,
                        short_times,
                    )

        if bool(args.include_post_setup):
            for max_bars in post_setup_max_values:
                for policy_name, long_trigger_col, short_trigger_col in post_setup_policies:
                    for mode_name, cooldown_bars, one_per_cluster in (
                        ("cooldown_only", int(args.cooldown_bars), False),
                        ("cooldown_cluster", int(args.cooldown_bars), True),
                    ):
                        (
                            row,
                            long_entries,
                            short_entries,
                            long_trigger,
                            short_trigger,
                            long_prices,
                            short_prices,
                            long_times,
                            short_times,
                        ) = _evaluate_post_setup_policy(
                            feature_df,
                            p_long,
                            p_short,
                            split_name=split_name,
                            eval_idx=eval_idx,
                            long_setup_threshold=float(args.long_setup_threshold),
                            short_setup_threshold=float(args.short_setup_threshold),
                            cooldown_bars=cooldown_bars,
                            one_per_setup_cluster=one_per_cluster,
                            policy_name=policy_name,
                            long_trigger_col=long_trigger_col,
                            short_trigger_col=short_trigger_col,
                            max_bars=int(max_bars),
                            horizon_bars=int(args.horizon_bars),
                            tp_atr=float(args.tp_atr),
                            sl_atr=float(args.sl_atr),
                            execution_1m=execution_1m,
                        )
                        row["mode"] = f"{mode_name}_max{max_bars}"
                        row["cooldown_bars"] = float(cooldown_bars)
                        rows.append(row)
                        entry_maps[(split_name, policy_name, str(row["mode"]))] = (
                            long_entries,
                            short_entries,
                            long_trigger,
                            short_trigger,
                            long_prices,
                            short_prices,
                            long_times,
                            short_times,
                        )

        if bool(args.include_asym_post_setup):
            for max_bars in post_setup_max_values:
                for long_policy_name, long_policy, long_trigger_col in asym_long_policies:
                    if long_trigger_col is not None and long_trigger_col not in feature_df.columns:
                        continue
                    for short_policy_name, short_policy, short_trigger_col in asym_short_policies:
                        if short_trigger_col is not None and short_trigger_col not in feature_df.columns:
                            continue
                        for mode_name, cooldown_bars, one_per_cluster in (
                            ("cooldown_only", int(args.cooldown_bars), False),
                            ("cooldown_cluster", int(args.cooldown_bars), True),
                        ):
                            (
                                row,
                                long_entries,
                                short_entries,
                                long_trigger,
                                short_trigger,
                                long_prices,
                                short_prices,
                                long_times,
                                short_times,
                            ) = _evaluate_asymmetric_policy(
                                feature_df,
                                p_long,
                                p_short,
                                split_name=split_name,
                                eval_idx=eval_idx,
                                long_setup_threshold=float(args.long_setup_threshold),
                                short_setup_threshold=float(args.short_setup_threshold),
                                cooldown_bars=cooldown_bars,
                                one_per_setup_cluster=one_per_cluster,
                                long_policy_name=long_policy_name,
                                long_policy=long_policy,
                                long_trigger_col=long_trigger_col,
                                short_policy_name=short_policy_name,
                                short_policy=short_policy,
                                short_trigger_col=short_trigger_col,
                                max_bars=int(max_bars),
                                horizon_bars=int(args.horizon_bars),
                                tp_atr=float(args.tp_atr),
                                sl_atr=float(args.sl_atr),
                                execution_1m=execution_1m,
                            )
                            row["mode"] = f"{mode_name}_longmax{max_bars}_shortmax{max_bars}"
                            row["cooldown_bars"] = float(cooldown_bars)
                            rows.append(row)
                            entry_maps[(split_name, str(row["variant"]), str(row["mode"]))] = (
                                long_entries,
                                short_entries,
                                long_trigger,
                                short_trigger,
                                long_prices,
                                short_prices,
                                long_times,
                                short_times,
                            )

        if bool(args.include_sr_baseline):
            for lookback in sr_lookback_values:
                for mode_name, cooldown_bars, one_per_cluster in (
                    ("cooldown_only", int(args.cooldown_bars), False),
                    ("cooldown_cluster", int(args.cooldown_bars), True),
                ):
                    (
                        row,
                        long_entries,
                        short_entries,
                        long_trigger,
                        short_trigger,
                        long_prices,
                        short_prices,
                        long_times,
                        short_times,
                    ) = _evaluate_sr_breakout_policy(
                        feature_df,
                        split_name=split_name,
                        eval_idx=eval_idx,
                        lookback_bars=int(lookback),
                        cooldown_bars=cooldown_bars,
                        one_per_setup_cluster=one_per_cluster,
                        horizon_bars=int(args.horizon_bars),
                        tp_atr=float(args.tp_atr),
                        sl_atr=float(args.sl_atr),
                        execution_1m=execution_1m,
                    )
                    row["mode"] = f"{mode_name}_lookback{lookback}"
                    row["cooldown_bars"] = float(cooldown_bars)
                    rows.append(row)
                    entry_maps[(split_name, str(row["variant"]), str(row["mode"]))] = (
                        long_entries,
                        short_entries,
                        long_trigger,
                        short_trigger,
                        long_prices,
                        short_prices,
                        long_times,
                        short_times,
                    )

    scoreboard = pd.DataFrame(rows)
    csv_path = out_dir / "phase4_trigger_scoreboard.csv"
    scoreboard.to_csv(csv_path, index=False)
    json_path = out_dir / "phase4_trigger_scoreboard.json"
    json_path.write_text(json.dumps(json.loads(scoreboard.to_json(orient="records")), indent=2))
    print(f"[phase4] wrote {csv_path}")
    print(f"[phase4] wrote {json_path}")

    test_scores = scoreboard[(scoreboard["split"] == "test") & (scoreboard["available"] == True)].copy()
    test_scores = test_scores.sort_values(
        ["total_ev_atr", "long_ev_atr", "short_ev_atr", "total_trades_per_day"],
        ascending=[False, False, False, True],
    )
    show_cols = [
        "variant",
        "mode",
        "setup_hold_bars",
        "long_setup_hold_bars",
        "short_setup_hold_bars",
        "post_setup_max_bars",
        "setup_invalidation",
        "total_ev_atr",
        "long_ev_atr",
        "short_ev_atr",
        "total_trades_per_day",
        "long_trades_per_day",
        "short_trades_per_day",
        "long_win_rate",
        "short_win_rate",
        "long_trades",
        "short_trades",
        "conflicts",
        "cooldown_removed_long",
        "cooldown_removed_short",
        "long_setup_invalidated",
        "short_setup_invalidated",
        "cluster_removed_long",
        "cluster_removed_short",
    ]
    print("[phase4] TEST scoreboard sorted by total_ev_atr:")
    print(test_scores[show_cols].head(20).to_string(index=False))
    best_key: tuple[str, str, str] | None = None
    if not test_scores.empty:
        best_row = test_scores.iloc[0]
        best_key = ("test", str(best_row["variant"]), str(best_row["mode"]))
        best_csv_path = out_dir / "best_phase4_trigger_scoreboard.csv"
        best_json_path = out_dir / "best_phase4_trigger_scoreboard.json"
        pd.DataFrame([best_row]).to_csv(best_csv_path, index=False)
        best_json_path.write_text(json.dumps(json.loads(pd.DataFrame([best_row]).to_json(orient="records"))[0], indent=2))
        print(f"[phase4] wrote best scoreboard row {best_csv_path}")
        print(f"[phase4] wrote best scoreboard row {best_json_path}")

    p_long_test = loaded["p_long_test"]
    p_short_test = loaded["p_short_test"]
    eval_idx = loaded["test"]
    finite = np.isfinite(p_long_test) & np.isfinite(p_short_test)
    eval_mask = np.zeros(len(feature_df), dtype=bool)
    eval_mask[eval_idx] = True
    long_setup_test = eval_mask & finite & (p_long_test >= float(args.long_setup_threshold))
    short_setup_test = eval_mask & finite & (p_short_test >= float(args.short_setup_threshold))
    for _, row in test_scores.head(max(0, int(args.plot_top_n))).iterrows():
        key = ("test", str(row["variant"]), str(row["mode"]))
        if key not in entry_maps:
            continue
        (
            long_entries,
            short_entries,
            long_trigger,
            short_trigger,
            long_prices,
            short_prices,
            long_times,
            short_times,
        ) = entry_maps[key]
        long_setup_hold = int(row["long_setup_hold_bars"]) if pd.notna(row.get("long_setup_hold_bars", np.nan)) else 0
        short_setup_hold = int(row["short_setup_hold_bars"]) if pd.notna(row.get("short_setup_hold_bars", np.nan)) else 0
        setup_invalidation = str(row.get("setup_invalidation", "none"))
        setup_invalidation_atr = float(row.get("setup_invalidation_atr", args.setup_invalidation_atr))
        if setup_invalidation == "post_setup":
            post_setup_max = int(row["post_setup_max_bars"]) if pd.notna(row.get("post_setup_max_bars", np.nan)) else 0
            plot_long_setup = _expand_recent_setup(long_setup_test, post_setup_max)
            plot_short_setup = _expand_recent_setup(short_setup_test, post_setup_max)
        elif setup_invalidation == "asym_post_setup":
            post_setup_max = int(row["post_setup_max_bars"]) if pd.notna(row.get("post_setup_max_bars", np.nan)) else 0
            plot_long_setup = _expand_recent_setup(long_setup_test, post_setup_max)
            plot_short_setup = _expand_recent_setup(short_setup_test, post_setup_max)
        else:
            plot_long_setup, _ = _expand_setup_with_invalidation(
                feature_df,
                long_setup_test,
                side="long",
                hold_bars=long_setup_hold,
                invalidation=setup_invalidation,
                invalidation_atr=setup_invalidation_atr,
            )
            plot_short_setup, _ = _expand_setup_with_invalidation(
                feature_df,
                short_setup_test,
                side="short",
                hold_bars=short_setup_hold,
                invalidation=setup_invalidation,
                invalidation_atr=setup_invalidation_atr,
            )
        plot_long_setup &= eval_mask
        plot_short_setup &= eval_mask
        plot_path = out_dir / f"phase4_{row['variant']}_{row['mode']}_test_tail.png"
        long_trace = _trace_trades_for_entries(
            feature_df,
            long_entries,
            side="long",
            split_name="test",
            variant=str(row["variant"]),
            mode=str(row["mode"]),
            p_long=p_long_test,
            p_short=p_short_test,
            setup=plot_long_setup,
            trigger=long_trigger,
            horizon_bars=int(args.horizon_bars),
            tp_atr=float(args.tp_atr),
            sl_atr=float(args.sl_atr),
            entry_prices=long_prices,
            entry_times=long_times,
            execution_1m=execution_1m,
        )
        short_trace = _trace_trades_for_entries(
            feature_df,
            short_entries,
            side="short",
            split_name="test",
            variant=str(row["variant"]),
            mode=str(row["mode"]),
            p_long=p_long_test,
            p_short=p_short_test,
            setup=plot_short_setup,
            trigger=short_trigger,
            horizon_bars=int(args.horizon_bars),
            tp_atr=float(args.tp_atr),
            sl_atr=float(args.sl_atr),
            entry_prices=short_prices,
            entry_times=short_times,
            execution_1m=execution_1m,
        )
        trace_df = pd.concat([long_trace, short_trace], ignore_index=True)
        if not trace_df.empty:
            trace_df = trace_df.sort_values(["entry_time", "side", "trace_id"])
        trace_path = out_dir / f"phase4_{row['variant']}_{row['mode']}_test_trades.csv"
        trace_df.to_csv(trace_path, index=False)
        print(f"[phase4] saved trade trace {trace_path}")
        best_trace_path = out_dir / f"best_phase4_{row['variant']}_{row['mode']}_test_trades.csv"
        if key == best_key:
            trace_df.to_csv(best_trace_path, index=False)
            print(f"[phase4] saved best trade trace {best_trace_path}")
        _plot_diagnostics(
            feature_df,
            p_long_test,
            p_short_test,
            plot_long_setup,
            plot_short_setup,
            long_entries,
            short_entries,
            long_trigger,
            short_trigger,
            y_long,
            y_short,
            title=(
                f"{normalize_ticker(args.ticker)} Phase 4 {row['variant']} {row['mode']} "
                f"EV={row['total_ev_atr']:.4f} ATR"
            ),
            save_path=plot_path,
            tail=int(args.tail),
            long_setup_threshold=float(args.long_setup_threshold),
            short_setup_threshold=float(args.short_setup_threshold),
        )
        if key == best_key:
            best_plot_path = out_dir / f"best_phase4_{row['variant']}_{row['mode']}_test_tail.png"
            shutil.copy2(plot_path, best_plot_path)
            print(f"[phase4] saved best plot {best_plot_path}")


if __name__ == "__main__":
    main()
