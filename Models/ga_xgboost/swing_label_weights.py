from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SwingLeg:
    leg_id: int
    direction: str
    start_idx: int
    end_idx: int
    start_pivot_kind: str
    end_pivot_kind: str


def compute_wilder_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    length: int = 14,
) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]
    )
    return (
        pd.Series(tr)
        .ewm(alpha=1.0 / max(int(length), 1), adjust=False, min_periods=max(int(length), 1))
        .mean()
        .to_numpy(dtype=float)
    )


def apply_swing_pivot_zone_weights(
    y: np.ndarray,
    *,
    positive_window_bars: int,
    ambiguous_window_bars: int,
    neighbor_weight: float,
    ambiguous_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand exact swing pivots into soft zones and return source codes.

    source codes:
    - 0: none / far negative
    - 1: core selected event
    - 2: positive neighbor
    - 3: ambiguous zero/low-weight row
    """
    y_base = (np.asarray(y) == 1).astype(np.int64)
    y_zone = y_base.copy()
    weights = np.ones(y_zone.shape[0], dtype=np.float32)
    source = np.zeros(y_zone.shape[0], dtype=np.int8)
    if y_zone.size == 0:
        return y_zone, weights, source

    positive_window = max(0, int(positive_window_bars))
    ambiguous_window = max(0, int(ambiguous_window_bars))
    ambiguous_window = max(ambiguous_window, positive_window)
    core_idx = np.flatnonzero(y_base == 1)
    if core_idx.size == 0:
        return y_zone, weights, source

    if ambiguous_window > 0:
        for idx in core_idx:
            lo = max(0, int(idx) - ambiguous_window)
            hi = min(y_zone.size, int(idx) + ambiguous_window + 1)
            weights[lo:hi] = float(ambiguous_weight)
            source[lo:hi] = 3

    if positive_window > 0:
        for idx in core_idx:
            lo = max(0, int(idx) - positive_window)
            hi = min(y_zone.size, int(idx) + positive_window + 1)
            y_zone[lo:hi] = 1
            weights[lo:hi] = float(neighbor_weight)
            source[lo:hi] = 2

    y_zone[core_idx] = 1
    weights[core_idx] = 1.0
    source[core_idx] = 1
    return y_zone, weights, source


def keep_first_same_side_event(
    long_y: np.ndarray,
    short_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep only the first event in each same-side run.

    The run resets only when the opposite side has a kept event. This is a
    simple top-level duplicate filter for sequences like SHORT, SHORT, SHORT.
    """
    long_in = (np.asarray(long_y) == 1).astype(np.int64)
    short_in = (np.asarray(short_y) == 1).astype(np.int64)
    if long_in.shape[0] != short_in.shape[0]:
        raise ValueError("long_y and short_y must have the same length.")

    long_out = long_in.copy()
    short_out = short_in.copy()
    suppressed_long = np.zeros(long_in.shape[0], dtype=bool)
    suppressed_short = np.zeros(short_in.shape[0], dtype=bool)
    active_side: str | None = None

    for idx, (is_long, is_short) in enumerate(zip(long_in == 1, short_in == 1)):
        if is_long and is_short:
            long_out[idx] = 0
            short_out[idx] = 0
            suppressed_long[idx] = True
            suppressed_short[idx] = True
            active_side = None
            continue
        if is_long:
            if active_side == "long":
                long_out[idx] = 0
                suppressed_long[idx] = True
            else:
                active_side = "long"
            continue
        if is_short:
            if active_side == "short":
                short_out[idx] = 0
                suppressed_short[idx] = True
            else:
                active_side = "short"

    return long_out, short_out, suppressed_long, suppressed_short


def source_codes_to_strings(source: np.ndarray) -> np.ndarray:
    out = np.full(source.shape[0], "none", dtype=object)
    out[source == 1] = "core"
    out[source == 2] = "neighbor"
    out[source == 3] = "ambiguous"
    out[source == 4] = "suppressed_leg"
    return out


def _read_binary(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise KeyError(f"Missing required label column: {col}")
    return pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=np.int64)


def collapse_alternating_zigzag_pivots(
    *,
    pivot_up: np.ndarray,
    pivot_down: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
) -> list[tuple[int, str]]:
    raw: list[tuple[int, str]] = []
    for idx in range(len(pivot_up)):
        is_high = int(pivot_up[idx]) == 1
        is_low = int(pivot_down[idx]) == 1
        if is_high and is_low:
            continue
        if is_high:
            raw.append((idx, "high"))
        elif is_low:
            raw.append((idx, "low"))

    collapsed: list[tuple[int, str]] = []
    for idx, kind in raw:
        if not collapsed or collapsed[-1][1] != kind:
            collapsed.append((idx, kind))
            continue

        prev_idx, _ = collapsed[-1]
        if kind == "high":
            prev_val = high[prev_idx]
            new_val = high[idx]
            replace = np.isfinite(new_val) and (
                not np.isfinite(prev_val) or new_val >= prev_val
            )
        else:
            prev_val = low[prev_idx]
            new_val = low[idx]
            replace = np.isfinite(new_val) and (
                not np.isfinite(prev_val) or new_val <= prev_val
            )
        if replace:
            collapsed[-1] = (idx, kind)
    return collapsed


def filter_macro_pivots(
    pivots: list[tuple[int, str]],
    *,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    min_leg_atr: float = 2.0,
    min_leg_bars: int = 6,
) -> list[tuple[int, str]]:
    """Second-pass pivot filter that keeps only broader ATR-confirmed swings."""
    if len(pivots) <= 2:
        return pivots
    min_leg_atr = float(min_leg_atr)
    min_leg_bars = int(min_leg_bars)
    if min_leg_atr <= 0.0 and min_leg_bars <= 1:
        return pivots

    finite_atr = atr[np.isfinite(atr) & (atr > 0)]
    fallback_atr = float(np.nanmedian(finite_atr)) if finite_atr.size else 1.0

    def _atr_ref(idx: int) -> float:
        value = float(atr[idx]) if 0 <= idx < len(atr) else np.nan
        return value if np.isfinite(value) and value > 0 else fallback_atr

    def _is_more_extreme(idx: int, prev_idx: int, kind: str) -> bool:
        if kind == "high":
            return high[idx] >= high[prev_idx]
        return low[idx] <= low[prev_idx]

    accepted: list[tuple[int, str]] = [pivots[0]]
    for idx, kind in pivots[1:]:
        last_idx, last_kind = accepted[-1]
        if kind == last_kind:
            if _is_more_extreme(idx, last_idx, kind):
                accepted[-1] = (idx, kind)
            continue

        if last_kind == "high" and kind == "low":
            move_atr = (high[last_idx] - low[idx]) / _atr_ref(last_idx)
        elif last_kind == "low" and kind == "high":
            move_atr = (high[idx] - low[last_idx]) / _atr_ref(last_idx)
        else:
            continue
        bars = int(idx) - int(last_idx)
        if move_atr >= min_leg_atr and bars >= min_leg_bars:
            accepted.append((idx, kind))
    return accepted


def build_zigzag_legs(
    *,
    pivot_up: np.ndarray,
    pivot_down: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray | None = None,
    use_macro_filter: bool = False,
    macro_min_leg_atr: float = 2.0,
    macro_min_leg_bars: int = 6,
) -> tuple[list[SwingLeg], list[tuple[int, str]]]:
    pivots = collapse_alternating_zigzag_pivots(
        pivot_up=pivot_up,
        pivot_down=pivot_down,
        high=high,
        low=low,
    )
    if use_macro_filter:
        if atr is None:
            raise ValueError("Macro pivot filtering requires ATR.")
        pivots = filter_macro_pivots(
            pivots,
            high=high,
            low=low,
            atr=atr,
            min_leg_atr=macro_min_leg_atr,
            min_leg_bars=macro_min_leg_bars,
        )
    legs: list[SwingLeg] = []
    leg_id = 0
    for (start_idx, start_kind), (end_idx, end_kind) in zip(pivots[:-1], pivots[1:]):
        if start_idx >= end_idx or start_kind == end_kind:
            continue
        if start_kind == "high" and end_kind == "low":
            direction = "down"
        elif start_kind == "low" and end_kind == "high":
            direction = "up"
        else:
            continue
        legs.append(
            SwingLeg(
                leg_id=leg_id,
                direction=direction,
                start_idx=int(start_idx),
                end_idx=int(end_idx),
                start_pivot_kind=start_kind,
                end_pivot_kind=end_kind,
            )
        )
        leg_id += 1
    return legs, pivots


def _candidate_mfe_mae_atr(
    *,
    side: str,
    idx: int,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    atr: np.ndarray,
    horizon_bars: int,
    entry_price_mode: str,
) -> tuple[float, float, bool]:
    if idx < 0 or idx >= len(close):
        return np.nan, np.nan, False
    atr_ref = float(atr[idx]) if np.isfinite(atr[idx]) and atr[idx] > 0 else np.nan
    if not np.isfinite(atr_ref):
        return np.nan, np.nan, False

    if entry_price_mode == "next_open":
        entry_idx = idx + 1
        if entry_idx >= len(open_) or not np.isfinite(open_[entry_idx]):
            return np.nan, np.nan, False
        entry = float(open_[entry_idx])
        first_forward_idx = entry_idx
    elif entry_price_mode == "close":
        entry = float(close[idx])
        first_forward_idx = idx + 1
    else:
        raise ValueError("entry_price_mode must be 'close' or 'next_open'.")

    if not np.isfinite(entry):
        return np.nan, np.nan, False

    end = min(len(close), first_forward_idx + max(int(horizon_bars), 0))
    if first_forward_idx >= end:
        return np.nan, np.nan, False
    future_high = high[first_forward_idx:end]
    future_low = low[first_forward_idx:end]
    if side == "long":
        mfe = (np.nanmax(future_high) - entry) / atr_ref
        mae = max(0.0, entry - np.nanmin(future_low)) / atr_ref
    elif side == "short":
        mfe = (entry - np.nanmin(future_low)) / atr_ref
        mae = max(0.0, np.nanmax(future_high) - entry) / atr_ref
    else:
        raise ValueError("side must be 'long' or 'short'.")
    return float(mfe), float(mae), True


def _tp_before_sl(
    *,
    side: str,
    idx: int,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    atr: np.ndarray,
    horizon_bars: int,
    tp_atr: float,
    sl_atr: float,
    entry_price_mode: str,
) -> tuple[bool, float, float]:
    mfe, mae, can_eval = _candidate_mfe_mae_atr(
        side=side,
        idx=idx,
        high=high,
        low=low,
        close=close,
        open_=open_,
        atr=atr,
        horizon_bars=horizon_bars,
        entry_price_mode=entry_price_mode,
    )
    if not can_eval:
        return False, mfe, mae

    atr_ref = float(atr[idx])
    if entry_price_mode == "next_open":
        entry_idx = idx + 1
        entry = float(open_[entry_idx])
        first_forward_idx = entry_idx
    else:
        entry = float(close[idx])
        first_forward_idx = idx + 1

    if side == "long":
        tp = entry + float(tp_atr) * atr_ref
        sl = entry - float(sl_atr) * atr_ref
        for j in range(first_forward_idx, min(len(close), first_forward_idx + int(horizon_bars))):
            if low[j] <= sl:
                return False, mfe, mae
            if high[j] >= tp:
                return True, mfe, mae
    else:
        tp = entry - float(tp_atr) * atr_ref
        sl = entry + float(sl_atr) * atr_ref
        for j in range(first_forward_idx, min(len(close), first_forward_idx + int(horizon_bars))):
            if high[j] >= sl:
                return False, mfe, mae
            if low[j] <= tp:
                return True, mfe, mae
    return False, mfe, mae


def build_phase3_swing_event_labels(
    df: pd.DataFrame,
    *,
    positive_window_bars: int = 1,
    ambiguous_window_bars: int = 3,
    neighbor_weight: float = 0.75,
    ambiguous_weight: float = 0.0,
    search_pre_bars: int = 6,
    search_post_bars: int = 3,
    horizon_bars: int = 12,
    tp_atr: float = 1.0,
    sl_atr: float = 0.8,
    entry_price_mode: str = "close",
    atr_length: int = 14,
    same_leg_non_event_weight: float = 0.0,
    use_macro_filter: bool = True,
    macro_min_leg_atr: float = 2.0,
    macro_min_leg_bars: int = 6,
    first_in_run_filter: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[SwingLeg], list[tuple[int, str]]]:
    required = ["open", "high", "low", "close", "pivot_up", "pivot_down"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required Phase 3 columns: {', '.join(missing)}")

    n = len(df)
    open_ = pd.to_numeric(df["open"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    pivot_up = _read_binary(df, "pivot_up")
    pivot_down = _read_binary(df, "pivot_down")
    atr = compute_wilder_atr(high, low, close, length=atr_length)

    long_core = np.zeros(n, dtype=np.int64)
    short_core = np.zeros(n, dtype=np.int64)
    leg_id_arr = np.full(n, -1, dtype=np.int64)
    leg_direction_arr = np.full(n, "none", dtype=object)
    long_candidate_valid = np.zeros(n, dtype=bool)
    short_candidate_valid = np.zeros(n, dtype=bool)
    candidate_forward_mfe_atr = np.full(n, np.nan, dtype=float)
    candidate_forward_mae_atr = np.full(n, np.nan, dtype=float)
    long_candidate_forward_mfe_atr = np.full(n, np.nan, dtype=float)
    long_candidate_forward_mae_atr = np.full(n, np.nan, dtype=float)
    short_candidate_forward_mfe_atr = np.full(n, np.nan, dtype=float)
    short_candidate_forward_mae_atr = np.full(n, np.nan, dtype=float)
    selected_long = np.zeros(n, dtype=bool)
    selected_short = np.zeros(n, dtype=bool)

    legs, pivots = build_zigzag_legs(
        pivot_up=pivot_up,
        pivot_down=pivot_down,
        high=high,
        low=low,
        atr=atr,
        use_macro_filter=use_macro_filter,
        macro_min_leg_atr=macro_min_leg_atr,
        macro_min_leg_bars=macro_min_leg_bars,
    )

    for leg in legs:
        leg_id_arr[leg.start_idx : leg.end_idx + 1] = leg.leg_id
        leg_direction_arr[leg.start_idx : leg.end_idx + 1] = leg.direction
        side = "long" if leg.direction == "down" else "short"
        lo = max(leg.start_idx, leg.end_idx - int(search_pre_bars), 0)
        hi = min(n - 1, leg.end_idx + int(search_post_bars))
        leg_id_arr[lo : hi + 1] = leg.leg_id
        leg_direction_arr[lo : hi + 1] = leg.direction
        selected_idx: int | None = None
        for idx in range(lo, hi + 1):
            valid, mfe, mae = _tp_before_sl(
                side=side,
                idx=idx,
                high=high,
                low=low,
                close=close,
                open_=open_,
                atr=atr,
                horizon_bars=horizon_bars,
                tp_atr=tp_atr,
                sl_atr=sl_atr,
                entry_price_mode=entry_price_mode,
            )
            candidate_forward_mfe_atr[idx] = mfe
            candidate_forward_mae_atr[idx] = mae
            if side == "long":
                long_candidate_valid[idx] = bool(valid)
                long_candidate_forward_mfe_atr[idx] = mfe
                long_candidate_forward_mae_atr[idx] = mae
            else:
                short_candidate_valid[idx] = bool(valid)
                short_candidate_forward_mfe_atr[idx] = mfe
                short_candidate_forward_mae_atr[idx] = mae
            if valid and selected_idx is None:
                selected_idx = idx

        if selected_idx is None:
            continue
        if side == "long":
            long_core[selected_idx] = 1
            selected_long[selected_idx] = True
        else:
            short_core[selected_idx] = 1
            selected_short[selected_idx] = True

    suppressed_run_long = np.zeros(n, dtype=bool)
    suppressed_run_short = np.zeros(n, dtype=bool)
    if first_in_run_filter:
        long_core, short_core, suppressed_run_long, suppressed_run_short = (
            keep_first_same_side_event(long_core, short_core)
        )
        selected_long &= long_core == 1
        selected_short &= short_core == 1

    long_label, long_weight, long_source_code = apply_swing_pivot_zone_weights(
        long_core,
        positive_window_bars=positive_window_bars,
        ambiguous_window_bars=ambiguous_window_bars,
        neighbor_weight=neighbor_weight,
        ambiguous_weight=ambiguous_weight,
    )
    short_label, short_weight, short_source_code = apply_swing_pivot_zone_weights(
        short_core,
        positive_window_bars=positive_window_bars,
        ambiguous_window_bars=ambiguous_window_bars,
        neighbor_weight=neighbor_weight,
        ambiguous_weight=ambiguous_weight,
    )

    for leg in legs:
        lo = max(leg.start_idx, 0)
        hi = min(n, leg.end_idx + 1)
        if leg.direction == "down":
            mask = long_source_code[lo:hi] == 0
            idx = np.arange(lo, hi)[mask]
            long_weight[idx] = float(same_leg_non_event_weight)
            long_source_code[idx] = 4
        elif leg.direction == "up":
            mask = short_source_code[lo:hi] == 0
            idx = np.arange(lo, hi)[mask]
            short_weight[idx] = float(same_leg_non_event_weight)
            short_source_code[idx] = 4

    overlap = (long_label == 1) & (short_label == 1)
    if np.any(overlap):
        long_core_overlap = overlap & (long_core == 1) & (short_core != 1)
        short_core_overlap = overlap & (short_core == 1) & (long_core != 1)
        both_core_overlap = overlap & (long_core == 1) & (short_core == 1)
        soft_overlap = overlap & ~long_core_overlap & ~short_core_overlap & ~both_core_overlap

        short_label[long_core_overlap] = 0
        short_weight[long_core_overlap] = 0.0
        short_source_code[long_core_overlap] = 3

        long_label[short_core_overlap] = 0
        long_weight[short_core_overlap] = 0.0
        long_source_code[short_core_overlap] = 3

        zero_both = both_core_overlap | soft_overlap
        long_label[zero_both] = 0
        short_label[zero_both] = 0
        long_weight[zero_both] = 0.0
        short_weight[zero_both] = 0.0
        long_source_code[zero_both] = 3
        short_source_code[zero_both] = 3

    meta = pd.DataFrame(index=df.index)
    meta["leg_id"] = leg_id_arr
    meta["leg_direction"] = leg_direction_arr
    meta["is_selected_long_event"] = selected_long
    meta["is_selected_short_event"] = selected_short
    meta["is_suppressed_long_run_event"] = suppressed_run_long
    meta["is_suppressed_short_run_event"] = suppressed_run_short
    meta["long_event_source"] = source_codes_to_strings(long_source_code)
    meta["short_event_source"] = source_codes_to_strings(short_source_code)
    meta["long_candidate_valid"] = long_candidate_valid
    meta["short_candidate_valid"] = short_candidate_valid
    meta["candidate_forward_mfe_atr"] = candidate_forward_mfe_atr
    meta["candidate_forward_mae_atr"] = candidate_forward_mae_atr
    meta["long_candidate_forward_mfe_atr"] = long_candidate_forward_mfe_atr
    meta["long_candidate_forward_mae_atr"] = long_candidate_forward_mae_atr
    meta["short_candidate_forward_mfe_atr"] = short_candidate_forward_mfe_atr
    meta["short_candidate_forward_mae_atr"] = short_candidate_forward_mae_atr
    meta["phase3_atr"] = atr

    return long_label, short_label, long_weight, short_weight, meta, legs, pivots
