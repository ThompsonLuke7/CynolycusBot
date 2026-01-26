# label_generators.py
#
# All label-generation functions live here.
# Import whatever you want in feature_engineering.py and wire it up as Y.

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

from Features.feature_sets.custom_indicators import add_fractal_pivots
from Features.feature_sets.feature_constants import SWING_LABEL_COLUMNS
from Features.multi_timeframe_features import ensure_time_index, resample_ohlcv


def _get_prob_array(
    df: pd.DataFrame,
    prob_col: str,
    *,
    binary_fallback_col: str | None = None,
    allow_binary_fallback: bool = True,
):
    """
    Fetch a probability series, optionally falling back to a binary column.
    """
    if prob_col in df.columns:
        return df[prob_col].fillna(0.0).to_numpy(dtype=float)

    if allow_binary_fallback and binary_fallback_col is not None:
        if binary_fallback_col in df.columns:
            return df[binary_fallback_col].fillna(0.0).astype(float).to_numpy()

    raise KeyError(
        f"Missing required column '{prob_col}'"
        + (f" and fallback '{binary_fallback_col}'" if binary_fallback_col else "")
    )


# ----------------------------------------------------------------------
# 1) ATR + Pivot-based swing labels (The best scheme)
# ----------------------------------------------------------------------


def add_atr_pivot_swing_labels(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    pivot_up_col: str = "pivot_up",
    pivot_down_col: str = "pivot_down",
    atr_length: int = 14,
    tp_mult: float = 1.0,
    sl_mult: float = 0.5,
    max_holding: int = 20,
    base_label_col: str = "atr_swing_label",
) -> pd.DataFrame:
    """
    ATR-based swing labeling using pivots only
    """

    high = df[high_col].to_numpy(dtype=float)
    low = df[low_col].to_numpy(dtype=float)
    close = df[close_col].to_numpy(dtype=float)

    pivot_up = df[pivot_up_col].to_numpy(dtype=int)
    pivot_down = df[pivot_down_col].to_numpy(dtype=int)

    n = len(df)

    # --- ATR ---
    df["atr"] = ta.atr(df[high_col], df[low_col], df[close_col], length=atr_length)
    atr = df["atr"].to_numpy(dtype=float)

    # --- outputs ---
    labels = np.zeros(n, dtype=float)  # -1, 0, +1
    entry_price = np.full(n, np.nan)
    exit_price = np.full(n, np.nan)
    holding_bars = np.full(n, np.nan)
    realized_ret = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        # ============================================================
        # LONG LOGIC: Standard Pivot
        # ============================================================
        is_pivot_long = pivot_down[i] == 1

        if is_pivot_long:
            ep = low[i]

            tp = ep + tp_mult * atr[i]
            sl = ep - sl_mult * atr[i]

            entry_price[i] = ep

            hit_label = 0
            hit_exit = ep
            hit_bars = 0

            for j in range(i + 1, min(i + 1 + max_holding, n)):
                # Stop check first (conservative)
                if low[j] <= sl:
                    hit_label = 0
                    hit_exit = sl
                    hit_bars = j - i
                    break
                if high[j] >= tp:
                    hit_label = 1
                    hit_exit = tp
                    hit_bars = j - i
                    break

            # If we already have a label (e.g. from pivot), overwrite it ONLY if
            # the capitulation logic found a winner where pivot failed.
            # But for simplicity, we just take the result if it's currently 0.
            if labels[i] == 0:
                labels[i] = hit_label
                exit_price[i] = hit_exit
                holding_bars[i] = hit_bars
                realized_ret[i] = hit_exit / ep - 1.0

        # ============================================================
        # SHORT LOGIC (Standard)
        # ============================================================
        elif pivot_up[i] == 1:
            ep = high[i]
            tp = ep - tp_mult * atr[i]  # profit target BELOW
            sl = ep + sl_mult * atr[i]  # stop ABOVE

            entry_price[i] = ep

            hit_label = 0
            hit_exit = ep
            hit_bars = 0

            for j in range(i + 1, min(i + 1 + max_holding, n)):
                if high[j] >= sl:
                    hit_label = 0  # stopped out
                    hit_exit = sl
                    hit_bars = j - i
                    break
                if low[j] <= tp:
                    hit_label = -1  # good short
                    hit_exit = tp
                    hit_bars = j - i
                    break

            labels[i] = hit_label
            exit_price[i] = hit_exit
            holding_bars[i] = hit_bars
            realized_ret[i] = hit_exit / ep - 1.0

    # Attach back to df
    df[base_label_col] = labels
    df["atr_entry_price"] = entry_price
    df["atr_exit_price"] = exit_price
    df["atr_holding_bars"] = holding_bars
    df["atr_realized_return"] = realized_ret

    # Binary labels
    df["long_swing_label"] = (df[base_label_col] == 1.0).astype("Int64")
    df["short_swing_label"] = (df[base_label_col] == -1.0).astype("Int64")

    return df


def add_atr_leg_segmentation_labels(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_col: str = "atr",
    atr_length: int = 14,
    reversal_atr: float = 2.0,
    start_atr: float | None = None,
    chop_atr_mult: float | None = 0.75,
    use_high_low_extremes: bool = True,
    label_col: str | None = None,
    pivot_price_col: str = "atr_leg_pivot_price",
    pivot_type_col: str = "atr_leg_pivot_type",
    up_label_col: str = "leg_up_label",
    down_label_col: str = "leg_down_label",
) -> pd.DataFrame:
    """
    ATR-based leg segmentation for leg-state labels (not entry outcome labels).

    Concept:
      - Track a current trend (up/down/unknown).
      - Track the extreme price during that trend.
      - Confirm a new pivot when price reverses from the extreme by
        >= reversal_atr * ATR.
      - Label bars between confirmed pivots as +1 for up legs, -1 for down legs
        (0 if unknown early).

    Outputs:
      - label_col (optional): Int64 {-1,0,+1} leg direction per bar
      - pivot_price_col: pivot price at pivot index (NaN otherwise)
      - pivot_type_col: +1 at swing LOW pivots, -1 at swing HIGH pivots
      - up_label_col/down_label_col: one-vs-all labels (1/0)
      - Also adds df[atr_col] if not present

    Notes:
      - Uses future info to confirm pivots (like zigzag). That is fine for labels.
      - start_atr controls how quickly the first trend is established; default
        uses reversal_atr for a slower, more stable leg-state.
      - chop_atr_mult gates weak moves: if price is within this ATR multiple
        of the last confirmed pivot, labels are set to 0 (neutral).
    """

    if start_atr is None:
        start_atr = reversal_atr

    if atr_col not in df.columns:
        df[atr_col] = ta.atr(
            df[high_col], df[low_col], df[close_col], length=atr_length
        )

    atr = df[atr_col].to_numpy(dtype=float)
    high = df[high_col].to_numpy(dtype=float)
    low = df[low_col].to_numpy(dtype=float)
    close = df[close_col].to_numpy(dtype=float)

    n = len(df)
    leg_label = np.zeros(n, dtype=float)
    pivot_price = np.full(n, np.nan)
    pivot_type = np.full(n, np.nan)
    last_pivot_price_series = np.full(n, np.nan)

    trend = 0  # 0 unknown, +1 up, -1 down
    last_pivot_idx = 0
    last_pivot_price = close[0] if n > 0 else np.nan
    extreme_idx = 0
    extreme_price = close[0] if n > 0 else np.nan

    def get_up_extreme_price(i: int) -> float:
        return high[i] if use_high_low_extremes else close[i]

    def get_dn_extreme_price(i: int) -> float:
        return low[i] if use_high_low_extremes else close[i]

    first_valid = np.where(~np.isnan(close))[0]
    if len(first_valid) == 0:
        leg_series = pd.Series(leg_label, index=df.index).astype("Int64")
        if label_col is not None:
            df[label_col] = leg_series
        df[pivot_price_col] = pivot_price
        df[pivot_type_col] = pivot_type
        df[up_label_col] = (leg_series == 1).astype("Int64")
        df[down_label_col] = (leg_series == -1).astype("Int64")
        return df

    start_i = int(first_valid[0])
    last_pivot_idx = start_i
    last_pivot_price = close[start_i]
    extreme_idx = start_i
    extreme_price = close[start_i]
    last_pivot_price_series[start_i] = last_pivot_price

    for i in range(start_i + 1, n):
        if np.isnan(close[i]) or np.isnan(atr[i]) or atr[i] == 0:
            continue

        thresh_start = start_atr * atr[i]
        thresh_rev = reversal_atr * atr[i]

        if trend == 0:
            if close[i] >= last_pivot_price + thresh_start:
                trend = 1
                extreme_idx = i
                extreme_price = get_up_extreme_price(i)
            elif close[i] <= last_pivot_price - thresh_start:
                trend = -1
                extreme_idx = i
                extreme_price = get_dn_extreme_price(i)
            else:
                leg_label[i] = 0
                continue

        if trend == 1:
            candidate = get_up_extreme_price(i)
            if candidate >= extreme_price:
                extreme_price = candidate
                extreme_idx = i

            if close[i] <= extreme_price - thresh_rev:
                pivot_price[extreme_idx] = extreme_price
                pivot_type[extreme_idx] = -1.0  # swing HIGH

                leg_label[last_pivot_idx:extreme_idx] = 1.0

                last_pivot_idx = extreme_idx
                last_pivot_price = extreme_price
                trend = -1

                extreme_idx = i
                extreme_price = get_dn_extreme_price(i)

        elif trend == -1:
            candidate = get_dn_extreme_price(i)
            if candidate <= extreme_price:
                extreme_price = candidate
                extreme_idx = i

            if close[i] >= extreme_price + thresh_rev:
                pivot_price[extreme_idx] = extreme_price
                pivot_type[extreme_idx] = 1.0  # swing LOW

                leg_label[last_pivot_idx:extreme_idx] = -1.0

                last_pivot_idx = extreme_idx
                last_pivot_price = extreme_price
                trend = 1

                extreme_idx = i
                extreme_price = get_up_extreme_price(i)

        last_pivot_price_series[i] = last_pivot_price

    if trend != 0 and last_pivot_idx < n:
        leg_label[last_pivot_idx:] = float(trend)

    if chop_atr_mult is not None:
        dist_from_pivot = np.abs(close - last_pivot_price_series)
        neutral_mask = (
            ~np.isfinite(dist_from_pivot)
            | ~np.isfinite(atr)
            | (dist_from_pivot < chop_atr_mult * atr)
        )
        leg_label[neutral_mask] = 0.0

    leg_series = pd.Series(leg_label, index=df.index).astype("Int64")
    if label_col is not None:
        df[label_col] = leg_series
    df[pivot_price_col] = pivot_price
    df[pivot_type_col] = pivot_type
    df[up_label_col] = (leg_series == 1).astype("Int64")
    df[down_label_col] = (leg_series == -1).astype("Int64")

    return df


def add_atr_continuation_entry_labels(
    df: pd.DataFrame,
    *,
    pivot_down_col: str = "pivot_down",
    pivot_up_col: str = "pivot_up",
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    atr_length: int = 14,
    tp_mult: float = 0.5,
    sl_mult: float = 0.5,
    max_holding: int = 20,
    exclude_pivot_bar: bool = True,
    stop_at_next_opposite_pivot: bool = True,
    base_label_col: str = "atr_cont_label",
    # --- NEW / IMPROVED KNOBS ---
    include_opposite_pivot_bar: bool = False,  # if False, cap at end-1 (recommended)
    min_runway: int = 3,                       # skip bars too close to horizon/leg end
    prevent_overwrite: bool = True,            # only write if labels[t] still 0
    tie_break: str = "stop",                   # "stop" | "tp" | "ignore"
) -> pd.DataFrame:
    """
    Continuation entry-quality labels INSIDE pivot-defined legs.

    Long leg: pivot_down -> next pivot_up
      label +1 if TP hit before SL within window else 0

    Short leg: pivot_up -> next pivot_down
      label -1 if TP hit before SL within window else 0

    Adds:
      base_label_col in {-1,0,+1}
      base_label_col_* metadata columns
      long_cont_label / short_cont_label (binary helpers)
    """

    # Ensure ATR exists
    if atr_col not in df.columns:
        df[atr_col] = ta.atr(
            df[high_col], df[low_col], df[close_col], length=atr_length
        )

    close = df[close_col].to_numpy(dtype=float)
    high = df[high_col].to_numpy(dtype=float)
    low = df[low_col].to_numpy(dtype=float)
    atr = df[atr_col].to_numpy(dtype=float)

    piv_dn = df[pivot_down_col].fillna(0).astype(int).to_numpy()
    piv_up = df[pivot_up_col].fillna(0).astype(int).to_numpy()

    n = len(df)

    labels = np.zeros(n, dtype=float)  # -1, 0, +1
    entry_price = np.full(n, np.nan)
    exit_price = np.full(n, np.nan)
    holding_bars = np.full(n, np.nan)
    realized_ret = np.full(n, np.nan)

    def _resolve_barrier(hit_tp: bool, hit_sl: bool) -> str | None:
        """
        Returns: "tp", "sl", or None if ambiguous and tie_break=="ignore"
        """
        if hit_tp and hit_sl:
            if tie_break == "stop":
                return "sl"
            if tie_break == "tp":
                return "tp"
            if tie_break == "ignore":
                return None
            # default fallback
            return "sl"
        if hit_tp:
            return "tp"
        if hit_sl:
            return "sl"
        return ""

    # Helper to evaluate a single entry with TP/SL barriers
    def eval_long_from(t: int, horizon_end: int):
        ep = close[t]
        if np.isnan(ep) or np.isnan(atr[t]) or atr[t] == 0:
            return 0.0, np.nan, 0, np.nan

        tp = ep + tp_mult * atr[t]
        sl = ep - sl_mult * atr[t]

        hit_label = 0.0
        hit_exit = ep
        hit_bars = 0

        for j in range(t + 1, horizon_end + 1):
            hit_sl = (low[j] <= sl)
            hit_tp = (high[j] >= tp)
            outcome = _resolve_barrier(hit_tp, hit_sl)

            if outcome is None and (hit_tp or hit_sl):
                # ambiguous + ignore => treat as no-label
                return 0.0, np.nan, 0, np.nan

            if outcome == "sl":
                hit_label = 0.0
                hit_exit = sl
                hit_bars = j - t
                break
            if outcome == "tp":
                hit_label = 1.0
                hit_exit = tp
                hit_bars = j - t
                break

        rr = (hit_exit / ep - 1.0) if ep != 0 and np.isfinite(hit_exit) else np.nan
        return hit_label, hit_exit, hit_bars, rr

    def eval_short_from(t: int, horizon_end: int):
        ep = close[t]
        if np.isnan(ep) or np.isnan(atr[t]) or atr[t] == 0:
            return 0.0, np.nan, 0, np.nan

        tp = ep - tp_mult * atr[t]
        sl = ep + sl_mult * atr[t]

        hit_label = 0.0
        hit_exit = ep
        hit_bars = 0

        for j in range(t + 1, horizon_end + 1):
            hit_sl = (high[j] >= sl)
            hit_tp = (low[j] <= tp)
            outcome = _resolve_barrier(hit_tp, hit_sl)

            if outcome is None and (hit_tp or hit_sl):
                return 0.0, np.nan, 0, np.nan

            if outcome == "sl":
                hit_label = 0.0
                hit_exit = sl
                hit_bars = j - t
                break
            if outcome == "tp":
                hit_label = -1.0
                hit_exit = tp
                hit_bars = j - t
                break

        # FIX: profit-positive realized return for shorts
        # If hit_exit < ep (short win), ep/hit_exit - 1 > 0
        rr = (ep / hit_exit - 1.0) if hit_exit != 0 and np.isfinite(hit_exit) else np.nan
        return hit_label, hit_exit, hit_bars, rr

    # Identify pivot segments
    pivot_dn_idx = np.where(piv_dn == 1)[0]
    pivot_up_idx = np.where(piv_up == 1)[0]

    # Long continuation labels inside (pivot_down -> next pivot_up)
    for start in pivot_dn_idx:
        nxt = pivot_up_idx[pivot_up_idx > start]
        if len(nxt) == 0:
            continue
        end = int(nxt[0])

        t0 = start + 1 if exclude_pivot_bar else start
        if t0 >= end:
            continue

        for t in range(t0, end):
            horizon_end = min(t + max_holding, n - 1)

            if stop_at_next_opposite_pivot:
                cap = end if include_opposite_pivot_bar else (end - 1)
                horizon_end = min(horizon_end, cap)

            if horizon_end <= t:
                continue
            if (horizon_end - t) < min_runway:
                continue

            y, xit, hb, rr_ = eval_long_from(t, horizon_end)

            if prevent_overwrite and labels[t] != 0.0:
                continue

            labels[t] = y
            entry_price[t] = close[t]
            exit_price[t] = xit
            holding_bars[t] = hb
            realized_ret[t] = rr_

    # Short continuation labels inside (pivot_up -> next pivot_down)
    for start in pivot_up_idx:
        nxt = pivot_dn_idx[pivot_dn_idx > start]
        if len(nxt) == 0:
            continue
        end = int(nxt[0])

        t0 = start + 1 if exclude_pivot_bar else start
        if t0 >= end:
            continue

        for t in range(t0, end):
            horizon_end = min(t + max_holding, n - 1)

            if stop_at_next_opposite_pivot:
                cap = end if include_opposite_pivot_bar else (end - 1)
                horizon_end = min(horizon_end, cap)

            if horizon_end <= t:
                continue
            if (horizon_end - t) < min_runway:
                continue

            y, xit, hb, rr_ = eval_short_from(t, horizon_end)

            if prevent_overwrite and labels[t] != 0.0:
                continue

            labels[t] = y
            entry_price[t] = close[t]
            exit_price[t] = xit
            holding_bars[t] = hb
            realized_ret[t] = rr_

    df[base_label_col] = labels
    df[f"{base_label_col}_entry_price"] = entry_price
    df[f"{base_label_col}_exit_price"] = exit_price
    df[f"{base_label_col}_holding_bars"] = holding_bars
    df[f"{base_label_col}_realized_return"] = realized_ret

    # Binary helpers (still produced)
    df["long_cont_label"] = (df[base_label_col] == 1.0).astype("Int64")
    df["short_cont_label"] = (df[base_label_col] == -1.0).astype("Int64")

    return df


def add_mfe_mae_labels(
    df: pd.DataFrame,
    *,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_col: str = "atr",
    atr_length: int = 14,
    horizon: int = 20,
    mfe_up_col: str = "mfe_up_atr",
    mfe_down_col: str = "mfe_down_atr",
) -> pd.DataFrame:
    """
    Dense MFE / MAE regression labels.

    For each bar t:
      MFE_up   = (max high[t+1 : t+H] - close[t]) / ATR[t]
      MFE_down = (close[t] - min low[t+1 : t+H]) / ATR[t]

    Uses future data by design (labels only).
    """

    if atr_col not in df.columns:
        df[atr_col] = ta.atr(
            df[high_col], df[low_col], df[close_col], length=atr_length
        )

    high = df[high_col].to_numpy(dtype=float)
    low = df[low_col].to_numpy(dtype=float)
    close = df[close_col].to_numpy(dtype=float)
    atr = df[atr_col].to_numpy(dtype=float)

    n = len(df)
    mfe_up = np.full(n, np.nan)
    mfe_dn = np.full(n, np.nan)

    for t in range(n):
        if np.isnan(close[t]) or np.isnan(atr[t]) or atr[t] == 0:
            continue

        end = min(t + horizon + 1, n)
        if t + 1 >= end:
            continue

        hi = np.nanmax(high[t + 1 : end])
        lo = np.nanmin(low[t + 1 : end])

        mfe_up[t] = (hi - close[t]) / atr[t]
        mfe_dn[t] = (close[t] - lo) / atr[t]

    df[mfe_up_col] = mfe_up
    df[mfe_down_col] = mfe_dn
    return df

def add_bars_to_exhaustion_label(
    df: pd.DataFrame,
    *,
    pivot_up_col: str = "pivot_up",
    pivot_down_col: str = "pivot_down",
    max_bars: int = 20,
    label_col: str = "bars_to_exhaustion",
) -> pd.DataFrame:
    """
    Bars until the next pivot (either up or down).

    For bar t:
      y = min(distance to next pivot, max_bars)

    Dense regression label.
    """

    piv_up = df[pivot_up_col].fillna(0).astype(int).to_numpy()
    piv_dn = df[pivot_down_col].fillna(0).astype(int).to_numpy()

    n = len(df)
    y = np.full(n, np.nan)

    pivot_idx = np.where((piv_up == 1) | (piv_dn == 1))[0]
    if len(pivot_idx) == 0:
        df[label_col] = y
        return df

    next_pivot_ptr = 0

    for t in range(n):
        while next_pivot_ptr < len(pivot_idx) and pivot_idx[next_pivot_ptr] <= t:
            next_pivot_ptr += 1

        if next_pivot_ptr < len(pivot_idx):
            dist = pivot_idx[next_pivot_ptr] - t
            y[t] = min(dist, max_bars)
        else:
            y[t] = max_bars

    df[label_col] = y
    return df


def add_bars_to_exhaustion_labels(
    df: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """
    Backwards-compatible alias for add_bars_to_exhaustion_label.
    """
    return add_bars_to_exhaustion_label(df, **kwargs)


def add_pivot_swing_state_probabilities(
    df: pd.DataFrame,
    *,
    p_pivot_long_col: str = "p_pivot_long",
    p_pivot_short_col: str = "p_pivot_short",
    pivot_down_col: str = "pivot_down",
    pivot_up_col: str = "pivot_up",
    decay: float = 0.9,
    mutual_inhibition: bool = True,
    long_state_col: str = "p_long_state",
    short_state_col: str = "p_short_state",
    allow_binary_fallback: bool = True,
) -> pd.DataFrame:
    """
    Post-process pivot probabilities into a short-lived swing state probability.

    - Carries forward recent pivot probability with exponential decay.
    - Optionally suppresses long state when short pivot spikes (and vice versa).
    - If probability columns are missing and allow_binary_fallback is True,
      falls back to binary pivot_up/pivot_down columns.
    """
    n = len(df)
    p_long = _get_prob_array(
        df,
        p_pivot_long_col,
        binary_fallback_col=pivot_down_col,
        allow_binary_fallback=allow_binary_fallback,
    )
    p_short = _get_prob_array(
        df,
        p_pivot_short_col,
        binary_fallback_col=pivot_up_col,
        allow_binary_fallback=allow_binary_fallback,
    )

    p_long_state = np.zeros(n, dtype=float)
    p_short_state = np.zeros(n, dtype=float)

    for t in range(n):
        prev_long = p_long_state[t - 1] * decay if t > 0 else 0.0
        prev_short = p_short_state[t - 1] * decay if t > 0 else 0.0

        p_long_state[t] = max(prev_long, p_long[t])
        p_short_state[t] = max(prev_short, p_short[t])

        if mutual_inhibition:
            p_long_state[t] *= 1.0 - p_short[t]
            p_short_state[t] *= 1.0 - p_long[t]

    df[long_state_col] = p_long_state
    df[short_state_col] = p_short_state
    return df


def add_pivot_swing_state_machine(
    df: pd.DataFrame,
    *,
    p_pivot_long_col: str = "p_pivot_long",
    p_pivot_short_col: str = "p_pivot_short",
    pivot_down_col: str = "pivot_down",
    pivot_up_col: str = "pivot_up",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_col: str = "atr",
    atr_length: int = 14,
    threshold: float = 0.65,
    confirm_threshold: float = 0.75,
    confirm_mult: float = 0.12,
    pending_max_bars: int = 20,
    tp_mult: float = 0.5,
    sl_mult: float = 1.0,
    max_holding: int = 120,
    cooldown_bars: int = 3,
    use_tp_exit: bool = False,
    state_id_col: str = "p_state_id",
    swing_id_col: str = "p_swing_id",
    bars_in_state_col: str = "p_bars_in_state",
    pending_age_col: str = "p_pending_age",
    cooldown_remaining_col: str = "p_cooldown_remaining",
    dist_to_sl_atr_col: str = "p_dist_to_sl_atr",
    dist_to_entry_atr_col: str = "p_dist_to_entry_atr",
    dist_to_tp_atr_col: str = "p_dist_to_tp_atr",
    session_phase_col: str = "session_phase",
    session_tz: str | None = "America/New_York",
    session_open_time: str = "09:30",
    session_close_time: str = "16:00",
    session_open_minutes: int = 30,
    session_late_minutes: int = 30,
    long_state_col: str = "p_long_state_gate",
    short_state_col: str = "p_short_state_gate",
    long_pending_col: str = "p_long_pending",
    short_pending_col: str = "p_short_pending",
    flat_state_col: str = "p_flat_state_gate",
    allow_binary_fallback: bool = True,
) -> pd.DataFrame:
    """
    Build a rule-based swing-state gate using pivot probabilities + ATR invalidation.

    - Enter pending long/short when pivot prob crosses threshold.
    - Confirm into a live state only if pivot prob reaches confirm_threshold
      and price confirms by confirm_mult * ATR.
    - Cancel pending if it times out (pending_max_bars) or invalidates via ATR.
    - Stay in state until ATR invalidation, opposite pivot spike, or max_holding.
    - Optional cooldown after any exit to avoid rapid re-entry.
    - Optional TP exit can be enabled with use_tp_exit.
    - Adds long_state_col / short_state_col as {0.0, 1.0} indicators.
    - Adds long_pending_col / short_pending_col / flat_state_col for visualization.
    - Adds state_id_col (0=FLAT,1=PEND_LONG,2=LONG,3=PEND_SHORT,4=SHORT),
      swing_id_col, bars_in_state_col, pending_age_col, cooldown_remaining_col.
    - Adds dist_to_sl_atr_col/dist_to_entry_atr_col/dist_to_tp_atr_col risk features.
    - Adds session_phase_col as {0=OPEN,1=MID,2=LATE} when index is intraday.
    """
    n = len(df)

    p_long = _get_prob_array(
        df,
        p_pivot_long_col,
        binary_fallback_col=pivot_down_col,
        allow_binary_fallback=allow_binary_fallback,
    )
    p_short = _get_prob_array(
        df,
        p_pivot_short_col,
        binary_fallback_col=pivot_up_col,
        allow_binary_fallback=allow_binary_fallback,
    )

    high = df[high_col].to_numpy(dtype=float)
    low = df[low_col].to_numpy(dtype=float)
    close = df[close_col].to_numpy(dtype=float)

    if atr_col not in df.columns:
        df[atr_col] = ta.atr(
            df[high_col], df[low_col], df[close_col], length=atr_length
        )
    atr = df[atr_col].to_numpy(dtype=float)

    state_id = np.zeros(n, dtype=float)
    swing_id = np.zeros(n, dtype=float)
    bars_in_state = np.zeros(n, dtype=float)
    pending_age_series = np.zeros(n, dtype=float)
    cooldown_remaining = np.zeros(n, dtype=float)
    dist_to_sl_atr = np.full(n, np.nan, dtype=float)
    dist_to_entry_atr = np.full(n, np.nan, dtype=float)
    dist_to_tp_atr = np.full(n, np.nan, dtype=float)

    long_state = np.zeros(n, dtype=float)
    short_state = np.zeros(n, dtype=float)
    long_pending = np.zeros(n, dtype=float)
    short_pending = np.zeros(n, dtype=float)
    flat_state = np.zeros(n, dtype=float)

    session_phase = np.full(n, np.nan, dtype=float)
    if isinstance(df.index, pd.DatetimeIndex) and n > 0:
        idx = df.index
        if session_tz is not None:
            if idx.tz is None:
                idx = idx.tz_localize(session_tz)
            else:
                idx = idx.tz_convert(session_tz)

        try:
            open_hour, open_minute = [int(x) for x in session_open_time.split(":")]
            close_hour, close_minute = [int(x) for x in session_close_time.split(":")]
        except ValueError:
            open_hour, open_minute = 9, 30
            close_hour, close_minute = 16, 0

        open_total = open_hour * 60 + open_minute
        close_total = close_hour * 60 + close_minute
        minutes = idx.hour * 60 + idx.minute
        in_session = (minutes >= open_total) & (minutes <= close_total)
        open_end = open_total + max(0, session_open_minutes)
        late_start = close_total - max(0, session_late_minutes)

        open_mask = in_session & (minutes < open_end)
        late_mask = in_session & (minutes >= late_start)
        mid_mask = in_session & ~open_mask & ~late_mask
        session_phase[open_mask] = 0
        session_phase[mid_mask] = 1
        session_phase[late_mask] = 2

    state = "FLAT"
    cooldown = 0
    pending_age = 0
    pending_entry = np.nan
    pending_sl = np.nan
    pending_tp = np.nan
    entry = np.nan
    sl = np.nan
    tp = np.nan
    age = 0
    state_age = 0
    swing_counter = 0
    state_id_map = {
        "FLAT": 0,
        "PENDING_LONG": 1,
        "LONG": 2,
        "PENDING_SHORT": 3,
        "SHORT": 4,
    }

    for t in range(n):
        prev_state = state
        if cooldown > 0:
            cooldown -= 1
            state = "FLAT"
        elif state == "FLAT":
            if p_long[t] >= threshold and not np.isnan(low[t]) and not np.isnan(atr[t]):
                state = "PENDING_LONG"
                pending_age = 0
                pending_entry = low[t]
                pending_sl = pending_entry - sl_mult * atr[t]
                pending_tp = pending_entry + tp_mult * atr[t] if use_tp_exit else np.nan
            elif (
                p_short[t] >= threshold
                and not np.isnan(high[t])
                and not np.isnan(atr[t])
            ):
                state = "PENDING_SHORT"
                pending_age = 0
                pending_entry = high[t]
                pending_sl = pending_entry + sl_mult * atr[t]
                pending_tp = pending_entry - tp_mult * atr[t] if use_tp_exit else np.nan

        elif state == "PENDING_LONG":
            pending_age += 1
            if p_short[t] >= threshold:
                state = "FLAT"
                if cooldown_bars > 0:
                    cooldown = cooldown_bars
            elif not np.isnan(low[t]) and low[t] <= pending_sl:
                state = "FLAT"
                if cooldown_bars > 0:
                    cooldown = cooldown_bars
            elif (
                p_long[t] >= confirm_threshold
                and not np.isnan(close[t])
                and not np.isnan(atr[t])
                and close[t] >= pending_entry + confirm_mult * atr[t]
            ):
                state = "LONG"
                entry = pending_entry
                sl = pending_sl
                tp = pending_tp
                age = 0
            elif pending_age >= pending_max_bars:
                state = "FLAT"
                if cooldown_bars > 0:
                    cooldown = cooldown_bars

        elif state == "PENDING_SHORT":
            pending_age += 1
            if p_long[t] >= threshold:
                state = "FLAT"
                if cooldown_bars > 0:
                    cooldown = cooldown_bars
            elif not np.isnan(high[t]) and high[t] >= pending_sl:
                state = "FLAT"
                if cooldown_bars > 0:
                    cooldown = cooldown_bars
            elif (
                p_short[t] >= confirm_threshold
                and not np.isnan(close[t])
                and not np.isnan(atr[t])
                and close[t] <= pending_entry - confirm_mult * atr[t]
            ):
                state = "SHORT"
                entry = pending_entry
                sl = pending_sl
                tp = pending_tp
                age = 0
            elif pending_age >= pending_max_bars:
                state = "FLAT"
                if cooldown_bars > 0:
                    cooldown = cooldown_bars

        elif state == "LONG":
            age += 1
            exit_now = False
            if not np.isnan(low[t]) and low[t] <= sl:
                exit_now = True
            elif p_short[t] >= threshold:
                exit_now = True
            elif age >= max_holding:
                exit_now = True
            elif use_tp_exit and not np.isnan(high[t]) and high[t] >= tp:
                exit_now = True

            if exit_now:
                state = "FLAT"
                if cooldown_bars > 0:
                    cooldown = cooldown_bars

        elif state == "SHORT":
            age += 1
            exit_now = False
            if not np.isnan(high[t]) and high[t] >= sl:
                exit_now = True
            elif p_long[t] >= threshold:
                exit_now = True
            elif age >= max_holding:
                exit_now = True
            elif use_tp_exit and not np.isnan(low[t]) and low[t] <= tp:
                exit_now = True

            if exit_now:
                state = "FLAT"
                if cooldown_bars > 0:
                    cooldown = cooldown_bars

        if prev_state != state:
            state_age = 0
            if state in ("LONG", "SHORT"):
                swing_counter += 1
        else:
            state_age += 1

        long_state[t] = 1.0 if state == "LONG" else 0.0
        short_state[t] = 1.0 if state == "SHORT" else 0.0
        long_pending[t] = 1.0 if state == "PENDING_LONG" else 0.0
        short_pending[t] = 1.0 if state == "PENDING_SHORT" else 0.0
        flat_state[t] = 1.0 if state == "FLAT" else 0.0
        state_id[t] = state_id_map[state]
        swing_id[t] = swing_counter
        bars_in_state[t] = state_age
        pending_age_series[t] = (
            pending_age if state in ("PENDING_LONG", "PENDING_SHORT") else 0
        )
        cooldown_remaining[t] = cooldown

        if not np.isnan(atr[t]) and atr[t] > 0 and not np.isnan(close[t]):
            if state in ("LONG", "PENDING_LONG"):
                ref_entry = entry if state == "LONG" else pending_entry
                ref_sl = sl if state == "LONG" else pending_sl
                ref_tp = tp if state == "LONG" else pending_tp
                if np.isfinite(ref_sl):
                    dist_to_sl_atr[t] = (close[t] - ref_sl) / atr[t]
                if np.isfinite(ref_entry):
                    dist_to_entry_atr[t] = (close[t] - ref_entry) / atr[t]
                if use_tp_exit and np.isfinite(ref_tp):
                    dist_to_tp_atr[t] = (ref_tp - close[t]) / atr[t]
            elif state in ("SHORT", "PENDING_SHORT"):
                ref_entry = entry if state == "SHORT" else pending_entry
                ref_sl = sl if state == "SHORT" else pending_sl
                ref_tp = tp if state == "SHORT" else pending_tp
                if np.isfinite(ref_sl):
                    dist_to_sl_atr[t] = (ref_sl - close[t]) / atr[t]
                if np.isfinite(ref_entry):
                    dist_to_entry_atr[t] = (ref_entry - close[t]) / atr[t]
                if use_tp_exit and np.isfinite(ref_tp):
                    dist_to_tp_atr[t] = (close[t] - ref_tp) / atr[t]

    df[long_state_col] = long_state
    df[short_state_col] = short_state
    df[long_pending_col] = long_pending
    df[short_pending_col] = short_pending
    df[flat_state_col] = flat_state
    df[state_id_col] = pd.Series(state_id, index=df.index).astype("Int64")
    df[swing_id_col] = pd.Series(swing_id, index=df.index).astype("Int64")
    df[bars_in_state_col] = pd.Series(bars_in_state, index=df.index).astype("Int64")
    df[pending_age_col] = pd.Series(pending_age_series, index=df.index).astype("Int64")
    df[cooldown_remaining_col] = pd.Series(cooldown_remaining, index=df.index).astype(
        "Int64"
    )
    df[dist_to_sl_atr_col] = dist_to_sl_atr
    df[dist_to_entry_atr_col] = dist_to_entry_atr
    df[dist_to_tp_atr_col] = dist_to_tp_atr
    df[session_phase_col] = pd.Series(session_phase, index=df.index).astype("Int64")
    return df


# ----------------------------------------------------------------------
# Helper: run all label builders in one call
# ----------------------------------------------------------------------
def add_all_labels(
    df: pd.DataFrame,
    *,
    atr_pivot_kwargs: dict | None = None,
    leg_state_kwargs: dict | None = None,
    swing_state_decay_kwargs: dict | None = None,
    swing_state_machine_kwargs: dict | None = None,
    continuation_kwargs: dict | None = None,
    mfe_mae_kwargs: dict | None = None,
    bars_to_exhaustion_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper that applies every label function in this module
    to a DataFrame, in a sensible order. Each label's parameters can be
    overridden via the corresponding *_kwargs dict.

    Example
    -------
    df = add_all_labels(
        df,
        atr_pivot_kwargs={"tp_mult": 2.0, "sl_mult": 1.0},
        fwd_ret_kwargs={"horizon": 10, "pct_threshold": 0.01},
    )
    """

    atr_pivot_kwargs = atr_pivot_kwargs or {}
    leg_state_kwargs = leg_state_kwargs or {}
    continuation_kwargs = continuation_kwargs or {}
    mfe_mae_kwargs = mfe_mae_kwargs or {}
    bars_to_exhaustion_kwargs = bars_to_exhaustion_kwargs or {}

    df = add_atr_pivot_swing_labels(df, **atr_pivot_kwargs)
    if leg_state_kwargs is not None:
        df = add_atr_leg_segmentation_labels(df, **leg_state_kwargs)

    if swing_state_decay_kwargs is not None:
        df = add_pivot_swing_state_probabilities(df, **swing_state_decay_kwargs)
    if swing_state_machine_kwargs is not None:
        df = add_pivot_swing_state_machine(df, **swing_state_machine_kwargs)
    if continuation_kwargs is not None:
        df = add_atr_continuation_entry_labels(df, **continuation_kwargs)
    if mfe_mae_kwargs is not None:
        df = add_mfe_mae_labels(df, **mfe_mae_kwargs)
    if bars_to_exhaustion_kwargs is not None:
        df = add_bars_to_exhaustion_label(df, **bars_to_exhaustion_kwargs)

    return df


def add_all_labels_on_timeframe(
    df: pd.DataFrame,
    *,
    label_timeframe: str = "15T",
    tz: str | None = "America/New_York",
    resample_label: str = "right",
    resample_closed: str = "right",
    shift_forward_bars: int = 1,
    atr_pivot_kwargs: dict | None = None,
    leg_state_kwargs: dict | None = None,
    swing_state_decay_kwargs: dict | None = None,
    swing_state_machine_kwargs: dict | None = None,
    continuation_kwargs: dict | None = None,
    mfe_mae_kwargs: dict | None = None,
    bars_to_exhaustion_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Compute labels on a higher timeframe and forward-fill them onto the base index.
    """
    base_index = df.index
    if not isinstance(base_index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    df_local = ensure_time_index(df, tz=tz)
    tf_df = resample_ohlcv(
        df_local,
        label_timeframe,
        label=resample_label,
        closed=resample_closed,
    )
    if tf_df.empty:
        return df

    tf_df = add_fractal_pivots(tf_df)
    tf_df = add_all_labels(
        tf_df,
        atr_pivot_kwargs=atr_pivot_kwargs,
        leg_state_kwargs=leg_state_kwargs,
        swing_state_decay_kwargs=swing_state_decay_kwargs,
        swing_state_machine_kwargs=swing_state_machine_kwargs,
        continuation_kwargs=continuation_kwargs,
        mfe_mae_kwargs=mfe_mae_kwargs,
        bars_to_exhaustion_kwargs=bars_to_exhaustion_kwargs,
    )

    label_cols = [c for c in SWING_LABEL_COLUMNS if c in tf_df.columns]
    if not label_cols:
        return df

    labels = tf_df[label_cols].copy()
    pivot_mask = None
    if "pivot_up" in tf_df.columns or "pivot_down" in tf_df.columns:
        pivot_up = (
            tf_df["pivot_up"].fillna(0).astype(int).to_numpy()
            if "pivot_up" in tf_df.columns
            else np.zeros(len(tf_df), dtype=int)
        )
        pivot_down = (
            tf_df["pivot_down"].fillna(0).astype(int).to_numpy()
            if "pivot_down" in tf_df.columns
            else np.zeros(len(tf_df), dtype=int)
        )
        pivot_mask = (pivot_up == 1) | (pivot_down == 1)

    if pivot_mask is not None:
        pivot_label_cols = {
            "atr_swing_label",
            "long_swing_label",
            "short_swing_label",
            "atr_entry_price",
            "atr_exit_price",
            "atr_holding_bars",
            "atr_realized_return",
        }
        for col in pivot_label_cols:
            if col in labels.columns:
                labels.loc[~pivot_mask, col] = np.nan
    if isinstance(labels.index, pd.DatetimeIndex):
        if base_index.tz is None and labels.index.tz is not None:
            labels.index = labels.index.tz_convert(None)
        elif base_index.tz is not None and labels.index.tz is None:
            labels.index = labels.index.tz_localize(base_index.tz)
        elif base_index.tz is not None and labels.index.tz is not None:
            if base_index.tz != labels.index.tz:
                labels = labels.tz_convert(base_index.tz)

    labels = labels.reindex(base_index, method="ffill")
    if shift_forward_bars:
        labels = labels.shift(shift_forward_bars)

    out = df.copy()
    for col in label_cols:
        out[col] = labels[col]
    return out


def main():
    print("goodbye")


if __name__ == "__main__":
    main()
