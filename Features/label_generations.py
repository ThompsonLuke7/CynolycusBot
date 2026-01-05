# label_generators.py
#
# All label-generation functions live here.
# Import whatever you want in feature_engineering.py and wire it up as Y.

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta


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
    tp_mult: float = 0.5,
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

            # If pivot failed (stopped out) but capitulation (wider stop) would have won,
            # you might want to force the capitulation logic.
            # Here, we prioritize the WIN (1) if both triggered.
            elif labels[i] == 0 and hit_label == 1:
                labels[i] = 1
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
    # Continuation entry barriers (can be different from pivot model)
    tp_mult: float = 0.5,
    sl_mult: float = 0.5,
    # How far ahead to evaluate the entry from this bar
    max_holding: int = 20,
    # If True, only label bars strictly AFTER the pivot (skip the pivot bar itself)
    exclude_pivot_bar: bool = True,
    # If True, restrict the lookahead window to not pass the next opposite pivot
    stop_at_next_opposite_pivot: bool = True,
    base_label_col: str = "atr_cont_label",
) -> pd.DataFrame:
    """
    Continuation entry-quality labels INSIDE pivot-defined legs.

    Long leg: from pivot_down (swing low) until next pivot_up (swing high).
      For each eligible bar t in that interval:
        entry = close[t]
        tp = entry + tp_mult * ATR[t]
        sl = entry - sl_mult * ATR[t]
        label = +1 if tp hit before sl within lookahead else 0

    Short leg: from pivot_up until next pivot_down.
      entry = close[t]
      tp = entry - tp_mult * ATR[t]
      sl = entry + sl_mult * ATR[t]
      label = -1 if tp hit before sl within lookahead else 0

    Notes:
      - Uses future data to evaluate outcomes (labels), which is intended.
      - Uses pivots to define "we are in a leg" (for label eligibility), which is also intended.
      - This is NOT a pivot detector label. It's "is entering now still good?"
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
            # conservative: stop first
            if low[j] <= sl:
                hit_label = 0.0
                hit_exit = sl
                hit_bars = j - t
                break
            if high[j] >= tp:
                hit_label = 1.0
                hit_exit = tp
                hit_bars = j - t
                break

        rr = (hit_exit / ep - 1.0) if ep != 0 else np.nan
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
            if high[j] >= sl:
                hit_label = 0.0
                hit_exit = sl
                hit_bars = j - t
                break
            if low[j] <= tp:
                hit_label = -1.0
                hit_exit = tp
                hit_bars = j - t
                break

        rr = (hit_exit / ep - 1.0) if ep != 0 else np.nan
        return hit_label, hit_exit, hit_bars, rr

    # Identify pivot segments
    pivot_dn_idx = np.where(piv_dn == 1)[0]
    pivot_up_idx = np.where(piv_up == 1)[0]

    # Long continuation labels inside (pivot_down -> next pivot_up)
    for start in pivot_dn_idx:
        # find next pivot_up after start
        nxt = pivot_up_idx[pivot_up_idx > start]
        if len(nxt) == 0:
            continue
        end = int(nxt[0])

        t0 = start + 1 if exclude_pivot_bar else start
        if t0 >= end:
            continue

        for t in range(t0, end):
            # Determine evaluation window end
            horizon_end = min(t + max_holding, n - 1)

            if stop_at_next_opposite_pivot:
                # Don't look past the next pivot_up (end) since leg is "done" there
                horizon_end = min(horizon_end, end)

            y, xit, hb, rr = eval_long_from(t, horizon_end)
            labels[t] = y
            entry_price[t] = close[t]
            exit_price[t] = xit
            holding_bars[t] = hb
            realized_ret[t] = rr

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
                horizon_end = min(horizon_end, end)

            y, xit, hb, rr = eval_short_from(t, horizon_end)
            labels[t] = y
            entry_price[t] = close[t]
            exit_price[t] = xit
            holding_bars[t] = hb
            realized_ret[t] = rr

    df[base_label_col] = labels
    df[f"{base_label_col}_entry_price"] = entry_price
    df[f"{base_label_col}_exit_price"] = exit_price
    df[f"{base_label_col}_holding_bars"] = holding_bars
    df[f"{base_label_col}_realized_return"] = realized_ret

    # Binary helpers
    df["long_cont_label"] = (df[base_label_col] == 1.0).astype("Int64")
    df["short_cont_label"] = (df[base_label_col] == -1.0).astype("Int64")

    return df


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
    atr_col: str = "atr",
    atr_length: int = 14,
    threshold: float = 0.8,
    tp_mult: float = 0.5,
    sl_mult: float = 0.5,
    max_holding: int = 20,
    long_state_col: str = "p_long_state_gate",
    short_state_col: str = "p_short_state_gate",
    allow_binary_fallback: bool = True,
) -> pd.DataFrame:
    """
    Build a rule-based swing-state gate using pivot probabilities + ATR invalidation.

    - Enter long/short when pivot prob crosses threshold.
    - Stay in state until TP/SL, opposite pivot signal, or max_holding bars.
    - Adds long_state_col / short_state_col as {0.0, 1.0} indicators.
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

    if atr_col not in df.columns:
        df[atr_col] = ta.atr(df[high_col], df[low_col], df["close"], length=atr_length)
    atr = df[atr_col].to_numpy(dtype=float)

    long_state = np.zeros(n, dtype=float)
    short_state = np.zeros(n, dtype=float)

    state = "FLAT"
    entry = np.nan
    sl = np.nan
    tp = np.nan
    age = 0

    for t in range(n):
        if state == "FLAT":
            if p_long[t] >= threshold and not np.isnan(low[t]) and not np.isnan(atr[t]):
                state = "LONG"
                entry = low[t]
                sl = entry - sl_mult * atr[t]
                tp = entry + tp_mult * atr[t]
                age = 0
            elif (
                p_short[t] >= threshold
                and not np.isnan(high[t])
                and not np.isnan(atr[t])
            ):
                state = "SHORT"
                entry = high[t]
                sl = entry + sl_mult * atr[t]
                tp = entry - tp_mult * atr[t]
                age = 0

        elif state == "LONG":
            age += 1
            if not np.isnan(low[t]) and low[t] <= sl:
                state = "FLAT"
            elif not np.isnan(high[t]) and high[t] >= tp:
                state = "FLAT"
            elif p_short[t] >= threshold:
                state = "FLAT"
            elif age >= max_holding:
                state = "FLAT"

        elif state == "SHORT":
            age += 1
            if not np.isnan(high[t]) and high[t] >= sl:
                state = "FLAT"
            elif not np.isnan(low[t]) and low[t] <= tp:
                state = "FLAT"
            elif p_long[t] >= threshold:
                state = "FLAT"
            elif age >= max_holding:
                state = "FLAT"

        long_state[t] = 1.0 if state == "LONG" else 0.0
        short_state[t] = 1.0 if state == "SHORT" else 0.0

    df[long_state_col] = long_state
    df[short_state_col] = short_state
    return df


# ----------------------------------------------------------------------
# Helper: run all label builders in one call
# ----------------------------------------------------------------------
def add_all_labels(
    df: pd.DataFrame,
    *,
    atr_pivot_kwargs: dict | None = None,
    swing_state_decay_kwargs: dict | None = None,
    swing_state_machine_kwargs: dict | None = None,
    continuation_kwargs: dict | None = None,
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
    continuation_kwargs = continuation_kwargs or {}

    df = add_atr_pivot_swing_labels(df, **atr_pivot_kwargs)

    if swing_state_decay_kwargs is not None:
        df = add_pivot_swing_state_probabilities(df, **swing_state_decay_kwargs)
    if swing_state_machine_kwargs is not None:
        df = add_pivot_swing_state_machine(df, **swing_state_machine_kwargs)
    if continuation_kwargs is not None:
        df = add_atr_continuation_entry_labels(df, **continuation_kwargs)

    return df


def main():
    print("goodbye")


if __name__ == "__main__":
    main()
