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


# ----------------------------------------------------------------------
# 2) Triple Barrier label (ATR-based)
# ----------------------------------------------------------------------


def add_triple_barrier_labels_atr(
    df,
    close_col="close",
    high_col="high",
    low_col="low",
    open_col="open",
    atr_col="atr",          # must exist on 15m
    atr_length=14,
    atr_warm_start=True,
    chop_atr_mult=0.5,
    k_up=2,
    k_dn=2,
    max_holding=10,
    base_label_col="tb_label",
):
    """
    ATR-based triple barrier with conservative handling for ambiguous bars and
    time-expiry exits.
    """
    if atr_col not in df.columns:
        df[atr_col] = ta.atr(df[high_col], df[low_col], df[close_col], length=atr_length)
    close = df[close_col].to_numpy(float)
    high  = df[high_col].to_numpy(float)
    low   = df[low_col].to_numpy(float)
    open_ = None
    if open_col in df.columns:
        open_ = df[open_col].to_numpy(float)
    atr   = df[atr_col].to_numpy(float, copy=True)
    if atr_warm_start:
        finite_idx = np.where(np.isfinite(atr))[0]
        if finite_idx.size:
            first_valid = int(finite_idx[0])
            if first_valid > 0:
                atr[:first_valid] = atr[first_valid]

    n = len(df)
    labels = np.zeros(n, dtype=float)
    entry_price = np.full(n, np.nan)
    exit_price = np.full(n, np.nan)
    holding_bars = np.full(n, np.nan)
    realized_ret = np.full(n, np.nan)

    for i in range(n):
        ep = close[i]
        if np.isnan(ep) or np.isnan(atr[i]) or atr[i] <= 0:
            continue

        upper = ep + k_up * atr[i]
        lower = ep - k_dn * atr[i]

        entry_price[i] = ep

        hit_label = 0.0
        hit_exit = ep
        hit_bars = 0

        for j in range(i + 1, min(i + 1 + max_holding, n)):
            hit_upper = high[j] >= upper
            hit_lower = low[j] <= lower

            if hit_upper and not hit_lower:
                hit_label = 1.0
                hit_exit = upper
                if open_ is not None and open_[j] > upper:
                    hit_exit = open_[j]
                hit_bars = j - i
                break
            if hit_lower and not hit_upper:
                hit_label = -1.0
                hit_exit = lower
                if open_ is not None and open_[j] < lower:
                    hit_exit = open_[j]
                hit_bars = j - i
                break
            if hit_upper and hit_lower:
                hit_label = -1.0
                hit_exit = lower
                if open_ is not None and open_[j] < lower:
                    hit_exit = open_[j]
                hit_bars = j - i
                break
        else:
            last_idx = min(i + max_holding, n - 1)
            hit_exit = close[last_idx]
            hit_bars = last_idx - i
            if np.isfinite(hit_exit):
                diff = hit_exit - ep
                if (
                    chop_atr_mult is not None
                    and chop_atr_mult > 0
                    and np.isfinite(atr[i])
                    and abs(diff) <= chop_atr_mult * atr[i]
                ):
                    hit_label = 0.0
                elif diff > 0:
                    hit_label = 1.0
                elif diff < 0:
                    hit_label = -1.0

        labels[i] = hit_label
        exit_price[i] = hit_exit
        holding_bars[i] = hit_bars
        realized_ret[i] = (hit_exit / ep - 1.0) if ep != 0 else np.nan

    df[base_label_col] = labels
    df["tb_entry_price"] = entry_price
    df["tb_exit_price"] = exit_price
    df["tb_holding_bars"] = holding_bars
    df["tb_realized_return"] = realized_ret
    df["tb_long_label"] = (df[base_label_col] == 1.0).astype("Int64")
    df["tb_short_label"] = (df[base_label_col] == -1.0).astype("Int64")
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
    leg_state_col: str = "leg_state",
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
      - leg_state_col: {0=neutral,1=down,2=up}
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
        leg_state = pd.Series(0, index=df.index).astype("Int64")
        leg_state[leg_series == 1] = 2
        leg_state[leg_series == -1] = 1
        df[leg_state_col] = leg_state
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
    leg_state = pd.Series(0, index=df.index).astype("Int64")
    leg_state[leg_series == 1] = 2
    leg_state[leg_series == -1] = 1
    df[leg_state_col] = leg_state

    return df


def add_atr_continuation_strength_labels(
    df: pd.DataFrame,
    *,
    pivot_down_col: str = "pivot_down",
    pivot_up_col: str = "pivot_up",
    leg_state_col: str | None = "atr_leg_label",
    max_holding: int = 20,
    exclude_pivot_bar: bool = True,
    strength_long_col: str = "cont_strength_long",
    strength_short_col: str = "cont_strength_short",
    combined_col: str = "cont_strength",
) -> pd.DataFrame:
    """
    Smooth continuation strength ∈ [0,1].

    Interpretation:
      1.0  -> very early in leg (strong continuation)
      0.5  -> mid-leg
      0.0  -> late / exhausted / invalid

    This is a STATE label, not an EVENT label.
    """

    piv_dn = df[pivot_down_col].fillna(0).astype(int).to_numpy()
    piv_up = df[pivot_up_col].fillna(0).astype(int).to_numpy()
    n = len(df)

    # --- find next pivots ---
    next_up = np.full(n, -1, dtype=int)
    next_dn = np.full(n, -1, dtype=int)
    nu, nd = -1, -1
    for i in range(n - 1, -1, -1):
        next_up[i] = nu
        next_dn[i] = nd
        if piv_up[i] == 1:
            nu = i
        if piv_dn[i] == 1:
            nd = i

    # --- leg state ---
    leg_state = None
    if leg_state_col and leg_state_col in df.columns:
        leg_state = df[leg_state_col].fillna(0).to_numpy(dtype=float)

    cont_long = np.full(n, np.nan, dtype=float)
    cont_short = np.full(n, np.nan, dtype=float)

    for t in range(n):
        if exclude_pivot_bar and (piv_up[t] == 1 or piv_dn[t] == 1):
            continue

        # LONG continuation: distance to next UP pivot
        if next_up[t] >= 0:
            remaining = next_up[t] - t
            strength = 1.0 - min(remaining, max_holding) / max_holding
            cont_long[t] = np.clip(1.0 - strength, 0.0, 1.0)

        # SHORT continuation: distance to next DOWN pivot
        if next_dn[t] >= 0:
            remaining = next_dn[t] - t
            strength = 1.0 - min(remaining, max_holding) / max_holding
            cont_short[t] = np.clip(1.0 - strength, 0.0, 1.0)

    # --- mask to active leg (important) ---
    if leg_state is not None:
        cont_long[leg_state != 1] = np.nan
        cont_short[leg_state != -1] = np.nan

    # --- combined (direction-aware) ---
    cont = np.full(n, np.nan, dtype=float)
    if leg_state is not None:
        cont[leg_state == 1] = cont_long[leg_state == 1]
        cont[leg_state == -1] = cont_short[leg_state == -1]

    df[strength_long_col] = cont_long
    df[strength_short_col] = cont_short
    df[combined_col] = cont

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
    mae_up_col: str = "mae_up_atr",
    mae_down_col: str = "mae_down_atr",
    compute_true_mae: bool = True,
    mask_to_leg: bool = True,
    leg_state_col: str | None = "atr_leg_label",
    leg_up_col: str = "leg_up_label",
    leg_down_col: str = "leg_down_label",
) -> pd.DataFrame:
    """
    Dense MFE / MAE regression labels.

    For each bar t:
      MFE_up   = (max high[t+1 : t+H] - close[t]) / ATR[t]
      MFE_down = (close[t] - min low[t+1 : t+H]) / ATR[t]

    If compute_true_mae is True, also compute:
      MAE_down = max adverse move before MFE_up (long adverse)
      MAE_up   = max adverse move before MFE_down (short adverse)

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
    mae_up = np.full(n, np.nan)
    mae_dn = np.full(n, np.nan)

    for t in range(n):
        if np.isnan(close[t]) or np.isnan(atr[t]) or atr[t] == 0:
            continue

        end = min(t + horizon + 1, n)
        if t + 1 >= end:
            continue

        window_high = high[t + 1 : end]
        window_low = low[t + 1 : end]
        if not np.isfinite(window_high).any() or not np.isfinite(window_low).any():
            continue

        hi_idx = int(np.nanargmax(window_high))
        lo_idx = int(np.nanargmin(window_low))
        hi = window_high[hi_idx]
        lo = window_low[lo_idx]

        mfe_up[t] = (hi - close[t]) / atr[t]
        mfe_dn[t] = (close[t] - lo) / atr[t]

        if compute_true_mae:
            if hi_idx >= 0:
                before_hi = window_low[: hi_idx + 1]
                if np.isfinite(before_hi).any():
                    mae_dn[t] = (close[t] - np.nanmin(before_hi)) / atr[t]
            if lo_idx >= 0:
                before_lo = window_high[: lo_idx + 1]
                if np.isfinite(before_lo).any():
                    mae_up[t] = (np.nanmax(before_lo) - close[t]) / atr[t]

    if mask_to_leg:
        active_leg = None
        if leg_state_col and leg_state_col in df.columns:
            leg_state = df[leg_state_col].fillna(0).to_numpy(dtype=float)
            active_leg = leg_state != 0
        elif leg_up_col in df.columns or leg_down_col in df.columns:
            leg_up = (
                df[leg_up_col].fillna(0).astype(int).to_numpy()
                if leg_up_col in df.columns
                else np.zeros(n, dtype=int)
            )
            leg_dn = (
                df[leg_down_col].fillna(0).astype(int).to_numpy()
                if leg_down_col in df.columns
                else np.zeros(n, dtype=int)
            )
            active_leg = (leg_up == 1) | (leg_dn == 1)

        if active_leg is not None:
            mfe_up[~active_leg] = np.nan
            mfe_dn[~active_leg] = np.nan
            if compute_true_mae:
                mae_up[~active_leg] = np.nan
                mae_dn[~active_leg] = np.nan

    df[mfe_up_col] = mfe_up
    df[mfe_down_col] = mfe_dn
    if compute_true_mae:
        df[mae_up_col] = mae_up
        df[mae_down_col] = mae_dn
    return df

def add_bars_to_exhaustion_label(
    df: pd.DataFrame,
    *,
    pivot_up_col: str = "pivot_up",
    pivot_down_col: str = "pivot_down",
    leg_state_col: str | None = "atr_leg_label",
    leg_up_col: str = "leg_up_label",
    leg_down_col: str = "leg_down_label",
    max_bars: int = 20,
    # --- existing outputs (backwards compatible) ---
    label_col: str = "bars_to_exhaustion",
    censored_col: str = "bars_to_exhaustion_censored",
    # --- NEW: directional outputs ---
    long_label_col: str = "bars_to_exhaustion_long",
    short_label_col: str = "bars_to_exhaustion_short",
    long_censored_col: str = "bars_to_exhaustion_long_censored",
    short_censored_col: str = "bars_to_exhaustion_short_censored",
    # --- NEW: normalized "progress" (0..1) ---
    progress_long_col: str = "exhaustion_progress_long",
    progress_short_col: str = "exhaustion_progress_short",
    # --- behavior knobs ---
    mask_to_leg: bool = True,          # only provide long target on up legs, short on down legs
    exclude_pivot_bar: bool = True,    # if True, pivot bar itself becomes NaN (prevents trivial 0/1 labels)
    fallback_to_any_when_flat: bool = False,  # if leg is 0, compute both anyway if False? (see below)
) -> pd.DataFrame:
    """
    Directional bars-to-exhaustion.

    - LONG exhaustion: bars until next pivot_up
    - SHORT exhaustion: bars until next pivot_down

    Also adds progress targets:
      progress = 1 - (bars_remaining / max_bars), clipped to [0,1]
    which behaves like "how late in the leg are we?"

    Backwards compatible:
      df[label_col] becomes direction-aware:
        - if in up leg => uses long
        - if in down leg => uses short
        - if flat/unknown => uses min(long, short) unless fallback_to_any_when_flat=False
    """

    piv_up = df[pivot_up_col].fillna(0).astype(int).to_numpy()
    piv_dn = df[pivot_down_col].fillna(0).astype(int).to_numpy()

    n = len(df)

    # ----- build next pivot indices (strictly in the future) -----
    next_up = np.full(n, -1, dtype=int)
    next_dn = np.full(n, -1, dtype=int)
    nxt_u = -1
    nxt_d = -1
    for i in range(n - 1, -1, -1):
        next_up[i] = nxt_u
        next_dn[i] = nxt_d
        if piv_up[i] == 1:
            nxt_u = i
        if piv_dn[i] == 1:
            nxt_d = i

    # ----- leg state -----
    leg_state = None
    if leg_state_col and leg_state_col in df.columns:
        leg_state = df[leg_state_col].fillna(0).to_numpy(dtype=float)
    else:
        # fallback from one-vs-all
        leg_up = (
            df[leg_up_col].fillna(0).astype(int).to_numpy()
            if leg_up_col in df.columns
            else np.zeros(n, dtype=int)
        )
        leg_dn = (
            df[leg_down_col].fillna(0).astype(int).to_numpy()
            if leg_down_col in df.columns
            else np.zeros(n, dtype=int)
        )
        leg_state = (leg_up == 1).astype(int) - (leg_dn == 1).astype(int)

    # ----- outputs -----
    y_long = np.full(n, np.nan, dtype=float)
    y_short = np.full(n, np.nan, dtype=float)
    c_long = np.full(n, np.nan, dtype=float)
    c_short = np.full(n, np.nan, dtype=float)

    # helper: distance to next idx
    def _dist_to(next_idx: int, t: int) -> tuple[float, float]:
        if next_idx >= 0:
            dist = next_idx - t
            val = float(min(dist, max_bars))
            cens = 1.0 if dist > max_bars else 0.0
            return val, cens
        # no pivot found -> censored at max
        return float(max_bars), 1.0

    for t in range(n):
        # exclude pivot bar itself if requested (prevents trivial y=0/1 on the event)
        if exclude_pivot_bar and (piv_up[t] == 1 or piv_dn[t] == 1):
            continue

        # compute both directionals always (we'll mask later if desired)
        yL, cL = _dist_to(next_up[t], t)
        yS, cS = _dist_to(next_dn[t], t)

        y_long[t], c_long[t] = yL, cL
        y_short[t], c_short[t] = yS, cS

    # ----- mask to leg direction (recommended) -----
    if mask_to_leg and leg_state is not None:
        up_mask = np.sign(leg_state) == 1
        dn_mask = np.sign(leg_state) == -1
        flat_mask = ~(up_mask | dn_mask)

        # long label only meaningful in up legs
        y_long[~up_mask] = np.nan
        c_long[~up_mask] = np.nan

        # short label only meaningful in down legs
        y_short[~dn_mask] = np.nan
        c_short[~dn_mask] = np.nan

        # if flat/unknown: optionally compute both anyway
        if fallback_to_any_when_flat:
            # restore for flat bars only
            for t in np.where(flat_mask)[0]:
                if exclude_pivot_bar and (piv_up[t] == 1 or piv_dn[t] == 1):
                    continue
                yL, cL = _dist_to(next_up[t], t)
                yS, cS = _dist_to(next_dn[t], t)
                y_long[t], c_long[t] = yL, cL
                y_short[t], c_short[t] = yS, cS

    # ----- progress targets (0..1) -----
    # progress = 1 - remaining/max_bars (late in swing => closer to 1)
    prog_long = np.where(np.isfinite(y_long), 1.0 - (y_long / float(max_bars)), np.nan)
    prog_short = np.where(np.isfinite(y_short), 1.0 - (y_short / float(max_bars)), np.nan)
    prog_long = np.clip(prog_long, 0.0, 1.0)
    prog_short = np.clip(prog_short, 0.0, 1.0)

    # ----- backwards compatible combined label -----
    # - if in up leg -> long
    # - if in down leg -> short
    # - if flat/unknown -> min(long, short) (closest pivot) unless you disabled both via mask
    y = np.full(n, np.nan, dtype=float)
    cens = np.full(n, np.nan, dtype=float)

    if leg_state is not None:
        up_mask = np.sign(leg_state) == 1
        dn_mask = np.sign(leg_state) == -1
        flat_mask = ~(up_mask | dn_mask)

        y[up_mask] = y_long[up_mask]
        cens[up_mask] = c_long[up_mask]
        y[dn_mask] = y_short[dn_mask]
        cens[dn_mask] = c_short[dn_mask]

        # flat: choose closest event if available
        if np.any(flat_mask):
            # compute "any" without unmasking directionals
            for t in np.where(flat_mask)[0]:
                if exclude_pivot_bar and (piv_up[t] == 1 or piv_dn[t] == 1):
                    continue
                yL, cL = _dist_to(next_up[t], t)
                yS, cS = _dist_to(next_dn[t], t)
                if yL <= yS:
                    y[t], cens[t] = yL, cL
                else:
                    y[t], cens[t] = yS, cS
    else:
        # no leg state: default to nearest of up/down pivots
        for t in range(n):
            if exclude_pivot_bar and (piv_up[t] == 1 or piv_dn[t] == 1):
                continue
            yL, cL = _dist_to(next_up[t], t)
            yS, cS = _dist_to(next_dn[t], t)
            if yL <= yS:
                y[t], cens[t] = yL, cL
            else:
                y[t], cens[t] = yS, cS

    # ----- attach -----
    df[long_label_col] = y_long
    df[short_label_col] = y_short
    df[long_censored_col] = pd.Series(c_long, index=df.index).astype("Int64")
    df[short_censored_col] = pd.Series(c_short, index=df.index).astype("Int64")

    df[progress_long_col] = prog_long
    df[progress_short_col] = prog_short

    df[label_col] = y
    df[censored_col] = pd.Series(cens, index=df.index).astype("Int64")

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
# 9) Trend phase labels (momentum/acceleration regime)
# ----------------------------------------------------------------------
def add_trend_phase_labels(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    ema_length: int = 8,
    return_col: str = "trend_phase_ret",
    momentum_col: str = "trend_phase_m",
    accel_raw_col: str = "trend_phase_dm",
    accel_col: str = "trend_phase_a",
    phase_col: str = "trend_phase_label",
    dead_col: str = "trend_phase_dead",
    ignition_col: str = "trend_phase_ignition",
    expansion_col: str = "trend_phase_expansion",
    saturation_col: str = "trend_phase_saturation",
    decay_col: str = "trend_phase_decay",
    exit_long_col: str = "trend_phase_exit_long",
    exit_short_col: str = "trend_phase_exit_short",
    dead_abs_m_threshold: float | None = None,
    dead_quantile: float = 0.10,
    high_m_threshold: float | None = None,
    high_quantile: float = 0.70,
    a_pos_eps: float | None = None,
    a_neg_eps: float | None = None,
    a_near_zero_band: float | None = None,
    a_near_zero_quantile: float = 0.35,
    a_eps_frac_of_zero_band: float = 0.1,
    min_positive_m_for_phase: float = 0.0,
    require_high_m_for_expansion: bool = False,
    min_hold_bars_for_exit: int = 2,
    write_phase_columns: bool = False,
    use_hazard_exit_labels: bool = True,
    exit_hazard_k: int = 2,
) -> pd.DataFrame:
    """
    Label momentum-decay exits and (optionally) phase regime columns.

    Definitions
    -----------
      m = EMA(return, n)
      a = EMA(delta(m), n)

    Notes
    -----
    - If thresholds are omitted, robust quantile-based defaults are derived from
      the current series. In particular, a_pos_eps/a_neg_eps default to a
      non-zero fraction of the near-zero acceleration band.
    - Long decay exit signal is emitted when acceleration flips negative while
      momentum stays positive.
    - Short decay exit signal is emitted when acceleration flips positive while
      momentum stays negative.
    - Both exits are gated by minimum regime age to reduce noise.
    - When use_hazard_exit_labels=True, outputs hazard labels where y_exit[t]=1
      if a point exit occurs within the next exit_hazard_k bars (including t).
    """
    if close_col not in df.columns:
        raise KeyError(f"Missing close column: {close_col}")
    if int(ema_length) <= 0:
        raise ValueError("ema_length must be > 0.")
    if int(exit_hazard_k) < 0:
        raise ValueError("exit_hazard_k must be >= 0.")

    close = pd.Series(df[close_col], index=df.index).astype(float)
    ret = close.pct_change()
    m = ret.ewm(span=int(ema_length), adjust=False, min_periods=1).mean()
    dm = m.diff()
    a = dm.ewm(span=int(ema_length), adjust=False, min_periods=1).mean()

    m_np = m.to_numpy(dtype=float)
    a_np = a.to_numpy(dtype=float)

    finite_m = np.isfinite(m_np)
    finite_a = np.isfinite(a_np)
    finite_both = finite_m & finite_a

    abs_m = np.abs(m_np)
    if dead_abs_m_threshold is None:
        if np.any(finite_m):
            q = float(np.clip(dead_quantile, 0.0, 1.0))
            dead_thr = float(np.nanquantile(abs_m[finite_m], q))
        else:
            dead_thr = 0.0
    else:
        dead_thr = max(0.0, float(dead_abs_m_threshold))

    pos_m = m_np[finite_m & (m_np > float(min_positive_m_for_phase))]
    if high_m_threshold is None:
        if pos_m.size:
            q = float(np.clip(high_quantile, 0.0, 1.0))
            high_thr = float(np.nanquantile(pos_m, q))
        else:
            high_thr = float(min_positive_m_for_phase)
    else:
        high_thr = float(high_m_threshold)

    abs_a = np.abs(a_np)
    if a_near_zero_band is None:
        if np.any(finite_a):
            q = float(np.clip(a_near_zero_quantile, 0.0, 1.0))
            a_zero_band = float(np.nanquantile(abs_a[finite_a], q))
        else:
            a_zero_band = 0.0
    else:
        a_zero_band = max(0.0, float(a_near_zero_band))

    eps_floor = 1e-12
    eps_frac = max(0.0, float(a_eps_frac_of_zero_band))
    if a_pos_eps is None:
        pos_eps = max(eps_floor, a_zero_band * eps_frac)
    else:
        pos_eps = max(0.0, float(a_pos_eps))
    if a_neg_eps is None:
        neg_eps = max(eps_floor, a_zero_band * eps_frac)
    else:
        neg_eps = max(0.0, abs(float(a_neg_eps)))
    min_m = float(min_positive_m_for_phase)

    phase = np.zeros(len(df), dtype=np.int8)
    decay_exit_long = np.zeros(len(df), dtype=np.int8)
    decay_exit_short = np.zeros(len(df), dtype=np.int8)
    min_hold = max(1, int(min_hold_bars_for_exit))
    positive_regime_age = 0
    negative_regime_age = 0
    for i in range(len(df)):
        if not finite_both[i]:
            positive_regime_age = 0
            negative_regime_age = 0
            continue
        m_i = m_np[i]
        a_i = a_np[i]
        abs_m_i = abs(m_i)

        in_positive_regime = abs_m_i > dead_thr and m_i > min_m
        in_negative_regime = abs_m_i > dead_thr and m_i < -min_m
        positive_regime_age = positive_regime_age + 1 if in_positive_regime else 0
        negative_regime_age = negative_regime_age + 1 if in_negative_regime else 0

        if abs(m_i) <= dead_thr:
            phase[i] = 0
            continue

        if m_i > min_m and a_i > pos_eps:
            phase[i] = 1
        elif m_i > min_m and a_i < -neg_eps:
            phase[i] = 3
        else:
            if require_high_m_for_expansion:
                phase[i] = 2 if (m_i > min_m and m_i >= high_thr and abs(a_i) <= a_zero_band) else 0
            else:
                phase[i] = 2 if m_i > min_m else 0

        # Exit trigger: acceleration flips negative while momentum remains positive.
        if in_positive_regime and a_i < -neg_eps and positive_regime_age >= min_hold:
            prev_a = a_np[i - 1] if i > 0 and np.isfinite(a_np[i - 1]) else np.nan
            prev_phase = phase[i - 1] if i > 0 else 0
            accel_flip = np.isfinite(prev_a) and prev_a >= 0.0 and a_i < 0.0
            phase_transition = prev_phase in (1, 2)
            if accel_flip or phase_transition:
                decay_exit_long[i] = 1

        # Symmetric short-side decay exit: down momentum weakens as acceleration turns positive.
        if in_negative_regime and negative_regime_age >= min_hold and a_i > pos_eps:
            prev_a = a_np[i - 1] if i > 0 and np.isfinite(a_np[i - 1]) else np.nan
            prev_in_negative_regime = (
                i > 0
                and np.isfinite(m_np[i - 1])
                and abs(m_np[i - 1]) > dead_thr
                and m_np[i - 1] < -min_m
            )
            accel_flip_short = np.isfinite(prev_a) and prev_a <= 0.0 and a_i > 0.0
            regime_transition_short = prev_in_negative_regime and (
                (not np.isfinite(prev_a)) or prev_a <= pos_eps
            )
            if accel_flip_short or regime_transition_short:
                decay_exit_short[i] = 1

    def _hazardize(point_events: np.ndarray, k: int) -> np.ndarray:
        if k <= 0:
            return point_events.astype(np.int8, copy=True)
        out = np.zeros_like(point_events, dtype=np.int8)
        hit_idx = np.flatnonzero(point_events == 1)
        for j in hit_idx:
            s = max(0, int(j) - int(k))
            out[s : int(j) + 1] = 1
        return out

    if use_hazard_exit_labels:
        exit_long_out = _hazardize(decay_exit_long, int(exit_hazard_k))
        exit_short_out = _hazardize(decay_exit_short, int(exit_hazard_k))
    else:
        exit_long_out = decay_exit_long
        exit_short_out = decay_exit_short

    phase_s = pd.Series(phase, index=df.index).astype("Int64")
    decay_exit_long_s = pd.Series(exit_long_out, index=df.index).astype("Int64")
    decay_exit_short_s = pd.Series(exit_short_out, index=df.index).astype("Int64")
    df[return_col] = ret
    df[momentum_col] = m
    df[accel_raw_col] = dm
    df[accel_col] = a
    if write_phase_columns:
        df[phase_col] = phase_s
        df[dead_col] = (phase_s == 0).astype("Int64")
        df[ignition_col] = (phase_s == 1).astype("Int64")
        df[expansion_col] = (phase_s == 2).astype("Int64")
        df[saturation_col] = (phase_s == 3).astype("Int64")
        df[decay_col] = (phase_s == 3).astype("Int64")
    df[exit_long_col] = decay_exit_long_s
    df[exit_short_col] = decay_exit_short_s
    return df


# ----------------------------------------------------------------------
# 10) Meta entry labels (event-based TP-before-SL with EOD cap)
# ----------------------------------------------------------------------
def build_meta_entry_labels(
    df,
    atr_col="atr",
    a_tp=1.6,
    b_sl=0.8,
    use_next_open=True,
    cost_bps=2.0,
    day_col="session_date",
) -> pd.DataFrame:
    """
    Build meta entry labels for long/short entries with same-session barriers.

    For each bar t:
      - entry is open[t+1] when use_next_open=True, else open[t]
      - TP/SL are ATR-scaled from atr[t]
      - forward scan is capped to bars in the same session only
      - label=1 if TP is hit strictly before SL, else 0

    Output columns:
      - y_enter_long: int8 in {0, 1}
      - y_enter_short: int8 in {0, 1}
    """
    required = {"open", "high", "low", atr_col, day_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    out = df.copy()
    n = len(out)
    y_long = np.zeros(n, dtype=np.int8)
    y_short = np.zeros(n, dtype=np.int8)
    if n == 0:
        out["y_enter_long"] = y_long
        out["y_enter_short"] = y_short
        return out

    open_px = pd.to_numeric(out["open"], errors="coerce").to_numpy(dtype=float)
    high_px = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=float)
    low_px = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(out[atr_col], errors="coerce").to_numpy(dtype=float)
    day_vals = out[day_col].to_numpy()

    entry_offset = 1 if bool(use_next_open) else 0
    tp_mult = float(a_tp)
    sl_mult = float(b_sl)
    cost_bps = max(0.0, float(cost_bps))

    if tp_mult <= 0.0:
        raise ValueError("a_tp must be > 0.")
    if sl_mult <= 0.0:
        raise ValueError("b_sl must be > 0.")

    boundaries = np.flatnonzero(day_vals[1:] != day_vals[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))

    for s, e in zip(starts, ends):
        if (e - s) <= entry_offset:
            continue
        for t in range(s, e):
            entry_idx = t + entry_offset
            if entry_idx >= e:
                break

            entry = open_px[entry_idx]
            atr_t = atr[t]
            if not np.isfinite(entry) or not np.isfinite(atr_t) or atr_t <= 0.0:
                continue

            tp_dist = tp_mult * atr_t
            sl_dist = sl_mult * atr_t
            if tp_dist <= 0.0 or sl_dist <= 0.0:
                continue

            # Skip events where target distance is not large enough to clear costs.
            cost_px = (cost_bps / 1e4) * entry
            if tp_dist <= cost_px:
                continue

            scan_start = entry_idx + 1
            if scan_start >= e:
                continue

            high_fwd = high_px[scan_start:e]
            low_fwd = low_px[scan_start:e]

            long_tp = entry + tp_dist
            long_sl = entry - sl_dist
            short_tp = entry - tp_dist
            short_sl = entry + sl_dist

            long_tp_hit = np.flatnonzero(high_fwd >= long_tp)
            long_sl_hit = np.flatnonzero(low_fwd <= long_sl)
            long_tp_i = int(long_tp_hit[0]) if long_tp_hit.size else None
            long_sl_i = int(long_sl_hit[0]) if long_sl_hit.size else None
            if long_tp_i is not None and (long_sl_i is None or long_tp_i < long_sl_i):
                y_long[t] = 1

            short_tp_hit = np.flatnonzero(low_fwd <= short_tp)
            short_sl_hit = np.flatnonzero(high_fwd >= short_sl)
            short_tp_i = int(short_tp_hit[0]) if short_tp_hit.size else None
            short_sl_i = int(short_sl_hit[0]) if short_sl_hit.size else None
            if short_tp_i is not None and (short_sl_i is None or short_tp_i < short_sl_i):
                y_short[t] = 1

    # Minimal sanity checks
    if y_long.shape[0] != n or y_short.shape[0] != n:
        raise RuntimeError("Meta-entry labels have unexpected length.")
    if not np.all((y_long == 0) | (y_long == 1)):
        raise RuntimeError("y_enter_long has non-binary values.")
    if not np.all((y_short == 0) | (y_short == 1)):
        raise RuntimeError("y_enter_short has non-binary values.")

    out["y_enter_long"] = y_long
    out["y_enter_short"] = y_short
    return out


# ----------------------------------------------------------------------
# 11) Meta exit labels (hazard-style: bars-until-exit <= K)
# ----------------------------------------------------------------------
def build_meta_exit_labels(
    df,
    atr_col="atr",
    enter_long_col="y_enter_long",
    enter_short_col="y_enter_short",
    a_tp=1.6,
    b_sl=0.8,
    use_next_open=True,
    cost_bps=2.0,
    K=2,
    day_col="session_date",
    close_col="close",
    use_decay_exit=True,
    decay_ema_length=8,
    decay_min_hold=2,
    decay_consecutive_bars=2,
    decay_a_eps=0.0,
    decay_require_m_trend=True,
    trail_activate_atr=1.0,
    trail_atr=1.0,
    trail_atr_after_tp=0.8,
    use_tp_to_tighten_trail=True,
    point_long_col="y_exit_long_point",
    point_short_col="y_exit_short_point",
    reason_long_col="exit_reason_long",
    reason_short_col="exit_reason_short",
    tp_hit_long_col="tp_hit_before_exit_long",
    tp_hit_short_col="tp_hit_before_exit_short",
) -> pd.DataFrame:
    """
    Build hybrid hazard-style exit labels for long/short trades driven by meta entries.

    Trade simulation (per session):
      - If y_enter_side[t] == 1, open at open[t+1] (or open[t] if use_next_open=False)
      - TP/SL are ATR-scaled off atr[t]
      - Optional decay trigger uses smoothed momentum/acceleration from close:
          m = EMA(pct_change(close), n)
          a = EMA(diff(m), n)
      - Canonical exit is earliest of {DECAY, TRAIL, SL, EOD}
      - Same-side overlapping entries are ignored while a trade is active
      - TP is analytics-only (can tighten/activate trailing), not a direct exit

    Exit labels:
      - For bars t in [entry_idx, exit_idx):
          y_exit_side[t] = 1 if (exit_idx - t) <= K else 0
      - Bars outside active trade windows are 0
      - Also writes point exit labels and exit reasons at exit bars:
          y_exit_*_point, exit_reason_*
      - Writes TP-hit analytics per side:
          tp_hit_before_exit_long, tp_hit_before_exit_short
    """
    required = {"open", "high", "low", atr_col, day_col, enter_long_col, enter_short_col}
    if use_decay_exit:
        required.add(close_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    out = df.copy()
    n = len(out)
    y_exit_long = np.zeros(n, dtype=np.int8)
    y_exit_short = np.zeros(n, dtype=np.int8)
    y_exit_long_point = np.zeros(n, dtype=np.int8)
    y_exit_short_point = np.zeros(n, dtype=np.int8)
    tp_hit_long = np.zeros(n, dtype=np.int8)
    tp_hit_short = np.zeros(n, dtype=np.int8)
    reason_long = np.full(n, "NONE", dtype=object)
    reason_short = np.full(n, "NONE", dtype=object)
    if n == 0:
        out["y_exit_long"] = y_exit_long
        out["y_exit_short"] = y_exit_short
        out[point_long_col] = y_exit_long_point
        out[point_short_col] = y_exit_short_point
        out[tp_hit_long_col] = tp_hit_long
        out[tp_hit_short_col] = tp_hit_short
        out[reason_long_col] = pd.Series(reason_long, index=out.index).astype("string")
        out[reason_short_col] = pd.Series(reason_short, index=out.index).astype("string")
        return out

    open_px = pd.to_numeric(out["open"], errors="coerce").to_numpy(dtype=float)
    high_px = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=float)
    low_px = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(out[atr_col], errors="coerce").to_numpy(dtype=float)
    day_vals = out[day_col].to_numpy()
    enter_long = out[enter_long_col].fillna(0).astype(int).to_numpy() == 1
    enter_short = out[enter_short_col].fillna(0).astype(int).to_numpy() == 1

    entry_offset = 1 if bool(use_next_open) else 0
    tp_mult = float(a_tp)
    sl_mult = float(b_sl)
    cost_bps = max(0.0, float(cost_bps))
    hazard_k = max(0, int(K))
    decay_span = max(1, int(decay_ema_length))
    min_hold = max(1, int(decay_min_hold))
    min_decay_bars = max(1, int(decay_consecutive_bars))
    decay_eps = max(0.0, float(decay_a_eps))
    trail_activate_mult = max(0.0, float(trail_activate_atr))
    trail_mult = float(trail_atr)
    trail_tight_mult = float(trail_atr_after_tp)

    if tp_mult <= 0.0:
        raise ValueError("a_tp must be > 0.")
    if sl_mult <= 0.0:
        raise ValueError("b_sl must be > 0.")
    if trail_mult <= 0.0:
        raise ValueError("trail_atr must be > 0.")
    if trail_tight_mult <= 0.0:
        trail_tight_mult = trail_mult

    m_np: np.ndarray | None = None
    a_np: np.ndarray | None = None
    if use_decay_exit:
        close = pd.to_numeric(out[close_col], errors="coerce")
        ret = close.pct_change()
        m = ret.ewm(span=decay_span, adjust=False, min_periods=1).mean()
        dm = m.diff()
        a = dm.ewm(span=decay_span, adjust=False, min_periods=1).mean()
        m_np = m.to_numpy(dtype=float)
        a_np = a.to_numpy(dtype=float)

    boundaries = np.flatnonzero(day_vals[1:] != day_vals[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))

    def _label_side(
        enter_sig: np.ndarray,
        y_out: np.ndarray,
        y_point_out: np.ndarray,
        tp_hit_out: np.ndarray,
        reason_out: np.ndarray,
        *,
        is_long: bool,
    ) -> None:
        for s, e in zip(starts, ends):
            if (e - s) <= entry_offset:
                continue

            i = int(s)
            while i < e:
                if not enter_sig[i]:
                    i += 1
                    continue

                entry_idx = i + entry_offset
                if entry_idx >= e:
                    i += 1
                    continue

                entry = open_px[entry_idx]
                atr_i = atr[i]
                if not np.isfinite(entry) or not np.isfinite(atr_i) or atr_i <= 0.0:
                    i += 1
                    continue

                tp_dist = tp_mult * atr_i
                sl_dist = sl_mult * atr_i
                if tp_dist <= 0.0 or sl_dist <= 0.0:
                    i += 1
                    continue

                cost_px = (cost_bps / 1e4) * entry
                if tp_dist <= cost_px:
                    i += 1
                    continue

                if is_long:
                    tp = entry + tp_dist
                    sl = entry - sl_dist
                else:
                    tp = entry - tp_dist
                    sl = entry + sl_dist

                exit_idx = e - 1
                exit_reason = "EOD"
                decay_streak = 0
                tp_seen = False
                trail_active = False
                trail_dist = trail_mult * atr_i
                trail_dist_tight = trail_tight_mult * atr_i
                peak = entry
                trough = entry
                for j in range(entry_idx + 1, e):
                    if is_long:
                        sl_hit = low_px[j] <= sl
                        tp_hit = high_px[j] >= tp
                        if np.isfinite(high_px[j]):
                            peak = max(peak, high_px[j])
                        if (peak - entry) >= (trail_activate_mult * atr_i):
                            trail_active = True
                        if tp_hit:
                            tp_seen = True
                            if use_tp_to_tighten_trail:
                                trail_active = True
                                trail_dist = min(trail_dist, trail_dist_tight)
                        trail_level = peak - trail_dist if trail_active else np.nan
                        trail_hit = trail_active and np.isfinite(trail_level) and (low_px[j] <= trail_level)
                    else:
                        sl_hit = high_px[j] >= sl
                        tp_hit = low_px[j] <= tp
                        if np.isfinite(low_px[j]):
                            trough = min(trough, low_px[j])
                        if (entry - trough) >= (trail_activate_mult * atr_i):
                            trail_active = True
                        if tp_hit:
                            tp_seen = True
                            if use_tp_to_tighten_trail:
                                trail_active = True
                                trail_dist = min(trail_dist, trail_dist_tight)
                        trail_level = trough + trail_dist if trail_active else np.nan
                        trail_hit = trail_active and np.isfinite(trail_level) and (high_px[j] >= trail_level)
                    decay_hit = False
                    if use_decay_exit and m_np is not None and a_np is not None:
                        if (j - entry_idx) >= min_hold and np.isfinite(m_np[j]) and np.isfinite(a_np[j]):
                            if is_long:
                                decay_cond = (m_np[j] > 0.0) and (a_np[j] < -decay_eps)
                                if decay_require_m_trend and j > 0 and np.isfinite(m_np[j - 1]):
                                    decay_cond = decay_cond and (m_np[j] < m_np[j - 1])
                            else:
                                decay_cond = (m_np[j] < 0.0) and (a_np[j] > decay_eps)
                                if decay_require_m_trend and j > 0 and np.isfinite(m_np[j - 1]):
                                    decay_cond = decay_cond and (m_np[j] > m_np[j - 1])
                            decay_streak = decay_streak + 1 if decay_cond else 0
                            decay_hit = decay_streak >= min_decay_bars
                        else:
                            decay_streak = 0

                    if sl_hit or trail_hit or decay_hit:
                        # Same-bar precedence: SL > TRAIL > DECAY.
                        exit_idx = int(j)
                        if sl_hit:
                            exit_reason = "SL"
                        elif trail_hit:
                            exit_reason = "TRAIL"
                        else:
                            exit_reason = "DECAY"
                        break

                if exit_idx > entry_idx:
                    idx = np.arange(entry_idx, exit_idx, dtype=int)
                    y_out[idx] = (exit_idx - idx <= hazard_k).astype(np.int8)
                    y_point_out[exit_idx] = 1
                    tp_hit_out[exit_idx] = np.int8(tp_seen)
                    reason_out[exit_idx] = exit_reason

                # Skip same-side signals while this trade is active.
                i = max(i + 1, exit_idx)

    _label_side(
        enter_long,
        y_exit_long,
        y_exit_long_point,
        tp_hit_long,
        reason_long,
        is_long=True,
    )
    _label_side(
        enter_short,
        y_exit_short,
        y_exit_short_point,
        tp_hit_short,
        reason_short,
        is_long=False,
    )

    # Minimal sanity checks
    if y_exit_long.shape[0] != n or y_exit_short.shape[0] != n:
        raise RuntimeError("Meta-exit labels have unexpected length.")
    if not np.all((y_exit_long == 0) | (y_exit_long == 1)):
        raise RuntimeError("y_exit_long has non-binary values.")
    if not np.all((y_exit_short == 0) | (y_exit_short == 1)):
        raise RuntimeError("y_exit_short has non-binary values.")
    if not np.all((y_exit_long_point == 0) | (y_exit_long_point == 1)):
        raise RuntimeError("y_exit_long_point has non-binary values.")
    if not np.all((y_exit_short_point == 0) | (y_exit_short_point == 1)):
        raise RuntimeError("y_exit_short_point has non-binary values.")
    if not np.all((tp_hit_long == 0) | (tp_hit_long == 1)):
        raise RuntimeError("tp_hit_before_exit_long has non-binary values.")
    if not np.all((tp_hit_short == 0) | (tp_hit_short == 1)):
        raise RuntimeError("tp_hit_before_exit_short has non-binary values.")

    out["y_exit_long"] = y_exit_long
    out["y_exit_short"] = y_exit_short
    out[point_long_col] = y_exit_long_point
    out[point_short_col] = y_exit_short_point
    out[tp_hit_long_col] = tp_hit_long
    out[tp_hit_short_col] = tp_hit_short
    out[reason_long_col] = pd.Series(reason_long, index=out.index).astype("string")
    out[reason_short_col] = pd.Series(reason_short, index=out.index).astype("string")
    return out


# ----------------------------------------------------------------------
# Helper: run all label builders in one call
# ----------------------------------------------------------------------
def add_all_labels(
    df: pd.DataFrame,
    *,
    label_horizon: int | None = None,
    atr_pivot_kwargs: dict | None = None,
    leg_state_kwargs: dict | None = None,
    swing_state_decay_kwargs: dict | None = None,
    swing_state_machine_kwargs: dict | None = None,
    triple_barrier_kwargs: dict | None = None,
    continuation_kwargs: dict | None = None,
    mfe_mae_kwargs: dict | None = None,
    bars_to_exhaustion_kwargs: dict | None = None,
    trend_phase_kwargs: dict | None = None,
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
    triple_barrier_kwargs = triple_barrier_kwargs or {}
    continuation_kwargs = continuation_kwargs or {}
    mfe_mae_kwargs = mfe_mae_kwargs or {}
    bars_to_exhaustion_kwargs = bars_to_exhaustion_kwargs or {}

    if label_horizon is not None:
        continuation_kwargs.setdefault("max_holding", label_horizon)
        triple_barrier_kwargs.setdefault("max_holding", label_horizon)
        mfe_mae_kwargs.setdefault("horizon", label_horizon)
        bars_to_exhaustion_kwargs.setdefault("max_bars", label_horizon)

    df = add_atr_pivot_swing_labels(df, **atr_pivot_kwargs)
    if leg_state_kwargs is not None:
        df = add_atr_leg_segmentation_labels(df, **leg_state_kwargs)
    if triple_barrier_kwargs is not None:
        df = add_triple_barrier_labels_atr(df, **triple_barrier_kwargs)

    if swing_state_decay_kwargs is not None:
        df = add_pivot_swing_state_probabilities(df, **swing_state_decay_kwargs)
    if swing_state_machine_kwargs is not None:
        df = add_pivot_swing_state_machine(df, **swing_state_machine_kwargs)
    if continuation_kwargs is not None:
        df = add_atr_continuation_strength_labels(df, **continuation_kwargs)
    if mfe_mae_kwargs is not None:
        df = add_mfe_mae_labels(df, **mfe_mae_kwargs)
    if bars_to_exhaustion_kwargs is not None:
        df = add_bars_to_exhaustion_label(df, **bars_to_exhaustion_kwargs)
    if trend_phase_kwargs is not None:
        df = add_trend_phase_labels(df, **trend_phase_kwargs)

    return df


def add_all_labels_on_timeframe(
    df: pd.DataFrame,
    *,
    label_timeframe: str = "15T",
    tz: str | None = "America/New_York",
    resample_label: str = "right",
    resample_closed: str = "right",
    shift_forward_bars: int = 1,
    label_horizon: int | None = None,
    atr_pivot_kwargs: dict | None = None,
    leg_state_kwargs: dict | None = None,
    swing_state_decay_kwargs: dict | None = None,
    swing_state_machine_kwargs: dict | None = None,
    triple_barrier_kwargs: dict | None = None,
    continuation_kwargs: dict | None = None,
    mfe_mae_kwargs: dict | None = None,
    bars_to_exhaustion_kwargs: dict | None = None,
    trend_phase_kwargs: dict | None = None,
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
        label_horizon=label_horizon,
        atr_pivot_kwargs=atr_pivot_kwargs,
        leg_state_kwargs=leg_state_kwargs,
        swing_state_decay_kwargs=swing_state_decay_kwargs,
        swing_state_machine_kwargs=swing_state_machine_kwargs,
        triple_barrier_kwargs=triple_barrier_kwargs,
        continuation_kwargs=continuation_kwargs,
        mfe_mae_kwargs=mfe_mae_kwargs,
        bars_to_exhaustion_kwargs=bars_to_exhaustion_kwargs,
        trend_phase_kwargs=trend_phase_kwargs,
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
