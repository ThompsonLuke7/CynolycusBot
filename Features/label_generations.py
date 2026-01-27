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

    if label_horizon is not None:
        continuation_kwargs.setdefault("max_holding", label_horizon)
        mfe_mae_kwargs.setdefault("horizon", label_horizon)
        bars_to_exhaustion_kwargs.setdefault("max_bars", label_horizon)

    df = add_atr_pivot_swing_labels(df, **atr_pivot_kwargs)
    if leg_state_kwargs is not None:
        df = add_atr_leg_segmentation_labels(df, **leg_state_kwargs)

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
        label_horizon=label_horizon,
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
