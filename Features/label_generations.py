# label_generators.py
#
# All label-generation functions live here.
# Import whatever you want in feature_engineering.py and wire it up as Y.

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1) Simple next-day direction label
# ----------------------------------------------------------------------
def add_next_day_direction_label(
    df: pd.DataFrame,
    close_col: str = "close",
    label_col: str = "direction_1d",
) -> pd.DataFrame:
    """
    Binary label: 1 if next close > current close, else 0.
    Last row has no future close -> NA.

    Parameters
    ----------
    df : DataFrame
    close_col : str
        Column name for close price.
    label_col : str
        Name of the label column to create.

    Returns
    -------
    DataFrame
        df with new column `label_col` (Int64: {0,1,NA}).
    """
    future_close = df[close_col].shift(-1)
    direction = np.where(future_close > df[close_col], 1, -1).astype("float")

    # No future close -> set label to NA
    direction[pd.isna(future_close.to_numpy())] = np.nan

    df[label_col] = pd.Series(direction, index=df.index).astype("Int64")
    return df


def compute_capitulation_pivots(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    low_col: str = "low",
    open_col: str = "open",
    volume_col: str = "volume",
    drawdown_lookback: int = 60,
    drawdown_thresh: float = -0.08,
    local_window: int = 5,
    snapback_bars: int = 3,
    volume_mult: float | None = 1.2,
    reversal_confirm_bars: int = 3,
    reversal_atr_mult: float = 0.5,
) -> pd.Series:
    """
    Fallback pivot detector for deep drawdowns that can break ATR-based logic.
    Triggers when price is deeply below a rolling high, prints a local-low
    (uses past+future window), and shows a snapback (bullish/reclaim or ATR
    retrace). Optional volume spike filter.
    """
    roll_high = df[close_col].rolling(drawdown_lookback, min_periods=5).max()
    drawdown_pct = (df[close_col] / roll_high) - 1.0
    is_extreme_dd = drawdown_pct <= drawdown_thresh

    past_min = df[low_col].rolling(local_window, min_periods=1).min()
    future_min = df[low_col].rolling(local_window, min_periods=1).min().shift(-(local_window - 1))
    local_low = (df[low_col] <= past_min) & (df[low_col] <= future_min)

    bullish_bar = df[close_col] > df[open_col]
    reclaim_prior = df[close_col] > df[close_col].shift(1)
    snapback = (bullish_bar | reclaim_prior).rolling(snapback_bars, min_periods=1).max().astype(bool)

    # ATR-based reclaim within a few bars after the extreme
    if "atr" in df.columns:
        atr = df["atr"]
    else:
        atr = ta.atr(df.get("high", df[close_col]), df.get("low", df[close_col]), df[close_col], length=14)
    reclaim_level = df[low_col] + reversal_atr_mult * atr
    fwd_reclaim = pd.concat(
        [(df[close_col].shift(-k) >= reclaim_level) for k in range(1, reversal_confirm_bars + 1)],
        axis=1,
    ).max(axis=1).astype(bool)
    snapback = snapback | fwd_reclaim

    volume_ok = pd.Series(True, index=df.index)
    if volume_mult is not None and volume_col in df.columns:
        vol_mean = df[volume_col].rolling(drawdown_lookback, min_periods=5).mean()
        volume_ok = df[volume_col] > (vol_mean * volume_mult)
        volume_ok = volume_ok.fillna(False)

    capitulation_pivot = (is_extreme_dd & local_low & snapback & volume_ok).fillna(False)
    return capitulation_pivot


# ----------------------------------------------------------------------
# 2) ATR + Pivot-based swing labels (The best scheme)
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
    *,
    enable_capitulation: bool = True,
    drawdown_lookback: int = 40,
    drawdown_thresh: float = -0.05,  # e.g., -5% or -10% drop
    local_window: int = 4,           # lookback for local min
    snapback_bars: int = 2,
    volume_mult: float | None = 1.5,
) -> pd.DataFrame:
    """
    ATR-based swing labeling with "Capitulation" logic to catch deep bottoms.
    """

    high = df[high_col].to_numpy(dtype=float)
    low = df[low_col].to_numpy(dtype=float)
    close = df[close_col].to_numpy(dtype=float)
    
    # Optional: Volume check if column exists
    if "volume" in df.columns and volume_mult is not None:
        vol = df["volume"].to_numpy(dtype=float)
        # Simple rolling avg volume for spike detection
        vol_ma = df["volume"].rolling(20).mean().to_numpy(dtype=float)
    else:
        vol = None
        vol_ma = None

    pivot_up = df[pivot_up_col].to_numpy(dtype=int)
    pivot_down = df[pivot_down_col].to_numpy(dtype=int)

    n = len(df)

    # --- ATR ---
    df["atr"] = ta.atr(df[high_col], df[low_col], df[close_col], length=atr_length)
    atr = df["atr"].to_numpy(dtype=float)

    # --- 1. Drawdown Calculation (New Factor) ---
    # Rolling Max High to define "Peak"
    rolling_peak = df[high_col].rolling(drawdown_lookback, min_periods=1).max()
    # Drawdown %
    dd_series = (df[close_col] - rolling_peak) / rolling_peak
    is_in_drawdown = (dd_series < drawdown_thresh).to_numpy()

    # --- outputs ---
    labels = np.zeros(n, dtype=float)          # -1, 0, +1
    entry_price = np.full(n, np.nan)
    exit_price = np.full(n, np.nan)
    holding_bars = np.full(n, np.nan)
    realized_ret = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        # ============================================================
        # LONG LOGIC: Standard Pivot OR Capitulation Catch
        # ============================================================
        is_pivot_long = (pivot_down[i] == 1)
        
        # Check Capitulation (The "Added Factor")
        is_capitulation_long = False
        if enable_capitulation and is_in_drawdown[i]:
            # 1. Are we at a local low relative to recent history?
            #    (Checks if current low is lowest in last 'local_window' bars)
            start_idx = max(0, i - local_window)
            if low[i] == np.min(low[start_idx : i + 1]):
                is_capitulation_long = True
                
                # Optional: Volume Spike Filter
                if vol is not None and vol_ma is not None and not np.isnan(vol_ma[i]):
                    if vol[i] < (vol_ma[i] * volume_mult):
                        is_capitulation_long = False

        if is_pivot_long or is_capitulation_long:
            ep = close[i]
            
            # --- INTELLIGENT STOP LOSS ---
            # If this is a capitulation trade (high volatility/fear), 
            # we widen the stop to prevent being shaken out by wicks.
            current_sl_mult = sl_mult * 2.0 if is_capitulation_long else sl_mult
            
            tp = ep + tp_mult * atr[i]
            sl = ep - current_sl_mult * atr[i]

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
                realized_ret[i] = (hit_exit / ep - 1.0)
            
            # If pivot failed (stopped out) but capitulation (wider stop) would have won,
            # you might want to force the capitulation logic. 
            # Here, we prioritize the WIN (1) if both triggered.
            elif labels[i] == 0 and hit_label == 1:
                labels[i] = 1
                exit_price[i] = hit_exit
                holding_bars[i] = hit_bars
                realized_ret[i] = (hit_exit / ep - 1.0)


        # ============================================================
        # SHORT LOGIC (Standard)
        # ============================================================
        elif pivot_up[i] == 1:
            ep = close[i]
            tp = ep - tp_mult * atr[i]   # profit target BELOW
            sl = ep + sl_mult * atr[i]   # stop ABOVE

            entry_price[i] = ep

            hit_label = 0
            hit_exit = ep
            hit_bars = 0

            for j in range(i + 1, min(i + 1 + max_holding, n)):
                if high[j] >= sl:
                    hit_label = 0        # stopped out
                    hit_exit = sl
                    hit_bars = j - i
                    break
                if low[j] <= tp:
                    hit_label = -1       # good short
                    hit_exit = tp
                    hit_bars = j - i
                    break

            labels[i] = hit_label
            exit_price[i] = hit_exit
            holding_bars[i] = hit_bars
            realized_ret[i] = (hit_exit / ep - 1.0)

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
# 3) Forward N-bar return sign (slope-based label)
# ----------------------------------------------------------------------
def add_forward_return_label(
    df: pd.DataFrame,
    horizon: int = 10,
    close_col: str = "close",
    pct_threshold: float = 0.03,
    label_col: str = "fwd_ret_label",
    ret_col: str | None = None,
) -> pd.DataFrame:
    """
    Label based on the sign of the N-bar forward return.

      r_t = (close[t + horizon] / close[t]) - 1

    If pct_threshold == 0:
        label = sign(r_t): +1, -1, or 0 if exactly 0 or NaN
    Else:
        label = +1 if r_t >= +pct_threshold
                -1 if r_t <= -pct_threshold
                 0 otherwise

    Parameters
    ----------
    df : DataFrame
    horizon : int
        Number of bars into the future for the return.
    close_col : str
    pct_threshold : float
        Minimum absolute return to count as +1/-1.
    label_col : str
    ret_col : str | None
        Optional column name to store the forward return. If None, uses
        f"{close_col}_fwd_ret_{horizon}".

    Returns
    -------
    DataFrame
        df with new columns: ret_col (float) and label_col (Int64).
    """
    if ret_col is None:
        ret_col = f"{close_col}_fwd_ret_{horizon}"

    future_price = df[close_col].shift(-horizon)
    fwd_ret = (future_price / df[close_col]) - 1.0
    df[ret_col] = fwd_ret

    labels = np.zeros(len(df), dtype=float)

    if pct_threshold <= 0:
        labels[fwd_ret > 0] = 1.0
        labels[fwd_ret < 0] = -1.0
    else:
        labels[fwd_ret >= pct_threshold] = 1.0
        labels[fwd_ret <= -pct_threshold] = -1.0

    labels[pd.isna(future_price.to_numpy())] = np.nan

    df[label_col] = pd.Series(labels, index=df.index).astype("Int64")
    return df


# ----------------------------------------------------------------------
# 4) Triple Barrier label (pure price-based, no pivots required)
# ----------------------------------------------------------------------
def add_triple_barrier_labels(
    df: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    up_pct: float = 0.03,
    down_pct: float = 0.02,
    max_holding: int = 20,
    base_label_col: str = "tb_label",
) -> pd.DataFrame:
    """
    Triple barrier label at each bar.

    For each index i:
        entry = close[i]
        upper = entry * (1 + up_pct)
        lower = entry * (1 - down_pct)

        Look ahead up to max_holding bars:
          - if high[j] >= upper before low[j] <= lower -> +1
          - if low[j] <= lower before high[j] >= upper -> -1
          - if neither hit -> 0

    Adds columns:
        base_label_col       - {-1, 0, +1}
        tb_entry_price
        tb_exit_price
        tb_holding_bars
        tb_realized_return
    """
    close = df[close_col].to_numpy(dtype=float)
    high = df[high_col].to_numpy(dtype=float)
    low = df[low_col].to_numpy(dtype=float)

    n = len(df)

    labels = np.zeros(n, dtype=float)
    entry_price = np.full(n, np.nan)
    exit_price = np.full(n, np.nan)
    holding_bars = np.full(n, np.nan)
    realized_ret = np.full(n, np.nan)

    for i in range(n):
        ep = close[i]
        if np.isnan(ep):
            continue

        upper = ep * (1.0 + up_pct)
        lower = ep * (1.0 - down_pct)

        entry_price[i] = ep

        hit_label = 0.0
        hit_exit = ep
        hit_bars = 0

        for j in range(i + 1, min(i + 1 + max_holding, n)):
            # check barriers
            hit_upper = high[j] >= upper
            hit_lower = low[j] <= lower

            if hit_upper and not hit_lower:
                hit_label = 1.0
                hit_exit = upper
                hit_bars = j - i
                break
            elif hit_lower and not hit_upper:
                hit_label = -1.0
                hit_exit = lower
                hit_bars = j - i
                break
            elif hit_upper and hit_lower:
                # tie-breaker: you can choose a rule; here we go neutral
                hit_label = 0.0
                hit_exit = ep
                hit_bars = j - i
                break

        labels[i] = hit_label
        exit_price[i] = hit_exit
        holding_bars[i] = hit_bars
        realized_ret[i] = (hit_exit / ep - 1.0) if ep != 0 else np.nan

    df[base_label_col] = labels
    df["tb_entry_price"] = entry_price
    df["tb_exit_price"] = exit_price
    df["tb_holding_bars"] = holding_bars
    df["tb_realized_return"] = realized_ret

    return df


# ----------------------------------------------------------------------
# 5) ZigZag leg labels (requires an existing zigzag series)
# ----------------------------------------------------------------------
def add_zigzag_leg_labels(
    df: pd.DataFrame,
    zz_col: str | None = None,
    label_col: str = "zz_leg_label",
    percent: float = 1.0,
    legs: int = 10,
) -> pd.DataFrame:
    """
    Creates a zigzag pivot series (if zz_col not provided/found) using pandas_ta
    and labels each bar with the current zigzag leg direction (+1/-1/0).

    percent lower => more sensitive => more pivots.
    """

    # 1) choose zigzag pivot series
    zz = None
    if zz_col is not None and zz_col in df.columns:
        zz = df[zz_col]
    else:
        # Try a known pandas_ta generated column if it exists
        candidate = f"ZIGZAGs_{percent:.1f}%_{legs}"
        if candidate in df.columns:
            zz = df[candidate]
        else:
            # Compute explicitly with chosen sensitivity
            zz_tmp = ta.zigzag(df["high"], df["low"], df["close"], percent=percent, legs=legs)
            zz = zz_tmp.iloc[:, 0] if isinstance(zz_tmp, pd.DataFrame) else zz_tmp

    zz_vals = zz.to_numpy(dtype=float)
    n = len(zz_vals)

    labels = np.zeros(n, dtype=float)

    pivot_idx = np.where(~np.isnan(zz_vals))[0]
    if len(pivot_idx) < 2:
        df[label_col] = pd.Series(labels, index=df.index).astype("Int64")
        return df

    pivot_vals = zz_vals[pivot_idx]

    for k in range(len(pivot_idx) - 1):
        i0 = pivot_idx[k]
        i1 = pivot_idx[k + 1]
        v0 = pivot_vals[k]
        v1 = pivot_vals[k + 1]

        if np.isnan(v0) or np.isnan(v1):
            continue

        leg_dir = 1.0 if v1 > v0 else (-1.0 if v1 < v0 else 0.0)
        labels[i0:i1] = leg_dir

    df[label_col] = pd.Series(labels, index=df.index).astype("Int64")
    return df

# ----------------------------------------------------------------------
# Helpers: handle multiple zigzag variants and visualize
# ----------------------------------------------------------------------
def add_multiple_zigzag_leg_labels(
    df: pd.DataFrame,
    zigzag_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Add leg labels for all provided zigzag columns.

    Default zigzag_cols tries the pandas-ta defaults:
      ["ZIGZAGs_5.0%_10", "ZIGZAGv_5.0%_10", "ZIGZAGd_5.0%_10"]
    """
    if zigzag_cols is None:
        zigzag_cols = ["ZIGZAGs_5.0%_10", "ZIGZAGv_5.0%_10", "ZIGZAGd_5.0%_10"]

    for col in zigzag_cols:
        if col not in df.columns:
            continue
        df = add_zigzag_leg_labels(df, zz_col=col, label_col=f"{col}_leg_label")
    return df


def plot_zigzags_vs_close(
    df: pd.DataFrame,
    zigzag_cols: list[str] | None = None,
    close_col: str = "close",
    title: str = "Close vs ZigZag pivots",
):
    """
    Plot close price with zigzag pivot series overlaid.
    Returns (fig, ax).
    """
    if zigzag_cols is None:
        zigzag_cols = ["ZIGZAGs_5.0%_10", "ZIGZAGv_5.0%_10", "ZIGZAGd_5.0%_10"]
    zigzag_cols = [c for c in zigzag_cols if c in df.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df[close_col], label=close_col, color="black", linewidth=1.2)

    for col in zigzag_cols:
        series = df[col]
        ax.plot(df.index, series, label=col, linewidth=1.0, alpha=0.8)
        # mark pivots as entry points
        pivots = series.dropna()
        ax.scatter(pivots.index, pivots.values, s=12, alpha=0.9)

    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig, ax

def add_atr_leg_segmentation_labels(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_length: int = 14,
    reversal_atr: float = 2.0,
    start_atr: float | None = None,
    use_high_low_extremes: bool = True,
    label_col: str = "atr_leg_label",
    pivot_price_col: str = "atr_leg_pivot_price",
    pivot_type_col: str = "atr_leg_pivot_type",
) -> pd.DataFrame:
    """
    ATR-based leg segmentation (swing segmentation), NOT TP/SL outcome labeling.

    Concept:
      - Track a current trend (up/down/unknown).
      - Track the extreme price during that trend (highest high in uptrend, lowest low in downtrend).
      - A new pivot is confirmed when price reverses from the extreme by >= reversal_atr * ATR.
      - Label bars between pivots as +1 for up legs, -1 for down legs (0 if unknown early).

    Outputs:
      - label_col: Int64 {-1,0,+1} leg direction per bar
      - pivot_price_col: pivot price at pivot index (NaN otherwise)
      - pivot_type_col: +1 at swing LOW pivots, -1 at swing HIGH pivots (NaN otherwise)
      - Also adds df["atr"] if not present

    Notes:
      - This uses future info to confirm pivots (like zigzag). That’s fine for LABELS.
      - Choose reversal_atr ~ 1.0–3.0 depending on how many swings you want.
      - start_atr controls how quickly the first trend is established; default = reversal_atr.
    """

    if start_atr is None:
        start_atr = reversal_atr

    # Compute ATR if missing
    if "atr" not in df.columns:
        df["atr"] = ta.atr(df[high_col], df[low_col], df[close_col], length=atr_length)

    atr = df["atr"].to_numpy(dtype=float)
    high = df[high_col].to_numpy(dtype=float)
    low = df[low_col].to_numpy(dtype=float)
    close = df[close_col].to_numpy(dtype=float)

    n = len(df)
    leg_label = np.zeros(n, dtype=float)
    pivot_price = np.full(n, np.nan)
    pivot_type = np.full(n, np.nan)

    # Trend state: 0 unknown, +1 up, -1 down
    trend = 0

    # Reference pivot index/price (last confirmed pivot)
    last_pivot_idx = 0
    last_pivot_price = close[0]

    # Track current extreme in the active trend
    extreme_idx = 0
    extreme_price = close[0]

    def get_up_extreme_price(i: int) -> float:
        return high[i] if use_high_low_extremes else close[i]

    def get_dn_extreme_price(i: int) -> float:
        return low[i] if use_high_low_extremes else close[i]

    # Initialize with first non-NaN close
    first_valid = np.where(~np.isnan(close))[0]
    if len(first_valid) == 0:
        df[label_col] = pd.Series(leg_label, index=df.index).astype("Int64")
        df[pivot_price_col] = pivot_price
        df[pivot_type_col] = pivot_type
        return df

    start_i = int(first_valid[0])
    last_pivot_idx = start_i
    last_pivot_price = close[start_i]
    extreme_idx = start_i
    extreme_price = close[start_i]

    # Walk forward
    for i in range(start_i + 1, n):
        if np.isnan(close[i]) or np.isnan(atr[i]) or atr[i] == 0:
            continue

        thresh_start = start_atr * atr[i]
        thresh_rev = reversal_atr * atr[i]

        if trend == 0:
            # Establish initial direction when we move enough from the starting pivot
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
            # Update extreme (highest)
            candidate = get_up_extreme_price(i)
            if candidate >= extreme_price:
                extreme_price = candidate
                extreme_idx = i

            # Check reversal down from extreme
            if close[i] <= extreme_price - thresh_rev:
                # Confirm swing HIGH pivot at extreme_idx
                pivot_price[extreme_idx] = extreme_price
                pivot_type[extreme_idx] = -1.0  # swing HIGH

                # Label bars from last pivot to this pivot as up leg
                leg_label[last_pivot_idx:extreme_idx] = 1.0

                # Start new down leg from that pivot
                last_pivot_idx = extreme_idx
                last_pivot_price = extreme_price
                trend = -1

                # Initialize extreme for downtrend at current bar
                extreme_idx = i
                extreme_price = get_dn_extreme_price(i)

        elif trend == -1:
            # Update extreme (lowest)
            candidate = get_dn_extreme_price(i)
            if candidate <= extreme_price:
                extreme_price = candidate
                extreme_idx = i

            # Check reversal up from extreme
            if close[i] >= extreme_price + thresh_rev:
                # Confirm swing LOW pivot at extreme_idx
                pivot_price[extreme_idx] = extreme_price
                pivot_type[extreme_idx] = 1.0  # swing LOW

                # Label bars from last pivot to this pivot as down leg
                leg_label[last_pivot_idx:extreme_idx] = -1.0

                # Start new up leg from that pivot
                last_pivot_idx = extreme_idx
                last_pivot_price = extreme_price
                trend = 1

                # Initialize extreme for uptrend at current bar
                extreme_idx = i
                extreme_price = get_up_extreme_price(i)

    # After loop, label the tail with current trend (optional but usually helpful)
    if trend != 0 and last_pivot_idx < n:
        leg_label[last_pivot_idx:] = float(trend)

    df[label_col] = pd.Series(leg_label, index=df.index).astype("Int64")
    df[pivot_price_col] = pivot_price
    df[pivot_type_col] = pivot_type

    return df

# ----------------------------------------------------------------------
# Helper: run all label builders in one call
# ----------------------------------------------------------------------
def add_all_labels(
    df: pd.DataFrame,
    *,
    next_day_kwargs: dict | None = None,
    atr_pivot_kwargs: dict | None = None,
    fwd_ret_kwargs: dict | None = None,
    triple_barrier_kwargs: dict | None = None,
    zigzag_kwargs: dict | None = None,
    atr_leg_kwargs: dict | None = None,
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

    next_day_kwargs = next_day_kwargs or {}
    atr_pivot_kwargs = atr_pivot_kwargs or {}
    fwd_ret_kwargs = fwd_ret_kwargs or {}
    triple_barrier_kwargs = triple_barrier_kwargs or {}
    zigzag_kwargs = zigzag_kwargs or {}
    atr_leg_kwargs = atr_leg_kwargs or {}

    df = add_next_day_direction_label(df, **next_day_kwargs)
    df = add_atr_pivot_swing_labels(df, **atr_pivot_kwargs)
    df = add_forward_return_label(df, **fwd_ret_kwargs)
    df = add_triple_barrier_labels(df, **triple_barrier_kwargs)
    df = add_zigzag_leg_labels(df, **zigzag_kwargs)
    df = add_atr_leg_segmentation_labels(df, **atr_leg_kwargs)

    return df

def plot_zig_zag(df):
    df = add_multiple_zigzag_leg_labels(df)
    fig, ax = plot_zigzags_vs_close(df)
    plt.show()
    
def main():
    print('goodbye')

if __name__ == "__main__":
    main()
    
