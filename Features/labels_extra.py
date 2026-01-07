import numpy as np
import pandas as pd
import pandas_ta as ta


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


def add_pullback_entry_labels(
    df: pd.DataFrame,
    *,
    pivot_down_col: str = "pivot_down",
    pivot_up_col: str = "pivot_up",
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    atr_length: int = 14,
    dd_atr: float = 1.0,
    max_entries: int = 2,
    rebound_mode: str = "break_prev_high",
    label_col: str = "pullback_entry",
    rank_col: str = "pullback_entry_rank",
) -> pd.DataFrame:
    """
    Tag up to `max_entries` higher-low pullbacks inside each upswing (pivot_down -> next pivot_up).
    A pullback must be at least `dd_atr` ATRs off the running high of the upswing.
    If rebound_mode == "break_prev_high", the label is placed on the first reclaim bar after the dip;
    otherwise the dip bar itself is labeled.
    """
    if pivot_down_col not in df.columns or pivot_up_col not in df.columns:
        return df

    if atr_col not in df.columns:
        df[atr_col] = ta.atr(
            df[high_col], df[low_col], df[close_col], length=atr_length
        )

    labels = pd.Series(0, index=df.index, dtype="Int64")
    ranks = pd.Series(pd.NA, index=df.index, dtype="Int64")

    pivot_down_idx = df.index[df[pivot_down_col].fillna(0).astype(int) == 1]
    pivot_up_idx = df.index[df[pivot_up_col].fillna(0).astype(int) == 1]

    for start in pivot_down_idx:
        end_candidates = pivot_up_idx[pivot_up_idx > start]
        end = end_candidates[0] if len(end_candidates) > 0 else df.index[-1]

        seg = df.loc[start:end]
        if seg.empty:
            continue

        run_high = seg[high_col].cummax()
        atr_seg = seg[atr_col].replace(0, np.nan)
        drawdown_atr = (run_high - seg[close_col]) / atr_seg

        higher_low = seg[low_col] > df.at[start, low_col]
        candidates = seg[higher_low & (drawdown_atr >= dd_atr)]
        if candidates.empty:
            continue

        candidates = candidates.copy()
        candidates["_dd_atr"] = drawdown_atr.loc[candidates.index]
        top_dips = candidates.nlargest(max_entries, columns="_dd_atr")

        for rank, dip_idx in enumerate(top_dips.index, start=1):
            entry_idx = dip_idx
            if rebound_mode == "break_prev_high":
                post = df.loc[dip_idx:]
                trigger = post[post[close_col] > post[high_col].shift(1)]
                if not trigger.empty:
                    entry_idx = trigger.index[0]
            labels.loc[entry_idx] = 1
            ranks.loc[entry_idx] = rank

    df[label_col] = labels
    df[rank_col] = ranks
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
