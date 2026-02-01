import numpy as np
import pandas as pd
import pandas_ta as ta

from Features.label_generations import add_triple_barrier_labels_atr


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


# Triple barrier label moved to Features/label_generations.py (kept import for compatibility).
