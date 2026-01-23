from __future__ import annotations

from typing import Mapping

import pandas as pd

from Features.feature_sets.custom_indicators import (
    add_atr_swing_state_features,
    add_fractal_pivots,
    add_rsilg_fe_gauss,
    add_tmo,
    add_vmd_return_features,
)
from Features.feature_sets.pandas_ta_indicators import add_all_pandasta_indicators

NY_TZ = "America/New_York"
DEFAULT_TIMEFRAMES: dict[str, str] = {
    "5m": "5T",
    "15m": "15T",
    "60m": "60T",
}


def ensure_time_index(
    df: pd.DataFrame,
    *,
    tz: str | None = NY_TZ,
    assume_tz: str = "UTC",
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize(assume_tz)
    if tz is not None:
        idx = idx.tz_convert(tz)

    out = df.copy()
    out.index = idx
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    return out


def resample_ohlcv(
    df: pd.DataFrame,
    rule: str,
    *,
    label: str = "left",
    closed: str = "left",
) -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    resampled = df[required].resample(rule, label=label, closed=closed).agg(agg)
    return resampled.dropna(subset=["open", "high", "low", "close"])


def _add_standard_indicators(
    df: pd.DataFrame, *, include_custom: bool, verbose: bool
) -> pd.DataFrame:
    df = add_all_pandasta_indicators(df, verbose=verbose)
    if include_custom:
        df = add_tmo(df)
        df = add_rsilg_fe_gauss(df)
        df = add_fractal_pivots(df)
        df = add_atr_swing_state_features(df)
        df = add_vmd_return_features(df)
    return df


def _align_to_base_index(
    df: pd.DataFrame, base_index: pd.DatetimeIndex, shift_bars: int
) -> pd.DataFrame:
    aligned = df.shift(shift_bars) if shift_bars else df
    return aligned.reindex(base_index, method="ffill")


def add_multi_timeframe_features(
    df_1m: pd.DataFrame,
    *,
    timeframes: Mapping[str, str] = DEFAULT_TIMEFRAMES,
    tz: str | None = NY_TZ,
    include_custom: bool = True,
    verbose: bool = False,
    shift_bars: int = 1,
) -> pd.DataFrame:
    """
    Build higher-timeframe features and align them back to the 1-minute index.
    """
    df_1m = ensure_time_index(df_1m, tz=tz)
    base_index = df_1m.index

    frames = []
    for label, rule in timeframes.items():
        tf_df = resample_ohlcv(df_1m, rule)
        if tf_df.empty:
            continue
        tf_df = _add_standard_indicators(
            tf_df, include_custom=include_custom, verbose=verbose
        )
        tf_df = tf_df.add_suffix(f"__{label}")
        aligned = _align_to_base_index(tf_df, base_index, shift_bars)
        aligned = aligned.dropna(axis=1, how="all")
        frames.append(aligned)

    if not frames:
        return pd.DataFrame(index=base_index)
    return pd.concat(frames, axis=1)
