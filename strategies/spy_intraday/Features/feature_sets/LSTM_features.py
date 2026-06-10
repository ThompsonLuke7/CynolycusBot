"""
Core sequence features (per 15-min bar) for the LSTM window.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import pandas_ta as ta

from strategies.spy_intraday.Features.feature_sets.pandas_ta_indicators import prepare_ohlcv_columns
from strategies.spy_intraday.Features.feature_sets.custom_indicators import add_atr_swing_state_features

EPS = 1e-12
REQUIRED_OHLCV = ("open", "high", "low", "close", "volume")


def _require_ohlcv(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_OHLCV if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing OHLCV columns: {missing_str}")


def _series_from_ta(
    result: pd.Series | pd.DataFrame, *, prefix: str | None = None
) -> pd.Series:
    if isinstance(result, pd.Series):
        return result
    if isinstance(result, pd.DataFrame):
        if prefix:
            for col in result.columns:
                if col.startswith(prefix):
                    return result[col]
        return result.iloc[:, 0]
    raise TypeError("Unexpected pandas_ta return type")


def _zscore(series: pd.Series, length: int) -> pd.Series:
    mean = series.rolling(length).mean()
    std = series.rolling(length).std()
    return (series - mean) / std.replace(0, np.nan)


def _parse_hhmm(value: str) -> tuple[int, int]:
    hour_str, minute_str = value.split(":")
    return int(hour_str), int(minute_str)


# Returns (stationary)
def add_return_features(
    df: pd.DataFrame,
    *,
    periods: Iterable[int] = (1, 2, 4, 8, 16),
    close_col: str = "close",
) -> pd.DataFrame:
    close = df[close_col].astype(float)
    for period in periods:
        df[f"ret_{period}"] = np.log(close / close.shift(period))
    return df


# Candle shape normalized
def add_candle_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    df["body_pct"] = (close - open_) / close.replace(0, np.nan)
    df["range_pct"] = (high - low) / close.replace(0, np.nan)
    df["upper_wick_pct"] = (high - np.maximum(open_, close)) / close.replace(
        0, np.nan
    )
    df["lower_wick_pct"] = (np.minimum(open_, close) - low) / close.replace(
        0, np.nan
    )
    range_span = (high - low).replace(0, np.nan)
    df["close_pos_in_range"] = (close - low) / range_span
    return df


# Volatility / realized vol
def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)

    atr = _series_from_ta(df.ta.atr(length=14, append=False))
    df["atr_14_pct"] = atr / close.replace(0, np.nan)

    tr = _series_from_ta(df.ta.true_range(append=False))
    df["tr_pct"] = tr / close.replace(0, np.nan)

    ret_1 = df.get("ret_1")
    if ret_1 is None:
        ret_1 = np.log(close / close.shift(1))
        df["ret_1"] = ret_1

    df["rv_10"] = ret_1.rolling(10).std()
    df["rv_20"] = ret_1.rolling(20).std()
    df["rv_ratio_10_20"] = df["rv_10"] / df["rv_20"].replace(0, np.nan)
    return df


# Trend + slope
def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)

    ema20 = _series_from_ta(df.ta.ema(length=20, append=False))
    ema50 = _series_from_ta(df.ta.ema(length=50, append=False))

    df["ema_20_ratio"] = close / ema20.replace(0, np.nan)
    df["ema_50_ratio"] = close / ema50.replace(0, np.nan)
    df["ema_20_50_ratio"] = ema20 / ema50.replace(0, np.nan)
    df["ema20_slope_pct"] = ema20.pct_change()
    df["ema50_slope_pct"] = ema50.pct_change()

    adx_df = df.ta.adx(length=14, append=False)
    df["adx_14"] = _series_from_ta(adx_df, prefix="ADX")
    return df


# Mean reversion / z-scores
def add_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    ret_1 = df.get("ret_1")
    if ret_1 is None:
        ret_1 = np.log(close / close.shift(1))
        df["ret_1"] = ret_1

    df["close_z_20"] = _zscore(close, 20)
    df["close_z_50"] = _zscore(close, 50)
    df["ret_z_20"] = _zscore(ret_1, 20)
    return df


# VWAP-based
def add_vwap_features(df: pd.DataFrame, *, anchor: str = "D") -> pd.DataFrame:
    close = df["close"].astype(float)
    vwap = _series_from_ta(df.ta.vwap(append=False, anchor=anchor))

    df["vwap_dist_pct"] = (close - vwap) / close.replace(0, np.nan)
    df["vwap_slope_pct"] = vwap.pct_change()
    df["above_vwap_flag"] = (close > vwap).astype(int)
    return df


# Volume (normalized)
def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    df["vol_z_20"] = _zscore(volume, 20)
    vol_sma_20 = _series_from_ta(df.ta.sma(close=volume, length=20, append=False))
    df["vol_ratio_20"] = volume / vol_sma_20.replace(0, np.nan)

    dollar_vol = close * volume
    df["dollar_vol_z_20"] = _zscore(dollar_vol, 20)
    return df


# Compression / expansion
def add_compression_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)

    bb = df.ta.bbands(length=20, append=False)
    bb_upper = _series_from_ta(bb, prefix="BBU")
    bb_lower = _series_from_ta(bb, prefix="BBL")

    df["bb_width_20_pct"] = (bb_upper - bb_lower) / close.replace(0, np.nan)
    bb_span = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_pos_20"] = (close - bb_lower) / bb_span

    kc = df.ta.kc(length=20, append=False)
    kc_upper = _series_from_ta(kc, prefix="KCU")
    kc_lower = _series_from_ta(kc, prefix="KCL")

    df["keltner_width_pct"] = (kc_upper - kc_lower) / close.replace(0, np.nan)
    df["squeeze_flag"] = (
        (bb_upper < kc_upper) & (bb_lower > kc_lower)
    ).astype(int)
    return df


# Momentum oscillators
def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    rsi = _series_from_ta(df.ta.rsi(length=14, append=False))
    df["rsi_14"] = rsi

    stoch = df.ta.stoch(k=14, d=3, append=False)
    df["stoch_k_14_3"] = _series_from_ta(stoch, prefix="STOCHk")

    roc = _series_from_ta(df.ta.roc(length=10, append=False))
    df["roc_10"] = roc / 100.0
    return df


# Time features (intraday context)
def add_time_features(
    df: pd.DataFrame,
    *,
    tz: str | None = "America/New_York",
    session_open: str = "09:30",
    session_close: str = "16:00",
    add_dow_onehot: bool = True,
    add_dow_sin_cos: bool = False,
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Time features require a DatetimeIndex.")

    idx = df.index
    if tz:
        if idx.tz is None:
            idx = idx.tz_localize(tz)
        else:
            idx = idx.tz_convert(tz)

    open_hour, open_minute = _parse_hhmm(session_open)
    close_hour, close_minute = _parse_hhmm(session_close)
    open_minutes = open_hour * 60 + open_minute
    close_minutes = close_hour * 60 + close_minute
    session_minutes = max(1, close_minutes - open_minutes)

    minutes = idx.hour * 60 + idx.minute + idx.second / 60.0
    minutes = np.asarray(minutes, dtype=float)
    minutes_since_open = minutes - open_minutes

    minutes_from_open = np.clip(
        minutes_since_open / session_minutes, 0.0, 1.0
    )
    df["minutes_from_open"] = minutes_from_open
    df["sin_time"] = np.sin(2 * np.pi * minutes_from_open)
    df["cos_time"] = np.cos(2 * np.pi * minutes_from_open)

    if add_dow_onehot:
        dow = idx.dayofweek
        dow_names = ("mon", "tue", "wed", "thu", "fri")
        for i, name in enumerate(dow_names):
            df[f"dow_{name}"] = (dow == i).astype(int)

    if add_dow_sin_cos:
        dow = idx.dayofweek.astype(float)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    df["is_opening_30m_flag"] = (
        (minutes_since_open >= 0) & (minutes_since_open <= 30)
    ).astype(int)
    df["is_power_hour_flag"] = (
        (minutes_since_open >= (session_minutes - 60))
        & (minutes_since_open <= session_minutes)
    ).astype(int)
    return df


def add_leg_structure_features(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    swing_prefix: str = "atr_swing",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Causal leg structure features based on ATR swing flips.

    No leakage: uses only past flips and current close.
    """
    flip_col = f"{swing_prefix}_flip"
    bars_col = f"{swing_prefix}_bars_since_flip"

    if flip_col not in df.columns or bars_col not in df.columns:
        df = add_atr_swing_state_features(df, prefix=swing_prefix)

    flip = df[flip_col].fillna(0).astype(int)
    df["swing_leg_count"] = flip.cumsum().astype(int)
    df["time_since_pivot"] = df[bars_col]
    return df


def add_lstm_features(
    df: pd.DataFrame,
    *,
    include_time_features: bool = True,
    vwap_anchor: str = "D",
    time_kwargs: dict | None = None,
) -> pd.DataFrame:
    df = prepare_ohlcv_columns(df)
    _require_ohlcv(df)

    df = add_return_features(df)
    df = add_candle_shape_features(df)
    df = add_volatility_features(df)
    df = add_trend_features(df)
    df = add_mean_reversion_features(df)
    df = add_vwap_features(df, anchor=vwap_anchor)
    df = add_volume_features(df)
    df = add_compression_features(df)
    df = add_momentum_features(df)
    df = add_leg_structure_features(df)

    if include_time_features:
        df = add_time_features(df, **(time_kwargs or {}))

    return df
