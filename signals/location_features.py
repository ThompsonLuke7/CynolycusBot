from __future__ import annotations

import numpy as np
import pandas as pd


def add_liquidity_zone_features(
    df: pd.DataFrame,
    *,
    prefix: str = "",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    atr_col: str | None = None,
    lookback: int = 78,
    swing_window: int = 20,
    zone_width_pct: float = 0.0015,
    wick_threshold: float = 0.45,
    volume_window: int = 50,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Add causal support/resistance zone and reaction features.

    The nearest zones are built from rolling swing highs/lows plus high-volume
    wick rejections. Current-bar OHLC is allowed because these exports train on
    bar-close decisions.
    """
    required = [open_col, high_col, low_col, close_col, volume_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"missing OHLCV columns for liquidity zones: {missing}")

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("liquidity zone features require a DatetimeIndex")
    out = out.sort_index()

    open_ = pd.to_numeric(out[open_col], errors="coerce").astype(float)
    high = pd.to_numeric(out[high_col], errors="coerce").astype(float)
    low = pd.to_numeric(out[low_col], errors="coerce").astype(float)
    close = pd.to_numeric(out[close_col], errors="coerce").astype(float)
    volume = pd.to_numeric(out[volume_col], errors="coerce").astype(float)
    atr = _atr(out, high, low, close, atr_col)

    candle_range = (high - low).replace(0, np.nan)
    upper_wick = (high - np.maximum(open_, close)).clip(lower=0)
    lower_wick = (np.minimum(open_, close) - low).clip(lower=0)
    upper_wick_ratio = upper_wick / candle_range
    lower_wick_ratio = lower_wick / candle_range
    close_pos = (close - low) / candle_range
    volume_rel = volume / volume.rolling(volume_window, min_periods=max(5, volume_window // 4)).mean().replace(0, np.nan)

    swing_low = low <= low.rolling(swing_window, min_periods=2).min()
    swing_high = high >= high.rolling(swing_window, min_periods=2).max()
    support_rejection = (lower_wick_ratio >= wick_threshold) & (close_pos >= 0.55) & (volume_rel >= 1.15)
    resistance_rejection = (upper_wick_ratio >= wick_threshold) & (close_pos <= 0.45) & (volume_rel >= 1.15)
    support_candidates = low.where(swing_low | support_rejection)
    resistance_candidates = high.where(swing_high | resistance_rejection)

    support_level = np.full(len(out), np.nan)
    resistance_level = np.full(len(out), np.nan)
    support_width = np.full(len(out), np.nan)
    resistance_width = np.full(len(out), np.nan)
    support_touches = np.full(len(out), np.nan)
    resistance_touches = np.full(len(out), np.nan)
    days_since_support = np.full(len(out), np.nan)
    days_since_resistance = np.full(len(out), np.nan)
    support_strength = np.full(len(out), np.nan)
    resistance_strength = np.full(len(out), np.nan)
    failed_breakout = np.zeros(len(out), dtype=float)
    failed_breakdown = np.zeros(len(out), dtype=float)

    idx = pd.DatetimeIndex(out.index)
    close_arr = close.to_numpy(float)
    high_arr = high.to_numpy(float)
    low_arr = low.to_numpy(float)
    atr_arr = atr.to_numpy(float)
    sup_arr = support_candidates.to_numpy(float)
    res_arr = resistance_candidates.to_numpy(float)
    vol_rel_arr = volume_rel.fillna(1.0).to_numpy(float)

    for i in range(len(out)):
        c = close_arr[i]
        if not np.isfinite(c) or c <= 0:
            continue
        start = max(0, i - lookback + 1)
        width = _zone_width(c, atr_arr[i], zone_width_pct)
        support_width[i] = width / c
        resistance_width[i] = width / c

        recent_sup = sup_arr[start : i + 1]
        valid_sup = recent_sup[np.isfinite(recent_sup) & (recent_sup <= c + width)]
        if valid_sup.size:
            level = float(np.max(valid_sup))
        else:
            lows = low_arr[start : i + 1]
            finite = lows[np.isfinite(lows)]
            level = float(np.min(finite)) if finite.size else np.nan
        support_level[i] = level
        if np.isfinite(level):
            touched = (low_arr[start : i + 1] <= level + width) & (high_arr[start : i + 1] >= level - width)
            support_touches[i] = float(np.sum(touched))
            touch_idx = np.where(touched)[0]
            if touch_idx.size:
                last = start + int(touch_idx[-1])
                days_since_support[i] = max(0.0, (idx[i] - idx[last]).total_seconds() / 86400.0)
            support_strength[i] = min(10.0, support_touches[i]) * float(np.nanmean(vol_rel_arr[start : i + 1]))

        recent_res = res_arr[start : i + 1]
        valid_res = recent_res[np.isfinite(recent_res) & (recent_res >= c - width)]
        if valid_res.size:
            level = float(np.min(valid_res))
        else:
            highs = high_arr[start : i + 1]
            finite = highs[np.isfinite(highs)]
            level = float(np.max(finite)) if finite.size else np.nan
        resistance_level[i] = level
        if np.isfinite(level):
            touched = (low_arr[start : i + 1] <= level + width) & (high_arr[start : i + 1] >= level - width)
            resistance_touches[i] = float(np.sum(touched))
            touch_idx = np.where(touched)[0]
            if touch_idx.size:
                last = start + int(touch_idx[-1])
                days_since_resistance[i] = max(0.0, (idx[i] - idx[last]).total_seconds() / 86400.0)
            resistance_strength[i] = min(10.0, resistance_touches[i]) * float(np.nanmean(vol_rel_arr[start : i + 1]))

        if np.isfinite(resistance_level[i]):
            failed_breakout[i] = float(high_arr[i] > resistance_level[i] + width and close_arr[i] < resistance_level[i])
        if np.isfinite(support_level[i]):
            failed_breakdown[i] = float(low_arr[i] < support_level[i] - width and close_arr[i] > support_level[i])

    p = prefix
    out[f"{p}nearest_support_zone"] = support_level
    out[f"{p}nearest_resistance_zone"] = resistance_level
    out[f"{p}distance_to_nearest_support_zone"] = (close - out[f"{p}nearest_support_zone"]) / close.replace(0, np.nan)
    out[f"{p}distance_to_nearest_resistance_zone"] = (out[f"{p}nearest_resistance_zone"] - close) / close.replace(0, np.nan)
    out[f"{p}inside_support_zone"] = (
        out[f"{p}distance_to_nearest_support_zone"].abs() <= support_width
    ).astype(float)
    out[f"{p}inside_resistance_zone"] = (
        out[f"{p}distance_to_nearest_resistance_zone"].abs() <= resistance_width
    ).astype(float)
    out[f"{p}support_zone_width_pct"] = support_width
    out[f"{p}resistance_zone_width_pct"] = resistance_width
    out[f"{p}support_touch_count"] = support_touches
    out[f"{p}resistance_touch_count"] = resistance_touches
    out[f"{p}days_since_support_touch"] = days_since_support
    out[f"{p}days_since_resistance_touch"] = days_since_resistance
    out[f"{p}support_zone_strength"] = support_strength
    out[f"{p}resistance_zone_strength"] = resistance_strength
    out[f"{p}wick_ratio"] = np.maximum(upper_wick_ratio, lower_wick_ratio)
    out[f"{p}close_position_in_candle"] = close_pos
    out[f"{p}rejection_strength"] = np.maximum(
        lower_wick_ratio * out[f"{p}inside_support_zone"],
        upper_wick_ratio * out[f"{p}inside_resistance_zone"],
    )
    out[f"{p}volume_on_rejection"] = volume_rel * (out[f"{p}rejection_strength"] > 0).astype(float)
    out[f"{p}failed_breakout_count"] = pd.Series(failed_breakout, index=out.index).rolling(20, min_periods=1).sum()
    out[f"{p}failed_breakdown_count"] = pd.Series(failed_breakdown, index=out.index).rolling(20, min_periods=1).sum()
    out[f"{p}breakout_proximity"] = (1.0 - (out[f"{p}distance_to_nearest_resistance_zone"] / (3.0 * out[f"{p}resistance_zone_width_pct"])).clip(0, 1)).clip(0, 1)
    out[f"{p}support_proximity"] = (1.0 - (out[f"{p}distance_to_nearest_support_zone"] / (3.0 * out[f"{p}support_zone_width_pct"])).clip(0, 1)).clip(0, 1)
    out[f"{p}resistance_proximity"] = out[f"{p}breakout_proximity"]

    feature_cols = [
        f"{p}distance_to_nearest_support_zone",
        f"{p}distance_to_nearest_resistance_zone",
        f"{p}inside_support_zone",
        f"{p}inside_resistance_zone",
        f"{p}support_zone_width_pct",
        f"{p}resistance_zone_width_pct",
        f"{p}support_touch_count",
        f"{p}resistance_touch_count",
        f"{p}days_since_support_touch",
        f"{p}days_since_resistance_touch",
        f"{p}support_zone_strength",
        f"{p}resistance_zone_strength",
        f"{p}wick_ratio",
        f"{p}close_position_in_candle",
        f"{p}rejection_strength",
        f"{p}volume_on_rejection",
        f"{p}failed_breakout_count",
        f"{p}failed_breakdown_count",
        f"{p}breakout_proximity",
        f"{p}support_proximity",
        f"{p}resistance_proximity",
    ]
    helper_cols = [f"{p}nearest_support_zone", f"{p}nearest_resistance_zone"]
    return out, feature_cols, helper_cols


def add_daily_high_low_gap_features(
    df: pd.DataFrame,
    *,
    prefix: str = "",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    open_col: str = "open",
) -> tuple[pd.DataFrame, list[str]]:
    """Add prior-day daily high/low and gap-location features to intraday bars."""
    missing = [c for c in [open_col, high_col, low_col, close_col] if c not in df.columns]
    if missing:
        raise KeyError(f"missing OHLC columns for daily location features: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("daily location features require a DatetimeIndex")

    out = df.copy().sort_index()
    session = out.index.tz_convert("America/New_York").normalize() if out.index.tz is not None else out.index.normalize()
    tmp = out[[open_col, high_col, low_col, close_col]].copy()
    tmp["_session"] = session
    daily = tmp.groupby("_session").agg(
        open=(open_col, "first"),
        high=(high_col, "max"),
        low=(low_col, "min"),
        close=(close_col, "last"),
    ).sort_index()

    close = pd.to_numeric(out[close_col], errors="coerce").astype(float)
    p = prefix
    for window, label in [(252, "52w"), (20, "recent_20d"), (60, "recent_60d")]:
        high_level = daily["high"].rolling(window, min_periods=5).max().shift(1)
        low_level = daily["low"].rolling(window, min_periods=5).min().shift(1)
        mapped_high = pd.Series(session, index=out.index).map(high_level)
        mapped_low = pd.Series(session, index=out.index).map(low_level)
        out[f"{p}distance_to_{label}_high"] = (mapped_high - close) / close.replace(0, np.nan)
        out[f"{p}distance_to_{label}_low"] = (close - mapped_low) / close.replace(0, np.nan)

    gap_up = daily["low"] > daily["high"].shift(1)
    gap_down = daily["high"] < daily["low"].shift(1)
    gap_mid = pd.Series(np.nan, index=daily.index, dtype=float)
    gap_mid.loc[gap_up] = (daily["low"].loc[gap_up] + daily["high"].shift(1).loc[gap_up]) / 2.0
    gap_mid.loc[gap_down] = (daily["high"].loc[gap_down] + daily["low"].shift(1).loc[gap_down]) / 2.0
    gap_fill = pd.Series(np.nan, index=daily.index, dtype=float)
    gap_fill.loc[gap_up] = (daily["low"].loc[gap_up] <= daily["high"].shift(1).loc[gap_up]).astype(float)
    gap_fill.loc[gap_down] = (daily["high"].loc[gap_down] >= daily["low"].shift(1).loc[gap_down]).astype(float)
    gap_fill_rate_20 = gap_fill.shift(1).rolling(20, min_periods=3).mean()
    gap_fill_rate_60 = gap_fill.shift(1).rolling(60, min_periods=5).mean()

    gap_above, gap_below = _nearest_gap_distances(daily, gap_mid)
    mapped_above = pd.Series(session, index=out.index).map(gap_above)
    mapped_below = pd.Series(session, index=out.index).map(gap_below)
    out[f"{p}distance_to_gap_above"] = (mapped_above - close) / close.replace(0, np.nan)
    out[f"{p}distance_to_gap_below"] = (close - mapped_below) / close.replace(0, np.nan)
    out[f"{p}gap_fill_rate_20d"] = pd.Series(session, index=out.index).map(gap_fill_rate_20)
    out[f"{p}gap_fill_rate_60d"] = pd.Series(session, index=out.index).map(gap_fill_rate_60)
    out[f"{p}breakout_proximity_daily"] = (1.0 - out[f"{p}distance_to_recent_20d_high"].clip(lower=0) / 0.03).clip(0, 1)
    out[f"{p}support_proximity_daily"] = (1.0 - out[f"{p}distance_to_recent_20d_low"].clip(lower=0) / 0.03).clip(0, 1)
    out[f"{p}resistance_proximity_daily"] = out[f"{p}breakout_proximity_daily"]

    feature_cols = [
        f"{p}distance_to_52w_high",
        f"{p}distance_to_52w_low",
        f"{p}distance_to_recent_20d_high",
        f"{p}distance_to_recent_20d_low",
        f"{p}distance_to_recent_60d_high",
        f"{p}distance_to_recent_60d_low",
        f"{p}distance_to_gap_above",
        f"{p}distance_to_gap_below",
        f"{p}gap_fill_rate_20d",
        f"{p}gap_fill_rate_60d",
        f"{p}breakout_proximity_daily",
        f"{p}support_proximity_daily",
        f"{p}resistance_proximity_daily",
    ]
    return out, feature_cols


def _atr(df: pd.DataFrame, high: pd.Series, low: pd.Series, close: pd.Series, atr_col: str | None) -> pd.Series:
    if atr_col and atr_col in df.columns:
        return pd.to_numeric(df[atr_col], errors="coerce").astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=1).mean()


def _zone_width(close: float, atr: float, zone_width_pct: float) -> float:
    pct_width = abs(close) * float(zone_width_pct)
    if np.isfinite(atr) and atr > 0:
        return max(pct_width, 0.25 * float(atr))
    return pct_width


def _nearest_gap_distances(daily: pd.DataFrame, gap_mid: pd.Series) -> tuple[pd.Series, pd.Series]:
    above = pd.Series(np.nan, index=daily.index, dtype=float)
    below = pd.Series(np.nan, index=daily.index, dtype=float)
    active: list[float] = []
    for i, (dt, row) in enumerate(daily.iterrows()):
        close = float(row["close"]) if np.isfinite(row["close"]) else np.nan
        if np.isfinite(close) and active:
            arr = np.array(active, dtype=float)
            up = arr[arr >= close]
            dn = arr[arr <= close]
            if up.size:
                above.loc[dt] = float(np.min(up))
            if dn.size:
                below.loc[dt] = float(np.max(dn))
        val = gap_mid.iloc[i]
        if np.isfinite(val):
            active.append(float(val))
            if len(active) > 100:
                active = active[-100:]
    return above, below
