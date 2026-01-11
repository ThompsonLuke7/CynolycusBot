from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def add_support_resistance_features(
    df: pd.DataFrame,
    *,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    local_windows: Iterable[int] = (6, 12, 24),
    pct_window: int | None = None,
    touch_window: int | None = None,
    touch_tolerance: float = 0.001,
    include_prior_day_vwap: bool = True,
    include_prior_week_vwap: bool = True,
    include_prior_month_levels: bool = False,
    include_market_profile: bool = False,
    round_levels: Iterable[int] = (5, 10),
    round_tolerance: float = 0.001,
    round_rejection_window: int | None = None,
    near_level_pct: float = 0.001,
    wick_ratio_threshold: float = 0.5,
    value_area_pct: float = 0.7,
    profile_bin_size: float = 0.25,
) -> pd.DataFrame:
    """
    Add support/resistance features based on distances, VWAP hierarchy,
    higher timeframe levels, round numbers, and wick rejection signals.

    Assumes df is indexed by a DatetimeIndex and contains OHLCV columns.
    """
    _require_columns(df, [open_col, high_col, low_col, close_col, volume_col])
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("support/resistance features require a DatetimeIndex")

    local_windows = tuple(int(w) for w in local_windows)
    if not local_windows:
        return df

    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)
    open_ = df[open_col].astype(float)
    volume = df[volume_col].astype(float)

    # --- Local structure (rolling highs/lows) ---
    for window in local_windows:
        rolling_high = high.rolling(window=window, min_periods=1).max()
        rolling_low = low.rolling(window=window, min_periods=1).min()
        df[f"dist_to_recent_high_{window}"] = rolling_high - close
        df[f"dist_to_recent_low_{window}"] = close - rolling_low

    pct_window = pct_window or max(local_windows)
    pct_high = high.rolling(window=pct_window, min_periods=1).max()
    pct_low = low.rolling(window=pct_window, min_periods=1).min()
    df["pct_from_recent_high"] = _safe_divide(close - pct_high, pct_high)
    df["pct_from_recent_low"] = _safe_divide(close - pct_low, pct_low)

    touch_window = touch_window or max(local_windows)
    df["recent_high_touch_count"] = _rolling_touch_count(
        high, touch_window, touch_tolerance, mode="high"
    )
    df["recent_low_touch_count"] = _rolling_touch_count(
        low, touch_window, touch_tolerance, mode="low"
    )

    # --- VWAP hierarchy ---
    session_key = df.index.normalize()
    session_series = pd.Series(session_key, index=df.index)
    vwap, vwap_std = _compute_session_vwap(high, low, close, volume, session_series)
    vwap_upper_1 = vwap + vwap_std
    vwap_lower_1 = vwap - vwap_std
    vwap_upper_2 = vwap + 2.0 * vwap_std
    vwap_lower_2 = vwap - 2.0 * vwap_std

    df["dist_to_vwap"] = close - vwap
    df["dist_to_vwap_upper_1"] = close - vwap_upper_1
    df["dist_to_vwap_lower_1"] = close - vwap_lower_1
    df["dist_to_vwap_upper_2"] = close - vwap_upper_2
    df["dist_to_vwap_lower_2"] = close - vwap_lower_2
    df["above_vwap_flag"] = (close > vwap).astype(int)
    df["vwap_slope"] = vwap.groupby(session_series).diff()

    prior_day_vwap = None
    prior_week_vwap = None
    if include_prior_day_vwap or include_prior_week_vwap:
        typical = (high + low + close) / 3.0
        daily_vwap = _compute_period_vwap(typical, volume, session_series).sort_index()
        if include_prior_day_vwap:
            prior_day_vwap = _map_by_key(session_series, daily_vwap.shift(1))
            df["dist_to_prior_day_vwap"] = close - prior_day_vwap

        if include_prior_week_vwap:
            week_key = df.index.to_period("W")
            week_series = pd.Series(week_key, index=df.index)
            weekly_vwap = _compute_period_vwap(typical, volume, week_series).sort_index()
            prior_week_vwap = _map_by_key(week_series, weekly_vwap.shift(1))
            df["dist_to_prior_week_vwap"] = close - prior_week_vwap

    # --- Higher timeframe levels ---
    pdh, pdl, pwh, pwl, pmh, pml = _compute_htf_levels(
        high,
        low,
        session_series,
        include_prior_month_levels=include_prior_month_levels,
    )
    df["dist_to_pdh"] = close - pdh
    df["dist_to_pdl"] = close - pdl
    df["dist_to_pwh"] = close - pwh
    df["dist_to_pwl"] = close - pwl
    if include_prior_month_levels:
        df["dist_to_pmh"] = close - pmh
        df["dist_to_pml"] = close - pml

    # --- Psych levels (round numbers) ---
    for level in round_levels:
        level = int(level)
        nearest = np.round(close / level) * level
        df[f"dist_to_round_{level}"] = close - nearest

    # --- Wick rejection dynamics ---
    upper_wick, lower_wick, total_range = _compute_wicks(open_, high, low, close)
    df["upper_wick_ratio"] = _safe_divide(upper_wick, total_range)
    df["lower_wick_ratio"] = _safe_divide(lower_wick, total_range)
    df["range_normalized_wick"] = _safe_divide(upper_wick + lower_wick, total_range)

    rejection_up = df["upper_wick_ratio"] >= wick_ratio_threshold
    rejection_down = df["lower_wick_ratio"] >= wick_ratio_threshold

    vwap_levels = [vwap, vwap_upper_1, vwap_lower_1, vwap_upper_2, vwap_lower_2]
    near_vwap = _near_any_level(close, vwap_levels, near_level_pct)
    df["rejection_near_vwap_flag"] = (near_vwap & (rejection_up | rejection_down)).astype(
        int
    )

    htf_high_levels = [pdh, pwh]
    htf_low_levels = [pdl, pwl]
    if include_prior_month_levels:
        htf_high_levels.append(pmh)
        htf_low_levels.append(pml)

    near_htf_high = _near_any_level(close, htf_high_levels, near_level_pct)
    near_htf_low = _near_any_level(close, htf_low_levels, near_level_pct)
    df["rejection_near_htf_flag"] = (
        (near_htf_high & rejection_up) | (near_htf_low & rejection_down)
    ).astype(int)

    if round_rejection_window:
        near_round = pd.Series(False, index=df.index)
        for level in round_levels:
            dist = df.get(f"dist_to_round_{int(level)}")
            if dist is None:
                continue
            near_round |= dist.abs() <= (close.abs() * round_tolerance)
        round_rejection = near_round & (rejection_up | rejection_down)
        df[f"round_rejection_count_last_{round_rejection_window}"] = (
            round_rejection.rolling(window=round_rejection_window, min_periods=1).sum()
        )

    # --- Market profile (optional, prior day to avoid lookahead) ---
    if include_market_profile:
        poc, vah, val = _compute_daily_market_profile(
            high,
            low,
            close,
            volume,
            session_series,
            value_area_pct=value_area_pct,
            bin_size=profile_bin_size,
        )
        df["dist_to_poc"] = close - poc
        df["inside_value_area_flag"] = (
            (close >= val) & (close <= vah)
        ).astype(int)

    return df


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = numerator.to_numpy(dtype=float)
    den = denominator.to_numpy(dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    np.divide(num, den, out=out, where=den != 0)
    return pd.Series(out, index=numerator.index)


def _rolling_touch_count(
    series: pd.Series,
    window: int,
    tolerance: float,
    *,
    mode: str,
) -> pd.Series:
    if window <= 0:
        return pd.Series(np.nan, index=series.index)

    def _count(arr: np.ndarray) -> float:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        if mode == "high":
            level = np.max(arr)
            return float(np.sum(arr >= level * (1.0 - tolerance)))
        level = np.min(arr)
        return float(np.sum(arr <= level * (1.0 + tolerance)))

    return series.rolling(window=window, min_periods=1).apply(_count, raw=True)


def _compute_session_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    session_key: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    typical = (high + low + close) / 3.0
    cum_vol = volume.groupby(session_key).cumsum()
    cum_pv = (typical * volume).groupby(session_key).cumsum()
    vwap = _safe_divide(cum_pv, cum_vol)

    cum_p2v = (typical * typical * volume).groupby(session_key).cumsum()
    mean_sq = _safe_divide(cum_p2v, cum_vol)
    var = (mean_sq - vwap**2).clip(lower=0)
    vwap_std = np.sqrt(var)
    return vwap, vwap_std


def _compute_period_vwap(
    typical: pd.Series,
    volume: pd.Series,
    period_key: pd.Series,
) -> pd.Series:
    total_pv = (typical * volume).groupby(period_key).sum()
    total_vol = volume.groupby(period_key).sum()
    return _safe_divide(total_pv, total_vol)


def _map_by_key(
    key: pd.Series,
    values_by_key: pd.Series,
) -> pd.Series:
    return key.map(values_by_key)


def _compute_htf_levels(
    high: pd.Series,
    low: pd.Series,
    session_key: pd.Series,
    *,
    include_prior_month_levels: bool,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series | None, pd.Series | None]:
    day_high = high.groupby(session_key).max().sort_index()
    day_low = low.groupby(session_key).min().sort_index()
    pdh = _map_by_key(session_key, day_high.shift(1))
    pdl = _map_by_key(session_key, day_low.shift(1))

    week_key = pd.Series(high.index.to_period("W"), index=high.index)
    week_high = high.groupby(week_key).max().sort_index()
    week_low = low.groupby(week_key).min().sort_index()
    pwh = _map_by_key(week_key, week_high.shift(1))
    pwl = _map_by_key(week_key, week_low.shift(1))

    pmh = None
    pml = None
    if include_prior_month_levels:
        month_key = pd.Series(high.index.to_period("M"), index=high.index)
        month_high = high.groupby(month_key).max().sort_index()
        month_low = low.groupby(month_key).min().sort_index()
        pmh = _map_by_key(month_key, month_high.shift(1))
        pml = _map_by_key(month_key, month_low.shift(1))

    return pdh, pdl, pwh, pwl, pmh, pml


def _compute_wicks(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    body_top = pd.concat([open_, close], axis=1).max(axis=1)
    body_bottom = pd.concat([open_, close], axis=1).min(axis=1)
    upper_wick = (high - body_top).clip(lower=0)
    lower_wick = (body_bottom - low).clip(lower=0)
    total_range = (high - low).clip(lower=0)
    return upper_wick, lower_wick, total_range


def _near_any_level(
    price: pd.Series,
    levels: Iterable[pd.Series],
    pct: float,
) -> pd.Series:
    level_list = [lvl.to_numpy(dtype=float) for lvl in levels if lvl is not None]
    if not level_list:
        return pd.Series(False, index=price.index)
    stacked = np.column_stack(level_list)
    diffs = np.abs(stacked - price.to_numpy(dtype=float)[:, None])
    diffs[~np.isfinite(diffs)] = np.inf
    min_diff = diffs.min(axis=1)
    threshold = price.to_numpy(dtype=float) * pct
    near = min_diff <= threshold
    return pd.Series(near, index=price.index)


def _compute_daily_market_profile(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    session_key: pd.Series,
    *,
    value_area_pct: float,
    bin_size: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    typical = (high + low + close) / 3.0
    daily = {}
    for day, idx in session_key.groupby(session_key).groups.items():
        prices = typical.loc[idx].to_numpy(dtype=float)
        vols = volume.loc[idx].to_numpy(dtype=float)
        if prices.size == 0:
            daily[day] = (np.nan, np.nan, np.nan)
            continue
        day_min = np.nanmin(prices)
        day_max = np.nanmax(prices)
        if not np.isfinite(day_min) or not np.isfinite(day_max):
            daily[day] = (np.nan, np.nan, np.nan)
            continue
        if day_max == day_min:
            daily[day] = (day_min, day_max, day_min)
            continue

        edges = np.arange(day_min, day_max + bin_size, bin_size)
        if edges.size < 2:
            edges = np.array([day_min, day_max])

        bin_idx = np.digitize(prices, edges) - 1
        bin_idx = np.clip(bin_idx, 0, edges.size - 2)
        vol_by_bin = np.zeros(edges.size - 1, dtype=float)
        np.add.at(vol_by_bin, bin_idx, vols)

        poc_idx = int(np.nanargmax(vol_by_bin))
        total_vol = np.nansum(vol_by_bin)
        if total_vol == 0:
            daily[day] = (np.nan, np.nan, np.nan)
            continue

        target = total_vol * value_area_pct
        lo = hi = poc_idx
        cum = vol_by_bin[poc_idx]
        while cum < target and (lo > 0 or hi < vol_by_bin.size - 1):
            next_lo = vol_by_bin[lo - 1] if lo > 0 else -np.inf
            next_hi = vol_by_bin[hi + 1] if hi < vol_by_bin.size - 1 else -np.inf
            if next_hi >= next_lo:
                hi += 1
                cum += vol_by_bin[hi]
            else:
                lo -= 1
                cum += vol_by_bin[lo]

        poc = (edges[poc_idx] + edges[poc_idx + 1]) / 2.0
        val = edges[lo]
        vah = edges[hi + 1]
        daily[day] = (poc, vah, val)

    daily_df = pd.DataFrame.from_dict(
        daily, orient="index", columns=["poc", "vah", "val"]
    ).sort_index()
    # Shift one day to avoid intraday lookahead.
    daily_df = daily_df.shift(1)
    poc = _map_by_key(session_key, daily_df["poc"])
    vah = _map_by_key(session_key, daily_df["vah"])
    val = _map_by_key(session_key, daily_df["val"])
    return poc, vah, val
