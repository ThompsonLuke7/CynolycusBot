"""
30-minute feature builder for the multi-ticker swing trading pipeline.

Computes all 11 feature categories per ticker, then assembles a combined
(ticker, timestamp) MultiIndex parquet for downstream label building and
model training.

Reuses from existing codebase:
  - Features.feature_sets.custom_indicators.add_fractal_pivots
  - Features.feature_sets.custom_indicators.add_atr_swing_state_features
"""
from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Feature building builds columns incrementally; fragmentation warnings are expected
# and harmless. Suppress them to keep the live log readable.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Existing custom indicators (reused directly)
from strategies.spy_intraday.Features.feature_sets.custom_indicators import (
    add_atr_swing_state_features,
)

from strategies.multi_ticker_swing.config.pipeline_config import (
    CONTEXT_TICKERS,
    FEATURES_COMBINED,
    MIN_BARS,
    PROCESSED_30M_DIR,
    RAW_30M_DIR,
    UNIVERSE_CSV,
)
from strategies.multi_ticker_swing.data.fetch_data import load_universe
from strategies.multi_ticker_swing.data.load_data import load_raw_30m

logger = logging.getLogger(__name__)


def _safe_logret(num: pd.Series, den: pd.Series) -> pd.Series:
    """``log(num/den)`` with non-positive ratios mapped to NaN.

    A zero/negative close (halted or bad bar) makes the ratio 0 → ``log(0) = -inf``,
    which spams ``RuntimeWarning: divide by zero encountered in log`` once per bad
    bar across the whole universe on every feature build. A return across a
    non-positive price is undefined, so NaN is the correct value anyway.
    """
    ratio = num / den
    return np.log(ratio.where(ratio > 0))


# 13 RTH bars/day at 30-min (09:30 – 16:00 = 13 bars)
BARS_PER_DAY = 13

# Swing setup score gate threshold
_SETUP_THRESHOLD = 0.60


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce parquet/RAM footprint without changing model-facing semantics."""
    out = df.copy()
    float_cols = out.select_dtypes(include=["float64"]).columns
    if len(float_cols):
        out[float_cols] = out[float_cols].astype(np.float32)
    int_cols = out.select_dtypes(include=["int64", "int32"]).columns
    if len(int_cols):
        out[int_cols] = out[int_cols].astype(np.int32)
    return out


def _append_parquet_frame(
    writer: pq.ParquetWriter | None,
    df: pd.DataFrame,
    path: Path,
) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(path, table.schema, compression="snappy")
    writer.write_table(table)
    return writer


# ===========================================================================
# Category helpers
# ===========================================================================

def _add_cat1_price(df: pd.DataFrame) -> pd.DataFrame:
    """Cat 1 – Price / return features."""
    c = df["close"]
    o = df["open"]
    h = df["high"]
    lo = df["low"]

    for n in (1, 2, 4, 8, 16):
        df[f"ret_{n}"] = c.pct_change(n)

    df["log_ret_1"] = _safe_logret(c, c.shift(1))

    # z-score of 1-bar close-to-close return over 20-bar rolling window
    r1 = df["log_ret_1"]
    roll_mean = r1.rolling(20).mean()
    roll_std  = r1.rolling(20).std().replace(0, np.nan)
    df["close_to_close_z_20"] = (r1 - roll_mean) / roll_std

    df["open_to_close_ret"]   = (c - o) / o.replace(0, np.nan)
    df["high_low_range_pct"]  = (h - lo) / c.replace(0, np.nan)
    denom = (h - lo).replace(0, np.nan)
    df["close_location_in_bar"] = (c - lo) / denom

    return df


def _add_cat2_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Cat 2 – Volatility / expansion."""
    atr  = df["atr_14"]
    c    = df["close"]
    h    = df["high"]
    lo   = df["low"]
    c_p  = c.shift(1)

    df["atr_pct_14"]  = atr / c.replace(0, np.nan)
    atr_pct = df["atr_pct_14"]

    roll_mean_64 = atr_pct.rolling(64).mean()
    roll_std_64  = atr_pct.rolling(64).std().replace(0, np.nan)
    df["atr_pct_z_64"] = (atr_pct - roll_mean_64) / roll_std_64

    log_r = _safe_logret(c, c_p)
    for n in (4, 16, 32):
        df[f"realized_vol_{n}"] = log_r.rolling(n).std()

    # Range expansion: current bar range vs N-bar average range
    bar_range = h - lo
    for n in (4, 16):
        avg = bar_range.rolling(n).mean().replace(0, np.nan)
        df[f"range_expansion_{n}"] = bar_range / avg

    # Range regime: 8-bar vol vs 32-bar vol ratio
    rv8  = log_r.rolling(8).std()
    rv32 = log_r.rolling(32).std().replace(0, np.nan)
    df["range_regime_8_32"] = rv8 / rv32

    # True range pct
    tr = pd.concat(
        [h - lo, (h - c_p).abs(), (lo - c_p).abs()], axis=1
    ).max(axis=1)
    df["true_range_pct"] = tr / c.replace(0, np.nan)

    # Vol of vol: rolling std of 4-bar realized vol over 16 bars
    df["vol_of_vol_16"] = df["realized_vol_4"].rolling(16).std()

    return df


def _add_cat3_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Cat 3 – Trend / structure."""
    c   = df["close"]
    atr = df["atr_14"].replace(0, np.nan)

    # EMAs
    ema10 = c.ewm(span=10, adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()

    # Clip to ±10 ATR — commodity ETFs/volatility products can briefly reach ±13
    df["ema_dist_10"] = ((c - ema10) / atr).clip(-10, 10)
    df["ema_dist_20"] = ((c - ema20) / atr).clip(-10, 10)
    df["ema_dist_50"] = ((c - ema50) / atr).clip(-10, 10)

    # EMA slopes (1-bar pct change of the EMA)
    df["ema_slope_10"] = ema10.pct_change(1)
    df["ema_slope_20"] = ema20.pct_change(1)

    # EMA stack state: encode which EMAs are above which
    # Bit 0: ema10 > ema20 | Bit 1: ema20 > ema50 | Bit 2: ema10 > ema50
    b0 = (ema10 > ema20).astype(int)
    b1 = (ema20 > ema50).astype(int)
    b2 = (ema10 > ema50).astype(int)
    df["ema_stack_state"] = b0 + 2 * b1 + 4 * b2   # 0-7

    # ADX / DI via pandas_ta
    try:
        import pandas_ta as ta
        adx_df = ta.adx(df["high"], df["low"], c, length=14)
        if adx_df is not None and not adx_df.empty:
            cols = adx_df.columns.tolist()
            # columns typically: ADX_14, DMP_14, DMN_14
            df["adx_14"] = adx_df.iloc[:, 0].values
            df["dmp_14"] = adx_df.iloc[:, 1].values
            df["dmn_14"] = adx_df.iloc[:, 2].values
        else:
            df["adx_14"] = np.nan
            df["dmp_14"] = np.nan
            df["dmn_14"] = np.nan
    except Exception:
        df["adx_14"] = np.nan
        df["dmp_14"] = np.nan
        df["dmn_14"] = np.nan

    # Clip DM indicators to valid [0, 100] range (pandas_ta can overflow on
    # extremely volatile tickers like UVXY where values reach >5000)
    for col in ("adx_14", "dmp_14", "dmn_14"):
        df[col] = df[col].clip(0, 100)

    # ADXR = (ADX + ADX[14]) / 2
    adx = df["adx_14"]
    df["adxr_14"] = (adx + adx.shift(14)) / 2
    df["adxr_14"] = df["adxr_14"].clip(0, 100)

    # Trend phase momentum / acceleration (normalized EMA velocity)
    ema_vel = ema20.diff()
    df["trend_phase_m"] = ema_vel / atr          # velocity normalized by ATR
    df["trend_phase_a"] = ema_vel.diff() / atr   # acceleration

    return df


def _add_cat4_swing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cat 4 – Swing / setup structure.

    CAUSAL: Uses only past/current data, no lookahead.
    - Rolling 20-bar extremes (causal lookback)
    - atr_swing_state from add_atr_swing_state_features (explicitly causal)
    - NOT fractal pivots (they require 3-bar lookahead to confirm)
    """
    c   = df["close"]
    h   = df["high"]
    lo  = df["low"]
    atr = df["atr_14"].replace(0, np.nan)

    # ------------------------------------------------------------------
    # Rolling 20-bar swing high / low (CAUSAL: lookback only)
    # ------------------------------------------------------------------
    roll_high20 = h.rolling(20).max()
    roll_low20  = lo.rolling(20).min()
    roll_range  = (roll_high20 - roll_low20).replace(0, np.nan)

    df["range_pos_20"]    = (c - roll_low20) / roll_range
    df["dist_20bar_high"] = (roll_high20 - c) / atr
    df["dist_20bar_low"]  = (c - roll_low20) / atr

    # Use rolling extremes as swing levels (avoids fractal pivot lookahead)
    df["dist_to_recent_swing_high"] = df["dist_20bar_high"]
    df["dist_to_recent_swing_low"]  = df["dist_20bar_low"]

    # ------------------------------------------------------------------
    # Compression score: short-term vol compressed vs long-term (CAUSAL)
    # ------------------------------------------------------------------
    rv4  = df.get("realized_vol_4",  pd.Series(np.nan, index=df.index))
    rv32 = df.get("realized_vol_32", pd.Series(np.nan, index=df.index))
    ratio = (rv4 / rv32.replace(0, np.nan)).clip(0, 3)
    compression = (1.0 - ratio / 3.0).clip(0, 1)
    df["compression_score"] = compression

    # ------------------------------------------------------------------
    # Breakout pressure: compression + rising volume (CAUSAL)
    # ------------------------------------------------------------------
    vol_rel = df.get("volume_rel_20", pd.Series(1.0, index=df.index))
    vol_component = ((vol_rel - 1.0).clip(0, 3) / 3.0)
    df["breakout_pressure_score"] = (0.6 * compression + 0.4 * vol_component).clip(0, 1)

    # ------------------------------------------------------------------
    # Swing setup fuzzy scores (CAUSAL: range_pos_20 + compression + EMA)
    # Avoid fractal pivots (they have 3-bar lookahead); use rolling extremes instead
    # ------------------------------------------------------------------
    range_pos = df["range_pos_20"].fillna(0.5)
    comp      = df["compression_score"].fillna(0.0)
    ema_stack = df.get("ema_stack_state", pd.Series(4, index=df.index))

    # atr_swing_state from add_atr_swing_state_features is explicitly causal
    swing_state = df.get("atr_swing_state", pd.Series(0, index=df.index))
    is_upswing  = (swing_state == 1).astype(float)
    is_downswing = (swing_state == -1).astype(float)

    # Long setup: at or near recent low, compressed, causal swing state supports
    long_pos_score  = (1.0 - range_pos).clip(0, 1)
    long_ema_score  = (ema_stack.isin([0, 2, 4, 6])).astype(float)
    long_state_score = is_downswing  # in downswing = potential long setup
    df["swing_setup_score_long"] = (
        0.35 * long_pos_score + 0.35 * comp + 0.15 * long_ema_score + 0.15 * long_state_score
    ).clip(0, 1)

    # Short setup: at or near recent high, compressed, downtrend
    short_pos_score  = range_pos.clip(0, 1)
    short_ema_score  = (ema_stack.isin([1, 3, 5, 7])).astype(float)
    short_state_score = is_upswing  # in upswing = potential short setup
    df["swing_setup_score_short"] = (
        0.35 * short_pos_score + 0.35 * comp + 0.15 * short_ema_score + 0.15 * short_state_score
    ).clip(0, 1)

    # ------------------------------------------------------------------
    # Bars since last long / short setup trigger (CAUSAL)
    # ------------------------------------------------------------------
    def _bars_since(flag: pd.Series) -> pd.Series:
        out = []
        count = 999
        for v in flag:
            if v:
                count = 0
            else:
                count = min(count + 1, 999)
            out.append(count)
        return pd.Series(out, index=flag.index, dtype=float)

    long_flag  = (df["swing_setup_score_long"]  >= _SETUP_THRESHOLD)
    short_flag = (df["swing_setup_score_short"] >= _SETUP_THRESHOLD)
    df["bars_since_long_setup"]  = _bars_since(long_flag)
    df["bars_since_short_setup"] = _bars_since(short_flag)

    return df


def _add_cat5_model_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cat 5 – Prior model output placeholders.
    Filled with NaN until GA-XGBoost pivot models are trained and wired in.
    """
    placeholder_cols = [
        "p_setup_long", "p_setup_short",
        "p_setup_long_lag1",  "p_setup_long_lag2",
        "p_setup_short_lag1", "p_setup_short_lag2",
        "p_setup_long_delta_1",  "p_setup_short_delta_1",
        "p_setup_long_max_last_4", "p_setup_short_max_last_4",
    ]
    for col in placeholder_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _add_cat6_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Cat 6 – Volume / participation."""
    v = df["volume"].replace(0, np.nan)
    c = df["close"]

    roll_mean_20 = v.rolling(20).mean().replace(0, np.nan)
    roll_std_20  = v.rolling(20).std().replace(0, np.nan)

    df["volume_z_20"]    = (v - v.rolling(20).mean()) / roll_std_20
    df["volume_rel_20"]  = v / roll_mean_20
    df["dollar_volume"]  = v * c
    dv_mean_20 = (v * c).rolling(20).mean().replace(0, np.nan)
    df["dollar_volume_rel_20"] = (v * c) / dv_mean_20

    # Rolling percentile rank of dollar volume (252-bar window)
    df["dollar_volume_pctile"] = (
        (v * c).rolling(252, min_periods=50).rank(pct=True)
    )

    # OBV slope normalized by rolling 20-bar mean volume so the signal is
    # cross-ticker comparable (raw OBV units depend on share volume scale)
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["obv_slope_10"] = obv.diff(10) / roll_mean_20 / 10

    # Acc/Dist slope normalized by rolling 10-bar mean dollar volume
    bar_range = (df["high"] - df["low"]).replace(0, np.nan)
    clv = ((c - df["low"]) - (df["high"] - c)) / bar_range
    ad  = (clv * v).fillna(0).cumsum()
    dv_mean_10 = (v * c).rolling(10).mean().replace(0, np.nan)
    df["accdist_slope_10"] = ad.diff(10) / dv_mean_10 / 10

    # Relative volume in the opening window (first 2 bars of session)
    # Defined as current volume / rolling mean of same session-bar-index volume
    # Approximated here as: is this bar in top-25% volume among the last 20 open-window bars?
    # For a proper implementation the intraday bar-index is needed (computed in Cat 10).
    # We'll fill this after Cat 10 is available; placeholder for now.
    df["relative_volume_open_window"] = np.nan   # filled by _finalize_open_window_vol

    return df


def _add_cat7_gap(df: pd.DataFrame, session_dates: pd.Series) -> pd.DataFrame:
    """Cat 7 – Gap / overnight context.  Requires session_dates Series (NY date per bar)."""
    c   = df["close"]
    o   = df["open"]
    atr = df["atr_14"].replace(0, np.nan)

    # Previous close: last close of the previous session date
    prev_close = c.copy()
    prev_close_map: dict = {}
    last_close_map: dict = {}
    for ts, row in df[["close"]].iterrows():
        sd = session_dates.loc[ts]
        prev_close_map[ts] = last_close_map.get(sd, np.nan)
        last_close_map[sd] = row["close"]

    pc_series = pd.Series(prev_close_map, dtype=float)

    df["gap_pct"]      = (o - pc_series) / pc_series.replace(0, np.nan)
    df["gap_to_atr"]   = df["gap_pct"] / (atr / c.replace(0, np.nan)).replace(0, np.nan)
    df["overnight_ret"] = df["gap_pct"]  # same concept for equity bars

    # Previous day return: yesterday's close / day-before-yesterday's close - 1
    # Build a daily close series first
    daily_close = (
        df.assign(_sd=session_dates)
        .groupby("_sd")["close"].last()
        .sort_index()
    )
    prev_day_close  = daily_close.shift(1)
    prev2_day_close = daily_close.shift(2)
    prev_day_ret_map = (prev_day_close / prev2_day_close.replace(0, np.nan) - 1).to_dict()
    df["prev_day_ret"] = session_dates.map(prev_day_ret_map).values

    # Previous day range pct
    daily_high = df.assign(_sd=session_dates).groupby("_sd")["high"].max()
    daily_low  = df.assign(_sd=session_dates).groupby("_sd")["low"].min()
    prev_day_range = ((daily_high.shift(1) - daily_low.shift(1)) / daily_close.shift(1).replace(0, np.nan))
    df["prev_day_range_pct"] = session_dates.map(prev_day_range.to_dict()).values

    # Gap followthrough: return of bar 1 and bar 2 from session open, forward-filled
    bars_from_open = df.get("bars_from_open", pd.Series(np.nan, index=df.index))
    ff1_raw = (c - o) / o.replace(0, np.nan)
    ff1_raw = ff1_raw.where(bars_from_open == 0)
    ff2_raw = c.pct_change(2).where(bars_from_open == 1)

    df["gap_followthrough_1"] = ff1_raw.ffill()
    df["gap_followthrough_2"] = ff2_raw.ffill()

    return df


def _add_cat8_context(
    df: pd.DataFrame,
    ticker: str,
    ctx: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Cat 8 – Relative strength / market context.

    Computes returns for all context tickers (SPY, QQQ, IWM, TLT, USO, GLD, …).
    For USO and GLD also computes relative-strength and 64-bar rolling correlation,
    giving the model causal information about what is driving commodity-correlated names.
    """
    c   = df["close"]
    idx = df.index

    own_logret = _safe_logret(c, c.shift(1))

    def _ctx_close(name: str) -> pd.Series:
        if name not in ctx:
            return pd.Series(np.nan, index=idx)
        return ctx[name]["close"].reindex(idx, method="ffill")

    def _ctx_ret(name: str, n: int) -> pd.Series:
        return _ctx_close(name).pct_change(n)

    def _ctx_logret(name: str) -> pd.Series:
        cc = _ctx_close(name)
        return _safe_logret(cc, cc.shift(1))

    def _rolling_beta(y: pd.Series, x: pd.Series, w: int) -> pd.Series:
        cov  = y.rolling(w).cov(x)
        xvar = x.rolling(w).var().replace(0, np.nan)
        return (cov / xvar).clip(-5, 5)

    def _rolling_corr(y: pd.Series, x: pd.Series, w: int) -> pd.Series:
        return y.rolling(w).corr(x)

    # ------------------------------------------------------------------
    # Equity index context — SPY, QQQ, IWM returns
    # ------------------------------------------------------------------
    for sym in ("SPY", "QQQ", "IWM"):
        key = sym.lower()
        for n in (1, 4, 16):
            df[f"{key}_ret_{n}"] = _ctx_ret(sym, n)

    # Relative strength vs SPY
    for n in (4, 16):
        spy_ret = _ctx_ret("SPY", n).replace(0, np.nan)
        df[f"rel_str_spy_{n}"] = (df[f"ret_{n}"] / spy_ret).clip(-10, 10)

    qqq_ret4 = _ctx_ret("QQQ", 4).replace(0, np.nan)
    df["rel_str_qqq_4"] = (df["ret_4"] / qqq_ret4).clip(-10, 10)

    # Rolling beta and correlation vs SPY
    spy_logret = _ctx_logret("SPY")
    if ticker == "SPY":
        df["beta_like_spy_64"] = 1.0
        df["corr_spy_64"]      = 1.0
    else:
        df["beta_like_spy_64"] = _rolling_beta(own_logret, spy_logret, 64)
        df["corr_spy_64"]      = _rolling_corr(own_logret, spy_logret, 64)

    # Market regime proxy: SPY 16-bar return sign
    spy16 = _ctx_ret("SPY", 16)
    df["market_regime_proxy"] = spy16.apply(lambda x: 1.0 if x > 0.01 else (-1.0 if x < -0.01 else 0.0))

    # ------------------------------------------------------------------
    # Rate context — TLT returns (rates-rising = TLT falling)
    # Gives model causal rate information for rate-sensitive sectors
    # ------------------------------------------------------------------
    for n in (1, 4, 16):
        df[f"tlt_ret_{n}"] = _ctx_ret("TLT", n)

    # ------------------------------------------------------------------
    # Crude oil context — USO returns + relative strength + correlation
    # Causal driver for energy names: model learns "USO falling → don't go long XOM"
    # ------------------------------------------------------------------
    uso_logret = _ctx_logret("USO")
    for n in (1, 4, 16):
        df[f"uso_ret_{n}"] = _ctx_ret("USO", n)

    uso_ret4 = _ctx_ret("USO", 4).replace(0, np.nan)
    df["rel_str_uso_4"] = (df.get("ret_4", pd.Series(np.nan, index=idx)) / uso_ret4).clip(-10, 10)

    if ticker == "USO":
        df["corr_uso_64"] = 1.0
    else:
        df["corr_uso_64"] = _rolling_corr(own_logret, uso_logret, 64)

    # ------------------------------------------------------------------
    # Gold context — GLD returns + relative strength + correlation
    # Causal driver for metals/miners: model learns "GLD falling → don't go long GDX"
    # ------------------------------------------------------------------
    gld_logret = _ctx_logret("GLD")
    for n in (1, 4, 16):
        df[f"gld_ret_{n}"] = _ctx_ret("GLD", n)

    gld_ret4 = _ctx_ret("GLD", 4).replace(0, np.nan)
    df["rel_str_gld_4"] = (df.get("ret_4", pd.Series(np.nan, index=idx)) / gld_ret4).clip(-10, 10)

    if ticker == "GLD":
        df["corr_gld_64"] = 1.0
    else:
        df["corr_gld_64"] = _rolling_corr(own_logret, gld_logret, 64)

    return df


def _add_cat9_location(
    df: pd.DataFrame, session_dates: pd.Series
) -> pd.DataFrame:
    """Cat 9 – Location / mean-reversion context."""
    c   = df["close"]
    atr = df["atr_14"].replace(0, np.nan)

    # Session VWAP (resets each day)
    typical = (df["high"] + df["low"] + c) / 3
    v = df["volume"].replace(0, np.nan)
    vwap_vals = []
    cum_tv = 0.0
    cum_v  = 0.0
    prev_sd = None
    for ts, (tp, vol) in zip(df.index, zip(typical, v)):
        sd = session_dates.loc[ts]
        if sd != prev_sd:
            cum_tv = 0.0
            cum_v  = 0.0
            prev_sd = sd
        cum_tv += tp * (vol if not math.isnan(vol) else 0.0)
        cum_v  += vol if not math.isnan(vol) else 0.0
        vwap_vals.append(cum_tv / cum_v if cum_v > 0 else float("nan"))

    vwap = pd.Series(vwap_vals, index=df.index, dtype=float)
    df["dist_to_vwap"] = (c - vwap) / atr

    # Previous day high / low / close
    daily_high  = df.assign(_sd=session_dates).groupby("_sd")["high"].max()
    daily_low   = df.assign(_sd=session_dates).groupby("_sd")["low"].min()
    daily_close = df.assign(_sd=session_dates).groupby("_sd")["close"].last()

    pdh = session_dates.map(daily_high.shift(1).to_dict()).values
    pdl = session_dates.map(daily_low.shift(1).to_dict()).values
    pdc = session_dates.map(daily_close.shift(1).to_dict()).values

    df["dist_to_pdh"] = (pd.Series(pdh, index=df.index, dtype=float) - c) / atr
    df["dist_to_pdl"] = (c - pd.Series(pdl, index=df.index, dtype=float)) / atr
    df["dist_to_pdc"] = (c - pd.Series(pdc, index=df.index, dtype=float)) / atr

    # Distance to rolling mean
    sma20 = c.rolling(20).mean()
    df["dist_to_rolling_mean_20"] = (c - sma20) / atr

    # Z-scores
    for w in (20, 64):
        mu  = c.rolling(w).mean()
        std = c.rolling(w).std().replace(0, np.nan)
        df[f"zscore_close_{w}"]      = (c - mu) / std
        df[f"percentile_close_{w}"]  = c.rolling(w).rank(pct=True)

    return df


def _add_cat10_time(df: pd.DataFrame, session_dates: pd.Series) -> pd.DataFrame:
    """Cat 10 – Time features."""
    idx_ny = df.index.tz_convert("America/New_York")

    dow = idx_ny.dayofweek.astype(float)
    df["day_of_week_sin"] = np.sin(2 * np.pi * dow / 5)
    df["day_of_week_cos"] = np.cos(2 * np.pi * dow / 5)

    df["is_monday"] = (dow == 0).astype(float)
    df["is_friday"] = (dow == 4).astype(float)

    # Bars from open / bars to close within each session
    open_min  = 9 * 60 + 30   # 570 minutes from midnight
    close_min = 15 * 60 + 30  # 930 minutes — last 30-min bar opens at 15:30
    session_len = close_min - open_min  # 360 minutes (13 bars × 30m = 390m open→close)

    bar_min = pd.Series(idx_ny.hour * 60 + idx_ny.minute, index=df.index)
    df["bars_from_open"] = ((bar_min - open_min) / 30).clip(0, BARS_PER_DAY - 1).astype(float)
    df["bars_to_close"]  = ((close_min - bar_min) / 30).clip(0, BARS_PER_DAY - 1).astype(float)

    # Session-relative fraction: 0.0 at open (09:30), 1.0 at session close (15:30).
    # Replaces the 24h sin/cos encoding where all RTH bars compressed into a tiny
    # arc of hour_cos ∈ [-1.0, -0.5] — essentially no useful variation.
    session_frac = ((bar_min - open_min) / session_len).clip(0, 1)
    df["session_frac"]     = session_frac
    df["session_frac_sin"] = np.sin(np.pi * session_frac)   # 0→0, 0.5→1, 1→0 (peaks mid-session)

    return df


def _add_cat11_behavioral(
    df: pd.DataFrame, meta: dict
) -> pd.DataFrame:
    """
    Cat 11 – Behavioral identity features.
    meta = {"sector_id": int, "market_cap_bucket": int, "asset_type": int}
    """
    df["sector_id"]        = float(meta.get("sector_id", 0))
    df["market_cap_bucket"] = float(meta.get("market_cap_bucket", 0))

    # Rolling percentile of atr_pct_14 (volatility regimen)
    atr_pct = df["atr_pct_14"] if "atr_pct_14" in df.columns else pd.Series(np.nan, index=df.index)
    df["volatility_pctile_rolling"] = atr_pct.rolling(252, min_periods=50).rank(pct=True)

    # Rolling percentile of dollar_volume
    dv = df["dollar_volume"] if "dollar_volume" in df.columns else pd.Series(np.nan, index=df.index)
    df["dollar_vol_pctile_rolling"] = dv.rolling(252, min_periods=50).rank(pct=True)

    # Gap frequency: rolling fraction of bars where |gap_pct| > 0.3 %
    gap = df["gap_pct"].abs() if "gap_pct" in df.columns else pd.Series(0.0, index=df.index)
    df["gap_frequency_rolling"] = (gap > 0.003).astype(float).rolling(252, min_periods=50).mean()

    # Trendiness score: rolling fraction of bars where ADX > 25
    adx = df["adx_14"] if "adx_14" in df.columns else pd.Series(np.nan, index=df.index)
    df["trendiness_score_rolling"] = (adx > 25).astype(float).rolling(252, min_periods=50).mean()

    # Beta bucket: discretize beta_like_spy_64
    beta = df.get("beta_like_spy_64", pd.Series(1.0, index=df.index))
    df["stock_beta_bucket"] = pd.cut(
        beta, bins=[-np.inf, 0.5, 1.0, 1.5, np.inf], labels=[0, 1, 2, 3]
    ).astype(float)

    return df


def _add_daily_context(df: pd.DataFrame, session_dates: pd.Series) -> pd.DataFrame:
    """
    Cat 12 – Daily timeframe features derived by resampling the 30m data.

    Leakage guard: ALL daily values are shifted forward by 1 day before being
    mapped back to 30m bars.  A bar on day D only ever sees data through the
    close of day D-1.  Day D's own daily bar is never used while it is still
    open.
    """
    # ------------------------------------------------------------------
    # Step 1: Build daily OHLCV by grouping 30m bars on their session date
    # ------------------------------------------------------------------
    daily = (
        df.assign(_sd=session_dates)
        .groupby("_sd")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .sort_index()
    )

    c  = daily["close"]
    h  = daily["high"]
    lo = daily["low"]
    v  = daily["volume"].replace(0, np.nan)

    # ------------------------------------------------------------------
    # Step 2: Compute daily indicators (on the full un-shifted daily series)
    # ------------------------------------------------------------------
    # Daily True Range / ATR-14
    c_p = c.shift(1)
    tr  = pd.concat([(h - lo), (h - c_p).abs(), (lo - c_p).abs()], axis=1).max(axis=1)
    daily_atr = tr.rolling(14).mean().replace(0, np.nan)

    # Daily return (close-to-close)
    daily_ret = c.pct_change(1)

    # Daily EMAs
    ema20d = c.ewm(span=20, adjust=False).mean()
    ema50d = c.ewm(span=50, adjust=False).mean()

    # Daily ATR pct (daily volatility level relative to price)
    daily_atr_pct = daily_atr / c.replace(0, np.nan)

    # Daily RSI-14
    try:
        import pandas_ta as ta
        daily_rsi = ta.rsi(c, length=14)
        if daily_rsi is None:
            daily_rsi = pd.Series(np.nan, index=daily.index)
    except Exception:
        daily_rsi = pd.Series(np.nan, index=daily.index)

    # EMA distances from close, ATR-normalised and clipped to ±10
    daily_ema_dist_20 = ((c - ema20d) / daily_atr).clip(-10, 10)
    daily_ema_dist_50 = ((c - ema50d) / daily_atr).clip(-10, 10)

    # Trend state: sign of (ema20 − ema50) → +1 bullish / -1 bearish / 0 flat
    daily_trend_state = np.sign(ema20d - ema50d)

    # Range position within the 20-day high / low channel [0, 1]
    roll_high20d = h.rolling(20).max()
    roll_low20d  = lo.rolling(20).min()
    roll_range20d = (roll_high20d - roll_low20d).replace(0, np.nan)
    daily_range_pos = ((c - roll_low20d) / roll_range20d).clip(0, 1)

    # Daily volume relative to 20-day mean
    vol_mean20d  = v.rolling(20).mean().replace(0, np.nan)
    daily_vol_rel = (v / vol_mean20d).clip(0, 10)

    # ------------------------------------------------------------------
    # Step 3: Shift ALL features forward 1 day — bars on day D see D-1 data
    # ------------------------------------------------------------------
    feat_daily = pd.DataFrame({
        "daily_ret_1":        daily_ret.shift(1),
        "daily_atr_pct":      daily_atr_pct.shift(1),
        "daily_rsi_14":       daily_rsi.shift(1),
        "daily_ema_dist_20":  daily_ema_dist_20.shift(1),
        "daily_ema_dist_50":  daily_ema_dist_50.shift(1),
        "daily_trend_state":  daily_trend_state.shift(1),
        "daily_range_pos_20": daily_range_pos.shift(1),
        "daily_vol_rel_20":   daily_vol_rel.shift(1),
    }, index=daily.index)

    # ------------------------------------------------------------------
    # Step 4: Map date-indexed daily features back to 30m bar timestamps
    # ------------------------------------------------------------------
    for col in feat_daily.columns:
        df[col] = session_dates.map(feat_daily[col].to_dict()).values

    return df


def _add_oscillators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cat 3b – Momentum oscillators missing from the original feature set.
    All values are bounded or normalized so they are cross-ticker comparable.
    """
    c  = df["close"]
    h  = df["high"]
    lo = df["low"]
    v  = df["volume"].replace(0, np.nan)

    try:
        import pandas_ta as ta

        # RSI [0, 100]
        rsi = ta.rsi(c, length=14)
        df["rsi_14"] = rsi.values if rsi is not None else np.nan

        # MACD histogram (fast=12, slow=26, signal=9)
        macd_df = ta.macd(c, fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            # pandas_ta column: MACDh_12_26_9
            hist_col = [col for col in macd_df.columns if col.startswith("MACDh")]
            if hist_col:
                atr = df["atr_14"].replace(0, np.nan)
                df["macd_hist_12_26_9"] = (macd_df[hist_col[0]].values / atr).clip(-5, 5)
            else:
                df["macd_hist_12_26_9"] = np.nan
        else:
            df["macd_hist_12_26_9"] = np.nan

        # Bollinger Band %B [0, 1] (values outside 0-1 indicate extreme moves)
        bb_df = ta.bbands(c, length=20, std=2)
        if bb_df is not None and not bb_df.empty:
            pctb_col = [col for col in bb_df.columns if col.startswith("BBP")]
            if pctb_col:
                df["bb_pct_b_20"] = bb_df[pctb_col[0]].values
            else:
                df["bb_pct_b_20"] = np.nan
        else:
            df["bb_pct_b_20"] = np.nan

        # Stochastic %K [0, 100]
        stoch_df = ta.stoch(h, lo, c, k=14, d=3, smooth_k=3)
        if stoch_df is not None and not stoch_df.empty:
            k_col = [col for col in stoch_df.columns if col.startswith("STOCHk")]
            if k_col:
                df["stoch_k_14"] = stoch_df[k_col[0]].values
            else:
                df["stoch_k_14"] = np.nan
        else:
            df["stoch_k_14"] = np.nan

        # Money Flow Index [0, 100]
        mfi = ta.mfi(h, lo, c, v, length=14)
        df["mfi_14"] = mfi.values if mfi is not None else np.nan

        # Choppiness Index [0, 100] (100 = max chop, 0 = trending)
        chop = ta.chop(h, lo, c, length=14)
        df["chop_14"] = chop.values if chop is not None else np.nan

    except Exception as exc:
        logger.warning("Oscillator computation failed: %s", exc)
        for col in ("rsi_14", "macd_hist_12_26_9", "bb_pct_b_20",
                    "stoch_k_14", "mfi_14", "chop_14"):
            if col not in df.columns:
                df[col] = np.nan

    return df


def _finalize_open_window_vol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill Cat 6 relative_volume_open_window after bars_from_open is available.
    For each bar at bars_from_open <= 1, computes its volume vs the rolling
    mean volume of bars with the same intraday index (approximated as overall mean).
    """
    bfo = df.get("bars_from_open", pd.Series(np.nan, index=df.index))
    v   = df["volume"].replace(0, np.nan)
    open_mask = bfo <= 1
    mean_v = v.rolling(40, min_periods=10).mean().replace(0, np.nan)
    df["relative_volume_open_window"] = (v / mean_v).where(open_mask).ffill()
    return df


# ===========================================================================
# Per-ticker entry point
# ===========================================================================

def build_ticker_features(
    ticker: str,
    df_30m: pd.DataFrame,
    context_data: dict[str, pd.DataFrame],
    meta: dict,
) -> pd.DataFrame | None:
    """
    Build all 11 feature categories for a single ticker's 30-minute OHLCV data.

    Parameters
    ----------
    ticker       : ticker symbol
    df_30m       : raw 30-minute OHLCV DataFrame, UTC DatetimeIndex
    context_data : {"SPY": df_spy_30m, "QQQ": ..., "IWM": ...}
    meta         : {"sector_id": int, "market_cap_bucket": int, "asset_type": int}

    Returns
    -------
    DataFrame with feature columns indexed by timestamp, or None if too few bars.
    """
    df = df_30m.copy()

    # Ensure lowercase OHLCV column names
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        logger.warning("[%s] missing OHLCV columns, skipping", ticker)
        return None

    if len(df) < MIN_BARS:
        logger.warning("[%s] only %d bars — below MIN_BARS=%d, skipping", ticker, len(df), MIN_BARS)
        return None

    # ------------------------------------------------------------------
    # Base ATR (used by almost every subsequent category)
    # ------------------------------------------------------------------
    try:
        import pandas_ta as ta
        atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["atr_14"] = atr_series.values if atr_series is not None else np.nan
    except Exception:
        df["atr_14"] = np.nan

    # ------------------------------------------------------------------
    # Existing custom indicators (causal swing state only)
    # Note: add_fractal_pivots uses 3-bar lookahead, so we don't use it in features.
    # Pivots are computed only in the label builder for label generation.
    # ------------------------------------------------------------------
    try:
        df = add_atr_swing_state_features(df)
    except Exception as exc:
        logger.debug("[%s] add_atr_swing_state_features failed: %s", ticker, exc)

    # ------------------------------------------------------------------
    # Session date index (NY timezone) — needed for multiple categories
    # ------------------------------------------------------------------
    idx_ny = df.index.tz_convert("America/New_York")
    session_dates = pd.Series(idx_ny.date, index=df.index)

    # ------------------------------------------------------------------
    # Build features by category
    # ------------------------------------------------------------------
    df = _add_cat10_time(df, session_dates)   # needed by cat 6 open-window vol
    df = _add_cat1_price(df)
    df = _add_cat2_volatility(df)
    df = _add_cat3_trend(df)
    df = _add_oscillators(df)                 # momentum oscillators (rsi, macd, bb, stoch, mfi, chop)
    df = _add_cat6_volume(df)                 # volume_rel_20 needed by cat 4
    df = _add_cat4_swing(df)
    df = _add_cat5_model_placeholders(df)
    df = _add_cat7_gap(df, session_dates)
    df = _add_cat8_context(df, ticker, context_data)
    df = _add_cat9_location(df, session_dates)
    df = _add_daily_context(df, session_dates)   # daily HTF features (shift(1) — no leakage)
    df = _add_cat11_behavioral(df, meta)
    df = _finalize_open_window_vol(df)

    return df


# ===========================================================================
# Pipeline orchestrator
# ===========================================================================

def _load_context_data(
    context_tickers: list[str],
    raw_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Load raw 30m data for context tickers."""
    ctx: dict[str, pd.DataFrame] = {}
    for sym in context_tickers:
        path = raw_dir / f"{sym}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]
            if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df = df.set_index("timestamp")
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            ctx[sym] = df.sort_index()
        else:
            logger.warning("Context ticker %s not found at %s", sym, path)
    return ctx


def _build_sector_cap_encodings(universe: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Return (sector_enc, cap_enc, asset_enc) dicts mapping string → int."""
    sectors = sorted(universe["sector"].unique())

    # Handle both old and new CSV column names
    cap_col = "market_cap_bucket" if "market_cap_bucket" in universe.columns else "cap_bucket"
    type_col = "type" if "type" in universe.columns else "asset_type"

    # Get unique values from the CSV (don't hardcode)
    caps   = sorted(universe[cap_col].dropna().unique().tolist())
    assets = sorted(universe[type_col].dropna().unique().tolist())

    cap_enc = {c: i for i, c in enumerate(caps)}
    # ETFs have NaN market_cap_bucket in the universe CSV (they don't have a market cap).
    # Give them a dedicated bucket rather than silently falling back to index 0 (="Large").
    cap_enc["ETF"] = len(caps)

    return (
        {s: i for i, s in enumerate(sectors)},
        cap_enc,
        {a: i for i, a in enumerate(assets)},
    )


def build_all_features(
    *,
    universe_csv: Path | str = UNIVERSE_CSV,
    raw_30m_dir: Path = RAW_30M_DIR,
    processed_30m_dir: Path = PROCESSED_30M_DIR,
    combined_path: Path = FEATURES_COMBINED,
    context_tickers: list[str] = CONTEXT_TICKERS,
    force: bool = False,
) -> None:
    """
    Build 30m features for every universe ticker, save per-ticker parquets,
    then combine into a single (ticker, timestamp) MultiIndex parquet.
    """
    if combined_path.exists() and not force:
        logger.info("Combined features already exist at %s — skipping. Use force=True to rebuild.", combined_path)
        return

    universe = load_universe(universe_csv)
    tickers  = universe["ticker"].tolist()
    sector_enc, cap_enc, asset_enc = _build_sector_cap_encodings(universe)

    processed_30m_dir.mkdir(parents=True, exist_ok=True)

    # Load context tickers once
    ctx = _load_context_data(context_tickers, raw_30m_dir)

    combined_writer: pq.ParquetWriter | None = None
    combined_rows = 0
    n = len(tickers)

    try:
        for i, row in universe.iterrows():
            ticker = row["ticker"]
            logger.info("(%d/%d) Building features for %s", i + 1, n, ticker)

            per_ticker_path = processed_30m_dir / f"{ticker}_features.parquet"
            if per_ticker_path.exists() and not force:
                logger.info("[%s] cached — loading", ticker)
                df_feat = pd.read_parquet(per_ticker_path)
            else:
                try:
                    df_raw = load_raw_30m(ticker, raw_30m_dir)
                except FileNotFoundError:
                    logger.warning("[%s] raw 30m data missing — skipping", ticker)
                    continue

                # Handle both old and new CSV column names
                cap_col = "market_cap_bucket" if "market_cap_bucket" in row.index else "cap_bucket"
                type_col = "type" if "type" in row.index else "asset_type"

                # ETFs have NaN market_cap_bucket — route them to the dedicated ETF bucket
                if str(row.get(type_col, "Stock")) == "ETF":
                    cap_bucket = cap_enc.get("ETF", len(cap_enc) - 1)
                else:
                    cap_bucket = cap_enc.get(row[cap_col], 0)

                meta = {
                    "sector_id":         sector_enc.get(row["sector"], 0),
                    "market_cap_bucket": cap_bucket,
                    "asset_type":        asset_enc.get(row[type_col], 0),
                }

                df_feat = build_ticker_features(ticker, df_raw, ctx, meta)
                if df_feat is None:
                    continue

                df_feat = _downcast_numeric(df_feat)
                df_feat.to_parquet(per_ticker_path)
                logger.info("[%s] done (%d bars)", ticker, len(df_feat))

            out = df_feat.reset_index().rename(columns={"index": "timestamp"})
            if "timestamp" not in out.columns:
                out = out.rename(columns={out.columns[0]: "timestamp"})
            out["ticker"] = ticker
            out = _downcast_numeric(out)
            combined_writer = _append_parquet_frame(combined_writer, out, combined_path)
            combined_rows += len(out)
    finally:
        if combined_writer is not None:
            combined_writer.close()

    if combined_rows == 0:
        logger.error("No feature frames built — check raw data.")
        return

    logger.info("Saved combined features → %s  (%d rows)", combined_path, combined_rows)
