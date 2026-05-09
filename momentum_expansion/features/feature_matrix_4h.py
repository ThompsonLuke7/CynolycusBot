"""
4H feature matrix for momentum_expansion.

All features are CAUSAL (no forward leakage) and cross-ticker comparable
(ATR-normalized / pct-change / cross-sectional rank).

Feature categories:
  trend        — EMA distance/slope, ADX, MA stack
  momentum     — multi-bar returns, RSI, MACD-hist (ATR-norm)
  rs           — relative strength vs SPY/QQQ/sector ETF, beta-like
  volume       — RVOL, dollar-volume rank
  volatility   — ATR pct, ATR expansion ratio, vol-of-vol
  structure    — distance to 52-week high, base/breakout, range position
  regime       — SPY trend state, VIX bucket
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from momentum_expansion.config.momentum_config import (
    CONTEXT_TICKERS,
    FEATURES_COMBINED,
    MIN_4H_BARS,
    PROCESSED_FEAT_DIR,
    RAW_4H_DIR,
    SECTOR_ETFS,
)
from momentum_expansion.data.load_bars import load_1d, load_4h

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector mapping (used for sector relative-strength feature)
# ---------------------------------------------------------------------------

# Coarse mapping of common tickers -> sector ETF. Tickers absent from this
# map fall back to XLK (broad). Light-weight; the universe selector / live
# system can override via config later.
SECTOR_MAP: dict[str, str] = {
    # Tech
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "AMD": "XLK",
    "ORCL": "XLK", "CRM": "XLK", "ADBE": "XLK", "INTC": "XLK", "MU": "XLK",
    "TSM": "XLK", "ASML": "XLK", "ARM": "XLK", "SMCI": "XLK", "PANW": "XLK",
    "CRWD": "XLK", "NET": "XLK", "DDOG": "XLK", "SNOW": "XLK", "PLTR": "XLK",
    "ZS": "XLK", "MSTR": "XLK",
    # Comm / Internet
    "GOOGL": "XLC", "GOOG": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC",
    "CMCSA": "XLC", "T": "XLC", "VZ": "XLC", "TMUS": "XLC",
    # Cons Disc
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "LOW": "XLY", "NKE": "XLY",
    "MCD": "XLY", "SBUX": "XLY", "F": "XLY", "GM": "XLY", "RIVN": "XLY",
    "LCID": "XLY", "ABNB": "XLY", "UBER": "XLY", "LYFT": "XLY", "DASH": "XLY",
    "BABA": "XLY", "JD": "XLY", "PDD": "XLY", "SHOP": "XLY",
    # Cons Staples
    "WMT": "XLP", "COST": "XLP", "TGT": "XLP", "PG": "XLP", "KO": "XLP",
    "PEP": "XLP", "PM": "XLP", "MO": "XLP",
    # Financials
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF",
    "C": "XLF", "V": "XLF", "MA": "XLF", "AXP": "XLF", "PYPL": "XLF",
    "SQ": "XLF", "COIN": "XLF", "SCHW": "XLF",
    # Healthcare
    "UNH": "XLV", "JNJ": "XLV", "LLY": "XLV", "PFE": "XLV", "MRK": "XLV",
    "ABBV": "XLV", "TMO": "XLV", "ABT": "XLV",
    # Energy
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "OXY": "XLE", "SLB": "XLE",
    "EOG": "XLE", "MPC": "XLE", "PSX": "XLE", "MARA": "XLE", "RIOT": "XLE",
    # Industrials
    "CAT": "XLI", "DE": "XLI", "BA": "XLI", "GE": "XLI", "HON": "XLI",
    "LMT": "XLI", "RTX": "XLI",
}


def _sector_etf_for(ticker: str) -> str:
    return SECTOR_MAP.get(ticker.upper(), "XLK")


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, lo, c_p = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h - lo), (h - c_p).abs(), (lo - c_p).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    try:
        import pandas_ta as ta
        out = ta.adx(df["high"], df["low"], df["close"], length=length)
        if out is None or out.empty:
            return pd.Series(np.nan, index=df.index)
        return out.iloc[:, 0].clip(0, 100)
    except Exception:
        return pd.Series(np.nan, index=df.index)


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    try:
        import pandas_ta as ta
        out = ta.rsi(close, length=length)
        return out if out is not None else pd.Series(np.nan, index=close.index)
    except Exception:
        return pd.Series(np.nan, index=close.index)


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    try:
        import pandas_ta as ta
        out = ta.macd(close, fast=fast, slow=slow, signal=signal)
        if out is None or out.empty:
            return pd.Series(np.nan, index=close.index)
        col = [c for c in out.columns if c.startswith("MACDh")]
        if not col:
            return pd.Series(np.nan, index=close.index)
        return out[col[0]]
    except Exception:
        return pd.Series(np.nan, index=close.index)


# ---------------------------------------------------------------------------
# Per-ticker builder
# ---------------------------------------------------------------------------

def build_ticker_features_4h(
    *,
    ticker: str,
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame | None,
    ctx_4h: dict[str, pd.DataFrame],
) -> pd.DataFrame | None:
    """
    Build 4H features for one ticker.

    Parameters
    ----------
    df_4h : 4H OHLCV with UTC DatetimeIndex (from data.load_bars.load_4h)
    df_1d : daily OHLCV (used for 52-week high distance / daily trend)
    ctx_4h: {"SPY": df, "QQQ": df, "IWM": df, "VIXY": df, "XLK": df, ...} on 4H
    """
    if df_4h is None or len(df_4h) < MIN_4H_BARS:
        return None

    df = df_4h.copy()
    df.columns = [c.lower() for c in df.columns]
    if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
        return None

    c = df["close"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"]

    # --- ATR (used everywhere)
    atr14 = _atr(df, 14)
    atr60 = _atr(df, 60)
    df["atr_14"] = atr14
    df["atr_pct_14"] = atr14 / c.replace(0, np.nan)

    # --- TREND ---
    ema10 = _ema(c, 10)
    ema20 = _ema(c, 20)
    ema50 = _ema(c, 50)
    ema100 = _ema(c, 100)
    df["ema_dist_10"] = ((c - ema10) / atr14).clip(-10, 10)
    df["ema_dist_20"] = ((c - ema20) / atr14).clip(-10, 10)
    df["ema_dist_50"] = ((c - ema50) / atr14).clip(-10, 10)
    df["ema_dist_100"] = ((c - ema100) / atr14).clip(-15, 15)
    df["ema_slope_10"] = ema10.pct_change(1)
    df["ema_slope_20"] = ema20.pct_change(1)
    df["ema_slope_50"] = ema50.pct_change(1)
    df["adx_14"] = _adx(df, 14)
    # Stack: bit0 ema10>ema20, bit1 ema20>ema50, bit2 ema50>ema100
    b0 = (ema10 > ema20).astype(int)
    b1 = (ema20 > ema50).astype(int)
    b2 = (ema50 > ema100).astype(int)
    df["ema_stack_4"] = b0 + 2 * b1 + 4 * b2

    # --- MOMENTUM ---
    for n in (1, 3, 5, 10, 20):
        df[f"ret_{n}"] = c.pct_change(n)
    df["rsi_14"] = _rsi(c, 14)
    df["macd_hist_norm"] = (_macd_hist(c) / atr14).clip(-5, 5)

    # --- VOLATILITY ---
    df["atr_expand_14_60"] = (atr14 / atr60).clip(0, 5)
    log_ret = np.log(c / c.shift(1))
    rv5 = log_ret.rolling(5).std()
    rv20 = log_ret.rolling(20).std()
    rv60 = log_ret.rolling(60).std()
    df["realized_vol_5"] = rv5
    df["realized_vol_20"] = rv20
    df["vol_regime_5_60"] = (rv5 / rv60.replace(0, np.nan)).clip(0, 5)
    df["vol_of_vol_20"] = rv5.rolling(20).std()

    # --- VOLUME ---
    vol_mean20 = v.rolling(20).mean().replace(0, np.nan)
    df["rvol_20"] = (v / vol_mean20).clip(0, 20)
    df["dollar_volume"] = (v * c)
    df["dollar_vol_pctile_252"] = (v * c).rolling(252, min_periods=50).rank(pct=True)
    df["dollar_vol_surge_20"] = ((v * c) / (v * c).rolling(20).mean().replace(0, np.nan)).clip(0, 20)

    # --- STRUCTURE ---
    # 52-week high distance — at 2 4H bars/day * ~252 trading days = ~504 bars
    roll_high_252d = h.rolling(504).max()
    roll_low_252d = lo.rolling(504).min()
    df["dist_to_52w_high_atr"] = ((roll_high_252d - c) / atr14).clip(0, 50)
    df["dist_to_52w_low_atr"] = ((c - roll_low_252d) / atr14).clip(0, 50)

    # 20-bar range position
    rh20 = h.rolling(20).max()
    rl20 = lo.rolling(20).min()
    rr20 = (rh20 - rl20).replace(0, np.nan)
    df["range_pos_20"] = ((c - rl20) / rr20).clip(0, 1)
    df["dist_20bar_high_atr"] = ((rh20 - c) / atr14).clip(0, 20)

    # Compression: vol_5 / vol_20 — lower = tighter
    df["compression_5_20"] = (rv5 / rv20.replace(0, np.nan)).clip(0, 5)

    # Breakout flag — close > 20-bar high (causal: shifted by 1 to avoid same-bar)
    df["breakout_20"] = (c > rh20.shift(1)).astype(float)

    # Base depth: 60-bar drawdown from rolling 60-bar high
    rh60 = h.rolling(60).max()
    df["drawdown_from_60h"] = ((rh60 - c) / rh60.replace(0, np.nan)).clip(0, 1)

    # --- RELATIVE STRENGTH ---
    def _ctx_close(name: str) -> pd.Series:
        df_ctx = ctx_4h.get(name)
        if df_ctx is None or df_ctx.empty:
            return pd.Series(np.nan, index=df.index)
        return df_ctx["close"].reindex(df.index, method="ffill")

    spy_c = _ctx_close("SPY")
    qqq_c = _ctx_close("QQQ")
    iwm_c = _ctx_close("IWM")
    sector_etf = _sector_etf_for(ticker)
    sec_c = _ctx_close(sector_etf)

    for n in (1, 5, 20):
        spy_ret = spy_c.pct_change(n)
        qqq_ret = qqq_c.pct_change(n)
        sec_ret = sec_c.pct_change(n)
        df[f"rs_spy_{n}"] = (df[f"ret_{n}"] - spy_ret).clip(-1, 1) if f"ret_{n}" in df.columns else np.nan
        if n in (1, 5, 20):
            df[f"rs_qqq_{n}"] = (df[f"ret_{n}"] - qqq_ret).clip(-1, 1)
        if n in (5, 20):
            df[f"rs_sector_{n}"] = (df[f"ret_{n}"] - sec_ret).clip(-1, 1)

    # Beta & corr to SPY (60-bar)
    own_lr = log_ret
    spy_lr = np.log(spy_c / spy_c.shift(1))
    if ticker == "SPY":
        df["beta_spy_60"] = 1.0
        df["corr_spy_60"] = 1.0
    else:
        cov60 = own_lr.rolling(60).cov(spy_lr)
        var60 = spy_lr.rolling(60).var().replace(0, np.nan)
        df["beta_spy_60"] = (cov60 / var60).clip(-5, 5)
        df["corr_spy_60"] = own_lr.rolling(60).corr(spy_lr)

    # --- REGIME ---
    spy_trend = np.sign(_ema(spy_c, 20) - _ema(spy_c, 50))
    df["regime_spy_trend"] = spy_trend
    spy_ret_20 = spy_c.pct_change(20)
    df["regime_spy_ret_20"] = spy_ret_20.fillna(0)

    vixy_c = _ctx_close("VIXY")
    if vixy_c.notna().any():
        vixy_z = (vixy_c - vixy_c.rolling(60).mean()) / vixy_c.rolling(60).std().replace(0, np.nan)
        df["regime_vix_z"] = vixy_z.clip(-5, 5)
        df["regime_vix_high"] = (vixy_z > 1.0).astype(float)
    else:
        df["regime_vix_z"] = 0.0
        df["regime_vix_high"] = 0.0

    # --- DAILY HTF context (shifted -1 to avoid leakage) ---
    if df_1d is not None and len(df_1d) > 60:
        d = df_1d.copy()
        d.columns = [x.lower() for x in d.columns]
        dc = d["close"]
        d_ret_1 = dc.pct_change(1)
        d_ret_5 = dc.pct_change(5)
        d_atr = _atr(d, 14)
        d_atr_pct = (d_atr / dc).shift(1)
        d_rsi = _rsi(dc, 14).shift(1)
        d_ema20 = _ema(dc, 20)
        d_ema50 = _ema(dc, 50)
        d_trend = np.sign(d_ema20 - d_ema50).shift(1)
        d_high_252 = d["high"].rolling(252).max()
        d_dist_52w = ((d_high_252 - dc) / d_atr.replace(0, np.nan)).shift(1).clip(0, 50)

        # Map daily series to 4H bars by date
        ny = df.index.tz_convert("America/New_York")
        date_index = pd.Series(ny.date, index=df.index)
        d.index = pd.to_datetime(d.index)
        d_dates = pd.Series(d.index.tz_convert("America/New_York").date if d.index.tz else pd.to_datetime(d.index).date,
                            index=d.index)

        def _map_to_4h(daily_series: pd.Series) -> pd.Series:
            mapping = pd.Series(daily_series.values, index=d_dates.values)
            mapping = mapping[~mapping.index.duplicated(keep="last")]
            return pd.Series([mapping.get(dt, np.nan) for dt in date_index.values],
                             index=df.index, dtype=float)

        df["daily_ret_1"] = _map_to_4h(d_ret_1.shift(1))
        df["daily_ret_5"] = _map_to_4h(d_ret_5.shift(1))
        df["daily_atr_pct"] = _map_to_4h(d_atr_pct)
        df["daily_rsi_14"] = _map_to_4h(d_rsi)
        df["daily_trend_state"] = _map_to_4h(d_trend)
        df["daily_dist_52w_atr"] = _map_to_4h(d_dist_52w)
    else:
        df["daily_ret_1"] = np.nan
        df["daily_ret_5"] = np.nan
        df["daily_atr_pct"] = np.nan
        df["daily_rsi_14"] = np.nan
        df["daily_trend_state"] = np.nan
        df["daily_dist_52w_atr"] = np.nan

    # --- BAR TIME (within session) ---
    ny = df.index.tz_convert("America/New_York")
    minutes = ny.hour * 60 + ny.minute
    df["is_morning_4h"] = (minutes < 13 * 60 + 30).astype(float)
    df["dow"] = pd.Series(ny.dayofweek.astype(float), index=df.index)

    return df


# ---------------------------------------------------------------------------
# Final ordered feature list (used by training + inference)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS_4H: list[str] = [
    # ATR baseline
    "atr_pct_14", "atr_expand_14_60",
    # Trend
    "ema_dist_10", "ema_dist_20", "ema_dist_50", "ema_dist_100",
    "ema_slope_10", "ema_slope_20", "ema_slope_50",
    "adx_14", "ema_stack_4",
    # Momentum
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "rsi_14", "macd_hist_norm",
    # Volatility
    "realized_vol_5", "realized_vol_20", "vol_regime_5_60", "vol_of_vol_20",
    # Volume
    "rvol_20", "dollar_vol_pctile_252", "dollar_vol_surge_20",
    # Structure
    "dist_to_52w_high_atr", "dist_to_52w_low_atr",
    "range_pos_20", "dist_20bar_high_atr",
    "compression_5_20", "breakout_20", "drawdown_from_60h",
    # Relative strength
    "rs_spy_1", "rs_spy_5", "rs_spy_20",
    "rs_qqq_1", "rs_qqq_5", "rs_qqq_20",
    "rs_sector_5", "rs_sector_20",
    "beta_spy_60", "corr_spy_60",
    # Regime
    "regime_spy_trend", "regime_spy_ret_20",
    "regime_vix_z", "regime_vix_high",
    # Daily HTF
    "daily_ret_1", "daily_ret_5", "daily_atr_pct", "daily_rsi_14",
    "daily_trend_state", "daily_dist_52w_atr",
    # Time
    "is_morning_4h", "dow",
]


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def _load_context_4h() -> dict[str, pd.DataFrame]:
    ctx: dict[str, pd.DataFrame] = {}
    for sym in list(CONTEXT_TICKERS) + list(SECTOR_ETFS):
        path = RAW_4H_DIR / f"{sym}.parquet"
        if not path.exists():
            logger.warning("Context 4h missing for %s — features that need it will be NaN", sym)
            continue
        try:
            ctx[sym] = load_4h(sym)
        except Exception as exc:
            logger.warning("[%s] load_4h failed: %s", sym, exc)
    return ctx


def build_all_features_4h(
    *,
    tickers: Iterable[str],
    out_path: Path = FEATURES_COMBINED,
    force: bool = False,
) -> pd.DataFrame | None:
    """
    Build 4H features for every ticker that has cached 4H data, persist
    per-ticker parquets, and combine into one (timestamp, ticker)
    MultiIndex parquet.
    """
    PROCESSED_FEAT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        logger.info("Features already exist at %s. Use force=True to rebuild.", out_path)
        return pd.read_parquet(out_path)

    ctx_4h = _load_context_4h()
    frames: list[pd.DataFrame] = []
    tickers = list(tickers)
    for i, t in enumerate(tickers, 1):
        per_path = PROCESSED_FEAT_DIR / f"{t}_features.parquet"
        if per_path.exists() and not force:
            df_feat = pd.read_parquet(per_path)
            df_feat["ticker"] = t
            frames.append(df_feat)
            continue

        try:
            df_4h = load_4h(t)
        except FileNotFoundError:
            logger.info("[%s] no 4h data — skip", t)
            continue
        try:
            df_1d = load_1d(t)
        except FileNotFoundError:
            df_1d = None

        feats = build_ticker_features_4h(
            ticker=t, df_4h=df_4h, df_1d=df_1d, ctx_4h=ctx_4h
        )
        if feats is None:
            logger.info("[%s] insufficient bars or missing OHLCV — skip", t)
            continue
        feats.to_parquet(per_path)
        feats["ticker"] = t
        frames.append(feats)
        if i % 25 == 0:
            logger.info("(%d/%d) features built for %s", i, len(tickers), t)

    if not frames:
        logger.error("No features built.")
        return None

    combined = pd.concat(frames, axis=0)
    combined = combined.reset_index().rename(columns={"index": "timestamp"})
    ts_col = "timestamp" if "timestamp" in combined.columns else combined.columns[0]
    combined = combined.set_index([ts_col, "ticker"])
    combined.to_parquet(out_path)
    logger.info("Saved combined 4H features (%d rows) -> %s", len(combined), out_path)
    return combined
