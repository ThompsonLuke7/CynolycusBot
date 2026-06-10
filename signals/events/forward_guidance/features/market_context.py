"""Market reaction/context features and leakage-safe forward labels."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from signals.events.forward_guidance.data.schema import EarningsEvent


def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={"index": "timestamp"})
        else:
            raise ValueError("bars must include timestamp column or DatetimeIndex")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    for col in ("open", "high", "low", "close", "volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def intraday_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    bars = _ensure_timestamp(df)
    if bars.empty:
        return pd.DataFrame(columns=["session", "open", "high", "low", "close", "volume"])
    ny = bars["timestamp"].dt.tz_convert("America/New_York")
    bars["session"] = ny.dt.tz_localize(None).dt.normalize()
    grouped = bars.groupby("session", as_index=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return add_daily_indicators(grouped)


def add_daily_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.sort_values("session").reset_index(drop=True).copy()
    if df.empty:
        return df
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = tr.rolling(14, min_periods=3).mean()
    df["rvol_20"] = df["volume"] / df["volume"].shift(1).rolling(20, min_periods=5).mean()
    df["ret_1d"] = df["close"].pct_change()
    df["ret_20d"] = df["close"].pct_change(20)
    df["ret_63d"] = df["close"].pct_change(63)
    df["ma_20"] = df["close"].rolling(20, min_periods=5).mean()
    df["ma_50"] = df["close"].rolling(50, min_periods=10).mean()
    return df


def _daily_for(symbol: str, bars_by_symbol: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    df = bars_by_symbol.get(symbol.upper())
    if df is None or df.empty:
        return pd.DataFrame()
    return intraday_to_daily(df)


def _row_at_or_before(daily: pd.DataFrame, session: pd.Timestamp) -> tuple[int | None, pd.Series | None]:
    if daily.empty:
        return None, None
    sessions = pd.to_datetime(daily["session"])
    matches = daily.loc[sessions <= session]
    if matches.empty:
        return None, None
    idx = int(matches.index[-1])
    return idx, daily.loc[idx]


def _value(row: pd.Series | None, col: str) -> float:
    if row is None or col not in row:
        return float("nan")
    try:
        return float(row[col])
    except Exception:
        return float("nan")


def compute_market_context(event: EarningsEvent, bars_by_symbol: Mapping[str, pd.DataFrame]) -> dict[str, float]:
    symbol = event.clean_ticker
    reaction = pd.Timestamp(event.reaction_date).normalize()
    daily = _daily_for(symbol, bars_by_symbol)
    idx, row = _row_at_or_before(daily, reaction)
    features: dict[str, float] = {
        "post_er_gap_pct": float("nan"),
        "post_er_move_pct": float("nan"),
        "intraday_reversal": float("nan"),
        "prior_3m_momentum": float("nan"),
        "atr_pct_14": float("nan"),
        "rvol_20": float("nan"),
        "sector_strength_20d": float("nan"),
        "spy_regime": float("nan"),
        "qqq_regime": float("nan"),
        "vix_regime": float("nan"),
        "bad_initial_reaction_flag": 0.0,
        "technical_stabilization_flag": 0.0,
    }
    if row is not None and idx is not None and idx > 0:
        prev = daily.loc[idx - 1]
        prev_close = _value(prev, "close")
        open_px = _value(row, "open")
        close_px = _value(row, "close")
        high_px = _value(row, "high")
        low_px = _value(row, "low")
        atr = _value(row, "atr_14")
        if prev_close > 0:
            features["post_er_gap_pct"] = open_px / prev_close - 1.0
            features["post_er_move_pct"] = close_px / prev_close - 1.0
        if open_px > 0:
            features["intraday_reversal"] = close_px / open_px - 1.0
        if close_px > 0:
            features["atr_pct_14"] = atr / close_px if atr == atr else float("nan")
        features["rvol_20"] = _value(row, "rvol_20")
        features["prior_3m_momentum"] = _value(prev, "ret_63d")
        bar_range = high_px - low_px
        close_location = (close_px - low_px) / bar_range if bar_range > 0 else float("nan")
        features["bad_initial_reaction_flag"] = float(features["post_er_move_pct"] < 0 or features["post_er_gap_pct"] < -0.02)
        features["technical_stabilization_flag"] = float(
            (features["intraday_reversal"] > 0) or (close_location == close_location and close_location >= 0.55)
        )

    sector = event.sector_etf
    if sector:
        sector_daily = _daily_for(sector, bars_by_symbol)
        _, sector_row = _row_at_or_before(sector_daily, reaction)
        features["sector_strength_20d"] = _value(sector_row, "ret_20d")

    for symbol_name, feature_name in (("SPY", "spy_regime"), ("QQQ", "qqq_regime")):
        ctx_daily = _daily_for(symbol_name, bars_by_symbol)
        _, ctx_row = _row_at_or_before(ctx_daily, reaction)
        close = _value(ctx_row, "close")
        ma20 = _value(ctx_row, "ma_20")
        ma50 = _value(ctx_row, "ma_50")
        if close == close and ma20 == ma20 and ma50 == ma50:
            features[feature_name] = float((close > ma20) + (ma20 > ma50)) / 2.0

    vix_daily = _daily_for("VIXY", bars_by_symbol)
    _, vix_row = _row_at_or_before(vix_daily, reaction)
    vix_ret20 = _value(vix_row, "ret_20d")
    if vix_ret20 == vix_ret20:
        features["vix_regime"] = float(vix_ret20 > 0)

    return features


def _future_close(daily: pd.DataFrame, start_idx: int, horizon: int) -> float:
    idx = start_idx + horizon
    if idx >= len(daily):
        return float("nan")
    return float(daily.loc[idx, "close"])


def _window_returns(daily: pd.DataFrame, start_idx: int, horizon: int) -> tuple[float, float, float]:
    start_close = float(daily.loc[start_idx, "close"])
    future = daily.iloc[start_idx + 1 : min(len(daily), start_idx + horizon + 1)]
    if start_close <= 0 or future.empty:
        return float("nan"), float("nan"), float("nan")
    terminal = _future_close(daily, start_idx, horizon)
    fwd_ret = terminal / start_close - 1.0 if terminal == terminal else float("nan")
    highs = future["high"] / start_close - 1.0
    lows = future["low"] / start_close - 1.0
    return fwd_ret, float(highs.max()), float(lows.min())


def _benchmark_return(symbol: str | None, bars_by_symbol: Mapping[str, pd.DataFrame], reaction: pd.Timestamp, horizon: int) -> float:
    if not symbol:
        return float("nan")
    daily = _daily_for(symbol, bars_by_symbol)
    idx, _ = _row_at_or_before(daily, reaction)
    if idx is None:
        return float("nan")
    start = float(daily.loc[idx, "close"])
    end = _future_close(daily, idx, horizon)
    if start <= 0 or not math.isfinite(end):
        return float("nan")
    return end / start - 1.0


def compute_forward_labels(event: EarningsEvent, bars_by_symbol: Mapping[str, pd.DataFrame]) -> dict[str, float]:
    daily = _daily_for(event.clean_ticker, bars_by_symbol)
    reaction = pd.Timestamp(event.reaction_date).normalize()
    idx, _ = _row_at_or_before(daily, reaction)
    labels: dict[str, float] = {
        "fwd_ret_5d": float("nan"),
        "fwd_ret_20d": float("nan"),
        "fwd_ret_60d": float("nan"),
        "fwd_60d_excess_ret_vs_spy": float("nan"),
        "fwd_60d_excess_ret_vs_sector": float("nan"),
        "max_runup": float("nan"),
        "max_drawdown": float("nan"),
        "target": float("nan"),
    }
    if idx is None:
        return labels
    for horizon in (5, 20, 60):
        fwd_ret, max_runup, max_drawdown = _window_returns(daily, idx, horizon)
        labels[f"fwd_ret_{horizon}d"] = fwd_ret
        if horizon == 60:
            labels["max_runup"] = max_runup
            labels["max_drawdown"] = max_drawdown
    spy_ret = _benchmark_return("SPY", bars_by_symbol, reaction, 60)
    sector_ret = _benchmark_return(event.sector_etf, bars_by_symbol, reaction, 60)
    if labels["fwd_ret_60d"] == labels["fwd_ret_60d"] and spy_ret == spy_ret:
        labels["fwd_60d_excess_ret_vs_spy"] = labels["fwd_ret_60d"] - spy_ret
    if labels["fwd_ret_60d"] == labels["fwd_ret_60d"] and sector_ret == sector_ret:
        labels["fwd_60d_excess_ret_vs_sector"] = labels["fwd_ret_60d"] - sector_ret
        labels["target"] = float(labels["fwd_60d_excess_ret_vs_sector"] > 0.10)
    return labels
