from __future__ import annotations

import argparse
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from API.Alpaca_API.inference.live_inference import LiveMetaXGBAgent
from scripts.compare_baseline_vs_profit_protect_exit import _load_one_min
from scripts.compare_hybrid_1m_vs_next_open import _equity_curve_from_events, _event_metrics
from scripts.replay_meta_independent import _load_meta_matrix, _normalize_bounds, _score_exit


@dataclass(frozen=True)
class EntryRule:
    name: str
    family: str
    confirm_bars: int = 1
    atr_mult: float = 0.0
    lookback: int = 0
    volume_mult: float = 0.0
    ema_length: int = 0


@dataclass
class TradeState:
    position: int = 0
    entry_price: float = np.nan
    entry_atr: float = np.nan
    bars_since_entry: int = 0
    favorable_anchor: float = np.nan
    adverse_anchor: float = np.nan
    tp_seen: bool = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep up to 20 cached 1m execution entry rules against the current 1-bar directional policy."
    )
    parser.add_argument(
        "--meta-matrix",
        default="Data/inference/spy/10min/debug_matrices_warmup/spy/live_meta_matrix_on_trace_ts_live_2026_03_27.parquet",
        help="Cached 10m meta matrix parquet.",
    )
    parser.add_argument(
        "--one-min-data",
        default="Data/raw/spy/spy_intraday_1min_live_2026_03_24_runtime_rth_cache.parquet",
        help="Raw 1m parquet for execution timing.",
    )
    parser.add_argument("--model-root", default="Data/models/meta_xgboost/10min", help="Meta model root.")
    parser.add_argument("--symbol", default="SPY", help="Symbol label.")
    parser.add_argument("--start", default=None, help="Optional UTC start timestamp.")
    parser.add_argument("--end", default=None, help="Optional UTC end timestamp.")
    parser.add_argument("--recent-days", type=int, default=60, help="Auto-window length when --start/--end are omitted.")
    parser.add_argument("--tz", default="America/New_York", help="Display timezone.")
    parser.add_argument("--entry-threshold", type=float, default=None, help="Optional override for both entry thresholds.")
    parser.add_argument("--exit-threshold", type=float, default=None, help="Optional override for both exit thresholds.")
    parser.add_argument("--min-hold-bars", type=int, default=2, help="Minimum 10m bars before soft exits.")
    parser.add_argument("--soft-exit-confirm-bars", type=int, default=2, help="Consecutive bars for soft exit confirmation.")
    parser.add_argument("--urgent-exit-prob", type=float, default=0.85, help="Immediate exit if p_exit_side exceeds this value.")
    parser.add_argument("--urgent-exit-delta", type=float, default=0.30, help="Immediate exit if p_exit_side - p_enter_side exceeds this value.")
    parser.add_argument("--opposite-dominance-delta", type=float, default=0.0, help="Opposite-side margin needed to invalidate a side intent.")
    parser.add_argument("--max-rules", type=int, default=20, help="Maximum number of entry rules to evaluate.")
    parser.add_argument(
        "--summary-out",
        default="Data/inference/spy/10min/meta/entry_execution_1m_sweep_summary.csv",
        help="Summary metrics CSV.",
    )
    parser.add_argument(
        "--events-out",
        default="Data/inference/spy/10min/meta/entry_execution_1m_sweep_events.csv",
        help="Combined event CSV with regime labels.",
    )
    parser.add_argument(
        "--equity-out",
        default="Data/inference/spy/10min/plots/entry_execution_1m_sweep_equity.png",
        help="Equity comparison PNG.",
    )
    return parser.parse_args()


def _candidate_rules() -> list[EntryRule]:
    return [
        EntryRule(name="baseline_next_open", family="next_open"),
        EntryRule(name="current_policy_trend_1bar", family="trend", confirm_bars=1),
        EntryRule(name="trend_2bar", family="trend", confirm_bars=2),
        EntryRule(name="trend_3bar", family="trend", confirm_bars=3),
        EntryRule(name="signal_open_cross", family="signal_open"),
        EntryRule(name="signal_close_cross", family="signal_close"),
        EntryRule(name="signal_body_break", family="signal_body"),
        EntryRule(name="signal_breakout_touch", family="breakout", atr_mult=0.0),
        EntryRule(name="signal_breakout_close", family="breakout_close", atr_mult=0.0),
        EntryRule(name="signal_breakout_0.05atr", family="breakout", atr_mult=0.05),
        EntryRule(name="signal_breakout_0.10atr", family="breakout", atr_mult=0.10),
        EntryRule(name="bos_2_touch", family="bos_touch", lookback=2),
        EntryRule(name="bos_3_touch", family="bos_touch", lookback=3),
        EntryRule(name="bos_2_close", family="bos_close", lookback=2),
        EntryRule(name="bos_3_close", family="bos_close", lookback=3),
        EntryRule(name="trend_volume_1.25x", family="trend_volume", confirm_bars=1, volume_mult=1.25),
        EntryRule(name="trend_volume_1.50x", family="trend_volume", confirm_bars=1, volume_mult=1.50),
        EntryRule(name="breakout_volume_1.25x", family="breakout_volume", atr_mult=0.0, volume_mult=1.25),
        EntryRule(name="support_resistance_bounce_5", family="sr_bounce", lookback=5, atr_mult=0.05),
        EntryRule(name="vwap_trend_1bar", family="vwap_trend", confirm_bars=1),
        EntryRule(name="ema8_pullback", family="ema_pullback", ema_length=8),
    ]


def _validity_flags(
    *,
    p_enter_long: float,
    p_enter_short: float,
    thr_enter_long: float,
    thr_enter_short: float,
    opposite_dominance_delta: float,
) -> tuple[bool, bool]:
    long_ready = np.isfinite(p_enter_long) and p_enter_long >= thr_enter_long
    short_ready = np.isfinite(p_enter_short) and p_enter_short >= thr_enter_short
    long_margin = (p_enter_long - thr_enter_long) if long_ready else -np.inf
    short_margin = (p_enter_short - thr_enter_short) if short_ready else -np.inf
    long_invalidated = bool(short_ready and short_margin > long_margin + opposite_dominance_delta)
    short_invalidated = bool(long_ready and long_margin > short_margin + opposite_dominance_delta)
    return bool(long_ready and not long_invalidated), bool(short_ready and not short_invalidated)


def _auto_bounds(
    *,
    meta_df: pd.DataFrame,
    one_min: pd.DataFrame,
    start_arg: str | None,
    end_arg: str | None,
    recent_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if start_arg or end_arg:
        start_ts, end_ts = _normalize_bounds(start_arg, end_arg)
        if start_ts is None:
            start_ts = min(pd.Timestamp(meta_df.index.min()), pd.Timestamp(one_min["timestamp"].min()))
        if end_ts is None:
            end_ts = max(pd.Timestamp(meta_df.index.max()) + pd.Timedelta(minutes=9), pd.Timestamp(one_min["timestamp"].max()))
        return start_ts, end_ts

    meta_end = pd.Timestamp(meta_df.index.max()) + pd.Timedelta(minutes=9)
    one_end = pd.Timestamp(one_min["timestamp"].max())
    end_ts = min(meta_end, one_end)
    start_ts = end_ts - pd.Timedelta(days=max(1, int(recent_days)))
    return start_ts, end_ts


def _build_one_min_feature_arrays(one_min: pd.DataFrame, *, tz: str) -> dict[str, object]:
    df = one_min.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")

    prev_vol_mean = df["volume"].shift(1).rolling(5, min_periods=3).mean()
    df["vol_ratio_5"] = df["volume"] / prev_vol_mean.replace(0.0, np.nan)
    df["prev_high_2"] = df["high"].shift(1).rolling(2, min_periods=2).max()
    df["prev_high_3"] = df["high"].shift(1).rolling(3, min_periods=3).max()
    df["prev_high_5"] = df["high"].shift(1).rolling(5, min_periods=5).max()
    df["prev_low_2"] = df["low"].shift(1).rolling(2, min_periods=2).min()
    df["prev_low_3"] = df["low"].shift(1).rolling(3, min_periods=3).min()
    df["prev_low_5"] = df["low"].shift(1).rolling(5, min_periods=5).min()
    df["ema_8"] = df["close"].ewm(span=8, adjust=False, min_periods=4).mean()

    local_ts = df["timestamp"].dt.tz_convert(tz)
    session_key = local_ts.dt.normalize()
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"].fillna(0.0)
    cum_pv = pv.groupby(session_key).cumsum()
    cum_vol = df["volume"].fillna(0.0).groupby(session_key).cumsum()
    df["session_vwap"] = cum_pv / cum_vol.replace(0.0, np.nan)

    return {
        "timestamp": df["timestamp"].tolist(),
        "open": df["open"].to_numpy(),
        "high": df["high"].to_numpy(),
        "low": df["low"].to_numpy(),
        "close": df["close"].to_numpy(),
        "volume": df["volume"].to_numpy(),
        "vol_ratio_5": df["vol_ratio_5"].to_numpy(),
        "prev_high_2": df["prev_high_2"].to_numpy(),
        "prev_high_3": df["prev_high_3"].to_numpy(),
        "prev_high_5": df["prev_high_5"].to_numpy(),
        "prev_low_2": df["prev_low_2"].to_numpy(),
        "prev_low_3": df["prev_low_3"].to_numpy(),
        "prev_low_5": df["prev_low_5"].to_numpy(),
        "ema_8": df["ema_8"].to_numpy(),
        "session_vwap": df["session_vwap"].to_numpy(),
    }


def _reset_intent_state(state: dict[str, object]) -> None:
    state["confirm_count"] = 0
    state["history"].clear()


def _append_history(state: dict[str, object], *, open_: float, high: float, low: float, close: float) -> None:
    state["history"].append({"open": open_, "high": high, "low": low, "close": close})


def _trigger_price_from_open(*, side: str, trigger: float, bar_open: float) -> float:
    if not np.isfinite(trigger):
        return float("nan")
    if not np.isfinite(bar_open):
        return float(trigger)
    if side == "long":
        return float(max(trigger, bar_open))
    return float(min(trigger, bar_open))


def _reset_trade_state() -> TradeState:
    return TradeState()


def _set_trade_entry(*, position: int, row: pd.Series, entry_price: float | None = None) -> TradeState:
    close = float(row.get("close", np.nan))
    atr = float(row.get("atr", np.nan))
    seed_entry = float(entry_price) if entry_price is not None and np.isfinite(entry_price) else close
    if not np.isfinite(seed_entry):
        seed_entry = close
    return TradeState(
        position=int(position),
        entry_price=float(seed_entry),
        entry_atr=float(atr) if np.isfinite(atr) and atr > 0.0 else float("nan"),
        bars_since_entry=0,
        favorable_anchor=float(close),
        adverse_anchor=float(close),
        tp_seen=False,
    )


def _advance_trade_state(
    *,
    state: TradeState,
    action: int,
    row: pd.Series,
    agent: LiveMetaXGBAgent,
) -> TradeState:
    high = float(row.get("high", np.nan))
    low = float(row.get("low", np.nan))
    close = float(row.get("close", np.nan))
    atr = float(row.get("atr", np.nan))

    if state.position != 0:
        state.bars_since_entry += 1
        if state.position > 0:
            if np.isfinite(high):
                state.favorable_anchor = max(float(state.favorable_anchor), high)
            if np.isfinite(low):
                state.adverse_anchor = min(float(state.adverse_anchor), low)
            tp = float(state.entry_price) + float(agent._a_tp) * float(state.entry_atr)
            if np.isfinite(high) and np.isfinite(tp) and high >= tp:
                state.tp_seen = True
        else:
            if np.isfinite(low):
                state.favorable_anchor = min(float(state.favorable_anchor), low)
            if np.isfinite(high):
                state.adverse_anchor = max(float(state.adverse_anchor), high)
            tp = float(state.entry_price) - float(agent._a_tp) * float(state.entry_atr)
            if np.isfinite(low) and np.isfinite(tp) and low <= tp:
                state.tp_seen = True

    if action == 0:
        return _reset_trade_state()
    if state.position == action:
        return state
    if not np.isfinite(close):
        return state
    return TradeState(
        position=int(action),
        entry_price=float(close),
        entry_atr=float(atr) if np.isfinite(atr) and atr > 0.0 else float("nan"),
        bars_since_entry=0,
        favorable_anchor=float(close),
        adverse_anchor=float(close),
        tp_seen=False,
    )


def _annotate_context(
    *,
    agent: LiveMetaXGBAgent,
    row: pd.Series,
    state: TradeState,
) -> pd.Series:
    out = row.copy()
    for side in ("long", "short"):
        out[f"in_{side}_trade"] = 0
        out[f"{side}_bars_since_entry"] = np.nan
        out[f"{side}_mfe_atr"] = np.nan
        out[f"{side}_mae_atr"] = np.nan
        out[f"{side}_tp_seen_run"] = 0
        out[f"{side}_trail_gap_atr"] = np.nan
        out[f"{side}_entry_price_ctx"] = np.nan
    if state.position == 0:
        return out

    entry = float(state.entry_price)
    atr_i = float(state.entry_atr)
    high = float(row.get("high", np.nan))
    low = float(row.get("low", np.nan))
    close = float(row.get("close", np.nan))
    if not np.isfinite(entry) or not np.isfinite(atr_i) or atr_i <= 0.0:
        return out

    side = "long" if state.position > 0 else "short"
    out[f"in_{side}_trade"] = 1
    out[f"{side}_bars_since_entry"] = float(state.bars_since_entry + 1)
    out[f"{side}_entry_price_ctx"] = entry

    if side == "long":
        favorable_anchor = max(float(state.favorable_anchor), high) if np.isfinite(high) else float(state.favorable_anchor)
        adverse_anchor = min(float(state.adverse_anchor), low) if np.isfinite(low) else float(state.adverse_anchor)
        tp = entry + float(agent._a_tp) * atr_i
        tp_seen = bool(state.tp_seen or (np.isfinite(high) and high >= tp))
        trail_active = (favorable_anchor - entry) >= float(agent._trail_activate_atr) * atr_i
        trail_dist = float(agent._trail_atr) * atr_i
        if tp_seen and bool(agent._use_tp_to_tighten_trail):
            trail_active = True
            trail_dist = min(trail_dist, max(float(agent._trail_atr_after_tp), 1e-9) * atr_i)
        trail_level = favorable_anchor - trail_dist if trail_active else np.nan
        out["long_mfe_atr"] = (favorable_anchor - entry) / atr_i
        out["long_mae_atr"] = (entry - adverse_anchor) / atr_i
        out["long_tp_seen_run"] = int(tp_seen)
        out["long_trail_gap_atr"] = ((close - trail_level) / atr_i) if np.isfinite(close) and np.isfinite(trail_level) else np.nan
        return out

    favorable_anchor = min(float(state.favorable_anchor), low) if np.isfinite(low) else float(state.favorable_anchor)
    adverse_anchor = max(float(state.adverse_anchor), high) if np.isfinite(high) else float(state.adverse_anchor)
    tp = entry - float(agent._a_tp) * atr_i
    tp_seen = bool(state.tp_seen or (np.isfinite(low) and low <= tp))
    trail_active = (entry - favorable_anchor) >= float(agent._trail_activate_atr) * atr_i
    trail_dist = float(agent._trail_atr) * atr_i
    if tp_seen and bool(agent._use_tp_to_tighten_trail):
        trail_active = True
        trail_dist = min(trail_dist, max(float(agent._trail_atr_after_tp), 1e-9) * atr_i)
    trail_level = favorable_anchor + trail_dist if trail_active else np.nan
    out["short_mfe_atr"] = (entry - favorable_anchor) / atr_i
    out["short_mae_atr"] = (adverse_anchor - entry) / atr_i
    out["short_tp_seen_run"] = int(tp_seen)
    out["short_trail_gap_atr"] = ((trail_level - close) / atr_i) if np.isfinite(close) and np.isfinite(trail_level) else np.nan
    return out


def _score_exit_for_state(
    *,
    agent: LiveMetaXGBAgent,
    row: pd.Series,
    state: TradeState,
    side: str,
) -> float:
    if side == "long" and state.position <= 0:
        return float("nan")
    if side == "short" and state.position >= 0:
        return float("nan")
    exit_row = _annotate_context(agent=agent, row=row, state=state)
    exit_df = pd.DataFrame([exit_row], index=[row.name])
    predictor = agent._exit_long if side == "long" else agent._exit_short
    return float(predictor.predict_row(exit_df, target_ts=row.name))


def _evaluate_entry(
    *,
    rule: EntryRule,
    side: str,
    one_min_idx: int,
    one_min_arrays: dict[str, object],
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    signal_open: float,
    signal_high: float,
    signal_low: float,
    signal_close: float,
    signal_atr: float,
    state: dict[str, object],
) -> tuple[bool, float, str | None]:
    family = str(rule.family)
    is_long = side == "long"
    atr = float(signal_atr) if np.isfinite(signal_atr) and signal_atr > 0.0 else float("nan")
    buffer_amt = float(rule.atr_mult) * atr if np.isfinite(atr) else float("nan")
    vol_ratio = float(one_min_arrays["vol_ratio_5"][one_min_idx]) if one_min_idx < len(one_min_arrays["vol_ratio_5"]) else float("nan")
    prev_high_key = f"prev_high_{int(rule.lookback)}"
    prev_low_key = f"prev_low_{int(rule.lookback)}"
    prev_high = float(one_min_arrays.get(prev_high_key, np.full(0, np.nan))[one_min_idx]) if prev_high_key in one_min_arrays else float("nan")
    prev_low = float(one_min_arrays.get(prev_low_key, np.full(0, np.nan))[one_min_idx]) if prev_low_key in one_min_arrays else float("nan")
    ema_8 = float(one_min_arrays["ema_8"][one_min_idx]) if one_min_idx < len(one_min_arrays["ema_8"]) else float("nan")
    vwap = float(one_min_arrays["session_vwap"][one_min_idx]) if one_min_idx < len(one_min_arrays["session_vwap"]) else float("nan")

    if family == "next_open":
        return (bool(np.isfinite(bar_open)), float(bar_open), "next_open") if np.isfinite(bar_open) else (False, float("nan"), None)

    if family == "trend":
        directional = bool(np.isfinite(bar_open) and np.isfinite(bar_close) and ((bar_close > bar_open) if is_long else (bar_close < bar_open)))
        state["confirm_count"] = int(state["confirm_count"]) + 1 if directional else 0
        if int(state["confirm_count"]) >= max(1, int(rule.confirm_bars)) and np.isfinite(bar_close):
            return True, float(bar_close), f"trend_{int(rule.confirm_bars)}bar"
        _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
        return False, float("nan"), None

    if family == "trend_volume":
        directional = bool(np.isfinite(bar_open) and np.isfinite(bar_close) and ((bar_close > bar_open) if is_long else (bar_close < bar_open)))
        vol_ok = bool(np.isfinite(vol_ratio) and vol_ratio >= float(rule.volume_mult))
        state["confirm_count"] = int(state["confirm_count"]) + 1 if directional else 0
        if int(state["confirm_count"]) >= max(1, int(rule.confirm_bars)) and vol_ok and np.isfinite(bar_close):
            return True, float(bar_close), f"trend_volume_{float(rule.volume_mult):.2f}x"
        _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
        return False, float("nan"), None

    if family == "signal_open":
        if not np.isfinite(signal_open):
            _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
            return False, float("nan"), None
        hit = bool(np.isfinite(bar_close) and ((bar_close >= signal_open) if is_long else (bar_close <= signal_open)))
        _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
        return (hit, float(bar_close), "signal_open_cross") if hit and np.isfinite(bar_close) else (False, float("nan"), None)

    if family == "signal_close":
        if not np.isfinite(signal_close):
            _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
            return False, float("nan"), None
        trigger = float(signal_close + (buffer_amt if np.isfinite(buffer_amt) else 0.0)) if is_long else float(signal_close - (buffer_amt if np.isfinite(buffer_amt) else 0.0))
        hit = bool(np.isfinite(bar_close) and ((bar_close >= trigger) if is_long else (bar_close <= trigger)))
        _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
        return (hit, float(bar_close), "signal_close_cross") if hit and np.isfinite(bar_close) else (False, float("nan"), None)

    if family == "signal_body":
        signal_body_high = max(signal_open, signal_close) if np.isfinite(signal_open) and np.isfinite(signal_close) else float("nan")
        signal_body_low = min(signal_open, signal_close) if np.isfinite(signal_open) and np.isfinite(signal_close) else float("nan")
        trigger = signal_body_high if is_long else signal_body_low
        if not np.isfinite(trigger):
            _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
            return False, float("nan"), None
        hit = bool(np.isfinite(bar_close) and ((bar_close >= trigger) if is_long else (bar_close <= trigger)))
        _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
        return (hit, float(bar_close), "signal_body_break") if hit and np.isfinite(bar_close) else (False, float("nan"), None)

    if family == "breakout":
        base = signal_high if is_long else signal_low
        if not np.isfinite(base):
            _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
            return False, float("nan"), None
        trigger = float(base + (buffer_amt if (is_long and np.isfinite(buffer_amt)) else 0.0)) if is_long else float(base - (buffer_amt if np.isfinite(buffer_amt) else 0.0))
        hit = bool(np.isfinite(bar_high if is_long else bar_low) and ((bar_high >= trigger) if is_long else (bar_low <= trigger)))
        fill = _trigger_price_from_open(side=side, trigger=trigger, bar_open=bar_open)
        _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
        return (hit and np.isfinite(fill), fill, "signal_breakout") if hit else (False, float("nan"), None)

    if family == "breakout_volume":
        base = signal_high if is_long else signal_low
        if not np.isfinite(base):
            _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
            return False, float("nan"), None
        trigger = float(base + (buffer_amt if (is_long and np.isfinite(buffer_amt)) else 0.0)) if is_long else float(base - (buffer_amt if np.isfinite(buffer_amt) else 0.0))
        probe = bar_high if is_long else bar_low
        hit = bool(np.isfinite(probe) and ((probe >= trigger) if is_long else (probe <= trigger)))
        vol_ok = bool(np.isfinite(vol_ratio) and vol_ratio >= float(rule.volume_mult))
        fill = _trigger_price_from_open(side=side, trigger=trigger, bar_open=bar_open)
        _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
        return (hit and vol_ok and np.isfinite(fill), fill, f"breakout_volume_{float(rule.volume_mult):.2f}x") if hit and vol_ok else (False, float("nan"), None)

    if family == "breakout_close":
        base = signal_high if is_long else signal_low
        if not np.isfinite(base):
            _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
            return False, float("nan"), None
        hit = bool(np.isfinite(bar_close) and ((bar_close >= base) if is_long else (bar_close <= base)))
        _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
        return (hit, float(bar_close), "signal_breakout_close") if hit and np.isfinite(bar_close) else (False, float("nan"), None)

    if family in {"bos_touch", "bos_close"}:
        history = list(state["history"])
        if len(history) < max(1, int(rule.lookback)):
            _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
            return False, float("nan"), None
        recent = history[-int(rule.lookback):]
        highs = [float(x["high"]) for x in recent if np.isfinite(x["high"])]
        lows = [float(x["low"]) for x in recent if np.isfinite(x["low"])]
        structure = max(highs) if is_long and highs else (min(lows) if (not is_long and lows) else float("nan"))
        if not np.isfinite(structure):
            _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
            return False, float("nan"), None
        if family == "bos_touch":
            probe = bar_high if is_long else bar_low
            hit = bool(np.isfinite(probe) and ((probe >= structure) if is_long else (probe <= structure)))
            fill = _trigger_price_from_open(side=side, trigger=structure, bar_open=bar_open)
            _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
            return (hit and np.isfinite(fill), fill, f"bos_{int(rule.lookback)}_touch") if hit else (False, float("nan"), None)
        hit = bool(np.isfinite(bar_close) and ((bar_close >= structure) if is_long else (bar_close <= structure)))
        _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
        return (hit, float(bar_close), f"bos_{int(rule.lookback)}_close") if hit and np.isfinite(bar_close) else (False, float("nan"), None)

    if family == "sr_bounce":
        tol = float(buffer_amt) if np.isfinite(buffer_amt) else 0.0
        if is_long:
            hit = bool(np.isfinite(prev_low) and np.isfinite(bar_low) and np.isfinite(bar_close) and np.isfinite(bar_open) and bar_low <= prev_low + tol and bar_close > bar_open and bar_close >= prev_low)
        else:
            hit = bool(np.isfinite(prev_high) and np.isfinite(bar_high) and np.isfinite(bar_close) and np.isfinite(bar_open) and bar_high >= prev_high - tol and bar_close < bar_open and bar_close <= prev_high)
        _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
        return (hit, float(bar_close), f"sr_bounce_{int(rule.lookback)}") if hit and np.isfinite(bar_close) else (False, float("nan"), None)

    if family == "vwap_trend":
        directional = bool(np.isfinite(bar_open) and np.isfinite(bar_close) and ((bar_close > bar_open) if is_long else (bar_close < bar_open)))
        vwap_ok = bool(np.isfinite(vwap) and np.isfinite(bar_close) and ((bar_close >= vwap) if is_long else (bar_close <= vwap)))
        state["confirm_count"] = int(state["confirm_count"]) + 1 if directional else 0
        if int(state["confirm_count"]) >= max(1, int(rule.confirm_bars)) and vwap_ok and np.isfinite(bar_close):
            return True, float(bar_close), "vwap_trend"
        _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
        return False, float("nan"), None

    if family == "ema_pullback":
        if not np.isfinite(ema_8):
            _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
            return False, float("nan"), None
        if is_long:
            hit = bool(np.isfinite(bar_low) and np.isfinite(bar_close) and np.isfinite(bar_open) and bar_low <= ema_8 and bar_close > bar_open and bar_close >= ema_8)
        else:
            hit = bool(np.isfinite(bar_high) and np.isfinite(bar_close) and np.isfinite(bar_open) and bar_high >= ema_8 and bar_close < bar_open and bar_close <= ema_8)
        _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
        return (hit, float(bar_close), f"ema_pullback_{int(rule.ema_length)}") if hit and np.isfinite(bar_close) else (False, float("nan"), None)

    if family == "adverse":
        if not (np.isfinite(signal_close) and np.isfinite(buffer_amt)):
            _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
            return False, float("nan"), None
        trigger = float(signal_close - buffer_amt) if is_long else float(signal_close + buffer_amt)
        probe = bar_low if is_long else bar_high
        hit = bool(np.isfinite(probe) and ((probe <= trigger) if is_long else (probe >= trigger)))
        fill = _trigger_price_from_open(side=side, trigger=trigger, bar_open=bar_open)
        _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
        return (hit and np.isfinite(fill), fill, "adverse_retest") if hit else (False, float("nan"), None)

    if family == "breakout_or_adverse":
        if not (np.isfinite(signal_high) and np.isfinite(signal_low) and np.isfinite(signal_close) and np.isfinite(buffer_amt)):
            _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
            return False, float("nan"), None
        breakout_trigger = float(signal_high) if is_long else float(signal_low)
        adverse_trigger = float(signal_close - buffer_amt) if is_long else float(signal_close + buffer_amt)
        breakout_probe = bar_high if is_long else bar_low
        adverse_probe = bar_low if is_long else bar_high
        breakout_hit = bool(np.isfinite(breakout_probe) and ((breakout_probe >= breakout_trigger) if is_long else (breakout_probe <= breakout_trigger)))
        adverse_hit = bool(np.isfinite(adverse_probe) and ((adverse_probe <= adverse_trigger) if is_long else (adverse_probe >= adverse_trigger)))
        chosen_trigger = float("nan")
        reason = None
        if breakout_hit and adverse_hit:
            breakout_dist = abs(float(breakout_trigger) - float(bar_open)) if np.isfinite(bar_open) else 0.0
            adverse_dist = abs(float(adverse_trigger) - float(bar_open)) if np.isfinite(bar_open) else 0.0
            if breakout_dist <= adverse_dist:
                chosen_trigger = breakout_trigger
                reason = "breakout"
            else:
                chosen_trigger = adverse_trigger
                reason = "adverse"
        elif breakout_hit:
            chosen_trigger = breakout_trigger
            reason = "breakout"
        elif adverse_hit:
            chosen_trigger = adverse_trigger
            reason = "adverse"
        fill = _trigger_price_from_open(side=side, trigger=chosen_trigger, bar_open=bar_open)
        _append_history(state, open_=bar_open, high=bar_high, low=bar_low, close=bar_close)
        return (np.isfinite(chosen_trigger) and np.isfinite(fill), fill, f"breakout_or_{reason}") if np.isfinite(chosen_trigger) else (False, float("nan"), None)

    raise ValueError(f"Unknown rule family: {family}")


def _simulate_rule(
    *,
    rule: EntryRule,
    meta_df: pd.DataFrame,
    one_min_arrays: dict[str, object],
    predictor_agent: LiveMetaXGBAgent,
    symbol: str,
    min_hold_bars: int,
    soft_exit_confirm_bars: int,
    urgent_exit_prob: float,
    urgent_exit_delta: float,
    opposite_dominance_delta: float,
    entry_long_probs: np.ndarray,
    entry_short_probs: np.ndarray,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    max_history = max((int(x.lookback) for x in _candidate_rules()), default=1) + 2
    long_state = {"confirm_count": 0, "history": deque(maxlen=max_history)}
    short_state = {"confirm_count": 0, "history": deque(maxlen=max_history)}

    long_trade = _reset_trade_state()
    short_trade = _reset_trade_state()

    long_active = False
    short_active = False
    long_intent_active = False
    short_intent_active = False
    pending_long_exit = False
    pending_short_exit = False
    long_soft_confirm = 0
    short_soft_confirm = 0
    one_min_pos = 0
    one_min_len = len(one_min_arrays["timestamp"])

    long_signal_row: pd.Series | None = None
    short_signal_row: pd.Series | None = None
    long_signal_open = float("nan")
    long_signal_high = float("nan")
    long_signal_low = float("nan")
    long_signal_close = float("nan")
    long_signal_atr = float("nan")
    short_signal_open = float("nan")
    short_signal_high = float("nan")
    short_signal_low = float("nan")
    short_signal_close = float("nan")
    short_signal_atr = float("nan")

    events: list[dict[str, object]] = []
    meta_index = meta_df.index.to_list()

    for idx, (_, row) in enumerate(meta_df.iterrows()):
        ts = pd.Timestamp(row.name)
        next_ts = meta_index[idx + 1] if idx + 1 < len(meta_index) else (ts + pd.Timedelta(minutes=10))
        decision_ts = ts + pd.Timedelta(minutes=10)
        next_decision_ts = next_ts + pd.Timedelta(minutes=10)

        p_enter_long = float(entry_long_probs[idx]) if idx < entry_long_probs.size else float("nan")
        p_enter_short = float(entry_short_probs[idx]) if idx < entry_short_probs.size else float("nan")
        work_row = row.copy()
        work_row["p_enter_long_oof"] = p_enter_long
        work_row["p_enter_short_oof"] = p_enter_short
        p_exit_long = _score_exit_for_state(agent=predictor_agent, row=work_row, state=long_trade, side="long") if long_active else float("nan")
        p_exit_short = _score_exit_for_state(agent=predictor_agent, row=work_row, state=short_trade, side="short") if short_active else float("nan")

        long_valid_signal, short_valid_signal = _validity_flags(
            p_enter_long=p_enter_long,
            p_enter_short=p_enter_short,
            thr_enter_long=float(thresholds["enter_long"]),
            thr_enter_short=float(thresholds["enter_short"]),
            opposite_dominance_delta=float(opposite_dominance_delta),
        )

        if not long_active:
            if long_valid_signal:
                if not long_intent_active:
                    _reset_intent_state(long_state)
                long_intent_active = True
                long_signal_row = row.copy()
                long_signal_open = float(row.get("open", np.nan))
                long_signal_high = float(row.get("high", np.nan))
                long_signal_low = float(row.get("low", np.nan))
                long_signal_close = float(row.get("close", np.nan))
                long_signal_atr = float(row.get("atr", np.nan))
            else:
                long_intent_active = False
                long_signal_row = None
                long_signal_open = float("nan")
                long_signal_high = float("nan")
                long_signal_low = float("nan")
                long_signal_close = float("nan")
                long_signal_atr = float("nan")
                _reset_intent_state(long_state)
        if not short_active:
            if short_valid_signal:
                if not short_intent_active:
                    _reset_intent_state(short_state)
                short_intent_active = True
                short_signal_row = row.copy()
                short_signal_open = float(row.get("open", np.nan))
                short_signal_high = float(row.get("high", np.nan))
                short_signal_low = float(row.get("low", np.nan))
                short_signal_close = float(row.get("close", np.nan))
                short_signal_atr = float(row.get("atr", np.nan))
            else:
                short_intent_active = False
                short_signal_row = None
                short_signal_open = float("nan")
                short_signal_high = float("nan")
                short_signal_low = float("nan")
                short_signal_close = float("nan")
                short_signal_atr = float("nan")
                _reset_intent_state(short_state)

        long_hold_ready = bool(long_active and int(long_trade.bars_since_entry) >= int(min_hold_bars))
        short_hold_ready = bool(short_active and int(short_trade.bars_since_entry) >= int(min_hold_bars))

        long_soft_exit_condition = bool(
            long_active and long_hold_ready and np.isfinite(p_enter_long) and p_enter_long < float(thresholds["enter_long"])
        )
        short_soft_exit_condition = bool(
            short_active and short_hold_ready and np.isfinite(p_enter_short) and p_enter_short < float(thresholds["enter_short"])
        )
        long_soft_confirm = long_soft_confirm + 1 if long_soft_exit_condition else 0
        short_soft_confirm = short_soft_confirm + 1 if short_soft_exit_condition else 0

        long_urgent_exit = bool(
            long_active
            and (
                (np.isfinite(p_exit_long) and p_exit_long >= float(urgent_exit_prob))
                or (
                    np.isfinite(p_exit_long)
                    and np.isfinite(p_enter_long)
                    and (p_exit_long - p_enter_long) >= float(urgent_exit_delta)
                )
            )
        )
        short_urgent_exit = bool(
            short_active
            and (
                (np.isfinite(p_exit_short) and p_exit_short >= float(urgent_exit_prob))
                or (
                    np.isfinite(p_exit_short)
                    and np.isfinite(p_enter_short)
                    and (p_exit_short - p_enter_short) >= float(urgent_exit_delta)
                )
            )
        )

        do_exit_long = bool(long_urgent_exit or long_soft_confirm >= int(soft_exit_confirm_bars))
        do_exit_short = bool(short_urgent_exit or short_soft_confirm >= int(soft_exit_confirm_bars))

        if long_active:
            long_trade = _advance_trade_state(
                state=long_trade,
                action=0 if do_exit_long else 1,
                row=work_row,
                agent=predictor_agent,
            )
            if do_exit_long:
                pending_long_exit = True
        if short_active:
            short_trade = _advance_trade_state(
                state=short_trade,
                action=0 if do_exit_short else -1,
                row=work_row,
                agent=predictor_agent,
            )
            if do_exit_short:
                pending_short_exit = True

        entered_long_this_interval = False
        entered_short_this_interval = False
        while one_min_pos < one_min_len and pd.Timestamp(one_min_arrays["timestamp"][one_min_pos]) < decision_ts:
            one_min_pos += 1
        while one_min_pos < one_min_len and pd.Timestamp(one_min_arrays["timestamp"][one_min_pos]) < next_decision_ts:
            bar_ts = pd.Timestamp(one_min_arrays["timestamp"][one_min_pos])
            bar_open = float(one_min_arrays["open"][one_min_pos])
            bar_high = float(one_min_arrays["high"][one_min_pos])
            bar_low = float(one_min_arrays["low"][one_min_pos])
            bar_close = float(one_min_arrays["close"][one_min_pos])

            if pending_long_exit and long_active and np.isfinite(bar_open):
                long_active = False
                pending_long_exit = False
                long_intent_active = False
                long_soft_confirm = 0
                long_trade = _reset_trade_state()
                long_signal_row = None
                _reset_intent_state(long_state)
                events.append({"regime": rule.name, "timestamp": bar_ts, "symbol": symbol, "event": "exit_long", "price": bar_open, "reason": "hybrid_exit"})
            if pending_short_exit and short_active and np.isfinite(bar_open):
                short_active = False
                pending_short_exit = False
                short_intent_active = False
                short_soft_confirm = 0
                short_trade = _reset_trade_state()
                short_signal_row = None
                _reset_intent_state(short_state)
                events.append({"regime": rule.name, "timestamp": bar_ts, "symbol": symbol, "event": "exit_short", "price": bar_open, "reason": "hybrid_exit"})

            if long_intent_active and (not long_active) and (not entered_long_this_interval) and long_signal_row is not None:
                hit, fill_price, reason = _evaluate_entry(
                    rule=rule,
                    side="long",
                    one_min_idx=one_min_pos,
                    one_min_arrays=one_min_arrays,
                    bar_open=bar_open,
                    bar_high=bar_high,
                    bar_low=bar_low,
                    bar_close=bar_close,
                    signal_open=long_signal_open,
                    signal_high=long_signal_high,
                    signal_low=long_signal_low,
                    signal_close=long_signal_close,
                    signal_atr=long_signal_atr,
                    state=long_state,
                )
                if hit:
                    long_active = True
                    long_intent_active = False
                    entered_long_this_interval = True
                    long_trade = _set_trade_entry(position=1, row=long_signal_row, entry_price=float(fill_price))
                    _reset_intent_state(long_state)
                    events.append({"regime": rule.name, "timestamp": bar_ts, "symbol": symbol, "event": "enter_long", "price": float(fill_price), "reason": reason or rule.name})

            if short_intent_active and (not short_active) and (not entered_short_this_interval) and short_signal_row is not None:
                hit, fill_price, reason = _evaluate_entry(
                    rule=rule,
                    side="short",
                    one_min_idx=one_min_pos,
                    one_min_arrays=one_min_arrays,
                    bar_open=bar_open,
                    bar_high=bar_high,
                    bar_low=bar_low,
                    bar_close=bar_close,
                    signal_open=short_signal_open,
                    signal_high=short_signal_high,
                    signal_low=short_signal_low,
                    signal_close=short_signal_close,
                    signal_atr=short_signal_atr,
                    state=short_state,
                )
                if hit:
                    short_active = True
                    short_intent_active = False
                    entered_short_this_interval = True
                    short_trade = _set_trade_entry(position=-1, row=short_signal_row, entry_price=float(fill_price))
                    _reset_intent_state(short_state)
                    events.append({"regime": rule.name, "timestamp": bar_ts, "symbol": symbol, "event": "enter_short", "price": float(fill_price), "reason": reason or rule.name})
            one_min_pos += 1

    return pd.DataFrame(events)


def _max_drawdown(equity: pd.Series) -> float:
    series = pd.to_numeric(equity, errors="coerce").dropna()
    if series.empty:
        return float("nan")
    running_peak = series.cummax()
    dd = series / running_peak - 1.0
    return float(dd.min())


def _save_equity_plot(
    *,
    curves: dict[str, pd.DataFrame],
    ordered_regimes: list[str],
    save_path: Path,
    symbol: str,
) -> None:
    fig, ax = plt.subplots(figsize=(17, 8))
    buy_hold_df = next(iter(curves.values()))
    buy_hold_x = pd.to_datetime(buy_hold_df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    ax.plot(buy_hold_x, buy_hold_df["buy_hold"], color="#444444", linewidth=1.5, label="buy_hold")

    palette = [
        "#1565C0",
        "#2E7D32",
        "#C62828",
        "#6A1B9A",
        "#EF6C00",
        "#00838F",
        "#AD1457",
    ]
    for idx, regime in enumerate(ordered_regimes):
        curve = curves.get(regime)
        if curve is None or curve.empty:
            continue
        x = pd.to_datetime(curve["timestamp"], utc=True).dt.tz_convert("America/New_York")
        ax.plot(
            x,
            curve["net_1x_style"],
            color=palette[idx % len(palette)],
            linewidth=2.0 if idx < 2 else 1.6,
            label=regime,
        )

    ax.set_title(f"{symbol} | 1m entry execution sweep | net 1x equity")
    ax.set_ylabel("Equity")
    ax.set_xlabel("Session Time (America/New_York)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d", tz=buy_hold_x.dt.tz))
    fig.autofmt_xdate()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    rules = _candidate_rules()[: max(1, int(args.max_rules))]
    meta_full = _load_meta_matrix(Path(args.meta_matrix), start=None, end=None, tz=args.tz)
    one_full = _load_one_min(Path(args.one_min_data), symbol=args.symbol, start=None, end=None)
    print(f"[entry-sweep] loaded raw inputs meta_rows={len(meta_full):,} one_min_rows={len(one_full):,}", flush=True)
    start_ts, end_ts = _auto_bounds(
        meta_df=meta_full,
        one_min=one_full,
        start_arg=args.start,
        end_arg=args.end,
        recent_days=args.recent_days,
    )

    meta_df = meta_full[(meta_full.index >= start_ts) & (meta_full.index <= end_ts)].copy()
    one_min = one_full[(one_full["timestamp"] >= start_ts) & (one_full["timestamp"] <= end_ts)].copy().reset_index(drop=True)
    if meta_df.empty:
        raise ValueError("Meta matrix is empty after applying the requested date window.")
    if one_min.empty:
        raise ValueError("1m data is empty after applying the requested date window.")
    print(f"[entry-sweep] windowed inputs meta_rows={len(meta_df):,} one_min_rows={len(one_min):,}", flush=True)
    one_min_arrays = _build_one_min_feature_arrays(one_min, tz=args.tz)

    base_agent = LiveMetaXGBAgent(
        model_root=Path(args.model_root),
        precomputed_base_frame=meta_df,
        entry_threshold_override=args.entry_threshold,
        exit_threshold_override=args.exit_threshold,
    )
    print("[entry-sweep] base agent loaded", flush=True)
    entry_long_probs = base_agent._entry_long.predict_frame(meta_df)
    entry_short_probs = base_agent._entry_short.predict_frame(meta_df)
    print("[entry-sweep] entry probability frames scored", flush=True)
    thresholds = base_agent.last_thresholds() or {
        "enter_long": np.nan,
        "enter_short": np.nan,
        "exit_long": np.nan,
        "exit_short": np.nan,
    }

    summary_rows: list[dict[str, object]] = []
    all_events: list[pd.DataFrame] = []
    curves: dict[str, pd.DataFrame] = {}
    sweep_start = time.perf_counter()
    for idx, rule in enumerate(rules, start=1):
        rule_start = time.perf_counter()
        events = _simulate_rule(
            rule=rule,
            meta_df=meta_df,
            one_min_arrays=one_min_arrays,
            predictor_agent=base_agent,
            symbol=args.symbol,
            min_hold_bars=max(1, int(args.min_hold_bars)),
            soft_exit_confirm_bars=max(1, int(args.soft_exit_confirm_bars)),
            urgent_exit_prob=float(args.urgent_exit_prob),
            urgent_exit_delta=float(args.urgent_exit_delta),
            opposite_dominance_delta=float(args.opposite_dominance_delta),
            entry_long_probs=np.asarray(entry_long_probs),
            entry_short_probs=np.asarray(entry_short_probs),
            thresholds=thresholds,
        )
        events = events.sort_values("timestamp").reset_index(drop=True) if not events.empty else pd.DataFrame(columns=["regime", "timestamp", "symbol", "event", "price", "reason"])
        curve = _equity_curve_from_events(events[["timestamp", "symbol", "event", "price"]].copy(), one_min)
        curves[rule.name] = curve
        metrics = _event_metrics(events[["timestamp", "symbol", "event", "price"]].copy())
        metrics.update(
            {
                "regime": rule.name,
                "family": rule.family,
                "confirm_bars": int(rule.confirm_bars),
                "atr_mult": float(rule.atr_mult),
                "lookback": int(rule.lookback),
                "net_1x_end": float(curve["net_1x_style"].iloc[-1]) if not curve.empty else np.nan,
                "buy_hold_end": float(curve["buy_hold"].iloc[-1]) if not curve.empty else np.nan,
                "max_drawdown_net_1x": _max_drawdown(curve["net_1x_style"]) if not curve.empty else np.nan,
            }
        )
        summary_rows.append(metrics)
        all_events.append(events)
        print(
            f"[entry-sweep] completed {idx}/{len(rules)} regime={rule.name} "
            f"net_1x_end={metrics['net_1x_end']:.6f} elapsed_sec={time.perf_counter() - rule_start:.2f}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise ValueError("No summary rows were produced.")

    current_name = "current_policy_trend_1bar"
    baseline_name = "baseline_next_open"
    current_net = float(summary.loc[summary["regime"].eq(current_name), "net_1x_end"].iloc[0]) if summary["regime"].eq(current_name).any() else np.nan
    current_gross = float(summary.loc[summary["regime"].eq(current_name), "combined_full_gross_end"].iloc[0]) if summary["regime"].eq(current_name).any() else np.nan
    baseline_net = float(summary.loc[summary["regime"].eq(baseline_name), "net_1x_end"].iloc[0]) if summary["regime"].eq(baseline_name).any() else np.nan
    summary["vs_current_net_1x"] = summary["net_1x_end"] - current_net if np.isfinite(current_net) else np.nan
    summary["vs_current_full_gross"] = summary["combined_full_gross_end"] - current_gross if np.isfinite(current_gross) else np.nan
    summary["vs_next_open_net_1x"] = summary["net_1x_end"] - baseline_net if np.isfinite(baseline_net) else np.nan
    summary = summary.sort_values(
        ["net_1x_end", "combined_full_gross_end", "short_win_rate", "long_win_rate"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    events_out = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    events_path = Path(args.events_out)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_out.to_csv(events_path, index=False)

    plot_regimes: list[str] = []
    for name in [current_name, baseline_name]:
        if name in curves and name not in plot_regimes:
            plot_regimes.append(name)
    for name in summary["regime"].head(5).tolist():
        if name in curves and name not in plot_regimes:
            plot_regimes.append(name)
    _save_equity_plot(curves=curves, ordered_regimes=plot_regimes, save_path=Path(args.equity_out), symbol=args.symbol)

    best = summary.iloc[0]
    print(
        f"[entry-sweep] window={start_ts.isoformat()}..{end_ts.isoformat()} "
        f"meta_rows={len(meta_df):,} one_min_rows={len(one_min):,} rules={len(rules)} total_sec={time.perf_counter() - sweep_start:.2f}"
    )
    print(
        f"[entry-sweep] best={best['regime']} net_1x_end={float(best['net_1x_end']):.6f} "
        f"full_gross_end={float(best['combined_full_gross_end']):.6f} "
        f"vs_current_net_1x={float(best['vs_current_net_1x']):+.6f}"
    )
    print(f"[entry-sweep] wrote summary={summary_path} events={events_path} equity_plot={Path(args.equity_out)}")


if __name__ == "__main__":
    main()
