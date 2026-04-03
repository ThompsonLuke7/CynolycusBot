from __future__ import annotations

import argparse
import math
import queue as queue_mod
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from alpaca.data.enums import DataFeed

from ..market_data.bar_aggregator import OhlcvAggregator
from ..market_data.bar_buffer import BarRingBuffer
from ..market_data.fetch_intraday import fetch_intraday
from ..inference.live_inference import (
    LiveIndependentMetaXGBAgent,
    LiveInferenceEngine,
    LiveMetaXGBAgent,
    LivePPOAgent,
    build_15m,
)
from ..market_data.live_stream import AlpacaBarStreamer
from Policy.execution_latch import DirectionExecutionLatch
from Policy.order_policy import OptionOrderPolicy, OptionOrderPolicyConfig


class LiveBarProcessor:
    def __init__(
        self,
        *,
        interval_minutes: int = 15,
        buffer_size: int | None = 5000,
        agg_label: str = "left",
        on_1m: Optional[Callable[[str, dict, BarRingBuffer], None]] = None,
        on_15m_close: Optional[Callable[[str, dict, BarRingBuffer], None]] = None,
        regular_hours_only: bool = False,
        tz_name: str = "America/New_York",
        session_open: str = "09:30",
        session_close: str = "16:00",
    ) -> None:
        self._interval_minutes = interval_minutes
        self._buffer_size = None if buffer_size is None or int(buffer_size) <= 0 else int(buffer_size)
        self._agg_label = agg_label
        self._on_1m = on_1m
        self._on_15m_close = on_15m_close
        self._regular_hours_only = bool(regular_hours_only)
        self._tz_name = str(tz_name)
        self._session_open = str(session_open)
        self._session_close = str(session_close)
        self._buffers: dict[str, BarRingBuffer] = {}
        self._aggregators: dict[str, OhlcvAggregator] = {}

    def _get_buffer(self, symbol: str) -> BarRingBuffer:
        if symbol not in self._buffers:
            self._buffers[symbol] = BarRingBuffer(maxlen=self._buffer_size)
        return self._buffers[symbol]

    def _get_aggregator(self, symbol: str) -> OhlcvAggregator:
        if symbol not in self._aggregators:
            self._aggregators[symbol] = OhlcvAggregator(
                interval_minutes=self._interval_minutes,
                label=self._agg_label,
            )
        return self._aggregators[symbol]

    def prefill(self, symbol: str, bars: list[dict]) -> None:
        if not bars:
            return
        buffer = self._get_buffer(symbol)
        if self._regular_hours_only:
            bars = [
                bar
                for bar in bars
                if _bar_in_regular_session(
                    bar,
                    tz_name=self._tz_name,
                    session_open=self._session_open,
                    session_close=self._session_close,
                )
            ]
            if not bars:
                return
        buffer.extend(bars)

    def handle_bar(self, bar: dict) -> None:
        symbol = str(bar.get("symbol", ""))
        if self._regular_hours_only and not _bar_in_regular_session(
            bar,
            tz_name=self._tz_name,
            session_open=self._session_open,
            session_close=self._session_close,
        ):
            return
        buffer = self._get_buffer(symbol)
        agg = self._get_aggregator(symbol)

        buffer.append(bar)
        if self._on_1m is not None:
            self._on_1m(symbol, bar, buffer)

        closed, _current = agg.update(bar)
        if closed and self._on_15m_close is not None:
            self._on_15m_close(symbol, closed, buffer)
        if (
            self._regular_hours_only
            and self._on_15m_close is not None
            and _bar_is_session_last_minute(
                bar,
                tz_name=self._tz_name,
                session_close=self._session_close,
            )
        ):
            final_closed = agg.flush_current()
            if final_closed is not None:
                self._on_15m_close(symbol, final_closed, buffer)


def _parse_feed(feed: str) -> DataFeed:
    feed_key = feed.strip().upper()
    if feed_key == "SIP":
        return DataFeed.SIP
    return DataFeed.IEX


def _format_ts_local(ts: object, *, tz: str = "America/New_York") -> str:
    if ts is None:
        return "None"
    try:
        t = pd.to_datetime(ts, utc=True, errors="coerce")
        if pd.isna(t):
            return str(ts)
        if tz:
            t = t.tz_convert(tz)
        return t.isoformat()
    except Exception:
        return str(ts)


def _hhmm_to_minutes(hhmm: str, *, default: int) -> int:
    try:
        parts = str(hhmm).strip().split(":")
        if len(parts) != 2:
            return int(default)
        return max(0, min(24 * 60, int(parts[0]) * 60 + int(parts[1])))
    except Exception:
        return int(default)


def _bar_in_regular_session(
    bar: dict,
    *,
    tz_name: str = "America/New_York",
    session_open: str = "09:30",
    session_close: str = "16:00",
) -> bool:
    ts = pd.to_datetime(bar.get("timestamp"), utc=True, errors="coerce")
    if pd.isna(ts):
        return False
    local_ts = ts.tz_convert(tz_name)
    minutes = int(local_ts.hour) * 60 + int(local_ts.minute)
    open_min = _hhmm_to_minutes(session_open, default=570)
    close_min = _hhmm_to_minutes(session_close, default=960)
    return open_min <= minutes < close_min


def _bar_is_session_last_minute(
    bar: dict,
    *,
    tz_name: str = "America/New_York",
    session_close: str = "16:00",
) -> bool:
    ts = pd.to_datetime(bar.get("timestamp"), utc=True, errors="coerce")
    if pd.isna(ts):
        return False
    local_ts = ts.tz_convert(tz_name)
    minutes = int(local_ts.hour) * 60 + int(local_ts.minute)
    close_min = _hhmm_to_minutes(session_close, default=960)
    return minutes == (close_min - 1)


def _fmt_prob(value: object) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return f"{v:.3f}"


def _print_meta_prob_log(*, prefix: str, probs: dict[str, float | None] | None, thresholds: dict[str, float] | None) -> None:
    if not probs and not thresholds:
        return
    probs = probs or {}
    thresholds = thresholds or {}
    print(
        f"{prefix} "
        f"p_enter_long={_fmt_prob(probs.get('p_enter_long'))} thr_enter_long={_fmt_prob(thresholds.get('enter_long'))} "
        f"p_enter_short={_fmt_prob(probs.get('p_enter_short'))} thr_enter_short={_fmt_prob(thresholds.get('enter_short'))} "
        f"p_exit_long={_fmt_prob(probs.get('p_exit_long'))} thr_exit_long={_fmt_prob(thresholds.get('exit_long'))} "
        f"p_exit_short={_fmt_prob(probs.get('p_exit_short'))} thr_exit_short={_fmt_prob(thresholds.get('exit_short'))}"
    )


def _action_to_position(action: float | int, *, deadband: float = 0.0) -> int:
    try:
        a = float(action)
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(a):
        return 0
    if abs(a - 2.0) < 1e-9:
        return -1
    if abs(a - 1.0) < 1e-9:
        return 1
    if abs(a) <= max(0.0, float(deadband)):
        return 0
    return 1 if a > 0.0 else -1


def _use_meta_direct_execution(inference: LiveInferenceEngine) -> bool:
    agent = getattr(inference, "_agent", None)
    return isinstance(agent, (LiveMetaXGBAgent, LiveIndependentMetaXGBAgent))


def _resolve_ga_feature_list_path(
    *,
    symbol: str,
    dataset_name: str,
    ga_feature_list: str | None,
    inference_enabled: bool,
    include_pivot_probs: bool,
    include_tb_probs: bool,
) -> str | None:
    if not inference_enabled:
        return ga_feature_list
    if ga_feature_list is not None:
        return str(ga_feature_list)
    if not (include_pivot_probs or include_tb_probs):
        return None
    try:
        from Data.load_data import get_ticker_processed_base_dir
        from Data.retrieve_data import normalize_ticker

        ticker = normalize_ticker(symbol)
        candidate = (
            get_ticker_processed_base_dir(ticker)
            / "datasets"
            / dataset_name
            / f"features_X_{dataset_name}_tree.txt"
        )
        if candidate.exists():
            return str(candidate)
    except Exception:
        return None
    return None


def _build_meta_agent(
    *,
    symbol: str,
    model_root: str,
    ga_model_root: str | None,
    ga_feature_list_path: str | None,
    include_pivot_probs: bool,
    include_tb_probs: bool,
    pivot_label_dir: str,
    tb_label_dir: str,
    tz: str | None,
    assume_tz: str,
    session_open: str,
    session_close: str,
    min_15m_bars: int,
    fill_missing_prob: float,
    resample_label: str,
    resample_closed: str,
    label_timeframe_rule: str,
    trail_activate_atr: float,
    trail_atr: float,
    trail_atr_after_tp: float,
    use_tp_to_tighten_trail: bool,
    entry_threshold_override: float | None,
    exit_threshold_override: float | None,
    ga_probs_frame: pd.DataFrame | None = None,
    ga_probs_mode: str = "xgb",
    precomputed_base_frame: pd.DataFrame | None = None,
    precomputed_append_lookback_days: int = 120,
    min_hold_bars: int = 2,
    exit_entry_delta: float = 0.15,
    soft_exit_confirm_bars: int = 2,
    urgent_exit_prob: float = 0.85,
    urgent_exit_delta: float = 0.30,
    profit_protect_enabled: bool = False,
    profit_protect_arm_atr: float = 2.0,
    profit_protect_giveback_atr_long: float = 0.75,
    profit_protect_giveback_atr_short: float = 1.0,
) -> LiveIndependentMetaXGBAgent:
    del symbol
    return LiveIndependentMetaXGBAgent(
        model_root=model_root,
        ga_model_root=ga_model_root if ga_feature_list_path else None,
        ga_feature_list_path=ga_feature_list_path,
        ga_probs_frame=ga_probs_frame,
        ga_probs_mode=ga_probs_mode,
        include_pivot_probs=include_pivot_probs,
        include_tb_probs=include_tb_probs,
        pivot_label_dir=pivot_label_dir,
        tb_label_dir=tb_label_dir,
        tz=tz or "America/New_York",
        assume_tz=assume_tz,
        session_open=session_open,
        session_close=session_close,
        min_15m_bars=min_15m_bars,
        fill_missing_prob=fill_missing_prob,
        resample_label=resample_label,
        resample_closed=resample_closed,
        label_timeframe_rule=label_timeframe_rule,
        trail_activate_atr=float(trail_activate_atr),
        trail_atr=float(trail_atr),
        trail_atr_after_tp=float(trail_atr_after_tp),
        use_tp_to_tighten_trail=bool(use_tp_to_tighten_trail),
        entry_threshold_override=entry_threshold_override,
        exit_threshold_override=exit_threshold_override,
        precomputed_base_frame=precomputed_base_frame,
        precomputed_append_lookback_days=int(precomputed_append_lookback_days),
        min_hold_bars=int(min_hold_bars),
        exit_entry_delta=float(exit_entry_delta),
        soft_exit_confirm_bars=int(soft_exit_confirm_bars),
        urgent_exit_prob=float(urgent_exit_prob),
        urgent_exit_delta=float(urgent_exit_delta),
        profit_protect_enabled=bool(profit_protect_enabled),
        profit_protect_arm_atr=float(profit_protect_arm_atr),
        profit_protect_giveback_atr_long=float(profit_protect_giveback_atr_long),
        profit_protect_giveback_atr_short=float(profit_protect_giveback_atr_short),
    )


def _build_ppo_agent(
    *,
    model_path: str,
    deterministic: bool,
    device: str,
    include_pivot_probs: bool,
    include_tb_probs: bool,
    tz: str | None,
    assume_tz: str,
    session_open: str,
    session_close: str,
    min_15m_bars: int,
    fill_missing_prob: float,
    ga_model_root: str | None,
    ga_feature_list_path: str | None,
    ga_pivot_label_dir: str,
    ga_tb_label_dir: str,
    ga_probs_frame: pd.DataFrame | None = None,
    ga_probs_mode: str = "xgb",
    require_probs: bool = False,
    resample_label: str = "left",
    resample_closed: str = "left",
    label_timeframe_rule: str = "10min",
) -> LivePPOAgent:
    return LivePPOAgent(
        model_path=model_path,
        deterministic=bool(deterministic),
        device=device,
        include_pivot_probs=include_pivot_probs,
        include_tb_probs=include_tb_probs,
        tz=tz or "America/New_York",
        assume_tz=assume_tz,
        session_open=session_open,
        session_close=session_close,
        min_15m_bars=min_15m_bars,
        fill_missing_prob=fill_missing_prob,
        ga_model_root=ga_model_root if ga_feature_list_path and ga_probs_mode != "frame" else None,
        ga_feature_list_path=ga_feature_list_path,
        ga_pivot_label_dir=ga_pivot_label_dir,
        ga_tb_label_dir=ga_tb_label_dir,
        ga_probs_frame=ga_probs_frame,
        ga_probs_mode=ga_probs_mode,
        require_probs=bool(require_probs),
        resample_label=resample_label,
        resample_closed=resample_closed,
        label_timeframe_rule=label_timeframe_rule,
    )


def _build_option_order_policy(
    *,
    symbol: str,
    env_file: str,
    tz_name: str | None,
    atr_multiplier: float,
    dte_cutoff_hhmm: str,
    qty: int,
    close_on_flat: bool,
    close_on_flip: bool,
    submit_orders: bool,
    opposite_confirm_bars: int = 2,
    opposite_min_abs_action: float = 0.10,
    opposite_min_prob_edge: float = 0.05,
    ema_alpha: float = 0.85,
    rebalance_deadband: float = 0.10,
    max_step_contracts: int = 2,
    price_mode: str = "ask",
    max_contracts_fallback: int = 1,
    max_contracts_cap: int = 0,
    meta_execute_on_interval_close: bool = False,
    meta_intrabar_execution_enabled: bool = True,
    meta_intrabar_breakout_entry_only: bool = False,
    meta_trailing_stop_enabled: bool = True,
    meta_trail_activate_atr: float = 0.75,
    meta_trail_atr: float = 0.8,
    meta_trail_atr_after_tp: float = 0.5,
    meta_use_tp_to_tighten_trail: bool = True,
    meta_soft_exit_confirm_bars: int = 2,
    meta_urgent_exit_prob: float = 0.85,
    meta_urgent_exit_delta: float = 0.30,
    meta_profit_protect_enabled: bool = False,
    meta_profit_protect_arm_atr: float = 2.0,
    meta_profit_protect_giveback_atr_long: float = 0.75,
    meta_profit_protect_giveback_atr_short: float = 1.0,
) -> OptionOrderPolicy:
    cfg = OptionOrderPolicyConfig(
        underlying=symbol,
        env_file=env_file,
        tz_name=tz_name or "America/New_York",
        atr_multiplier=float(atr_multiplier),
        dte_cutoff_hhmm=dte_cutoff_hhmm,
        qty=int(qty),
        close_on_flat=bool(close_on_flat),
        close_on_flip=bool(close_on_flip),
        opposite_confirm_bars=int(opposite_confirm_bars),
        opposite_min_abs_action=float(opposite_min_abs_action),
        opposite_min_prob_edge=float(opposite_min_prob_edge),
        submit_orders=bool(submit_orders),
        ema_alpha=float(ema_alpha),
        rebalance_deadband=float(rebalance_deadband),
        max_step_contracts=int(max_step_contracts),
        price_mode=str(price_mode),
        max_contracts_fallback=int(max_contracts_fallback),
        max_contracts_cap=int(max_contracts_cap),
        meta_trailing_stop_enabled=bool(meta_trailing_stop_enabled),
        meta_trail_activate_atr=float(meta_trail_activate_atr),
        meta_trail_atr=float(meta_trail_atr),
        meta_trail_atr_after_tp=float(meta_trail_atr_after_tp),
        meta_use_tp_to_tighten_trail=bool(meta_use_tp_to_tighten_trail),
        meta_execute_on_interval_close=bool(meta_execute_on_interval_close),
        meta_intrabar_execution_enabled=bool(meta_intrabar_execution_enabled),
        meta_intrabar_breakout_entry_only=bool(meta_intrabar_breakout_entry_only),
        meta_soft_exit_confirm_bars=int(meta_soft_exit_confirm_bars),
        meta_urgent_exit_prob=float(meta_urgent_exit_prob),
        meta_urgent_exit_delta=float(meta_urgent_exit_delta),
        meta_profit_protect_enabled=bool(meta_profit_protect_enabled),
        meta_profit_protect_arm_atr=float(meta_profit_protect_arm_atr),
        meta_profit_protect_giveback_atr_long=float(meta_profit_protect_giveback_atr_long),
        meta_profit_protect_giveback_atr_short=float(meta_profit_protect_giveback_atr_short),
    )
    return OptionOrderPolicy(cfg)


def _make_1m_handler(
    *,
    print_tz: str,
    print_1m: bool,
    order_policies: dict[str, OptionOrderPolicy] | None = None,
    trace_rows: list[dict] | None = None,
) -> Callable[[str, dict, BarRingBuffer], None]:
    def _handler(symbol: str, bar: dict, _buffer: BarRingBuffer) -> None:
        if order_policies is not None and symbol in order_policies:
            result = order_policies[symbol].on_1m_bar(bar=bar)
            event = str(result.get("event", "unknown"))
            if trace_rows is not None:
                trace_rows.append(
                    {
                        "trace_kind": "policy_1m",
                        "symbol": symbol,
                        "timestamp": bar.get("timestamp"),
                        "open": bar.get("open"),
                        "high": bar.get("high"),
                        "low": bar.get("low"),
                        "close": bar.get("close"),
                        "volume": bar.get("volume"),
                        "policy_event": event,
                        "policy_mode": result.get("mode"),
                        "policy_reason": result.get("reason"),
                        "policy_position": result.get("position"),
                        "policy_signed_contracts": result.get("signed_contracts"),
                        "policy_long_contracts": result.get("long_contracts"),
                        "policy_short_contracts": result.get("short_contracts"),
                        "policy_open_long_symbol": result.get("open_long_symbol"),
                        "policy_open_short_symbol": result.get("open_short_symbol"),
                        "target_long_contracts": result.get("target_long_contracts"),
                        "target_short_contracts": result.get("target_short_contracts"),
                    }
                )
            if event not in {"hold", "no_change"}:
                print(f"{symbol} order_policy 1m event={event} details={result}")
        if print_1m:
            ts = _format_ts_local(bar.get("timestamp"), tz=print_tz)
            print(
                f"{symbol} 1m: {ts} o={bar.get('open')} h={bar.get('high')} "
                f"l={bar.get('low')} c={bar.get('close')} v={bar.get('volume')}"
            )
    return _handler


def _make_15m_handler(
    *,
    inference: LiveInferenceEngine,
    interval_minutes: int,
    print_15m: bool,
    print_tz: str,
    execution_latches: dict[str, DirectionExecutionLatch],
    order_policies: dict[str, OptionOrderPolicy] | None = None,
) -> Callable[[str, dict, BarRingBuffer], None]:
    def _handler(symbol: str, bar15: dict, buffer: BarRingBuffer) -> None:
        if print_15m:
            ts = _format_ts_local(bar15.get("timestamp"), tz=print_tz)
            print(
                f"{symbol} {interval_minutes}m closed: {ts} o={bar15.get('open')} h={bar15.get('high')} "
                f"l={bar15.get('low')} c={bar15.get('close')} v={bar15.get('volume')}"
            )
        if order_policies is not None and symbol in order_policies:
            order_policies[symbol].on_interval_bar(closed_bar=bar15)
        action = inference.on_15m_close(df_1m=buffer.to_dataframe(), closed_bar=bar15)
        if action is not None:
            raw_action = float(action)
            raw_pos = _action_to_position(raw_action)
            if _use_meta_direct_execution(inference):
                exec_pos = int(raw_pos)
                gate_status = "meta_direct"
            else:
                gate = execution_latches[symbol].step(raw_pos)
                exec_pos = int(gate.executed_pos)
                gate_status = str(gate.status)
            probs = inference.last_probs() or {}
            thresholds = inference.last_thresholds() or {}
            _print_meta_prob_log(
                prefix=f"{symbol} meta:",
                probs=probs,
                thresholds=thresholds,
            )
            print(
                f"{symbol} inference raw={raw_action:+.4f} raw_pos={raw_pos:+d} "
                f"exec={exec_pos:+d} gate={gate_status}"
            )
            if order_policies is not None and symbol in order_policies:
                policy_bar = dict(bar15)
                policy_bar.update({k: v for k, v in probs.items() if v is not None})
                policy_bar.update(
                    {
                        "thr_enter_long": thresholds.get("enter_long"),
                        "thr_enter_short": thresholds.get("enter_short"),
                        "thr_exit_long": thresholds.get("exit_long"),
                        "thr_exit_short": thresholds.get("exit_short"),
                    }
                )
                result = order_policies[symbol].on_decision(
                    action=float(raw_action if _use_meta_direct_execution(inference) else exec_pos),
                    closed_bar=policy_bar,
                    update_bar_state=False,
                )
                event = str(result.get("event", "unknown"))
                if event not in {"hold", "no_change", "intent_update"}:
                    print(f"{symbol} order_policy event={event} details={result}")
    return _handler


def _load_prefill_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prefill file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Prefill file must be .csv or .parquet")

    if "timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
        else:
            raise ValueError("Prefill data must include a timestamp column or DatetimeIndex.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df


def _default_runtime_prefill_cache_path(prefill_path: Path) -> Path:
    suffix = prefill_path.suffix.lower()
    if suffix not in {".parquet", ".csv"}:
        suffix = ".parquet"
    return prefill_path.with_name(f"{prefill_path.stem}_runtime_rth_cache{suffix}")


def _default_runtime_vix_cache_path() -> Path:
    return Path("Data") / "raw" / "vix" / "vixy_10min_live_runtime.parquet"


def _prepare_runtime_prefill_frame(
    *,
    df: pd.DataFrame,
    precomputed_meta_frame: pd.DataFrame | None,
    append_lookback_days: int,
    tz_name: str = "America/New_York",
    session_open: str = "09:30",
    session_close: str = "16:00",
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.upper()

    ts_local = out["timestamp"].dt.tz_convert(tz_name)
    minutes = ts_local.dt.hour * 60 + ts_local.dt.minute
    open_min = _hhmm_to_minutes(session_open, default=570)
    close_min = _hhmm_to_minutes(session_close, default=960)
    out = out.loc[minutes.between(open_min, close_min)].copy()
    if out.empty:
        return out

    if isinstance(precomputed_meta_frame, pd.DataFrame) and not precomputed_meta_frame.empty:
        cached_max = precomputed_meta_frame.index.max()
        if isinstance(cached_max, pd.Timestamp) and not pd.isna(cached_max):
            if cached_max.tz is None:
                cached_max = cached_max.tz_localize(tz_name)
            keep_start_local = cached_max - pd.Timedelta(days=max(1, int(append_lookback_days)))
            keep_start_utc = keep_start_local.tz_convert("UTC")
            out = out.loc[out["timestamp"] >= keep_start_utc].copy()

    out = out.sort_values("timestamp")
    if "symbol" in out.columns:
        out = out.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    else:
        out = out.drop_duplicates(subset=["timestamp"], keep="last")
    return out.reset_index(drop=True)


def _persist_runtime_prefill_cache(
    *,
    df: pd.DataFrame,
    cache_path: Path,
) -> None:
    if df is None or df.empty:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.suffix.lower() == ".csv":
        df.to_csv(cache_path, index=False)
    else:
        df.to_parquet(cache_path, index=False)


def _load_runtime_vix_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing VIX cache file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("VIX cache file must be .csv or .parquet")
    return _normalize_runtime_vix_frame(df)


def _normalize_runtime_vix_frame(df: pd.DataFrame, *, symbol: str = "VIXY") -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        "Date": "timestamp",
        "date": "timestamp",
        "Datetime": "timestamp",
        "datetime": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    out = out.rename(columns=rename_map)
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"VIX cache missing required columns: {missing}")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).copy()
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out["symbol"] = symbol
    out = out[["timestamp", "open", "high", "low", "close", "volume", "symbol"]].copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    return out


def _prepare_runtime_vix_frame(
    *,
    df: pd.DataFrame,
    tz_name: str = "America/New_York",
    session_open: str = "09:30",
    session_close: str = "16:00",
    max_rows: int = 10000,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = _normalize_runtime_vix_frame(df)
    ts_local = out["timestamp"].dt.tz_convert(tz_name)
    minutes = ts_local.dt.hour * 60 + ts_local.dt.minute
    open_min = _hhmm_to_minutes(session_open, default=570)
    close_min = _hhmm_to_minutes(session_close, default=960)
    out = out.loc[minutes.between(open_min, close_min - 1)].copy()
    out = out.sort_values("timestamp")
    out = out.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    if max_rows and max_rows > 0 and len(out) > max_rows:
        out = out.tail(int(max_rows)).copy()
    return out.reset_index(drop=True)


def _persist_runtime_vix_cache(
    *,
    df: pd.DataFrame,
    cache_path: Path,
) -> None:
    if df is None or df.empty:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.suffix.lower() == ".csv":
        df.to_csv(cache_path, index=False)
    else:
        df.to_parquet(cache_path, index=False)


def _extend_vix_with_alpaca_gap(
    *,
    df: pd.DataFrame,
    feed: DataFeed,
    ticker: str = "VIXY",
    timeframe: str = "10Min",
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = _normalize_runtime_vix_frame(df, symbol=ticker)
    latest_ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dropna().max()
    if pd.isna(latest_ts):
        return out
    fetch_start = latest_ts + pd.Timedelta(minutes=10 if str(timeframe).lower() == "10min" else 1)
    now_utc = pd.Timestamp.now(tz="UTC")
    if fetch_start >= now_utc:
        return out
    fetched_df = fetch_intraday(
        ticker=ticker,
        start=fetch_start.isoformat(),
        timeframe=timeframe,
        limit=100000,
        feed=feed,
        save_path=None,
    )
    if fetched_df is None or fetched_df.empty:
        return out
    fetched_df = _normalize_runtime_vix_frame(fetched_df, symbol=ticker)
    fetched_df = fetched_df[fetched_df["timestamp"] > latest_ts].copy()
    if fetched_df.empty:
        return out
    combined = pd.concat([out, fetched_df], axis=0, ignore_index=True)
    combined = combined.sort_values("timestamp")
    combined = combined.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    return combined.reset_index(drop=True)


def _resolve_symbolized_path(template: str | Path, *, symbol: str) -> Path:
    text = str(template)
    return Path(
        text.format(
            symbol=symbol.upper(),
            symbol_lower=symbol.lower(),
        )
    )


def _load_precomputed_meta_frame(path: Path, *, tz: str = "America/New_York") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing meta base frame file: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Meta base frame must be .csv or .parquet")

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.loc[ts.notna()].copy()
        df.index = ts[ts.notna()]
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Meta base frame must include a timestamp column or DatetimeIndex.")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    if tz:
        df.index = df.index.tz_convert(tz)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _prefill_buffers(
    *,
    processor: LiveBarProcessor,
    df: pd.DataFrame,
    symbols: list[str],
    tail: int | None,
) -> None:
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Prefill data missing required columns: {missing}")

    df = df.copy()
    if "symbol" not in df.columns:
        if len(symbols) != 1:
            raise ValueError("Prefill data missing symbol column; use single --symbols or add symbol column.")
        df["symbol"] = symbols[0]
    df["symbol"] = df["symbol"].astype(str).str.upper()

    df = df[df["symbol"].isin(symbols)]
    if df.empty:
        print("[live] Prefill skipped: no matching symbols in prefill data.")
        return

    df = df.sort_values("timestamp")
    effective_tail = int(tail) if tail is not None else (
        int(processor._buffer_size) if processor._buffer_size is not None else None
    )
    for symbol in symbols:
        sym_df = df[df["symbol"] == symbol]
        if sym_df.empty:
            continue
        if effective_tail is not None and effective_tail > 0:
            sym_df = sym_df.tail(effective_tail)
        bars = sym_df[required + ["symbol"]].to_dict("records")
        processor.prefill(symbol, bars)
        cap_text = "unlimited" if effective_tail is None else f"{effective_tail:,}"
        print(
            f"[live] Prefilled {len(bars):,} bars for {symbol} "
            f"(cap={cap_text})."
        )


def _prefill_from_alpaca(
    *,
    processor: LiveBarProcessor,
    symbols: list[str],
    start: str,
    tail: int | None,
    split_dataset_name: str = "15min",
    split_x_filename: str = "X_15min_tree.parquet",
    prepend_split_test_warmup: bool = True,
    prepend_warmup_bars: int = 500,
    feed: DataFeed = DataFeed.IEX,
) -> None:
    feed_enum = feed if isinstance(feed, DataFeed) else _parse_feed(str(feed))
    effective_tail = int(tail) if tail is not None else (
        int(processor._buffer_size) if processor._buffer_size is not None else None
    )
    warmup_target = max(0, int(prepend_warmup_bars))
    if effective_tail is not None and effective_tail > 0:
        warmup_target = min(warmup_target, effective_tail)
    for symbol in symbols:
        fetched_df = fetch_intraday(
            ticker=symbol,
            start=start,
            timeframe="1Min",
            limit=100000,
            feed=feed_enum,
            save_path=None,
        )
        frames: list[pd.DataFrame] = []
        raw_warmup_count = 0
        used_warmup_count = 0
        if prepend_split_test_warmup:
            warmup_df = _load_test_split_warmup_1m(
                symbol=symbol,
                dataset_name=split_dataset_name,
                x_filename=split_x_filename,
            )
            if warmup_df is not None and not warmup_df.empty:
                raw_warmup_count = int(len(warmup_df))
                if warmup_target > 0:
                    warmup_df = warmup_df.tail(warmup_target)
                else:
                    warmup_df = warmup_df.iloc[0:0]
                used_warmup_count = int(len(warmup_df))
                if used_warmup_count > 0:
                    frames.append(warmup_df)

        raw_fetched_count = 0
        used_fetched_count = 0
        if fetched_df is not None and not fetched_df.empty:
            fetched_df = _normalize_prefill_1m_frame(fetched_df, symbol=symbol)
            raw_fetched_count = int(len(fetched_df))
            fetched_latest_df = fetched_df
            if effective_tail is not None and effective_tail > 0:
                fetch_tail_cap = max(effective_tail - used_warmup_count, 0)
                if fetch_tail_cap > 0:
                    fetched_df = fetched_df.tail(fetch_tail_cap)
                else:
                    fetched_df = fetched_df.iloc[0:0]
            used_fetched_count = int(len(fetched_df))
            if used_fetched_count > 0:
                frames.append(fetched_df)
            last_fetch_ts = pd.to_datetime(
                fetched_latest_df["timestamp"],
                utc=True,
                errors="coerce",
            ).max()
            if pd.notna(last_fetch_ts):
                now_ny = pd.Timestamp.now(tz="America/New_York")
                last_fetch_ny = last_fetch_ts.tz_convert("America/New_York")
                print(
                    f"[live] Prefill {symbol}: latest fetched 1m bar="
                    f"{last_fetch_ny.isoformat()} (feed={feed_enum.value})."
                )
                if last_fetch_ny.normalize() < now_ny.normalize():
                    print(
                        f"[live] Prefill {symbol}: no fetched bars on {now_ny.date()} yet; "
                        "market may be closed/holiday or selected feed has no newer bars."
                    )

        if not frames:
            print(f"[live] Prefill skipped: no historical bars for {symbol}.")
            continue

        combined = pd.concat(frames, axis=0, ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
        combined = combined.dropna(subset=["timestamp"])
        combined = combined.sort_values("timestamp")
        combined = combined.drop_duplicates(subset=["symbol", "timestamp"], keep="first")
        if effective_tail is not None and effective_tail > 0 and len(combined) > effective_tail:
            warmup_keep = min(used_warmup_count, len(combined), effective_tail)
            remaining_cap = max(effective_tail - warmup_keep, 0)
            keep_warmup = combined.head(warmup_keep) if warmup_keep > 0 else combined.iloc[0:0]
            keep_fetched = combined.tail(remaining_cap) if remaining_cap > 0 else combined.iloc[0:0]
            combined = pd.concat([keep_warmup, keep_fetched], axis=0, ignore_index=True)
            combined = combined.drop_duplicates(subset=["symbol", "timestamp"], keep="first")
            combined = combined.sort_values("timestamp")
        used_combined_count = int(len(combined))

        _prefill_buffers(
            processor=processor,
            df=combined,
            symbols=[symbol],
            tail=effective_tail,
        )
        print(
            f"[live] Prefill source breakdown for {symbol}: "
            f"split_test_warmup(raw={raw_warmup_count:,}, used={used_warmup_count:,}, target={warmup_target:,}), "
            f"alpaca_fetch(raw={raw_fetched_count:,}, used={used_fetched_count:,}), "
            f"combined_used={used_combined_count:,}, cap={effective_tail:,}"
        )


def _extend_prefill_with_alpaca_gap(
    *,
    df: pd.DataFrame,
    symbols: list[str],
    feed: DataFeed,
) -> pd.DataFrame:
    if df.empty:
        return df
    out_frames: list[pd.DataFrame] = [df.copy()]
    for symbol in symbols:
        sym_df = df[df["symbol"].astype(str).str.upper() == symbol].copy() if "symbol" in df.columns else df.copy()
        if sym_df.empty:
            continue
        latest_ts = pd.to_datetime(sym_df["timestamp"], utc=True, errors="coerce").dropna().max()
        if pd.isna(latest_ts):
            continue
        fetch_start = latest_ts + pd.Timedelta(minutes=1)
        now_utc = pd.Timestamp.now(tz="UTC")
        if fetch_start >= now_utc:
            print(f"[live] Prefill gap bridge {symbol}: local history already up to date ({latest_ts}).")
            continue
        fetched_df = fetch_intraday(
            ticker=symbol,
            start=fetch_start.isoformat(),
            timeframe="1Min",
            limit=100000,
            feed=feed,
            save_path=None,
        )
        if fetched_df is None or fetched_df.empty:
            print(f"[live] Prefill gap bridge {symbol}: no newer Alpaca 1m bars after {latest_ts}.")
            continue
        fetched_df = _normalize_prefill_1m_frame(fetched_df, symbol=symbol)
        fetched_df = fetched_df[fetched_df["timestamp"] > latest_ts].copy()
        if fetched_df.empty:
            print(f"[live] Prefill gap bridge {symbol}: fetched bars were all duplicates.")
            continue
        out_frames.append(fetched_df)
        print(
            f"[live] Prefill gap bridge {symbol}: appended {len(fetched_df):,} "
            f"1m bars from {fetched_df['timestamp'].min()} to {fetched_df['timestamp'].max()}."
        )
    combined = pd.concat(out_frames, axis=0, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["timestamp"]).sort_values("timestamp")
    if "symbol" in combined.columns:
        combined["symbol"] = combined["symbol"].astype(str).str.upper()
        combined = combined.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    else:
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    return combined.sort_values("timestamp").reset_index(drop=True)


def _normalize_prefill_1m_frame(df: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={out.index.name or "index": "timestamp"})
        else:
            raise ValueError("Prefill source frame must have 'timestamp' column or DatetimeIndex.")

    rename_map = {
        "Date": "timestamp",
        "date": "timestamp",
        "Datetime": "timestamp",
        "datetime": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    out = out.rename(columns=rename_map)
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns in prefill source: {missing}")

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])
    if "symbol" not in out.columns:
        out["symbol"] = symbol
    out["symbol"] = out["symbol"].astype(str).str.upper()
    return out[["timestamp", "open", "high", "low", "close", "volume", "symbol"]]


def _warmup_order_policies_from_prefill(
    *,
    processor: LiveBarProcessor,
    order_policies: dict[str, OptionOrderPolicy],
    symbols: list[str],
) -> dict[str, dict]:
    """
    Seed option-policy ATR state from prefilled 1m history so orders can be
    evaluated on the first live actionable bar.
    """
    latest_closed_by_symbol: dict[str, dict] = {}
    for symbol in symbols:
        policy = order_policies.get(symbol)
        buffer = processor._buffers.get(symbol)
        if policy is None or buffer is None:
            continue

        df = buffer.to_dataframe()
        if df is None or df.empty:
            continue

        # BarRingBuffer.to_dataframe() returns timestamp as index, so materialize
        # it as a column for deterministic warmup processing.
        if "timestamp" not in df.columns:
            df = df.reset_index()

        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            continue

        sym_df = df.sort_values("timestamp")
        agg = OhlcvAggregator(
            interval_minutes=processor._interval_minutes,
            label=processor._agg_label,
        )
        closed_count = 0
        last_atr = float("nan")
        last_closed: dict | None = None
        for row in sym_df.itertuples(index=False):
            ts = getattr(row, "timestamp", None)
            if ts is None:
                continue
            if not isinstance(ts, datetime):
                ts = pd.to_datetime(ts, utc=True, errors="coerce")
                if pd.isna(ts):
                    continue
            bar = {
                "timestamp": ts,
                "open": float(getattr(row, "open")),
                "high": float(getattr(row, "high")),
                "low": float(getattr(row, "low")),
                "close": float(getattr(row, "close")),
                "volume": float(getattr(row, "volume")),
                "symbol": symbol,
            }
            closed, _current = agg.update(bar)
            if closed:
                last_closed = closed
                last_atr = policy.on_interval_bar(closed_bar=closed)
                closed_count += 1

        if closed_count == 0:
            print(f"[live] Prefill ATR warmup for {symbol}: no completed 15m bars.")
            continue
        if last_closed is not None:
            latest_closed_by_symbol[symbol] = last_closed

        ready = math.isfinite(last_atr)
        print(
            f"[live] Prefill ATR warmup for {symbol}: "
            f"seeded_15m_bars={closed_count:,}, atr_ready={ready}, atr={last_atr}"
        )
    return latest_closed_by_symbol


def _run_startup_catchup_decision(
    *,
    processor: LiveBarProcessor,
    inference: LiveInferenceEngine,
    order_policies: dict[str, OptionOrderPolicy],
    execution_latches: dict[str, DirectionExecutionLatch] | None = None,
    latest_closed_by_symbol: dict[str, dict],
    symbols: list[str],
    print_tz: str,
    max_age_min: int,
) -> None:
    now_utc = pd.Timestamp.now(tz="UTC")
    for symbol in symbols:
        bar15 = latest_closed_by_symbol.get(symbol)
        if not bar15:
            print(f"[live] Startup catch-up {symbol}: skipped (no completed 15m bar).")
            continue

        ts = pd.to_datetime(bar15.get("timestamp"), utc=True, errors="coerce")
        if pd.isna(ts):
            print(f"[live] Startup catch-up {symbol}: skipped (invalid bar timestamp).")
            continue

        if max_age_min > 0:
            age_min = float((now_utc - ts).total_seconds() / 60.0)
            if age_min > float(max_age_min):
                print(
                    f"[live] Startup catch-up {symbol}: skipped (latest 15m bar age "
                    f"{age_min:.1f}m > {max_age_min}m)."
                )
                continue

        buffer = processor._buffers.get(symbol)
        if buffer is None:
            print(f"[live] Startup catch-up {symbol}: skipped (no prefilled buffer).")
            continue

        df_1m = buffer.to_dataframe()
        if df_1m.empty:
            print(f"[live] Startup catch-up {symbol}: skipped (empty prefilled buffer).")
            continue

        action = inference.on_15m_close(df_1m=df_1m, closed_bar=bar15)
        if action is None:
            print(f"[live] Startup catch-up {symbol}: no action (insufficient features/bars).")
            continue

        action_raw = float(action)
        action_pos = _action_to_position(action_raw)
        gate_status = "disabled"
        exec_pos = action_pos
        if _use_meta_direct_execution(inference):
            gate_status = "meta_direct"
        elif execution_latches is not None and symbol in execution_latches:
            gate = execution_latches[symbol].step(action_pos)
            exec_pos = int(gate.executed_pos)
            gate_status = str(gate.status)

        print(
            f"[live] Startup catch-up {symbol}: ts={_format_ts_local(ts, tz=print_tz)} "
            f"raw_action={action_raw:+.4f} raw_pos={action_pos:+d} "
            f"exec_pos={exec_pos:+d} gate={gate_status}"
        )
        policy_bar = dict(bar15)
        probs = inference.last_probs() or {}
        thresholds = inference.last_thresholds() or {}
        _print_meta_prob_log(
            prefix=f"[live] Startup catch-up {symbol} meta:",
            probs=probs,
            thresholds=thresholds,
        )
        policy_bar.update({k: v for k, v in probs.items() if v is not None})
        policy_bar.update(
            {
                "thr_enter_long": thresholds.get("enter_long"),
                "thr_enter_short": thresholds.get("enter_short"),
                "thr_exit_long": thresholds.get("exit_long"),
                "thr_exit_short": thresholds.get("exit_short"),
            }
        )
        result = order_policies[symbol].on_decision(
            action=float(action_raw if _use_meta_direct_execution(inference) else exec_pos),
            closed_bar=policy_bar,
            update_bar_state=False,
        )
        event = str(result.get("event", "unknown"))
        if event not in {"hold", "no_change", "intent_update"}:
            print(f"[live] Startup catch-up {symbol} order_policy event={event} details={result}")


def _replay_warmup_actions_from_prefill(
    *,
    processor: LiveBarProcessor,
    inference: LiveInferenceEngine,
    agent: object | None,
    execution_latches: dict[str, DirectionExecutionLatch],
    symbols: list[str],
    interval_minutes: int,
    print_tz: str,
) -> dict[str, int]:
    if agent is None:
        return {}
    counts: dict[str, int] = {}
    rule = str(getattr(inference, "_rule", f"{int(interval_minutes)}min"))
    label = str(getattr(inference, "_label", "left"))
    closed = str(getattr(inference, "_closed", "left"))
    tz = getattr(inference, "_tz", None)
    assume_tz = str(getattr(inference, "_assume_tz", "UTC"))

    for symbol in symbols:
        buffer = processor._buffers.get(symbol)
        if buffer is None:
            continue
        df_1m = buffer.to_dataframe()
        if df_1m is None or df_1m.empty:
            continue
        if not isinstance(df_1m.index, pd.DatetimeIndex):
            if "timestamp" not in df_1m.columns:
                continue
            df_1m = df_1m.copy()
            df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"], utc=True, errors="coerce")
            df_1m = df_1m.dropna(subset=["timestamp"]).set_index("timestamp")
        else:
            df_1m = df_1m.copy()
        if df_1m.index.tz is None:
            df_1m.index = df_1m.index.tz_localize(assume_tz)
        df_1m_utc = df_1m.sort_index().tz_convert("UTC")

        try:
            df_15m = build_15m(
                df_1m_utc,
                rule=rule,
                label=label,
                closed=closed,
                tz=tz,
                assume_tz=assume_tz,
            )
        except Exception:
            continue
        if df_15m.empty:
            continue

        action_count = 0
        try:
            records = agent.replay_warmup_actions(
                df_1m=df_1m_utc,
                df_15m=df_15m,
                apply_ga_probs=True,
            )
        except Exception:
            records = []
        for rec in records:
            ts = pd.to_datetime(rec.get("timestamp"), utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            raw_action = float(rec.get("action", 0.0))
            raw_pos = _action_to_position(raw_action)
            if isinstance(agent, (LiveMetaXGBAgent, LiveIndependentMetaXGBAgent)):
                exec_pos = int(raw_pos)
                gate_status = "meta_direct"
            else:
                gate = execution_latches[symbol].step(raw_pos)
                exec_pos = int(gate.executed_pos)
                gate_status = str(gate.status)
            print(
                f"[live] warmup {symbol} 15m={_format_ts_local(ts, tz=print_tz)} "
                f"raw={raw_action:+.4f} raw_pos={raw_pos:+d} exec={exec_pos:+d} gate={gate_status}"
            )
            action_count += 1

        if action_count > 0:
            counts[symbol] = action_count
    return counts


def _seed_agent_from_sync(
    *,
    processor: LiveBarProcessor,
    agent: object | None,
    sync_results: dict[str, dict[str, object]],
    symbols: list[str],
) -> None:
    if agent is None:
        return
    sync_fn = getattr(agent, "sync_live_position", None)
    if sync_fn is None:
        return
    for symbol in symbols:
        sync_result = sync_results.get(symbol) or {}
        desired_position = int(sync_result.get("position", 0) or 0)
        buffer = processor._buffers.get(symbol)
        if buffer is None:
            continue
        df_1m = buffer.to_dataframe()
        if df_1m is None or df_1m.empty:
            continue
        try:
            seed_result = sync_fn(
                desired_position=desired_position,
                df_1m=df_1m,
                entry_price=sync_result.get("avg_entry_price"),
            )
            print(f"[live] Meta startup seed {symbol}: {seed_result}")
        except Exception as exc:
            print(f"[live] Meta startup seed {symbol} failed: {exc}")


def _resolve_split_test_idx_path(*, split_root: Path, dataset_name: str, x_filename: str) -> Path | None:
    x_stem = Path(x_filename).stem
    candidates = [
        split_root / dataset_name / x_stem / "test_idx.npy",
        split_root / dataset_name / "test_idx.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_test_split_warmup_1m(
    *,
    symbol: str,
    dataset_name: str,
    x_filename: str,
) -> pd.DataFrame | None:
    try:
        from Data.load_data import (
            get_ticker_processed_base_dir,
            get_ticker_processed_split_dir,
            get_ticker_raw_dir,
        )
        from Data.retrieve_data import normalize_ticker

        clean = normalize_ticker(symbol)
        processed_root = get_ticker_processed_base_dir(clean)
        split_root = get_ticker_processed_split_dir(clean)
        test_idx_path = _resolve_split_test_idx_path(
            split_root=split_root,
            dataset_name=dataset_name,
            x_filename=x_filename,
        )
        if test_idx_path is None:
            return None

        plot_frame_path = processed_root / "datasets" / dataset_name / "plot_frame.parquet"
        if not plot_frame_path.exists():
            return None
        plot_df = pd.read_parquet(plot_frame_path)
        if not isinstance(plot_df.index, pd.DatetimeIndex):
            return None

        test_idx = np.load(test_idx_path).astype(int)
        if test_idx.size == 0:
            return None
        test_idx = test_idx[(test_idx >= 0) & (test_idx < len(plot_df))]
        if test_idx.size == 0:
            return None

        test_idx = np.sort(test_idx)
        ts_index = pd.DatetimeIndex(plot_df.index)
        start_15 = ts_index[int(test_idx[0])]
        end_15 = ts_index[int(test_idx[-1])]
        if start_15.tz is None:
            start_15 = start_15.tz_localize("UTC")
        if end_15.tz is None:
            end_15 = end_15.tz_localize("UTC")

        # 15m labels are bar starts; extend end by one 15m interval to capture full bar.
        start_utc = start_15.tz_convert("UTC")
        end_utc = (end_15 + pd.Timedelta(minutes=15)).tz_convert("UTC")

        raw_dir = get_ticker_raw_dir(clean)
        slug = clean.lower()
        candidates = [
            raw_dir / "1m_train.parquet",
            raw_dir / "train.parquet",
            raw_dir / f"{slug}_intraday_1min.parquet",
        ]
        raw_path = next((p for p in candidates if p.exists()), None)
        if raw_path is None:
            return None

        raw_df = pd.read_parquet(raw_path)
        raw_df = _normalize_prefill_1m_frame(raw_df, symbol=clean)
        raw_df = raw_df[
            (raw_df["timestamp"] >= start_utc)
            & (raw_df["timestamp"] <= end_utc)
        ]
        if raw_df.empty:
            return None

        raw_df = raw_df.sort_values("timestamp")
        return raw_df
    except Exception:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream 1m bars from Alpaca and aggregate into 15m candles."
    )
    parser.add_argument("--symbols", default="SPY", help="Comma-separated symbols.")
    parser.add_argument("--feed", default="IEX", help="IEX or SIP.")
    parser.add_argument("--interval", type=int, default=10, help="Aggregation interval in minutes.")
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=0,
        help="Ring buffer size in 1m bars. Use 0 for unlimited history in memory.",
    )
    parser.add_argument("--queue-size", type=int, default=5000, help="Max queued bars before dropping.")
    parser.add_argument("--print-1m", action="store_true", help="Print each 1m bar.")
    parser.add_argument("--print-15m", action="store_true", help="Print completed 15m bars.")
    parser.add_argument("--resample-label", default="left", help="Resample label (left/right).")
    parser.add_argument("--resample-closed", default="left", help="Resample closed (left/right).")
    parser.add_argument("--tz", default="America/New_York", help="Timezone for resampling (e.g. America/New_York).")
    parser.add_argument("--assume-tz", default="UTC", help="Assume timezone for naive timestamps.")
    parser.add_argument("--inference-mode", choices=["meta", "ppo", "none"], default="meta", help="Inference controller to run.")
    parser.add_argument("--model-path", default="Data/outputs/agent/ppo_model.pt", help="PPO model checkpoint.")
    parser.add_argument("--no-agent", action="store_true", help="Disable PPO inference.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions from policy (default is deterministic mean).")
    parser.add_argument("--device", default="auto", help="Device for inference (auto/cpu/cuda/mps).")
    parser.add_argument("--min-15m-bars", type=int, default=20, help="Minimum 15m bars before inference.")
    parser.add_argument("--no-pivot-probs", action="store_true", help="Disable pivot probability features.")
    parser.add_argument("--no-tb-probs", action="store_true", help="Disable triple-barrier probability features.")
    parser.add_argument("--fill-missing-prob", type=float, default=0.0, help="Value for missing prob features.")
    parser.add_argument("--session-open", default="09:30", help="Session open for time features.")
    parser.add_argument("--session-close", default="16:00", help="Session close for time features.")
    parser.add_argument("--ga-model-root", default="Data/models/ga_xgboost/10min", help="GA-XGB model root.")
    parser.add_argument("--ga-feature-list", default=None, help="Path to GA-XGB feature list txt.")
    parser.add_argument("--ga-dataset-name", default="10min", help="Dataset name for GA-XGB feature list fallback.")
    parser.add_argument(
        "--split-x-filename",
        default="X_10min_tree.parquet",
        help="Feature filename stem used to locate split indices for test warmup preload.",
    )
    parser.add_argument("--ga-pivot-label-dir", default="swing", help="Label dir for pivot GA-XGB models.")
    parser.add_argument("--ga-tb-label-dir", default="tb", help="Label dir for TB GA-XGB models.")
    parser.add_argument("--meta-model-root", default="Data/models/meta_xgboost/10min", help="Meta-XGB model root.")
    parser.add_argument(
        "--meta-base-frame-path",
        default="Data/inference/spy/10min/debug_matrices_warmup/spy/live_meta_matrix_on_trace_ts_live_2026_03_24.parquet",
        help="Optional cached 10m meta feature matrix path. Supports {symbol} and {symbol_lower}.",
    )
    parser.add_argument(
        "--meta-base-frame-append-lookback-days",
        type=int,
        default=120,
        help="When live bars extend past the cached matrix, recompute only this many days of 1m history and append.",
    )
    parser.add_argument(
        "--meta-entry-threshold",
        type=float,
        default=None,
        help="Optional execution threshold override for both meta long/short entries.",
    )
    parser.add_argument(
        "--meta-exit-threshold",
        type=float,
        default=None,
        help="Optional execution threshold override for both meta long/short exits.",
    )
    parser.add_argument("--meta-trail-activate-atr", type=float, default=0.75, help="Trail activation ATR used to build live exit context.")
    parser.add_argument("--meta-trail-atr", type=float, default=0.8, help="Base trail ATR used to build live exit context.")
    parser.add_argument("--meta-trail-atr-after-tp", type=float, default=0.5, help="Tightened trail ATR after TP is seen.")
    parser.add_argument("--meta-use-tp-to-tighten-trail", action=argparse.BooleanOptionalAction, default=True, help="Mirror training trail-tightening behavior in live exit context.")
    parser.add_argument("--env-file", default=".env", help="Path to .env with Alpaca credentials.")
    parser.add_argument(
        "--prefill-path",
        default="Data/raw/spy/spy_intraday_1min_live_2026_03_24.parquet",
        help="Optional CSV/Parquet path to prefill the 1m buffer for warm start. If --no-prefill-fetch is not set, Alpaca will only fetch and append the missing gap after the latest local bar.",
    )
    parser.add_argument(
        "--prefill-start",
        default="2026-01-30",
        help="Fetch historical 1m bars from Alpaca starting at this date/time (UTC).",
    )
    parser.add_argument(
        "--no-prefill-fetch",
        action="store_true",
        help="Disable all Alpaca historical prefill fetching, including gap-bridging on top of --prefill-path.",
    )
    parser.add_argument(
        "--meta-execution-mode",
        choices=["interval", "intrabar"],
        default="intrabar",
        help="For meta option execution: interval=execute on 10min close, intrabar=cache 10min intent and execute via 1m monitoring.",
    )
    parser.add_argument(
        "--prefill-tail",
        type=int,
        default=None,
        help="If set, use the most recent N rows per symbol; otherwise defaults to --buffer-size.",
    )
    parser.add_argument(
        "--prefill-prepend-warmup-bars",
        type=int,
        default=500,
        help="Always reserve and prepend this many split-test warmup 1m bars before fetched history (0 disables).",
    )
    parser.add_argument(
        "--enable-option-orders",
        action="store_true",
        help="Enable option order policy execution on each 15m inference action.",
    )
    parser.add_argument(
        "--no-startup-sync",
        action="store_true",
        help="Disable startup broker sync of open option positions into policy state.",
    )
    parser.add_argument(
        "--option-order-qty",
        type=int,
        default=1,
        help="Fallback max contracts when account/quote sizing is unavailable.",
    )
    parser.add_argument(
        "--option-price-mode",
        default="ask",
        choices=["ask", "mid", "bid", "last", "mark"],
        help="Price input for sizing max contracts (ask is conservative, mid for sim).",
    )
    parser.add_argument(
        "--option-action-ema-alpha",
        type=float,
        default=0.85,
        help="EMA alpha for action smoothing (higher = smoother).",
    )
    parser.add_argument(
        "--option-rebalance-deadband",
        type=float,
        default=0.10,
        help="Ignore action changes smaller than this after smoothing.",
    )
    parser.add_argument(
        "--option-max-step-contracts",
        type=int,
        default=2,
        help="Max absolute signed-contract change per decision step.",
    )
    parser.add_argument(
        "--option-max-contracts-cap",
        type=int,
        default=0,
        help="Optional hard cap on max contracts (<=0 disables cap).",
    )
    parser.add_argument(
        "--option-atr-mult",
        type=float,
        default=1.0,
        help="ATR multiplier for target strike distance (default 1.0 ATR).",
    )
    parser.add_argument(
        "--option-dte-cutoff",
        default="13:00",
        help="Local HH:MM cutoff; before cutoff use 0DTE, otherwise 1DTE.",
    )
    parser.add_argument(
        "--simulate-orders",
        action="store_true",
        help="Do not submit to Alpaca; print intended order payloads only.",
    )
    parser.add_argument(
        "--option-no-close-on-flat",
        action="store_true",
        help="Do not auto close open option when agent action goes flat.",
    )
    parser.add_argument(
        "--option-no-close-on-flip",
        action="store_true",
        help="Do not auto close existing option before flipping side.",
    )
    parser.add_argument(
        "--exec-entry-confirm-bars",
        type=int,
        default=1,
        help="Consecutive bars required to confirm a new entry while flat.",
    )
    parser.add_argument(
        "--exec-exit-confirm-bars",
        type=int,
        default=2,
        help="Consecutive bars required to confirm exit/flip while in-position.",
    )
    parser.add_argument(
        "--startup-catchup-order",
        action="store_true",
        help="On startup, evaluate the latest completed 15m bar from prefill and place any missed order immediately.",
    )
    parser.add_argument(
        "--startup-catchup-max-age-min",
        type=int,
        default=120,
        help="Skip startup catch-up if latest completed 15m bar is older than this many minutes (<=0 disables age guard).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    feed = _parse_feed(args.feed)
    inference_mode = "none" if args.no_agent else str(args.inference_mode).strip().lower()

    bar_queue: queue_mod.Queue = queue_mod.Queue(maxsize=args.queue_size)
    stop_event = threading.Event()

    agent = None
    precomputed_meta_frame: pd.DataFrame | None = None
    ga_feature_list = _resolve_ga_feature_list_path(
        symbol=symbols[0],
        dataset_name=args.ga_dataset_name,
        ga_feature_list=args.ga_feature_list,
        inference_enabled=inference_mode != "none",
        include_pivot_probs=not args.no_pivot_probs,
        include_tb_probs=not args.no_tb_probs,
    )

    if inference_mode != "none" and ga_feature_list is None and not (args.no_pivot_probs and args.no_tb_probs):
        print("[live] Warning: GA-XGB feature list not found; pivot/TB probs will be filled with defaults.")

    if inference_mode == "ppo":
        model_path = args.model_path
        if not model_path:
            raise SystemExit("Missing --model-path for PPO inference.")
        agent = _build_ppo_agent(
            model_path=model_path,
            deterministic=not args.stochastic,
            device=args.device,
            include_pivot_probs=not args.no_pivot_probs,
            include_tb_probs=not args.no_tb_probs,
            tz=args.tz or "America/New_York",
            assume_tz=args.assume_tz,
            session_open=args.session_open,
            session_close=args.session_close,
            min_15m_bars=args.min_15m_bars,
            fill_missing_prob=args.fill_missing_prob,
            ga_model_root=args.ga_model_root,
            ga_feature_list_path=ga_feature_list,
            ga_pivot_label_dir=args.ga_pivot_label_dir,
            ga_tb_label_dir=args.ga_tb_label_dir,
            ga_probs_mode="xgb",
            resample_label=args.resample_label,
            resample_closed=args.resample_closed,
            label_timeframe_rule=f"{args.interval}min",
        )
    elif inference_mode == "meta":
        if int(args.interval) != 10:
            print(f"[live] Warning: meta inference is trained for 10min bars; current --interval={args.interval}.")
        if args.meta_base_frame_path:
            try:
                meta_base_path = _resolve_symbolized_path(args.meta_base_frame_path, symbol=symbols[0])
                precomputed_meta_frame = _load_precomputed_meta_frame(
                    meta_base_path,
                    tz=args.tz or "America/New_York",
                )
                if not precomputed_meta_frame.empty:
                    print(
                        f"[live] Loaded cached meta base frame: rows={len(precomputed_meta_frame):,} "
                        f"range={precomputed_meta_frame.index.min()}..{precomputed_meta_frame.index.max()} "
                        f"path={meta_base_path}"
                    )
            except Exception as exc:
                print(f"[live] Cached meta base frame unavailable: {exc}")
        agent = _build_meta_agent(
            symbol=symbols[0],
            model_root=args.meta_model_root,
            ga_model_root=args.ga_model_root,
            ga_feature_list_path=ga_feature_list,
            include_pivot_probs=not args.no_pivot_probs,
            include_tb_probs=not args.no_tb_probs,
            pivot_label_dir=args.ga_pivot_label_dir,
            tb_label_dir=args.ga_tb_label_dir,
            tz=args.tz or "America/New_York",
            assume_tz=args.assume_tz,
            session_open=args.session_open,
            session_close=args.session_close,
            min_15m_bars=args.min_15m_bars,
            fill_missing_prob=args.fill_missing_prob,
            resample_label=args.resample_label,
            resample_closed=args.resample_closed,
            label_timeframe_rule=f"{args.interval}min",
            ga_probs_mode="xgb",
            trail_activate_atr=float(args.meta_trail_activate_atr),
            trail_atr=float(args.meta_trail_atr),
            trail_atr_after_tp=float(args.meta_trail_atr_after_tp),
            use_tp_to_tighten_trail=bool(args.meta_use_tp_to_tighten_trail),
            entry_threshold_override=args.meta_entry_threshold,
            exit_threshold_override=args.meta_exit_threshold,
            precomputed_base_frame=precomputed_meta_frame,
            precomputed_append_lookback_days=int(args.meta_base_frame_append_lookback_days),
            min_hold_bars=2,
            exit_entry_delta=0.15,
            soft_exit_confirm_bars=2,
            urgent_exit_prob=0.85,
            urgent_exit_delta=0.30,
            profit_protect_enabled=False,
            profit_protect_arm_atr=2.0,
            profit_protect_giveback_atr_long=0.75,
            profit_protect_giveback_atr_short=1.0,
        )
        print(
            f"[live] Meta-XGB inference enabled: model_root={args.meta_model_root} "
            f"timeframe={args.interval}min"
        )

    inference = LiveInferenceEngine(
        agent=agent,
        label=args.resample_label,
        closed=args.resample_closed,
        rule=f"{args.interval}min",
        tz=args.tz,
        assume_tz=args.assume_tz,
    )
    execution_latches: dict[str, DirectionExecutionLatch] = {
        symbol: DirectionExecutionLatch(
            entry_confirm_bars=max(1, int(args.exec_entry_confirm_bars)),
            exit_confirm_bars=max(1, int(args.exec_exit_confirm_bars)),
            initial_position=0,
        )
        for symbol in symbols
    }

    order_policies: dict[str, OptionOrderPolicy] | None = None
    startup_sync_results: dict[str, dict[str, object]] = {}
    if args.enable_option_orders:
        order_policies = {}
        for symbol in symbols:
            order_policies[symbol] = _build_option_order_policy(
                symbol=symbol,
                env_file=args.env_file,
                tz_name=args.tz,
                atr_multiplier=float(args.option_atr_mult),
                dte_cutoff_hhmm=args.option_dte_cutoff,
                qty=int(args.option_order_qty),
                close_on_flat=not args.option_no_close_on_flat,
                close_on_flip=not args.option_no_close_on_flip,
                submit_orders=not args.simulate_orders,
                ema_alpha=float(args.option_action_ema_alpha),
                rebalance_deadband=float(args.option_rebalance_deadband),
                max_step_contracts=int(args.option_max_step_contracts),
                price_mode=str(args.option_price_mode),
                max_contracts_fallback=int(args.option_order_qty),
                max_contracts_cap=int(args.option_max_contracts_cap),
                meta_execute_on_interval_close=str(args.meta_execution_mode).strip().lower() == "interval",
                meta_intrabar_execution_enabled=str(args.meta_execution_mode).strip().lower() == "intrabar",
                meta_trail_activate_atr=float(args.meta_trail_activate_atr),
                meta_trail_atr=float(args.meta_trail_atr),
                meta_trail_atr_after_tp=float(args.meta_trail_atr_after_tp),
                meta_use_tp_to_tighten_trail=bool(args.meta_use_tp_to_tighten_trail),
                meta_soft_exit_confirm_bars=2,
                meta_urgent_exit_prob=0.85,
                meta_urgent_exit_delta=0.30,
                meta_profit_protect_enabled=False,
                meta_profit_protect_arm_atr=2.0,
                meta_profit_protect_giveback_atr_long=0.75,
                meta_profit_protect_giveback_atr_short=1.0,
            )
        mode = "SIMULATED" if args.simulate_orders else "LIVE"
        print(f"[live] Option order policy enabled ({mode}) for symbols: {', '.join(symbols)}")
        if not args.no_startup_sync:
            for symbol in symbols:
                try:
                    sync_result = order_policies[symbol].sync_from_broker()
                    startup_sync_results[symbol] = sync_result
                    execution_latches[symbol].set_position(
                        order_policies[symbol].snapshot_state().get("position", 0)
                    )
                    print(f"[live] Startup sync {symbol}: {sync_result}")
                except Exception as exc:
                    print(f"[live] Startup sync {symbol} failed: {exc}")

    on_1m = (
        _make_1m_handler(
            print_tz=args.tz or "America/New_York",
            print_1m=bool(args.print_1m),
            order_policies=order_policies,
        )
        if (args.print_1m or order_policies is not None)
        else None
    )
    on_15m = _make_15m_handler(
        inference=inference,
        interval_minutes=int(args.interval),
        print_15m=args.print_15m,
        print_tz=args.tz or "America/New_York",
        execution_latches=execution_latches,
        order_policies=order_policies,
    )

    processor = LiveBarProcessor(
        interval_minutes=args.interval,
        buffer_size=args.buffer_size,
        agg_label=args.resample_label,
        on_1m=on_1m,
        on_15m_close=on_15m,
        regular_hours_only=True,
        tz_name=args.tz or "America/New_York",
        session_open=args.session_open,
        session_close=args.session_close,
    )
    print(
        f"[live] RTH-only live bar filter enabled: "
        f"{args.session_open}-{args.session_close} {args.tz or 'America/New_York'}"
    )
    if args.prefill_path:
        configured_prefill_path = Path(args.prefill_path)
        runtime_prefill_cache_path = _default_runtime_prefill_cache_path(configured_prefill_path)
        prefill_source_path = runtime_prefill_cache_path if runtime_prefill_cache_path.exists() else configured_prefill_path
        if prefill_source_path == runtime_prefill_cache_path:
            print(f"[live] Using runtime prefill cache: {runtime_prefill_cache_path}")
        prefill_df = _load_prefill_frame(prefill_source_path)
        if not args.no_prefill_fetch:
            try:
                prefill_df = _extend_prefill_with_alpaca_gap(
                    df=prefill_df,
                    symbols=symbols,
                    feed=feed,
                )
            except Exception as exc:
                print(f"[live] Prefill gap bridge failed: {exc}")
        compact_prefill_df = _prepare_runtime_prefill_frame(
            df=prefill_df,
            precomputed_meta_frame=precomputed_meta_frame,
            append_lookback_days=int(args.meta_base_frame_append_lookback_days),
            tz_name=args.tz or "America/New_York",
            session_open=args.session_open,
            session_close=args.session_close,
        )
        try:
            _persist_runtime_prefill_cache(
                df=compact_prefill_df,
                cache_path=runtime_prefill_cache_path,
            )
            print(
                f"[live] Updated runtime prefill cache: {runtime_prefill_cache_path} "
                f"rows={len(compact_prefill_df):,}"
            )
        except Exception as exc:
            print(f"[live] Runtime prefill cache update failed: {exc}")
        if args.buffer_size and args.buffer_size > 0 and args.prefill_tail and args.prefill_tail > args.buffer_size:
            print("[live] Warning: prefill-tail exceeds buffer-size; oldest rows will be dropped.")
        _prefill_buffers(
            processor=processor,
            df=compact_prefill_df,
            symbols=symbols,
            tail=args.prefill_tail,
        )
    elif args.prefill_start and not args.no_prefill_fetch:
        if args.buffer_size and args.buffer_size > 0 and args.prefill_tail and args.prefill_tail > args.buffer_size:
            print("[live] Warning: prefill-tail exceeds buffer-size; oldest rows will be dropped.")
        try:
            _prefill_from_alpaca(
                processor=processor,
                symbols=symbols,
                start=args.prefill_start,
                tail=args.prefill_tail,
                split_dataset_name=args.ga_dataset_name,
                split_x_filename=args.split_x_filename,
                prepend_split_test_warmup=True,
                prepend_warmup_bars=max(0, int(args.prefill_prepend_warmup_bars)),
                feed=feed,
            )
        except Exception as exc:
            print(f"[live] Prefill fetch failed: {exc}")
    if agent is not None:
        warmup_action_counts = _replay_warmup_actions_from_prefill(
            processor=processor,
            inference=inference,
            agent=agent,
            execution_latches=execution_latches,
            symbols=symbols,
            interval_minutes=int(args.interval),
            print_tz=args.tz or "America/New_York",
        )
        if warmup_action_counts:
            print(f"[live] Warmup action history replayed: {warmup_action_counts}")
        else:
            print("[live] Warmup action replay: no actionable 15m decisions from prefill.")
        if startup_sync_results:
            _seed_agent_from_sync(
                processor=processor,
                agent=agent,
                sync_results=startup_sync_results,
                symbols=symbols,
            )
    latest_closed_by_symbol: dict[str, dict] = {}
    if order_policies:
        latest_closed_by_symbol = _warmup_order_policies_from_prefill(
            processor=processor,
            order_policies=order_policies,
            symbols=symbols,
        )
        if args.startup_catchup_order:
            _run_startup_catchup_decision(
                processor=processor,
                inference=inference,
                order_policies=order_policies,
                execution_latches=execution_latches,
                latest_closed_by_symbol=latest_closed_by_symbol,
                symbols=symbols,
                print_tz=args.tz or "America/New_York",
                max_age_min=int(args.startup_catchup_max_age_min),
            )

    streamer = AlpacaBarStreamer(
        symbols=symbols,
        feed=feed,
        env_file=args.env_file,
        queue=bar_queue,
    )
    streamer.start_in_thread()

    print("Streaming started. Ctrl+C to stop.")
    def _handle_sigint(_signum, _frame) -> None:
        stop_event.set()
        streamer.stop()

    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except Exception:
        # Signal registration can fail in some environments; fallback to KeyboardInterrupt.
        pass

    try:
        last_broker_poll = 0.0
        while not stop_event.is_set():
            if order_policies is not None:
                now = time.monotonic()
                if now - last_broker_poll >= 30.0:
                    last_broker_poll = now
                    for symbol, policy in order_policies.items():
                        if policy.has_pending_broker_reconcile():
                            reconcile_result = policy.reconcile_pending_broker_order(logger=print)
                            if args.use_execution_latch:
                                execution_latches[symbol].set_position(
                                    policy.snapshot_state().get("position", 0)
                                )
                            print(f"[live] Broker reconcile {symbol}: {reconcile_result}")
                        else:
                            reconcile_result = policy.reconcile_with_broker(logger=print, force=True)
                            if reconcile_result.get("changed"):
                                if args.use_execution_latch:
                                    execution_latches[symbol].set_position(
                                        policy.snapshot_state().get("position", 0)
                                    )
                                print(f"[live] Broker sync {symbol}: {reconcile_result}")
            try:
                bar = bar_queue.get(timeout=0.5)
            except queue_mod.Empty:
                continue
            processor.handle_bar(bar)
    except KeyboardInterrupt:
        stop_event.set()
        print("Stopping stream...")
    finally:
        streamer.stop()
        streamer.join(timeout=5)
        time.sleep(0.2)


if __name__ == "__main__":
    main()
