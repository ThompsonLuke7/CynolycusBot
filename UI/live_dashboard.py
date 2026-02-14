from __future__ import annotations

import argparse
import json
import queue as queue_mod
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TYPE_CHECKING, Callable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from API.Alpaca_API.market_data.bar_aggregator import OhlcvAggregator
from API.Alpaca_API.market_data.bar_buffer import BarRingBuffer

UI_BUILD = "2026-02-10-dashboard-replay-v2"

if TYPE_CHECKING:
    from API.Alpaca_API.inference.live_inference import LiveInferenceEngine, LivePPOAgent
    from API.Alpaca_API.market_data.live_stream import AlpacaBarStreamer
    from Policy.order_policy import OptionOrderPolicy


def _ts_iso(ts: Any) -> str | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.isoformat()
    if isinstance(ts, pd.Timestamp):
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        return ts.isoformat()
    parsed = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(parsed):
        return str(ts)
    return parsed.isoformat()


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return _ts_iso(value)
    if isinstance(value, pd.Series):
        return {str(k): _json_safe(v) for k, v in value.to_dict().items()}
    if isinstance(value, pd.DataFrame):
        return [_json_safe(x) for x in value.to_dict("records")]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, deque)):
        return [_json_safe(x) for x in value]
    return str(value)


def _coerce_int(raw: Any, default: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _coerce_float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _coerce_bool(raw: Any, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    key = str(raw).strip().lower()
    if key in {"1", "true", "yes", "y", "on"}:
        return True
    if key in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_bar(bar: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "symbol": str(bar.get("symbol", "")).upper(),
        "timestamp": _ts_iso(bar.get("timestamp")),
    }
    for key in ("open", "high", "low", "close", "volume", "trade_count", "vwap"):
        if key in bar:
            val = bar.get(key)
            num = _coerce_float(val, float("nan"))
            out[key] = num if np.isfinite(num) else None
    for key, val in bar.items():
        if not isinstance(key, str) or not key.startswith("p_"):
            continue
        num = _coerce_float(val, float("nan"))
        out[key] = num if np.isfinite(num) else None
    return out


def _action_to_position(action: int) -> int:
    if int(action) == 1:
        return 1
    if int(action) == 2:
        return -1
    return 0


def _agent_action_event(prev_action: int | None, action: int) -> str | None:
    if prev_action is None:
        return None
    prev_pos = _action_to_position(prev_action)
    cur_pos = _action_to_position(action)
    if prev_pos == cur_pos:
        return None
    if prev_pos == 0 and cur_pos == 1:
        return "enter_long"
    if prev_pos == 0 and cur_pos == -1:
        return "enter_short"
    if prev_pos == 1 and cur_pos == 0:
        return "exit_long"
    if prev_pos == -1 and cur_pos == 0:
        return "exit_short"
    if prev_pos == -1 and cur_pos == 1:
        return "flip_to_long"
    if prev_pos == 1 and cur_pos == -1:
        return "flip_to_short"
    return None


def _load_history_file(path: Path, *, assume_tz: str = "UTC") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing history file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("History file must be .csv or .parquet")

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
    df = df.rename(columns=rename_map)
    if "timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
        else:
            raise ValueError("History data must include a timestamp column or DatetimeIndex.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if getattr(df["timestamp"].dt, "tz", None) is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(assume_tz)
    return df


def _apply_regular_hours(df: pd.DataFrame, *, tz: str = "America/New_York") -> pd.DataFrame:
    ts = df["timestamp"].dt.tz_convert(tz)
    minutes = ts.dt.hour * 60 + ts.dt.minute
    regular_mask = minutes.between(570, 960)
    return df.loc[regular_mask].copy()


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


def _load_test_split_warmup_1m(*, symbol: str, dataset_name: str, x_filename: str) -> pd.DataFrame | None:
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
        return raw_df.sort_values("timestamp")
    except Exception:
        return None


def _load_agent_matrix_probs(*, symbol: str, dataset_name: str) -> pd.DataFrame | None:
    try:
        from Data.retrieve_data import normalize_ticker

        clean = normalize_ticker(symbol).lower()
        path = (
            Path("Data")
            / "models"
            / "agent"
            / dataset_name
            / clean
            / "agent_matrix.parquet"
        )
        if not path.exists():
            return None
        df = pd.read_parquet(
            path,
            columns=[
                "timestamp",
                "p_pivot_long",
                "p_pivot_short",
                "p_tb_long",
                "p_tb_short",
            ],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")
        return df
    except Exception:
        return None


class ReplayBarProcessor:
    def __init__(
        self,
        *,
        interval_minutes: int = 15,
        buffer_size: int = 5000,
        agg_label: str = "left",
        on_1m: Callable[[str, dict, BarRingBuffer], None] | None = None,
        on_15m_close: Callable[[str, dict, BarRingBuffer], None] | None = None,
    ) -> None:
        self._interval_minutes = int(interval_minutes)
        self._buffer_size = int(buffer_size)
        self._agg_label = str(agg_label)
        self._on_1m = on_1m
        self._on_15m_close = on_15m_close
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

    def handle_bar(self, bar: dict) -> None:
        symbol = str(bar.get("symbol", "")).upper()
        buffer = self._get_buffer(symbol)
        agg = self._get_aggregator(symbol)
        buffer.append(bar)
        if self._on_1m is not None:
            self._on_1m(symbol, bar, buffer)
        closed, _current = agg.update(bar)
        if closed and self._on_15m_close is not None:
            self._on_15m_close(symbol, closed, buffer)


@dataclass(frozen=True)
class SessionConfig:
    symbols: list[str]
    runner_mode: str = "live"
    feed: str = "IEX"
    interval: int = 15
    buffer_size: int = 5000
    queue_size: int = 5000
    resample_label: str = "left"
    resample_closed: str = "left"
    tz: str = "America/New_York"
    assume_tz: str = "UTC"
    model_path: str = "Data/outputs/agent/ppo_model.pt"
    no_agent: bool = False
    stochastic: bool = False
    device: str = "auto"
    min_15m_bars: int = 20
    no_pivot_probs: bool = False
    no_tb_probs: bool = False
    fill_missing_prob: float = 0.0
    session_open: str = "09:30"
    session_close: str = "16:00"
    ga_model_root: str = "Data/models/ga_xgboost/15min"
    ga_feature_list: str | None = None
    ga_dataset_name: str = "15min"
    split_x_filename: str = "X_15min_tree.parquet"
    ga_pivot_label_dir: str = "pivots"
    ga_tb_label_dir: str = "tb"
    env_file: str = ".env"
    prefill_path: str | None = None
    prefill_start: str = "2026-01-30"
    no_prefill_fetch: bool = False
    prefill_tail: int | None = None
    enable_option_orders: bool = False
    no_startup_sync: bool = False
    option_order_qty: int = 1
    option_atr_mult: float = 1.0
    option_dte_cutoff: str = "14:00"
    simulate_orders: bool = True
    option_no_close_on_flat: bool = False
    option_no_close_on_flip: bool = False
    startup_catchup_order: bool = False
    startup_catchup_max_age_min: int = 120
    replay_data_path: str = "Data/raw/spy/inference_buffer_1m.parquet"
    replay_start: str | None = None
    replay_end: str | None = None
    replay_regular_only: bool = False
    replay_sleep: float = 0.0
    replay_max_bars: int | None = None
    replay_no_prepend_split_test_warmup: bool = False

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SessionConfig":
        symbols_raw = str(payload.get("symbols", "SPY"))
        symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
        if not symbols:
            symbols = ["SPY"]
        return cls(
            symbols=symbols,
            runner_mode=str(payload.get("runner_mode", "live")).strip().lower(),
            feed=str(payload.get("feed", "IEX")).upper(),
            interval=max(1, _coerce_int(payload.get("interval"), 15)),
            buffer_size=max(100, _coerce_int(payload.get("buffer_size"), 5000)),
            queue_size=max(100, _coerce_int(payload.get("queue_size"), 5000)),
            resample_label=str(payload.get("resample_label", "left")).strip().lower(),
            resample_closed=str(payload.get("resample_closed", "left")).strip().lower(),
            tz=str(payload.get("tz", "America/New_York")),
            assume_tz=str(payload.get("assume_tz", "UTC")),
            model_path=str(payload.get("model_path", "Data/outputs/agent/ppo_model.pt")),
            no_agent=_coerce_bool(payload.get("no_agent"), False),
            stochastic=_coerce_bool(payload.get("stochastic"), False),
            device=str(payload.get("device", "auto")),
            min_15m_bars=max(5, _coerce_int(payload.get("min_15m_bars"), 20)),
            no_pivot_probs=_coerce_bool(payload.get("no_pivot_probs"), False),
            no_tb_probs=_coerce_bool(payload.get("no_tb_probs"), False),
            fill_missing_prob=_coerce_float(payload.get("fill_missing_prob"), 0.0),
            session_open=str(payload.get("session_open", "09:30")),
            session_close=str(payload.get("session_close", "16:00")),
            ga_model_root=str(payload.get("ga_model_root", "Data/models/ga_xgboost/15min")),
            ga_feature_list=payload.get("ga_feature_list"),
            ga_dataset_name=str(payload.get("ga_dataset_name", "15min")),
            split_x_filename=str(payload.get("split_x_filename", "X_15min_tree.parquet")),
            ga_pivot_label_dir=str(payload.get("ga_pivot_label_dir", "pivots")),
            ga_tb_label_dir=str(payload.get("ga_tb_label_dir", "tb")),
            env_file=str(payload.get("env_file", ".env")),
            prefill_path=payload.get("prefill_path"),
            prefill_start=str(payload.get("prefill_start", "2026-01-30")),
            no_prefill_fetch=_coerce_bool(payload.get("no_prefill_fetch"), False),
            prefill_tail=(
                None
                if payload.get("prefill_tail") in (None, "")
                else max(50, _coerce_int(payload.get("prefill_tail"), 0))
            ),
            enable_option_orders=_coerce_bool(payload.get("enable_option_orders"), False),
            no_startup_sync=_coerce_bool(payload.get("no_startup_sync"), False),
            option_order_qty=max(1, _coerce_int(payload.get("option_order_qty"), 1)),
            option_atr_mult=max(0.0, _coerce_float(payload.get("option_atr_mult"), 1.0)),
            option_dte_cutoff=str(payload.get("option_dte_cutoff", "14:00")),
            simulate_orders=_coerce_bool(payload.get("simulate_orders"), True),
            option_no_close_on_flat=_coerce_bool(payload.get("option_no_close_on_flat"), False),
            option_no_close_on_flip=_coerce_bool(payload.get("option_no_close_on_flip"), False),
            startup_catchup_order=_coerce_bool(payload.get("startup_catchup_order"), False),
            startup_catchup_max_age_min=_coerce_int(payload.get("startup_catchup_max_age_min"), 120),
            replay_data_path=str(payload.get("replay_data_path", "Data/raw/spy/inference_buffer_1m.parquet")),
            replay_start=payload.get("replay_start"),
            replay_end=payload.get("replay_end"),
            replay_regular_only=_coerce_bool(payload.get("replay_regular_only"), False),
            replay_sleep=max(0.0, _coerce_float(payload.get("replay_sleep"), 0.0)),
            replay_max_bars=(
                None
                if payload.get("replay_max_bars") in (None, "")
                else max(1, _coerce_int(payload.get("replay_max_bars"), 0))
            ),
            replay_no_prepend_split_test_warmup=_coerce_bool(
                payload.get("replay_no_prepend_split_test_warmup"),
                False,
            ),
        )


class EventBroker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue_mod.Queue] = []

    def subscribe(self) -> queue_mod.Queue:
        q: queue_mod.Queue = queue_mod.Queue(maxsize=2000)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue_mod.Queue) -> None:
        with self._lock:
            self._subscribers = [sub for sub in self._subscribers if sub is not q]

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except queue_mod.Full:
                try:
                    q.get_nowait()
                except queue_mod.Empty:
                    pass
                try:
                    q.put_nowait(event)
                except queue_mod.Full:
                    pass


class DashboardStore:
    def __init__(self, *, max_bars: int = 1000, max_events: int = 250) -> None:
        self._max_bars = int(max_bars)
        self._lock = threading.Lock()
        self._running = False
        self._started_at: str | None = None
        self._stopped_at: str | None = None
        self._last_error: str | None = None
        self._status_message: str = "idle"
        self._config: dict[str, Any] = {}
        self._symbols: list[str] = []
        self._bars_1m: dict[str, deque] = {}
        self._bars_15m: dict[str, deque] = {}
        self._last_actions: dict[str, dict[str, Any]] = {}
        self._agent_state: dict[str, Any] | None = None
        self._policy_state: dict[str, dict[str, Any]] = {}
        self._broker_state: dict[str, dict[str, Any]] = {}
        self._action_events: deque = deque(maxlen=max_events * 6)
        self._trade_events: deque = deque(maxlen=max_events)
        self._log_events: deque = deque(maxlen=max_events)

    def start(self, config: SessionConfig) -> None:
        symbols = list(config.symbols)
        with self._lock:
            self._running = True
            self._started_at = _ts_iso(datetime.utcnow())
            self._stopped_at = None
            self._last_error = None
            self._status_message = "starting"
            self._config = _json_safe(config.__dict__)
            self._symbols = symbols
            self._bars_1m = {s: deque(maxlen=self._max_bars) for s in symbols}
            self._bars_15m = {s: deque(maxlen=self._max_bars) for s in symbols}
            self._last_actions = {}
            self._agent_state = None
            self._policy_state = {}
            self._broker_state = {}
            self._action_events.clear()
            self._trade_events.clear()
            self._log_events.clear()

    def stop(self, *, error: str | None = None) -> None:
        with self._lock:
            self._running = False
            self._stopped_at = _ts_iso(datetime.utcnow())
            if error:
                self._last_error = str(error)
            self._status_message = "stopped" if not error else f"error: {error}"

    def set_status_message(self, message: str) -> None:
        with self._lock:
            self._status_message = str(message)

    def add_1m_bar(self, symbol: str, bar: dict[str, Any], *, buffer_size: int | None = None) -> None:
        payload = _normalize_bar(bar)
        if buffer_size is not None:
            payload["buffer_size"] = int(buffer_size)
        with self._lock:
            self._bars_1m.setdefault(symbol, deque(maxlen=self._max_bars)).append(payload)

    def add_15m_bar(self, symbol: str, bar: dict[str, Any]) -> None:
        payload = _normalize_bar(bar)
        with self._lock:
            self._bars_15m.setdefault(symbol, deque(maxlen=self._max_bars)).append(payload)

    def set_last_action(self, symbol: str, *, action: int, ts: Any, close: Any) -> None:
        close_val = _coerce_float(close, float("nan"))
        payload = {
            "symbol": symbol,
            "action": int(action),
            "timestamp": _ts_iso(ts),
            "close": close_val if np.isfinite(close_val) else None,
        }
        with self._lock:
            self._last_actions[symbol] = payload
            self._action_events.append(payload)

    def set_agent_state(self, state: dict[str, Any] | None) -> None:
        with self._lock:
            self._agent_state = _json_safe(state) if state is not None else None

    def set_policy_state(self, symbol: str, state: dict[str, Any] | None) -> None:
        if state is None:
            return
        with self._lock:
            self._policy_state[symbol] = _json_safe(state)

    def set_broker_state(self, symbol: str, state: dict[str, Any] | None) -> None:
        if state is None:
            return
        with self._lock:
            self._broker_state[symbol] = _json_safe(state)

    def add_trade_event(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._trade_events.append(_json_safe(payload))

    def add_log_event(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._log_events.append(_json_safe(payload))

    def seed_from_buffers(self, buffers: dict[str, Any]) -> None:
        with self._lock:
            for symbol, buf in buffers.items():
                if buf is None:
                    continue
                df = buf.to_dataframe()
                if df.empty:
                    continue
                if "timestamp" not in df.columns:
                    df = df.reset_index()
                records = df.tail(self._max_bars).to_dict("records")
                target = self._bars_1m.setdefault(symbol, deque(maxlen=self._max_bars))
                target.clear()
                for row in records:
                    target.append(
                        _normalize_bar(
                            {
                                "symbol": symbol,
                                "timestamp": row.get("timestamp"),
                                "open": row.get("open"),
                                "high": row.get("high"),
                                "low": row.get("low"),
                                "close": row.get("close"),
                                "volume": row.get("volume"),
                            }
                        )
                    )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ui_build": UI_BUILD,
                "running": self._running,
                "started_at": self._started_at,
                "stopped_at": self._stopped_at,
                "last_error": self._last_error,
                "status_message": self._status_message,
                "config": _json_safe(self._config),
                "symbols": list(self._symbols),
                "bars_1m": {k: list(v) for k, v in self._bars_1m.items()},
                "bars_15m": {k: list(v) for k, v in self._bars_15m.items()},
                "last_actions": _json_safe(self._last_actions),
                "agent_state": _json_safe(self._agent_state),
                "policy_state": _json_safe(self._policy_state),
                "broker_state": _json_safe(self._broker_state),
                "action_events": list(self._action_events),
                "trade_events": list(self._trade_events),
                "logs": list(self._log_events),
            }


class LiveSession:
    def __init__(self, store: DashboardStore, broker: EventBroker) -> None:
        self._store = store
        self._broker = broker
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._streamer: AlpacaBarStreamer | None = None
        self._config: SessionConfig | None = None

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._thread and self._thread.is_alive())

    def start(self, config: SessionConfig) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeError("Live session already running.")
            self._config = config
            self._stop_event = threading.Event()
            self._store.start(config)
            self._emit_status(running=True, message="starting", extra={"config": config.__dict__})
            thread = threading.Thread(target=self._run, name="live-dashboard-session", daemon=True)
            self._thread = thread
            thread.start()

    def stop(self) -> None:
        with self._lock:
            stop_event = self._stop_event
            streamer = self._streamer
            thread = self._thread
        if stop_event is not None:
            stop_event.set()
        if streamer is not None:
            try:
                streamer.stop()
            except Exception:
                pass
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=6.0)
            if thread.is_alive():
                # Fallback: release UI controls even if background thread is still blocked.
                with self._lock:
                    self._thread = None
                    self._stop_event = None
                    self._streamer = None
                msg = "stop timed out; session marked stopped"
                self._store.stop(error=msg)
                self._emit_status(running=False, message=msg)

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "type": event_type,
            "timestamp": _ts_iso(datetime.utcnow()),
            "payload": _json_safe(payload),
        }
        self._broker.publish(event)

    def _emit_status(self, *, running: bool, message: str, extra: dict[str, Any] | None = None) -> None:
        self._store.set_status_message(message)
        payload: dict[str, Any] = {"running": bool(running), "message": str(message)}
        if extra:
            payload.update(extra)
        self._emit("status", payload)

    def _emit_state_sync(self) -> None:
        self._emit("state_sync", {"state": self._store.snapshot()})

    def _policy_logger(self, symbol: str) -> Callable[[str], None]:
        def _logger(message: str) -> None:
            payload = {"symbol": symbol, "message": str(message)}
            self._store.add_log_event(payload)
            self._emit("log", payload)

        return _logger

    def _resolve_ga_feature_list(self, cfg: SessionConfig) -> str | None:
        if cfg.ga_feature_list:
            return cfg.ga_feature_list
        try:
            from Data.load_data import get_ticker_processed_base_dir
            from Data.retrieve_data import normalize_ticker

            ticker = normalize_ticker(cfg.symbols[0])
            candidate = (
                get_ticker_processed_base_dir(ticker)
                / "datasets"
                / cfg.ga_dataset_name
                / f"features_X_{cfg.ga_dataset_name}_tree.txt"
            )
            if candidate.exists():
                return str(candidate)
        except Exception:
            return None
        return None

    def _seed_15m_from_prefill(
        self,
        *,
        cfg: SessionConfig,
        processor: Any,
        agent: Any | None,
    ) -> None:
        from API.Alpaca_API.inference.live_inference import build_15m

        seeded_counts: dict[str, int] = {}
        for symbol in cfg.symbols:
            buffer = processor._buffers.get(symbol)
            if buffer is None:
                continue
            df_1m = buffer.to_dataframe()
            if df_1m is None or df_1m.empty:
                continue

            try:
                df_15m = build_15m(
                    df_1m,
                    rule=f"{cfg.interval}min",
                    label=cfg.resample_label,
                    closed=cfg.resample_closed,
                    tz=cfg.tz,
                    assume_tz=cfg.assume_tz,
                )
            except Exception:
                continue
            if df_15m.empty:
                continue

            df_plot = df_15m
            if agent is not None:
                try:
                    # Precompute display probabilities from prefill without
                    # emitting actions/orders before live bars arrive.
                    df_plot = agent._maybe_add_ga_probs(df_1m=df_1m, df_15m=df_15m, target_ts=None)  # noqa: SLF001
                except Exception:
                    df_plot = df_15m

            if "timestamp" not in df_plot.columns:
                df_plot = df_plot.reset_index().rename(columns={df_plot.index.name or "index": "timestamp"})
            else:
                df_plot = df_plot.reset_index(drop=True)
            df_plot = df_plot.tail(self._store._max_bars)

            count = 0
            for row in df_plot.to_dict("records"):
                bar = {
                    "symbol": symbol,
                    "timestamp": row.get("timestamp"),
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": row.get("close"),
                    "volume": row.get("volume"),
                }
                for key in ("p_pivot_long", "p_pivot_short", "p_tb_long", "p_tb_short"):
                    if key in row:
                        bar[key] = row.get(key)
                self._store.add_15m_bar(symbol, bar)
                count += 1

            if count > 0:
                seeded_counts[symbol] = count

        if seeded_counts:
            self._emit(
                "log",
                {
                    "symbol": "SYSTEM",
                    "message": f"[live] seeded 15m/prob history from prefill: {seeded_counts}",
                },
            )

    def _run_replay(self, *, cfg: SessionConfig, stop_event: threading.Event) -> None:
        symbols = cfg.symbols
        self._emit_status(running=True, message=f"replay loading {cfg.replay_data_path}")
        self._store.add_log_event(
            {
                "symbol": "SYSTEM",
                "message": f"[replay] loading data: {cfg.replay_data_path}",
            }
        )

        df = _load_history_file(Path(cfg.replay_data_path), assume_tz=cfg.assume_tz)
        df["__source"] = "replay_file"
        if cfg.replay_regular_only:
            df = _apply_regular_hours(df, tz=cfg.tz)

        if "symbol" not in df.columns:
            if len(symbols) != 1:
                raise ValueError("Replay data missing symbol column; use one symbol or include symbol column.")
            df["symbol"] = symbols[0]
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df = df[df["symbol"].isin(symbols)]

        if cfg.replay_start:
            start = pd.to_datetime(cfg.replay_start, utc=True, errors="coerce")
            df = df[df["timestamp"] >= start]
        if cfg.replay_end:
            end = pd.to_datetime(cfg.replay_end, utc=True, errors="coerce")
            df = df[df["timestamp"] <= end]

        if not cfg.replay_no_prepend_split_test_warmup:
            warm_frames: list[pd.DataFrame] = []
            for symbol in symbols:
                warm_df = _load_test_split_warmup_1m(
                    symbol=symbol,
                    dataset_name=cfg.ga_dataset_name,
                    x_filename=cfg.split_x_filename,
                )
                if warm_df is None or warm_df.empty:
                    continue
                if cfg.replay_regular_only:
                    warm_df = _apply_regular_hours(warm_df, tz=cfg.tz)
                warm_df = warm_df.copy()
                warm_df["__source"] = "warmup_split"
                warm_frames.append(warm_df)
            if warm_frames:
                warmup = pd.concat(warm_frames, axis=0, ignore_index=True)
                df = pd.concat([warmup, df], axis=0, ignore_index=True)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
                df = df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
                self._emit(
                    "log",
                    {
                        "symbol": "SYSTEM",
                        "message": (
                            f"[replay] prepended split-test warmup bars: {len(warmup):,} "
                            f"(combined rows: {len(df):,})"
                        ),
                    },
                )

        required = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Replay data missing required columns: {missing}")
        df = df.sort_values("timestamp")

        # Replay cap is applied to the most recent bars, so N=5000 means
        # "start ~5000 bars ago and play forward to latest."
        if cfg.replay_max_bars is not None:
            keep_n = int(cfg.replay_max_bars)
            if keep_n > 0:
                df = df.tail(keep_n).copy()

        if "__source" in df.columns:
            src_counts = (
                df["__source"]
                .astype(str)
                .value_counts(dropna=False)
                .to_dict()
            )
            self._emit(
                "log",
                {
                    "symbol": "SYSTEM",
                    "message": f"[replay] selected-bar source mix: {src_counts}",
                },
            )
        self._store.add_log_event(
            {
                "symbol": "SYSTEM",
                "message": f"[replay] rows loaded: {len(df):,}.",
            }
        )

        agent = None
        inference = None
        if not cfg.no_agent:
            self._emit_status(running=True, message="replay loading PPO agent")
            from API.Alpaca_API.inference.live_inference import LiveInferenceEngine, LivePPOAgent

            ga_feature_list = self._resolve_ga_feature_list(cfg)
            ga_probs_frame = _load_agent_matrix_probs(symbol=symbols[0], dataset_name=cfg.ga_dataset_name)
            ga_probs_mode = "frame" if ga_probs_frame is not None else "xgb"

            if ga_probs_frame is not None:
                # If the frame doesn't overlap the replay time range, fall back to XGB.
                try:
                    frame_idx = ga_probs_frame.index
                    if isinstance(frame_idx, pd.DatetimeIndex):
                        ts_min = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").min()
                        ts_max = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").max()
                        if pd.notna(ts_min) and pd.notna(ts_max):
                            if frame_idx.tz is None:
                                frame_min = frame_idx.min().tz_localize("UTC")
                                frame_max = frame_idx.max().tz_localize("UTC")
                            else:
                                frame_min = frame_idx.min().tz_convert("UTC")
                                frame_max = frame_idx.max().tz_convert("UTC")
                            if frame_max < ts_min or frame_min > ts_max:
                                ga_probs_frame = None
                                ga_probs_mode = "xgb"
                                self._emit(
                                    "log",
                                    {
                                        "symbol": "SYSTEM",
                                        "message": "[replay] agent_matrix probs out of range; falling back to XGB.",
                                    },
                                )
                    if ga_probs_frame is not None:
                        self._emit(
                            "log",
                            {
                                "symbol": "SYSTEM",
                                "message": "[replay] using agent_matrix probabilities for PPO features.",
                            },
                        )
                except Exception:
                    pass

            if ga_probs_frame is None and ga_feature_list is None and not cfg.no_pivot_probs and not cfg.no_tb_probs:
                self._emit(
                    "log",
                    {
                        "symbol": "SYSTEM",
                        "message": "[replay] GA-XGB feature list missing; pivot/TB probs will be zeros.",
                    },
                )
            agent = LivePPOAgent(
                model_path=cfg.model_path,
                deterministic=not cfg.stochastic,
                device=cfg.device,
                include_pivot_probs=not cfg.no_pivot_probs,
                include_tb_probs=not cfg.no_tb_probs,
                tz=cfg.tz or "America/New_York",
                assume_tz=cfg.assume_tz,
                session_open=cfg.session_open,
                session_close=cfg.session_close,
                min_15m_bars=cfg.min_15m_bars,
                fill_missing_prob=cfg.fill_missing_prob,
                ga_model_root=cfg.ga_model_root if ga_feature_list else None,
                ga_feature_list_path=ga_feature_list,
                ga_pivot_label_dir=cfg.ga_pivot_label_dir,
                ga_tb_label_dir=cfg.ga_tb_label_dir,
                ga_probs_frame=ga_probs_frame,
                ga_probs_mode=ga_probs_mode,
                resample_label=cfg.resample_label,
                resample_closed=cfg.resample_closed,
                label_timeframe_rule=f"{cfg.interval}min",
            )
            self._store.set_agent_state(agent.snapshot_state())
            self._emit("agent_state", {"state": agent.snapshot_state()})

            inference = LiveInferenceEngine(
                agent=agent,
                label=cfg.resample_label,
                closed=cfg.resample_closed,
                rule=f"{cfg.interval}min",
                tz=cfg.tz,
                assume_tz=cfg.assume_tz,
            )

        order_policies = None
        simulated_orders: dict[str, deque] = {}
        last_action_by_symbol: dict[str, int] = {}
        if cfg.enable_option_orders:
            from Policy.order_policy import OptionOrderPolicy, OptionOrderPolicyConfig

            order_policies = {}
            for symbol in symbols:
                pol_cfg = OptionOrderPolicyConfig(
                    underlying=symbol,
                    env_file=cfg.env_file,
                    tz_name=cfg.tz or "America/New_York",
                    atr_multiplier=float(cfg.option_atr_mult),
                    dte_cutoff_hhmm=cfg.option_dte_cutoff,
                    qty=int(cfg.option_order_qty),
                    close_on_flat=not cfg.option_no_close_on_flat,
                    close_on_flip=not cfg.option_no_close_on_flip,
                    submit_orders=False,
                )
                policy = OptionOrderPolicy(pol_cfg)
                order_policies[symbol] = policy
                simulated_orders[symbol] = deque(maxlen=20)
                self._store.set_policy_state(symbol, policy.snapshot_state())
            self._emit(
                "log",
                {
                    "symbol": "SYSTEM",
                    "message": "[replay] option orders enabled in SIMULATED mode.",
                },
            )

        def _set_sim_broker_state(symbol: str) -> dict[str, Any]:
            if order_policies is None or symbol not in order_policies:
                return {"simulation": True, "positions": [], "recent_orders": []}
            pol_state = order_policies[symbol].snapshot_state()
            open_symbol = pol_state.get("open_symbol")
            positions = []
            if open_symbol:
                positions.append(
                    {
                        "symbol": open_symbol,
                        "side": "long",
                        "qty": pol_state.get("qty"),
                        "avg_entry_price": None,
                        "market_value": None,
                        "unrealized_pl": None,
                    }
                )
            out = {
                "underlying": symbol,
                "ok": True,
                "simulation": True,
                "positions": positions,
                "recent_orders": list(simulated_orders.get(symbol, [])),
            }
            self._store.set_broker_state(symbol, out)
            self._emit("broker_state", {"symbol": symbol, "state": out})
            return out

        def _record_sim_orders(symbol: str, result: dict[str, Any], ts_iso: str | None) -> None:
            if symbol not in simulated_orders:
                return
            for key in ("open_order", "close_order", "flip_close_order"):
                item = result.get(key)
                if not isinstance(item, dict):
                    continue
                payload = item.get("payload") if isinstance(item.get("payload"), dict) else None
                if not payload:
                    continue
                simulated_orders[symbol].appendleft(
                    {
                        "id": None,
                        "symbol": payload.get("symbol"),
                        "side": payload.get("side"),
                        "status": "simulated",
                        "qty": str(payload.get("qty")) if payload.get("qty") is not None else None,
                        "filled_qty": str(payload.get("qty")) if payload.get("qty") is not None else None,
                        "filled_avg_price": None,
                        "submitted_at": ts_iso,
                        "filled_at": ts_iso,
                    }
                )

        def _on_1m(symbol: str, bar: dict, buffer: Any) -> None:
            size = len(buffer) if buffer is not None else None
            self._store.add_1m_bar(symbol, bar, buffer_size=size)
            self._emit(
                "bar_1m",
                {
                    "symbol": symbol,
                    "bar": _normalize_bar(bar),
                    "buffer_size": size,
                },
            )

        def _on_15m(symbol: str, bar15: dict, buffer: Any) -> None:
            if order_policies is not None and symbol in order_policies:
                order_policies[symbol].on_15m_bar(closed_bar=bar15)
                self._store.set_policy_state(symbol, order_policies[symbol].snapshot_state())

            if inference is None:
                self._store.add_15m_bar(symbol, bar15)
                self._emit("bar_15m", {"symbol": symbol, "bar": _normalize_bar(bar15)})
                return
            action = inference.on_15m_close(df_1m=buffer.to_dataframe(), closed_bar=bar15)
            if action is None:
                self._store.add_15m_bar(symbol, bar15)
                self._emit("bar_15m", {"symbol": symbol, "bar": _normalize_bar(bar15)})
                return

            action_int = int(action)
            prev_action = last_action_by_symbol.get(symbol)
            probs = agent.last_probs() if agent is not None else None
            bar_payload = dict(bar15)
            if probs:
                bar_payload.update({k: v for k, v in probs.items() if v is not None})
            self._store.add_15m_bar(symbol, bar_payload)
            self._emit("bar_15m", {"symbol": symbol, "bar": _normalize_bar(bar_payload)})
            self._store.set_last_action(
                symbol,
                action=action_int,
                ts=bar15.get("timestamp"),
                close=bar15.get("close"),
            )
            last_action_by_symbol[symbol] = action_int
            agent_state = agent.snapshot_state() if agent is not None else None
            if agent_state is not None:
                self._store.set_agent_state(agent_state)
            self._emit(
                "action",
                {
                    "symbol": symbol,
                    "action": action_int,
                    "timestamp": _ts_iso(bar15.get("timestamp")),
                    "close": _coerce_float(bar15.get("close"), float("nan")),
                    "agent_state": agent_state,
                },
            )

            if order_policies is None or symbol not in order_policies:
                event_key = _agent_action_event(prev_action, action_int)
                if event_key is not None:
                    trade_payload = {
                        "symbol": symbol,
                        "timestamp": _ts_iso(bar15.get("timestamp")),
                        "result": {
                            "event": event_key,
                            "source": "agent_action",
                            "prev_action": prev_action,
                            "action": action_int,
                            "simulated": True,
                        },
                    }
                    self._store.add_trade_event(trade_payload)
                    self._emit("trade_event", trade_payload)
                return

            policy = order_policies[symbol]
            result = policy.on_decision(
                action=action_int,
                closed_bar=bar15,
                update_bar_state=False,
                logger=self._policy_logger(symbol),
            )
            ts_iso = _ts_iso(bar15.get("timestamp"))
            _record_sim_orders(symbol, result, ts_iso)
            policy_state = policy.snapshot_state()
            self._store.set_policy_state(symbol, policy_state)
            broker_state = _set_sim_broker_state(symbol)
            event_payload = {
                "symbol": symbol,
                "timestamp": ts_iso,
                "result": result,
                "policy_state": policy_state,
                "broker_state": broker_state,
            }
            self._emit("order_policy", event_payload)
            if str(result.get("event", "")).strip().lower() not in {"hold", "no_change"}:
                self._store.add_trade_event(event_payload)

        processor = ReplayBarProcessor(
            interval_minutes=cfg.interval,
            buffer_size=cfg.buffer_size,
            agg_label=cfg.resample_label,
            on_1m=_on_1m,
            on_15m_close=_on_15m,
        )

        for symbol in symbols:
            if order_policies is not None and symbol in order_policies:
                _set_sim_broker_state(symbol)

        total_rows = len(df)
        self._emit_status(running=True, message=f"replay started ({total_rows:,} bars queued)")
        self._emit_state_sync()
        count = 0
        for row in df.itertuples(index=False):
            if stop_event.is_set():
                break
            bar = {
                "symbol": getattr(row, "symbol"),
                "timestamp": getattr(row, "timestamp"),
                "open": float(getattr(row, "open")),
                "high": float(getattr(row, "high")),
                "low": float(getattr(row, "low")),
                "close": float(getattr(row, "close")),
                "volume": float(getattr(row, "volume")),
            }
            processor.handle_bar(bar)
            count += 1
            if count % 500 == 0:
                self._emit_status(running=True, message=f"replay processing {count:,}/{total_rows:,} bars")
            if cfg.replay_max_bars is not None and count >= int(cfg.replay_max_bars):
                break
            if cfg.replay_sleep > 0.0 and stop_event.wait(timeout=float(cfg.replay_sleep)):
                break

        self._emit(
            "log",
            {
                "symbol": "SYSTEM",
                "message": f"[replay] done. bars processed: {count:,}.",
            },
        )
        self._emit_status(running=True, message=f"replay complete ({count:,} bars)")
        self._emit_state_sync()

    def _run(self) -> None:
        error_message: str | None = None
        cfg = self._config
        stop_event = self._stop_event
        if cfg is None or stop_event is None:
            return

        try:
            mode = str(cfg.runner_mode or "live").strip().lower()
            if mode == "replay":
                self._run_replay(cfg=cfg, stop_event=stop_event)
                return
            if mode != "live":
                raise ValueError(f"Unknown runner_mode: {cfg.runner_mode}")

            from API.Alpaca_API.inference.live_inference import LiveInferenceEngine, LivePPOAgent
            from API.Alpaca_API.market_data.live_stream import AlpacaBarStreamer
            from API.Alpaca_API.runners import live_runner as lr
            from Policy.order_policy import OptionOrderPolicy, OptionOrderPolicyConfig

            symbols = cfg.symbols
            last_action_by_symbol: dict[str, int] = {}
            feed = lr._parse_feed(cfg.feed)
            bar_queue: queue_mod.Queue = queue_mod.Queue(maxsize=cfg.queue_size)

            agent: LivePPOAgent | None = None
            if not cfg.no_agent:
                ga_feature_list = self._resolve_ga_feature_list(cfg)
                agent = LivePPOAgent(
                    model_path=cfg.model_path,
                    deterministic=not cfg.stochastic,
                    device=cfg.device,
                    include_pivot_probs=not cfg.no_pivot_probs,
                    include_tb_probs=not cfg.no_tb_probs,
                    tz=cfg.tz or "America/New_York",
                    assume_tz=cfg.assume_tz,
                    session_open=cfg.session_open,
                    session_close=cfg.session_close,
                    min_15m_bars=cfg.min_15m_bars,
                    fill_missing_prob=cfg.fill_missing_prob,
                    ga_model_root=cfg.ga_model_root if ga_feature_list else None,
                    ga_feature_list_path=ga_feature_list,
                    ga_pivot_label_dir=cfg.ga_pivot_label_dir,
                    ga_tb_label_dir=cfg.ga_tb_label_dir,
                    resample_label=cfg.resample_label,
                    resample_closed=cfg.resample_closed,
                    label_timeframe_rule=f"{cfg.interval}min",
                )
                self._store.set_agent_state(agent.snapshot_state())
                self._emit("agent_state", {"state": agent.snapshot_state()})

            inference = LiveInferenceEngine(
                agent=agent,
                label=cfg.resample_label,
                closed=cfg.resample_closed,
                rule=f"{cfg.interval}min",
                tz=cfg.tz,
                assume_tz=cfg.assume_tz,
            )

            order_policies: dict[str, OptionOrderPolicy] | None = None
            if cfg.enable_option_orders:
                order_policies = {}
                for symbol in symbols:
                    pol_cfg = OptionOrderPolicyConfig(
                        underlying=symbol,
                        env_file=cfg.env_file,
                        tz_name=cfg.tz or "America/New_York",
                        atr_multiplier=float(cfg.option_atr_mult),
                        dte_cutoff_hhmm=cfg.option_dte_cutoff,
                        qty=int(cfg.option_order_qty),
                        close_on_flat=not cfg.option_no_close_on_flat,
                        close_on_flip=not cfg.option_no_close_on_flip,
                        submit_orders=not cfg.simulate_orders,
                    )
                    policy = OptionOrderPolicy(pol_cfg)
                    order_policies[symbol] = policy
                    self._store.set_policy_state(symbol, policy.snapshot_state())
                    broker_snap = policy.snapshot_broker_state(orders_limit=20)
                    self._store.set_broker_state(symbol, broker_snap)
                    self._emit("broker_state", {"symbol": symbol, "state": broker_snap})

                if not cfg.no_startup_sync:
                    for symbol in symbols:
                        policy = order_policies[symbol]
                        sync_result = policy.sync_from_broker(logger=self._policy_logger(symbol))
                        self._store.set_policy_state(symbol, policy.snapshot_state())
                        self._emit("startup_sync", {"symbol": symbol, "result": sync_result})
                        broker_snap = policy.snapshot_broker_state(orders_limit=20)
                        self._store.set_broker_state(symbol, broker_snap)
                        self._emit("broker_state", {"symbol": symbol, "state": broker_snap})

            def _on_1m(symbol: str, bar: dict, buffer: Any) -> None:
                size = len(buffer) if buffer is not None else None
                self._store.add_1m_bar(symbol, bar, buffer_size=size)
                self._emit(
                    "bar_1m",
                    {
                        "symbol": symbol,
                        "bar": _normalize_bar(bar),
                        "buffer_size": size,
                    },
                )

            def _on_15m(symbol: str, bar15: dict, buffer: Any) -> None:
                if order_policies is not None and symbol in order_policies:
                    policy = order_policies[symbol]
                    policy.on_15m_bar(closed_bar=bar15)
                    self._store.set_policy_state(symbol, policy.snapshot_state())

                action = inference.on_15m_close(df_1m=buffer.to_dataframe(), closed_bar=bar15)
                if action is None:
                    self._store.add_15m_bar(symbol, bar15)
                    self._emit("bar_15m", {"symbol": symbol, "bar": _normalize_bar(bar15)})
                    return

                action_int = int(action)
                prev_action = last_action_by_symbol.get(symbol)
                probs = agent.last_probs() if agent is not None else None
                bar_payload = dict(bar15)
                if probs:
                    bar_payload.update({k: v for k, v in probs.items() if v is not None})
                self._store.add_15m_bar(symbol, bar_payload)
                self._emit("bar_15m", {"symbol": symbol, "bar": _normalize_bar(bar_payload)})
                self._store.set_last_action(
                    symbol,
                    action=action_int,
                    ts=bar15.get("timestamp"),
                    close=bar15.get("close"),
                )
                last_action_by_symbol[symbol] = action_int

                agent_state = agent.snapshot_state() if agent is not None else None
                if agent_state is not None:
                    self._store.set_agent_state(agent_state)
                self._emit(
                    "action",
                    {
                        "symbol": symbol,
                        "action": action_int,
                        "timestamp": _ts_iso(bar15.get("timestamp")),
                        "close": _coerce_float(bar15.get("close"), float("nan")),
                        "agent_state": agent_state,
                    },
                )

                if order_policies is None or symbol not in order_policies:
                    event_key = _agent_action_event(prev_action, action_int)
                    if event_key is not None:
                        trade_payload = {
                            "symbol": symbol,
                            "timestamp": _ts_iso(bar15.get("timestamp")),
                            "result": {
                                "event": event_key,
                                "source": "agent_action",
                                "prev_action": prev_action,
                                "action": action_int,
                                "simulated": True,
                            },
                        }
                        self._store.add_trade_event(trade_payload)
                        self._emit("trade_event", trade_payload)
                    return

                policy = order_policies[symbol]
                result = policy.on_decision(
                    action=action_int,
                    closed_bar=bar15,
                    update_bar_state=False,
                    logger=self._policy_logger(symbol),
                )
                policy_state = policy.snapshot_state()
                self._store.set_policy_state(symbol, policy_state)
                broker_snap = policy.snapshot_broker_state(orders_limit=20)
                self._store.set_broker_state(symbol, broker_snap)
                event_payload = {
                    "symbol": symbol,
                    "timestamp": _ts_iso(bar15.get("timestamp")),
                    "result": result,
                    "policy_state": policy_state,
                    "broker_state": broker_snap,
                }
                self._emit("order_policy", event_payload)
                self._emit("broker_state", {"symbol": symbol, "state": broker_snap})
                event_key = str(result.get("event", "")).strip().lower()
                if event_key not in {"hold", "no_change"}:
                    self._store.add_trade_event(event_payload)

            processor = lr.LiveBarProcessor(
                interval_minutes=cfg.interval,
                buffer_size=cfg.buffer_size,
                agg_label=cfg.resample_label,
                on_1m=_on_1m,
                on_15m_close=_on_15m,
            )

            if cfg.prefill_tail and cfg.prefill_tail > cfg.buffer_size:
                self._emit(
                    "log",
                    {
                        "symbol": "SYSTEM",
                        "message": "[live] prefill-tail exceeds buffer-size; oldest rows will be dropped.",
                    },
                )
            if cfg.prefill_path:
                prefill_df = lr._load_prefill_frame(Path(cfg.prefill_path))
                lr._prefill_buffers(
                    processor=processor,
                    df=prefill_df,
                    symbols=symbols,
                    tail=cfg.prefill_tail,
                )
            elif cfg.prefill_start and not cfg.no_prefill_fetch:
                lr._prefill_from_alpaca(
                    processor=processor,
                    symbols=symbols,
                    start=cfg.prefill_start,
                    tail=cfg.prefill_tail,
                    split_dataset_name=cfg.ga_dataset_name,
                    split_x_filename=cfg.split_x_filename,
                    prepend_split_test_warmup=True,
                )

            self._store.seed_from_buffers(processor._buffers)
            seed_counts = {sym: len(buf) for sym, buf in processor._buffers.items()}
            self._emit(
                "log",
                {
                    "symbol": "SYSTEM",
                    "message": f"[live] prefill loaded bars: {seed_counts}",
                },
            )
            self._seed_15m_from_prefill(cfg=cfg, processor=processor, agent=agent)
            self._emit_state_sync()

            if order_policies:
                latest_closed = lr._warmup_order_policies_from_prefill(
                    processor=processor,
                    order_policies=order_policies,
                    symbols=symbols,
                )
                for symbol in symbols:
                    self._store.set_policy_state(symbol, order_policies[symbol].snapshot_state())
                if cfg.startup_catchup_order:
                    lr._run_startup_catchup_decision(
                        processor=processor,
                        inference=inference,
                        order_policies=order_policies,
                        latest_closed_by_symbol=latest_closed,
                        symbols=symbols,
                        print_tz=cfg.tz,
                        max_age_min=int(cfg.startup_catchup_max_age_min),
                    )
                    for symbol in symbols:
                        self._store.set_policy_state(symbol, order_policies[symbol].snapshot_state())

            streamer = AlpacaBarStreamer(
                symbols=symbols,
                feed=feed,
                env_file=cfg.env_file,
                queue=bar_queue,
            )
            with self._lock:
                self._streamer = streamer

            streamer.start_in_thread()
            self._emit_status(running=True, message="live stream started (waiting for real-time bars)")
            self._emit(
                "log",
                {
                    "symbol": "SYSTEM",
                    "message": "[live] stream connected; waiting for fresh market bars.",
                },
            )

            last_broker_poll = 0.0
            while not stop_event.is_set():
                # Keep broker-side positions/orders fresh in UI between trade events.
                if order_policies is not None:
                    now = time.monotonic()
                    if now - last_broker_poll >= 30.0:
                        last_broker_poll = now
                        for symbol, policy in order_policies.items():
                            broker_snap = policy.snapshot_broker_state(orders_limit=20)
                            self._store.set_broker_state(symbol, broker_snap)
                            self._emit("broker_state", {"symbol": symbol, "state": broker_snap})
                try:
                    bar = bar_queue.get(timeout=0.5)
                except queue_mod.Empty:
                    continue
                processor.handle_bar(bar)

        except Exception as exc:
            error_message = f"{exc}"
            self._store.add_log_event(
                {
                    "symbol": "SYSTEM",
                    "message": f"{exc}",
                    "traceback": traceback.format_exc(limit=12),
                }
            )
            self._emit_status(
                running=False,
                message=f"error: {exc}",
                extra={
                    "error": f"{exc}",
                    "traceback": traceback.format_exc(limit=12),
                },
            )
        finally:
            with self._lock:
                streamer = self._streamer
                self._streamer = None
            if streamer is not None:
                try:
                    streamer.stop()
                except Exception:
                    pass
                try:
                    streamer.join(timeout=5.0)
                except Exception:
                    pass
            self._store.stop(error=error_message)
            self._emit_status(running=False, message="stopped", extra={"error": error_message})
            with self._lock:
                self._thread = None


class DashboardApp:
    def __init__(self) -> None:
        self.store = DashboardStore(max_bars=900, max_events=400)
        self.broker = EventBroker()
        self.session = LiveSession(self.store, self.broker)

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        cfg = SessionConfig.from_payload(payload)
        if self.session.is_running():
            self.session.stop()
            if self.session.is_running():
                raise RuntimeError("Existing session is still running; stop it first.")
        self.session.start(cfg)
        return self.store.snapshot()

    def stop(self) -> dict[str, Any]:
        self.session.stop()
        return self.store.snapshot()

    def snapshot(self) -> dict[str, Any]:
        snap = self.store.snapshot()
        snap["session_alive"] = self.session.is_running()
        return snap


class DashboardHTTPServer(ThreadingHTTPServer):
    app: DashboardApp


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "CynolycusLiveDashboard/1.0"

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _write_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(_json_safe(payload)).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _write_text(self, body: str, *, status: int = HTTPStatus.OK, content_type: str = "text/plain") -> None:
        blob = body.encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _serve_index(self) -> None:
        root = Path(__file__).resolve().parent
        index_path = root / "index.html"
        if not index_path.exists():
            self._write_text("Missing UI/index.html", status=HTTPStatus.NOT_FOUND)
            return
        html = index_path.read_text(encoding="utf-8")
        self._write_text(html, status=HTTPStatus.OK, content_type="text/html")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        app = self._app()

        if parsed.path in {"/", "/index.html"}:
            self._serve_index()
            return
        if parsed.path == "/api/state":
            self._write_json(app.snapshot())
            return
        if parsed.path == "/api/events":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            q = app.broker.subscribe()
            try:
                hello = {"type": "hello", "timestamp": _ts_iso(datetime.utcnow()), "payload": app.snapshot()}
                self.wfile.write(f"data: {json.dumps(_json_safe(hello))}\n\n".encode("utf-8"))
                self.wfile.flush()
                while True:
                    try:
                        event = q.get(timeout=12.0)
                    except queue_mod.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    data = json.dumps(_json_safe(event))
                    self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                app.broker.unsubscribe(q)
            return

        self._write_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        app = self._app()
        try:
            payload = self._read_json_body()
        except Exception as exc:
            self._write_json({"error": f"invalid_json: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            if parsed.path == "/api/start":
                state = app.start(payload)
                self._write_json({"ok": True, "state": state})
                return
            if parsed.path == "/api/stop":
                state = app.stop()
                self._write_json({"ok": True, "state": state})
                return
        except Exception as exc:
            self._write_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._write_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def _app(self) -> DashboardApp:
        server = self.server
        if not isinstance(server, DashboardHTTPServer):
            raise RuntimeError("Handler attached to unexpected server type.")
        return server.app


def run_server(*, host: str, port: int) -> None:
    app = DashboardApp()
    server = DashboardHTTPServer((host, int(port)), DashboardHandler)
    server.daemon_threads = True
    server.app = app

    print(f"[ui] Live dashboard: http://{host}:{port}")
    print("[ui] Use Start in the browser to launch live_runner session.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[ui] Shutting down...")
    finally:
        try:
            app.stop()
        except Exception:
            pass
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local web dashboard for live_runner.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port.")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
