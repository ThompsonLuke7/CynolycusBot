from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue as queue_mod
import re
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TYPE_CHECKING, Callable
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from API.Alpaca_API.market_data.bar_aggregator import OhlcvAggregator
from API.Alpaca_API.market_data.bar_buffer import BarRingBuffer
from Policy.execution_latch import DirectionExecutionLatch
from Policy.order_policy import PHASE4_SWING_SETUP_BODYCLOSE_BODYCLOSE_V1
from Policy.replay_option_proxy import ReplayOptionPriceProxy

UI_BUILD = "2026-04-15-sim-replay-offline"
DEFAULT_SPY_1M_PATH = "Data/raw/spy/1m_train.parquet"

if TYPE_CHECKING:
    from API.Alpaca_API.inference.live_inference import (
        LiveIndependentMetaXGBAgent,
        LiveInferenceEngine,
        LiveMetaXGBAgent,
        LivePPOAgent,
    )
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


def _coerce_optional_float(raw: Any) -> float | None:
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


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
        if not isinstance(key, str) or (not key.startswith("p_") and not key.startswith("thr_")):
            continue
        num = _coerce_float(val, float("nan"))
        out[key] = num if np.isfinite(num) else None
    return out


def _default_runtime_policy_cache_path(prefill_path: Path) -> Path:
    suffix = prefill_path.suffix or ".json"
    if suffix.lower() != ".json":
        suffix = ".json"
    return prefill_path.with_name(f"{prefill_path.stem}_runtime_policy_state{suffix}")


def _load_runtime_policy_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _persist_runtime_policy_cache(cache_path: Path, payload: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _default_replay_snapshot_cache_dir() -> Path:
    return Path("Data") / "inference" / "dashboard_cache"


def _default_live_audit_root() -> Path:
    return Path("Data") / "inference" / "live_runs"


def _slugify_token(value: Any, *, default: str = "na") -> str:
    raw = str(value or "").strip().lower()
    raw = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    return raw or default


def _path_fingerprint(raw_path: str | None) -> dict[str, Any]:
    if not raw_path:
        return {"path": None, "exists": False}
    path = Path(raw_path)
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "exists": True,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except Exception:
        return {"path": str(path), "exists": False}


def _replay_snapshot_signature(cfg: "SessionConfig") -> dict[str, Any]:
    semantic_cfg = {
        "runner_mode": str(cfg.runner_mode),
        "symbols": list(cfg.symbols),
        "inference_mode": str(cfg.inference_mode),
        "interval": int(cfg.interval),
        "resample_label": str(cfg.resample_label),
        "resample_closed": str(cfg.resample_closed),
        "tz": str(cfg.tz),
        "assume_tz": str(cfg.assume_tz),
        "meta_model_root": str(cfg.meta_model_root),
        "meta_entry_prob_source": str(cfg.meta_entry_prob_source),
        "swing_setup_single_model_dir": str(cfg.swing_setup_single_model_dir),
        "meta_base_frame_append_lookback_days": int(cfg.meta_base_frame_append_lookback_days),
        "no_agent": bool(cfg.no_agent),
        "no_pivot_probs": bool(cfg.no_pivot_probs),
        "no_tb_probs": bool(cfg.no_tb_probs),
        "fill_missing_prob": float(cfg.fill_missing_prob),
        "session_open": str(cfg.session_open),
        "session_close": str(cfg.session_close),
        "ga_model_root": str(cfg.ga_model_root),
        "ga_feature_list": cfg.ga_feature_list,
        "ga_dataset_name": str(cfg.ga_dataset_name),
        "split_x_filename": str(cfg.split_x_filename),
        "ga_pivot_label_dir": str(cfg.ga_pivot_label_dir),
        "ga_tb_label_dir": str(cfg.ga_tb_label_dir),
        "meta_entry_threshold": cfg.meta_entry_threshold,
        "meta_exit_threshold": cfg.meta_exit_threshold,
        "meta_intrabar_entry_policy": str(cfg.meta_intrabar_entry_policy),
        "meta_intrabar_setup_max_bars": int(cfg.meta_intrabar_setup_max_bars),
        "meta_intrabar_long_setup_threshold": cfg.meta_intrabar_long_setup_threshold,
        "meta_intrabar_short_setup_threshold": cfg.meta_intrabar_short_setup_threshold,
        "meta_hard_stop_atr": float(cfg.meta_hard_stop_atr),
        "meta_trail_activate_atr": float(cfg.meta_trail_activate_atr),
        "meta_trail_atr": float(cfg.meta_trail_atr),
        "meta_trail_atr_after_tp": float(cfg.meta_trail_atr_after_tp),
        "meta_use_tp_to_tighten_trail": bool(cfg.meta_use_tp_to_tighten_trail),
        "enable_option_orders": bool(cfg.enable_option_orders),
        "simulate_orders": bool(cfg.simulate_orders),
        "option_order_qty": int(cfg.option_order_qty),
        "option_atr_mult": float(cfg.option_atr_mult),
        "option_dte_cutoff": str(cfg.option_dte_cutoff),
        "option_exit_policy": str(cfg.option_exit_policy),
        "option_exit_take_profit_pct": float(cfg.option_exit_take_profit_pct),
        "option_exit_stop_loss_pct": float(cfg.option_exit_stop_loss_pct),
        "option_exit_profit_lock_arm_pct": float(cfg.option_exit_profit_lock_arm_pct),
        "option_exit_profit_lock_floor_pct": float(cfg.option_exit_profit_lock_floor_pct),
        "option_exit_trailing_arm_pct": float(cfg.option_exit_trailing_arm_pct),
        "option_exit_trailing_giveback_pct": float(cfg.option_exit_trailing_giveback_pct),
        "option_exit_time_decay_minutes": int(cfg.option_exit_time_decay_minutes),
        "option_exit_time_decay_progress_pct": float(cfg.option_exit_time_decay_progress_pct),
        "option_exit_opposite_prob": float(cfg.option_exit_opposite_prob),
        "option_exit_quote_mode": str(cfg.option_exit_quote_mode),
        "replay_option_proxy_mode": str(cfg.replay_option_proxy_mode),
        "replay_option_proxy_expiry_hhmm": str(cfg.replay_option_proxy_expiry_hhmm),
        "replay_option_proxy_iv_floor": float(cfg.replay_option_proxy_iv_floor),
        "replay_option_proxy_iv_ceiling": float(cfg.replay_option_proxy_iv_ceiling),
        "replay_option_proxy_iv_multiplier": float(cfg.replay_option_proxy_iv_multiplier),
        "replay_option_proxy_min_dte_minutes": float(cfg.replay_option_proxy_min_dte_minutes),
        "option_no_close_on_flat": bool(cfg.option_no_close_on_flat),
        "option_no_close_on_flip": bool(cfg.option_no_close_on_flip),
        "startup_catchup_order": bool(cfg.startup_catchup_order),
        "startup_catchup_max_age_min": int(cfg.startup_catchup_max_age_min),
        "exec_entry_confirm_bars": int(cfg.exec_entry_confirm_bars),
        "exec_exit_confirm_bars": int(cfg.exec_exit_confirm_bars),
        "exec_flip_abs_threshold": float(cfg.exec_flip_abs_threshold),
        "use_execution_latch": bool(cfg.use_execution_latch),
        "replay_start": cfg.replay_start,
        "replay_end": cfg.replay_end,
        "replay_regular_only": bool(cfg.replay_regular_only),
        "replay_max_bars": cfg.replay_max_bars,
        "replay_warmup_bars": int(cfg.replay_warmup_bars),
        "replay_no_prepend_split_test_warmup": bool(cfg.replay_no_prepend_split_test_warmup),
        "eval_parity_mode": bool(cfg.eval_parity_mode),
    }
    return {
        "ui_build": UI_BUILD,
        "semantic_cfg": semantic_cfg,
        "replay_data": _path_fingerprint(cfg.replay_data_path),
        "meta_base_frame": _path_fingerprint(cfg.meta_base_frame_path),
    }


def _default_replay_snapshot_cache_path(cfg: "SessionConfig") -> Path:
    sig = _replay_snapshot_signature(cfg)
    digest = hashlib.sha256(
        json.dumps(_json_safe(sig), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return _default_replay_snapshot_cache_dir() / f"replay_snapshot_{digest}.json"


def _load_replay_snapshot_cache(cache_path: Path, *, cfg: "SessionConfig") -> dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    expected = _replay_snapshot_signature(cfg)
    if payload.get("signature") != _json_safe(expected):
        return {}
    state = payload.get("state")
    if not isinstance(state, dict):
        return {}
    return payload


def _persist_replay_snapshot_cache(cache_path: Path, *, cfg: "SessionConfig", state: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": _ts_iso(datetime.now(timezone.utc)),
        "signature": _replay_snapshot_signature(cfg),
        "state": _json_safe(state),
    }
    cache_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _action_to_position(action: float | int, *, deadband: float = 0.0) -> int:
    a = _coerce_float(action, float("nan"))
    if not np.isfinite(a):
        return 0
    # Backward compatibility with legacy discrete actions {0,1,2}.
    if abs(a - 2.0) < 1e-9:
        return -1
    if abs(a - 1.0) < 1e-9:
        return 1
    if abs(a) <= max(0.0, float(deadband)):
        return 0
    return 1 if a > 0.0 else -1


def _agent_action_event(prev_action: float | int | None, action: float | int) -> str | None:
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


def _apply_flip_abs_threshold(
    *,
    action_raw: float,
    prev_exec_action: float | int | None,
    flip_abs_threshold: float,
) -> tuple[float, int, int, bool]:
    """
    Gate weak opposite-direction flips based on raw action magnitude.

    Returns:
      - gated_action_raw: possibly sign-adjusted raw action
      - model_raw_pos: sign class from unmodified model action
      - exec_raw_pos: sign class after threshold gate
      - flip_blocked: whether threshold blocked a flip
    """
    threshold = max(0.0, float(flip_abs_threshold))
    model_raw_pos = _action_to_position(action_raw, deadband=threshold)
    if threshold <= 0.0 or prev_exec_action is None:
        return action_raw, model_raw_pos, model_raw_pos, False

    prev_pos = _action_to_position(prev_exec_action, deadband=threshold)
    if prev_pos == 0 or model_raw_pos == 0 or model_raw_pos == prev_pos:
        return action_raw, model_raw_pos, model_raw_pos, False

    if (not np.isfinite(action_raw)) or abs(float(action_raw)) < threshold:
        gated = float(np.sign(prev_pos) * abs(action_raw)) if np.isfinite(action_raw) else float(prev_pos)
        return gated, model_raw_pos, prev_pos, True

    return action_raw, model_raw_pos, model_raw_pos, False


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
        prob_cols = ["p_pivot_long", "p_pivot_short", "p_tb_long", "p_tb_short"]
        candidates = [
            Path("Data") / "inference" / clean / dataset_name / "agent" / "agent_matrix.parquet",
            Path("Data") / "inference" / clean / dataset_name / "agent" / "agent_matrix.csv",
            Path("Data") / "models" / "agent" / dataset_name / clean / "agent_matrix.parquet",
            Path("Data") / "models" / "agent" / dataset_name / clean / "agent_matrix.csv",
        ]

        def _series_from_probs_parquet(path: Path, value_col: str) -> pd.Series | None:
            if not path.exists():
                return None
            try:
                df = pd.read_parquet(path)
            except Exception:
                return None
            if value_col not in df.columns:
                return None
            if "timestamp" in df.columns:
                ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            else:
                ts = pd.to_datetime(df.index, utc=True, errors="coerce")
            out = pd.Series(pd.to_numeric(df[value_col], errors="coerce").to_numpy(), index=ts)
            out = out[out.index.notna()]
            if out.empty:
                return None
            return out.sort_index()

        frames: list[pd.DataFrame] = []
        loaded_paths: list[str] = []

        # Backfill source: GA-XGB full-series probabilities usually cover the full
        # train window and can provide early replay timestamps before agent_matrix starts.
        ga_roots = [
            Path("Data") / "models" / "ga_xgboost" / dataset_name,
            Path("Data") / "inference" / clean / dataset_name / "models" / "ga_xgboost" / dataset_name,
        ]
        ga_specs = [
            ("p_pivot_long", "long", "swing", "p_long_probs.parquet", "p_long_oof_train", "p_long_test", "p_long_full"),
            ("p_pivot_short", "short", "swing", "p_short_probs.parquet", "p_short_oof_train", "p_short_test", "p_short_full"),
            ("p_tb_long", "long", "tb", "p_long_probs.parquet", "p_long_oof_train", "p_long_test", "p_long_full"),
            ("p_tb_short", "short", "tb", "p_short_probs.parquet", "p_short_oof_train", "p_short_test", "p_short_full"),
        ]
        ga_cols: dict[str, pd.Series] = {}
        for root in ga_roots:
            found_any = False
            for out_col, side, label_dir, fname, oof_col, test_col, full_col in ga_specs:
                candidate_paths = [
                    root / side / label_dir / fname,
                    root / side / "probs" / label_dir / fname,
                    root / side / fname,
                ]
                series = None
                for candidate_path in candidate_paths:
                    if not candidate_path.exists():
                        continue
                    try:
                        df = pd.read_parquet(candidate_path)
                    except Exception:
                        continue
                    picked = None
                    if oof_col in df.columns and test_col in df.columns:
                        picked = pd.to_numeric(df[oof_col], errors="coerce").combine_first(
                            pd.to_numeric(df[test_col], errors="coerce")
                        )
                        if full_col in df.columns:
                            picked = picked.combine_first(pd.to_numeric(df[full_col], errors="coerce"))
                    elif full_col in df.columns:
                        picked = pd.to_numeric(df[full_col], errors="coerce")
                    if picked is None:
                        continue
                    if "timestamp" in df.columns:
                        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                    else:
                        ts = pd.to_datetime(df.index, utc=True, errors="coerce")
                    series = pd.Series(picked.to_numpy(), index=ts)
                    series = series[series.index.notna()].sort_index()
                    if not series.empty:
                        break
                if series is None or series.empty:
                    continue
                ga_cols[out_col] = series
                found_any = True
            if found_any:
                idx = None
                for s in ga_cols.values():
                    idx = s.index if idx is None else idx.union(s.index)
                if idx is not None and len(idx) > 0:
                    ga_df = pd.DataFrame(index=idx.sort_values())
                    for col in prob_cols:
                        ga_df[col] = ga_cols[col].reindex(ga_df.index) if col in ga_cols else np.nan
                    frames.append(ga_df)
                    loaded_paths.append(f"{root}/*/(swing|tb)/*_probs.parquet")
                break

        for path in candidates:
            if not path.exists():
                continue
            try:
                if path.suffix.lower() == ".parquet":
                    cols = pd.read_parquet(path, columns=None).columns.tolist()
                    keep = [c for c in ["timestamp", *prob_cols] if c in cols]
                    if "timestamp" not in keep:
                        continue
                    df = pd.read_parquet(path, columns=keep)
                else:
                    df = pd.read_csv(path, usecols=lambda c: c in {"timestamp", *prob_cols})
                    if "timestamp" not in df.columns:
                        continue
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
                have_any_prob = any(col in df.columns for col in prob_cols)
                if not have_any_prob:
                    continue
                for col in prob_cols:
                    if col not in df.columns:
                        df[col] = np.nan
                frames.append(df[prob_cols])
                loaded_paths.append(str(path))
            except Exception:
                continue

        if not frames:
            return None

        merged = pd.concat(frames, axis=0).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        merged.attrs["source_paths"] = loaded_paths
        return merged
    except Exception:
        return None


def _load_swing_setup_probs_frame(
    *,
    model_dir: str | Path,
    tz: str | None = "America/New_York",
) -> pd.DataFrame | None:
    path = Path(model_dir) / "p_swing_probs.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            ts_col = None
            for candidate in ("timestamp", "date", "datetime", "index"):
                if candidate in df.columns:
                    ts_col = candidate
                    break
            if ts_col is None:
                return None
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            df = df.dropna(subset=[ts_col]).set_index(ts_col)
        else:
            idx = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df.loc[pd.notna(idx)].copy()
            df.index = pd.DatetimeIndex(idx[pd.notna(idx)])
        if df.empty:
            return None
        if tz:
            df.index = pd.DatetimeIndex(df.index).tz_convert(tz)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
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
    runner_mode: str = "replay"
    inference_mode: str = "meta"
    feed: str = "IEX"
    interval: int = 10
    buffer_size: int = 0
    queue_size: int = 5000
    resample_label: str = "left"
    resample_closed: str = "left"
    tz: str = "America/New_York"
    assume_tz: str = "UTC"
    model_path: str = "Data/outputs/agent/ppo_model.pt"
    meta_model_root: str = "Data/models/meta_xgboost/10min"
    meta_entry_prob_source: str = "swing_support_single"
    swing_setup_single_model_dir: str = "Data/models/ga_xgboost/10min/single/swing_support_single"
    meta_base_frame_path: str = "Data/inference/spy/10min/debug_matrices_warmup/spy/live_meta_matrix_on_trace_ts_live_2026_03_27.parquet"
    meta_base_frame_append_lookback_days: int = 120
    no_agent: bool = False
    stochastic: bool = False
    device: str = "auto"
    min_15m_bars: int = 20
    no_pivot_probs: bool = False
    no_tb_probs: bool = False
    fill_missing_prob: float = 0.0
    session_open: str = "09:30"
    session_close: str = "16:00"
    ga_model_root: str = "Data/models/ga_xgboost/10min"
    ga_feature_list: str | None = None
    ga_dataset_name: str = "10min"
    split_x_filename: str = "X_10min_tree.parquet"
    ga_pivot_label_dir: str = "swing"
    ga_tb_label_dir: str = "tb"
    meta_entry_threshold: float | None = None
    meta_exit_threshold: float | None = None
    meta_intrabar_entry_policy: str = PHASE4_SWING_SETUP_BODYCLOSE_BODYCLOSE_V1
    meta_intrabar_setup_max_bars: int = 4
    meta_intrabar_long_setup_threshold: float | None = 0.42
    meta_intrabar_short_setup_threshold: float | None = 0.15
    meta_hard_stop_atr: float = 0.0
    meta_trail_activate_atr: float = 2.0
    meta_trail_atr: float = 1.0
    meta_trail_atr_after_tp: float = 0.8
    meta_use_tp_to_tighten_trail: bool = True
    env_file: str = ".env"
    prefill_path: str | None = DEFAULT_SPY_1M_PATH
    prefill_start: str = "2026-01-30"
    no_prefill_fetch: bool = False
    prefill_tail: int | None = None
    enable_option_orders: bool = True
    no_startup_sync: bool = False
    option_order_qty: int = 1
    option_atr_mult: float = 1.0
    option_dte_cutoff: str = "13:00"
    option_exit_policy: str = "option_adaptive_trail_v1"
    option_exit_take_profit_pct: float = 0.0
    option_exit_stop_loss_pct: float = 1.0
    option_exit_profit_lock_arm_pct: float = 2.0
    option_exit_profit_lock_floor_pct: float = 0.25
    option_exit_trailing_arm_pct: float = 2.0
    option_exit_trailing_giveback_pct: float = 0.25
    option_exit_time_decay_minutes: int = 80
    option_exit_time_decay_progress_pct: float = 1.0
    option_exit_opposite_prob: float = 0.60
    option_exit_quote_mode: str = "bid"
    replay_option_proxy_mode: str = "black_scholes"
    replay_option_proxy_expiry_hhmm: str = "15:40"
    replay_option_proxy_iv_floor: float = 0.12
    replay_option_proxy_iv_ceiling: float = 0.90
    replay_option_proxy_iv_multiplier: float = 1.50
    replay_option_proxy_min_dte_minutes: float = 1.0
    simulate_orders: bool = True
    option_no_close_on_flat: bool = False
    option_no_close_on_flip: bool = False
    startup_catchup_order: bool = False
    startup_catchup_max_age_min: int = 120
    exec_entry_confirm_bars: int = 1
    exec_exit_confirm_bars: int = 2
    exec_flip_abs_threshold: float = 0.05
    use_execution_latch: bool = False
    replay_data_path: str = DEFAULT_SPY_1M_PATH
    replay_start: str | None = "2026-03-14T00:00:00Z"
    replay_end: str | None = "2026-03-23T23:59:59Z"
    replay_regular_only: bool = False
    replay_sleep: float = 0.0
    replay_max_bars: int | None = None
    replay_warmup_bars: int = 5000
    replay_no_prepend_split_test_warmup: bool = False
    eval_parity_mode: bool = False
    audit_enabled: bool = True
    audit_root: str = "Data/inference/live_runs"

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SessionConfig":
        symbols_raw = str(payload.get("symbols", "SPY"))
        symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
        if not symbols:
            symbols = ["SPY"]
        return cls(
            symbols=symbols,
            runner_mode=str(payload.get("runner_mode", "replay")).strip().lower(),
            inference_mode=str(payload.get("inference_mode", "meta")).strip().lower(),
            feed=str(payload.get("feed", "IEX")).upper(),
            interval=max(1, _coerce_int(payload.get("interval"), 10)),
            buffer_size=max(0, _coerce_int(payload.get("buffer_size"), 0)),
            queue_size=max(100, _coerce_int(payload.get("queue_size"), 5000)),
            resample_label=str(payload.get("resample_label", "left")).strip().lower(),
            resample_closed=str(payload.get("resample_closed", "left")).strip().lower(),
            tz=str(payload.get("tz", "America/New_York")),
            assume_tz=str(payload.get("assume_tz", "UTC")),
            model_path=str(payload.get("model_path", "Data/outputs/agent/ppo_model.pt")),
            meta_model_root=str(payload.get("meta_model_root", "Data/models/meta_xgboost/10min")),
            meta_entry_prob_source=str(payload.get("meta_entry_prob_source", "swing_support_single")).strip().lower(),
            swing_setup_single_model_dir=str(
                payload.get(
                    "swing_setup_single_model_dir",
                    "Data/models/ga_xgboost/10min/single/swing_support_single",
                )
            ),
            meta_base_frame_path=str(
                payload.get(
                    "meta_base_frame_path",
                    "Data/inference/spy/10min/debug_matrices_warmup/spy/live_meta_matrix_on_trace_ts_live_2026_03_27.parquet",
                )
            ),
            meta_base_frame_append_lookback_days=max(
                1,
                _coerce_int(payload.get("meta_base_frame_append_lookback_days"), 120),
            ),
            no_agent=_coerce_bool(payload.get("no_agent"), False),
            stochastic=_coerce_bool(payload.get("stochastic"), False),
            device=str(payload.get("device", "auto")),
            min_15m_bars=max(5, _coerce_int(payload.get("min_15m_bars"), 20)),
            no_pivot_probs=_coerce_bool(payload.get("no_pivot_probs"), False),
            no_tb_probs=_coerce_bool(payload.get("no_tb_probs"), False),
            fill_missing_prob=_coerce_float(payload.get("fill_missing_prob"), 0.0),
            session_open=str(payload.get("session_open", "09:30")),
            session_close=str(payload.get("session_close", "16:00")),
            ga_model_root=str(payload.get("ga_model_root", "Data/models/ga_xgboost/10min")),
            ga_feature_list=payload.get("ga_feature_list"),
            ga_dataset_name=str(payload.get("ga_dataset_name", "10min")),
            split_x_filename=str(payload.get("split_x_filename", "X_10min_tree.parquet")),
            ga_pivot_label_dir=str(payload.get("ga_pivot_label_dir", "swing")),
            ga_tb_label_dir=str(payload.get("ga_tb_label_dir", "tb")),
            meta_entry_threshold=_coerce_optional_float(payload.get("meta_entry_threshold")),
            meta_exit_threshold=_coerce_optional_float(payload.get("meta_exit_threshold")),
            meta_intrabar_entry_policy=str(
                payload.get("meta_intrabar_entry_policy", PHASE4_SWING_SETUP_BODYCLOSE_BODYCLOSE_V1)
            ),
            meta_intrabar_setup_max_bars=max(1, _coerce_int(payload.get("meta_intrabar_setup_max_bars"), 4)),
            meta_intrabar_long_setup_threshold=_coerce_optional_float(
                payload.get("meta_intrabar_long_setup_threshold", 0.42)
            ),
            meta_intrabar_short_setup_threshold=_coerce_optional_float(
                payload.get("meta_intrabar_short_setup_threshold", 0.15)
            ),
            meta_hard_stop_atr=max(0.0, _coerce_float(payload.get("meta_hard_stop_atr"), 0.0)),
            meta_trail_activate_atr=_coerce_float(payload.get("meta_trail_activate_atr"), 2.0),
            meta_trail_atr=_coerce_float(payload.get("meta_trail_atr"), 1.0),
            meta_trail_atr_after_tp=_coerce_float(payload.get("meta_trail_atr_after_tp"), 0.8),
            meta_use_tp_to_tighten_trail=_coerce_bool(payload.get("meta_use_tp_to_tighten_trail"), True),
            env_file=str(payload.get("env_file", ".env")),
            prefill_path=payload.get("prefill_path", DEFAULT_SPY_1M_PATH),
            prefill_start=str(payload.get("prefill_start", "2026-01-30")),
            no_prefill_fetch=_coerce_bool(payload.get("no_prefill_fetch"), False),
            prefill_tail=(
                None
                if payload.get("prefill_tail") in (None, "")
                else max(50, _coerce_int(payload.get("prefill_tail"), 0))
            ),
            enable_option_orders=_coerce_bool(payload.get("enable_option_orders"), True),
            no_startup_sync=_coerce_bool(payload.get("no_startup_sync"), False),
            option_order_qty=max(1, _coerce_int(payload.get("option_order_qty"), 1)),
            option_atr_mult=max(0.0, _coerce_float(payload.get("option_atr_mult"), 1.0)),
            option_dte_cutoff=str(payload.get("option_dte_cutoff", "13:00")),
            option_exit_policy=str(payload.get("option_exit_policy", "option_adaptive_trail_v1")).strip().lower(),
            option_exit_take_profit_pct=max(
                0.0,
                _coerce_float(payload.get("option_exit_take_profit_pct"), 0.0),
            ),
            option_exit_stop_loss_pct=max(
                0.0,
                _coerce_float(payload.get("option_exit_stop_loss_pct"), 1.0),
            ),
            option_exit_profit_lock_arm_pct=max(
                0.0,
                _coerce_float(payload.get("option_exit_profit_lock_arm_pct"), 2.0),
            ),
            option_exit_profit_lock_floor_pct=max(
                0.0,
                _coerce_float(payload.get("option_exit_profit_lock_floor_pct"), 0.25),
            ),
            option_exit_trailing_arm_pct=max(
                0.0,
                _coerce_float(payload.get("option_exit_trailing_arm_pct"), 2.0),
            ),
            option_exit_trailing_giveback_pct=max(
                0.0,
                _coerce_float(payload.get("option_exit_trailing_giveback_pct"), 0.25),
            ),
            option_exit_time_decay_minutes=max(
                0,
                _coerce_int(payload.get("option_exit_time_decay_minutes"), 80),
            ),
            option_exit_time_decay_progress_pct=max(
                0.0,
                _coerce_float(payload.get("option_exit_time_decay_progress_pct"), 1.0),
            ),
            option_exit_opposite_prob=max(
                0.0,
                _coerce_float(payload.get("option_exit_opposite_prob"), 0.60),
            ),
            option_exit_quote_mode=str(payload.get("option_exit_quote_mode", "bid")).strip().lower(),
            replay_option_proxy_mode=str(payload.get("replay_option_proxy_mode", "black_scholes")).strip().lower(),
            replay_option_proxy_expiry_hhmm=str(payload.get("replay_option_proxy_expiry_hhmm", "15:40")),
            replay_option_proxy_iv_floor=max(
                0.0,
                _coerce_float(payload.get("replay_option_proxy_iv_floor"), 0.12),
            ),
            replay_option_proxy_iv_ceiling=max(
                0.0,
                _coerce_float(payload.get("replay_option_proxy_iv_ceiling"), 0.90),
            ),
            replay_option_proxy_iv_multiplier=max(
                0.0,
                _coerce_float(payload.get("replay_option_proxy_iv_multiplier"), 1.50),
            ),
            replay_option_proxy_min_dte_minutes=max(
                0.0,
                _coerce_float(payload.get("replay_option_proxy_min_dte_minutes"), 1.0),
            ),
            simulate_orders=_coerce_bool(payload.get("simulate_orders"), True),
            option_no_close_on_flat=_coerce_bool(payload.get("option_no_close_on_flat"), False),
            option_no_close_on_flip=_coerce_bool(payload.get("option_no_close_on_flip"), False),
            startup_catchup_order=_coerce_bool(payload.get("startup_catchup_order"), False),
            startup_catchup_max_age_min=_coerce_int(payload.get("startup_catchup_max_age_min"), 120),
            exec_entry_confirm_bars=max(1, _coerce_int(payload.get("exec_entry_confirm_bars"), 1)),
            exec_exit_confirm_bars=max(1, _coerce_int(payload.get("exec_exit_confirm_bars"), 2)),
            exec_flip_abs_threshold=max(0.0, _coerce_float(payload.get("exec_flip_abs_threshold"), 0.05)),
            use_execution_latch=_coerce_bool(payload.get("use_execution_latch"), False),
            replay_data_path=str(payload.get("replay_data_path", DEFAULT_SPY_1M_PATH)),
            replay_start=payload.get("replay_start", "2026-03-14T00:00:00Z"),
            replay_end=payload.get("replay_end", "2026-03-23T23:59:59Z"),
            replay_regular_only=_coerce_bool(payload.get("replay_regular_only"), False),
            replay_sleep=max(0.0, _coerce_float(payload.get("replay_sleep"), 0.0)),
            replay_max_bars=(
                None
                if payload.get("replay_max_bars") in (None, "")
                else max(1, _coerce_int(payload.get("replay_max_bars"), 0))
            ),
            replay_warmup_bars=max(0, _coerce_int(payload.get("replay_warmup_bars"), 5000)),
            replay_no_prepend_split_test_warmup=_coerce_bool(
                payload.get("replay_no_prepend_split_test_warmup"),
                False,
            ),
            eval_parity_mode=_coerce_bool(payload.get("eval_parity_mode"), False),
            audit_enabled=_coerce_bool(payload.get("audit_enabled"), True),
            audit_root=str(payload.get("audit_root", "Data/inference/live_runs")),
        )


class LiveAuditWriter:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self._queue: queue_mod.Queue = queue_mod.Queue(maxsize=20000)
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self._dropped = 0
        self._lock = threading.Lock()
        self._meta: dict[str, Any] = {}

    def start(self, *, metadata: dict[str, Any]) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._meta = dict(_json_safe(metadata) or {})
        self._meta["session_dir"] = str(self.session_dir)
        self._write_meta()
        self._thread = threading.Thread(target=self._run, name="live-audit-writer", daemon=True)
        self._thread.start()

    def enqueue(self, stream: str, payload: dict[str, Any]) -> None:
        if self._stopped.is_set():
            return
        item = {
            "stream": str(stream),
            "recorded_at": _ts_iso(datetime.now(timezone.utc)),
            "payload": _json_safe(payload),
        }
        try:
            self._queue.put_nowait(item)
        except queue_mod.Full:
            with self._lock:
                self._dropped += 1
            try:
                self._queue.get_nowait()
            except queue_mod.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue_mod.Full:
                with self._lock:
                    self._dropped += 1

    def stop(self) -> None:
        self._stopped.set()
        try:
            self._queue.put_nowait(None)
        except queue_mod.Full:
            pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._meta["stopped_at"] = _ts_iso(datetime.now(timezone.utc))
        with self._lock:
            self._meta["dropped_records"] = int(self._dropped)
        self._write_meta()

    def _write_meta(self) -> None:
        meta_path = self.session_dir / "session_meta.json"
        meta_path.write_text(json.dumps(_json_safe(self._meta), indent=2), encoding="utf-8")

    def _run(self) -> None:
        handles: dict[str, Any] = {}
        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.5)
                except queue_mod.Empty:
                    if self._stopped.is_set():
                        break
                    continue
                if item is None:
                    break
                stream = _slugify_token(item.get("stream"), default="events")
                handle = handles.get(stream)
                if handle is None:
                    path = self.session_dir / f"{stream}.jsonl"
                    handle = path.open("a", encoding="utf-8", buffering=1)
                    handles[stream] = handle
                handle.write(json.dumps(_json_safe(item), separators=(",", ":")) + "\n")
        finally:
            for handle in handles.values():
                try:
                    handle.close()
                except Exception:
                    pass


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
        self._max_bars_1m = int(max_bars) * 20
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
        self._realized_total: float = 0.0
        self._realized_prev_day: str | None = None
        self._realized_prev_today: float | None = None

    def start(self, config: SessionConfig) -> None:
        symbols = list(config.symbols)
        with self._lock:
            self._running = True
            self._started_at = _ts_iso(datetime.now(timezone.utc))
            self._stopped_at = None
            self._last_error = None
            self._status_message = "starting"
            self._config = _json_safe(config.__dict__)
            self._symbols = symbols
            self._bars_1m = {s: deque(maxlen=self._max_bars_1m) for s in symbols}
            self._bars_15m = {s: deque(maxlen=self._max_bars) for s in symbols}
            self._last_actions = {}
            self._agent_state = None
            self._policy_state = {}
            self._broker_state = {}
            self._action_events.clear()
            self._trade_events.clear()
            self._log_events.clear()
            self._realized_total = 0.0
            self._realized_prev_day = None
            self._realized_prev_today = None

    def stop(self, *, error: str | None = None) -> None:
        with self._lock:
            self._running = False
            self._stopped_at = _ts_iso(datetime.now(timezone.utc))
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
            self._bars_1m.setdefault(symbol, deque(maxlen=self._max_bars_1m)).append(payload)

    def add_15m_bar(self, symbol: str, bar: dict[str, Any]) -> None:
        payload = _normalize_bar(bar)
        with self._lock:
            self._bars_15m.setdefault(symbol, deque(maxlen=self._max_bars)).append(payload)

    def set_last_action(
        self,
        symbol: str,
        *,
        action: float,
        action_class: int | None,
        ts: Any,
        close: Any,
        record_event: bool = True,
    ) -> None:
        close_val = _coerce_float(close, float("nan"))
        raw_val = _coerce_float(action, float("nan"))
        cls_val = int(action_class) if action_class is not None else _action_to_position(raw_val)
        payload = {
            "symbol": symbol,
            "action": cls_val,
            "action_class": cls_val,
            "action_raw": raw_val if np.isfinite(raw_val) else None,
            "timestamp": _ts_iso(ts),
            "close": close_val if np.isfinite(close_val) else None,
        }
        with self._lock:
            self._last_actions[symbol] = payload
            if record_event:
                self._action_events.append(payload)

    def set_agent_state(self, state: dict[str, Any] | None) -> None:
        with self._lock:
            safe = _json_safe(state) if state is not None else None
            if isinstance(safe, dict):
                last_probs = safe.get("last_probs")
                if isinstance(last_probs, dict):
                    for key, value in last_probs.items():
                        safe[key] = value
                last_prob_sources = safe.get("last_prob_sources")
                if isinstance(last_prob_sources, dict):
                    for key, value in last_prob_sources.items():
                        safe[key] = value
                last_vix_status = safe.get("last_vix_status")
                if isinstance(last_vix_status, dict):
                    for key, value in last_vix_status.items():
                        safe[f"vix_{key}"] = value
                last_context_status = safe.get("last_context_status")
                if isinstance(last_context_status, dict):
                    for symbol, payload in last_context_status.items():
                        sym = str(symbol or "").strip().lower()
                        if not sym or not isinstance(payload, dict):
                            continue
                        for key, value in payload.items():
                            safe[f"ctx_{sym}_{key}"] = value
                last_thresholds = safe.get("last_thresholds")
                if isinstance(last_thresholds, dict):
                    for key, value in last_thresholds.items():
                        safe[f"thr_{key}"] = value
                day = safe.get("last_session_day")
                day_key = str(day) if day is not None else None
                today_val = _coerce_float(safe.get("realized_pnl_today"), float("nan"))
                if np.isfinite(today_val):
                    if self._realized_prev_day is None:
                        self._realized_prev_day = day_key
                        self._realized_prev_today = 0.0
                    elif day_key != self._realized_prev_day:
                        self._realized_prev_day = day_key
                        self._realized_prev_today = 0.0
                    prev_today = float(self._realized_prev_today or 0.0)
                    delta = float(today_val - prev_today)
                    if np.isfinite(delta):
                        self._realized_total += delta
                    self._realized_prev_today = float(today_val)
                safe["realized_pnl_total"] = float(self._realized_total)
            self._agent_state = safe

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
            data = dict(payload or {})
            data.setdefault("timestamp", _ts_iso(datetime.now(timezone.utc)))
            self._log_events.append(_json_safe(data))

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
                records = df.tail(self._max_bars_1m).to_dict("records")
                target = self._bars_1m.setdefault(symbol, deque(maxlen=self._max_bars_1m))
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

    def restore_snapshot(
        self,
        snap: dict[str, Any],
        *,
        running: bool,
        status_message: str,
    ) -> None:
        with self._lock:
            self._running = bool(running)
            self._started_at = _ts_iso(datetime.now(timezone.utc))
            self._stopped_at = None
            self._last_error = None
            self._status_message = str(status_message)
            self._config = _json_safe(snap.get("config", self._config or {}))
            self._symbols = [str(x).upper() for x in (snap.get("symbols") or [])]
            self._bars_1m = {
                str(symbol).upper(): deque((rows or []), maxlen=self._max_bars_1m)
                for symbol, rows in (snap.get("bars_1m") or {}).items()
            }
            self._bars_15m = {
                str(symbol).upper(): deque((rows or []), maxlen=self._max_bars)
                for symbol, rows in (snap.get("bars_15m") or {}).items()
            }
            self._last_actions = _json_safe(snap.get("last_actions") or {})
            self._agent_state = _json_safe(snap.get("agent_state"))
            self._policy_state = _json_safe(snap.get("policy_state") or {})
            self._broker_state = _json_safe(snap.get("broker_state") or {})
            self._action_events = deque((snap.get("action_events") or []), maxlen=self._action_events.maxlen)
            self._trade_events = deque((snap.get("trade_events") or []), maxlen=self._trade_events.maxlen)
            self._log_events = deque((snap.get("logs") or []), maxlen=self._log_events.maxlen)


class LiveSession:
    def __init__(self, store: DashboardStore, broker: EventBroker) -> None:
        self._store = store
        self._broker = broker
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._streamer: AlpacaBarStreamer | None = None
        self._config: SessionConfig | None = None
        self._audit_writer: LiveAuditWriter | None = None

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
            "timestamp": _ts_iso(datetime.now(timezone.utc)),
            "payload": _json_safe(payload),
        }
        self._broker.publish(event)
        if event_type == "log":
            self._audit("logs", payload)

    def _audit(self, stream: str, payload: dict[str, Any]) -> None:
        writer = self._audit_writer
        if writer is None:
            return
        writer.enqueue(stream, payload)

    def _emit_status(self, *, running: bool, message: str, extra: dict[str, Any] | None = None) -> None:
        self._store.set_status_message(message)
        payload: dict[str, Any] = {"running": bool(running), "message": str(message)}
        if extra:
            payload.update(extra)
        self._emit("status", payload)
        self._audit("status", payload)

    def _emit_state_sync(self) -> None:
        self._emit("state_sync", {"state": self._store.snapshot()})

    def _policy_logger(self, symbol: str) -> Callable[[str], None]:
        def _logger(message: str) -> None:
            payload = {"symbol": symbol, "message": str(message)}
            self._store.add_log_event(payload)
            self._emit("log", payload)

        return _logger

    def _resolve_ga_feature_list(self, cfg: SessionConfig) -> str | None:
        try:
            from API.Alpaca_API.runners import live_runner as lr

            shared = lr._resolve_ga_feature_list_path(
                symbol=cfg.symbols[0],
                dataset_name=cfg.ga_dataset_name,
                ga_feature_list=cfg.ga_feature_list,
                inference_enabled=not cfg.no_agent,
                include_pivot_probs=not cfg.no_pivot_probs,
                include_tb_probs=not cfg.no_tb_probs,
            )
            if shared:
                return shared
        except Exception:
            pass
        try:
            from Data.load_data import get_ticker_processed_base_dir
            from Data.retrieve_data import normalize_ticker

            ticker = normalize_ticker(cfg.symbols[0])
            candidates = [
                get_ticker_processed_base_dir(ticker)
                / "datasets"
                / cfg.ga_dataset_name
                / f"features_X_{cfg.ga_dataset_name}_tree.txt",
                Path("Data")
                / "inference"
                / ticker.lower()
                / cfg.ga_dataset_name
                / "datasets"
                / cfg.ga_dataset_name
                / f"features_X_{cfg.ga_dataset_name}_tree.txt",
            ]
            for candidate in candidates:
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
        inference: Any | None = None,
        execution_latches: dict[str, DirectionExecutionLatch] | None = None,
        last_exec_action_by_symbol: dict[str, int] | None = None,
    ) -> None:
        from API.Alpaca_API.inference.live_inference import LiveIndependentMetaXGBAgent, LiveMetaXGBAgent, build_15m

        seeded_counts: dict[str, int] = {}
        warmup_action_counts: dict[str, int] = {}
        skip_meta_warmup_replay = False
        for symbol in cfg.symbols:
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
                df_1m.index = df_1m.index.tz_localize(cfg.assume_tz)
            df_1m_utc = df_1m.sort_index().tz_convert("UTC")

            try:
                df_15m = build_15m(
                    df_1m_utc,
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
                    df_plot = agent._maybe_add_ga_probs(df_1m=df_1m_utc, df_15m=df_15m, target_ts=None)  # noqa: SLF001
                except Exception:
                    df_plot = df_15m

            if "timestamp" not in df_plot.columns:
                df_plot = df_plot.reset_index().rename(columns={df_plot.index.name or "index": "timestamp"})
            else:
                df_plot = df_plot.reset_index(drop=True)
            df_plot = df_plot.tail(self._store._max_bars)

            warmup_actions = 0
            warmup_by_ts: dict[pd.Timestamp, dict[str, Any]] = {}
            if inference is not None and agent is not None and not skip_meta_warmup_replay:
                try:
                    warmup_records = agent.replay_warmup_actions(
                        df_1m=df_1m_utc,
                        df_15m=df_plot.set_index("timestamp"),
                        apply_ga_probs=False,
                    )
                except Exception:
                    warmup_records = []
                warmup_by_ts = {
                    pd.to_datetime(rec.get("timestamp"), utc=True, errors="coerce"): rec
                    for rec in warmup_records
                    if pd.notna(pd.to_datetime(rec.get("timestamp"), utc=True, errors="coerce"))
                }
                for rec in warmup_records:
                    action_raw = _coerce_float(rec.get("action"), float("nan"))
                    prev_exec_action = (
                        last_exec_action_by_symbol.get(symbol)
                        if last_exec_action_by_symbol is not None
                        else None
                    )
                    gated_action_raw, _model_raw_pos, raw_action_pos, _flip_blocked = _apply_flip_abs_threshold(
                        action_raw=action_raw,
                        prev_exec_action=prev_exec_action,
                        flip_abs_threshold=float(cfg.exec_flip_abs_threshold),
                    )
                    if cfg.eval_parity_mode:
                        selected_action = (
                            gated_action_raw
                            if np.isfinite(gated_action_raw)
                            else float(raw_action_pos)
                        )
                        selected_action_class = raw_action_pos
                    elif (
                        cfg.use_execution_latch
                        and execution_latches is not None
                        and symbol in execution_latches
                    ):
                        gate = execution_latches[symbol].step(raw_action_pos)
                        exec_action_pos = int(gate.executed_pos)
                        selected_action = float(exec_action_pos)
                        selected_action_class = exec_action_pos
                    else:
                        selected_action = (
                            gated_action_raw
                            if np.isfinite(gated_action_raw)
                            else float(raw_action_pos)
                        )
                        selected_action_class = raw_action_pos
                    self._store.set_last_action(
                        symbol,
                        action=float(selected_action),
                        action_class=selected_action_class,
                        ts=rec.get("timestamp"),
                        close=rec.get("close"),
                    )
                    if last_exec_action_by_symbol is not None:
                        last_exec_action_by_symbol[symbol] = selected_action_class
                    warmup_actions += 1
                if warmup_actions > 0:
                    self._store.set_agent_state(agent.snapshot_state())

            count = 0
            for row in df_plot.to_dict("records"):
                ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
                rec = warmup_by_ts.get(ts)
                bar = {
                    "symbol": symbol,
                    "timestamp": row.get("timestamp"),
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": row.get("close"),
                    "volume": row.get("volume"),
                }
                for key in (
                    "p_pivot_long",
                    "p_pivot_short",
                    "p_tb_long",
                    "p_tb_short",
                    "p_enter_long",
                    "p_enter_short",
                    "p_exit_long",
                    "p_exit_short",
                    "thr_enter_long",
                    "thr_enter_short",
                    "thr_exit_long",
                    "thr_exit_short",
                ):
                    if rec is not None and key in rec:
                        bar[key] = rec.get(key)
                    elif key in row:
                        bar[key] = row.get(key)
                self._store.add_15m_bar(symbol, bar)
                count += 1

            if count > 0:
                seeded_counts[symbol] = count
            if warmup_actions > 0:
                warmup_action_counts[symbol] = warmup_actions

        if seeded_counts:
            self._emit(
                "log",
                {
                    "symbol": "SYSTEM",
                    "message": f"[live] seeded 15m/prob history from prefill: {seeded_counts}",
                },
            )
        if warmup_action_counts:
            self._emit(
                "log",
                {
                    "symbol": "SYSTEM",
                    "message": f"[live] seeded warmup action history from prefill: {warmup_action_counts}",
                },
            )
        elif skip_meta_warmup_replay:
            self._emit(
                "log",
                {
                    "symbol": "SYSTEM",
                    "message": "[live] skipped warmup action replay because cached meta base frame is loaded.",
                },
            )

    def _run_replay(self, *, cfg: SessionConfig, stop_event: threading.Event) -> None:
        symbols = cfg.symbols
        replay_snapshot_cache_path = _default_replay_snapshot_cache_path(cfg)
        can_use_snapshot_cache = float(cfg.replay_sleep or 0.0) <= 0.0
        if can_use_snapshot_cache:
            cached = _load_replay_snapshot_cache(replay_snapshot_cache_path, cfg=cfg)
            cached_state = cached.get("state") if isinstance(cached, dict) else None
            if isinstance(cached_state, dict):
                self._store.restore_snapshot(
                    cached_state,
                    running=True,
                    status_message="replay cache loaded",
                )
                self._store.add_log_event(
                    {
                        "symbol": "SYSTEM",
                        "message": f"[replay] loaded cached dashboard replay snapshot: {replay_snapshot_cache_path}",
                    }
                )
                self._emit(
                    "log",
                    {
                        "symbol": "SYSTEM",
                        "message": f"[replay] loaded cached dashboard replay snapshot: {replay_snapshot_cache_path}",
                    },
                )
                self._emit_status(running=True, message="replay cache loaded")
                self._emit_state_sync()
                return

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
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        replay_all_df = df.copy()

        start = pd.to_datetime(cfg.replay_start, utc=True, errors="coerce") if cfg.replay_start else pd.NaT
        end = pd.to_datetime(cfg.replay_end, utc=True, errors="coerce") if cfg.replay_end else pd.NaT
        visible_df = replay_all_df.copy()
        if pd.notna(start):
            visible_df = visible_df[visible_df["timestamp"] >= start]
        if pd.notna(end):
            visible_df = visible_df[visible_df["timestamp"] <= end]

        if cfg.replay_max_bars is not None:
            keep_n = int(cfg.replay_max_bars)
            if keep_n > 0:
                visible_df = visible_df.tail(keep_n).copy()

        visible_start_marker = pd.NaT
        if not visible_df.empty:
            visible_start_marker = pd.to_datetime(visible_df["timestamp"], utc=True, errors="coerce").min()
        elif pd.notna(start):
            visible_start_marker = start

        warmup_frames: list[pd.DataFrame] = []
        warmup_n = max(0, int(cfg.replay_warmup_bars))
        prepend_replay_warmup = (not cfg.replay_no_prepend_split_test_warmup) and warmup_n > 0 and pd.notna(visible_start_marker)
        if prepend_replay_warmup:
            for symbol in symbols:
                sym_warm = replay_all_df[
                    (replay_all_df["symbol"].astype(str).str.upper() == str(symbol).upper())
                    & (replay_all_df["timestamp"] < visible_start_marker)
                ].tail(warmup_n)
                if sym_warm.empty:
                    continue
                sym_warm = sym_warm.copy()
                sym_warm["__source"] = "replay_warmup"
                warmup_frames.append(sym_warm)
            if warmup_frames:
                warmup = pd.concat(warmup_frames, axis=0, ignore_index=True)
                df = pd.concat([warmup, visible_df], axis=0, ignore_index=True)
                df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
                df = df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
                self._emit(
                    "log",
                    {
                        "symbol": "SYSTEM",
                        "message": (
                            f"[replay] prepended same-file warmup bars: {len(warmup):,} "
                            f"(hidden before visible start {_ts_iso(visible_start_marker)}, requested={warmup_n:,})"
                        ),
                    },
                )
            else:
                df = visible_df.copy()
                self._emit(
                    "log",
                    {
                        "symbol": "SYSTEM",
                        "message": (
                            "[replay] no same-file bars found before replay visible start; "
                            "continuing without split-test warmup."
                        ),
                    },
                )
        else:
            df = visible_df.copy()

        required = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Replay data missing required columns: {missing}")
        df = df.sort_values("timestamp")

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
        visible_start_ts = visible_start_marker
        last_warmup_interval_ts_by_symbol: dict[str, pd.Timestamp] = {}
        visible_agent_primed: set[str] = set()

        def _is_replay_warmup_bar(ts: Any) -> bool:
            if pd.isna(visible_start_ts):
                return False
            parsed = pd.to_datetime(ts, utc=True, errors="coerce")
            return bool(pd.notna(parsed) and parsed < visible_start_ts)

        def _to_agent_index_ts(ts: Any) -> pd.Timestamp | None:
            parsed = pd.to_datetime(ts, utc=True, errors="coerce")
            if pd.isna(parsed):
                return None
            try:
                return pd.Timestamp(parsed).tz_convert(cfg.tz or "America/New_York")
            except Exception:
                return pd.Timestamp(parsed)

        def _prime_agent_at_visible_start(symbol: str) -> None:
            if symbol in visible_agent_primed:
                return
            visible_agent_primed.add(symbol)
            if agent is None:
                return
            last_warmup_ts = last_warmup_interval_ts_by_symbol.get(symbol)
            if last_warmup_ts is None:
                return
            if hasattr(agent, "_last_processed_ts"):
                try:
                    setattr(agent, "_last_processed_ts", last_warmup_ts)
                except Exception:
                    pass

        agent = None
        inference = None
        inference_mode = "none" if cfg.no_agent else str(cfg.inference_mode or "meta").strip().lower()
        if inference_mode != "none":
            self._emit_status(running=True, message=f"replay loading {inference_mode} agent")
            from API.Alpaca_API.inference.live_inference import LiveInferenceEngine
            from API.Alpaca_API.runners import live_runner as lr

            ga_feature_list = self._resolve_ga_feature_list(cfg)
            ga_probs_frame = _load_agent_matrix_probs(symbol=symbols[0], dataset_name=cfg.ga_dataset_name)
            if cfg.eval_parity_mode and ga_probs_frame is not None and ga_feature_list:
                ga_probs_mode = "hybrid"
            elif cfg.eval_parity_mode and ga_probs_frame is not None:
                ga_probs_mode = "frame"
            elif ga_probs_frame is not None and ga_feature_list:
                ga_probs_mode = "hybrid"
            elif ga_probs_frame is not None:
                ga_probs_mode = "frame"
            else:
                ga_probs_mode = "xgb"
            if ga_probs_frame is not None:
                try:
                    src = ga_probs_frame.attrs.get("source_paths", [])
                    rng_min = ga_probs_frame.index.min()
                    rng_max = ga_probs_frame.index.max()
                    self._emit(
                        "log",
                        {
                            "symbol": "SYSTEM",
                            "message": (
                                "[replay] loaded agent_matrix probs "
                                f"(rows={len(ga_probs_frame):,}, "
                                f"range={_ts_iso(rng_min)}..{_ts_iso(rng_max)}, "
                                f"sources={len(src)})"
                            ),
                        },
                    )
                except Exception:
                    pass

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
                                "message": f"[replay] using {ga_probs_mode} probability mode for PPO features.",
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
            precomputed_meta_frame = None
            if inference_mode == "meta" and cfg.meta_base_frame_path:
                try:
                    meta_base_path = lr._resolve_symbolized_path(cfg.meta_base_frame_path, symbol=symbols[0])
                    precomputed_meta_frame = lr._load_precomputed_meta_frame(
                        meta_base_path,
                        tz=cfg.tz or "America/New_York",
                    )
                    if not precomputed_meta_frame.empty:
                        self._emit(
                            "log",
                            {
                                "symbol": "SYSTEM",
                                "message": (
                                    f"[replay] loaded cached meta base frame: rows={len(precomputed_meta_frame):,} "
                                    f"range={_ts_iso(precomputed_meta_frame.index.min())}..{_ts_iso(precomputed_meta_frame.index.max())} "
                                    f"path={meta_base_path}"
                                ),
                            },
                        )
                except Exception as exc:
                    self._emit(
                        "log",
                        {
                            "symbol": "SYSTEM",
                            "message": f"[replay] cached meta base frame unavailable: {exc}",
                        },
                    )

            swing_setup_probs_frame = None
            if (
                inference_mode == "meta"
                and str(cfg.meta_entry_prob_source or "").strip().lower() == "swing_support_single"
            ):
                swing_setup_probs_frame = _load_swing_setup_probs_frame(
                    model_dir=cfg.swing_setup_single_model_dir,
                    tz=cfg.tz or "America/New_York",
                )
                if swing_setup_probs_frame is not None and not swing_setup_probs_frame.empty:
                    self._emit(
                        "log",
                        {
                            "symbol": "SYSTEM",
                            "message": (
                                "[replay] using saved swing setup probabilities "
                                f"(rows={len(swing_setup_probs_frame):,}, "
                                f"range={_ts_iso(swing_setup_probs_frame.index.min())}.."
                                f"{_ts_iso(swing_setup_probs_frame.index.max())}, "
                                f"path={Path(cfg.swing_setup_single_model_dir) / 'p_swing_probs.parquet'})"
                            ),
                        },
                    )
                else:
                    self._emit(
                        "log",
                        {
                            "symbol": "SYSTEM",
                            "message": (
                                "[replay] saved swing setup probabilities unavailable; "
                                "falling back to live feature recomputation."
                            ),
                        },
                    )

            if inference_mode == "meta":
                agent = lr._build_meta_agent(
                    symbol=symbols[0],
                    model_root=cfg.meta_model_root,
                    ga_model_root=cfg.ga_model_root,
                    ga_feature_list_path=ga_feature_list,
                    ga_probs_frame=ga_probs_frame,
                    ga_probs_mode=ga_probs_mode,
                    include_pivot_probs=not cfg.no_pivot_probs,
                    include_tb_probs=not cfg.no_tb_probs,
                    pivot_label_dir=cfg.ga_pivot_label_dir,
                    tb_label_dir=cfg.ga_tb_label_dir,
                    tz=cfg.tz or "America/New_York",
                    assume_tz=cfg.assume_tz,
                    session_open=cfg.session_open,
                    session_close=cfg.session_close,
                    min_15m_bars=cfg.min_15m_bars,
                    fill_missing_prob=cfg.fill_missing_prob,
                    resample_label=cfg.resample_label,
                    resample_closed=cfg.resample_closed,
                    label_timeframe_rule=f"{cfg.interval}min",
                    trail_activate_atr=float(cfg.meta_trail_activate_atr),
                    trail_atr=float(cfg.meta_trail_atr),
                    trail_atr_after_tp=float(cfg.meta_trail_atr_after_tp),
                    use_tp_to_tighten_trail=bool(cfg.meta_use_tp_to_tighten_trail),
                    entry_threshold_override=cfg.meta_entry_threshold,
                    exit_threshold_override=cfg.meta_exit_threshold,
                    precomputed_base_frame=precomputed_meta_frame,
                    precomputed_append_lookback_days=int(cfg.meta_base_frame_append_lookback_days),
                    min_hold_bars=2,
                    exit_entry_delta=0.15,
                    soft_exit_confirm_bars=2,
                    urgent_exit_prob=0.85,
                    urgent_exit_delta=0.30,
                    profit_protect_enabled=False,
                    profit_protect_arm_atr=2.0,
                    profit_protect_giveback_atr_long=0.75,
                    profit_protect_giveback_atr_short=1.0,
                    entry_prob_source=cfg.meta_entry_prob_source,
                    swing_setup_single_model_dir=cfg.swing_setup_single_model_dir,
                    swing_setup_probs_frame=swing_setup_probs_frame,
                )
                self._emit(
                    "log",
                    {
                        "symbol": "SYSTEM",
                        "message": (
                            (
                                f"[replay] Swing setup wrapper enabled: model_dir={cfg.swing_setup_single_model_dir} "
                                f"timeframe={cfg.interval}min"
                            )
                            if str(cfg.meta_entry_prob_source or "").strip().lower() == "swing_support_single"
                            else (
                                    f"[replay] Legacy probability wrapper enabled: model_root={cfg.meta_model_root} "
                                f"entry_source={cfg.meta_entry_prob_source} timeframe={cfg.interval}min"
                            )
                        ),
                    },
                )
            else:
                agent = lr._build_ppo_agent(
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
                    ga_model_root=cfg.ga_model_root,
                    ga_feature_list_path=ga_feature_list,
                    ga_pivot_label_dir=cfg.ga_pivot_label_dir,
                    ga_tb_label_dir=cfg.ga_tb_label_dir,
                    ga_probs_frame=ga_probs_frame,
                    ga_probs_mode=ga_probs_mode,
                    require_probs=True,
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
        replay_option_proxies: dict[str, ReplayOptionPriceProxy] = {}
        simulated_orders: dict[str, deque] = {}
        last_exec_action_by_symbol: dict[str, int] = {}
        execution_latches: dict[str, DirectionExecutionLatch] = {
            symbol: DirectionExecutionLatch(
                entry_confirm_bars=int(cfg.exec_entry_confirm_bars),
                exit_confirm_bars=int(cfg.exec_exit_confirm_bars),
                initial_position=0,
            )
            for symbol in symbols
        }
        if cfg.enable_option_orders:
            from Policy.order_policy import OptionOrderPolicy

            order_policies = {}
            for symbol in symbols:
                policy = lr._build_option_order_policy(
                    symbol=symbol,
                    env_file=cfg.env_file,
                    tz_name=cfg.tz,
                    atr_multiplier=float(cfg.option_atr_mult),
                    dte_cutoff_hhmm=cfg.option_dte_cutoff,
                    qty=int(cfg.option_order_qty),
                    close_on_flat=not cfg.option_no_close_on_flat,
                    close_on_flip=not cfg.option_no_close_on_flip,
                    submit_orders=False,
                    opposite_confirm_bars=1,
                    opposite_min_abs_action=float(cfg.exec_flip_abs_threshold),
                    opposite_min_prob_edge=0.0,
                    meta_execute_on_interval_close=False,
                    meta_intrabar_execution_enabled=True,
                    meta_intrabar_breakout_entry_only=True,
                    meta_intrabar_entry_policy=str(cfg.meta_intrabar_entry_policy),
                    meta_intrabar_setup_max_bars=int(cfg.meta_intrabar_setup_max_bars),
                    meta_intrabar_setup_bar_minutes=int(cfg.interval),
                    meta_intrabar_long_setup_threshold=cfg.meta_intrabar_long_setup_threshold,
                    meta_intrabar_short_setup_threshold=cfg.meta_intrabar_short_setup_threshold,
                    meta_hard_stop_atr=float(cfg.meta_hard_stop_atr),
                    meta_trail_activate_atr=float(cfg.meta_trail_activate_atr),
                    meta_trail_atr=float(cfg.meta_trail_atr),
                    meta_trail_atr_after_tp=float(cfg.meta_trail_atr_after_tp),
                    meta_use_tp_to_tighten_trail=bool(cfg.meta_use_tp_to_tighten_trail),
                    meta_soft_exit_confirm_bars=2,
                    meta_urgent_exit_prob=0.85,
                    meta_urgent_exit_delta=0.30,
                    meta_profit_protect_enabled=False,
                    meta_profit_protect_arm_atr=2.0,
                    meta_profit_protect_giveback_atr_long=0.75,
                    meta_profit_protect_giveback_atr_short=1.0,
                    option_exit_policy=str(cfg.option_exit_policy),
                    option_exit_take_profit_pct=float(cfg.option_exit_take_profit_pct),
                    option_exit_stop_loss_pct=float(cfg.option_exit_stop_loss_pct),
                    option_exit_profit_lock_arm_pct=float(cfg.option_exit_profit_lock_arm_pct),
                    option_exit_profit_lock_floor_pct=float(cfg.option_exit_profit_lock_floor_pct),
                    option_exit_trailing_arm_pct=float(cfg.option_exit_trailing_arm_pct),
                    option_exit_trailing_giveback_pct=float(cfg.option_exit_trailing_giveback_pct),
                    option_exit_time_decay_minutes=int(cfg.option_exit_time_decay_minutes),
                    option_exit_time_decay_progress_pct=float(cfg.option_exit_time_decay_progress_pct),
                    option_exit_opposite_prob=float(cfg.option_exit_opposite_prob),
                    option_exit_quote_mode=str(cfg.option_exit_quote_mode),
                )
                if str(cfg.replay_option_proxy_mode).lower() == "black_scholes":
                    proxy = ReplayOptionPriceProxy(
                        tz_name=cfg.tz,
                        expiry_hhmm=str(cfg.replay_option_proxy_expiry_hhmm),
                        iv_floor=float(cfg.replay_option_proxy_iv_floor),
                        iv_ceiling=float(cfg.replay_option_proxy_iv_ceiling),
                        iv_multiplier=float(cfg.replay_option_proxy_iv_multiplier),
                        min_dte_minutes=float(cfg.replay_option_proxy_min_dte_minutes),
                    )
                    policy.set_contract_price_provider(
                        lambda *, symbol, mode=None, proxy=proxy: proxy.price(symbol, mode=mode)
                    )
                    replay_option_proxies[symbol] = proxy
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
            if replay_option_proxies:
                self._emit(
                    "log",
                    {
                        "symbol": "SYSTEM",
                        "message": "[replay] simulated option pricing uses Black-Scholes proxy from replay bars.",
                    },
                )

        def _set_sim_broker_state(symbol: str) -> dict[str, Any]:
            if order_policies is None or symbol not in order_policies:
                return {"simulation": True, "positions": [], "recent_orders": []}
            pol_state = order_policies[symbol].snapshot_state()
            positions = []
            if pol_state.get("open_long_symbol"):
                positions.append(
                    {
                        "symbol": pol_state.get("open_long_symbol"),
                        "side": "long",
                        "qty": pol_state.get("long_contracts"),
                        "avg_entry_price": None,
                        "market_value": None,
                        "unrealized_pl": None,
                    }
                )
            if pol_state.get("open_short_symbol"):
                positions.append(
                    {
                        "symbol": pol_state.get("open_short_symbol"),
                        "side": "short",
                        "qty": pol_state.get("short_contracts"),
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
            legs = result.get("orders")
            if isinstance(legs, list):
                for leg in legs:
                    if not isinstance(leg, dict):
                        continue
                    resp = leg.get("response") if isinstance(leg.get("response"), dict) else {}
                    payload = resp.get("payload") if isinstance(resp.get("payload"), dict) else {}
                    side = resp.get("side", payload.get("side"))
                    qty_val = resp.get("qty", payload.get("qty", leg.get("qty")))
                    simulated_orders[symbol].appendleft(
                        {
                            "id": None,
                            "symbol": leg.get("symbol", payload.get("symbol")),
                            "side": side,
                            "status": "simulated",
                            "qty": str(qty_val) if qty_val is not None else None,
                            "filled_qty": str(qty_val) if qty_val is not None else None,
                            "filled_avg_price": None,
                            "submitted_at": ts_iso,
                            "filled_at": ts_iso,
                        }
                    )
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
            if _is_replay_warmup_bar(bar.get("timestamp")):
                if order_policies is not None and symbol in order_policies:
                    order_policies[symbol].prefill_1m_bar(bar=bar)
                return
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
            if order_policies is None or symbol not in order_policies:
                return
            policy = order_policies[symbol]
            result = policy.on_1m_bar(bar=bar, logger=self._policy_logger(symbol))
            ts_iso = _ts_iso(bar.get("timestamp"))
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
            if str(result.get("event", "")).strip().lower() not in {"hold", "no_change", "intent_update"}:
                self._store.set_last_action(
                    symbol,
                    action=float(policy_state.get("position", 0) or 0),
                    action_class=int(policy_state.get("position", 0) or 0),
                    ts=bar.get("timestamp"),
                    close=bar.get("close"),
                    record_event=False,
                )
                self._store.add_trade_event(event_payload)

        def _on_15m(symbol: str, bar15: dict, buffer: Any) -> None:
            if _is_replay_warmup_bar(bar15.get("timestamp")):
                agent_ts = _to_agent_index_ts(bar15.get("timestamp"))
                if agent_ts is not None:
                    last_warmup_interval_ts_by_symbol[symbol] = agent_ts
                if order_policies is not None and symbol in order_policies:
                    order_policies[symbol].on_interval_bar(closed_bar=bar15)
                return

            _prime_agent_at_visible_start(symbol)
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

            prev_exec_action = last_exec_action_by_symbol.get(symbol)
            action_raw = _coerce_float(action, float("nan"))
            gated_action_raw, model_raw_action_pos, raw_action_pos, flip_blocked_by_threshold = _apply_flip_abs_threshold(
                action_raw=action_raw,
                prev_exec_action=prev_exec_action,
                flip_abs_threshold=float(cfg.exec_flip_abs_threshold),
            )
            if cfg.eval_parity_mode:
                selected_action = (
                    gated_action_raw
                    if np.isfinite(gated_action_raw)
                    else float(raw_action_pos)
                )
                selected_action_class = raw_action_pos
                gate_status = "bypassed_parity_mode"
                gate_changed = (
                    prev_exec_action is not None
                    and int(prev_exec_action) != int(selected_action_class)
                )
                gate_pending_target: int | None = None
                gate_pending_count = 0
            elif cfg.use_execution_latch:
                gate = execution_latches[symbol].step(raw_action_pos)
                selected_action_class = int(gate.executed_pos)
                selected_action = float(selected_action_class)
                gate_status = str(gate.status)
                gate_changed = bool(gate.changed)
                gate_pending_target = gate.pending_target
                gate_pending_count = int(gate.pending_count)
            else:
                selected_action = (
                    gated_action_raw
                    if np.isfinite(gated_action_raw)
                    else float(raw_action_pos)
                )
                selected_action_class = raw_action_pos
                gate_status = "bypassed_latch_disabled"
                gate_changed = (
                    prev_exec_action is not None
                    and int(prev_exec_action) != int(selected_action_class)
                )
                gate_pending_target = None
                gate_pending_count = 0
            probs = agent.last_probs() if agent is not None else None
            thresholds = inference.last_thresholds() if inference is not None else None
            bar_payload = dict(bar15)
            if probs:
                bar_payload.update({k: v for k, v in probs.items() if v is not None})
            if thresholds:
                bar_payload.update(
                    {
                        "thr_enter_long": thresholds.get("enter_long"),
                        "thr_enter_short": thresholds.get("enter_short"),
                        "thr_exit_long": thresholds.get("exit_long"),
                        "thr_exit_short": thresholds.get("exit_short"),
                    }
                )
            self._store.add_15m_bar(symbol, bar_payload)
            self._emit("bar_15m", {"symbol": symbol, "bar": _normalize_bar(bar_payload)})
            self._store.set_last_action(
                symbol,
                action=float(selected_action),
                action_class=selected_action_class,
                ts=bar15.get("timestamp"),
                close=bar15.get("close"),
            )
            last_exec_action_by_symbol[symbol] = selected_action_class
            agent_state = agent.snapshot_state() if agent is not None else None
            if agent_state is not None and thresholds:
                agent_state = dict(agent_state)
                agent_state["last_thresholds"] = thresholds
            if agent_state is not None:
                self._store.set_agent_state(agent_state)
            self._emit(
                "action",
                {
                    "symbol": symbol,
                    "action": float(selected_action),
                    "action_class": selected_action_class,
                    "action_raw": action_raw if np.isfinite(action_raw) else None,
                    "action_raw_exec": float(selected_action),
                    "raw_action_class": raw_action_pos,
                    "raw_action_class_model": model_raw_action_pos,
                    "execution_gate": {
                        "status": gate_status,
                        "changed": gate_changed,
                        "pending_target": gate_pending_target,
                        "pending_count": gate_pending_count,
                        "exec_pos": selected_action_class,
                        "flip_abs_threshold": float(cfg.exec_flip_abs_threshold),
                        "flip_blocked_by_threshold": bool(flip_blocked_by_threshold),
                    },
                    "timestamp": _ts_iso(bar15.get("timestamp")),
                    "close": _coerce_float(bar15.get("close"), float("nan")),
                    "agent_state": agent_state,
                },
            )

            if order_policies is None or symbol not in order_policies:
                event_key = _agent_action_event(prev_exec_action, selected_action_class)
                if event_key is not None:
                    trade_payload = {
                        "symbol": symbol,
                        "timestamp": _ts_iso(bar15.get("timestamp")),
                        "result": {
                            "event": event_key,
                            "source": "agent_action",
                            "prev_action": prev_exec_action,
                            "action": selected_action_class,
                            "raw_action_class": raw_action_pos,
                            "raw_action_class_model": model_raw_action_pos,
                            "action_raw": action_raw if np.isfinite(action_raw) else None,
                            "action_raw_exec": float(selected_action),
                            "execution_gate_status": gate_status,
                            "flip_abs_threshold": float(cfg.exec_flip_abs_threshold),
                            "flip_blocked_by_threshold": bool(flip_blocked_by_threshold),
                            "simulated": True,
                        },
                    }
                    self._store.add_trade_event(trade_payload)
                    self._emit("trade_event", trade_payload)
                return

            policy = order_policies[symbol]
            result = policy.on_decision(
                action=float(selected_action),
                closed_bar=bar_payload,
                update_bar_state=False,
                logger=self._policy_logger(symbol),
            )
            ts_iso = _ts_iso(bar15.get("timestamp"))
            _record_sim_orders(symbol, result, ts_iso)
            policy_state = policy.snapshot_state()
            self._store.set_policy_state(symbol, policy_state)
            policy_pos = int(policy_state.get("position", 0) or 0)
            self._store.set_last_action(
                symbol,
                action=float(policy_pos),
                action_class=policy_pos,
                ts=bar15.get("timestamp"),
                close=bar15.get("close"),
                record_event=False,
            )
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
            proxy = replay_option_proxies.get(str(bar["symbol"]).upper())
            if proxy is not None:
                proxy.update_bar(str(bar["symbol"]).upper(), bar)
            processor.handle_bar(bar)
            count += 1
            if count % 500 == 0:
                self._emit_status(running=True, message=f"replay processing {count:,}/{total_rows:,} bars")
            if cfg.replay_sleep > 0.0 and stop_event.wait(timeout=float(cfg.replay_sleep)):
                break

        self._emit(
            "log",
            {
                "symbol": "SYSTEM",
                "message": f"[replay] done. bars processed: {count:,}.",
            },
        )
        if can_use_snapshot_cache and not stop_event.is_set():
            try:
                _persist_replay_snapshot_cache(
                    replay_snapshot_cache_path,
                    cfg=cfg,
                    state=self._store.snapshot(),
                )
                self._store.add_log_event(
                    {
                        "symbol": "SYSTEM",
                        "message": f"[replay] saved dashboard replay snapshot: {replay_snapshot_cache_path}",
                    }
                )
                self._emit(
                    "log",
                    {
                        "symbol": "SYSTEM",
                        "message": f"[replay] saved dashboard replay snapshot: {replay_snapshot_cache_path}",
                    },
                )
            except Exception as exc:
                self._store.add_log_event(
                    {
                        "symbol": "SYSTEM",
                        "message": f"[replay] replay snapshot cache save failed: {exc}",
                    }
                )
                self._emit(
                    "log",
                    {
                        "symbol": "SYSTEM",
                        "message": f"[replay] replay snapshot cache save failed: {exc}",
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
        audit_writer: LiveAuditWriter | None = None

        try:
            if bool(cfg.audit_enabled):
                started_at = datetime.now(timezone.utc)
                session_stamp = started_at.strftime("%Y%m%d_%H%M%S")
                symbols_slug = "-".join(_slugify_token(sym, default="sym") for sym in (cfg.symbols or ["SPY"]))
                mode_slug = _slugify_token(cfg.runner_mode, default="live")
                session_dir = Path(cfg.audit_root or _default_live_audit_root()) / f"{session_stamp}_{mode_slug}_{symbols_slug}"
                audit_writer = LiveAuditWriter(session_dir)
                audit_writer.start(
                    metadata={
                        "started_at": _ts_iso(started_at),
                        "runner_mode": cfg.runner_mode,
                        "symbols": list(cfg.symbols),
                        "config": cfg.__dict__,
                    }
                )
                self._audit_writer = audit_writer
                self._audit(
                    "session",
                    {
                        "event": "session_started",
                        "started_at": _ts_iso(started_at),
                        "runner_mode": cfg.runner_mode,
                        "symbols": list(cfg.symbols),
                        "audit_dir": str(session_dir),
                    },
                )
                self._emit(
                    "log",
                    {
                        "symbol": "SYSTEM",
                        "message": f"[live] audit trail enabled: {session_dir}",
                    },
                )
            mode = str(cfg.runner_mode or "live").strip().lower()
            if mode == "replay":
                self._run_replay(cfg=cfg, stop_event=stop_event)
                return
            if mode != "live":
                raise ValueError(f"Unknown runner_mode: {cfg.runner_mode}")

            from API.Alpaca_API.inference.live_inference import LiveInferenceEngine
            from API.Alpaca_API.market_data.live_stream import AlpacaBarStreamer
            from API.Alpaca_API.runners import live_runner as lr
            from Policy.order_policy import OptionOrderPolicy

            symbols = cfg.symbols
            last_exec_action_by_symbol: dict[str, int] = {}
            execution_latches: dict[str, DirectionExecutionLatch] = {
                symbol: DirectionExecutionLatch(
                    entry_confirm_bars=int(cfg.exec_entry_confirm_bars),
                    exit_confirm_bars=int(cfg.exec_exit_confirm_bars),
                    initial_position=0,
                )
                for symbol in symbols
            }
            feed = lr._parse_feed(cfg.feed)
            bar_queue: queue_mod.Queue = queue_mod.Queue(maxsize=cfg.queue_size)
            precomputed_meta_frame = None
            inference_mode = "none" if cfg.no_agent else str(cfg.inference_mode or "meta").strip().lower()
            live_vix_symbol = "VIXY"
            live_vix_enabled = inference_mode != "none"
            vix_runtime_cache_path = lr._default_runtime_vix_cache_path()
            vix_runtime_cache_df: pd.DataFrame | None = None
            live_context_symbols: list[str] = []
            context_runtime_cache_by_symbol: dict[str, pd.DataFrame | None] = {}
            context_processors: dict[str, lr.LiveBarProcessor] = {}

            def _build_live_context_status(target_ts: object | None = None) -> dict[str, dict[str, object]]:
                target = pd.to_datetime(target_ts, utc=True, errors="coerce") if target_ts is not None else pd.NaT
                interval_minutes = max(1, int(cfg.interval or 10))
                soft_lag = pd.Timedelta(minutes=interval_minutes)
                hard_lag = pd.Timedelta(minutes=interval_minutes * 3)
                out: dict[str, dict[str, object]] = {}

                def _status_from_df(df: pd.DataFrame | None) -> dict[str, object]:
                    if df is None or df.empty or "timestamp" not in df.columns:
                        return {"status": "missing", "last_ts": None, "rows": 0, "lag_min": None}
                    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
                    if ts.empty:
                        return {"status": "missing", "last_ts": None, "rows": int(len(df)), "lag_min": None}
                    last_ts = ts.max()
                    lag_min: float | None = None
                    if pd.isna(target):
                        label = "present"
                    else:
                        lag = target - last_ts
                        lag_min = max(0.0, lag.total_seconds() / 60.0)
                        if last_ts >= target or lag <= soft_lag:
                            label = "present"
                        elif lag <= hard_lag:
                            label = "lagging"
                        else:
                            label = "stale"
                    return {
                        "status": label,
                        "last_ts": _ts_iso(last_ts),
                        "rows": int(len(df)),
                        "lag_min": lag_min,
                    }

                if live_vix_enabled:
                    out[live_vix_symbol] = _status_from_df(vix_runtime_cache_df)
                for symbol in live_context_symbols:
                    out[symbol] = _status_from_df(context_runtime_cache_by_symbol.get(symbol))
                if "QQQ" in out and "IWM" in out:
                    qqq_status = str(out["QQQ"].get("status") or "").lower()
                    iwm_status = str(out["IWM"].get("status") or "").lower()
                    statuses = {qqq_status, iwm_status}
                    breadth_last = min(
                        [ts for ts in (out["QQQ"].get("last_ts"), out["IWM"].get("last_ts")) if ts],
                        default=None,
                    )
                    breadth_rows = min(
                        [
                            int(val)
                            for val in (out["QQQ"].get("rows"), out["IWM"].get("rows"))
                            if isinstance(val, (int, float))
                        ],
                        default=0,
                    )
                    breadth_lag = max(
                        [
                            float(val)
                            for val in (out["QQQ"].get("lag_min"), out["IWM"].get("lag_min"))
                            if isinstance(val, (int, float)) and math.isfinite(float(val))
                        ],
                        default=None,
                    )
                    if statuses <= {"present"}:
                        breadth_status = "present"
                    elif statuses <= {"present", "lagging"}:
                        breadth_status = "lagging"
                    else:
                        breadth_status = "stale"
                    out["BREADTH_PROXY"] = {
                        "status": breadth_status,
                        "last_ts": breadth_last,
                        "rows": breadth_rows,
                        "lag_min": breadth_lag,
                    }
                return out

            agent: LivePPOAgent | LiveMetaXGBAgent | LiveIndependentMetaXGBAgent | None = None
            if inference_mode != "none":
                ga_feature_list = self._resolve_ga_feature_list(cfg)
                ga_probs_frame = _load_agent_matrix_probs(
                    symbol=symbols[0],
                    dataset_name=cfg.ga_dataset_name,
                )
                if cfg.eval_parity_mode and inference_mode == "ppo" and ga_probs_frame is not None and ga_feature_list:
                    ga_probs_mode = "hybrid"
                elif cfg.eval_parity_mode and inference_mode == "ppo" and ga_probs_frame is not None:
                    ga_probs_mode = "frame"
                elif ga_probs_frame is not None and ga_feature_list:
                    ga_probs_mode = "hybrid"
                elif ga_probs_frame is not None:
                    ga_probs_mode = "frame"
                else:
                    ga_probs_mode = "xgb"
                if cfg.eval_parity_mode and inference_mode == "ppo" and ga_probs_frame is None:
                    self._emit(
                        "log",
                        {
                            "symbol": "SYSTEM",
                            "message": "[live] eval parity mode requested, but no agent_matrix probs were found; falling back to XGB probs.",
                        },
                    )
                if ga_probs_frame is not None:
                    try:
                        src = ga_probs_frame.attrs.get("source_paths", [])
                        rng_min = ga_probs_frame.index.min()
                        rng_max = ga_probs_frame.index.max()
                        self._emit(
                            "log",
                            {
                                "symbol": "SYSTEM",
                                "message": (
                                    "[live] loaded agent_matrix probs "
                                    f"(rows={len(ga_probs_frame):,}, "
                                    f"range={_ts_iso(rng_min)}..{_ts_iso(rng_max)}, "
                                    f"sources={len(src)}, mode={ga_probs_mode})"
                                ),
                            },
                        )
                    except Exception:
                        pass
                    if cfg.eval_parity_mode and inference_mode == "ppo" and ga_probs_mode == "frame":
                        self._emit(
                            "log",
                            {
                                "symbol": "SYSTEM",
                                "message": "[live] eval parity mode using frame-only probs (no GA feature list for XGB fallback).",
                            },
                        )
                if inference_mode == "meta":
                    if cfg.meta_base_frame_path:
                        try:
                            meta_base_path = lr._resolve_symbolized_path(cfg.meta_base_frame_path, symbol=symbols[0])
                            precomputed_meta_frame = lr._load_precomputed_meta_frame(
                                meta_base_path,
                                tz=cfg.tz or "America/New_York",
                            )
                            if not precomputed_meta_frame.empty:
                                self._emit(
                                    "log",
                                    {
                                        "symbol": "SYSTEM",
                                        "message": (
                                            f"[live] Loaded cached context base frame: rows={len(precomputed_meta_frame):,} "
                                            f"range={_ts_iso(precomputed_meta_frame.index.min())}..{_ts_iso(precomputed_meta_frame.index.max())} "
                                            f"path={meta_base_path}"
                                        ),
                                    },
                                )
                        except Exception as exc:
                            self._emit(
                                "log",
                                {
                                    "symbol": "SYSTEM",
                                    "message": f"[live] Cached context base frame unavailable: {exc}",
                                },
                            )
                    agent = lr._build_meta_agent(
                        symbol=symbols[0],
                        model_root=cfg.meta_model_root,
                        ga_model_root=cfg.ga_model_root,
                        ga_feature_list_path=ga_feature_list,
                        ga_probs_frame=ga_probs_frame,
                        ga_probs_mode=ga_probs_mode,
                        include_pivot_probs=not cfg.no_pivot_probs,
                        include_tb_probs=not cfg.no_tb_probs,
                        pivot_label_dir=cfg.ga_pivot_label_dir,
                        tb_label_dir=cfg.ga_tb_label_dir,
                        tz=cfg.tz or "America/New_York",
                        assume_tz=cfg.assume_tz,
                        session_open=cfg.session_open,
                        session_close=cfg.session_close,
                        min_15m_bars=cfg.min_15m_bars,
                        fill_missing_prob=cfg.fill_missing_prob,
                        resample_label=cfg.resample_label,
                        resample_closed=cfg.resample_closed,
                        label_timeframe_rule=f"{cfg.interval}min",
                        trail_activate_atr=float(cfg.meta_trail_activate_atr),
                        trail_atr=float(cfg.meta_trail_atr),
                        trail_atr_after_tp=float(cfg.meta_trail_atr_after_tp),
                        use_tp_to_tighten_trail=bool(cfg.meta_use_tp_to_tighten_trail),
                        entry_threshold_override=cfg.meta_entry_threshold,
                        exit_threshold_override=cfg.meta_exit_threshold,
                        precomputed_base_frame=precomputed_meta_frame,
                        precomputed_append_lookback_days=int(cfg.meta_base_frame_append_lookback_days),
                        min_hold_bars=2,
                        exit_entry_delta=0.15,
                        soft_exit_confirm_bars=2,
                        urgent_exit_prob=0.85,
                        urgent_exit_delta=0.30,
                        profit_protect_enabled=False,
                        profit_protect_arm_atr=2.0,
                        profit_protect_giveback_atr_long=0.75,
                        profit_protect_giveback_atr_short=1.0,
                        entry_prob_source=cfg.meta_entry_prob_source,
                        swing_setup_single_model_dir=cfg.swing_setup_single_model_dir,
                    )
                    self._emit(
                        "log",
                        {
                            "symbol": "SYSTEM",
                            "message": (
                                (
                                    f"[live] Swing setup wrapper enabled: model_dir={cfg.swing_setup_single_model_dir} "
                                    f"timeframe={cfg.interval}min"
                                )
                                if str(cfg.meta_entry_prob_source or "").strip().lower() == "swing_support_single"
                                else (
                                    f"[live] Legacy probability wrapper enabled: model_root={cfg.meta_model_root} "
                                    f"entry_source={cfg.meta_entry_prob_source} timeframe={cfg.interval}min"
                                )
                            ),
                        },
                    )
                else:
                    agent = lr._build_ppo_agent(
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
                        ga_model_root=cfg.ga_model_root,
                        ga_feature_list_path=ga_feature_list,
                        ga_pivot_label_dir=cfg.ga_pivot_label_dir,
                        ga_tb_label_dir=cfg.ga_tb_label_dir,
                        ga_probs_frame=ga_probs_frame,
                        ga_probs_mode=ga_probs_mode,
                        require_probs=True,
                        resample_label=cfg.resample_label,
                        resample_closed=cfg.resample_closed,
                        label_timeframe_rule=f"{cfg.interval}min",
                    )
                initial_agent_state = agent.snapshot_state()
                if isinstance(initial_agent_state, dict):
                    initial_agent_state = dict(initial_agent_state)
                    initial_agent_state["last_context_status"] = _build_live_context_status()
                self._store.set_agent_state(initial_agent_state)
                self._emit("agent_state", {"state": initial_agent_state})

            inference = LiveInferenceEngine(
                agent=agent,
                label=cfg.resample_label,
                closed=cfg.resample_closed,
                rule=f"{cfg.interval}min",
                tz=cfg.tz,
                assume_tz=cfg.assume_tz,
            )

            order_policies: dict[str, OptionOrderPolicy] | None = None
            runtime_policy_cache_path: Path | None = None
            runtime_policy_cache: dict[str, Any] = {}
            if cfg.enable_option_orders:
                order_policies = {}
                for symbol in symbols:
                    policy = lr._build_option_order_policy(
                        symbol=symbol,
                        env_file=cfg.env_file,
                        tz_name=cfg.tz,
                        atr_multiplier=float(cfg.option_atr_mult),
                        dte_cutoff_hhmm=cfg.option_dte_cutoff,
                        qty=int(cfg.option_order_qty),
                        close_on_flat=not cfg.option_no_close_on_flat,
                        close_on_flip=not cfg.option_no_close_on_flip,
                        submit_orders=not cfg.simulate_orders,
                        opposite_confirm_bars=1,
                        opposite_min_abs_action=float(cfg.exec_flip_abs_threshold),
                        opposite_min_prob_edge=0.0,
                        meta_execute_on_interval_close=False,
                        meta_intrabar_execution_enabled=True,
                        meta_intrabar_breakout_entry_only=True,
                        meta_intrabar_entry_policy=str(cfg.meta_intrabar_entry_policy),
                        meta_intrabar_setup_max_bars=int(cfg.meta_intrabar_setup_max_bars),
                        meta_intrabar_setup_bar_minutes=int(cfg.interval),
                        meta_intrabar_long_setup_threshold=cfg.meta_intrabar_long_setup_threshold,
                        meta_intrabar_short_setup_threshold=cfg.meta_intrabar_short_setup_threshold,
                        meta_hard_stop_atr=float(cfg.meta_hard_stop_atr),
                        meta_trail_activate_atr=float(cfg.meta_trail_activate_atr),
                        meta_trail_atr=float(cfg.meta_trail_atr),
                        meta_trail_atr_after_tp=float(cfg.meta_trail_atr_after_tp),
                        meta_use_tp_to_tighten_trail=bool(cfg.meta_use_tp_to_tighten_trail),
                        meta_soft_exit_confirm_bars=2,
                        meta_urgent_exit_prob=0.85,
                        meta_urgent_exit_delta=0.30,
                        meta_profit_protect_enabled=False,
                        meta_profit_protect_arm_atr=2.0,
                        meta_profit_protect_giveback_atr_long=0.75,
                        meta_profit_protect_giveback_atr_short=1.0,
                        option_exit_policy=str(cfg.option_exit_policy),
                        option_exit_take_profit_pct=float(cfg.option_exit_take_profit_pct),
                        option_exit_stop_loss_pct=float(cfg.option_exit_stop_loss_pct),
                        option_exit_profit_lock_arm_pct=float(cfg.option_exit_profit_lock_arm_pct),
                        option_exit_profit_lock_floor_pct=float(cfg.option_exit_profit_lock_floor_pct),
                        option_exit_trailing_arm_pct=float(cfg.option_exit_trailing_arm_pct),
                        option_exit_trailing_giveback_pct=float(cfg.option_exit_trailing_giveback_pct),
                        option_exit_time_decay_minutes=int(cfg.option_exit_time_decay_minutes),
                        option_exit_time_decay_progress_pct=float(cfg.option_exit_time_decay_progress_pct),
                        option_exit_opposite_prob=float(cfg.option_exit_opposite_prob),
                        option_exit_quote_mode=str(cfg.option_exit_quote_mode),
                    )
                    order_policies[symbol] = policy
                    self._store.set_policy_state(symbol, policy.snapshot_state())
                    broker_snap = policy.snapshot_broker_state(orders_limit=20)
                    self._store.set_broker_state(symbol, broker_snap)
                    self._emit("broker_state", {"symbol": symbol, "state": broker_snap})

                cache_base = Path(cfg.prefill_path or cfg.replay_data_path or "Data/raw/spy/live_runtime_policy_state.json")
                runtime_policy_cache_path = _default_runtime_policy_cache_path(cache_base)
                runtime_policy_cache = _load_runtime_policy_cache(runtime_policy_cache_path)

                def _persist_policy_runtime_cache() -> None:
                    if order_policies is None or runtime_policy_cache_path is None:
                        return
                    payload = {
                        "symbols": list(symbols),
                        "updated_at": _ts_iso(datetime.now(timezone.utc)),
                        "policies": {
                            sym: pol.export_runtime_state()
                            for sym, pol in order_policies.items()
                        },
                    }
                    _persist_runtime_policy_cache(runtime_policy_cache_path, payload)

                if not cfg.no_startup_sync:
                    for symbol in symbols:
                        policy = order_policies[symbol]
                        sync_result = policy.sync_from_broker(logger=self._policy_logger(symbol))
                        if cfg.use_execution_latch:
                            execution_latches[symbol].set_position(policy.snapshot_state().get("position", 0))
                        self._store.set_policy_state(symbol, policy.snapshot_state())
                        _persist_policy_runtime_cache()
                        self._emit("startup_sync", {"symbol": symbol, "result": sync_result})
                        self._audit("startup_sync", {"symbol": symbol, "result": sync_result})
                        broker_snap = policy.snapshot_broker_state(orders_limit=20)
                        self._store.set_broker_state(symbol, broker_snap)
                        self._emit("broker_state", {"symbol": symbol, "state": broker_snap})
                        self._audit("broker_state", {"symbol": symbol, "state": broker_snap})

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
                if order_policies is None or symbol not in order_policies:
                    return
                policy = order_policies[symbol]
                result = policy.on_1m_bar(bar=bar, logger=self._policy_logger(symbol))
                policy_state = policy.snapshot_state()
                self._store.set_policy_state(symbol, policy_state)
                _persist_policy_runtime_cache()
                broker_snap = policy.snapshot_broker_state(orders_limit=20)
                self._store.set_broker_state(symbol, broker_snap)
                self._emit("broker_state", {"symbol": symbol, "state": broker_snap})
                self._audit("broker_state", {"symbol": symbol, "state": broker_snap})
                event_payload = {
                    "symbol": symbol,
                    "timestamp": _ts_iso(bar.get("timestamp")),
                    "result": result,
                    "policy_state": policy_state,
                    "broker_state": broker_snap,
                }
                self._emit("order_policy", event_payload)
                event_key = str(result.get("event", "")).strip().lower()
                if event_key not in {"hold", "no_change", "intent_update"}:
                    self._audit("order_policy", event_payload)
                if event_key not in {"hold", "no_change", "intent_update"}:
                    self._store.set_last_action(
                        symbol,
                        action=float(policy_state.get("position", 0) or 0),
                        action_class=int(policy_state.get("position", 0) or 0),
                        ts=bar.get("timestamp"),
                        close=bar.get("close"),
                        record_event=False,
                    )
                    self._store.add_trade_event(event_payload)
                    self._audit("trade_events", event_payload)

            def _on_15m(symbol: str, bar15: dict, buffer: Any) -> None:
                if order_policies is not None and symbol in order_policies:
                    policy = order_policies[symbol]
                    policy.on_15m_bar(closed_bar=bar15)
                    self._store.set_policy_state(symbol, policy.snapshot_state())
                    _persist_policy_runtime_cache()

                action = inference.on_15m_close(df_1m=buffer.to_dataframe(), closed_bar=bar15)
                if action is None:
                    self._store.add_15m_bar(symbol, bar15)
                    self._emit("bar_15m", {"symbol": symbol, "bar": _normalize_bar(bar15)})
                    return

                prev_exec_action = last_exec_action_by_symbol.get(symbol)
                action_raw = _coerce_float(action, float("nan"))
                gated_action_raw, model_raw_action_pos, raw_action_pos, flip_blocked_by_threshold = _apply_flip_abs_threshold(
                    action_raw=action_raw,
                    prev_exec_action=prev_exec_action,
                    flip_abs_threshold=float(cfg.exec_flip_abs_threshold),
                )
                if cfg.eval_parity_mode:
                    selected_action = (
                        gated_action_raw
                        if np.isfinite(gated_action_raw)
                        else float(raw_action_pos)
                    )
                    selected_action_class = raw_action_pos
                    gate_status = "bypassed_parity_mode"
                    gate_changed = (
                        prev_exec_action is not None
                        and int(prev_exec_action) != int(selected_action_class)
                    )
                    gate_pending_target: int | None = None
                    gate_pending_count = 0
                elif cfg.use_execution_latch:
                    gate = execution_latches[symbol].step(raw_action_pos)
                    selected_action_class = int(gate.executed_pos)
                    selected_action = float(selected_action_class)
                    gate_status = str(gate.status)
                    gate_changed = bool(gate.changed)
                    gate_pending_target = gate.pending_target
                    gate_pending_count = int(gate.pending_count)
                else:
                    selected_action = (
                        gated_action_raw
                        if np.isfinite(gated_action_raw)
                        else float(raw_action_pos)
                    )
                    selected_action_class = raw_action_pos
                    gate_status = "bypassed_latch_disabled"
                    gate_changed = (
                        prev_exec_action is not None
                        and int(prev_exec_action) != int(selected_action_class)
                    )
                    gate_pending_target = None
                    gate_pending_count = 0
                probs = agent.last_probs() if agent is not None else None
                thresholds = inference.last_thresholds() if inference is not None else None
                bar_payload = dict(bar15)
                if probs:
                    bar_payload.update({k: v for k, v in probs.items() if v is not None})
                if thresholds:
                    bar_payload.update(
                        {
                            "thr_enter_long": thresholds.get("enter_long"),
                            "thr_enter_short": thresholds.get("enter_short"),
                            "thr_exit_long": thresholds.get("exit_long"),
                            "thr_exit_short": thresholds.get("exit_short"),
                        }
                    )
                self._store.add_15m_bar(symbol, bar_payload)
                self._emit("bar_15m", {"symbol": symbol, "bar": _normalize_bar(bar_payload)})
                self._store.set_last_action(
                    symbol,
                    action=float(selected_action),
                    action_class=selected_action_class,
                    ts=bar15.get("timestamp"),
                    close=bar15.get("close"),
                )
                last_exec_action_by_symbol[symbol] = selected_action_class

                agent_state = agent.snapshot_state() if agent is not None else None
                if agent_state is not None and thresholds:
                    agent_state = dict(agent_state)
                    agent_state["last_thresholds"] = thresholds
                if agent_state is not None:
                    agent_state = dict(agent_state)
                    agent_state["last_context_status"] = _build_live_context_status(bar15.get("timestamp"))
                    self._store.set_agent_state(agent_state)
                self._emit(
                    "action",
                    {
                        "symbol": symbol,
                        "action": float(selected_action),
                        "action_class": selected_action_class,
                        "action_raw": action_raw if np.isfinite(action_raw) else None,
                        "action_raw_exec": float(selected_action),
                        "raw_action_class": raw_action_pos,
                        "raw_action_class_model": model_raw_action_pos,
                        "execution_gate": {
                            "status": gate_status,
                            "changed": gate_changed,
                            "pending_target": gate_pending_target,
                            "pending_count": gate_pending_count,
                            "exec_pos": selected_action_class,
                            "flip_abs_threshold": float(cfg.exec_flip_abs_threshold),
                            "flip_blocked_by_threshold": bool(flip_blocked_by_threshold),
                        },
                        "timestamp": _ts_iso(bar15.get("timestamp")),
                        "close": _coerce_float(bar15.get("close"), float("nan")),
                        "agent_state": agent_state,
                    },
                )
                decision_context = {
                    "symbol": symbol,
                    "timestamp": _ts_iso(bar15.get("timestamp")),
                    "bar": _normalize_bar(bar_payload),
                    "action": {
                        "selected_action": float(selected_action),
                        "selected_action_class": int(selected_action_class),
                        "action_raw": action_raw if np.isfinite(action_raw) else None,
                        "raw_action_class": raw_action_pos,
                        "raw_action_class_model": model_raw_action_pos,
                        "gate_status": gate_status,
                        "gate_changed": bool(gate_changed),
                        "gate_pending_target": gate_pending_target,
                        "gate_pending_count": int(gate_pending_count),
                        "flip_abs_threshold": float(cfg.exec_flip_abs_threshold),
                        "flip_blocked_by_threshold": bool(flip_blocked_by_threshold),
                    },
                    "agent_state": agent_state,
                }
                self._audit("actions", decision_context)

                if order_policies is None or symbol not in order_policies:
                    event_key = _agent_action_event(prev_exec_action, selected_action_class)
                    if event_key is not None:
                        trade_payload = {
                            "symbol": symbol,
                            "timestamp": _ts_iso(bar15.get("timestamp")),
                            "result": {
                                "event": event_key,
                                "source": "agent_action",
                                "prev_action": prev_exec_action,
                                "action": selected_action_class,
                                "raw_action_class": raw_action_pos,
                                "raw_action_class_model": model_raw_action_pos,
                                "action_raw": action_raw if np.isfinite(action_raw) else None,
                                "action_raw_exec": float(selected_action),
                                "execution_gate_status": gate_status,
                                "flip_abs_threshold": float(cfg.exec_flip_abs_threshold),
                                "flip_blocked_by_threshold": bool(flip_blocked_by_threshold),
                                "simulated": True,
                            },
                        }
                        self._store.add_trade_event(trade_payload)
                        self._emit("trade_event", trade_payload)
                        self._audit("trade_events", trade_payload)
                    self._audit("decision_10m", decision_context)
                    return

                policy = order_policies[symbol]
                result = policy.on_decision(
                    action=float(selected_action),
                    closed_bar=bar_payload,
                    update_bar_state=False,
                    logger=self._policy_logger(symbol),
                )
                policy_state = policy.snapshot_state()
                self._store.set_policy_state(symbol, policy_state)
                _persist_policy_runtime_cache()
                broker_snap = policy.snapshot_broker_state(orders_limit=20)
                self._store.set_broker_state(symbol, broker_snap)
                self._audit("broker_state", {"symbol": symbol, "state": broker_snap})
                policy_pos = int(policy_state.get("position", 0) or 0)
                self._store.set_last_action(
                    symbol,
                    action=float(policy_pos),
                    action_class=policy_pos,
                    ts=bar15.get("timestamp"),
                    close=bar15.get("close"),
                    record_event=False,
                )
                event_payload = {
                    "symbol": symbol,
                    "timestamp": _ts_iso(bar15.get("timestamp")),
                    "result": result,
                    "policy_state": policy_state,
                    "broker_state": broker_snap,
                }
                self._emit("order_policy", event_payload)
                self._emit("broker_state", {"symbol": symbol, "state": broker_snap})
                self._audit("order_policy", event_payload)
                event_key = str(result.get("event", "")).strip().lower()
                if event_key not in {"hold", "no_change"}:
                    self._store.add_trade_event(event_payload)
                    self._audit("trade_events", event_payload)
                self._audit(
                    "decision_10m",
                    {
                        **decision_context,
                        "policy_result": result,
                        "policy_state": policy_state,
                        "broker_state": broker_snap,
                    },
                )

            processor = lr.LiveBarProcessor(
                interval_minutes=cfg.interval,
                buffer_size=cfg.buffer_size,
                agg_label=cfg.resample_label,
                on_1m=_on_1m,
                on_15m_close=_on_15m,
                regular_hours_only=True,
                tz_name=cfg.tz or "America/New_York",
                session_open=cfg.session_open,
                session_close=cfg.session_close,
            )

            def _log_vix_runtime_cache(message_prefix: str) -> None:
                if vix_runtime_cache_df is None or vix_runtime_cache_df.empty:
                    self._emit(
                        "log",
                        {
                            "symbol": "SYSTEM",
                            "message": f"[live] {message_prefix}: no VIX runtime cache rows available.",
                        },
                    )
                    return
                ts = pd.to_datetime(vix_runtime_cache_df["timestamp"], utc=True, errors="coerce").dropna()
                if ts.empty:
                    self._emit(
                        "log",
                        {
                            "symbol": "SYSTEM",
                            "message": f"[live] {message_prefix}: VIX runtime cache has no valid timestamps.",
                        },
                    )
                    return
                self._emit(
                    "log",
                    {
                        "symbol": "SYSTEM",
                        "message": (
                            f"[live] {message_prefix}: {vix_runtime_cache_path} "
                            f"rows={len(vix_runtime_cache_df):,} "
                            f"range={_ts_iso(ts.min())}..{_ts_iso(ts.max())}"
                        ),
                    },
                )

            def _log_context_runtime_cache(message_prefix: str, symbol: str) -> None:
                cache_df = context_runtime_cache_by_symbol.get(symbol)
                cache_path = lr._default_runtime_context_cache_path(symbol)
                if cache_df is None or cache_df.empty:
                    self._emit(
                        "log",
                        {
                            "symbol": "SYSTEM",
                            "message": f"[live] {message_prefix} {symbol}: no runtime cache rows available.",
                        },
                    )
                    return
                ts = pd.to_datetime(cache_df["timestamp"], utc=True, errors="coerce").dropna()
                if ts.empty:
                    self._emit(
                        "log",
                        {
                            "symbol": "SYSTEM",
                            "message": f"[live] {message_prefix} {symbol}: runtime cache has no valid timestamps.",
                        },
                    )
                    return
                self._emit(
                    "log",
                    {
                        "symbol": "SYSTEM",
                        "message": (
                            f"[live] {message_prefix} {symbol}: {cache_path} "
                            f"rows={len(cache_df):,} range={_ts_iso(ts.min())}..{_ts_iso(ts.max())}"
                        ),
                    },
                )

            def _on_vix_interval(symbol: str, bar_tf: dict, _buffer: Any) -> None:
                nonlocal vix_runtime_cache_df
                bar_df = pd.DataFrame(
                    [
                        {
                            "timestamp": pd.to_datetime(bar_tf.get("timestamp"), utc=True, errors="coerce"),
                            "open": bar_tf.get("open"),
                            "high": bar_tf.get("high"),
                            "low": bar_tf.get("low"),
                            "close": bar_tf.get("close"),
                            "volume": bar_tf.get("volume"),
                            "symbol": symbol,
                        }
                    ]
                )
                if vix_runtime_cache_df is None or vix_runtime_cache_df.empty:
                    merged_vix = bar_df
                else:
                    merged_vix = pd.concat([vix_runtime_cache_df, bar_df], axis=0, ignore_index=True)
                vix_runtime_cache_df = lr._prepare_runtime_vix_frame(
                    df=merged_vix,
                    tz_name=cfg.tz or "America/New_York",
                    session_open=cfg.session_open,
                    session_close=cfg.session_close,
                )
                try:
                    lr._persist_runtime_vix_cache(
                        df=vix_runtime_cache_df,
                        cache_path=vix_runtime_cache_path,
                    )
                except Exception as exc:
                    self._emit(
                        "log",
                        {
                            "symbol": "SYSTEM",
                            "message": f"[live] runtime VIX cache update failed: {exc}",
                        },
                    )

            def _make_context_interval_handler(symbol: str, cache_path: Path):
                def _handler(_symbol: str, bar_tf: dict, _buffer: Any) -> None:
                    current_df = context_runtime_cache_by_symbol.get(symbol)
                    bar_df = pd.DataFrame(
                        [
                            {
                                "timestamp": pd.to_datetime(bar_tf.get("timestamp"), utc=True, errors="coerce"),
                                "open": bar_tf.get("open"),
                                "high": bar_tf.get("high"),
                                "low": bar_tf.get("low"),
                                "close": bar_tf.get("close"),
                                "volume": bar_tf.get("volume"),
                                "symbol": symbol,
                            }
                        ]
                    )
                    merged_df = bar_df if current_df is None or current_df.empty else pd.concat([current_df, bar_df], axis=0, ignore_index=True)
                    prepared_df = lr._prepare_runtime_context_frame(
                        df=merged_df,
                        symbol=symbol,
                        tz_name=cfg.tz or "America/New_York",
                        session_open=cfg.session_open,
                        session_close=cfg.session_close,
                    )
                    context_runtime_cache_by_symbol[symbol] = prepared_df
                    try:
                        lr._persist_runtime_context_cache(
                            df=prepared_df,
                            cache_path=cache_path,
                        )
                    except Exception as exc:
                        self._emit(
                            "log",
                            {
                                "symbol": "SYSTEM",
                                "message": f"[live] runtime cache update failed for {symbol}: {exc}",
                            },
                        )

                return _handler

            vix_processor = (
                lr.LiveBarProcessor(
                    interval_minutes=cfg.interval,
                    buffer_size=2048,
                    agg_label=cfg.resample_label,
                    on_15m_close=_on_vix_interval,
                    regular_hours_only=True,
                    tz_name=cfg.tz or "America/New_York",
                    session_open=cfg.session_open,
                    session_close=cfg.session_close,
                )
                if live_vix_enabled
                else None
            )
            self._emit(
                "log",
                {
                    "symbol": "SYSTEM",
                    "message": (
                        f"[live] RTH-only live bar filter enabled: "
                        f"{cfg.session_open}-{cfg.session_close} {cfg.tz or 'America/New_York'}"
                    ),
                },
            )

            if live_vix_enabled:
                vix_source_path = vix_runtime_cache_path if vix_runtime_cache_path.exists() else (Path("Data") / "raw" / "vix" / "vixy_10min.parquet")
                try:
                    vix_runtime_cache_df = lr._load_runtime_vix_frame(vix_source_path)
                except Exception as exc:
                    self._emit(
                        "log",
                        {
                            "symbol": "SYSTEM",
                            "message": f"[live] VIX runtime cache load failed from {vix_source_path}: {exc}",
                        },
                    )
                    vix_runtime_cache_df = None
                if vix_runtime_cache_df is not None and not vix_runtime_cache_df.empty and not cfg.no_prefill_fetch:
                    try:
                        vix_runtime_cache_df = lr._extend_vix_with_alpaca_gap(
                            df=vix_runtime_cache_df,
                            feed=feed,
                            ticker=live_vix_symbol,
                            timeframe=f"{int(cfg.interval)}Min",
                        )
                    except Exception as exc:
                        self._emit(
                            "log",
                            {
                                "symbol": "SYSTEM",
                                "message": f"[live] VIX gap bridge failed: {exc}",
                            },
                        )
                if vix_runtime_cache_df is not None and not vix_runtime_cache_df.empty:
                    vix_runtime_cache_df = lr._prepare_runtime_vix_frame(
                        df=vix_runtime_cache_df,
                        tz_name=cfg.tz or "America/New_York",
                        session_open=cfg.session_open,
                        session_close=cfg.session_close,
                    )
                    try:
                        lr._persist_runtime_vix_cache(
                            df=vix_runtime_cache_df,
                            cache_path=vix_runtime_cache_path,
                        )
                        _log_vix_runtime_cache("prepared runtime VIX cache")
                    except Exception as exc:
                        self._emit(
                        "log",
                        {
                            "symbol": "SYSTEM",
                            "message": f"[live] runtime VIX cache persist failed: {exc}",
                        },
                    )

                live_context_symbols = []
                for context_symbol in lr.META_CONTEXT_SYMBOLS:
                    clean = str(context_symbol).upper()
                    if clean in symbols or clean in live_context_symbols:
                        continue
                    live_context_symbols.append(clean)

                for context_symbol in live_context_symbols:
                    cache_path = lr._default_runtime_context_cache_path(context_symbol)
                    source_path = cache_path if cache_path.exists() else lr._default_static_context_cache_path(context_symbol)
                    try:
                        context_df = lr._load_runtime_context_frame(source_path, symbol=context_symbol)
                    except Exception as exc:
                        self._emit(
                            "log",
                            {
                                "symbol": "SYSTEM",
                                "message": f"[live] runtime cache load failed for {context_symbol} from {source_path}: {exc}",
                            },
                        )
                        context_df = None
                    if context_df is not None and not context_df.empty and not cfg.no_prefill_fetch:
                        try:
                            context_df = lr._extend_context_with_alpaca_gap(
                                df=context_df,
                                feed=feed,
                                ticker=context_symbol,
                                timeframe=f"{int(cfg.interval)}Min",
                            )
                        except Exception as exc:
                            self._emit(
                                "log",
                                {
                                    "symbol": "SYSTEM",
                                    "message": f"[live] runtime gap bridge failed for {context_symbol}: {exc}",
                                },
                            )
                    if context_df is not None and not context_df.empty:
                        prepared_context_df = lr._prepare_runtime_context_frame(
                            df=context_df,
                            symbol=context_symbol,
                            tz_name=cfg.tz or "America/New_York",
                            session_open=cfg.session_open,
                            session_close=cfg.session_close,
                        )
                        context_runtime_cache_by_symbol[context_symbol] = prepared_context_df
                        try:
                            lr._persist_runtime_context_cache(
                                df=prepared_context_df,
                                cache_path=cache_path,
                            )
                            _log_context_runtime_cache("prepared runtime cache", context_symbol)
                        except Exception as exc:
                            self._emit(
                                "log",
                                {
                                    "symbol": "SYSTEM",
                                    "message": f"[live] runtime cache persist failed for {context_symbol}: {exc}",
                                },
                            )
                    else:
                        context_runtime_cache_by_symbol[context_symbol] = context_df
                    context_processors[context_symbol] = lr.LiveBarProcessor(
                        interval_minutes=cfg.interval,
                        buffer_size=2048,
                        agg_label=cfg.resample_label,
                        on_15m_close=_make_context_interval_handler(context_symbol, cache_path),
                        regular_hours_only=True,
                        tz_name=cfg.tz or "America/New_York",
                        session_open=cfg.session_open,
                        session_close=cfg.session_close,
                    )

            if cfg.buffer_size and cfg.buffer_size > 0 and cfg.prefill_tail and cfg.prefill_tail > cfg.buffer_size:
                self._emit(
                    "log",
                    {
                        "symbol": "SYSTEM",
                        "message": "[live] prefill-tail exceeds buffer-size; oldest rows will be dropped.",
                    },
                )
            if cfg.prefill_path:
                self._emit_status(running=True, message="loading prefill file")
                configured_prefill_path = Path(cfg.prefill_path)
                runtime_prefill_cache_path = lr._default_runtime_prefill_cache_path(configured_prefill_path)
                prefill_source_path = runtime_prefill_cache_path if runtime_prefill_cache_path.exists() else configured_prefill_path
                if prefill_source_path == runtime_prefill_cache_path:
                    self._emit(
                        "log",
                        {
                            "symbol": "SYSTEM",
                            "message": f"[live] using runtime prefill cache: {runtime_prefill_cache_path}",
                        },
                    )
                self._emit(
                    "log",
                    {
                        "symbol": "SYSTEM",
                        "message": f"[live] loading prefill file: {prefill_source_path}",
                    },
                )
                prefill_df = lr._load_prefill_frame(prefill_source_path)
                if not cfg.no_prefill_fetch:
                    self._emit_status(running=True, message="gap-bridging prefill from Alpaca")
                    self._emit(
                        "log",
                        {
                            "symbol": "SYSTEM",
                            "message": "[live] gap-bridging prefill from Alpaca...",
                        },
                    )
                    try:
                        prefill_df = lr._extend_prefill_with_alpaca_gap(
                            df=prefill_df,
                            symbols=symbols,
                            feed=feed,
                        )
                    except Exception as exc:
                        self._emit(
                            "log",
                            {
                                "symbol": "SYSTEM",
                            "message": f"[live] Prefill gap bridge failed: {exc}",
                        },
                    )
                else:
                    self._emit(
                        "log",
                        {
                            "symbol": "SYSTEM",
                            "message": "[live] skipping Alpaca prefill fetch because no_prefill_fetch is enabled.",
                        },
                    )
                compact_prefill_df = lr._prepare_runtime_prefill_frame(
                    df=prefill_df,
                    precomputed_meta_frame=precomputed_meta_frame,
                    append_lookback_days=int(cfg.meta_base_frame_append_lookback_days),
                    tz_name=cfg.tz or "America/New_York",
                    session_open=cfg.session_open,
                    session_close=cfg.session_close,
                )
                try:
                    lr._persist_runtime_prefill_cache(
                        df=compact_prefill_df,
                        cache_path=runtime_prefill_cache_path,
                    )
                    self._emit(
                        "log",
                        {
                            "symbol": "SYSTEM",
                            "message": (
                                f"[live] updated runtime prefill cache: {runtime_prefill_cache_path} "
                                f"rows={len(compact_prefill_df):,}"
                            ),
                        },
                    )
                except Exception as exc:
                    self._emit(
                        "log",
                        {
                            "symbol": "SYSTEM",
                            "message": f"[live] runtime prefill cache update failed: {exc}",
                        },
                    )
                self._emit_status(running=True, message="seeding live buffers from prefill")
                lr._prefill_buffers(
                    processor=processor,
                    df=compact_prefill_df,
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
                    feed=feed,
                )

            self._store.seed_from_buffers(processor._buffers)
            seed_counts = {sym: len(buf) for sym, buf in processor._buffers.items()}
            tz_name = cfg.tz or "America/New_York"
            now_local = pd.Timestamp.now(tz=tz_name)
            for symbol in symbols:
                buf = processor._buffers.get(symbol)
                if buf is None or len(buf) == 0:
                    continue
                buf_df = buf.to_dataframe()
                if buf_df is None or buf_df.empty:
                    continue
                if "timestamp" in buf_df.columns:
                    ts_series = pd.to_datetime(buf_df["timestamp"], utc=True, errors="coerce")
                else:
                    ts_series = pd.to_datetime(buf_df.index, utc=True, errors="coerce")
                last_ts = ts_series.max()
                if pd.isna(last_ts):
                    continue
                last_local = last_ts.tz_convert(tz_name)
                self._emit(
                    "log",
                    {
                        "symbol": "SYSTEM",
                        "message": f"[live] prefill latest 1m for {symbol}: {last_local.isoformat()}",
                    },
                )
                if last_local.normalize() < now_local.normalize():
                    self._emit(
                        "log",
                        {
                            "symbol": "SYSTEM",
                            "message": (
                                f"[live] no prefill bars on {now_local.date()} for {symbol}; "
                                "market may be closed/holiday or feed has no newer bars."
                            ),
                        },
                    )
            self._emit(
                "log",
                {
                    "symbol": "SYSTEM",
                    "message": f"[live] prefill loaded bars: {seed_counts}",
                },
            )
            self._seed_15m_from_prefill(
                cfg=cfg,
                processor=processor,
                agent=agent,
                inference=inference,
                execution_latches=execution_latches,
                last_exec_action_by_symbol=last_exec_action_by_symbol,
            )
            self._emit_state_sync()

            if order_policies:
                latest_closed = lr._warmup_order_policies_from_prefill(
                    processor=processor,
                    order_policies=order_policies,
                    symbols=symbols,
                )
                for symbol in symbols:
                    self._store.set_policy_state(symbol, order_policies[symbol].snapshot_state())
                saved_policy_states = runtime_policy_cache.get("policies") if isinstance(runtime_policy_cache, dict) else None
                if isinstance(saved_policy_states, dict):
                    restored_symbols: list[str] = []
                    for symbol in symbols:
                        saved_state = saved_policy_states.get(symbol)
                        if order_policies[symbol].load_runtime_state(saved_state, logger=self._policy_logger(symbol)):
                            restored_symbols.append(symbol)
                            self._store.set_policy_state(symbol, order_policies[symbol].snapshot_state())
                    if restored_symbols:
                        self._emit(
                            "log",
                            {
                                "symbol": "SYSTEM",
                                "message": f"[live] restored runtime policy state cache for: {', '.join(restored_symbols)}",
                            },
                        )
                _persist_policy_runtime_cache()
                if cfg.startup_catchup_order:
                    lr._run_startup_catchup_decision(
                        processor=processor,
                        inference=inference,
                        order_policies=order_policies,
                        execution_latches=execution_latches,
                        latest_closed_by_symbol=latest_closed,
                        symbols=symbols,
                        print_tz=cfg.tz,
                        max_age_min=int(cfg.startup_catchup_max_age_min),
                    )
                    for symbol in symbols:
                        self._store.set_policy_state(symbol, order_policies[symbol].snapshot_state())
                    _persist_policy_runtime_cache()

            stream_symbols = list(symbols)
            if live_vix_enabled and live_vix_symbol not in stream_symbols:
                stream_symbols.append(live_vix_symbol)
            for context_symbol in context_processors:
                if context_symbol not in stream_symbols:
                    stream_symbols.append(context_symbol)

            streamer = AlpacaBarStreamer(
                symbols=stream_symbols,
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
                            reconcile_event: dict[str, Any] | None = None
                            if policy.has_pending_broker_reconcile():
                                reconcile_result = policy.reconcile_pending_broker_order(
                                    logger=self._policy_logger(symbol)
                                )
                                if cfg.use_execution_latch:
                                    execution_latches[symbol].set_position(
                                        policy.snapshot_state().get("position", 0)
                                    )
                                self._store.set_policy_state(symbol, policy.snapshot_state())
                                _persist_policy_runtime_cache()
                                self._emit(
                                    "order_policy",
                                    {
                                        "symbol": symbol,
                                        "timestamp": _ts_iso(datetime.now(tz=ZoneInfo(cfg.tz or "America/New_York"))),
                                        "result": {
                                            "event": "broker_reconcile",
                                            "pending_reconcile_result": reconcile_result,
                                        },
                                        "policy_state": policy.snapshot_state(),
                                    },
                                )
                                self._audit(
                                    "order_policy",
                                    {
                                        "symbol": symbol,
                                        "timestamp": _ts_iso(datetime.now(tz=ZoneInfo(cfg.tz or "America/New_York"))),
                                        "result": {
                                            "event": "broker_reconcile",
                                            "pending_reconcile_result": reconcile_result,
                                        },
                                        "policy_state": policy.snapshot_state(),
                                    },
                                )
                            else:
                                reconcile_result = policy.reconcile_with_broker(
                                    logger=self._policy_logger(symbol),
                                    force=True,
                                )
                                if reconcile_result.get("changed"):
                                    if cfg.use_execution_latch:
                                        execution_latches[symbol].set_position(
                                            policy.snapshot_state().get("position", 0)
                                        )
                                    self._store.set_policy_state(symbol, policy.snapshot_state())
                                    _persist_policy_runtime_cache()
                                    reconcile_event = {
                                        "symbol": symbol,
                                        "timestamp": _ts_iso(datetime.now(tz=ZoneInfo(cfg.tz or "America/New_York"))),
                                        "result": {
                                            "event": "broker_sync",
                                            "broker_sync_result": reconcile_result,
                                        },
                                        "policy_state": policy.snapshot_state(),
                                    }
                            if reconcile_event is not None:
                                self._emit("order_policy", reconcile_event)
                                self._audit("order_policy", reconcile_event)
                            broker_snap = policy.snapshot_broker_state(orders_limit=20)
                            self._store.set_broker_state(symbol, broker_snap)
                            self._emit("broker_state", {"symbol": symbol, "state": broker_snap})
                            self._audit("broker_state", {"symbol": symbol, "state": broker_snap})
                try:
                    bar = bar_queue.get(timeout=0.5)
                except queue_mod.Empty:
                    continue
                bar_symbol = str(bar.get("symbol", "")).upper()
                context_processor = context_processors.get(bar_symbol)
                if context_processor is not None:
                    context_processor.handle_bar(bar)
                    continue
                if live_vix_enabled and vix_processor is not None and bar_symbol == live_vix_symbol:
                    vix_processor.handle_bar(bar)
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
            self._audit(
                "session",
                {
                    "event": "session_error",
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
            self._audit(
                "session",
                {
                    "event": "session_stopped",
                    "error": error_message,
                    "stopped_at": _ts_iso(datetime.now(timezone.utc)),
                },
            )
            if audit_writer is not None:
                audit_writer.stop()
            self._audit_writer = None
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
        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        self.session.stop()
        return self.snapshot()

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
        self.send_header("Cache-Control", "no-store")
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
                hello = {
                    "type": "hello",
                    "timestamp": _ts_iso(datetime.now(timezone.utc)),
                    "payload": app.snapshot(),
                }
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
