from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from Data.load_data import load_ticker_parquet


DEFAULT_COMMAND_COLS = [
    "htf_dir",
    "htf_conf",
    "time_since_flip_min",
    "htf_atr_pct",
    "htf_expected_edge",
]


def _read_frame(path_like: str | Path) -> pd.DataFrame:
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def _ensure_timestamp_col(df: pd.DataFrame, *, tz: str | None) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        ts = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    else:
        ts = pd.to_datetime(out.index, errors="coerce", utc=True)
    if tz:
        ts = ts.dt.tz_convert(tz)
    out["timestamp"] = ts
    out = out[out["timestamp"].notna()].copy()
    return out.sort_values("timestamp").reset_index(drop=True)


def _ensure_ohlcv_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    out = out.rename(columns=rename_map)
    required = ("open", "high", "low", "close")
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")
    if "volume" not in out.columns:
        out["volume"] = 0.0
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"]).copy()
    return out


def load_1m_frame(
    *,
    ticker: str,
    raw_1m_path: str | Path,
    tz: str | None = "America/New_York",
) -> pd.DataFrame:
    df = load_ticker_parquet(ticker, parquet_path=raw_1m_path).reset_index()
    if "date" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    df = _ensure_timestamp_col(df, tz=tz)
    df = _ensure_ohlcv_cols(df)
    df["day_id"] = pd.Series(df["timestamp"].dt.normalize()).factorize()[0]
    return df


def _action_dir_from_trace(df: pd.DataFrame) -> pd.Series:
    if "htf_dir" in df.columns:
        return pd.to_numeric(df["htf_dir"], errors="coerce").fillna(0.0).clip(-1, 1)
    if "action_dir_idx" in df.columns:
        idx = pd.to_numeric(df["action_dir_idx"], errors="coerce")
        out = pd.Series(0.0, index=df.index)
        out[idx == 1] = 1.0
        out[idx == 2] = -1.0
        return out
    if "action" in df.columns:
        return np.sign(pd.to_numeric(df["action"], errors="coerce").fillna(0.0)).astype(float)
    raise ValueError("Intent frame is missing direction columns: htf_dir/action_dir_idx/action")


def _confidence_from_trace(df: pd.DataFrame) -> pd.Series:
    if "htf_conf" in df.columns:
        return pd.to_numeric(df["htf_conf"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if "action_mag" in df.columns:
        return pd.to_numeric(df["action_mag"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if "action" in df.columns:
        return pd.to_numeric(df["action"], errors="coerce").abs().fillna(0.0).clip(0.0, 1.0)
    return pd.Series(0.0, index=df.index, dtype=float)


def load_htf_intent_frame(
    intent_path: str | Path,
    *,
    tz: str | None = "America/New_York",
) -> pd.DataFrame:
    raw = _read_frame(intent_path)
    raw = _ensure_timestamp_col(raw, tz=tz)
    out = pd.DataFrame({"timestamp": raw["timestamp"]})
    out["htf_dir"] = _action_dir_from_trace(raw).astype(float)
    out["htf_conf"] = _confidence_from_trace(raw).astype(float)
    if "htf_atr_pct" in raw.columns:
        out["htf_atr_pct"] = pd.to_numeric(raw["htf_atr_pct"], errors="coerce")
    elif "convex_atr_scale" in raw.columns:
        out["htf_atr_pct"] = pd.to_numeric(raw["convex_atr_scale"], errors="coerce")
    else:
        out["htf_atr_pct"] = np.nan
    if "htf_expected_edge" in raw.columns:
        out["htf_expected_edge"] = pd.to_numeric(raw["htf_expected_edge"], errors="coerce")
    else:
        out["htf_expected_edge"] = np.nan
    out = out.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    out["htf_dir"] = out["htf_dir"].fillna(0.0).clip(-1, 1)
    out["htf_conf"] = out["htf_conf"].fillna(0.0).clip(0.0, 1.0)
    return out


def _compute_1m_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    open_ = out["open"].astype(float)
    volume = out["volume"].astype(float)

    out["ret_1m_1"] = close.pct_change(1)
    out["ret_1m_3"] = close.pct_change(3)
    out["ret_1m_5"] = close.pct_change(5)
    out["mom_1m_3"] = close / close.rolling(3, min_periods=1).mean() - 1.0
    out["mom_1m_10"] = close / close.rolling(10, min_periods=1).mean() - 1.0
    out["range_1m_pct"] = (high - low) / close.replace(0.0, np.nan)
    out["body_1m_pct"] = (close - open_) / close.replace(0.0, np.nan)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=1).mean()
    out["atr_1m_pct"] = atr14 / close.replace(0.0, np.nan)

    vol_mu = volume.rolling(20, min_periods=5).mean()
    vol_sd = volume.rolling(20, min_periods=5).std(ddof=0).replace(0.0, np.nan)
    out["vol_z_20"] = (volume - vol_mu) / vol_sd

    if "vwap" in out.columns:
        out["dist_to_vwap_1m"] = (
            (close - pd.to_numeric(out["vwap"], errors="coerce")) / close.replace(0.0, np.nan)
        )
    return out


def _compute_time_since_flip(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    dir_series = pd.to_numeric(df["htf_dir"], errors="coerce").fillna(0.0)
    flipped = dir_series.ne(dir_series.shift(1))
    last_flip_ts = ts.where(flipped).ffill()
    minutes = (ts - last_flip_ts).dt.total_seconds() / 60.0
    return minutes.fillna(0.0).clip(lower=0.0)


def align_htf_intent_to_1m(
    one_min_df: pd.DataFrame,
    htf_intent_df: pd.DataFrame,
    *,
    allow_exact_matches: bool = False,
) -> pd.DataFrame:
    left = one_min_df.sort_values("timestamp").copy()
    right = htf_intent_df.sort_values("timestamp").copy()
    merged = pd.merge_asof(
        left,
        right,
        on="timestamp",
        direction="backward",
        allow_exact_matches=bool(allow_exact_matches),
    )
    merged["htf_dir"] = pd.to_numeric(merged["htf_dir"], errors="coerce").fillna(0.0).clip(-1, 1)
    merged["htf_conf"] = pd.to_numeric(merged["htf_conf"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    merged["htf_atr_pct"] = pd.to_numeric(merged["htf_atr_pct"], errors="coerce")
    merged["htf_expected_edge"] = pd.to_numeric(merged["htf_expected_edge"], errors="coerce")
    merged["time_since_flip_min"] = _compute_time_since_flip(merged)
    prev_dir = merged["htf_dir"].shift(1).fillna(merged["htf_dir"])
    merged["htf_flip"] = merged["htf_dir"].ne(prev_dir).astype(np.int8)
    enter_event = (prev_dir == 0.0) & (merged["htf_dir"] != 0.0) & (merged["htf_flip"] == 1)
    exit_or_switch_event = (prev_dir != 0.0) & (merged["htf_dir"] != prev_dir) & (merged["htf_flip"] == 1)
    event_code = np.zeros(len(merged), dtype=np.int8)
    event_code[enter_event.to_numpy()] = 1
    event_code[exit_or_switch_event.to_numpy()] = -1
    merged["event_just_flipped"] = merged["htf_flip"].astype(np.int8)
    merged["event_type_code"] = pd.Series(event_code, index=merged.index).astype(np.float32)
    return merged


def build_execution_frame(
    *,
    ticker: str,
    raw_1m_path: str | Path,
    htf_intent_path: str | Path,
    tz: str | None = "America/New_York",
    allow_exact_matches: bool = False,
) -> pd.DataFrame:
    one_min = load_1m_frame(ticker=ticker, raw_1m_path=raw_1m_path, tz=tz)
    htf = load_htf_intent_frame(htf_intent_path, tz=tz)
    merged = align_htf_intent_to_1m(one_min, htf, allow_exact_matches=allow_exact_matches)
    merged = _compute_1m_features(merged)
    merged["ret_next"] = merged["close"].shift(-1) / merged["close"] - 1.0
    return merged


def default_execution_feature_cols(df: pd.DataFrame) -> list[str]:
    drop_cols = {
        "timestamp",
        "day_id",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ret_next",
        "oracle_enter",
        "oracle_exit",
        "oracle_score",
        "oracle_exit_score",
    }
    return [c for c in df.columns if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)]


def ensure_numeric_non_nan(
    df: pd.DataFrame,
    *,
    feature_cols: Iterable[str],
) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    mask = out[list(feature_cols)].notna().all(axis=1)
    return out.loc[mask].reset_index(drop=True)
