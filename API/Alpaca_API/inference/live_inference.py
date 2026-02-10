from __future__ import annotations

from dataclasses import dataclass
import contextlib
import io
from pathlib import Path
from typing import Callable, Optional
import warnings

import numpy as np
import pandas as pd
import pandas_ta as ta  # registers df.ta accessor
import torch
import xgboost as xgb

from Policy.Agent.env import sincos_time_of_day
from Policy.Agent.model import ActorCritic
from Features.feature_matrix import DEFAULT_FEATURE_TIMEFRAMES, _add_feature_set, _align_htf_features
from Features.feature_matrix_agent import (
    _add_pivot_features,
    _compute_prior_day_high,
    _compute_time_sin_cos,
    _series_from_ta,
)
from Features.multi_timeframe_features import ensure_time_index, resample_ohlcv


def _resolve_device(device: str) -> torch.device:
    dev = (device or "auto").lower()
    if dev in ("auto", "gpu", "cuda"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


@contextlib.contextmanager
def _quiet_feature_ops() -> None:
    """
    Suppress noisy indicator/progress output during live/replay feature builds.
    """
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink), warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        yield


def build_15m(
    df_1m: pd.DataFrame,
    *,
    rule: str = "15min",
    label: str = "left",
    closed: str = "left",
    tz: str | None = None,
    assume_tz: str = "UTC",
) -> pd.DataFrame:
    """
    Resample 1-minute OHLCV data into 15-minute bars.

    Defaults match training pipeline settings (label=left, closed=left).
    """
    if df_1m.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if not isinstance(df_1m.index, pd.DatetimeIndex):
        if "timestamp" not in df_1m.columns:
            raise ValueError("df_1m must have a DatetimeIndex or a 'timestamp' column")
        df_1m = df_1m.copy()
        df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"], utc=True, errors="coerce")
        df_1m = df_1m.dropna(subset=["timestamp"]).set_index("timestamp")

    if df_1m.index.tz is None:
        df_1m.index = df_1m.index.tz_localize(assume_tz)
    if tz is not None:
        df_1m.index = df_1m.index.tz_convert(tz)

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df_15 = df_1m.resample(rule, label=label, closed=closed).agg(agg)
    df_15 = df_15.dropna(subset=["open", "high", "low", "close"])
    return df_15


def build_tree_feature_frame_from_1m(
    df_1m: pd.DataFrame,
    *,
    label_timeframe: str = "15min",
    feature_timeframes: dict[str, str] | None = None,
    include_custom: bool = True,
    include_date_features: bool = True,
    include_htf_date_features: bool = False,
    shift_htf_bars: int = 1,
    resample_label: str = "left",
    resample_closed: str = "left",
    tz: str | None = "America/New_York",
    assume_tz: str = "UTC",
) -> pd.DataFrame:
    if df_1m.empty:
        return pd.DataFrame()

    df = df_1m.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" not in df.columns:
            raise ValueError("df_1m must have a DatetimeIndex or a 'timestamp' column")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")

    df = ensure_time_index(df, tz=tz, assume_tz=assume_tz)

    feature_timeframes = feature_timeframes or DEFAULT_FEATURE_TIMEFRAMES

    with _quiet_feature_ops():
        df_15m = resample_ohlcv(
            df, label_timeframe, label=resample_label, closed=resample_closed
        )
        if df_15m.empty:
            return pd.DataFrame()

        f15 = _add_feature_set(
            df_15m,
            include_custom=include_custom,
            include_date_features=include_date_features,
            verbose=False,
            model="tree",
        )

        frames = [f15]
        for tf_label, tf_rule in feature_timeframes.items():
            tf_df = resample_ohlcv(df, tf_rule, label=resample_label, closed=resample_closed)
            if tf_df.empty:
                continue
            tf_feat = _add_feature_set(
                tf_df,
                include_custom=include_custom,
                include_date_features=include_htf_date_features,
                verbose=False,
                model="tree",
            )
            aligned = _align_htf_features(
                tf_feat,
                base_index=f15.index,
                suffix=tf_label,
                shift_bars=shift_htf_bars,
            )
            frames.append(aligned)

    return pd.concat(frames, axis=1)


def build_agent_feature_frame_from_15m(
    df_15m: pd.DataFrame,
    *,
    include_pivot_probs: bool = True,
    include_tb_probs: bool = True,
    tz: str | None = "America/New_York",
    assume_tz: str = "UTC",
    session_open: str = "09:30",
    session_close: str = "16:00",
    fill_missing_prob: float = 0.0,
    include_state_placeholders: bool = True,
) -> pd.DataFrame:
    if df_15m.empty:
        return df_15m.copy()

    df = df_15m.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" not in df.columns:
            raise ValueError("df_15m must have a DatetimeIndex or a 'timestamp' column")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")

    if df.index.tz is None:
        df.index = df.index.tz_localize(assume_tz)
    if tz is not None:
        df.index = df.index.tz_convert(tz)

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    if include_pivot_probs:
        if "p_pivot_long" not in df.columns:
            df["p_pivot_long"] = float(fill_missing_prob)
        if "p_pivot_short" not in df.columns:
            df["p_pivot_short"] = float(fill_missing_prob)
        df = _add_pivot_features(df, "p_pivot_long")
        df = _add_pivot_features(df, "p_pivot_short")

    if include_tb_probs:
        if "p_tb_long" not in df.columns:
            df["p_tb_long"] = float(fill_missing_prob)
        if "p_tb_short" not in df.columns:
            df["p_tb_short"] = float(fill_missing_prob)
        df = _add_pivot_features(df, "p_tb_long")
        df = _add_pivot_features(df, "p_tb_short")

    def _quiet_ta(fn, *args, **kwargs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            return fn(*args, **kwargs)

    sin_time, cos_time = _compute_time_sin_cos(
        df.index, tz=tz, session_open=session_open, session_close=session_close
    )
    df["sin_time_of_day"] = sin_time
    df["cos_time_of_day"] = cos_time

    atr = _series_from_ta(_quiet_ta(df.ta.atr, length=14, append=False))
    df["atr_pct"] = atr / df["close"].replace(0, np.nan)

    vwap = _series_from_ta(_quiet_ta(df.ta.vwap, append=False, anchor="D"))
    df["dist_to_vwap"] = (df["close"] - vwap) / df["close"].replace(0, np.nan)

    pdh = _compute_prior_day_high(df)
    df["dist_to_pdh"] = df["close"] - pdh

    adx_df = _quiet_ta(df.ta.adx, length=14, append=False)
    df["trend_strength"] = _series_from_ta(adx_df, prefix="ADX")

    df["timestamp"] = df.index
    df["day_id"] = pd.Series(df.index.normalize()).factorize()[0]

    close = df["close"].replace(0, np.nan).astype(float)
    for lag in (1, 2, 4, 8, 16):
        df[f"ret_{lag}"] = close.pct_change(lag)

    if include_state_placeholders:
        df["current_position"] = 0.0
        df["time_in_position"] = 0.0
        df["bars_since_last_trade"] = 0.0
        df["unrealized_pnl"] = 0.0
        df["realized_pnl_today"] = 0.0

    cols = [
        "timestamp",
        "day_id",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sin_time_of_day",
        "cos_time_of_day",
        "atr_pct",
        "dist_to_vwap",
        "dist_to_pdh",
        "trend_strength",
        "ret_1",
        "ret_2",
        "ret_4",
        "ret_8",
        "ret_16",
    ]
    if include_pivot_probs:
        cols.extend(
            [
                "p_pivot_long",
                "p_pivot_long_lag1",
                "p_pivot_long_lag2",
                "p_pivot_long_max_last_4",
                "p_pivot_long_delta_1",
                "p_pivot_short",
                "p_pivot_short_lag1",
                "p_pivot_short_lag2",
                "p_pivot_short_max_last_4",
                "p_pivot_short_delta_1",
            ]
        )
    if include_tb_probs:
        cols.extend(
            [
                "p_tb_long",
                "p_tb_long_lag1",
                "p_tb_long_lag2",
                "p_tb_long_max_last_4",
                "p_tb_long_delta_1",
                "p_tb_short",
                "p_tb_short_lag1",
                "p_tb_short_lag2",
                "p_tb_short_max_last_4",
                "p_tb_short_delta_1",
            ]
        )

    if include_state_placeholders:
        cols.extend(
            [
                "current_position",
                "time_in_position",
                "bars_since_last_trade",
                "unrealized_pnl",
                "realized_pnl_today",
            ]
        )

    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


class LiveGAXGBPredictor:
    def __init__(
        self,
        *,
        model_root: str | Path,
        feature_list_path: str | Path,
        include_pivot_probs: bool = True,
        include_tb_probs: bool = True,
        pivot_label_dir: str = "pivots",
        tb_label_dir: str = "tb",
    ) -> None:
        self._model_root = Path(model_root)
        self._feature_list = self._load_feature_list(Path(feature_list_path))
        self._include_pivot = bool(include_pivot_probs)
        self._include_tb = bool(include_tb_probs)

        self._pivot_long = None
        self._pivot_short = None
        self._tb_long = None
        self._tb_short = None

        if self._include_pivot:
            self._pivot_long = self._load_model("long", label_dir=pivot_label_dir)
            self._pivot_short = self._load_model("short", label_dir=pivot_label_dir)
        if self._include_tb:
            self._tb_long = self._load_model("long", label_dir=tb_label_dir)
            self._tb_short = self._load_model("short", label_dir=tb_label_dir)

    @staticmethod
    def _load_feature_list(path: Path) -> list[str]:
        if not path.exists():
            raise FileNotFoundError(f"Missing GA-XGB feature list: {path}")
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    def _load_model(self, side: str, *, label_dir: str | None = None) -> tuple[np.ndarray, xgb.Booster]:
        candidates = []
        base_side = self._model_root / side.lower()
        if label_dir:
            candidates.append(base_side / "probs" / label_dir)
        candidates.append(base_side)

        for candidate in candidates:
            mask_path = candidate / "best_mask.npy"
            model_path = candidate / "xgb_model.json"
            if mask_path.exists() and model_path.exists():
                mask = np.load(mask_path).astype(bool)
                model = xgb.Booster()
                model.load_model(str(model_path))
                return mask, model

        raise FileNotFoundError(
            f"Missing GA-XGB artifacts under {', '.join(str(c) for c in candidates)}"
        )

    def _predict_one(
        self,
        x_row: np.ndarray,
        mask: np.ndarray,
        model: xgb.Booster,
    ) -> float:
        if mask.size != x_row.shape[0]:
            raise ValueError("Mask length does not match feature vector length.")
        x_sel = x_row[mask].reshape(1, -1)
        dmat = xgb.DMatrix(x_sel)
        return float(model.predict(dmat)[0])

    def _predict_many(
        self,
        x_mat: np.ndarray,
        mask: np.ndarray,
        model: xgb.Booster,
    ) -> np.ndarray:
        if mask.size != x_mat.shape[1]:
            raise ValueError("Mask length does not match feature matrix width.")
        x_sel = x_mat[:, mask]
        dmat = xgb.DMatrix(x_sel)
        return model.predict(dmat).astype(np.float32, copy=False)

    def predict_frame(self, x_df: pd.DataFrame) -> pd.DataFrame:
        if x_df.empty:
            return pd.DataFrame(index=x_df.index)

        x_aligned = x_df.reindex(columns=self._feature_list)
        x_mat = x_aligned.to_numpy(dtype=np.float32)
        if x_mat.ndim != 2:
            x_mat = np.atleast_2d(x_mat)

        out: dict[str, np.ndarray] = {}
        if self._include_pivot and self._pivot_long and self._pivot_short:
            mask, model = self._pivot_long
            out["p_pivot_long"] = self._predict_many(x_mat, mask, model)
            mask, model = self._pivot_short
            out["p_pivot_short"] = self._predict_many(x_mat, mask, model)
        if self._include_tb and self._tb_long and self._tb_short:
            mask, model = self._tb_long
            out["p_tb_long"] = self._predict_many(x_mat, mask, model)
            mask, model = self._tb_short
            out["p_tb_short"] = self._predict_many(x_mat, mask, model)

        if not out:
            return pd.DataFrame(index=x_aligned.index)
        return pd.DataFrame(out, index=x_aligned.index)

    def predict_row(
        self,
        x_df: pd.DataFrame,
        *,
        target_ts: pd.Timestamp | None = None,
    ) -> dict[str, float]:
        if x_df.empty:
            return {}

        if target_ts is not None and target_ts in x_df.index:
            row_df = x_df.loc[[target_ts]]
        else:
            row_df = x_df.tail(1)
        pred = self.predict_frame(row_df)
        if pred.empty:
            return {}
        return {k: float(v) for k, v in pred.iloc[-1].to_dict().items()}


@dataclass
class _PositionState:
    position: int = 0
    entry_price: float = np.nan
    time_in_pos: int = 0
    realized_pnl_today: float = 0.0
    unrealized_pnl: float = 0.0
    last_session_day: Optional[pd.Timestamp] = None


class LivePPOAgent:
    def __init__(
        self,
        *,
        model_path: str,
        deterministic: bool = True,
        device: str = "auto",
        include_pivot_probs: bool = True,
        include_tb_probs: bool = True,
        tz: str | None = "America/New_York",
        assume_tz: str = "UTC",
        session_open: str = "09:30",
        session_close: str = "16:00",
        min_15m_bars: int = 20,
        fill_missing_prob: float = 0.0,
        skip_on_nan: bool = True,
        ga_model_root: str | None = None,
        ga_feature_list_path: str | None = None,
        ga_pivot_label_dir: str = "pivots",
        ga_tb_label_dir: str = "tb",
        ga_probs_frame: pd.DataFrame | None = None,
        ga_probs_mode: str = "xgb",
        resample_label: str = "left",
        resample_closed: str = "left",
        label_timeframe_rule: str = "15min",
    ) -> None:
        ckpt = torch.load(model_path, map_location="cpu")
        feature_cols = ckpt.get("feature_cols")
        if not feature_cols:
            raise ValueError("Checkpoint missing feature_cols; cannot run live inference.")

        self._feature_cols = list(feature_cols)
        self._obs_dim = int(ckpt["obs_dim"])
        self._n_actions = int(ckpt.get("n_actions", 3))
        self._deterministic = bool(deterministic)
        self._device = _resolve_device(device)
        self._tz = tz
        self._assume_tz = assume_tz
        self._session_open = session_open
        self._session_close = session_close
        self._include_pivot_probs = bool(include_pivot_probs)
        self._include_tb_probs = bool(include_tb_probs)
        self._min_15m_bars = int(min_15m_bars)
        self._fill_missing_prob = float(fill_missing_prob)
        self._skip_on_nan = bool(skip_on_nan)
        self._resample_label = resample_label
        self._resample_closed = resample_closed
        self._label_timeframe_rule = label_timeframe_rule
        self._ga_probs_frame = ga_probs_frame
        self._ga_probs_mode = str(ga_probs_mode or "xgb").strip().lower()
        self._warned_ga_frame = False
        self._last_probs: dict[str, float | None] | None = None

        state_dict = ckpt["state_dict"]
        has_head_mlps = any(
            k.startswith("policy_mlp.") or k.startswith("value_mlp.")
            for k in state_dict
        )
        self._model = ActorCritic(
            obs_dim=self._obs_dim,
            n_actions=self._n_actions,
            head_mlp=has_head_mlps,
        )
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

        self._add_time_features, self._add_position_features = self._infer_extra_flags()
        self._state = _PositionState()
        self._warned_missing_cols = False
        self._warned_ga_error = False

        self._ga_predictor: LiveGAXGBPredictor | None = None
        if (self._include_pivot_probs or self._include_tb_probs) and ga_model_root and ga_feature_list_path:
            self._ga_predictor = LiveGAXGBPredictor(
                model_root=ga_model_root,
                feature_list_path=ga_feature_list_path,
                include_pivot_probs=self._include_pivot_probs,
                include_tb_probs=self._include_tb_probs,
                pivot_label_dir=ga_pivot_label_dir,
                tb_label_dir=ga_tb_label_dir,
            )

    def _apply_ga_probs_frame(self, df_15m: pd.DataFrame) -> pd.DataFrame:
        frame = self._ga_probs_frame
        if frame is None or frame.empty:
            return df_15m
        df = df_15m.copy()
        idx = frame.index
        if not isinstance(idx, pd.DatetimeIndex):
            return df
        base = frame
        if idx.tz is None:
            base = base.copy()
            base.index = base.index.tz_localize(self._assume_tz)
        if self._tz is not None:
            base = base.copy()
            base.index = base.index.tz_convert(self._tz)
        aligned = base.reindex(df.index)
        if aligned.empty and not self._warned_ga_frame:
            print("[live] GA prob frame had no overlapping timestamps with replay bars.")
            self._warned_ga_frame = True
            return df
        cols = [c for c in ("p_pivot_long", "p_pivot_short", "p_tb_long", "p_tb_short") if c in aligned.columns]
        if not cols:
            return df
        for col in cols:
            if self._ga_probs_mode == "frame":
                df[col] = aligned[col].astype(float)
            else:
                # Hybrid: fill missing only.
                if col in df.columns:
                    df[col] = df[col].fillna(aligned[col].astype(float))
                else:
                    df[col] = aligned[col].astype(float)
            if col in df.columns:
                df[col] = df[col].fillna(self._fill_missing_prob)
        return df

    def _infer_extra_flags(self) -> tuple[bool, bool]:
        extra = self._obs_dim - len(self._feature_cols)
        if extra == 0:
            return False, False
        if extra == 2:
            return True, False
        if extra == 4:
            return False, True
        if extra == 6:
            return True, True
        raise ValueError(
            f"Obs dim {self._obs_dim} does not match feature_cols ({len(self._feature_cols)}) plus expected extras."
        )

    def _normalize_ts(self, ts: pd.Timestamp) -> pd.Timestamp:
        if ts.tzinfo is None:
            ts = ts.tz_localize(self._assume_tz)
        if self._tz is not None:
            ts = ts.tz_convert(self._tz)
        return ts

    def _resolve_target_index(self, df: pd.DataFrame, target_ts: pd.Timestamp | None) -> pd.Timestamp:
        if target_ts is not None:
            ts = self._normalize_ts(pd.to_datetime(target_ts, utc=True, errors="coerce"))
            if ts in df.index:
                return ts
        return df.index[-1]

    def _update_day_state(self, ts: pd.Timestamp, price: float) -> None:
        day = ts.normalize()
        if self._state.last_session_day is None:
            self._state.last_session_day = day
            return
        if day != self._state.last_session_day:
            self._state.last_session_day = day
            self._state.realized_pnl_today = 0.0
            if self._state.position != 0 and np.isfinite(self._state.entry_price):
                self._state.unrealized_pnl = (
                    price / self._state.entry_price - 1.0
                ) * float(self._state.position)
            else:
                self._state.unrealized_pnl = 0.0
                if self._state.position == 0:
                    self._state.time_in_pos = 0

    def _build_obs(self, features: np.ndarray, ts: pd.Timestamp) -> np.ndarray:
        parts = [features]
        if self._add_time_features:
            s, c = sincos_time_of_day(ts)
            parts.append(np.array([s, c], dtype=np.float32))
        if self._add_position_features:
            parts.append(
                np.array(
                    [
                        float(self._state.position),
                        float(self._state.time_in_pos),
                        float(self._state.unrealized_pnl),
                        float(self._state.realized_pnl_today),
                    ],
                    dtype=np.float32,
                )
            )
        obs = np.concatenate(parts, axis=0).astype(np.float32)
        if obs.shape[0] != self._obs_dim:
            raise RuntimeError(f"Obs dim mismatch: got {obs.shape[0]} expected {self._obs_dim}")
        return obs

    def _mark_to_market(self, price: float) -> None:
        if self._state.position != 0 and np.isfinite(self._state.entry_price):
            self._state.unrealized_pnl = (
                price / self._state.entry_price - 1.0
            ) * float(self._state.position)
        else:
            self._state.unrealized_pnl = 0.0
            if self._state.position == 0:
                self._state.time_in_pos = 0

    def _select_row(self, df: pd.DataFrame, target_ts: pd.Timestamp | None) -> pd.Series:
        if target_ts is not None:
            ts = self._normalize_ts(pd.to_datetime(target_ts, utc=True, errors="coerce"))
            if ts in df.index:
                row = df.loc[ts]
                if isinstance(row, pd.DataFrame):
                    return row.iloc[-1]
                return row
        return df.iloc[-1]

    def _maybe_add_ga_probs(
        self,
        df_1m: pd.DataFrame,
        df_15m: pd.DataFrame,
        target_ts: pd.Timestamp | None,
    ) -> pd.DataFrame:
        if self._ga_probs_mode == "frame" and self._ga_probs_frame is not None:
            return self._apply_ga_probs_frame(df_15m)
        if self._ga_predictor is None:
            return self._apply_ga_probs_frame(df_15m)

        try:
            norm_target = None
            if target_ts is not None:
                norm_target = self._normalize_ts(
                    pd.to_datetime(target_ts, utc=True, errors="coerce")
                )
            x_tree = build_tree_feature_frame_from_1m(
                df_1m,
                label_timeframe=self._label_timeframe_rule,
                resample_label=self._resample_label,
                resample_closed=self._resample_closed,
                tz=self._tz,
                assume_tz=self._assume_tz,
            )
            if x_tree.empty:
                return df_15m
            probs_df = self._ga_predictor.predict_frame(x_tree)
            if probs_df.empty:
                return df_15m
            df = df_15m.copy()
            for key in probs_df.columns:
                df[key] = probs_df[key].reindex(df.index)
            for col in ("p_pivot_long", "p_pivot_short", "p_tb_long", "p_tb_short"):
                if col in df.columns:
                    df[col] = df[col].fillna(self._fill_missing_prob)
            return self._apply_ga_probs_frame(df)
        except Exception as exc:
            if not self._warned_ga_error:
                print(f"[live] GA-XGB inference failed: {exc}")
                self._warned_ga_error = True
            return self._apply_ga_probs_frame(df_15m)

    def act(
        self,
        *,
        df_1m: pd.DataFrame,
        df_15m: pd.DataFrame,
        target_ts: pd.Timestamp | None = None,
    ) -> Optional[int]:
        if df_15m.empty or len(df_15m) < self._min_15m_bars:
            return None

        df_15m = self._maybe_add_ga_probs(df_1m, df_15m, target_ts)

        feat_df = build_agent_feature_frame_from_15m(
            df_15m,
            include_pivot_probs=self._include_pivot_probs,
            include_tb_probs=self._include_tb_probs,
            tz=self._tz,
            assume_tz=self._assume_tz,
            session_open=self._session_open,
            session_close=self._session_close,
            fill_missing_prob=self._fill_missing_prob,
            include_state_placeholders=False,
        )
        if feat_df.empty:
            return None

        missing = [c for c in self._feature_cols if c not in feat_df.columns]
        if missing and not self._warned_missing_cols:
            print(f"[live] Missing {len(missing)} PPO feature columns: {missing}")
            self._warned_missing_cols = True

        for col in missing:
            feat_df[col] = 0.0

        row_full = self._select_row(feat_df, target_ts)
        price = float(row_full.get("close", np.nan))
        self._last_probs = {
            "p_pivot_long": float(row_full.get("p_pivot_long", np.nan)),
            "p_pivot_short": float(row_full.get("p_pivot_short", np.nan)),
            "p_tb_long": float(row_full.get("p_tb_long", np.nan)),
            "p_tb_short": float(row_full.get("p_tb_short", np.nan)),
        }
        for key, val in list(self._last_probs.items()):
            if not np.isfinite(val):
                self._last_probs[key] = None
        row = row_full.reindex(self._feature_cols)
        if self._skip_on_nan and row.isna().any():
            return None

        features = row.to_numpy(dtype=np.float32)
        ts = row.name
        if not isinstance(ts, pd.Timestamp):
            ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
        if ts is None or pd.isna(ts):
            return None
        ts = self._normalize_ts(ts)
        if not np.isfinite(price):
            return None

        self._update_day_state(ts, price)
        # Align observation state with current bar before action selection.
        self._mark_to_market(price)
        obs = self._build_obs(features, ts)
        x = torch.as_tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)
        logits, _ = self._model(x)
        if self._deterministic:
            action = int(torch.argmax(logits, dim=-1).item())
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = int(dist.sample().item())

        self._apply_action(action, price)
        return action

    def _apply_action(self, action: int, price: float) -> None:
        desired_pos = 0 if action == 0 else (1 if action == 1 else -1)
        if desired_pos != self._state.position:
            if self._state.position != 0 and np.isfinite(self._state.entry_price):
                move = (price / self._state.entry_price - 1.0) * float(self._state.position)
                self._state.realized_pnl_today += move

            if desired_pos != 0:
                self._state.entry_price = price
                self._state.time_in_pos = 0
            else:
                self._state.entry_price = np.nan
                self._state.time_in_pos = 0

            self._state.position = desired_pos

        if self._state.position != 0 and np.isfinite(self._state.entry_price):
            self._state.time_in_pos += 1
        else:
            self._state.unrealized_pnl = 0.0
            self._state.time_in_pos = 0

    def snapshot_state(self) -> dict[str, object]:
        """
        Return a JSON-safe snapshot of the internal live position state.
        """
        last_day = self._state.last_session_day
        entry = self._state.entry_price
        return {
            "position": int(self._state.position),
            "entry_price": float(entry) if np.isfinite(entry) else None,
            "time_in_position": int(self._state.time_in_pos),
            "realized_pnl_today": float(self._state.realized_pnl_today),
            "unrealized_pnl": float(self._state.unrealized_pnl),
            "last_session_day": last_day.isoformat() if isinstance(last_day, pd.Timestamp) else None,
            "last_probs": self._last_probs,
        }

    def last_probs(self) -> dict[str, float | None] | None:
        return self._last_probs


class LiveInferenceEngine:
    """
    Wrapper for live inference on completed 15-minute bars.

    You can inject your own feature/predict functions, or attach a LivePPOAgent.
    """

    def __init__(
        self,
        *,
        feature_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        predict_fn: Optional[Callable[[pd.DataFrame], object]] = None,
        agent: Optional[LivePPOAgent] = None,
        label: str = "left",
        closed: str = "left",
        rule: str = "15min",
        tz: str | None = None,
        assume_tz: str = "UTC",
    ) -> None:
        self._feature_fn = feature_fn
        self._predict_fn = predict_fn
        self._agent = agent
        self._label = label
        self._closed = closed
        self._rule = rule
        self._tz = tz
        self._assume_tz = assume_tz

    def on_15m_close(self, *, df_1m: pd.DataFrame, closed_bar: dict | None = None) -> object:
        df_15m = build_15m(
            df_1m,
            rule=self._rule,
            label=self._label,
            closed=self._closed,
            tz=self._tz,
            assume_tz=self._assume_tz,
        )
        if df_15m.empty:
            return None

        target_ts = None
        if closed_bar and "timestamp" in closed_bar:
            target_ts = closed_bar["timestamp"]

        if self._agent is not None:
            return self._agent.act(df_1m=df_1m, df_15m=df_15m, target_ts=target_ts)
        if self._feature_fn is None or self._predict_fn is None:
            return None
        features = self._feature_fn(df_15m)
        return self._predict_fn(features)
