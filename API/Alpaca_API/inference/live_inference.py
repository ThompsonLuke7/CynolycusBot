from __future__ import annotations

from dataclasses import dataclass
import contextlib
import io
import json
from pathlib import Path
from typing import Callable, Iterator, Optional
import warnings

import numpy as np
import pandas as pd
import pandas_ta as ta  # registers df.ta accessor
import torch
import xgboost as xgb

from Policy.Agent.env import sincos_time_of_day
from Policy.Agent.model import ActorCritic
from Features.feature_matrix import (
    DEFAULT_FEATURE_TIMEFRAMES,
    _add_feature_set,
    _add_lstm_features_for_tree,
    _add_vix_suite_to_frame,
    _align_htf_features,
    _load_vix_1m,
)
from Features.feature_matrix_regime import (
    AgentFeatureConfig,
    VIX_FEATURE_COLUMNS,
    _add_intraday_sr_distance_features,
    _add_pivot_features,
    _add_probability_confidence_features,
    _add_vix_feature_suite,
    _add_volatility_regime_features,
    _compute_prior_day_high,
    _compute_time_features,
    _ensure_vix_feature_cols,
    _load_align_vix_ohlcv,
    _series_from_ta,
)
from Features.label_generations import add_trend_phase_labels
from Features.multi_timeframe_features import ensure_time_index, resample_ohlcv


_EMITTED_RUNTIME_WARNINGS: set[str] = set()


def _warn_once(message: str) -> None:
    if message in _EMITTED_RUNTIME_WARNINGS:
        return
    _EMITTED_RUNTIME_WARNINGS.add(message)
    print(message)


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
def _quiet_feature_ops() -> Iterator[None]:
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
    include_htf_date_features: bool = True,
    shift_htf_bars: int = 1,
    resample_label: str = "left",
    resample_closed: str = "left",
    tz: str | None = "America/New_York",
    assume_tz: str = "UTC",
    include_vix_features: bool = True,
    vix_ticker: str = "VIXY",
    vix_parquet_path: str | Path | None = None,
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
    vix_1m: pd.DataFrame | None = None
    if include_vix_features:
        try:
            vix_1m = _load_vix_1m(
                vix_ticker=vix_ticker,
                vix_parquet_path=vix_parquet_path,
                tz=tz,
            )
        except Exception:
            vix_1m = None

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
        f15 = _add_lstm_features_for_tree(
            f15,
            include_time_features=True,
            tz=tz,
        )
        if include_vix_features:
            f15 = _add_vix_suite_to_frame(
                f15,
                vix_1m=vix_1m,
                timeframe_rule=label_timeframe,
                resample_label=resample_label,
                resample_closed=resample_closed,
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
            tf_feat = _add_lstm_features_for_tree(
                tf_feat,
                include_time_features=include_htf_date_features,
                tz=tz,
            )
            if include_vix_features:
                tf_feat = _add_vix_suite_to_frame(
                    tf_feat,
                    vix_1m=vix_1m,
                    timeframe_rule=tf_rule,
                    resample_label=resample_label,
                    resample_closed=resample_closed,
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
    include_vix_features: bool = True,
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
    df = _add_probability_confidence_features(df)

    def _quiet_ta(fn, *args, **kwargs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            return fn(*args, **kwargs)

    (
        sin_time,
        cos_time,
        minutes_since_open,
        minutes_to_close,
        day_of_week_sin,
        day_of_week_cos,
    ) = _compute_time_features(
        df.index, tz=tz, session_open=session_open, session_close=session_close
    )
    df["sin_time_of_day"] = sin_time
    df["cos_time_of_day"] = cos_time
    df["minutes_since_open"] = minutes_since_open
    df["minutes_to_close"] = minutes_to_close
    df["day_of_week_sin"] = day_of_week_sin
    df["day_of_week_cos"] = day_of_week_cos

    atr = _series_from_ta(_quiet_ta(df.ta.atr, length=14, append=False))
    df["atr_pct"] = atr / df["close"].replace(0, np.nan)

    vwap = _series_from_ta(_quiet_ta(df.ta.vwap, append=False, anchor="D"))
    df["dist_to_vwap"] = (df["close"] - vwap) / df["close"].replace(0, np.nan)

    pdh = _compute_prior_day_high(df)
    df["dist_to_pdh"] = df["close"] - pdh
    df = _add_intraday_sr_distance_features(
        df,
        atr=atr,
        tz=tz,
        session_open=session_open,
        open_range_minutes=30,
    )

    adx_df = _quiet_ta(df.ta.adx, length=14, append=False)
    df["trend_strength"] = _series_from_ta(adx_df, prefix="ADX")

    df["timestamp"] = df.index
    df["day_id"] = pd.Series(df.index.normalize()).factorize()[0]

    close = df["close"].replace(0, np.nan).astype(float)
    for lag in (1, 2, 4, 8, 16):
        df[f"ret_{lag}"] = close.pct_change(lag)
    df = _add_volatility_regime_features(df)

    if include_vix_features:
        vix_cfg = AgentFeatureConfig(
            dataset_name="live",
            tz=tz,
            session_open=session_open,
            session_close=session_close,
            include_vix_features=True,
            vix_fetch_if_missing=False,
            vix_refetch_if_low_coverage=False,
            vix_warn_on_missing=False,
        )
        try:
            vix_ohlcv = _load_align_vix_ohlcv(cfg=vix_cfg, target_index=df.index)
            df = _add_vix_feature_suite(df, vix_ohlcv=vix_ohlcv)
        except Exception:
            df = _ensure_vix_feature_cols(df)
    else:
        df = _ensure_vix_feature_cols(df)

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
        "minutes_since_open",
        "minutes_to_close",
        "day_of_week_sin",
        "day_of_week_cos",
        "atr_pct",
        "dist_to_vwap",
        "dist_to_pdh",
        "dist_to_day_high_so_far_atr",
        "dist_to_day_low_so_far_atr",
        "dist_to_or_high_30m_atr",
        "dist_to_or_low_30m_atr",
        "trend_strength",
        "ret_1",
        "ret_2",
        "ret_4",
        "ret_8",
        "ret_16",
        "atr_pct_z_64",
        "atr_pct_rank_64",
        "realized_vol_4",
        "realized_vol_16",
        "realized_vol_32",
        "range_regime_8_32",
        "range_expansion_32",
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
                "pivot_edge",
                "pivot_edge_abs",
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
                "tb_edge",
                "tb_edge_abs",
            ]
        )
    if include_pivot_probs and include_tb_probs:
        cols.extend(["edge_disagreement_abs", "edge_sign_disagreement"])
    if include_vix_features:
        cols.extend(VIX_FEATURE_COLUMNS)

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


def build_meta_feature_frame_from_1m(
    df_1m: pd.DataFrame,
    *,
    rule: str = "10min",
    label: str = "left",
    closed: str = "left",
    tz: str | None = "America/New_York",
    assume_tz: str = "UTC",
    include_pivot_probs: bool = True,
    include_tb_probs: bool = True,
    include_vix_features: bool = True,
    fill_missing_prob: float = 0.0,
    session_open: str = "09:30",
    session_close: str = "16:00",
    ga_predictor: LiveGAXGBPredictor | None = None,
    ga_probs_frame: pd.DataFrame | None = None,
    ga_probs_mode: str = "xgb",
) -> pd.DataFrame:
    prob_cols = ("p_pivot_long", "p_pivot_short", "p_tb_long", "p_tb_short")
    df_tf = build_15m(
        df_1m,
        rule=rule,
        label=label,
        closed=closed,
        tz=tz,
        assume_tz=assume_tz,
    )
    if df_tf.empty:
        return df_tf
    prob_sources = pd.DataFrame(index=df_tf.index)

    if ga_predictor is not None:
        try:
            x_tree = build_tree_feature_frame_from_1m(
                df_1m,
                label_timeframe=rule,
                resample_label=label,
                resample_closed=closed,
                tz=tz,
                assume_tz=assume_tz,
            )
            if not x_tree.empty:
                probs_df = ga_predictor.predict_frame(x_tree)
                for col in probs_df.columns:
                    aligned = pd.to_numeric(probs_df[col].reindex(df_tf.index), errors="coerce")
                    df_tf[col] = aligned
                    if col in prob_cols:
                        prob_sources[col] = np.where(aligned.notna(), "xgb", None)
        except Exception as exc:
            _warn_once(f"[live] GA-XGB inference failed: {exc}")

    if ga_probs_frame is not None and not ga_probs_frame.empty:
        idx = ga_probs_frame.index
        if isinstance(idx, pd.DatetimeIndex):
            base = ga_probs_frame
            if idx.tz is None:
                base = base.copy()
                base.index = base.index.tz_localize(assume_tz)
            if tz is not None:
                base = base.copy()
                base.index = base.index.tz_convert(tz)
            aligned = base.reindex(df_tf.index)
            for col in prob_cols:
                if col not in aligned.columns:
                    continue
                aligned_col = pd.to_numeric(aligned[col], errors="coerce")
                existing = (
                    pd.to_numeric(df_tf[col], errors="coerce")
                    if col in df_tf.columns
                    else pd.Series(np.nan, index=df_tf.index, dtype=float)
                )
                df_tf[col] = existing.where(aligned_col.isna(), aligned_col)
                existing_src = (
                    prob_sources[col]
                    if col in prob_sources.columns
                    else pd.Series(index=df_tf.index, dtype=object)
                )
                prob_sources[col] = existing_src.where(aligned_col.isna(), "frame")

    for col in prob_cols:
        if col not in df_tf.columns:
            df_tf[col] = float(fill_missing_prob)
            prob_sources[col] = "fill"
            continue
        numeric = pd.to_numeric(df_tf[col], errors="coerce")
        missing = numeric.isna()
        if missing.any():
            numeric = numeric.fillna(float(fill_missing_prob))
        df_tf[col] = numeric
        existing_src = (
            prob_sources[col]
            if col in prob_sources.columns
            else pd.Series(index=df_tf.index, dtype=object)
        )
        prob_sources[col] = existing_src.where(~missing, "fill")
        prob_sources[col] = prob_sources[col].fillna("fill")

    feat_df = build_agent_feature_frame_from_15m(
        df_tf,
        include_pivot_probs=include_pivot_probs,
        include_tb_probs=include_tb_probs,
        include_vix_features=include_vix_features,
        tz=tz,
        assume_tz=assume_tz,
        session_open=session_open,
        session_close=session_close,
        fill_missing_prob=fill_missing_prob,
        include_state_placeholders=False,
    )
    if feat_df.empty:
        return feat_df
    feat_df.attrs["ga_prob_sources"] = prob_sources

    if "atr" not in feat_df.columns:
        feat_df["atr"] = ta.atr(feat_df["high"], feat_df["low"], feat_df["close"], length=14)
    feat_df = add_trend_phase_labels(
        feat_df,
        close_col="close",
        momentum_col="trend_phase_m",
        accel_col="trend_phase_a",
        phase_col="trend_phase_label",
        ignition_col="trend_phase_ignition",
        expansion_col="trend_phase_expansion",
        saturation_col="trend_phase_saturation",
        decay_col="trend_phase_decay",
        exit_long_col="trend_phase_exit_long",
        exit_short_col="trend_phase_exit_short",
        write_phase_columns=True,
        use_hazard_exit_labels=False,
    )
    return feat_df


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
            candidates.append(base_side / label_dir)
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


class _LiveXGBArtifact:
    def __init__(self, side_dir: Path) -> None:
        self._side_dir = Path(side_dir)
        self.feature_cols = self._load_feature_cols(self._side_dir / "feature_columns.txt")
        meta = {}
        meta_path = self._side_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        constant_prob = meta.get("constant_prob")
        self.constant_prob = float(constant_prob) if constant_prob is not None else None
        self.model: xgb.Booster | None = None
        model_path = self._side_dir / "xgb_model.json"
        if model_path.exists():
            booster = xgb.Booster()
            booster.load_model(str(model_path))
            self.model = booster

    @staticmethod
    def _load_feature_cols(path: Path) -> list[str]:
        if not path.exists():
            raise FileNotFoundError(f"Missing feature list: {path}")
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.asarray([], dtype=np.float32)
        if self.constant_prob is not None:
            return np.full(len(frame), float(self.constant_prob), dtype=np.float32)
        if self.model is None:
            raise FileNotFoundError(f"Missing xgb_model.json under {self._side_dir}")
        aligned = frame.reindex(columns=self.feature_cols)
        dmat = xgb.DMatrix(aligned.to_numpy(dtype=np.float32), missing=np.nan)
        return self.model.predict(dmat).astype(np.float32, copy=False)

    def predict_row(self, frame: pd.DataFrame, *, target_ts: pd.Timestamp | None = None) -> float:
        if frame.empty:
            return float("nan")
        if target_ts is not None and target_ts in frame.index:
            row_df = frame.loc[[target_ts]]
        else:
            row_df = frame.tail(1)
        preds = self.predict_frame(row_df)
        if preds.size == 0:
            return float("nan")
        return float(preds[-1])


@dataclass
class _MetaTradeState:
    position: int = 0
    entry_price: float = np.nan
    entry_atr: float = np.nan
    bars_since_entry: int = 0
    favorable_anchor: float = np.nan
    adverse_anchor: float = np.nan
    tp_seen: bool = False


class LiveMetaXGBAgent:
    def __init__(
        self,
        *,
        model_root: str | Path,
        ga_model_root: str | None = None,
        ga_feature_list_path: str | None = None,
        include_pivot_probs: bool = True,
        include_tb_probs: bool = True,
        include_vix_features: bool = True,
        pivot_label_dir: str = "swing",
        tb_label_dir: str = "tb",
        tz: str | None = "America/New_York",
        assume_tz: str = "UTC",
        session_open: str = "09:30",
        session_close: str = "16:00",
        min_15m_bars: int = 20,
        fill_missing_prob: float = 0.0,
        resample_label: str = "left",
        resample_closed: str = "left",
        label_timeframe_rule: str = "10min",
        a_tp: float = 1.6,
        trail_activate_atr: float = 2.0,
        trail_atr: float = 1.0,
        trail_atr_after_tp: float = 0.8,
        use_tp_to_tighten_trail: bool = True,
        entry_threshold_override: float | None = None,
        exit_threshold_override: float | None = None,
        ga_probs_frame: pd.DataFrame | None = None,
        ga_probs_mode: str = "xgb",
        precomputed_base_frame: pd.DataFrame | None = None,
        precomputed_append_lookback_days: int = 120,
    ) -> None:
        self._model_root = Path(model_root)
        self._include_pivot_probs = bool(include_pivot_probs)
        self._include_tb_probs = bool(include_tb_probs)
        self._include_vix_features = bool(include_vix_features)
        self._tz = tz
        self._assume_tz = assume_tz
        self._session_open = session_open
        self._session_close = session_close
        self._min_bars = int(min_15m_bars)
        self._fill_missing_prob = float(fill_missing_prob)
        self._resample_label = resample_label
        self._resample_closed = resample_closed
        self._label_timeframe_rule = label_timeframe_rule
        self._a_tp = float(a_tp)
        self._trail_activate_atr = float(trail_activate_atr)
        self._trail_atr = float(trail_atr)
        self._trail_atr_after_tp = float(trail_atr_after_tp)
        self._use_tp_to_tighten_trail = bool(use_tp_to_tighten_trail)
        self._ga_probs_frame = ga_probs_frame
        self._ga_probs_mode = str(ga_probs_mode or "xgb").strip().lower()
        self._precomputed_base_frame = self._normalize_precomputed_base_frame(precomputed_base_frame)
        self._precomputed_append_lookback_days = max(1, int(precomputed_append_lookback_days))
        self._entry_threshold_override = (
            float(entry_threshold_override)
            if entry_threshold_override is not None and np.isfinite(entry_threshold_override)
            else None
        )
        self._exit_threshold_override = (
            float(exit_threshold_override)
            if exit_threshold_override is not None and np.isfinite(exit_threshold_override)
            else None
        )
        self._state = _MetaTradeState()
        self._last_probs: dict[str, float | None] | None = None
        self._last_prob_sources: dict[str, str | None] | None = None

        self._ga_predictor: LiveGAXGBPredictor | None = None
        if (self._include_pivot_probs or self._include_tb_probs) and ga_model_root and ga_feature_list_path:
            self._ga_predictor = LiveGAXGBPredictor(
                model_root=ga_model_root,
                feature_list_path=ga_feature_list_path,
                include_pivot_probs=self._include_pivot_probs,
                include_tb_probs=self._include_tb_probs,
                pivot_label_dir=pivot_label_dir,
                tb_label_dir=tb_label_dir,
            )

        self._entry_long = _LiveXGBArtifact(self._model_root / "entry" / "long")
        self._entry_short = _LiveXGBArtifact(self._model_root / "entry" / "short")
        self._exit_long = _LiveXGBArtifact(self._model_root / "exit" / "long")
        self._exit_short = _LiveXGBArtifact(self._model_root / "exit" / "short")
        self._entry_thresholds = self._load_thresholds(
            self._model_root / "entry" / "entry_thresholds.json",
            defaults={"enter_long": 0.8, "enter_short": 0.8},
        )
        self._exit_thresholds = self._load_thresholds(
            self._model_root / "exit" / "exit_thresholds.json",
            defaults={"exit_long": 0.8, "exit_short": 0.8},
        )
        if self._entry_threshold_override is not None:
            self._entry_thresholds["enter_long"] = float(self._entry_threshold_override)
            self._entry_thresholds["enter_short"] = float(self._entry_threshold_override)
        if self._exit_threshold_override is not None:
            self._exit_thresholds["exit_long"] = float(self._exit_threshold_override)
            self._exit_thresholds["exit_short"] = float(self._exit_threshold_override)

    @staticmethod
    def _load_thresholds(path: Path, *, defaults: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = dict(defaults)
        if not path.exists():
            print(f"[meta] Threshold file missing ({path}); using defaults: {defaults}")
            return out

        payload = json.loads(path.read_text(encoding="utf-8"))
        for old_key, new_key in (
            ("y_enter_long", "enter_long"),
            ("y_enter_short", "enter_short"),
            ("y_exit_long", "exit_long"),
            ("y_exit_short", "exit_short"),
        ):
            key = new_key if new_key in payload else old_key
            if key in payload and "threshold" in payload[key]:
                out[new_key] = float(payload[key]["threshold"])
        return out

    @staticmethod
    def _normalize_ts(ts: pd.Timestamp, *, assume_tz: str, tz: str | None) -> pd.Timestamp:
        if ts.tzinfo is None:
            ts = ts.tz_localize(assume_tz)
        if tz is not None:
            ts = ts.tz_convert(tz)
        return ts

    def _normalize_precomputed_base_frame(self, frame: pd.DataFrame | None) -> pd.DataFrame | None:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return None
        out = frame.copy()
        if "timestamp" in out.columns:
            ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
            out = out.loc[ts.notna()].copy()
            out.index = ts[ts.notna()]
        elif not isinstance(out.index, pd.DatetimeIndex):
            return None
        if out.index.tz is None:
            out.index = out.index.tz_localize(self._assume_tz)
        if self._tz is not None:
            out.index = out.index.tz_convert(self._tz)
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
        return out

    def _reset_trade_state(self) -> None:
        self._state = _MetaTradeState()

    def _set_trade_entry(self, *, position: int, row: pd.Series, entry_price: float | None = None) -> None:
        close = float(row.get("close", np.nan))
        atr = float(row.get("atr", np.nan))
        seed_entry = float(entry_price) if entry_price is not None and np.isfinite(entry_price) else close
        if not np.isfinite(seed_entry):
            seed_entry = close
        self._state = _MetaTradeState(
            position=int(position),
            entry_price=float(seed_entry),
            entry_atr=float(atr) if np.isfinite(atr) and atr > 0.0 else float("nan"),
            bars_since_entry=0,
            favorable_anchor=float(close),
            adverse_anchor=float(close),
            tp_seen=False,
        )

    def _build_base_frame(self, *, df_1m: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self._precomputed_base_frame, pd.DataFrame) and not self._precomputed_base_frame.empty:
            pre = self._precomputed_base_frame
            if df_1m is None or df_1m.empty:
                return pre

            if not isinstance(df_1m.index, pd.DatetimeIndex):
                if "timestamp" not in df_1m.columns:
                    return pre
                raw = df_1m.copy()
                raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
                raw = raw.dropna(subset=["timestamp"]).set_index("timestamp")
            else:
                raw = df_1m.copy()

            if raw.index.tz is None:
                raw.index = raw.index.tz_localize(self._assume_tz)
            if self._tz is not None:
                raw.index = raw.index.tz_convert(self._tz)
            raw = raw.sort_index()
            if raw.empty:
                return pre

            try:
                latest_base_ts = build_15m(
                    raw,
                    rule=self._label_timeframe_rule,
                    label=self._resample_label,
                    closed=self._resample_closed,
                    tz=self._tz,
                    assume_tz=self._assume_tz,
                ).index.max()
            except Exception:
                latest_base_ts = None

            cached_max = pre.index.max()
            if latest_base_ts is None or pd.isna(latest_base_ts) or latest_base_ts <= cached_max:
                return pre

            overlap_start = cached_max - pd.Timedelta(days=self._precomputed_append_lookback_days)
            raw_tail = raw.loc[raw.index >= overlap_start].copy()
            computed_tail = build_meta_feature_frame_from_1m(
                raw_tail,
                rule=self._label_timeframe_rule,
                label=self._resample_label,
                closed=self._resample_closed,
                tz=self._tz,
                assume_tz=self._assume_tz,
                include_pivot_probs=self._include_pivot_probs,
                include_tb_probs=self._include_tb_probs,
                include_vix_features=self._include_vix_features,
                fill_missing_prob=self._fill_missing_prob,
                session_open=self._session_open,
                session_close=self._session_close,
                ga_predictor=self._ga_predictor,
                ga_probs_frame=self._ga_probs_frame,
                ga_probs_mode=self._ga_probs_mode,
            )
            if computed_tail.empty:
                return pre
            merged = pd.concat(
                [pre.loc[pre.index < computed_tail.index.min()], computed_tail],
                axis=0,
            ).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            self._precomputed_base_frame = merged
            return merged
        return build_meta_feature_frame_from_1m(
            df_1m,
            rule=self._label_timeframe_rule,
            label=self._resample_label,
            closed=self._resample_closed,
            tz=self._tz,
            assume_tz=self._assume_tz,
            include_pivot_probs=self._include_pivot_probs,
            include_tb_probs=self._include_tb_probs,
            include_vix_features=self._include_vix_features,
            fill_missing_prob=self._fill_missing_prob,
            session_open=self._session_open,
            session_close=self._session_close,
            ga_predictor=self._ga_predictor,
            ga_probs_frame=self._ga_probs_frame,
            ga_probs_mode=self._ga_probs_mode,
        )

    @staticmethod
    def _extract_last_prob_sources(base_frame: pd.DataFrame, ts: pd.Timestamp) -> dict[str, str | None] | None:
        src_df = base_frame.attrs.get("ga_prob_sources")
        if not isinstance(src_df, pd.DataFrame) or src_df.empty:
            return None
        if ts not in src_df.index:
            return None
        row = src_df.loc[ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        out: dict[str, str | None] = {}
        for key in ("p_pivot_long", "p_pivot_short", "p_tb_long", "p_tb_short"):
            value = row.get(key)
            out[f"{key}_source"] = None if pd.isna(value) else str(value)
        return out

    def _annotate_current_context(self, row: pd.Series) -> pd.Series:
        out = row.copy()
        for side in ("long", "short"):
            out[f"in_{side}_trade"] = 0
            out[f"{side}_bars_since_entry"] = np.nan
            out[f"{side}_mfe_atr"] = np.nan
            out[f"{side}_mae_atr"] = np.nan
            out[f"{side}_tp_seen_run"] = 0
            out[f"{side}_trail_gap_atr"] = np.nan
            out[f"{side}_entry_price_ctx"] = np.nan
        if self._state.position == 0:
            return out

        entry = float(self._state.entry_price)
        atr_i = float(self._state.entry_atr)
        high = float(row.get("high", np.nan))
        low = float(row.get("low", np.nan))
        close = float(row.get("close", np.nan))
        if not np.isfinite(entry) or not np.isfinite(atr_i) or atr_i <= 0.0:
            return out

        side = "long" if self._state.position > 0 else "short"
        out[f"in_{side}_trade"] = 1
        out[f"{side}_bars_since_entry"] = float(self._state.bars_since_entry + 1)
        out[f"{side}_entry_price_ctx"] = entry

        if side == "long":
            favorable_anchor = max(float(self._state.favorable_anchor), high) if np.isfinite(high) else float(self._state.favorable_anchor)
            adverse_anchor = min(float(self._state.adverse_anchor), low) if np.isfinite(low) else float(self._state.adverse_anchor)
            tp = entry + self._a_tp * atr_i
            tp_seen = bool(self._state.tp_seen or (np.isfinite(high) and high >= tp))
            trail_active = (favorable_anchor - entry) >= self._trail_activate_atr * atr_i
            trail_dist = self._trail_atr * atr_i
            if tp_seen and self._use_tp_to_tighten_trail:
                trail_active = True
                trail_dist = min(trail_dist, max(self._trail_atr_after_tp, 1e-9) * atr_i)
            trail_level = favorable_anchor - trail_dist if trail_active else np.nan
            out["long_mfe_atr"] = (favorable_anchor - entry) / atr_i
            out["long_mae_atr"] = (entry - adverse_anchor) / atr_i
            out["long_tp_seen_run"] = int(tp_seen)
            out["long_trail_gap_atr"] = (
                (close - trail_level) / atr_i if np.isfinite(close) and np.isfinite(trail_level) else np.nan
            )
        else:
            favorable_anchor = min(float(self._state.favorable_anchor), low) if np.isfinite(low) else float(self._state.favorable_anchor)
            adverse_anchor = max(float(self._state.adverse_anchor), high) if np.isfinite(high) else float(self._state.adverse_anchor)
            tp = entry - self._a_tp * atr_i
            tp_seen = bool(self._state.tp_seen or (np.isfinite(low) and low <= tp))
            trail_active = (entry - favorable_anchor) >= self._trail_activate_atr * atr_i
            trail_dist = self._trail_atr * atr_i
            if tp_seen and self._use_tp_to_tighten_trail:
                trail_active = True
                trail_dist = min(trail_dist, max(self._trail_atr_after_tp, 1e-9) * atr_i)
            trail_level = favorable_anchor + trail_dist if trail_active else np.nan
            out["short_mfe_atr"] = (entry - favorable_anchor) / atr_i
            out["short_mae_atr"] = (adverse_anchor - entry) / atr_i
            out["short_tp_seen_run"] = int(tp_seen)
            out["short_trail_gap_atr"] = (
                (trail_level - close) / atr_i if np.isfinite(close) and np.isfinite(trail_level) else np.nan
            )
        return out

    def _decide_action(self, *, p_enter_long: float, p_enter_short: float, p_exit_long: float, p_exit_short: float) -> int:
        if self._state.position > 0:
            return 0 if np.isfinite(p_exit_long) and p_exit_long >= self._exit_thresholds["exit_long"] else 1
        if self._state.position < 0:
            return 0 if np.isfinite(p_exit_short) and p_exit_short >= self._exit_thresholds["exit_short"] else -1

        long_thr = self._entry_thresholds["enter_long"]
        short_thr = self._entry_thresholds["enter_short"]
        long_ready = np.isfinite(p_enter_long) and p_enter_long >= long_thr
        short_ready = np.isfinite(p_enter_short) and p_enter_short >= short_thr
        if long_ready and short_ready:
            long_margin = p_enter_long - long_thr
            short_margin = p_enter_short - short_thr
            if abs(long_margin - short_margin) <= 1e-9:
                return 0
            return 1 if long_margin > short_margin else -1
        if long_ready:
            return 1
        if short_ready:
            return -1
        return 0

    def _advance_state(self, *, action: int, row: pd.Series) -> None:
        high = float(row.get("high", np.nan))
        low = float(row.get("low", np.nan))
        close = float(row.get("close", np.nan))
        atr = float(row.get("atr", np.nan))

        if self._state.position != 0:
            self._state.bars_since_entry += 1
            if self._state.position > 0:
                if np.isfinite(high):
                    self._state.favorable_anchor = max(float(self._state.favorable_anchor), high)
                if np.isfinite(low):
                    self._state.adverse_anchor = min(float(self._state.adverse_anchor), low)
                tp = float(self._state.entry_price) + self._a_tp * float(self._state.entry_atr)
                if np.isfinite(high) and high >= tp:
                    self._state.tp_seen = True
            else:
                if np.isfinite(low):
                    self._state.favorable_anchor = min(float(self._state.favorable_anchor), low)
                if np.isfinite(high):
                    self._state.adverse_anchor = max(float(self._state.adverse_anchor), high)
                tp = float(self._state.entry_price) - self._a_tp * float(self._state.entry_atr)
                if np.isfinite(low) and low <= tp:
                    self._state.tp_seen = True

        if action == 0:
            self._reset_trade_state()
            return
        if self._state.position == action:
            return
        if not np.isfinite(close):
            return
        self._state = _MetaTradeState(
            position=int(action),
            entry_price=float(close),
            entry_atr=float(atr) if np.isfinite(atr) and atr > 0.0 else float("nan"),
            bars_since_entry=0,
            favorable_anchor=float(close),
            adverse_anchor=float(close),
            tp_seen=False,
        )

    def sync_live_position(
        self,
        *,
        desired_position: int,
        df_1m: pd.DataFrame,
        entry_price: float | None = None,
    ) -> dict[str, object]:
        side = 1 if int(desired_position) > 0 else (-1 if int(desired_position) < 0 else 0)
        if side == 0:
            self._reset_trade_state()
            return {"synced": True, "position": 0, "reason": "flat"}

        base_frame = self._build_base_frame(df_1m=df_1m)
        if base_frame.empty:
            self._reset_trade_state()
            self._state.position = side
            if entry_price is not None and np.isfinite(entry_price):
                self._state.entry_price = float(entry_price)
            return {"synced": False, "position": side, "reason": "empty_feature_frame"}

        side_probs = (
            self._entry_long.predict_frame(base_frame)
            if side > 0
            else self._entry_short.predict_frame(base_frame)
        )
        side_thr = self._entry_thresholds["enter_long"] if side > 0 else self._entry_thresholds["enter_short"]
        above = np.isfinite(side_probs) & (side_probs >= float(side_thr))
        rising = above.copy()
        if rising.size:
            rising[1:] = above[1:] & (~above[:-1])
        candidate_idx = np.flatnonzero(rising)
        if candidate_idx.size:
            start_idx = int(candidate_idx[-1])
            seed_mode = "threshold_cross"
        else:
            active_idx = np.flatnonzero(above)
            if active_idx.size:
                start_idx = int(active_idx[-1])
                seed_mode = "threshold_active"
            else:
                start_idx = int(len(base_frame) - 1)
                seed_mode = "latest_bar"

        seed_rows = base_frame.iloc[start_idx:]
        self._reset_trade_state()
        first = True
        for _, row in seed_rows.iterrows():
            if first:
                self._set_trade_entry(position=side, row=row, entry_price=entry_price)
                first = False
                continue
            self._advance_state(action=side, row=row)

        last_row = base_frame.iloc[-1].copy()
        self._last_prob_sources = self._extract_last_prob_sources(base_frame, last_row.name)
        p_enter_long = self._entry_long.predict_row(base_frame.tail(1), target_ts=last_row.name)
        p_enter_short = self._entry_short.predict_row(base_frame.tail(1), target_ts=last_row.name)
        last_row["p_enter_long_oof"] = p_enter_long
        last_row["p_enter_short_oof"] = p_enter_short
        exit_row = self._annotate_current_context(last_row)
        exit_df = pd.DataFrame([exit_row], index=[last_row.name])
        p_exit_long = self._exit_long.predict_row(exit_df, target_ts=last_row.name) if side > 0 else float("nan")
        p_exit_short = self._exit_short.predict_row(exit_df, target_ts=last_row.name) if side < 0 else float("nan")
        self._last_probs = {
            "p_pivot_long": float(last_row.get("p_pivot_long", np.nan)),
            "p_pivot_short": float(last_row.get("p_pivot_short", np.nan)),
            "p_tb_long": float(last_row.get("p_tb_long", np.nan)),
            "p_tb_short": float(last_row.get("p_tb_short", np.nan)),
            "p_enter_long": p_enter_long,
            "p_enter_short": p_enter_short,
            "p_exit_long": p_exit_long,
            "p_exit_short": p_exit_short,
        }
        for key, value in list(self._last_probs.items()):
            if not np.isfinite(value):
                self._last_probs[key] = None
        return {
            "synced": True,
            "position": side,
            "seed_mode": seed_mode,
            "seed_start_ts": str(seed_rows.index[0]),
            "bars_since_entry": int(self._state.bars_since_entry),
            "entry_price": float(self._state.entry_price) if np.isfinite(self._state.entry_price) else None,
        }

    def act(
        self,
        *,
        df_1m: pd.DataFrame,
        df_15m: pd.DataFrame,
        target_ts: pd.Timestamp | None = None,
    ) -> Optional[float]:
        del df_15m
        base_frame = self._build_base_frame(df_1m=df_1m)

        ts = target_ts
        if ts is not None:
            ts = self._normalize_ts(pd.to_datetime(ts, utc=True, errors="coerce"), assume_tz=self._assume_tz, tz=self._tz)
            if ts in base_frame.index:
                base_frame = base_frame.loc[:ts]
        if base_frame.empty or len(base_frame) < self._min_bars:
            return None
        row_df = base_frame.loc[[ts]] if ts is not None and ts in base_frame.index else base_frame.tail(1)
        row = row_df.iloc[-1].copy()
        self._last_prob_sources = self._extract_last_prob_sources(base_frame, row.name)

        p_enter_long = self._entry_long.predict_row(row_df, target_ts=row.name)
        p_enter_short = self._entry_short.predict_row(row_df, target_ts=row.name)
        row["p_enter_long_oof"] = p_enter_long
        row["p_enter_short_oof"] = p_enter_short
        exit_row = self._annotate_current_context(row)
        exit_df = pd.DataFrame([exit_row], index=[row.name])
        p_exit_long = self._exit_long.predict_row(exit_df, target_ts=row.name) if self._state.position > 0 else float("nan")
        p_exit_short = self._exit_short.predict_row(exit_df, target_ts=row.name) if self._state.position < 0 else float("nan")

        action = self._decide_action(
            p_enter_long=p_enter_long,
            p_enter_short=p_enter_short,
            p_exit_long=p_exit_long,
            p_exit_short=p_exit_short,
        )
        self._last_probs = {
            "p_pivot_long": float(row.get("p_pivot_long", np.nan)),
            "p_pivot_short": float(row.get("p_pivot_short", np.nan)),
            "p_tb_long": float(row.get("p_tb_long", np.nan)),
            "p_tb_short": float(row.get("p_tb_short", np.nan)),
            "p_enter_long": p_enter_long,
            "p_enter_short": p_enter_short,
            "p_exit_long": p_exit_long,
            "p_exit_short": p_exit_short,
        }
        for key, value in list(self._last_probs.items()):
            if not np.isfinite(value):
                self._last_probs[key] = None
        self._advance_state(action=action, row=row)
        return float(action)

    def replay_warmup_actions(
        self,
        *,
        df_1m: pd.DataFrame,
        df_15m: pd.DataFrame,
        apply_ga_probs: bool = True,
    ) -> list[dict[str, object]]:
        del df_15m, apply_ga_probs
        base_frame = self._build_base_frame(df_1m=df_1m)
        if base_frame.empty or len(base_frame) < self._min_bars:
            return []
        entry_long_probs = self._entry_long.predict_frame(base_frame)
        entry_short_probs = self._entry_short.predict_frame(base_frame)
        out: list[dict[str, object]] = []
        self._reset_trade_state()
        for idx, (_, row) in enumerate(base_frame.iterrows()):
            p_enter_long = float(entry_long_probs[idx]) if idx < entry_long_probs.size else float("nan")
            p_enter_short = float(entry_short_probs[idx]) if idx < entry_short_probs.size else float("nan")
            row = row.copy()
            row["p_enter_long_oof"] = p_enter_long
            row["p_enter_short_oof"] = p_enter_short
            exit_row = self._annotate_current_context(row)
            exit_df = pd.DataFrame([exit_row], index=[row.name])
            p_exit_long = self._exit_long.predict_row(exit_df, target_ts=row.name) if self._state.position > 0 else float("nan")
            p_exit_short = self._exit_short.predict_row(exit_df, target_ts=row.name) if self._state.position < 0 else float("nan")
            action = self._decide_action(
                p_enter_long=p_enter_long,
                p_enter_short=p_enter_short,
                p_exit_long=p_exit_long,
                p_exit_short=p_exit_short,
            )
            self._last_probs = {
                "p_pivot_long": float(row.get("p_pivot_long", np.nan)),
                "p_pivot_short": float(row.get("p_pivot_short", np.nan)),
                "p_tb_long": float(row.get("p_tb_long", np.nan)),
                "p_tb_short": float(row.get("p_tb_short", np.nan)),
                "p_enter_long": p_enter_long,
                "p_enter_short": p_enter_short,
                "p_exit_long": p_exit_long,
                "p_exit_short": p_exit_short,
            }
            for key, value in list(self._last_probs.items()):
                if not np.isfinite(value):
                    self._last_probs[key] = None
            self._last_prob_sources = self._extract_last_prob_sources(base_frame, row.name)
            self._advance_state(action=action, row=row)
            out.append({"timestamp": row.name, "action": float(action), "close": float(row.get("close", np.nan))})
        return out

    def snapshot_state(self) -> dict[str, object]:
        return {
            "position": float(self._state.position),
            "entry_price": float(self._state.entry_price) if np.isfinite(self._state.entry_price) else None,
            "time_in_position": int(self._state.bars_since_entry),
            "last_probs": self._last_probs,
            "last_prob_sources": self._last_prob_sources,
        }

    def last_probs(self) -> dict[str, float | None] | None:
        return self._last_probs

    def last_thresholds(self) -> dict[str, float] | None:
        return {
            "enter_long": float(self._entry_thresholds.get("enter_long", float("nan"))),
            "enter_short": float(self._entry_thresholds.get("enter_short", float("nan"))),
            "exit_long": float(self._exit_thresholds.get("exit_long", float("nan"))),
            "exit_short": float(self._exit_thresholds.get("exit_short", float("nan"))),
        }


class LiveIndependentMetaXGBAgent:
    """
    Replay-compatible independent long/short meta scorer for live use.

    This keeps separate long and short trade state so exit probabilities are
    computed the same way as the independent offline replay, instead of being
    tied to one signed position state.
    """

    def __init__(
        self,
        *,
        model_root: str | Path,
        ga_model_root: str | None = None,
        ga_feature_list_path: str | None = None,
        include_pivot_probs: bool = True,
        include_tb_probs: bool = True,
        include_vix_features: bool = True,
        pivot_label_dir: str = "swing",
        tb_label_dir: str = "tb",
        tz: str | None = "America/New_York",
        assume_tz: str = "UTC",
        session_open: str = "09:30",
        session_close: str = "16:00",
        min_15m_bars: int = 20,
        fill_missing_prob: float = 0.0,
        resample_label: str = "left",
        resample_closed: str = "left",
        label_timeframe_rule: str = "10min",
        a_tp: float = 1.6,
        trail_activate_atr: float = 2.0,
        trail_atr: float = 1.0,
        trail_atr_after_tp: float = 0.8,
        use_tp_to_tighten_trail: bool = True,
        entry_threshold_override: float | None = None,
        exit_threshold_override: float | None = None,
        ga_probs_frame: pd.DataFrame | None = None,
        ga_probs_mode: str = "xgb",
        precomputed_base_frame: pd.DataFrame | None = None,
        precomputed_append_lookback_days: int = 120,
        min_hold_bars: int = 2,
        exit_entry_delta: float = 0.15,
    ) -> None:
        common_kwargs = dict(
            model_root=model_root,
            ga_model_root=ga_model_root,
            ga_feature_list_path=ga_feature_list_path,
            include_pivot_probs=include_pivot_probs,
            include_tb_probs=include_tb_probs,
            include_vix_features=include_vix_features,
            pivot_label_dir=pivot_label_dir,
            tb_label_dir=tb_label_dir,
            tz=tz,
            assume_tz=assume_tz,
            session_open=session_open,
            session_close=session_close,
            min_15m_bars=min_15m_bars,
            fill_missing_prob=fill_missing_prob,
            resample_label=resample_label,
            resample_closed=resample_closed,
            label_timeframe_rule=label_timeframe_rule,
            a_tp=a_tp,
            trail_activate_atr=trail_activate_atr,
            trail_atr=trail_atr,
            trail_atr_after_tp=trail_atr_after_tp,
            use_tp_to_tighten_trail=use_tp_to_tighten_trail,
            entry_threshold_override=entry_threshold_override,
            exit_threshold_override=exit_threshold_override,
            ga_probs_frame=ga_probs_frame,
            ga_probs_mode=ga_probs_mode,
            precomputed_base_frame=precomputed_base_frame,
            precomputed_append_lookback_days=precomputed_append_lookback_days,
        )
        self._base_agent = LiveMetaXGBAgent(**common_kwargs)
        self._long_agent = LiveMetaXGBAgent(**common_kwargs)
        self._short_agent = LiveMetaXGBAgent(**common_kwargs)
        self._min_bars = int(self._base_agent._min_bars)
        self._min_hold_bars = max(0, int(min_hold_bars))
        self._exit_entry_delta = float(exit_entry_delta)
        self._entry_thresholds = dict(self._base_agent._entry_thresholds)
        self._exit_thresholds = dict(self._base_agent._exit_thresholds)
        self._long_active = False
        self._short_active = False
        self._long_bars_held = -1
        self._short_bars_held = -1
        self._last_probs: dict[str, float | None] | None = None
        self._last_prob_sources: dict[str, str | None] | None = None
        self._last_processed_ts: pd.Timestamp | None = None

    def _reset_state(self) -> None:
        self._long_agent._reset_trade_state()
        self._short_agent._reset_trade_state()
        self._long_active = False
        self._short_active = False
        self._long_bars_held = -1
        self._short_bars_held = -1
        self._last_probs = None
        self._last_prob_sources = None
        self._last_processed_ts = None

    def _row_with_entries(self, base_frame: pd.DataFrame, row: pd.Series) -> tuple[pd.Series, float, float]:
        p_enter_long = self._base_agent._entry_long.predict_row(base_frame, target_ts=row.name)
        p_enter_short = self._base_agent._entry_short.predict_row(base_frame, target_ts=row.name)
        work_row = row.copy()
        work_row["p_enter_long_oof"] = p_enter_long
        work_row["p_enter_short_oof"] = p_enter_short
        return work_row, float(p_enter_long), float(p_enter_short)

    def _score_row(self, base_frame: pd.DataFrame, row: pd.Series) -> tuple[pd.Series, dict[str, float | None]]:
        work_row, p_enter_long, p_enter_short = self._row_with_entries(base_frame, row)
        if self._long_active:
            exit_row_long = self._long_agent._annotate_current_context(work_row)
            exit_df_long = pd.DataFrame([exit_row_long], index=[row.name])
            p_exit_long = float(self._base_agent._exit_long.predict_row(exit_df_long, target_ts=row.name))
        else:
            p_exit_long = float("nan")
        if self._short_active:
            exit_row_short = self._short_agent._annotate_current_context(work_row)
            exit_df_short = pd.DataFrame([exit_row_short], index=[row.name])
            p_exit_short = float(self._base_agent._exit_short.predict_row(exit_df_short, target_ts=row.name))
        else:
            p_exit_short = float("nan")
        probs = {
            "p_pivot_long": float(row.get("p_pivot_long", np.nan)),
            "p_pivot_short": float(row.get("p_pivot_short", np.nan)),
            "p_tb_long": float(row.get("p_tb_long", np.nan)),
            "p_tb_short": float(row.get("p_tb_short", np.nan)),
            "p_enter_long": p_enter_long,
            "p_enter_short": p_enter_short,
            "p_exit_long": p_exit_long,
            "p_exit_short": p_exit_short,
        }
        return work_row, probs

    def _advance_independent_state(self, *, work_row: pd.Series, probs: dict[str, float | None]) -> int:
        p_enter_long = float(probs.get("p_enter_long", np.nan))
        p_enter_short = float(probs.get("p_enter_short", np.nan))
        p_exit_long = float(probs.get("p_exit_long", np.nan))
        p_exit_short = float(probs.get("p_exit_short", np.nan))

        long_exit_threshold_hit = bool(
            self._long_active and np.isfinite(p_exit_long) and p_exit_long >= float(self._exit_thresholds["exit_long"])
        )
        short_exit_threshold_hit = bool(
            self._short_active and np.isfinite(p_exit_short) and p_exit_short >= float(self._exit_thresholds["exit_short"])
        )
        long_hold_ready = bool(self._long_active and self._long_bars_held >= self._min_hold_bars)
        short_hold_ready = bool(self._short_active and self._short_bars_held >= self._min_hold_bars)

        long_entry_still_supports = bool(
            np.isfinite(p_enter_long)
            and p_enter_long >= float(self._entry_thresholds["enter_long"])
            and (not np.isfinite(p_exit_long) or (p_exit_long - p_enter_long) < self._exit_entry_delta)
        )
        short_entry_still_supports = bool(
            np.isfinite(p_enter_short)
            and p_enter_short >= float(self._entry_thresholds["enter_short"])
            and (not np.isfinite(p_exit_short) or (p_exit_short - p_enter_short) < self._exit_entry_delta)
        )

        do_exit_long = bool(long_exit_threshold_hit and long_hold_ready and not long_entry_still_supports)
        do_exit_short = bool(short_exit_threshold_hit and short_hold_ready and not short_entry_still_supports)
        do_entry_long = bool((not self._long_active) and np.isfinite(p_enter_long) and p_enter_long >= float(self._entry_thresholds["enter_long"]))
        do_entry_short = bool((not self._short_active) and np.isfinite(p_enter_short) and p_enter_short >= float(self._entry_thresholds["enter_short"]))

        next_long_active = bool((self._long_active and not do_exit_long) or do_entry_long)
        next_short_active = bool((self._short_active and not do_exit_short) or do_entry_short)

        self._long_agent._advance_state(action=1 if next_long_active else 0, row=work_row)
        self._short_agent._advance_state(action=-1 if next_short_active else 0, row=work_row)
        self._long_active = next_long_active
        self._short_active = next_short_active
        if do_entry_long:
            self._long_bars_held = 0
        elif self._long_active:
            self._long_bars_held = max(0, self._long_bars_held + 1)
        else:
            self._long_bars_held = -1
        if do_entry_short:
            self._short_bars_held = 0
        elif self._short_active:
            self._short_bars_held = max(0, self._short_bars_held + 1)
        else:
            self._short_bars_held = -1

        if self._long_active and not self._short_active:
            return 1
        if self._short_active and not self._long_active:
            return -1
        return 0

    def _process_rows(self, *, base_frame: pd.DataFrame, rows: pd.DataFrame) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for _, row in rows.iterrows():
            work_row, probs = self._score_row(base_frame, row)
            self._last_probs = {
                key: (None if not np.isfinite(val) else float(val))
                for key, val in probs.items()
            }
            self._last_prob_sources = self._base_agent._extract_last_prob_sources(base_frame, row.name)
            action = self._advance_independent_state(work_row=work_row, probs=probs)
            self._last_processed_ts = pd.Timestamp(row.name)
            out.append({"timestamp": row.name, "action": float(action), "close": float(row.get("close", np.nan))})
        return out

    def replay_warmup_actions(
        self,
        *,
        df_1m: pd.DataFrame,
        df_15m: pd.DataFrame,
        apply_ga_probs: bool = True,
    ) -> list[dict[str, object]]:
        del df_15m, apply_ga_probs
        base_frame = self._base_agent._build_base_frame(df_1m=df_1m)
        if base_frame.empty or len(base_frame) < self._min_bars:
            return []
        self._reset_state()
        return self._process_rows(base_frame=base_frame, rows=base_frame)

    def act(self, *, df_1m: pd.DataFrame, df_15m: pd.DataFrame, target_ts: pd.Timestamp | None = None) -> float | None:
        del df_15m
        base_frame = self._base_agent._build_base_frame(df_1m=df_1m)
        if base_frame.empty or len(base_frame) < self._min_bars:
            return None
        if target_ts is not None:
            ts = pd.to_datetime(target_ts, utc=True, errors="coerce")
            if pd.isna(ts) or ts not in base_frame.index:
                rows = base_frame.tail(1)
            else:
                rows = base_frame.loc[base_frame.index > self._last_processed_ts] if self._last_processed_ts is not None else base_frame
                rows = rows.loc[rows.index <= ts]
                if rows.empty:
                    rows = base_frame.loc[[ts]]
        else:
            rows = base_frame.loc[base_frame.index > self._last_processed_ts] if self._last_processed_ts is not None else base_frame.tail(1)
            if rows.empty:
                rows = base_frame.tail(1)
        actions = self._process_rows(base_frame=base_frame, rows=rows)
        if not actions:
            return None
        return float(actions[-1]["action"])

    def snapshot_state(self) -> dict[str, object]:
        return {
            "position": float((1 if self._long_active else 0) + (-1 if self._short_active else 0)),
            "long_active": int(self._long_active),
            "short_active": int(self._short_active),
            "long_bars_held": int(self._long_bars_held),
            "short_bars_held": int(self._short_bars_held),
            "last_probs": self._last_probs,
            "last_prob_sources": self._last_prob_sources,
        }

    def last_probs(self) -> dict[str, float | None] | None:
        return self._last_probs

    def last_thresholds(self) -> dict[str, float] | None:
        return {
            "enter_long": float(self._entry_thresholds.get("enter_long", float("nan"))),
            "enter_short": float(self._entry_thresholds.get("enter_short", float("nan"))),
            "exit_long": float(self._exit_thresholds.get("exit_long", float("nan"))),
            "exit_short": float(self._exit_thresholds.get("exit_short", float("nan"))),
        }


@dataclass
class _PositionState:
    position: float = 0.0
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
        require_probs: bool = True,
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
        self._action_type = str(ckpt.get("action_type", "discrete")).strip().lower()
        self._hybrid_action = self._action_type in {"hybrid", "hybrid_dir_mag"}
        self._continuous_action = self._action_type in {
            "continuous",
            "continuous_tanh",
            "gaussian_tanh",
        } or self._hybrid_action
        self._n_actions = int(ckpt.get("n_actions", 3))
        self._action_dim = int(ckpt.get("action_dim", 1))
        self._action_low = float(ckpt.get("action_low", -1.0))
        self._action_high = float(ckpt.get("action_high", 1.0))
        self._action_deadband = max(0.0, float(ckpt.get("action_deadband", 1e-3)))
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
        self._require_probs = bool(require_probs)
        self._warned_ga_frame = False
        self._last_probs: dict[str, float | None] | None = None
        self._warned_missing_live_probs = False

        state_dict = ckpt["state_dict"]
        has_head_mlps = any(
            k.startswith("policy_mlp.") or k.startswith("value_mlp.")
            for k in state_dict
        )
        policy_head_mlp = bool(ckpt.get("policy_head_mlp", has_head_mlps))
        policy_hidden_size = int(ckpt.get("policy_hidden_size", 128))
        policy_layer_norm = bool(ckpt.get("policy_layer_norm", False))
        policy_dropout_p = float(ckpt.get("policy_dropout_p", 0.0))
        self._model = ActorCritic(
            obs_dim=self._obs_dim,
            n_actions=self._n_actions,
            action_type=self._action_type,
            action_dim=self._action_dim,
            hidden=policy_hidden_size,
            head_mlp=policy_head_mlp,
            use_layer_norm=policy_layer_norm,
            dropout_p=policy_dropout_p,
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
                # Hybrid: frame values override XGB where available.
                aligned_col = aligned[col].astype(float)
                if col in df.columns:
                    df[col] = df[col].where(aligned_col.isna(), aligned_col)
                else:
                    df[col] = aligned_col
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

    def _is_flat(self, pos: float) -> bool:
        return abs(float(pos)) <= self._action_deadband

    def _action_to_exposure(self, action: float) -> float:
        if self._hybrid_action:
            if isinstance(action, (list, tuple, np.ndarray)):
                arr = np.asarray(action, dtype=np.float32).reshape(-1)
                out = float(arr[0]) if arr.size else 0.0
            else:
                try:
                    out = float(action)
                except (TypeError, ValueError):
                    out = 0.0
            if not np.isfinite(out):
                out = 0.0
            out = float(np.clip(out, self._action_low, self._action_high))
            if abs(out) <= self._action_deadband:
                return 0.0
            return out
        if self._continuous_action:
            try:
                out = float(action)
            except (TypeError, ValueError):
                out = 0.0
            if not np.isfinite(out):
                out = 0.0
            out = float(np.clip(out, self._action_low, self._action_high))
            if abs(out) <= self._action_deadband:
                return 0.0
            return out
        # Backward compatibility for discrete checkpoints.
        try:
            act = int(action)
        except (TypeError, ValueError):
            act = 0
        return 0.0 if act == 0 else (1.0 if act == 1 else -1.0)

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
            if (not self._is_flat(self._state.position)) and np.isfinite(self._state.entry_price):
                self._state.unrealized_pnl = (
                    price / self._state.entry_price - 1.0
                ) * float(self._state.position)
            else:
                self._state.unrealized_pnl = 0.0
                if self._is_flat(self._state.position):
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
        if (not self._is_flat(self._state.position)) and np.isfinite(self._state.entry_price):
            self._state.unrealized_pnl = (
                price / self._state.entry_price - 1.0
            ) * float(self._state.position)
        else:
            self._state.unrealized_pnl = 0.0
            if self._is_flat(self._state.position):
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
    ) -> Optional[float]:
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

        if self._require_probs:
            missing_prob_keys: list[str] = []
            if self._include_pivot_probs:
                if self._last_probs.get("p_pivot_long") is None:
                    missing_prob_keys.append("p_pivot_long")
                if self._last_probs.get("p_pivot_short") is None:
                    missing_prob_keys.append("p_pivot_short")
            if self._include_tb_probs:
                if self._last_probs.get("p_tb_long") is None:
                    missing_prob_keys.append("p_tb_long")
                if self._last_probs.get("p_tb_short") is None:
                    missing_prob_keys.append("p_tb_short")
            if missing_prob_keys:
                if not self._warned_missing_live_probs:
                    print(
                        "[live] Skipping decision: missing required probs "
                        f"for current bar: {sorted(set(missing_prob_keys))}"
                    )
                    self._warned_missing_live_probs = True
                return None
            self._warned_missing_live_probs = False

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
        policy_out, _ = self._model(x)
        if self._hybrid_action:
            if self._deterministic:
                action_t, _logp, _entropy = self._model._sample_hybrid(  # noqa: SLF001
                    policy_out,
                    deterministic=True,
                )
            else:
                action_t, _logp, _entropy = self._model._sample_hybrid(  # noqa: SLF001
                    policy_out,
                    deterministic=False,
                )
            action = float(action_t[:, 0].squeeze().item())
        elif self._continuous_action:
            if self._deterministic:
                action_t = torch.tanh(policy_out)
            else:
                action_t, _logp, _entropy = self._model._sample_continuous(  # noqa: SLF001
                    policy_out,
                    deterministic=False,
                )
            action = float(action_t.squeeze().item())
        else:
            if self._deterministic:
                class_action = int(torch.argmax(policy_out, dim=-1).item())
            else:
                dist = torch.distributions.Categorical(logits=policy_out)
                class_action = int(dist.sample().item())
            action = self._action_to_exposure(float(class_action))

        self._apply_action(action, price)
        return action

    def _apply_action(self, action: float, price: float) -> None:
        prev_pos = float(self._state.position)
        desired_pos = self._action_to_exposure(action)
        trade_units = abs(desired_pos - prev_pos)

        if trade_units > self._action_deadband:
            if (not self._is_flat(prev_pos)) and np.isfinite(self._state.entry_price):
                same_sign = prev_pos * desired_pos > 0.0
                prev_abs = abs(prev_pos)
                desired_abs = abs(desired_pos)
                closed_units = max(0.0, prev_abs - desired_abs) if same_sign else prev_abs
                if closed_units > self._action_deadband:
                    move_per_unit = (price / self._state.entry_price - 1.0) * float(np.sign(prev_pos))
                    self._state.realized_pnl_today += move_per_unit * closed_units

            if self._is_flat(desired_pos):
                self._state.position = 0.0
                self._state.entry_price = np.nan
                self._state.time_in_pos = 0
            else:
                if self._is_flat(prev_pos) or (prev_pos * desired_pos <= 0.0) or (not np.isfinite(self._state.entry_price)):
                    self._state.entry_price = price
                    self._state.time_in_pos = 0
                else:
                    prev_abs = abs(prev_pos)
                    desired_abs = abs(desired_pos)
                    if desired_abs > prev_abs + self._action_deadband:
                        add_units = desired_abs - prev_abs
                        self._state.entry_price = (
                            (prev_abs * float(self._state.entry_price)) + (add_units * price)
                        ) / max(desired_abs, 1e-12)
                self._state.position = desired_pos

        if (not self._is_flat(self._state.position)) and np.isfinite(self._state.entry_price):
            self._state.time_in_pos += 1
        else:
            self._state.unrealized_pnl = 0.0
            self._state.time_in_pos = 0

    def replay_warmup_actions(
        self,
        *,
        df_1m: pd.DataFrame,
        df_15m: pd.DataFrame,
        apply_ga_probs: bool = True,
    ) -> list[dict[str, object]]:
        """
        Run a sequential warmup over prefilled 15m bars and return per-bar actions.
        This is optimized for startup seeding (single feature build, sequential state updates).
        """
        if df_15m.empty or len(df_15m) < self._min_15m_bars:
            return []

        if apply_ga_probs:
            df_15m = self._maybe_add_ga_probs(df_1m=df_1m, df_15m=df_15m, target_ts=None)
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
            return []

        missing = [c for c in self._feature_cols if c not in feat_df.columns]
        for col in missing:
            feat_df[col] = 0.0

        out: list[dict[str, object]] = []
        for _, row_full in feat_df.iterrows():
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

            if self._require_probs:
                missing_prob_keys: list[str] = []
                if self._include_pivot_probs:
                    if self._last_probs.get("p_pivot_long") is None:
                        missing_prob_keys.append("p_pivot_long")
                    if self._last_probs.get("p_pivot_short") is None:
                        missing_prob_keys.append("p_pivot_short")
                if self._include_tb_probs:
                    if self._last_probs.get("p_tb_long") is None:
                        missing_prob_keys.append("p_tb_long")
                    if self._last_probs.get("p_tb_short") is None:
                        missing_prob_keys.append("p_tb_short")
                if missing_prob_keys:
                    continue

            row = row_full.reindex(self._feature_cols)
            if self._skip_on_nan and row.isna().any():
                continue

            ts = row.name
            if not isinstance(ts, pd.Timestamp):
                ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
            if ts is None or pd.isna(ts) or (not np.isfinite(price)):
                continue
            ts = self._normalize_ts(ts)

            self._update_day_state(ts, price)
            self._mark_to_market(price)
            obs = self._build_obs(row.to_numpy(dtype=np.float32), ts)
            x = torch.as_tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)
            policy_out, _ = self._model(x)
            if self._hybrid_action:
                if self._deterministic:
                    action_t, _logp, _entropy = self._model._sample_hybrid(  # noqa: SLF001
                        policy_out,
                        deterministic=True,
                    )
                else:
                    action_t, _logp, _entropy = self._model._sample_hybrid(  # noqa: SLF001
                        policy_out,
                        deterministic=False,
                    )
                action = float(action_t[:, 0].squeeze().item())
            elif self._continuous_action:
                if self._deterministic:
                    action_t = torch.tanh(policy_out)
                else:
                    action_t, _logp, _entropy = self._model._sample_continuous(  # noqa: SLF001
                        policy_out,
                        deterministic=False,
                    )
                action = float(action_t.squeeze().item())
            else:
                if self._deterministic:
                    class_action = int(torch.argmax(policy_out, dim=-1).item())
                else:
                    dist = torch.distributions.Categorical(logits=policy_out)
                    class_action = int(dist.sample().item())
                action = self._action_to_exposure(float(class_action))

            self._apply_action(action, price)
            out.append(
                {
                    "timestamp": ts,
                    "action": float(action),
                    "close": float(price),
                }
            )
        return out

    def snapshot_state(self) -> dict[str, object]:
        """
        Return a JSON-safe snapshot of the internal live position state.
        """
        last_day = self._state.last_session_day
        entry = self._state.entry_price
        return {
            "position": float(self._state.position),
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
        agent: Optional[object] = None,
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
        target_ts = None
        if closed_bar and "timestamp" in closed_bar:
            target_ts = closed_bar["timestamp"]

        if self._agent is not None:
            if isinstance(self._agent, (LiveMetaXGBAgent, LiveIndependentMetaXGBAgent)):
                return self._agent.act(df_1m=df_1m, df_15m=pd.DataFrame(), target_ts=target_ts)
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
            return self._agent.act(df_1m=df_1m, df_15m=df_15m, target_ts=target_ts)
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
        if self._feature_fn is None or self._predict_fn is None:
            return None
        features = self._feature_fn(df_15m)
        return self._predict_fn(features)

    def last_probs(self) -> dict[str, float | None] | None:
        if self._agent is None:
            return None
        getter = getattr(self._agent, "last_probs", None)
        if getter is None:
            return None
        return getter()

    def last_thresholds(self) -> dict[str, float] | None:
        if self._agent is None:
            return None
        getter = getattr(self._agent, "last_thresholds", None)
        if getter is None:
            return None
        return getter()

    def snapshot_state(self) -> dict[str, object] | None:
        if self._agent is None:
            return None
        getter = getattr(self._agent, "snapshot_state", None)
        if getter is None:
            return None
        state = getter()
        return state if isinstance(state, dict) else None

    def last_prob_sources(self) -> dict[str, str | None] | None:
        state = self.snapshot_state()
        if not isinstance(state, dict):
            return None
        sources = state.get("last_prob_sources")
        return sources if isinstance(sources, dict) else None
