from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from signals.location_features import add_liquidity_zone_features


@dataclass
class EmpiricalScoreCalibrator:
    """Map ranker scores to their empirical percentile on a calibration split."""

    sorted_scores: np.ndarray

    @classmethod
    def fit(cls, scores: np.ndarray) -> "EmpiricalScoreCalibrator":
        values = np.asarray(scores, dtype=float)
        values = np.sort(values[np.isfinite(values)])
        if values.size == 0:
            raise ValueError("Cannot calibrate an empty score vector.")
        return cls(sorted_scores=values)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        out = np.full(values.shape, np.nan, dtype=float)
        finite = np.isfinite(values)
        if finite.any():
            ranks = np.searchsorted(self.sorted_scores, values[finite], side="right")
            out[finite] = ranks / float(self.sorted_scores.size)
        return out


class CompetitionSwingRanker:
    """Runtime adapter for the paired long/short SPY competition rankers."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        manifest_path = self.model_dir / "long_swing_label" / "feature_manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        self.feature_columns = list(self.manifest["feature_columns"])
        self.models: dict[str, xgb.Booster] = {}
        self.best_iterations: dict[str, int] = {}
        self.calibrators: dict[str, EmpiricalScoreCalibrator] = {}
        for side in ("long", "short"):
            side_dir = self.model_dir / f"{side}_swing_label"
            meta = json.loads((side_dir / "competition_meta.json").read_text())
            model = xgb.Booster()
            model.load_model(side_dir / "winner_model.ubj")
            model.set_param({"device": "cpu"})
            self.models[side] = model
            self.best_iterations[side] = int(meta["best"]["best_iteration"])
            self.calibrators[side] = joblib.load(side_dir / "score_percentile_calibrator.joblib")

    def prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.reindex(columns=self.feature_columns)
        work = frame.copy()
        if not isinstance(work.index, pd.DatetimeIndex):
            raise ValueError("Competition ranker frame requires a DatetimeIndex.")
        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in work.columns]
        if missing:
            raise ValueError(f"Competition ranker frame is missing OHLCV columns: {missing}")

        enriched, liquidity_cols, _ = add_liquidity_zone_features(
            work,
            lookback=78,
            swing_window=18,
            zone_width_pct=0.0015,
            volume_window=39,
        )
        for col in liquidity_cols:
            if col not in work.columns and col in enriched.columns:
                work[col] = enriched[col]

        # Match the untouched evaluation period: options, historical bid/ask,
        # and OI-wall features were unavailable and therefore remained missing.
        work = work.reindex(columns=self.feature_columns)
        for col in ("inside_call_wall_zone", "inside_put_wall_zone", "between_major_walls"):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
        return work

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        aligned = self.prepare_frame(frame)
        matrix = (
            aligned.apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(np.float32)
        )
        dmat = xgb.DMatrix(matrix, missing=np.nan)
        out = {}
        for side in ("long", "short"):
            raw = self.models[side].predict(
                dmat,
                iteration_range=(0, self.best_iterations[side] + 1),
            )
            out[side] = self.calibrators[side].transform(raw)
        result = pd.DataFrame(out, index=frame.index)
        result["neutral"] = 1.0 - result[["long", "short"]].max(axis=1)
        return result[["short", "neutral", "long"]]
