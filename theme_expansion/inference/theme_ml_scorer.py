"""
Theme ML model live inference scorer.

The nightly theme pipeline already produces rule-based heat/regime scores
(`theme_scores.parquet`). This module adds the *learned* signal: it loads the
trained `theme_xgb` booster and scores each (date, theme) row, emitting
`theme_ml_score` — the same out-of-fold target the Meta Ranker consumes.

The booster was trained on one-hot categoricals (category / benchmark /
signal_market_playbook) plus numeric theme-signal features, so this scorer
rebuilds those dummies on the fly and aligns to the model's exact feature order.

Cadence: run nightly after close (or pre-open) on the latest theme signal-label
rows so the next session's bars get a fresh prior-day theme_ml_score.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "bundle" / "theme_xgb.json"
EVAL_METRICS_PATH = ROOT / "models" / "bundle" / "eval_metrics.json"
SIGNAL_LABELS_PATH = ROOT / "outputs" / "theme_signal_labels.parquet"

CATEGORICAL_COLS = ["category", "benchmark", "signal_market_playbook"]


def _load_feature_columns() -> list[str]:
    obj = json.loads(EVAL_METRICS_PATH.read_text())
    cols = obj.get("feature_columns")
    if not cols:
        raise FileNotFoundError(f"No feature_columns in {EVAL_METRICS_PATH}")
    return list(cols)


class ThemeMLScorer:
    def __init__(self, *, booster_path: Path = MODEL_PATH, feature_columns: list[str] | None = None):
        self.booster_path = Path(booster_path)
        self.feature_columns = feature_columns or _load_feature_columns()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import xgboost as xgb
            if not self.booster_path.exists():
                raise FileNotFoundError(
                    f"Booster missing at {self.booster_path}. Train + export theme_xgb first."
                )
            booster = xgb.Booster()
            booster.load_model(str(self.booster_path))
            self._model = booster
        return self._model

    def _build_X(self, rows: pd.DataFrame) -> pd.DataFrame:
        numeric = [f for f in self.feature_columns if f in rows.columns]
        present_cats = [c for c in CATEGORICAL_COLS if c in rows.columns]
        dummies = pd.get_dummies(rows[present_cats], prefix=present_cats) if present_cats else pd.DataFrame(index=rows.index)
        X = pd.concat([rows[numeric].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        return X.reindex(columns=self.feature_columns, fill_value=0).astype(float)

    def score(self, rows: pd.DataFrame) -> pd.Series:
        """Score a frame of theme signal-label rows -> theme_ml_score per row."""
        import xgboost as xgb
        X = self._build_X(rows)
        dmat = xgb.DMatrix(X.values, missing=np.nan, feature_names=self.feature_columns)
        preds = self.model.predict(dmat)
        return pd.Series(preds, index=rows.index, name="theme_ml_score")

    def score_labels(self, labels: pd.DataFrame | None = None, *, on_date=None) -> pd.DataFrame:
        """Score theme_signal_labels (all rows, or a single date). Returns
        [date, theme, theme_ml_score] sorted by score within date."""
        if labels is None:
            labels = pd.read_parquet(SIGNAL_LABELS_PATH)
        labels = labels.copy()
        labels["date"] = pd.to_datetime(labels["date"])
        if on_date is not None:
            labels = labels[labels["date"] == pd.Timestamp(on_date)]
        s = self.score(labels)
        out = labels[["date", "theme"]].copy()
        out["theme_ml_score"] = s.values
        return out.sort_values(["date", "theme_ml_score"], ascending=[True, False]).reset_index(drop=True)

    def latest(self, top: int | None = None) -> pd.DataFrame:
        labels = pd.read_parquet(SIGNAL_LABELS_PATH)
        labels["date"] = pd.to_datetime(labels["date"])
        out = self.score_labels(labels, on_date=labels["date"].max())
        return out.head(top) if top else out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scorer = ThemeMLScorer()
    latest = scorer.latest(top=10)
    print(f"Theme ML top-10 on {latest['date'].iloc[0].date()}:")
    print(latest.to_string(index=False))
