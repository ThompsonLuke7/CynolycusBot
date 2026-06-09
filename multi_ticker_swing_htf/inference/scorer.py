"""
HTF swing live inference scorer.

Loads the installed booster (`models/htf_swing_xgb.json`) and scores the 4H
feature matrix, returning the continuous `htf_swing_score` per (timestamp,
ticker). Mirrors momentum_expansion.inference.ranker so the Meta Ranker and the
live stack consume all modules through the same shape.

The HTF model uses the shared 4H feature set (the same FEATURE_COLUMNS_4H the
momentum features matrix provides), so any (timestamp, ticker) MultiIndex frame
carrying those columns can be scored.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from multi_ticker_swing_htf.config import MODELS_DIR

logger = logging.getLogger(__name__)

MODEL_PATH = MODELS_DIR / "htf_swing_xgb.json"
MANIFEST_PATH = MODELS_DIR / "feature_manifest.json"
EVAL_METRICS_PATH = MODELS_DIR / "eval_metrics.json"


def _load_feature_columns() -> list[str]:
    for p in (MANIFEST_PATH, EVAL_METRICS_PATH):
        if p.exists():
            obj = json.loads(p.read_text())
            cols = obj.get("feature_columns")
            if cols:
                return list(cols)
    raise FileNotFoundError(
        f"No feature_columns in {MANIFEST_PATH} or {EVAL_METRICS_PATH}. "
        "Run import_from_colab to install the HTF bundle first."
    )


class HTFSwingScorer:
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
                    f"Booster missing at {self.booster_path}. "
                    "Run Colab training + import_from_colab first."
                )
            booster = xgb.Booster()
            booster.load_model(str(self.booster_path))
            self._model = booster
        return self._model

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Return htf_swing_score for each row of a feature frame."""
        import xgboost as xgb
        cols = [c for c in self.feature_columns if c in X.columns]
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            logger.warning("HTF scorer: %d feature columns missing (scoring on %d)", len(missing), len(cols))
        dmat = xgb.DMatrix(X[cols].astype(float).values, missing=np.nan, feature_names=cols)
        preds = self.model.predict(dmat)
        return pd.Series(preds, index=X.index, name="htf_swing_score")

    def score_at(self, bar_ts: pd.Timestamp, features: pd.DataFrame) -> pd.DataFrame:
        """Rank all names at a single 4H bar. Returns [ticker, htf_swing_score, rank]."""
        if not isinstance(features.index, pd.MultiIndex):
            raise ValueError("features must have (timestamp, ticker) MultiIndex")
        ts_level = features.index.names[0]
        rows = features.xs(bar_ts, level=ts_level)
        rows = rows.dropna(subset=[c for c in self.feature_columns if c in rows.columns], how="all")
        if rows.empty:
            return pd.DataFrame(columns=["ticker", "htf_swing_score", "rank"])
        s = self.score(rows)
        out = pd.DataFrame({"ticker": rows.index, "htf_swing_score": s.values})
        out = out.sort_values("htf_swing_score", ascending=False).reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1)
        return out

    def score_over_range(self, features: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
        """Score every (timestamp, ticker) row; optional time filter. Long frame."""
        df = features
        ts = df.index.get_level_values(0)
        if start is not None:
            df = df[ts >= pd.Timestamp(start)]; ts = df.index.get_level_values(0)
        if end is not None:
            df = df[ts <= pd.Timestamp(end)]
        s = self.score(df)
        return s.reset_index().rename(columns={0: "htf_swing_score"})


if __name__ == "__main__":
    # Inference smoke test (no training): score a recent slice of the shared 4H
    # feature matrix that the HTF model shares with momentum.
    logging.basicConfig(level=logging.INFO)
    import pyarrow.parquet as pq
    feats_path = Path("momentum_expansion/data/processed/features_4h.parquet")
    scorer = HTFSwingScorer()
    tcol = pd.to_datetime(pq.read_table(feats_path, columns=["timestamp"]).column("timestamp").to_pandas(), utc=True)
    cut = tcol.max() - pd.Timedelta(days=5)
    df = pd.read_parquet(feats_path, filters=[("timestamp", ">=", cut.to_pydatetime())])
    if "timestamp" not in df.columns:
        df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index(["timestamp", "ticker"]).sort_index()
    latest = df.index.get_level_values(0).max()
    top = scorer.score_at(latest, df).head(10)
    print(f"HTF top-10 at {latest}:")
    print(top.to_string(index=False))
