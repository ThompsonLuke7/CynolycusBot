"""
4H expansion-score ranker.

Loads the trained XGBoost booster, scores each ticker's most recent
4H feature row, applies the active week's universe filter, and returns
the top-N candidates for trade evaluation.

Backtest path: scores can be batched over a date range (returns a per-bar
top-N DataFrame).

Live path: `score_at(timestamp)` returns the active top-N for a given bar.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from momentum_expansion.config.momentum_config import (
    FEATURES_COMBINED,
    MODEL_PATH,
    RANKING_CONFIG,
)
from momentum_expansion.data.universe import load_snapshot_for
from momentum_expansion.features.feature_matrix_4h import FEATURE_COLUMNS_4H

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Booster loader
# ---------------------------------------------------------------------------

class ExpansionRanker:
    def __init__(
        self,
        *,
        booster_path: Path = MODEL_PATH,
        feature_columns: list[str] | None = None,
    ):
        self.booster_path = Path(booster_path)
        self.feature_columns = feature_columns or FEATURE_COLUMNS_4H
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise RuntimeError("xgboost not installed; required for ranker") from exc
            if not self.booster_path.exists():
                raise FileNotFoundError(
                    f"Booster missing at {self.booster_path}. "
                    "Run Colab training and import_from_colab first."
                )
            booster = xgb.Booster()
            booster.load_model(str(self.booster_path))
            self._model = booster
        return self._model

    def score(self, X: pd.DataFrame) -> pd.Series:
        import xgboost as xgb
        cols = [c for c in self.feature_columns if c in X.columns]
        Xv = X[cols].astype(float).values
        dmat = xgb.DMatrix(Xv, feature_names=cols)
        preds = self.model.predict(dmat)
        return pd.Series(preds, index=X.index, name="expansion_score")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enforce_universe(df_scored: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot is None or snapshot.empty:
        return df_scored
    allowed = set(snapshot["ticker"].astype(str).tolist())
    if "ticker" in df_scored.columns:
        return df_scored[df_scored["ticker"].isin(allowed)].copy()
    if "ticker" in df_scored.index.names:
        idx_t = df_scored.index.get_level_values("ticker")
        return df_scored[idx_t.isin(list(allowed))].copy()
    return df_scored


def top_n_at(
    *,
    bar_ts: pd.Timestamp,
    features: pd.DataFrame,
    ranker: ExpansionRanker,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Compute top-N candidates for the given 4H bar timestamp.

    `features` should be the combined (timestamp, ticker) MultiIndex
    feature DataFrame produced by build_all_features_4h.
    """
    cfg = {**RANKING_CONFIG, **(cfg or {})}

    if not isinstance(features.index, pd.MultiIndex):
        raise ValueError("features must have (timestamp, ticker) MultiIndex")
    ts_level = features.index.names[0]
    rows = features.xs(bar_ts, level=ts_level)
    if rows.empty:
        return pd.DataFrame(columns=["ticker", "expansion_score", "rank"])

    snapshot = load_snapshot_for(bar_ts)
    if not snapshot.empty:
        allowed = set(snapshot["ticker"].astype(str).tolist())
        rows = rows[rows.index.isin(allowed)]
    if rows.empty:
        return pd.DataFrame(columns=["ticker", "expansion_score", "rank"])

    rows = rows.dropna(subset=[c for c in ranker.feature_columns if c in rows.columns])
    if rows.empty:
        return pd.DataFrame(columns=["ticker", "expansion_score", "rank"])

    scores = ranker.score(rows)
    out = pd.DataFrame({"ticker": rows.index, "expansion_score": scores.values})
    out = out.sort_values("expansion_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)

    n_pct = max(1, int(cfg["top_pct"] * len(out)))
    n_eff = min(int(cfg["top_n"]), n_pct)
    floor = float(cfg["min_score"])
    keep = out.head(n_eff)
    keep = keep[keep["expansion_score"] >= floor]
    return keep.reset_index(drop=True)


def rank_over_range(
    *,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    features: pd.DataFrame,
    ranker: ExpansionRanker,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Run top-N ranking on every 4H timestamp in [start, end]. Returns a long
    DataFrame: [bar_ts, ticker, expansion_score, rank].
    """
    if not isinstance(features.index, pd.MultiIndex):
        raise ValueError("features must have (timestamp, ticker) MultiIndex")
    ts_level = features.index.names[0]
    all_ts = features.index.get_level_values(ts_level).unique()
    mask = (all_ts >= pd.Timestamp(start)) & (all_ts <= pd.Timestamp(end))
    keep_ts = all_ts[mask]

    out_rows: list[pd.DataFrame] = []
    for ts in keep_ts:
        ranked = top_n_at(bar_ts=ts, features=features, ranker=ranker, cfg=cfg)
        if not ranked.empty:
            ranked["bar_ts"] = ts
            out_rows.append(ranked)
    if not out_rows:
        return pd.DataFrame(columns=["bar_ts", "ticker", "expansion_score", "rank"])
    return pd.concat(out_rows, axis=0, ignore_index=True)
