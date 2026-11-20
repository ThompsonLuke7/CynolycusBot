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

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from strategies.momentum_expansion.config.momentum_config import (
    EVAL_METRICS_PATH,
    FEATURE_MANIFEST,
    FEATURES_COMBINED,
    MODEL_PATH,
    RANKING_CONFIG,
)
from strategies.momentum_expansion.data.universe import load_snapshot_for
from strategies.momentum_expansion.inference.candidate_filter import filter_momentum_candidates

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Booster loader
# ---------------------------------------------------------------------------

def _load_feature_columns() -> list[str]:
    """Source of truth for the INSTALLED booster's expected features: its own
    feature_manifest.json (falls back to eval_metrics.json). Mirrors
    strategies.multi_ticker_swing_htf.inference.scorer._load_feature_columns —
    do NOT default to the feature_matrix_4h.FEATURE_COLUMNS_4H superset, which
    drifts from whatever subset a given training run's GA feature selection
    actually kept (see feature_manifest.json vs feature_matrix_4h.py history)."""
    for p in (FEATURE_MANIFEST, EVAL_METRICS_PATH):
        if p.exists():
            obj = json.loads(p.read_text())
            cols = obj.get("feature_columns")
            if cols:
                return list(cols)
    raise FileNotFoundError(
        f"No feature_columns in {FEATURE_MANIFEST} or {EVAL_METRICS_PATH}. "
        "Run Colab training + import_from_colab first."
    )


class ExpansionRanker:
    def __init__(
        self,
        *,
        booster_path: Path = MODEL_PATH,
        feature_columns: list[str] | None = None,
    ):
        self.booster_path = Path(booster_path)
        self.feature_columns = feature_columns or _load_feature_columns()
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
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            logger.warning(
                "ExpansionRanker: %d/%d manifest columns missing (scoring on %d): %s",
                len(missing), len(self.feature_columns), len(cols), missing[:10],
            )
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


CROSS_SECTION_GATES = ("in_universe", "candidate_filter", "complete_features", "top_n", "min_score")


def score_cross_section(
    *,
    bar_ts: pd.Timestamp,
    features: pd.DataFrame,
    ranker: ExpansionRanker,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Every name on ``bar_ts`` with its gate outcomes, score and rank.

    One row per ticker on the bar (index = ticker). Columns: ``expansion_score``
    (NaN when a model feature is missing), ``rank`` (1 = best among names that
    pass universe, candidate filter and feature completeness; NaN otherwise), a
    boolean per gate in ``CROSS_SECTION_GATES`` (True = passed) and
    ``selected`` (passed every gate). ``top_n_at`` is exactly the selected rows;
    the rest is what the decision-state archive needs to record the misses.
    """
    cfg = {**RANKING_CONFIG, **(cfg or {})}

    if not isinstance(features.index, pd.MultiIndex):
        raise ValueError("features must have (timestamp, ticker) MultiIndex")
    ts_level = features.index.names[0]
    rows = features.xs(bar_ts, level=ts_level)
    out = pd.DataFrame(index=rows.index)
    out["expansion_score"] = np.nan
    out["rank"] = np.nan
    for gate in CROSS_SECTION_GATES:
        out[gate] = False
    out["selected"] = False
    if rows.empty:
        return out

    snapshot = load_snapshot_for(bar_ts)
    if not snapshot.empty:
        allowed = set(snapshot["ticker"].astype(str).tolist())
        out["in_universe"] = rows.index.isin(allowed)
    else:
        out["in_universe"] = True
    if cfg.get("use_momentum_candidate_filter", True):
        passed = filter_momentum_candidates(rows).index
        out["candidate_filter"] = rows.index.isin(passed)
    else:
        out["candidate_filter"] = True
    model_cols = [c for c in ranker.feature_columns if c in rows.columns]
    complete = (rows[model_cols].notna().all(axis=1).to_numpy(dtype=bool) if model_cols
                else np.ones(len(rows), dtype=bool))
    out["complete_features"] = complete

    # Positional throughout. Scores stay float32 (the booster's dtype) and the
    # eligible rows are sorted in their original row order with the same sort
    # call as before this refactor, so even ties resolve exactly as they did.
    scores = np.full(len(rows), np.nan, dtype=np.float32)
    if complete.any():
        scores[complete] = ranker.score(rows.loc[complete]).to_numpy(dtype=np.float32)
    eligible = (out["in_universe"].to_numpy(dtype=bool) & out["candidate_filter"].to_numpy(dtype=bool)
                & complete)
    elig_pos = np.flatnonzero(eligible)
    order = pd.DataFrame({"pos": elig_pos, "expansion_score": scores[elig_pos]}).sort_values(
        "expansion_score", ascending=False)["pos"].to_numpy()
    rank = np.full(len(rows), np.nan)
    rank[order] = np.arange(1, len(order) + 1)

    n_pct = max(1, int(cfg["top_pct"] * len(order)))
    n_eff = min(int(cfg["top_n"]), n_pct)
    floor = float(cfg["min_score"])
    out["expansion_score"] = scores
    out["rank"] = rank
    out["top_n"] = rank <= n_eff
    out["min_score"] = scores >= floor
    out["selected"] = eligible & out["top_n"].to_numpy() & out["min_score"].to_numpy()
    return out


def top_n_at(
    *,
    bar_ts: pd.Timestamp,
    features: pd.DataFrame,
    ranker: ExpansionRanker,
    cfg: dict | None = None,
    cross_section: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute top-N candidates for the given 4H bar timestamp.

    `features` should be the combined (timestamp, ticker) MultiIndex
    feature DataFrame produced by build_all_features_4h. Pass a precomputed
    ``cross_section`` (from ``score_cross_section``) to avoid scoring twice.
    """
    xs = cross_section if cross_section is not None else score_cross_section(
        bar_ts=bar_ts, features=features, ranker=ranker, cfg=cfg)
    keep = xs.loc[xs["selected"]].sort_values("rank")
    if keep.empty:
        return pd.DataFrame(columns=["ticker", "expansion_score", "rank"])
    out = pd.DataFrame({
        "ticker": keep.index.to_numpy(),
        "expansion_score": keep["expansion_score"].to_numpy(dtype=np.float32),
        "rank": keep["rank"].to_numpy(dtype=np.int64),
    })
    return out.reset_index(drop=True)


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
