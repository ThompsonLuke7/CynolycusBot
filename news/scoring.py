"""Similarity-based catalyst scoring for news records.

The scorer walks records in chronological order and, when scoring record ``i``,
only consults priors whose forward-return label was already realized by ``i``'s
timestamp. Without that horizon, the score at time ``t`` leaks information from
records at ``t - k`` whose 10-day forward return wouldn't be known until
``t - k + 10 trading days``. See ``LABEL_HORIZON_DAYS``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meta_context.config import CONTEXT_BACKTEST_UNIVERSE_PATH
from news.config import (
    LABEL_HORIZON_BARS_PER_DAY,
    LABEL_HORIZON_DAYS,
    NEWS_EMBEDDINGS_PATH,
    NEWS_LABELS_PATH,
    NEWS_RECORDS_PATH,
    NEWS_SCORES_PATH,
    ensure_data_dirs,
)
from news.pipeline import parse_embedding


def _load_universe_profile() -> pd.DataFrame:
    if not CONTEXT_BACKTEST_UNIVERSE_PATH.exists():
        return pd.DataFrame(columns=["ticker", "sector", "market_cap_bucket", "type"])
    out = pd.read_csv(CONTEXT_BACKTEST_UNIVERSE_PATH)
    keep = [col for col in ["ticker", "sector", "market_cap_bucket", "type"] if col in out.columns]
    out = out[keep].copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    return out


def _target_score(row: pd.Series) -> float:
    fwd_5d = row.get("forward_5d_return")
    max_fwd = row.get("max_forward_return")
    max_dd = row.get("max_drawdown")
    values = []
    if pd.notna(fwd_5d):
        values.append(float(fwd_5d))
    if pd.notna(max_fwd):
        values.append(0.5 * float(max_fwd))
    if pd.notna(max_dd):
        values.append(0.35 * float(max_dd))
    raw = float(np.nanmean(values)) if values else 0.0
    return float(np.tanh(raw * 2.0))


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    out = vector.astype(np.float32)
    return out / max(float(np.linalg.norm(out)), 1e-12)


def _label_ready_timestamps(timestamps: pd.Series, *, horizon_days: int) -> pd.Series:
    """Earliest moment at which each record's forward-return label would be known."""
    return pd.to_datetime(timestamps, utc=True, errors="coerce") + pd.Timedelta(days=int(horizon_days))


def build_news_similarity_scores(
    *,
    news_path: str = str(NEWS_RECORDS_PATH),
    embeddings_path: str = str(NEWS_EMBEDDINGS_PATH),
    labels_path: str = str(NEWS_LABELS_PATH),
    output_path: str = str(NEWS_SCORES_PATH),
    min_similarity: float = 0.72,
    top_k: int = 25,
    label_horizon_days: int = LABEL_HORIZON_DAYS,
) -> pd.DataFrame:
    """Score news from -1 to 1 using prior similar records whose labels were realized in time.

    Leak prevention: a prior record at time ``t_prior`` only contributes when
    ``t_prior + label_horizon_days <= t_current``. Sorting is deterministic on
    ``(timestamp, record_id)`` so tie-broken priors don't shuffle between runs.
    """
    ensure_data_dirs()
    news = pd.read_parquet(news_path)
    emb = pd.read_parquet(embeddings_path)
    labels = pd.read_parquet(labels_path)
    universe = _load_universe_profile()
    df = (
        news.merge(
            emb.drop(columns=["text", "catalyst_family", "catalyst_subtype"], errors="ignore"),
            on=["record_id", "ticker", "timestamp"],
            how="left",
        )
        .merge(labels, on=["record_id", "ticker", "timestamp"], how="left")
        .merge(universe, on="ticker", how="left")
        .sort_values(["timestamp", "record_id"])
        .reset_index(drop=True)
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    vectors = [parse_embedding(value) for value in df.get("embedding", [])]
    target_scores = np.asarray(
        [_target_score(row) if pd.notna(row.get("forward_5d_return")) else np.nan for _, row in df.iterrows()],
        dtype=np.float32,
    )
    label_ready_ts = _label_ready_timestamps(df["timestamp"], horizon_days=label_horizon_days).to_numpy()
    current_ts = df["timestamp"].to_numpy()

    # Priors stored as plain lists; we matmul against a freshly stacked submatrix
    # of just the eligible priors at each step. Avoids the O(n^2) np.append churn.
    prior_vecs: list[np.ndarray] = []
    prior_scores: list[float] = []
    prior_ready_ts: list[np.datetime64] = []
    prior_relation: list[str] = []
    prior_sector: list[str] = []
    prior_market_cap: list[str] = []
    prior_direct: list[float] = []
    prior_role: list[str] = []
    prior_family: list[str] = []
    prior_subtype: list[str] = []

    rows = []
    for idx, row in df.iterrows():
        vector = vectors[idx]
        score = float("nan")
        neighbor_count = 0
        similarity_max = float("nan")
        # Determine how many priors are already label-ready at this row's timestamp.
        # prior_ready_ts is kept in insertion order (which is timestamp order), so we
        # can searchsorted it.
        if vector is not None and prior_vecs:
            ready_array = np.asarray(prior_ready_ts, dtype="datetime64[ns]")
            cutoff = pd.Timestamp(current_ts[idx]).tz_convert("UTC").tz_localize(None).to_datetime64()
            eligible_end = int(np.searchsorted(ready_array, cutoff, side="right"))
            if eligible_end > 0:
                query = _normalize_vector(vector)
                matrix = np.vstack(prior_vecs[:eligible_end]).astype(np.float32)
                sims = matrix @ query
                weights = np.maximum(sims - float(min_similarity), 0.0)
                relation = str(row.get("relation_type"))
                sector = str(row.get("sector"))
                market_cap = str(row.get("market_cap_bucket"))
                direct = float(row.get("is_direct_catalyst") or 0.0)
                role = str(row.get("impact_role"))
                family = str(row.get("catalyst_family"))
                subtype = str(row.get("catalyst_subtype"))
                p_relation = np.asarray(prior_relation[:eligible_end], dtype=object)
                p_sector = np.asarray(prior_sector[:eligible_end], dtype=object)
                p_cap = np.asarray(prior_market_cap[:eligible_end], dtype=object)
                p_direct = np.asarray(prior_direct[:eligible_end], dtype=np.float32)
                p_role = np.asarray(prior_role[:eligible_end], dtype=object)
                p_family = np.asarray(prior_family[:eligible_end], dtype=object)
                p_subtype = np.asarray(prior_subtype[:eligible_end], dtype=object)
                weights = weights * np.where(p_family == family, 1.60, 0.15)
                weights = weights * np.where(p_subtype == subtype, 1.35, 1.0)
                weights = weights * np.where(p_relation == relation, 1.25, 1.0)
                if sector != "Unknown":
                    weights = weights * np.where(p_sector == sector, 1.15, 1.0)
                weights = weights * np.where(p_cap == market_cap, 1.10, 1.0)
                weights = weights * np.where(p_direct == direct, 1.10, 1.0)
                weights = weights * np.where(p_role == role, 1.20, 1.0)
                valid = np.flatnonzero(weights > 0)
                if len(valid):
                    ranked = valid[np.argsort(weights[valid])[-int(top_k):]]
                    score = float(np.average(np.asarray(prior_scores[:eligible_end])[ranked], weights=weights[ranked]))
                    neighbor_count = int(len(ranked))
                    similarity_max = float(np.nanmax(sims[ranked]))
        rows.append(
            {
                "record_id": row["record_id"],
                "ticker": row["ticker"],
                "timestamp": row["timestamp"],
                "relation_type": row.get("relation_type"),
                "impact_role": row.get("impact_role"),
                "catalyst_family": row.get("catalyst_family"),
                "catalyst_subtype": row.get("catalyst_subtype"),
                "news_cluster_id": row.get("news_cluster_id"),
                "news_similarity_score": score,
                "news_similarity_neighbor_count": neighbor_count,
                "news_similarity_max": similarity_max,
                "realized_news_score": target_scores[idx] if np.isfinite(target_scores[idx]) else np.nan,
            }
        )
        if vector is not None and np.isfinite(target_scores[idx]):
            prior_vecs.append(_normalize_vector(vector))
            prior_scores.append(float(target_scores[idx]))
            prior_ready_ts.append(np.datetime64(pd.Timestamp(label_ready_ts[idx]).tz_convert("UTC").tz_localize(None)))
            prior_relation.append(str(row.get("relation_type")))
            prior_sector.append(str(row.get("sector")))
            prior_market_cap.append(str(row.get("market_cap_bucket")))
            prior_direct.append(float(row.get("is_direct_catalyst") or 0.0))
            prior_role.append(str(row.get("impact_role")))
            prior_family.append(str(row.get("catalyst_family")))
            prior_subtype.append(str(row.get("catalyst_subtype")))
    out = pd.DataFrame(rows)
    out.to_parquet(output_path, index=False)
    return out
