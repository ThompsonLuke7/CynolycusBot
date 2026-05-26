"""Similarity-based catalyst scoring for news records."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meta_context.config import CONTEXT_BACKTEST_UNIVERSE_PATH
from news.config import NEWS_EMBEDDINGS_PATH, NEWS_LABELS_PATH, NEWS_RECORDS_PATH, NEWS_SCORES_PATH, ensure_data_dirs
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


def build_news_similarity_scores(
    *,
    news_path: str = str(NEWS_RECORDS_PATH),
    embeddings_path: str = str(NEWS_EMBEDDINGS_PATH),
    labels_path: str = str(NEWS_LABELS_PATH),
    output_path: str = str(NEWS_SCORES_PATH),
    min_similarity: float = 0.72,
    top_k: int = 25,
) -> pd.DataFrame:
    """Score news from -1 to 1 using prior similar labeled records only."""
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
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    vectors = [parse_embedding(value) for value in df.get("embedding", [])]
    target_scores = np.asarray([_target_score(row) if pd.notna(row.get("forward_5d_return")) else np.nan for _, row in df.iterrows()], dtype=np.float32)
    rows = []
    prior_matrix = np.empty((0, 0), dtype=np.float32)
    prior_scores: list[float] = []
    prior_relation = np.asarray([], dtype=object)
    prior_sector = np.asarray([], dtype=object)
    prior_market_cap = np.asarray([], dtype=object)
    prior_direct = np.asarray([], dtype=np.float32)
    prior_role = np.asarray([], dtype=object)
    prior_family = np.asarray([], dtype=object)
    prior_subtype = np.asarray([], dtype=object)
    for idx, row in df.iterrows():
        vector = vectors[idx]
        score = float("nan")
        neighbor_count = 0
        similarity_max = float("nan")
        if vector is not None and prior_matrix.size:
            query = _normalize_vector(vector)
            sims = prior_matrix @ query
            weights = np.maximum(sims - float(min_similarity), 0.0)
            relation = str(row.get("relation_type"))
            sector = str(row.get("sector"))
            market_cap = str(row.get("market_cap_bucket"))
            direct = float(row.get("is_direct_catalyst") or 0.0)
            role = str(row.get("impact_role"))
            family = str(row.get("catalyst_family"))
            subtype = str(row.get("catalyst_subtype"))
            family_match = prior_family == family
            weights *= np.where(family_match, 1.60, 0.15)
            weights *= np.where(prior_subtype == subtype, 1.35, 1.0)
            weights *= np.where(prior_relation == relation, 1.25, 1.0)
            if sector != "Unknown":
                weights *= np.where(prior_sector == sector, 1.15, 1.0)
            weights *= np.where(prior_market_cap == market_cap, 1.10, 1.0)
            weights *= np.where(prior_direct == direct, 1.10, 1.0)
            weights *= np.where(prior_role == role, 1.20, 1.0)
            valid = np.flatnonzero(weights > 0)
            if len(valid):
                ranked = valid[np.argsort(weights[valid])[-int(top_k):]]
                score = float(np.average(np.asarray(prior_scores)[ranked], weights=weights[ranked]))
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
            normalized = _normalize_vector(vector).reshape(1, -1)
            if prior_matrix.size:
                prior_matrix = np.vstack([prior_matrix, normalized])
            else:
                prior_matrix = normalized
            prior_scores.append(float(target_scores[idx]))
            prior_relation = np.append(prior_relation, str(row.get("relation_type")))
            prior_sector = np.append(prior_sector, str(row.get("sector")))
            prior_market_cap = np.append(prior_market_cap, str(row.get("market_cap_bucket")))
            prior_direct = np.append(prior_direct, float(row.get("is_direct_catalyst") or 0.0))
            prior_role = np.append(prior_role, str(row.get("impact_role")))
            prior_family = np.append(prior_family, str(row.get("catalyst_family")))
            prior_subtype = np.append(prior_subtype, str(row.get("catalyst_subtype")))
    out = pd.DataFrame(rows)
    out.to_parquet(output_path, index=False)
    return out
