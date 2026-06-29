"""Step 8 — Ticker Theme Membership Scores.

For every ticker compute cosine_similarity(ticker_embedding, theme_centroid)
across all themes. Allows multi-theme membership with soft scores.

Output: ticker_theme_membership.parquet
Schema : ticker | theme | membership_score | date
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from themes.dynamic_theme.config import (
    TICKER_CLUSTERS_PATH,
    TICKER_EMBEDDINGS_PATH,
    TICKER_MEMBERSHIP_PATH,
    ensure_outputs,
)
from themes.dynamic_theme.stages.step02_embed import load_embeddings_matrix
from themes.dynamic_theme.stages.step03_cluster import compute_centroids
from themes.dynamic_theme.stages.step05_claude_labeling import load_registry

logger = logging.getLogger(__name__)


def _cosine_sim_matrix(ticker_matrix: np.ndarray, centroid_matrix: np.ndarray) -> np.ndarray:
    """Return [N_tickers, N_themes] cosine similarity matrix.

    Both inputs are assumed L2-normalized, so dot product == cosine similarity.
    """
    tn = ticker_matrix / (np.linalg.norm(ticker_matrix, axis=1, keepdims=True) + 1e-10)
    cn = centroid_matrix / (np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + 1e-10)
    return (tn @ cn.T).astype(np.float32)


def compute_memberships(
    embeddings_df: pd.DataFrame | None = None,
    clusters_df: pd.DataFrame | None = None,
    registry_df: pd.DataFrame | None = None,
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute soft theme membership scores for every ticker."""
    ensure_outputs()
    as_of = (as_of or pd.Timestamp.now(tz="UTC")).normalize().tz_localize(None)

    if embeddings_df is None:
        tickers, matrix, date = load_embeddings_matrix()
        embeddings_df = None  # matrix already loaded
    else:
        tickers = embeddings_df["ticker"].astype(str).tolist()
        matrix = np.array(embeddings_df["embedding"].tolist(), dtype=np.float32)

    if clusters_df is None:
        clusters_df = pd.read_parquet(TICKER_CLUSTERS_PATH)

    if registry_df is None:
        registry_df = load_registry(latest_only=True)

    if registry_df.empty:
        logger.warning("Theme registry is empty — cannot compute memberships")
        return pd.DataFrame(columns=["ticker", "theme", "membership_score", "date"])

    # Build centroid per cluster_id
    centroids_by_id = compute_centroids(matrix, tickers, clusters_df)

    # Map cluster_id → theme_name
    id_to_theme = dict(zip(registry_df["cluster_id"], registry_df["theme_name"]))

    # Inject hand-pinned seed themes (e.g. memory_storage) so they always exist
    # and any similar ticker maps to them, regardless of what HDBSCAN produced.
    # Done here (not via clusters_df) so seeds work in both daily and weekly runs.
    # Skip any seed whose name the emergent taxonomy already produced this run, so
    # a seed only *fills a gap* — it never duplicates a theme HDBSCAN found itself.
    from themes.dynamic_theme.seed_themes import seed_centroids
    seed_cents, seed_names = seed_centroids(tickers, matrix)
    existing_names = set(id_to_theme.values())
    for cid, name in seed_names.items():
        if name in existing_names:
            logger.info("Seed theme '%s' already emerged this run — not injecting seed", name)
            continue
        centroids_by_id[cid] = seed_cents[cid]
        id_to_theme[cid] = name

    # Only keep clusters that appear in the registry
    valid_ids = [cid for cid in centroids_by_id if cid in id_to_theme]
    if not valid_ids:
        logger.warning("No cluster centroids match current registry")
        return pd.DataFrame(columns=["ticker", "theme", "membership_score", "date"])

    theme_names = [id_to_theme[cid] for cid in valid_ids]
    centroid_matrix = np.array([centroids_by_id[cid] for cid in valid_ids], dtype=np.float32)

    logger.info("Computing membership scores: %d tickers × %d themes", len(tickers), len(theme_names))

    sim_matrix = _cosine_sim_matrix(matrix, centroid_matrix)  # [N, T]

    rows = []
    for i, ticker in enumerate(tickers):
        for j, theme in enumerate(theme_names):
            score = float(sim_matrix[i, j])
            if score > 0.0:  # exclude zero-similarity (noise)
                rows.append({"ticker": ticker, "theme": theme, "membership_score": score, "date": as_of})

    out = pd.DataFrame(rows)
    out.to_parquet(TICKER_MEMBERSHIP_PATH, index=False)
    logger.info(
        "Wrote %s  rows=%d  avg_score=%.3f",
        TICKER_MEMBERSHIP_PATH,
        len(out),
        out["membership_score"].mean() if not out.empty else 0.0,
    )
    return out


def load_memberships(*, latest_only: bool = True) -> pd.DataFrame:
    if not TICKER_MEMBERSHIP_PATH.exists():
        return pd.DataFrame(columns=["ticker", "theme", "membership_score", "date"])
    df = pd.read_parquet(TICKER_MEMBERSHIP_PATH)
    if latest_only and not df.empty and "date" in df.columns:
        latest = df["date"].max()
        df = df[df["date"] == latest].copy()
    return df


def get_primary_theme(memberships_df: pd.DataFrame) -> pd.DataFrame:
    """Return {ticker, primary_theme, membership_score} — highest scoring theme per ticker."""
    if memberships_df.empty:
        return pd.DataFrame(columns=["ticker", "primary_theme", "membership_score"])
    idx = memberships_df.groupby("ticker")["membership_score"].idxmax()
    top = memberships_df.loc[idx, ["ticker", "theme", "membership_score"]].rename(
        columns={"theme": "primary_theme"}
    )
    return top.reset_index(drop=True)
