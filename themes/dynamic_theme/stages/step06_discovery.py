"""Step 6 — Theme Discovery Engine.

Each week compare current cluster centroids against the prior week's centroids
using cosine similarity. Any current cluster with max similarity < threshold
is considered a NEW theme and gets labeled by Claude immediately.

Returns (new_theme_count, updated_registry_df).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from themes.dynamic_theme.config import (
    NEW_THEME_SIMILARITY_THRESHOLD,
    TICKER_CLUSTERS_PATH,
)
from themes.dynamic_theme.stages.step03_cluster import compute_centroids
from themes.dynamic_theme.stages.step05_claude_labeling import _load_registry_or_empty

logger = logging.getLogger(__name__)


def _load_prior_centroids(registry: pd.DataFrame, embeddings_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Reconstruct prior-week theme centroids from the registry and prior embeddings.

    Returns {theme_name: centroid_vector}.
    We don't store historical centroids directly; instead we compute them from
    the ticker_theme_membership parquet if available, or fall back to the
    embedding mean of tickers in each prior cluster.
    """
    # Attempt to load prior ticker clusters (previous week's parquet may not exist)
    prior_cluster_path = TICKER_CLUSTERS_PATH.with_suffix(".prior.parquet")
    if not prior_cluster_path.exists():
        return {}

    try:
        prior_clusters = pd.read_parquet(prior_cluster_path)
    except Exception:
        return {}

    tickers = embeddings_df["ticker"].astype(str).tolist()
    matrix = np.array(embeddings_df["embedding"].tolist(), dtype=np.float32)
    centroids_by_id = compute_centroids(matrix, tickers, prior_clusters)

    # map cluster_id → theme_name via last week's registry
    prior_registry = registry[registry["date"] < registry["date"].max()].copy() if not registry.empty else pd.DataFrame()
    if prior_registry.empty:
        return {str(cid): c for cid, c in centroids_by_id.items()}

    latest_prior = prior_registry["date"].max()
    prior_reg = prior_registry[prior_registry["date"] == latest_prior][["cluster_id", "theme_name"]]
    id_to_name = dict(zip(prior_reg["cluster_id"], prior_reg["theme_name"]))
    return {
        id_to_name.get(cid, f"cluster_{cid}"): c
        for cid, c in centroids_by_id.items()
    }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def discover_new_themes(
    current_clusters_df: pd.DataFrame,
    current_embeddings_df: pd.DataFrame,
    cluster_summaries: list[dict[str, Any]],
    *,
    as_of: pd.Timestamp | None = None,
    similarity_threshold: float = NEW_THEME_SIMILARITY_THRESHOLD,
) -> tuple[int, pd.DataFrame]:
    """Compare current clusters to prior; label and register new themes.

    Returns (new_theme_count, updated_registry_df).
    """
    as_of = (as_of or pd.Timestamp.now(tz="UTC")).normalize().tz_localize(None)
    registry = _load_registry_or_empty()

    tickers = current_embeddings_df["ticker"].astype(str).tolist()
    matrix = np.array(current_embeddings_df["embedding"].tolist(), dtype=np.float32)
    current_centroids = compute_centroids(matrix, tickers, current_clusters_df)
    prior_centroids = _load_prior_centroids(registry, current_embeddings_df)

    # NOTE: as of the label-stability change, step05 already (a) carries forward
    # stable prior labels and (b) Claude-labels clusters with no strong prior
    # match — i.e. exactly the "new" clusters this step used to label. So we no
    # longer re-label here (that would double-spend Claude and write duplicate
    # (cluster_id, date) rows). This step now just *reports* how many clusters
    # look new and — critically — persists the current clusters as the ".prior"
    # snapshot that step05's stability matching reads next week.
    new_count = 0
    for cid, centroid in current_centroids.items():
        if not prior_centroids:
            new_count += 1
            continue
        max_sim = max(_cosine_similarity(centroid, pc) for pc in prior_centroids.values())
        if max_sim < similarity_threshold:
            new_count += 1
            logger.info("New theme detected: cluster %d (max_sim=%.3f < %.3f)",
                        cid, max_sim, similarity_threshold)

    logger.info("Discovery: %d cluster(s) look new (labeled in step05; none re-labeled here)", new_count)

    # save current clusters as "prior" for next week's stability + discovery compare
    current_clusters_df.to_parquet(TICKER_CLUSTERS_PATH.with_suffix(".prior.parquet"), index=False)

    return new_count, registry
