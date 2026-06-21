"""Step 3 — Cluster tickers with HDBSCAN.

Embeddings are already L2-normalized (BGE with normalize=True), so cosine
distance is equivalent to Euclidean on the unit sphere. We use metric='euclidean'
which HDBSCAN handles efficiently.

Output: ticker_clusters.parquet
Schema : ticker | cluster_id | probability | date
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from themes.dynamic_theme.config import (
    HDBSCAN_CLUSTER_SELECTION_METHOD,
    HDBSCAN_METRIC,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    TICKER_CLUSTERS_PATH,
    TICKER_EMBEDDINGS_PATH,
    UMAP_METRIC,
    UMAP_MIN_DIST,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
    ensure_outputs,
)
from themes.dynamic_theme.stages.step02_embed import load_embeddings_matrix

logger = logging.getLogger(__name__)


def cluster_tickers(
    embeddings_df: pd.DataFrame | None = None,
    *,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    metric: str = HDBSCAN_METRIC,
    cluster_selection_method: str = HDBSCAN_CLUSTER_SELECTION_METHOD,
) -> pd.DataFrame:
    """Run UMAP + HDBSCAN on the ticker embedding matrix and write ticker_clusters.parquet.

    UMAP reduces the 384-dim BGE embeddings to a low-dim space where HDBSCAN
    can discover fine-grained cluster structure (otherwise collapses to 2-3 blobs).
    Soft membership (step08) scores every ticker against every cluster centroid,
    so hard cluster assignment precision matters less than richness of structure.
    """
    ensure_outputs()

    try:
        import hdbscan as hdbscan_lib
    except ImportError as exc:
        raise ImportError("Install hdbscan: pip install hdbscan") from exc

    try:
        import umap
    except ImportError as exc:
        raise ImportError("Install umap-learn: pip install umap-learn") from exc

    if embeddings_df is not None:
        tickers = embeddings_df["ticker"].astype(str).tolist()
        matrix = np.array(embeddings_df["embedding"].tolist(), dtype=np.float32)
        date = pd.to_datetime(embeddings_df["date"].iloc[0]) if "date" in embeddings_df.columns else pd.Timestamp.now()
    else:
        tickers, matrix, date = load_embeddings_matrix()

    if len(tickers) == 0:
        logger.warning("No embeddings to cluster")
        return pd.DataFrame(columns=["ticker", "cluster_id", "probability", "date"])

    # ── UMAP dimensionality reduction ─────────────────────────────────────────
    n_components = min(UMAP_N_COMPONENTS, matrix.shape[1] - 1, len(tickers) - 2)
    logger.info(
        "UMAP: %d-dim → %d-dim  (n_neighbors=%d, min_dist=%.2f, metric=%s) ...",
        matrix.shape[1], n_components, UMAP_N_NEIGHBORS, UMAP_MIN_DIST, UMAP_METRIC,
    )
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=42,
        low_memory=False,
    )
    reduced = reducer.fit_transform(matrix).astype(np.float32)

    # ── HDBSCAN on UMAP output ────────────────────────────────────────────────
    logger.info(
        "HDBSCAN: min_cluster_size=%d, min_samples=%d, metric=%s, selection=%s ...",
        min_cluster_size, min_samples, metric, cluster_selection_method,
    )
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,
    )
    clusterer.fit(reduced)

    labels: np.ndarray = clusterer.labels_
    probs: np.ndarray = clusterer.probabilities_

    n_clusters = int(labels.max()) + 1
    n_noise = int((labels == -1).sum())
    logger.info(
        "HDBSCAN found %d clusters, %d noise points (%.1f%%)",
        n_clusters, n_noise, 100 * n_noise / len(labels),
    )

    df = pd.DataFrame(
        {
            "ticker": tickers,
            "cluster_id": labels.astype(int),
            "probability": probs.astype(float),
            "date": date,
        }
    )
    df.to_parquet(TICKER_CLUSTERS_PATH, index=False)
    logger.info("Wrote %s  clusters=%d  noise=%d", TICKER_CLUSTERS_PATH, n_clusters, n_noise)
    return df


def load_clusters() -> pd.DataFrame:
    return pd.read_parquet(TICKER_CLUSTERS_PATH)


def compute_centroids(
    matrix: np.ndarray,
    tickers: list[str],
    clusters_df: pd.DataFrame,
) -> dict[int, np.ndarray]:
    """Return {cluster_id: centroid_vector} for all non-noise clusters."""
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    centroids: dict[int, np.ndarray] = {}
    for cid, grp in clusters_df[clusters_df["cluster_id"] >= 0].groupby("cluster_id"):
        idxs = [ticker_to_idx[t] for t in grp["ticker"] if t in ticker_to_idx]
        if idxs:
            vecs = matrix[idxs]
            c = vecs.mean(axis=0)
            norm = np.linalg.norm(c)
            centroids[int(cid)] = c / norm if norm > 0 else c
    return centroids
