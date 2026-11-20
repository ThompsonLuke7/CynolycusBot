"""Step 3 — Cluster tickers with HDBSCAN.

Embeddings are already L2-normalized (BGE with normalize=True), so cosine
distance is equivalent to Euclidean on the unit sphere. We use metric='euclidean'
which HDBSCAN handles efficiently.

Groups are FIXED between re-fits (research/theme_stability_2026-11-18: re-clustering
identical inputs with another UMAP seed keeps only ~0.57 of the grouping). A re-fit
runs when requested, when no anchor exists, when the identity recipe changed, or when
the anchor is THEME_REFIT_DAYS old; it writes ticker_cluster_anchor.parquet. Every
other run reuses the anchor: continuing tickers keep their group, new tickers get
cluster_id -1 (placed by industry peers downstream) until the next re-fit.

Output: ticker_clusters.parquet
Schema : ticker | cluster_id | probability | date | refit_id
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from themes.dynamic_theme.config import (
    COMOVEMENT_WEIGHT,
    HDBSCAN_CLUSTER_SELECTION_METHOD,
    HDBSCAN_METRIC,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    IDENTITY_TEXT_VERSION,
    THEME_REFIT_DAYS,
    TICKER_CLUSTER_ANCHOR_PATH,
    TICKER_CLUSTERS_PATH,
    TICKER_EMBEDDINGS_PATH,
    UMAP_METRIC,
    UMAP_MIN_DIST,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
    ensure_outputs,
)
from themes.dynamic_theme.stages.step02_embed import load_embeddings_matrix
from themes.dynamic_theme.sklearn_compat import load_hdbscan, load_umap

logger = logging.getLogger(__name__)

_CLUSTER_COLUMNS = ["ticker", "cluster_id", "probability", "date", "refit_id"]


def clustering_recipe(
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    metric: str = HDBSCAN_METRIC,
    cluster_selection_method: str = HDBSCAN_CLUSTER_SELECTION_METHOD,
) -> str:
    """Everything that defines the groups; an anchor built under another recipe is re-fit."""
    return (
        f"{IDENTITY_TEXT_VERSION}|comovement={COMOVEMENT_WEIGHT}|umap={UMAP_N_COMPONENTS},"
        f"{UMAP_N_NEIGHBORS},{UMAP_MIN_DIST},{UMAP_METRIC}|hdbscan={min_cluster_size},"
        f"{min_samples},{metric},{cluster_selection_method}"
    )


def load_anchor() -> pd.DataFrame | None:
    if not TICKER_CLUSTER_ANCHOR_PATH.exists():
        return None
    anchor = pd.read_parquet(TICKER_CLUSTER_ANCHOR_PATH)
    return anchor if not anchor.empty else None


def refit_reason(anchor: pd.DataFrame | None, *, refit: bool, as_of: pd.Timestamp,
                 recipe: str) -> str | None:
    """Why the groups must be re-fit now, or None to reuse the anchor."""
    if refit:
        return "requested"
    if anchor is None:
        return "no anchor"
    if str(anchor["recipe"].iloc[0]) != recipe:
        return "clustering recipe changed"
    age = (as_of.normalize() - pd.Timestamp(anchor["refit_id"].iloc[0])).days
    if age >= THEME_REFIT_DAYS:
        return f"anchor is {age} days old (>= {THEME_REFIT_DAYS})"
    return None


def clusters_from_anchor(anchor: pd.DataFrame, tickers: list[str], date: pd.Timestamp) -> pd.DataFrame:
    """Continuing tickers keep their anchored group; new ones are unassigned (-1)."""
    a = anchor.drop_duplicates("ticker").set_index("ticker")
    cluster_id = [int(a["cluster_id"].get(t, -1)) for t in tickers]
    probability = [float(a["probability"].get(t, 0.0)) for t in tickers]
    return pd.DataFrame({
        "ticker": tickers, "cluster_id": cluster_id, "probability": probability,
        "date": date, "refit_id": str(anchor["refit_id"].iloc[0]),
    })[_CLUSTER_COLUMNS]


def cluster_tickers(
    embeddings_df: pd.DataFrame | None = None,
    *,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    metric: str = HDBSCAN_METRIC,
    cluster_selection_method: str = HDBSCAN_CLUSTER_SELECTION_METHOD,
    refit: bool = False,
) -> pd.DataFrame:
    """Write ticker_clusters.parquet from the fixed groups, re-fitting only when due.

    A re-fit runs UMAP + HDBSCAN on the ticker embedding matrix (below) and stores
    the result as the new anchor; see the module docstring for when that happens.

    UMAP reduces the 384-dim BGE embeddings to a low-dim space where HDBSCAN
    can discover fine-grained cluster structure (otherwise collapses to 2-3 blobs).
    Soft membership (step08) scores every ticker against every cluster centroid,
    so hard cluster assignment precision matters less than richness of structure.
    """
    ensure_outputs()

    if embeddings_df is not None:
        tickers = embeddings_df["ticker"].astype(str).tolist()
        matrix = np.array(embeddings_df["embedding"].tolist(), dtype=np.float32)
        date = pd.to_datetime(embeddings_df["date"].iloc[0]) if "date" in embeddings_df.columns else pd.Timestamp.now()
    else:
        tickers, matrix, date = load_embeddings_matrix()

    if len(tickers) == 0:
        logger.warning("No embeddings to cluster")
        return pd.DataFrame(columns=_CLUSTER_COLUMNS)

    date = pd.Timestamp(date)
    date = date.tz_convert(None) if date.tzinfo is not None else date
    recipe = clustering_recipe(min_cluster_size, min_samples, metric, cluster_selection_method)
    anchor = load_anchor()
    reason = refit_reason(anchor, refit=refit, as_of=date, recipe=recipe)
    if reason is None:
        df = clusters_from_anchor(anchor, tickers, date)
        n_new = int((~pd.Series(tickers).isin(set(anchor["ticker"]))).sum())
        logger.info(
            "Fixed groups from re-fit %s: %d clusters, %d new ticker(s) unassigned until the next re-fit",
            df["refit_id"].iloc[0], int(df.loc[df["cluster_id"] >= 0, "cluster_id"].nunique()), n_new,
        )
        df.to_parquet(TICKER_CLUSTERS_PATH, index=False)
        return df
    logger.info("Re-fitting theme groups: %s", reason)
    hdbscan_lib = load_hdbscan()
    umap = load_umap()

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

    refit_id = date.normalize().strftime("%Y-%m-%d")
    df = pd.DataFrame(
        {
            "ticker": tickers,
            "cluster_id": labels.astype(int),
            "probability": probs.astype(float),
            "date": date,
            "refit_id": refit_id,
        }
    )
    df.to_parquet(TICKER_CLUSTERS_PATH, index=False)
    df[["ticker", "cluster_id", "probability", "refit_id"]].assign(recipe=recipe).to_parquet(
        TICKER_CLUSTER_ANCHOR_PATH, index=False,
    )
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
