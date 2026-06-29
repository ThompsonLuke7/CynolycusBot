"""Seeded / anchor themes.

Hand-pinned themes that must always exist regardless of what unsupervised
HDBSCAN produces. Each seed's anchor centroid is the mean embedding of its
``anchor_tickers`` (so it lives in the same space as the emergent cluster
centroids), injected alongside them in step08 membership scoring. Seeds get
reserved (negative) cluster ids that never collide with HDBSCAN ids and are
never sent to Claude for labeling — their name/parent/description come straight
from config.

See ``SEED_THEMES`` in config.py for the catalog and rationale.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from themes.dynamic_theme.config import SEED_CLUSTER_ID_BASE, SEED_THEMES

logger = logging.getLogger(__name__)


def seed_cluster_id(i: int) -> int:
    """Reserved cluster id for the i-th seed theme (distinct, < HDBSCAN ids)."""
    return SEED_CLUSTER_ID_BASE - i


def seed_id_to_name() -> dict[int, str]:
    return {seed_cluster_id(i): s["theme_name"] for i, s in enumerate(SEED_THEMES)}


def seed_registry_rows(as_of: pd.Timestamp) -> pd.DataFrame:
    """Registry rows for every seed theme (schema matches step05's registry)."""
    rows = []
    for i, s in enumerate(SEED_THEMES):
        rows.append(
            {
                "cluster_id": seed_cluster_id(i),
                "theme_name": str(s["theme_name"]),
                "parent_theme": str(s.get("parent_theme", "unknown")),
                "description": str(s.get("description", "")),
                "related_themes": json.dumps(s.get("related_themes") or []),
                "confidence": 1.0,  # hand-pinned
                "date": as_of,
            }
        )
    return pd.DataFrame(rows)


def seed_centroids(
    tickers: list[str], matrix: np.ndarray
) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    """Build anchor centroids for seed themes from anchor-ticker embeddings.

    Returns ({cluster_id: centroid_vector}, {cluster_id: theme_name}). A seed is
    skipped (with a warning) if none of its anchor tickers are in the universe.
    """
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    centroids: dict[int, np.ndarray] = {}
    id_to_name: dict[int, str] = {}
    for i, s in enumerate(SEED_THEMES):
        cid = seed_cluster_id(i)
        idxs = [ticker_to_idx[t] for t in s.get("anchor_tickers", []) if t in ticker_to_idx]
        if not idxs:
            logger.warning(
                "Seed theme '%s': none of its anchor tickers are in the universe — skipping",
                s["theme_name"],
            )
            continue
        c = matrix[idxs].mean(axis=0)
        norm = np.linalg.norm(c)
        centroids[cid] = (c / norm) if norm > 0 else c
        id_to_name[cid] = str(s["theme_name"])
        if len(idxs) < len(s.get("anchor_tickers", [])):
            logger.info(
                "Seed theme '%s': anchored on %d/%d tickers present",
                s["theme_name"], len(idxs), len(s.get("anchor_tickers", [])),
            )
    return centroids, id_to_name
