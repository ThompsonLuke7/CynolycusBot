"""Step 5 — Claude theme labeling.

Calls Claude once per cluster to assign:
  theme_name, parent_theme, description, related_themes, confidence

Merges with any existing theme_registry.parquet (upsert by cluster_id + date).

Output: theme_registry.parquet
Schema : cluster_id | theme_name | parent_theme | description | related_themes | confidence | date
"""
from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from themes.dynamic_theme.client.claude_client import call_claude_json
from themes.dynamic_theme.config import THEME_REGISTRY_PATH, ensure_outputs

logger = logging.getLogger(__name__)

_LABEL_PROMPT_TEMPLATE = """\
You are an institutional equity research taxonomy system.

Analyze this cluster of tickers and generate a theme label.

Cluster:
{cluster_json}

Return JSON only — no prose, no markdown fences.

{{
  "theme_name": "",
  "parent_theme": "",
  "description": "",
  "related_themes": [],
  "confidence": 0.0
}}

Rules:
- theme_name must be snake_case, 1-4 words (e.g. nuclear_energy, ai_infrastructure)
- parent_theme must be a broader snake_case category
- related_themes is a list of snake_case theme names that are economically related
- confidence is 0.0-1.0 reflecting how clearly defined this cluster is
"""


def _label_cluster(summary: dict[str, Any]) -> dict[str, Any]:
    cluster_json = json.dumps(
        {
            "cluster_id": summary["cluster_id"],
            "tickers": summary["tickers"],
            "top_keywords": summary["top_keywords"],
            "sample_headlines": summary["sample_headlines"],
        },
        indent=2,
    )
    prompt = _LABEL_PROMPT_TEMPLATE.format(cluster_json=cluster_json)
    try:
        result = call_claude_json(prompt)
        result["cluster_id"] = summary["cluster_id"]
        result["related_themes"] = result.get("related_themes") or []
        return result
    except Exception as exc:
        logger.error("Claude labeling failed for cluster %d: %s", summary["cluster_id"], exc)
        return {
            "cluster_id": summary["cluster_id"],
            "theme_name": f"cluster_{summary['cluster_id']}",
            "parent_theme": "unknown",
            "description": "",
            "related_themes": [],
            "confidence": 0.0,
        }


def label_clusters(
    cluster_summaries: list[dict[str, Any]],
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Label clusters and write/upsert theme_registry.parquet — *stably*.

    HDBSCAN cluster ids are not stable week to week, so naively labeling every
    cluster re-names themes that were already correct. Instead each current
    cluster is matched to the prior week's theme centroids; if the best match
    clears ``LABEL_STABILITY_THRESHOLD`` the prior label is carried forward (no
    Claude call). Only clusters with no strong prior match get (re)labeled.
    Hand-pinned seed themes are always appended.
    """
    ensure_outputs()
    as_of = (as_of or pd.Timestamp.now(tz="UTC")).normalize().tz_localize(None)

    if not cluster_summaries:
        logger.warning("No cluster summaries provided — writing seed themes only")
        return _write_registry([], as_of)

    # Build current + prior theme centroids for stability matching. Best-effort:
    # if anything is missing (e.g. first run, no prior), fall back to labeling all.
    cur_centroids, prior_centroids = _stability_centroids()
    prior_meta = _latest_theme_metadata()  # theme_name -> registry row (carry-forward)

    n = len(cluster_summaries)
    logger.info("Labeling %d clusters (stable carry-forward + Claude for new only)...", n)
    rows, reused, relabeled = [], 0, 0
    for summary in cluster_summaries:
        cid = int(summary["cluster_id"])
        prior_name = _match_prior_theme(cid, cur_centroids, prior_centroids)
        if prior_name is not None and prior_name in prior_meta:
            pr = prior_meta[prior_name]
            rows.append({
                "cluster_id": cid,
                "theme_name": prior_name,
                "parent_theme": str(pr.get("parent_theme", "unknown")),
                "description": str(pr.get("description", "")),
                "related_themes": pr.get("related_themes", "[]"),
                "confidence": float(pr.get("confidence", 0.0)),
                "date": as_of,
            })
            reused += 1
        else:
            label = _label_cluster(summary)
            rows.append({
                "cluster_id": cid,
                "theme_name": str(label.get("theme_name", f"cluster_{cid}")),
                "parent_theme": str(label.get("parent_theme", "unknown")),
                "description": str(label.get("description", "")),
                "related_themes": json.dumps(label.get("related_themes") or []),
                "confidence": float(label.get("confidence", 0.0)),
                "date": as_of,
            })
            relabeled += 1
            logger.info("  cluster %3d → %-30s (NEW/relabeled, conf %.2f)",
                        cid, label.get("theme_name", "?"), label.get("confidence", 0.0))

    logger.info("Labeling done: %d carried forward (stable), %d Claude-labeled", reused, relabeled)
    return _write_registry(rows, as_of)


def _write_registry(rows: list[dict], as_of: pd.Timestamp) -> pd.DataFrame:
    """Upsert today's cluster rows + seed themes (only those not already emergent)."""
    from themes.dynamic_theme.seed_themes import seed_registry_rows

    labeled_names = {r["theme_name"] for r in rows}
    seeds = seed_registry_rows(as_of)
    seeds = seeds[~seeds["theme_name"].isin(labeled_names)]  # gap-fill only, no dupes

    parts = ([pd.DataFrame(rows)] if rows else []) + ([seeds] if not seeds.empty else [])
    new_df = pd.concat(parts, ignore_index=True) if parts else seed_registry_rows(as_of)

    existing = _load_registry_or_empty()
    if not existing.empty:
        existing = existing[existing["date"] < as_of]
    registry = pd.concat([existing, new_df], ignore_index=True)
    registry = registry.sort_values(["date", "cluster_id"]).reset_index(drop=True)
    registry.to_parquet(THEME_REGISTRY_PATH, index=False)
    logger.info("Wrote %s  total_rows=%d (+%d seed themes added)",
                THEME_REGISTRY_PATH, len(registry), len(seeds))
    return registry


def _latest_theme_metadata() -> dict[str, dict]:
    """{theme_name: latest registry row} for carrying parent/description forward."""
    reg = load_registry(latest_only=True)
    if reg.empty:
        return {}
    return {str(r["theme_name"]): r.to_dict() for _, r in reg.iterrows()}


def _stability_centroids() -> tuple[dict[int, "object"], dict[str, "object"]]:
    """(current cluster_id→centroid, prior theme_name→centroid). Empty on any miss."""
    try:
        import numpy as np  # noqa: F401
        from themes.dynamic_theme.config import TICKER_CLUSTERS_PATH
        from themes.dynamic_theme.stages.step02_embed import load_embeddings_matrix
        from themes.dynamic_theme.stages.step03_cluster import compute_centroids
        from themes.dynamic_theme.stages.step06_discovery import _load_prior_centroids

        tickers, matrix, _ = load_embeddings_matrix()
        cur_clusters = pd.read_parquet(TICKER_CLUSTERS_PATH)
        cur = compute_centroids(matrix, tickers, cur_clusters)
        emb_df = pd.DataFrame({"ticker": tickers, "embedding": list(matrix)})
        prior = _load_prior_centroids(_load_registry_or_empty(), emb_df)
        return cur, prior
    except Exception as exc:
        logger.warning("Stability matching unavailable (%s) — labeling all clusters", exc)
        return {}, {}


def _match_prior_theme(cid: int, cur_centroids: dict, prior_centroids: dict) -> str | None:
    """Return a prior theme_name to carry forward if cluster cid matches it well."""
    from themes.dynamic_theme.config import LABEL_STABILITY_THRESHOLD
    from themes.dynamic_theme.stages.step06_discovery import _cosine_similarity

    if not prior_centroids or cid not in cur_centroids:
        return None
    best_name, best_sim = None, -1.0
    for name, pc in prior_centroids.items():
        s = _cosine_similarity(cur_centroids[cid], pc)
        if s > best_sim:
            best_name, best_sim = name, s
    if best_name and best_sim >= LABEL_STABILITY_THRESHOLD and not str(best_name).startswith("cluster_"):
        return str(best_name)
    return None


def _load_registry_or_empty() -> pd.DataFrame:
    if THEME_REGISTRY_PATH.exists():
        try:
            return pd.read_parquet(THEME_REGISTRY_PATH)
        except Exception:
            pass
    return pd.DataFrame(
        columns=["cluster_id", "theme_name", "parent_theme", "description", "related_themes", "confidence", "date"]
    )


def load_registry(*, latest_only: bool = True) -> pd.DataFrame:
    """Load theme registry. If latest_only=True returns only the most recent date's rows."""
    df = _load_registry_or_empty()
    if latest_only and not df.empty and "date" in df.columns:
        latest = df["date"].max()
        df = df[df["date"] == latest].copy()
    return df
