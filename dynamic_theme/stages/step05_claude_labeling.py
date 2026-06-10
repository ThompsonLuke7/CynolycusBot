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

from dynamic_theme.client.claude_client import call_claude_json
from dynamic_theme.config import THEME_REGISTRY_PATH, ensure_outputs

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
    """Label each cluster via Claude and write/upsert theme_registry.parquet."""
    ensure_outputs()
    as_of = (as_of or pd.Timestamp.now(tz="UTC")).normalize().tz_localize(None)

    if not cluster_summaries:
        logger.warning("No cluster summaries provided — nothing to label")
        return _load_registry_or_empty()

    logger.info("Calling Claude to label %d clusters ...", len(cluster_summaries))
    rows = []
    for summary in cluster_summaries:
        label = _label_cluster(summary)
        rows.append(
            {
                "cluster_id": int(label["cluster_id"]),
                "theme_name": str(label.get("theme_name", f"cluster_{label['cluster_id']}")),
                "parent_theme": str(label.get("parent_theme", "unknown")),
                "description": str(label.get("description", "")),
                "related_themes": json.dumps(label.get("related_themes") or []),
                "confidence": float(label.get("confidence", 0.0)),
                "date": as_of,
            }
        )
        logger.info(
            "  cluster %3d → %-30s (parent: %s, conf: %.2f)",
            label["cluster_id"],
            label.get("theme_name", "?"),
            label.get("parent_theme", "?"),
            label.get("confidence", 0.0),
        )

    new_df = pd.DataFrame(rows)

    # upsert: keep older dates from existing registry, add/replace today's rows
    existing = _load_registry_or_empty()
    if not existing.empty:
        existing = existing[existing["date"] < as_of]
    registry = pd.concat([existing, new_df], ignore_index=True)
    registry = registry.sort_values(["date", "cluster_id"]).reset_index(drop=True)
    registry.to_parquet(THEME_REGISTRY_PATH, index=False)
    logger.info("Wrote %s  total_rows=%d", THEME_REGISTRY_PATH, len(registry))
    return registry


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
