"""Step 7 — Theme Relationship Graph.

Calls Claude once with all current theme names to determine economic and
supply-chain relationships between themes.

Output: theme_relationships.parquet
Schema : source | target | relationship | strength | date
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from themes.dynamic_theme.client.claude_client import call_claude_json
from themes.dynamic_theme.config import THEME_RELATIONSHIPS_PATH, ensure_outputs
from themes.dynamic_theme.stages.step05_claude_labeling import load_registry

logger = logging.getLogger(__name__)

_RELATIONSHIP_PROMPT = """\
You are an institutional equity research taxonomy system.

Determine economic and supply-chain relationships between the following themes.

Themes:
{themes_json}

Return JSON only — no prose, no markdown fences.

{{
  "relationships": [
    {{
      "source": "",
      "target": "",
      "relationship": "",
      "strength": 0.0
    }}
  ]
}}

Rules:
- Only include relationships where strength >= 0.5
- relationship must be one of: drives_demand, supply_chain, competes_with, correlated, enables
- strength is 0.0-1.0 reflecting how strong the economic linkage is
- Be selective — include only high-conviction linkages
"""


def build_relationship_graph(
    registry_df: pd.DataFrame | None = None,
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Call Claude to map theme relationships and write theme_relationships.parquet."""
    ensure_outputs()
    as_of = (as_of or pd.Timestamp.now(tz="UTC")).normalize().tz_localize(None)

    if registry_df is None:
        registry_df = load_registry(latest_only=True)

    if registry_df.empty or "theme_name" not in registry_df.columns:
        logger.warning("No themes in registry — skipping relationship graph")
        return pd.DataFrame(columns=["source", "target", "relationship", "strength", "date"])

    theme_names = registry_df["theme_name"].dropna().unique().tolist()
    logger.info("Building relationship graph for %d themes ...", len(theme_names))

    prompt = _RELATIONSHIP_PROMPT.format(themes_json=json.dumps(theme_names, indent=2))

    try:
        result = call_claude_json(prompt)
        relationships = result.get("relationships", [])
    except Exception as exc:
        logger.error("Claude relationship graph call failed: %s", exc)
        relationships = []

    rows = []
    for rel in relationships:
        src = str(rel.get("source", "")).strip()
        tgt = str(rel.get("target", "")).strip()
        relation = str(rel.get("relationship", "")).strip()
        strength = float(rel.get("strength", 0.0))
        if src and tgt and src != tgt and strength >= 0.5:
            rows.append(
                {
                    "source": src,
                    "target": tgt,
                    "relationship": relation,
                    "strength": strength,
                    "date": as_of,
                }
            )

    new_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source", "target", "relationship", "strength", "date"]
    )

    # upsert: preserve older dates, replace today's
    existing = _load_relationships_or_empty()
    if not existing.empty:
        existing = existing[existing["date"] < as_of]
    out = pd.concat([existing, new_df], ignore_index=True)
    out.to_parquet(THEME_RELATIONSHIPS_PATH, index=False)
    logger.info(
        "Wrote %s  new_edges=%d  total_rows=%d",
        THEME_RELATIONSHIPS_PATH, len(rows), len(out),
    )
    return out


def _load_relationships_or_empty() -> pd.DataFrame:
    if THEME_RELATIONSHIPS_PATH.exists():
        try:
            return pd.read_parquet(THEME_RELATIONSHIPS_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=["source", "target", "relationship", "strength", "date"])


def load_relationships(*, latest_only: bool = True) -> pd.DataFrame:
    df = _load_relationships_or_empty()
    if latest_only and not df.empty and "date" in df.columns:
        latest = df["date"].max()
        df = df[df["date"] == latest].copy()
    return df
