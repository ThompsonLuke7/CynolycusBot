"""Step 7 — Theme Relationship Graph.

Calls Claude once with all current theme names to determine economic and
supply-chain relationships between themes.

Output: theme_relationships.parquet
Schema : source | target | relationship | strength | date
"""
from __future__ import annotations

import json
import logging
import re

import pandas as pd

from themes.dynamic_theme.client.claude_client import call_claude, call_claude_json
from themes.dynamic_theme.config import (
    CLAUDE_RELATIONSHIP_MAX_TOKENS,
    RELATIONSHIP_FULL_REBUILD_NEW_FRACTION,
    THEME_RELATIONSHIPS_PATH,
    ensure_outputs,
)
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

_INCREMENTAL_RELATIONSHIP_PROMPT = """\
You are an institutional equity research taxonomy system.

The full current theme universe is:
{themes_json}

These themes are NEW this week and have no relationships on file yet:
{new_themes_json}

Determine economic and supply-chain relationships involving the NEW themes above —
either between two new themes, or between a new theme and an existing theme from the
full universe list. Every relationship you return must include at least one theme
from the NEW list. Do not return a relationship where both themes are absent from
the NEW list — those already have relationships on file and should not be repeated.

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
- Every relationship must include at least one theme from the NEW themes list
"""


def build_relationship_graph(
    registry_df: pd.DataFrame | None = None,
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Call Claude to map theme relationships and write theme_relationships.parquet.

    Only asks Claude about themes that are new since the prior run. Edges
    between two themes that both existed last week are carried forward
    unchanged — economic/supply-chain relationships don't shift week to
    week, so re-deriving them from scratch every run only adds cost, token
    truncation, and destabilizing edge churn. Falls back to a full rebuild
    when there's no prior snapshot, or when the new-theme fraction is large
    enough that "incremental" would barely save anything.
    """
    ensure_outputs()
    as_of = (as_of or pd.Timestamp.now(tz="UTC")).normalize().tz_localize(None)

    if registry_df is None:
        registry_df = load_registry(latest_only=True)

    if registry_df.empty or "theme_name" not in registry_df.columns:
        logger.warning("No themes in registry — skipping relationship graph")
        return pd.DataFrame(columns=["source", "target", "relationship", "strength", "date"])

    # Callers may pass the full multi-date accumulated registry (label_clusters
    # / discover_new_themes both return it as-is) — restrict to the *currently
    # active* theme set (this run's date), not every theme name ever labeled
    # across past weeks. Without this, stale/renamed themes from prior weeks
    # keep inflating the graph indefinitely.
    reg_dates = sorted(registry_df["date"].unique())
    latest_reg_date = reg_dates[-1]
    theme_names = sorted(
        registry_df.loc[registry_df["date"] == latest_reg_date, "theme_name"].dropna().unique().tolist()
    )
    active_set = set(theme_names)
    logger.info("Active themes this run: %d", len(theme_names))

    existing = _load_relationships_or_empty()
    prior_edge_dates = sorted(existing["date"].unique()) if not existing.empty else []
    prior_edges = existing[existing["date"] == prior_edge_dates[-1]] if prior_edge_dates else pd.DataFrame()

    prior_reg_date = reg_dates[-2] if len(reg_dates) >= 2 else None
    prior_active = (
        set(registry_df.loc[registry_df["date"] == prior_reg_date, "theme_name"].dropna())
        if prior_reg_date is not None else set()
    )
    new_themes = sorted(active_set - prior_active)
    new_fraction = (len(new_themes) / len(theme_names)) if theme_names else 1.0

    carried_edges = (
        prior_edges[prior_edges["source"].isin(active_set) & prior_edges["target"].isin(active_set)]
        [["source", "target", "relationship", "strength"]]
        if not prior_edges.empty else pd.DataFrame(columns=["source", "target", "relationship", "strength"])
    )

    new_set = set(new_themes)
    incremental = False

    if not prior_edges.empty and not new_themes:
        logger.info(
            "No new themes since last run (%d active, 0 new) — reusing %d prior edges, no Claude call",
            len(theme_names), len(carried_edges),
        )
        relationships = []
    elif not prior_edges.empty and new_fraction <= RELATIONSHIP_FULL_REBUILD_NEW_FRACTION:
        logger.info(
            "%d new theme(s) of %d active (%.0f%%) — asking Claude to place only the new ones; "
            "%d prior edges carried forward untouched",
            len(new_themes), len(theme_names), new_fraction * 100, len(carried_edges),
        )
        incremental = True
        prompt = _INCREMENTAL_RELATIONSHIP_PROMPT.format(
            themes_json=json.dumps(theme_names, indent=2),
            new_themes_json=json.dumps(new_themes, indent=2),
        )
        relationships = _call_relationship_prompt(prompt)
    else:
        logger.info(
            "%d new theme(s) of %d active (%.0f%%, no prior edges or above rebuild threshold) — "
            "building the full relationship graph",
            len(new_themes), len(theme_names), new_fraction * 100,
        )
        carried_edges = pd.DataFrame(columns=["source", "target", "relationship", "strength"])
        prompt = _RELATIONSHIP_PROMPT.format(themes_json=json.dumps(theme_names, indent=2))
        relationships = _call_relationship_prompt(prompt)

    new_rows = []
    for rel in relationships:
        src = str(rel.get("source", "")).strip()
        tgt = str(rel.get("target", "")).strip()
        relation = str(rel.get("relationship", "")).strip()
        strength = float(rel.get("strength", 0.0))
        if not (src and tgt and src != tgt and strength >= 0.5 and src in active_set and tgt in active_set):
            continue
        # Defensive: the incremental prompt asks Claude to only return edges
        # touching a NEW theme, but don't trust that blindly — an old<->old
        # edge here would silently clobber the carried-forward (authoritative)
        # edge for that pair below.
        if incremental and src not in new_set and tgt not in new_set:
            continue
        new_rows.append({"source": src, "target": tgt, "relationship": relation, "strength": strength})

    # Merge carried-forward edges with freshly-derived ones, deduping by
    # unordered (source, target) pair — a fresh edge wins over a carried one
    # (e.g. if strength was re-assessed), but we never keep both.
    new_edges_df = pd.DataFrame(new_rows, columns=["source", "target", "relationship", "strength"])
    parts = [df for df in (carried_edges, new_edges_df) if not df.empty]
    combined = pd.concat(parts, ignore_index=True) if parts else new_edges_df
    if not combined.empty:
        pair_key = combined.apply(lambda r: tuple(sorted((r["source"], r["target"]))), axis=1)
        combined = combined[~pair_key.duplicated(keep="last")]
    combined["date"] = as_of

    # upsert: preserve older dates, replace today's
    if not existing.empty:
        existing = existing[existing["date"] < as_of]
    out_parts = [df for df in (existing, combined) if not df.empty]
    out = pd.concat(out_parts, ignore_index=True) if out_parts else combined
    out.to_parquet(THEME_RELATIONSHIPS_PATH, index=False)
    logger.info(
        "Wrote %s  carried=%d  new=%d  total_today=%d  total_rows=%d",
        THEME_RELATIONSHIPS_PATH, len(carried_edges), len(new_rows), len(combined), len(out),
    )
    return out


def _call_relationship_prompt(prompt: str) -> list[dict]:
    try:
        result = call_claude_json(prompt, max_tokens=CLAUDE_RELATIONSHIP_MAX_TOKENS)
        return result.get("relationships", [])
    except Exception as exc:
        # A truncated/oversized response is no longer valid JSON. Rather than
        # discard the whole graph (which strands the registry with stale edges),
        # salvage every complete relationship object from the raw text.
        logger.warning("Relationship JSON parse failed (%s) — salvaging partial edges", exc)
        return _salvage_relationships(prompt)


def _salvage_relationships(prompt: str) -> list[dict]:
    """Re-call Claude and regex-extract every complete relationship object.

    Tolerant of a truncated trailing object: only fully-closed { ... } blocks
    containing the four expected keys are kept.
    """
    try:
        raw = call_claude(prompt, max_tokens=CLAUDE_RELATIONSHIP_MAX_TOKENS)
    except Exception as exc:
        logger.error("Relationship salvage re-call failed: %s", exc)
        return []
    salvaged: list[dict] = []
    for blob in re.findall(r"\{[^{}]*\}", raw):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if {"source", "target", "relationship", "strength"} <= obj.keys():
            salvaged.append(obj)
    logger.info("Salvaged %d relationship edges from partial response", len(salvaged))
    return salvaged


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
